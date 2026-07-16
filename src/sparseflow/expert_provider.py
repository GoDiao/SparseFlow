from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .loader import ShardReader
from .locator import ExpertLocator
from .moe_probe import CachedExpertProvider, _decode_span


ExpertWeights = Mapping[str, Any]


@runtime_checkable
class ExpertProvider(Protocol):
    """Storage-policy boundary consumed by the shared SparseFlow MoE runtime."""

    backend_id: str

    def get(self, layer: int, expert_id: int) -> ExpertWeights: ...

    def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None: ...

    def begin_forward(self, forward: int, phase: str) -> None: ...

    def observe_routes(self, layer: int, selected_experts) -> None: ...

    def predict(self, routes_by_layer: Mapping[int, tuple[int, ...]]) -> None: ...

    def finish_generation(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def counters(self) -> dict[str, Any]: ...

    def storage_report(self) -> dict[str, Any]: ...

    def prefetch_stats(self) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


StreamingExpertProvider = CachedExpertProvider


class ResidentExpertProvider:
    """Keep every routed expert in RAM and expose zero-copy per-expert views.

    Qwen3.6 stores one fused gate/up tensor and one fused down tensor per layer.
    The provider reads each fused tensor once, owns its writable backing buffer,
    and returns views along expert axis zero.  Both resident and streaming
    providers therefore feed identically shaped tensors to the shared kernel.
    """

    backend_id = "sparseflow-resident"
    required_parts = ("gate_up_proj", "down_proj")

    def __init__(
        self,
        model_dir: str | Path,
        reader: ShardReader,
        torch,
        layers: tuple[int, ...] | None = None,
    ):
        self.locator = ExpertLocator(model_dir)
        self.reader = reader
        self.torch = torch
        self.layers = self.locator.layers if layers is None else tuple(layers)
        if not self.layers:
            raise ValueError("resident provider requires at least one expert layer")
        unknown = sorted(set(self.layers) - set(self.locator.layers))
        if unknown:
            raise ValueError(f"resident provider requested unknown layers: {unknown}")

        self._backings: dict[tuple[int, str], bytearray] = {}
        self._fused: dict[int, dict[str, Any]] = {}
        self._requests = 0
        self._closed = False
        self._preload_calls_before = reader.read_calls
        self._preload_bytes_before = reader.read_bytes
        started = time.perf_counter()
        try:
            self._preload()
        except Exception:
            self.close()
            raise
        self.preload_seconds = time.perf_counter() - started
        self._preload_calls = reader.read_calls - self._preload_calls_before
        self._preload_bytes = reader.read_bytes - self._preload_bytes_before
        self._reads_after_preload = (reader.read_calls, reader.read_bytes)

    def _preload(self) -> None:
        for layer in self.layers:
            spans = self.locator.fused_parts(layer)
            missing = [part for part in self.required_parts if part not in spans]
            if missing:
                raise ValueError(f"layer {layer} is missing routed expert parts: {missing}")
            weights: dict[str, Any] = {}
            for part in self.required_parts:
                span = spans[part]
                if not span.shape or span.shape[0] != self.locator.num_experts:
                    raise ValueError(
                        f"resident fused tensor has invalid expert axis for layer={layer} "
                        f"part={part}: shape={span.shape}"
                    )
                payload = self.reader.read(span.shard, span.data_offset, span.nbytes)
                backing = bytearray(payload)
                del payload
                tensor = _decode_span(span, backing, self.torch, copy=False)
                if tuple(tensor.shape) != span.shape:
                    raise ValueError(
                        f"resident tensor shape mismatch for layer={layer} part={part}: "
                        f"expected={span.shape}, actual={tuple(tensor.shape)}"
                    )
                self._backings[(layer, part)] = backing
                weights[part] = tensor
            self._fused[layer] = weights

    def get(self, layer: int, expert_id: int) -> ExpertWeights:
        if self._closed:
            raise RuntimeError("resident expert provider is closed")
        if not 0 <= expert_id < self.locator.num_experts:
            raise ValueError(
                f"expert id {expert_id} is outside [0, {self.locator.num_experts})"
            )
        try:
            weights = self._fused[layer]
        except KeyError as exc:
            raise ValueError(f"resident expert layer is not loaded: {layer}") from exc
        self._requests += 1
        return {part: weights[part][expert_id] for part in self.required_parts}

    def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None:
        if self._closed:
            raise RuntimeError("resident expert provider is closed")
        if layer not in self._fused:
            raise ValueError(f"resident expert layer is not loaded: {layer}")
        for expert_id in expert_ids:
            if not 0 <= int(expert_id) < self.locator.num_experts:
                raise ValueError(
                    f"expert id {expert_id} is outside [0, {self.locator.num_experts})"
                )

    def begin_forward(self, forward: int, phase: str) -> None:
        del forward, phase
        if self._closed:
            raise RuntimeError("resident expert provider is closed")

    def observe_routes(self, layer: int, selected_experts) -> None:
        del selected_experts
        if layer not in self._fused:
            raise ValueError(f"resident expert layer is not loaded: {layer}")

    def predict(self, routes_by_layer: Mapping[int, tuple[int, ...]]) -> None:
        del routes_by_layer
        if self._closed:
            raise RuntimeError("resident expert provider is closed")

    def finish_generation(self) -> None:
        if self._closed:
            raise RuntimeError("resident expert provider is closed")

    def snapshot(self) -> dict[str, Any]:
        return self.counters()

    def counters(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "reader_calls": self.reader.read_calls,
            "reader_bytes": self.reader.read_bytes,
            "requests": self._requests,
            "resident_layers": len(self._fused),
            "resident_experts": len(self._fused) * self.locator.num_experts,
            "resident_buffers": len(self._backings),
            "resident_bytes": sum(len(buffer) for buffer in self._backings.values()),
        }

    def storage_report(self) -> dict[str, Any]:
        snapshot = self.counters()
        return {
            **snapshot,
            "policy": "all routed experts resident in fused layer buffers",
            "preload_read_calls": self._preload_calls,
            "preload_read_bytes": self._preload_bytes,
            "reads_after_preload": self.reader.read_calls - self._reads_after_preload[0],
            "bytes_after_preload": self.reader.read_bytes - self._reads_after_preload[1],
            "preload_seconds": self.preload_seconds,
        }

    def prefetch_stats(self) -> None:
        return None

    @property
    def decoded_entries(self) -> int:
        return len(self._fused) * self.locator.num_experts

    def close(self) -> None:
        self._fused.clear()
        self._backings.clear()
        self._closed = True

    def __enter__(self) -> "ResidentExpertProvider":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


# [Main Dev]

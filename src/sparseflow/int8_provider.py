from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

from .cache import CachedExpert, ExpertCache
from .int8_container import Int8ExpertIndex, Int8ExpertLocation
from .loader import ShardReader
from .moe_probe import CachedExpertProvider


def dequantize_expert(location, payloads: Mapping[str, bytes | bytearray | memoryview], torch):
    weights = {}
    for part in location.parts:
        data = payloads[f"{part.part}.data"]
        scales = payloads[f"{part.part}.scales"]
        quantized = torch.frombuffer(data, dtype=torch.int8).reshape(part.shape)
        scale_tensor = torch.frombuffer(scales, dtype=torch.float16)
        weights[part.part] = (
            quantized.float() * scale_tensor.float().unsqueeze(1)
        ).to(torch.bfloat16)
    return weights


class Int8ResidentExpertProvider:

    def __init__(self, container_dir: str | Path, torch, native: bool = False):
        self.index = Int8ExpertIndex.from_dir(container_dir)
        self.torch = torch
        self.native = native
        self.backend_id = (
            "sparseflow-int8-native-resident"
            if native
            else "sparseflow-int8-reference-resident"
        )
        self._backings: dict[Path, bytearray] = {}
        self._native_views: dict[tuple[int, int], dict[str, Any]] = {}
        self._requests = 0
        self._closed = False
        started = time.perf_counter()
        for layer in self.index.layers:
            path = self.index.locate(layer, 0).file
            buffer = bytearray(path.stat().st_size)
            with path.open("rb", buffering=0) as stream:
                read_bytes = stream.readinto(buffer)
            if read_bytes != len(buffer):
                raise OSError(f"short INT8 resident preload: {path}")
            self._backings[path] = buffer
        self.preload_seconds = time.perf_counter() - started
        self._resident_bytes = sum(len(value) for value in self._backings.values())

    def get(self, layer: int, expert_id: int):
        if self._closed:
            raise RuntimeError("INT8 resident provider is closed")
        location = self.index.locate(layer, expert_id)
        backing = self._backings[location.file]
        payloads = {}
        for part in location.parts:
            payloads[f"{part.part}.data"] = memoryview(backing)[
                part.data_offset : part.data_offset + part.data_nbytes
            ]
            payloads[f"{part.part}.scales"] = memoryview(backing)[
                part.scale_offset : part.scale_offset + part.scale_nbytes
            ]
        self._requests += 1
        if self.native:
            from .native_int8 import prepare_native_weights

            key = (layer, expert_id)
            weights = self._native_views.get(key)
            if weights is None:
                weights = prepare_native_weights(location, payloads, self.torch)
                self._native_views[key] = weights
            return {"gate_up_proj": weights, "down_proj": None}
        return dequantize_expert(location, payloads, self.torch)

    def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None:
        for expert_id in expert_ids:
            self.index.locate(layer, int(expert_id))

    def begin_forward(self, forward: int, phase: str) -> None:
        del forward, phase

    def observe_routes(self, layer: int, selected_experts) -> None:
        del layer, selected_experts

    def predict(self, routes_by_layer) -> None:
        del routes_by_layer

    def finish_generation(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return self.counters()

    def counters(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "reader_calls": 0,
            "reader_bytes": 0,
            "requests": self._requests,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_evictions": 0,
            "cached_experts": 0,
            "cached_bytes": 0,
            "loaded_bytes": 0,
            "hit_bytes": 0,
            "miss_bytes": 0,
            "admission_rejections": 0,
            "decoded_entries": 0,
            "transient_prefetch_entries": 0,
            "demand_read_calls": 0,
            "demand_read_bytes": 0,
            "demand_read_ms_total": 0.0,
            "demand_requests": self._requests,
            "demand_reuse_hits": self._requests,
            "demand_prefetch_served": 0,
            "demand_misses": 0,
            "prefetch_submitted": 0,
            "prefetch_hits": 0,
            "prefetch_late": 0,
            "prefetch_hit_bytes": 0,
            "prefetch_wasted_ready_bytes": 0,
        }

    def storage_report(self) -> dict[str, Any]:
        return {
            **self.counters(),
            "policy": "all canonical INT8 routed experts resident by layer file",
            "resident_bytes": self._resident_bytes,
            "resident_layer_files": len(self._backings),
            "resident_layers": len(self.index.layers),
            "resident_experts": len(self.index.layers) * self.index.num_experts,
            "resident_buffers": len(self._backings),
            "preload_seconds": self.preload_seconds,
            "preload_read_bytes": self._resident_bytes,
            "bytes_after_preload": 0,
            "format_id": self.index.manifest["format_id"],
            "native_views": len(self._native_views),
        }

    def prefetch_stats(self) -> None:
        return None

    def close(self) -> None:
        self._backings.clear()
        self._native_views.clear()
        self._resident_bytes = 0
        self._closed = True


@dataclass(frozen=True)
class _Int8ReadPart:
    part: str
    shard: Path
    file_offset: int
    nbytes: int


@dataclass(frozen=True)
class _Int8ReadLocation:
    source: Int8ExpertLocation
    parts: tuple[_Int8ReadPart, ...]

    @property
    def layer(self) -> int:
        return self.source.layer

    @property
    def expert_id(self) -> int:
        return self.source.expert_id

    @property
    def nbytes(self) -> int:
        return self.source.nbytes

    def __iter__(self):
        return iter(self.parts)


class _Int8ReadIndex:
    """Expose canonical INT8 data/scales through the generic range interface."""

    def __init__(self, index: Int8ExpertIndex):
        self.index = index
        self.layers = index.layers
        self._locations: dict[tuple[int, int], _Int8ReadLocation] = {}

    def locate(self, layer: int, expert_id: int) -> _Int8ReadLocation:
        key = (layer, expert_id)
        location = self._locations.get(key)
        if location is not None:
            return location
        source = self.index.locate(layer, expert_id)
        spans = []
        for part in source.parts:
            spans.extend(
                (
                    _Int8ReadPart(
                        part=f"{part.part}.data",
                        shard=source.file,
                        file_offset=part.data_offset,
                        nbytes=part.data_nbytes,
                    ),
                    _Int8ReadPart(
                        part=f"{part.part}.scales",
                        shard=source.file,
                        file_offset=part.scale_offset,
                        nbytes=part.scale_nbytes,
                    ),
                )
            )
        location = _Int8ReadLocation(source=source, parts=tuple(spans))
        self._locations[key] = location
        return location


class Int8StreamingExpertProvider(CachedExpertProvider):
    """Canonical INT8 storage adapter for the shared cache/prefetch lifecycle."""

    def __init__(
        self,
        container_dir: str | Path,
        cache: ExpertCache,
        reader: ShardReader,
        torch,
        native: bool = False,
        prefetch_workers: int = 0,
        coalesce_gap: int = 0,
        prefetch_policy: str = "current-route",
        prefetch_budget_ratio: float = 0.25,
    ):
        self.index = Int8ExpertIndex.from_dir(container_dir)
        self.native = native
        self._closed = False
        locator = _Int8ReadIndex(self.index)
        super().__init__(
            container_dir,
            cache,
            reader,
            torch,
            prefetch_workers=prefetch_workers,
            coalesce_gap=coalesce_gap,
            prefetch_policy=prefetch_policy,
            prefetch_budget_ratio=prefetch_budget_ratio,
            locator=locator,
        )
        self.backend_id = (
            "sparseflow-int8-native-streaming"
            if native
            else "sparseflow-int8-reference-streaming"
        )

    def get(self, layer: int, expert_id: int):
        if self._closed:
            raise RuntimeError("INT8 streaming provider is closed")
        return super().get(layer, expert_id)

    def _decode_entry(
        self,
        key: tuple[int, int],
        location: _Int8ReadLocation,
        entry: CachedExpert,
    ) -> dict[str, Any]:
        existing = self._decoded.get(key) if self.native else None
        if existing is not None and existing[0] is entry:
            return existing[1]

        started = time.perf_counter() if self.cache.collect_timings else None
        if self.native:
            from .native_int8 import prepare_native_weights

            weights = prepare_native_weights(location.source, entry.parts, self.torch)
            result = {"gate_up_proj": weights, "down_proj": None}
        else:
            result = dequantize_expert(location.source, entry.parts, self.torch)
        if started is not None:
            self._timings_ms["tensor_decode_view"] += (
                time.perf_counter() - started
            ) * 1000.0
        if self.native and self.cache.peek(*key) is entry:
            self._decoded[key] = (entry, result)
        return result

    def counters(self) -> dict[str, Any]:
        result = super().counters()
        result["native_views"] = self.decoded_entries if self.native else 0
        return result

    def storage_report(self) -> dict[str, Any]:
        return {
            **super().storage_report(),
            "policy": "canonical INT8 routed experts loaded through ExpertCache",
            "format_id": self.index.manifest["format_id"],
        }

    def close(self) -> None:
        if self._closed:
            return
        super().close()
        self._decoded.clear()
        self._closed = True


# [Main Dev]

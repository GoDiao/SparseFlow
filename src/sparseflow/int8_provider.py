from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any, Mapping

from .cache import ExpertCache
from .int8_container import Int8ExpertIndex, Int8ExpertLocation
from .loader import ShardReader


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
    backend_id = "sparseflow-int8-reference-resident"

    def __init__(self, container_dir: str | Path, torch):
        self.index = Int8ExpertIndex.from_dir(container_dir)
        self.torch = torch
        self._backings: dict[Path, bytearray] = {}
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
        }

    def prefetch_stats(self) -> None:
        return None

    def close(self) -> None:
        self._backings.clear()
        self._resident_bytes = 0
        self._closed = True


class Int8StreamingExpertProvider:
    backend_id = "sparseflow-int8-reference-streaming"

    def __init__(
        self,
        container_dir: str | Path,
        cache: ExpertCache,
        reader: ShardReader,
        torch,
    ):
        self.index = Int8ExpertIndex.from_dir(container_dir)
        self.cache = cache
        self.reader = reader
        self.torch = torch
        self._closed = False
        self._forward = -1
        self._phase = "unknown"
        self._demand_requests = 0
        self._demand_reuse_hits = 0
        self._demand_misses = 0
        self._read_ms_total = 0.0
        self._dequant_ms_total = 0.0

    def get(self, layer: int, expert_id: int):
        if self._closed:
            raise RuntimeError("INT8 streaming provider is closed")
        location = self.index.locate(layer, expert_id)
        entry = self.cache.lookup(layer, expert_id, expected_nbytes=location.nbytes)
        self._demand_requests += 1
        if entry is None:
            self._demand_misses += 1
            started = time.perf_counter()
            payloads = self._read_location(location)
            self._read_ms_total += (time.perf_counter() - started) * 1000.0
            entry = self.cache.put_loaded(layer, expert_id, payloads)
        else:
            self._demand_reuse_hits += 1
        started = time.perf_counter()
        weights = dequantize_expert(location, entry.parts, self.torch)
        self._dequant_ms_total += (time.perf_counter() - started) * 1000.0
        return weights

    def _read_location(self, location: Int8ExpertLocation) -> dict[str, bytearray]:
        payloads = {}
        for part in location.parts:
            data = bytearray(part.data_nbytes)
            scales = bytearray(part.scale_nbytes)
            self.reader.readinto(location.file, part.data_offset, data)
            self.reader.readinto(location.file, part.scale_offset, scales)
            payloads[f"{part.part}.data"] = data
            payloads[f"{part.part}.scales"] = scales
        return payloads

    def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None:
        for expert_id in expert_ids:
            self.index.locate(layer, int(expert_id))

    def begin_forward(self, forward: int, phase: str) -> None:
        self._forward = forward
        self._phase = phase
        self.cache.begin_forward(forward, phase)

    def observe_routes(self, layer: int, selected_experts) -> None:
        counts = Counter(int(value) for value in selected_experts.reshape(-1).tolist())
        self.cache.observe_routes(layer, counts)

    def predict(self, routes_by_layer) -> None:
        del routes_by_layer

    def finish_generation(self) -> None:
        pass

    def snapshot(self) -> dict[str, Any]:
        return self.counters()

    def counters(self) -> dict[str, Any]:
        cache = self.cache.counters()
        return {
            "backend_id": self.backend_id,
            "reader_calls": self.reader.read_calls,
            "reader_bytes": self.reader.read_bytes,
            "requests": cache["requests"],
            "cache_hits": cache["hits"],
            "cache_misses": cache["misses"],
            "cache_evictions": cache["evictions"],
            "cached_experts": cache["cache_entries"],
            "cached_bytes": cache["cached_bytes"],
            "loaded_bytes": cache["loaded_bytes"],
            "hit_bytes": cache["hit_bytes"],
            "miss_bytes": cache["miss_bytes"],
            "admission_rejections": cache["admission_rejections"],
            "decoded_entries": 0,
            "transient_prefetch_entries": 0,
            "demand_read_calls": self.reader.read_calls,
            "demand_read_bytes": self.reader.read_bytes,
            "demand_read_ms_total": self._read_ms_total,
            "demand_requests": self._demand_requests,
            "demand_reuse_hits": self._demand_reuse_hits,
            "demand_prefetch_served": 0,
            "demand_misses": self._demand_misses,
            "prefetch_submitted": 0,
            "prefetch_hits": 0,
            "prefetch_late": 0,
            "prefetch_hit_bytes": 0,
            "prefetch_wasted_ready_bytes": 0,
            "timing_pread_ms_total": self._read_ms_total,
            "timing_tensor_decode_view_ms_total": self._dequant_ms_total,
            **{key: value for key, value in cache.items() if key.startswith("timing_")},
        }

    def storage_report(self) -> dict[str, Any]:
        return {
            **self.counters(),
            "policy": "canonical INT8 routed experts loaded through ExpertCache",
            "cache_policy": self.cache.policy.snapshot(),
            "format_id": self.index.manifest["format_id"],
            "prefetch": None,
        }

    def prefetch_stats(self) -> dict[str, int]:
        return {
            "submitted": 0,
            "failed": 0,
            "hits": 0,
            "late": 0,
            "wasted_ready_bytes": 0,
        }

    def close(self) -> None:
        self._closed = True


# [Main Dev]

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import Counter
from pathlib import Path
from threading import RLock
import time
from typing import Any, Callable

from .cache import CachedExpert, ExpertCache
from .buffers import BufferPool
from .loader import ShardReader, read_expert_bytes
from .locator import ExpertLocator


def run_expert_kernel(hidden_states, gate_up_proj, down_proj):
    """The shared routed-expert kernel used by both resident and streaming paths."""

    if isinstance(gate_up_proj, dict) and gate_up_proj.get("native_int8"):
        from .native_int8 import run_native_expert

        return run_native_expert(hidden_states, gate_up_proj, __import__("torch"))
    import torch.nn.functional as functional

    gate, up = functional.linear(hidden_states, gate_up_proj).chunk(2, dim=-1)
    activated = functional.silu(gate) * up
    return functional.linear(activated, down_proj)


def compare_expert_paths(
    model_dir: str | Path,
    layer: int,
    expert_id: int,
    rows: int = 2,
    seed: int = 1234,
) -> dict[str, Any]:
    """Compare resident and repeated-streaming execution for one expert."""

    if rows < 1:
        raise ValueError("rows must be positive")

    import torch

    locator = ExpertLocator(model_dir)
    location = locator.locate(layer, expert_id)
    resident_bytes = read_expert_bytes(location)
    resident_weights = _decode_bf16(location, resident_bytes, torch)
    hidden_size = resident_weights["gate_up_proj"].shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        rows,
        hidden_size,
        generator=generator,
        dtype=resident_weights["gate_up_proj"].dtype,
    )

    with torch.inference_mode():
        resident_output = run_expert_kernel(
            hidden_states,
            resident_weights["gate_up_proj"],
            resident_weights["down_proj"],
        )

        streaming_bytes = read_expert_bytes(location)
        streaming_weights = _decode_bf16(location, streaming_bytes, torch)
        streaming_output = run_expert_kernel(
            hidden_states,
            streaming_weights["gate_up_proj"],
            streaming_weights["down_proj"],
        )

    difference = (resident_output - streaming_output).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = resident_output.abs().clamp_min(torch.finfo(resident_output.dtype).eps)
    max_rel = float((difference / denominator).max().item()) if difference.numel() else 0.0
    return {
        "schema_version": 1,
        "model": str(Path(model_dir).expanduser().resolve()),
        "layer": layer,
        "expert_id": expert_id,
        "rows": rows,
        "dtype": str(resident_weights["gate_up_proj"].dtype).replace("torch.", ""),
        "parts": {
            part: {
                "shape": list(weight.shape),
                "nbytes": location.part(part).nbytes,
            }
            for part, weight in resident_weights.items()
        },
        "resident_vs_streaming": {
            "exact_equal": bool(torch.equal(resident_output, streaming_output)),
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
            "resident_output_shape": list(resident_output.shape),
        },
        "kernel": "linear(gate_up).chunk -> silu(gate)*up -> linear(down)",
    }


class CachedExpertProvider:
    """Turn :class:`ExpertCache` entries into reusable Qwen expert tensors.

    The storage cache remains raw-byte and model-independent.  This adapter
    owns only decoded tensor views and ties each view to the exact
    :class:`CachedExpert` entry that backs it.  An eviction therefore causes a
    new decode on the next miss, while cache hits avoid both disk I/O and
    tensor decoding.
    """

    backend_id = "sparseflow-streaming"

    def __init__(
        self,
        model_dir: str | Path,
        cache: ExpertCache,
        reader: ShardReader,
        torch,
        prefetch_workers: int = 0,
        coalesce_gap: int = 0,
        prefetch_policy: str = "current-route",
        prefetch_budget_ratio: float = 0.25,
        locator: Any | None = None,
    ):
        if prefetch_workers < 0:
            raise ValueError("prefetch_workers must be non-negative")
        if coalesce_gap < 0:
            raise ValueError("coalesce_gap must be non-negative")
        if prefetch_policy not in {"none", "current-route", "previous-token", "hot-set"}:
            raise ValueError(f"unknown prefetch policy: {prefetch_policy}")
        if not 0.0 <= prefetch_budget_ratio <= 1.0:
            raise ValueError("prefetch_budget_ratio must be in [0, 1]")
        self.locator = locator if locator is not None else ExpertLocator(model_dir)
        self.cache = cache
        self.reader = reader
        self.torch = torch
        self.prefetch_workers = prefetch_workers
        self.coalesce_gap = coalesce_gap
        self.prefetch_policy = prefetch_policy
        self.prefetch_budget_ratio = prefetch_budget_ratio
        self._forward = -1
        self._phase = "unknown"
        self._prediction_history: list[dict[int, tuple[int, ...]]] = []
        self._decoded: dict[tuple[int, int], tuple[CachedExpert, dict[str, Any]]] = {}
        self.cache.add_eviction_listener(self._on_cache_evict)
        self._prefetch_lock = RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._inflight: dict[tuple[int, int], Future] = {}
        self._future_keys: dict[Future, tuple[tuple[int, int], ...]] = {}
        self._future_submitted: dict[Future, float] = {}
        self._future_reasons: dict[Future, str] = {}
        self._prefetched_ready: dict[tuple[int, int], int] = {}
        self._transient_ready: dict[tuple[int, int], CachedExpert] = {}
        self._prefetch_metrics = {
            "submitted": 0,
            "batches": 0,
            "completed": 0,
            "failed": 0,
            "waits": 0,
            "prefetched_experts": 0,
            "prefetched_bytes": 0,
            "coalesced_ranges": 0,
            "logical_ranges": 0,
            "physical_bytes": 0,
            "hit_bytes": 0,
            "wasted_bytes": 0,
            "read_ms_total": 0.0,
            "wait_ms_total": 0.0,
            "hits": 0,
            "late": 0,
            "dropped": 0,
            "useful_bytes": 0,
            "wasted_ready": 0,
            "wasted_ready_bytes": 0,
            "current_route_submitted": 0,
            "previous_token_submitted": 0,
            "hot_set_submitted": 0,
            "cancelled": 0,
        }
        self._demand_metrics = {
            "requests": 0,
            "reuse_hits": 0,
            "prefetch_served": 0,
            "misses": 0,
            "read_calls": 0,
            "read_bytes": 0,
            "read_ms_total": 0.0,
        }
        self._timings_ms = {
            "tensor_decode_view": 0.0,
        }
        pool_budget = min(
            64 * 1024**2,
            cache.max_bytes if cache.max_bytes is not None else 64 * 1024**2,
        )
        self._buffer_pool = (
            BufferPool(pool_budget)
            if prefetch_workers == 0
            and pool_budget > 0
            and cache.capacity_per_layer != 0
            and cache.policy.policy_id != "none"
            else None
        )

    def __enter__(self) -> "CachedExpertProvider":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def get(self, layer: int, expert_id: int) -> dict[str, Any]:
        if self.prefetch_workers == 0:
            return self._get_without_prefetch(layer, expert_id)

        key = (layer, expert_id)
        location = self.locator.locate(layer, expert_id)
        with self._prefetch_lock:
            entry = self.cache.lookup(layer, expert_id, expected_nbytes=location.nbytes)
            transient = self._transient_ready.pop(key, None) if entry is None else None
            if transient is not None:
                entry = transient
            future = self._inflight.get(key) if entry is None else None
            if entry is not None:
                prefetched = transient is not None or key in self._prefetched_ready
                self._mark_prefetch_used(
                    key,
                    fallback_bytes=transient.nbytes if transient is not None else None,
                )
                if prefetched:
                    self._demand_metrics["prefetch_served"] += 1
                else:
                    self._demand_metrics["reuse_hits"] += 1
            if future is not None:
                self._prefetch_metrics["waits"] += 1
                self._prefetch_metrics["late"] += 1
            self._demand_metrics["requests"] += 1
            if entry is None and future is None:
                self._demand_metrics["misses"] += 1

        if entry is None and future is not None:
            entries = self._consume_future(future)
            entry = entries.get(key)
            if entry is None:
                raise RuntimeError(f"prefetch completed without expert payload: {key}")
            with self._prefetch_lock:
                self._transient_ready.pop(key, None)
                self._mark_prefetch_used(key, fallback_bytes=entry.nbytes)
                self._demand_metrics["prefetch_served"] += 1
        elif entry is None:
            calls_before = self.reader.read_calls
            bytes_before = self.reader.read_bytes
            started = time.perf_counter()
            payloads = self._read_demand_payloads(location)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._prefetch_lock:
                entry = self.cache.put_loaded(layer, expert_id, payloads)
                self._demand_metrics["read_calls"] += self.reader.read_calls - calls_before
                self._demand_metrics["read_bytes"] += self.reader.read_bytes - bytes_before
                self._demand_metrics["read_ms_total"] += elapsed_ms
                self._reconcile_prefetched_ready()

        return self._decode_entry(key, location, entry)

    def _get_without_prefetch(self, layer: int, expert_id: int) -> dict[str, Any]:
        key = (layer, expert_id)
        location = self.locator.locate(layer, expert_id)
        entry = self.cache.lookup(layer, expert_id, expected_nbytes=location.nbytes)
        self._demand_metrics["requests"] += 1
        if entry is not None:
            self._demand_metrics["reuse_hits"] += 1
        else:
            self._demand_metrics["misses"] += 1
            calls_before = self.reader.read_calls
            bytes_before = self.reader.read_bytes
            started = time.perf_counter()
            payloads = self._read_demand_payloads(location)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            entry = self.cache.put_loaded(layer, expert_id, payloads)
            self._demand_metrics["read_calls"] += self.reader.read_calls - calls_before
            self._demand_metrics["read_bytes"] += self.reader.read_bytes - bytes_before
            self._demand_metrics["read_ms_total"] += elapsed_ms
        return self._decode_entry(key, location, entry)

    def _decode_entry(self, key, location, entry: CachedExpert) -> dict[str, Any]:
        existing = self._decoded.get(key)
        if existing is not None and existing[0] is entry:
            return existing[1]

        decode_started = time.perf_counter() if self.cache.collect_timings else None
        weights = {
            part: _decode_span(
                location.part(part),
                entry.parts[part],
                self.torch,
                copy=False,
            )
            for part in ("gate_up_proj", "down_proj")
        }
        if decode_started is not None:
            self._timings_ms["tensor_decode_view"] += (
                time.perf_counter() - decode_started
            ) * 1000.0
        layer, expert_id = key
        if self.cache.peek(layer, expert_id) is entry:
            self._decoded[key] = (entry, weights)
        else:
            # A zero/smaller-than-entry budget deliberately does not retain
            # decoded state beyond the current kernel call.
            self._decoded.pop(key, None)
        return weights

    def _read_demand_payloads(self, location) -> dict[str, bytearray]:
        if self._buffer_pool is None:
            return self.reader.read_expert_into(location)
        payloads: dict[str, bytearray] = {}
        try:
            for item in location:
                payload = self._buffer_pool.acquire(item.nbytes)
                self.reader.readinto(item.shard, item.file_offset, payload)
                payloads[item.part] = payload
        except Exception:
            for payload in payloads.values():
                self._buffer_pool.release(payload)
            raise
        return payloads

    def _on_cache_evict(self, entry: CachedExpert) -> None:
        key = (entry.layer, entry.expert_id)
        decoded = self._decoded.get(key)
        if decoded is not None and decoded[0] is entry:
            self._decoded.pop(key, None)
        if self._buffer_pool is not None:
            for payload in entry.parts.values():
                if isinstance(payload, bytearray):
                    self._buffer_pool.release(payload)

    @property
    def decoded_entries(self) -> int:
        return len(self._decoded)

    def prefetch(self, layer: int, expert_ids: list[int] | tuple[int, ...]) -> None:
        """Submit one coalesced read for selected cache-miss experts."""

        self._prefetch(layer, expert_ids, reason="current-route")

    def _prefetch(
        self,
        layer: int,
        expert_ids: list[int] | tuple[int, ...],
        reason: str,
    ) -> None:

        if self.prefetch_workers <= 0:
            raise RuntimeError("prefetch is disabled; construct provider with prefetch_workers > 0")
        unique_keys = tuple(dict.fromkeys((layer, int(expert_id)) for expert_id in expert_ids))
        with self._prefetch_lock:
            live = set(self.cache.cached_keys())
            pending = tuple(
                key for key in unique_keys
                if key not in live and key not in self._inflight
            )
            if not pending:
                self._prefetch_metrics["dropped"] += len(unique_keys)
                return
            locations = tuple(self.locator.locate(*key) for key in pending)
            executor = self._executor
            if executor is None:
                executor = ThreadPoolExecutor(max_workers=self.prefetch_workers)
                self._executor = executor
            future = executor.submit(
                self._read_prefetch_batch,
                locations,
            )
            self._future_keys[future] = pending
            self._future_submitted[future] = time.perf_counter()
            self._future_reasons[future] = reason
            for key in pending:
                self._inflight[key] = future
            self._prefetch_metrics["submitted"] += len(pending)
            self._prefetch_metrics["batches"] += 1
            metric = reason.replace("-", "_") + "_submitted"
            if metric in self._prefetch_metrics:
                self._prefetch_metrics[metric] += len(pending)

    def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None:
        """Prepare selected experts when asynchronous prefetch is enabled."""

        if self.prefetch_workers > 0 and self.prefetch_policy in {
            "current-route",
            "previous-token",
        }:
            self._prefetch(layer, expert_ids, reason="current-route")

    def begin_forward(self, forward: int, phase: str) -> None:
        self._forward = forward
        self._phase = phase
        with self._prefetch_lock:
            self.cache.begin_forward(forward, phase)
            self._reconcile_prefetched_ready()

    def observe_routes(self, layer: int, selected_experts) -> None:
        values = selected_experts.detach().to(device="cpu").reshape(-1).tolist()
        counts = Counter(int(value) for value in values)
        with self._prefetch_lock:
            self.cache.observe_routes(layer, counts)

    def predict(self, routes_by_layer: dict[int, tuple[int, ...]]) -> None:
        if self.prefetch_workers <= 0 or self.prefetch_policy in {"none", "current-route"}:
            return
        hot_by_layer: dict[int, list[int]] = {}
        for layer, expert_id in self.cache.policy.hot_keys():
            hot_by_layer.setdefault(layer, []).append(expert_id)
        stable_routes: dict[int, tuple[int, ...]] = {}
        if self.prefetch_policy == "previous-token" and len(self._prediction_history) >= 2:
            for layer, experts in routes_by_layer.items():
                previous = set(experts)
                for history in self._prediction_history:
                    previous.intersection_update(history.get(layer, ()))
                stable_routes[layer] = tuple(
                    expert_id for expert_id in experts if expert_id in previous
                )
        if routes_by_layer:
            self._prediction_history.append(dict(routes_by_layer))
            self._prediction_history = self._prediction_history[-2:]
        hot_keys = [
            (layer, expert_id)
            for layer in sorted(hot_by_layer)
            for expert_id in hot_by_layer[layer]
        ]
        hot_key_set = set(hot_keys)
        route_keys = [
            (layer, expert_id)
            for layer in sorted(stable_routes)
            for expert_id in stable_routes[layer]
            if (layer, expert_id) not in hot_key_set
        ]
        candidates = hot_keys if self.prefetch_policy == "hot-set" else hot_keys + route_keys
        if self.cache.max_bytes is not None:
            budget = int(self.cache.max_bytes * self.prefetch_budget_ratio)
        elif self.cache.capacity_per_layer is not None:
            sample = self.locator.locate(self.locator.layers[0], 0).nbytes
            budget = int(
                self.cache.capacity_per_layer
                * len(self.locator.layers)
                * sample
                * self.prefetch_budget_ratio
            )
        else:
            budget = 0
        selected: dict[int, list[int]] = {}
        selected_bytes = 0
        with self._prefetch_lock:
            live = set(self.cache.cached_keys())
            inflight = set(self._inflight)
        for layer, expert_id in candidates:
            if (layer, expert_id) in live or (layer, expert_id) in inflight:
                continue
            nbytes = self.locator.locate(layer, expert_id).nbytes
            if selected_bytes + nbytes > budget:
                continue
            selected.setdefault(layer, []).append(expert_id)
            selected_bytes += nbytes
        reason = "hot-set" if self.prefetch_policy == "hot-set" else "previous-token"
        for layer, expert_ids in selected.items():
            self._prefetch(layer, tuple(expert_ids), reason=reason)

    def _consume_future(self, future: Future) -> dict[tuple[int, int], CachedExpert]:
        try:
            payloads, read_ms = future.result()
        except Exception:
            with self._prefetch_lock:
                self._prefetch_metrics["failed"] += 1
                keys = self._future_keys.pop(future, ())
                self._future_submitted.pop(future, None)
                self._future_reasons.pop(future, None)
                for key in keys:
                    if self._inflight.get(key) is future:
                        self._inflight.pop(key, None)
            raise

        with self._prefetch_lock:
            keys = self._future_keys.pop(future, ())
            submitted = self._future_submitted.pop(future, time.perf_counter())
            reason = self._future_reasons.pop(future, "current-route")
            for key in keys:
                if self._inflight.get(key) is future:
                    self._inflight.pop(key, None)
            entries: dict[tuple[int, int], CachedExpert] = {}
            for key in keys:
                entry = self.cache.put_loaded(
                    key[0],
                    key[1],
                    payloads[key],
                    source=(
                        "current-route"
                        if reason == "current-route"
                        else "prefetch"
                    ),
                )
                entries[key] = entry
                if self.cache.peek(*key) is entry:
                    self._prefetched_ready[key] = entry.nbytes
                else:
                    self._transient_ready[key] = entry
            self._prefetch_metrics["completed"] += 1
            self._prefetch_metrics["read_ms_total"] += read_ms
            self._prefetch_metrics["wait_ms_total"] += (time.perf_counter() - submitted) * 1000.0
            self._prefetch_metrics["prefetched_experts"] += len(payloads)
            self._prefetch_metrics["prefetched_bytes"] += sum(
                sum(len(part) for part in parts.values())
                for parts in payloads.values()
            )
            batch_stats = getattr(payloads, "stats", None)
            if batch_stats is not None:
                self._prefetch_metrics["coalesced_ranges"] += batch_stats.ranges
                self._prefetch_metrics["logical_ranges"] += batch_stats.logical_ranges
                self._prefetch_metrics["physical_bytes"] += batch_stats.physical_bytes
                self._prefetch_metrics["useful_bytes"] += batch_stats.useful_bytes
                self._prefetch_metrics["wasted_bytes"] += batch_stats.wasted_bytes
            self._reconcile_prefetched_ready()
            return entries

    def _read_prefetch_batch(self, locations) -> tuple[Any, float]:
        started = time.perf_counter()
        payloads = self.reader.read_locations(locations, self.coalesce_gap)
        return payloads, (time.perf_counter() - started) * 1000.0

    def prefetch_stats(self) -> dict[str, int | float]:
        with self._prefetch_lock:
            self._reconcile_prefetched_ready()
            return dict(self._prefetch_metrics)

    def finish_generation(self) -> None:
        """Close predictive accounting without shutting down reusable workers."""

        with self._prefetch_lock:
            futures = tuple(set(self._inflight.values()))
        for future in futures:
            if future.cancel():
                with self._prefetch_lock:
                    keys = self._future_keys.pop(future, ())
                    self._future_submitted.pop(future, None)
                    self._future_reasons.pop(future, None)
                    for key in keys:
                        if self._inflight.get(key) is future:
                            self._inflight.pop(key, None)
                    self._prefetch_metrics["cancelled"] += len(keys)
                continue
            self._consume_future(future)
        with self._prefetch_lock:
            for nbytes in self._prefetched_ready.values():
                self._prefetch_metrics["wasted_ready"] += 1
                self._prefetch_metrics["wasted_ready_bytes"] += nbytes
            for entry in self._transient_ready.values():
                self._prefetch_metrics["wasted_ready"] += 1
                self._prefetch_metrics["wasted_ready_bytes"] += entry.nbytes
            self._prefetched_ready.clear()
            self._transient_ready.clear()

    def _mark_prefetch_used(
        self,
        key: tuple[int, int],
        fallback_bytes: int | None = None,
    ) -> None:
        nbytes = self._prefetched_ready.pop(key, fallback_bytes)
        if nbytes is not None:
            self._prefetch_metrics["hits"] += 1
            self._prefetch_metrics["hit_bytes"] += nbytes

    def _reconcile_prefetched_ready(self) -> None:
        live = set(self.cache.cached_keys())
        for key in tuple(self._prefetched_ready):
            if key not in live:
                self._prefetch_metrics["wasted_ready"] += 1
                self._prefetch_metrics["wasted_ready_bytes"] += self._prefetched_ready.pop(key)

    def snapshot(self) -> dict[str, Any]:
        return self.counters()

    def counters(self) -> dict[str, Any]:
        with self._prefetch_lock:
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
                "decoded_entries": self.decoded_entries,
                "transient_prefetch_entries": len(self._transient_ready),
                "demand_read_calls": self._demand_metrics["read_calls"],
                "demand_read_bytes": self._demand_metrics["read_bytes"],
                "demand_read_ms_total": self._demand_metrics["read_ms_total"],
                "demand_requests": self._demand_metrics["requests"],
                "demand_reuse_hits": self._demand_metrics["reuse_hits"],
                "demand_prefetch_served": self._demand_metrics["prefetch_served"],
                "demand_misses": self._demand_metrics["misses"],
                "prefetch_submitted": self._prefetch_metrics["submitted"],
                "prefetch_hits": self._prefetch_metrics["hits"],
                "prefetch_late": self._prefetch_metrics["late"],
                "prefetch_hit_bytes": self._prefetch_metrics["hit_bytes"],
                "prefetch_wasted_ready_bytes": self._prefetch_metrics[
                    "wasted_ready_bytes"
                ],
                "timing_pread_ms_total": (
                    self._demand_metrics["read_ms_total"]
                    + self._prefetch_metrics["read_ms_total"]
                ),
                **{
                    f"timing_{key}_ms_total": value
                    for key, value in self._timings_ms.items()
                },
                **{
                    key: value
                    for key, value in cache.items()
                    if key.startswith("timing_")
                },
                "forward": self._forward,
                "phase": self._phase,
                "buffer_pool_reuses": (
                    self._buffer_pool.stats.reuses if self._buffer_pool is not None else 0
                ),
            }

    def storage_report(self) -> dict[str, Any]:
        return {
            **self.counters(),
            "policy": "routed experts loaded on demand through ExpertCache",
            "cache_policy": self.cache.policy.snapshot(),
            "preload_read_calls": 0,
            "preload_read_bytes": 0,
            "prefetch_policy": self.prefetch_policy,
            "prefetch_budget_ratio": self.prefetch_budget_ratio,
            "prefetch": self.prefetch_stats(),
            "buffer_pool": (
                self._buffer_pool.as_dict() if self._buffer_pool is not None else None
            ),
        }

    def close(self) -> None:
        self.finish_generation()
        self.cache.remove_eviction_listener(self._on_cache_evict)
        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=True)
        if self._buffer_pool is not None:
            self._buffer_pool.clear()
            self._executor = None


def compare_moe_paths(
    model_dir: str | Path,
    layer: int = 0,
    rows: int = 1,
    seed: int = 1234,
    cache_slots: int | None = None,
    cache_bytes: int | None = None,
) -> dict[str, Any]:
    """Compare a complete Qwen3.6 MoE layer with two routed-weight policies.

    The router, routing weights, shared expert, and shared-expert gate are
    loaded once.  The resident path then preloads every routed expert in the
    layer, while the streaming path reads only the unique experts selected by
    the same route.  Both paths call :func:`run_moe_kernel`, so the comparison
    isolates storage policy from MoE arithmetic.

    This is intentionally a layer probe rather than a complete Qwen3.6
    transformer forward.  It is the correctness boundary needed before
    integrating the storage runtime with attention and generation.
    """

    if rows < 1:
        raise ValueError("rows must be positive")

    import torch

    locator = ExpertLocator(model_dir)
    spans = _qwen36_moe_spans(locator, layer)
    _validate_moe_spans(spans)
    cache = _make_expert_cache(locator.num_experts, cache_slots, cache_bytes)

    with ShardReader() as reader:
        common_before = (reader.read_calls, reader.read_bytes)
        common = {
            name: _read_tensor(span, reader, torch)
            for name, span in spans.items()
            if name not in {"gate_up_proj", "down_proj"}
        }
        common_after = (reader.read_calls, reader.read_bytes)

        gate_up_span = spans["gate_up_proj"]
        down_span = spans["down_proj"]
        resident_before = (reader.read_calls, reader.read_bytes)
        resident_weights = {
            "gate_up_proj": _read_tensor(gate_up_span, reader, torch),
            "down_proj": _read_tensor(down_span, reader, torch),
        }
        resident_after = (reader.read_calls, reader.read_bytes)

        dtype = resident_weights["gate_up_proj"].dtype
        hidden_size = int(gate_up_span.shape[-1])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        hidden_states = torch.randn(
            rows,
            hidden_size,
            generator=generator,
            dtype=dtype,
        )
        top_k = int(locator.config.text_config.get("num_experts_per_tok", 0) or 0)
        if top_k <= 0 or top_k > locator.num_experts:
            raise ValueError(f"invalid num_experts_per_tok: {top_k}")

        with torch.inference_mode():
            resident_routing_weights, resident_selected_experts, resident_router_logits = _route_hidden_states(
                hidden_states,
                common["router"],
                top_k,
            )

            resident_result = run_moe_kernel(
                hidden_states,
                common,
                resident_selected_experts,
                resident_routing_weights,
                lambda expert_id: {
                    "gate_up_proj": resident_weights["gate_up_proj"][expert_id],
                    "down_proj": resident_weights["down_proj"][expert_id],
                },
            )

            provider = CachedExpertProvider(model_dir, cache, reader, torch)
            streaming_before = (reader.read_calls, reader.read_bytes)
            streaming_routing_weights, streaming_selected_experts, streaming_router_logits = _route_hidden_states(
                hidden_states,
                common["router"],
                top_k,
            )
            streaming_result = run_moe_kernel(
                hidden_states,
                common,
                streaming_selected_experts,
                streaming_routing_weights,
                lambda expert_id: provider.get(layer, expert_id),
            )
            streaming_after = (reader.read_calls, reader.read_bytes)

    comparisons = {
        "selected_experts": _compare_tensor(
            resident_selected_experts,
            streaming_selected_experts,
            torch,
        ),
        "routing_weights": _compare_tensor(
            resident_routing_weights,
            streaming_routing_weights,
            torch,
        ),
        "router_logits": _compare_tensor(resident_router_logits, streaming_router_logits, torch),
        "routed_output": _compare_tensor(
            resident_result["routed_output"],
            streaming_result["routed_output"],
            torch,
        ),
        "shared_output": _compare_tensor(
            resident_result["shared_output"],
            streaming_result["shared_output"],
            torch,
        ),
        "final_output": _compare_tensor(
            resident_result["final_output"],
            streaming_result["final_output"],
            torch,
        ),
    }
    selected_ids = sorted(int(value) for value in streaming_selected_experts.unique().tolist())
    resident_read_calls = resident_after[0] - resident_before[0]
    resident_read_bytes = resident_after[1] - resident_before[1]
    streaming_read_calls = streaming_after[0] - streaming_before[0]
    streaming_read_bytes = streaming_after[1] - streaming_before[1]

    return {
        "schema_version": 1,
        "kind": "qwen3_5_moe_layer_correctness",
        "agent": "Main Dev",
        "model": str(Path(model_dir).expanduser().resolve()),
        "layer": layer,
        "rows": rows,
        "seed": seed,
        "hidden_size": hidden_size,
        "num_experts": locator.num_experts,
        "top_k": top_k,
        "dtype": str(dtype).replace("torch.", ""),
        "selected_experts": resident_selected_experts.tolist(),
        "routing_weights": resident_routing_weights.tolist(),
        "streaming_expert_ids": selected_ids,
        "resident_storage": {
            "policy": "preload all routed experts in the layer",
            "expert_count": locator.num_experts,
            "read_calls": resident_read_calls,
            "read_bytes": resident_read_bytes,
            "expected_routed_bytes": gate_up_span.nbytes + down_span.nbytes,
        },
        "streaming_storage": {
            "policy": "read each unique selected expert once",
            "expert_count": len(selected_ids),
            "read_calls": streaming_read_calls,
            "read_bytes": streaming_read_bytes,
            "expected_selected_bytes": sum(locator.locate(layer, expert).nbytes for expert in selected_ids),
        },
        "cache": {
            **cache.stats_dict(),
            "decoded_entries": provider.decoded_entries,
        },
        "common_storage": {
            "read_calls": common_after[0] - common_before[0],
            "read_bytes": common_after[1] - common_before[1],
        },
        "comparison": comparisons,
        "kernel": "router softmax/topk -> routed weighted accumulation + shared gated MLP",
    }


def compare_moe_cache_paths(
    model_dir: str | Path,
    layer: int = 0,
    rows: int = 1,
    forwards: int = 4,
    repeats: int = 2,
    seed: int = 1234,
    cache_slots: int | None = 4,
    cache_bytes: int | None = None,
    prefetch_workers: int = 0,
    coalesce_gap: int = 0,
) -> dict[str, Any]:
    """Run repeated resident/cached-streaming MoE forwards.

    ``repeats`` deliberately reuses each generated hidden-state batch that
    many times.  This gives the cache an observable hit path while later
    batches still exercise misses and evictions when the budget is small.
    Every forward is compared with the resident path before its cache metrics
    are recorded.
    """

    if rows < 1:
        raise ValueError("rows must be positive")
    if forwards < 1:
        raise ValueError("forwards must be positive")
    if repeats < 1:
        raise ValueError("repeats must be positive")

    import torch

    locator = ExpertLocator(model_dir)
    spans = _qwen36_moe_spans(locator, layer)
    _validate_moe_spans(spans)
    cache = _make_expert_cache(locator.num_experts, cache_slots, cache_bytes)

    with ShardReader() as reader:
        common_before = (reader.read_calls, reader.read_bytes)
        common = {
            name: _read_tensor(span, reader, torch)
            for name, span in spans.items()
            if name not in {"gate_up_proj", "down_proj"}
        }
        common_after = (reader.read_calls, reader.read_bytes)
        resident_before = (reader.read_calls, reader.read_bytes)
        resident_weights = {
            "gate_up_proj": _read_tensor(spans["gate_up_proj"], reader, torch),
            "down_proj": _read_tensor(spans["down_proj"], reader, torch),
        }
        resident_after = (reader.read_calls, reader.read_bytes)
        stream_before = (reader.read_calls, reader.read_bytes)
        provider = CachedExpertProvider(
            model_dir,
            cache,
            reader,
            torch,
            prefetch_workers=prefetch_workers,
            coalesce_gap=coalesce_gap,
        )
        top_k = int(locator.config.text_config.get("num_experts_per_tok", 0) or 0)
        if top_k <= 0 or top_k > locator.num_experts:
            raise ValueError(f"invalid num_experts_per_tok: {top_k}")
        hidden_size = int(spans["gate_up_proj"].shape[-1])
        per_forward: list[dict[str, Any]] = []

        for forward in range(forwards):
            generator = torch.Generator(device="cpu").manual_seed(seed + forward // repeats)
            hidden_states = torch.randn(
                rows,
                hidden_size,
                generator=generator,
                dtype=resident_weights["gate_up_proj"].dtype,
            )
            cache_before_stats = _cache_stats_snapshot(cache)
            reader_before_forward = (reader.read_calls, reader.read_bytes)

            with torch.inference_mode():
                resident_weights_for_route, resident_selected, resident_logits = _route_hidden_states(
                    hidden_states,
                    common["router"],
                    top_k,
                )
                resident_result = run_moe_kernel(
                    hidden_states,
                    common,
                    resident_selected,
                    resident_weights_for_route,
                    lambda expert_id: {
                        "gate_up_proj": resident_weights["gate_up_proj"][expert_id],
                        "down_proj": resident_weights["down_proj"][expert_id],
                    },
                )

                streaming_weights_for_route, streaming_selected, streaming_logits = _route_hidden_states(
                    hidden_states,
                    common["router"],
                    top_k,
                )
                streaming_result = run_moe_kernel(
                    hidden_states,
                    common,
                    streaming_selected,
                    streaming_weights_for_route,
                    lambda expert_id: provider.get(layer, expert_id),
                    (lambda expert_ids: provider.prefetch(layer, expert_ids))
                    if prefetch_workers > 0
                    else None,
                )

            cache_after_stats = _cache_stats_snapshot(cache)
            reader_after_forward = (reader.read_calls, reader.read_bytes)
            comparisons = {
                "selected_experts": _compare_tensor(resident_selected, streaming_selected, torch),
                "routing_weights": _compare_tensor(
                    resident_weights_for_route,
                    streaming_weights_for_route,
                    torch,
                ),
                "router_logits": _compare_tensor(resident_logits, streaming_logits, torch),
                "routed_output": _compare_tensor(
                    resident_result["routed_output"],
                    streaming_result["routed_output"],
                    torch,
                ),
                "shared_output": _compare_tensor(
                    resident_result["shared_output"],
                    streaming_result["shared_output"],
                    torch,
                ),
                "final_output": _compare_tensor(
                    resident_result["final_output"],
                    streaming_result["final_output"],
                    torch,
                ),
            }
            cache_delta = _cache_stats_delta(cache_before_stats, cache_after_stats)
            cache_delta["cached_bytes"] = cache_after_stats["cached_bytes"]
            cache_delta["cache_entries"] = cache_after_stats["cache_entries"]
            per_forward.append(
                {
                    "forward": forward,
                    "route_batch": forward // repeats,
                    "rows": rows,
                    "unique_selected_experts": len(streaming_selected.unique().tolist()),
                    "selected_experts": streaming_selected.tolist(),
                    "cache": cache_delta,
                    "reader": {
                        "read_calls": reader_after_forward[0] - reader_before_forward[0],
                        "read_bytes": reader_after_forward[1] - reader_before_forward[1],
                    },
                    "comparison": comparisons,
                }
            )

        provider.close()
        cache_result = cache.stats_dict()
        stream_read_calls = reader.read_calls - stream_before[0]
        stream_read_bytes = reader.read_bytes - stream_before[1]
        resident_read_calls = resident_after[0] - resident_before[0]
        resident_read_bytes = resident_after[1] - resident_before[1]

    comparison_names = (
        "selected_experts",
        "routing_weights",
        "router_logits",
        "routed_output",
        "shared_output",
        "final_output",
    )
    all_exact = all(
        item["comparison"][name]["exact_equal"]
        for item in per_forward
        for name in comparison_names
    )
    max_abs = max(
        (item["comparison"][name].get("max_abs_error", 0.0)
         for item in per_forward for name in comparison_names),
        default=0.0,
    )
    max_rel = max(
        (item["comparison"][name].get("max_rel_error", 0.0)
         for item in per_forward for name in comparison_names),
        default=0.0,
    )
    invariants = {
        "cache_request_partition": cache_result["requests"]
        == cache_result["hits"] + cache_result["misses"],
        "loaded_bytes_match_reader": cache_result["loaded_bytes"] == stream_read_bytes,
        "cached_bytes_within_budget": (
            cache.max_bytes is None or cache_result["cached_bytes"] <= cache.max_bytes
        ),
        "per_forward_request_partition": all(
            item["cache"]["requests"]
            == item["cache"]["hits"] + item["cache"]["misses"]
            for item in per_forward
        ),
    }
    return {
        "schema_version": 1,
        "kind": "qwen3_5_moe_cache_correctness",
        "agent": "Main Dev",
        "model": str(Path(model_dir).expanduser().resolve()),
        "layer": layer,
        "rows": rows,
        "forwards": forwards,
        "repeats": repeats,
        "seed": seed,
        "hidden_size": hidden_size,
        "num_experts": locator.num_experts,
        "top_k": top_k,
        "dtype": str(resident_weights["gate_up_proj"].dtype).replace("torch.", ""),
        "cache_policy": {
            "capacity_per_layer": cache.capacity_per_layer,
            "max_bytes": cache.max_bytes,
            "prefetch_workers": prefetch_workers,
            "coalesce_gap": coalesce_gap,
        },
        "resident_storage": {
            "policy": "preload all routed experts in the layer",
            "read_calls": resident_read_calls,
            "read_bytes": resident_read_bytes,
            "expected_routed_bytes": spans["gate_up_proj"].nbytes + spans["down_proj"].nbytes,
        },
        "common_storage": {
            "read_calls": common_after[0] - common_before[0],
            "read_bytes": common_after[1] - common_before[1],
        },
        "streaming_storage": {
            "policy": "ExpertCache-backed unique selected experts",
            "read_calls": stream_read_calls,
            "read_bytes": stream_read_bytes,
            "loaded_bytes": cache_result["loaded_bytes"],
            "decoded_entries": provider.decoded_entries,
        },
        "cache": cache_result,
        "prefetch": provider.prefetch_stats(),
        "correctness": {
            "all_exact_equal": all_exact,
            "max_abs_error": max_abs,
            "max_rel_error": max_rel,
        },
        "invariants": invariants,
        "forwards_detail": per_forward,
        "kernel": "router softmax/topk -> routed weighted accumulation + shared gated MLP",
    }


def _make_expert_cache(
    num_experts: int,
    capacity_per_layer: int | None,
    max_bytes: int | None,
) -> ExpertCache:
    if capacity_per_layer is None and max_bytes is None:
        capacity_per_layer = num_experts
    return ExpertCache(capacity_per_layer=capacity_per_layer, max_bytes=max_bytes)


def _cache_stats_snapshot(cache: ExpertCache) -> dict[str, int | float]:
    return cache.stats_dict()


def _cache_stats_delta(
    before: dict[str, int | float],
    after: dict[str, int | float],
) -> dict[str, int | float]:
    fields = ("requests", "hits", "misses", "evictions", "loaded_bytes")
    result: dict[str, int | float] = {
        field: after[field] - before[field]
        for field in fields
    }
    requests = int(result["requests"])
    result["hit_rate"] = int(result["hits"]) / requests if requests else 0.0
    return result


def run_moe_kernel(
    hidden_states,
    common_weights: dict[str, Any],
    selected_experts,
    routing_weights,
    routed_loader: Callable[[int], dict[str, Any]],
    prepare_routed: Callable[[tuple[int, ...]], None] | None = None,
) -> dict[str, Any]:
    """Run the Qwen3.6 sparse MoE calculation for already computed routes."""

    import torch.nn.functional as functional

    expert_ids = tuple(int(value) for value in selected_experts.unique(sorted=True).tolist())
    if prepare_routed is not None:
        prepare_routed(expert_ids)

    # Compute the shared path after reads are submitted so an asynchronous
    # provider can overlap shared-MoE arithmetic with routed expert I/O.
    gate = functional.linear(hidden_states, common_weights["shared_gate_proj"])
    shared_gate = gate.sigmoid()
    shared_hidden = functional.linear(hidden_states, common_weights["shared_gate_proj_mlp"])
    shared_up = functional.linear(hidden_states, common_weights["shared_up_proj"])
    shared_output = functional.linear(
        functional.silu(shared_hidden) * shared_up,
        common_weights["shared_down_proj"],
    )
    shared_output = shared_gate * shared_output

    routed_output = run_routed_experts(
        hidden_states,
        selected_experts,
        routing_weights,
        routed_loader,
        prepare_routed=None,
    )

    return {
        "routed_output": routed_output,
        "shared_output": shared_output,
        "final_output": routed_output + shared_output,
    }


def run_routed_experts(
    hidden_states,
    selected_experts,
    routing_weights,
    routed_loader: Callable[[int], dict[str, Any]],
    prepare_routed: Callable[[tuple[int, ...]], None] | None = None,
    timing_callback: Callable[[str, float], None] | None = None,
):
    """Run only routed experts, reusable by a full Transformer MoE module."""

    dispatch_started = time.perf_counter() if timing_callback is not None else None
    expert_ids = tuple(int(value) for value in selected_experts.unique(sorted=True).tolist())
    if dispatch_started is not None:
        timing_callback("dispatch", (time.perf_counter() - dispatch_started) * 1000.0)
    if prepare_routed is not None:
        started = time.perf_counter() if timing_callback is not None else None
        prepare_routed(expert_ids)
        if started is not None:
            timing_callback("prepare", (time.perf_counter() - started) * 1000.0)
    dispatch_started = time.perf_counter() if timing_callback is not None else None
    routed_output = hidden_states.new_zeros(hidden_states.shape)
    if dispatch_started is not None:
        timing_callback("dispatch", (time.perf_counter() - dispatch_started) * 1000.0)
    for expert_id in expert_ids:
        # Match Transformers' Qwen3.5-MoE dispatch order exactly.  Its
        # expert mask is laid out as [top_k, token], so torch.where visits
        # top-k positions before token positions.  The mathematically
        # equivalent [token, top_k] order can produce different BF16 GEMM
        # rounding on CPU and the error compounds across decoder layers.
        dispatch_started = time.perf_counter() if timing_callback is not None else None
        top_positions, token_indices = (
            (selected_experts == expert_id)
            .transpose(0, 1)
            .nonzero(as_tuple=True)
        )
        current_state = hidden_states[token_indices]
        if dispatch_started is not None:
            timing_callback("dispatch", (time.perf_counter() - dispatch_started) * 1000.0)
        started = time.perf_counter() if timing_callback is not None else None
        weights = routed_loader(expert_id)
        if started is not None:
            timing_callback("provider_get", (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter() if timing_callback is not None else None
        current_output = run_expert_kernel(
            current_state,
            weights["gate_up_proj"],
            weights["down_proj"],
        )
        if started is not None:
            timing_callback("expert_kernel", (time.perf_counter() - started) * 1000.0)
        started = time.perf_counter() if timing_callback is not None else None
        current_output = current_output * routing_weights[token_indices, top_positions, None]
        routed_output.index_add_(
            0,
            token_indices,
            current_output.to(routed_output.dtype),
        )
        if started is not None:
            timing_callback(
                "routing_accumulation",
                (time.perf_counter() - started) * 1000.0,
            )
    return routed_output


def _route_hidden_states(hidden_states, router_weight, top_k: int):
    import torch
    import torch.nn.functional as functional

    router_logits = functional.linear(hidden_states, router_weight)
    router_probs = functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    return router_top_value.to(router_logits.dtype), router_indices, router_logits


def _qwen36_moe_spans(locator: ExpertLocator, layer: int):
    prefix = f"model.language_model.layers.{layer}.mlp."
    names = {
        "router": prefix + "gate.weight",
        "gate_up_proj": prefix + "experts.gate_up_proj",
        "down_proj": prefix + "experts.down_proj",
        "shared_gate_proj_mlp": prefix + "shared_expert.gate_proj.weight",
        "shared_up_proj": prefix + "shared_expert.up_proj.weight",
        "shared_down_proj": prefix + "shared_expert.down_proj.weight",
        "shared_gate_proj": prefix + "shared_expert_gate.weight",
    }
    try:
        return {key: locator.index.find(name) for key, name in names.items()}
    except KeyError as exc:
        raise ValueError(f"missing Qwen3.6 MoE tensor for layer {layer}: {exc.args[0]}") from exc


def _validate_moe_spans(spans) -> None:
    expected = {
        "router": ("router",),
        "gate_up_proj": ("routed",),
        "down_proj": ("routed",),
        "shared_gate_proj_mlp": ("shared",),
        "shared_up_proj": ("shared",),
        "shared_down_proj": ("shared",),
        "shared_gate_proj": ("shared_gate",),
    }
    for key, kind in expected.items():
        span = spans[key]
        if span.dtype not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unsupported {kind[0]} dtype for {key}: {span.dtype}")
        if span.nbytes <= 0:
            raise ValueError(f"empty MoE tensor: {span.name}")
    if spans["gate_up_proj"].shape[0] != spans["down_proj"].shape[0]:
        raise ValueError("routed expert tensors disagree on expert count")
    if spans["router"].shape[0] != spans["gate_up_proj"].shape[0]:
        raise ValueError("router and routed tensors disagree on expert count")
    if spans["router"].shape[1] != spans["gate_up_proj"].shape[2]:
        raise ValueError("router hidden size does not match routed expert input size")
    if spans["gate_up_proj"].shape[1] % 2:
        raise ValueError("gate_up_proj output dimension must be even")
    if spans["gate_up_proj"].shape[2] != spans["down_proj"].shape[1]:
        raise ValueError("routed gate/up input and down output dimensions disagree")
    if spans["gate_up_proj"].shape[1] // 2 != spans["down_proj"].shape[2]:
        raise ValueError("routed intermediate dimensions disagree")
    hidden = spans["router"].shape[1]
    if spans["shared_gate_proj_mlp"].shape[1] != hidden or spans["shared_up_proj"].shape[1] != hidden:
        raise ValueError("shared expert input dimension does not match router hidden size")
    if spans["shared_gate_proj_mlp"].shape[0] != spans["shared_up_proj"].shape[0]:
        raise ValueError("shared gate/up dimensions disagree")
    if spans["shared_down_proj"].shape != (hidden, spans["shared_gate_proj_mlp"].shape[0]):
        raise ValueError("shared down projection shape is incompatible")
    if spans["shared_gate_proj"].shape != (1, hidden):
        raise ValueError("shared expert gate must have shape [1, hidden_size]")
    dtypes = {span.dtype for span in spans.values()}
    if len(dtypes) != 1:
        raise ValueError(f"MoE tensors use inconsistent dtypes: {sorted(dtypes)}")


def _read_tensor(span, reader: ShardReader, torch):
    return _decode_span(span, reader.read(span.shard, span.data_offset, span.nbytes), torch)


def _decode_span(span, payload, torch, copy: bool = True):
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
    }
    try:
        dtype = dtype_map[span.dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype: {span.dtype}") from exc
    shape = getattr(span, "shape", None) or getattr(span, "expert_shape", None)
    element_count = getattr(span, "numel", None)
    if element_count is None:
        element_count = getattr(span, "element_count")
    expected = element_count * (2 if span.dtype in {"BF16", "F16"} else 4)
    if len(payload) != expected:
        raise ValueError(
            f"short tensor payload for {span.name}: expected {expected}, got {len(payload)}"
        )
    buffer = bytearray(payload) if copy else payload
    return torch.frombuffer(buffer, dtype=dtype).reshape(shape)


def _compare_tensor(left, right, torch) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "exact_equal": False,
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    difference = (left - right).abs().float()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    denominator = right.abs().float().clamp_min(torch.finfo(torch.float32).eps)
    max_rel = float((difference / denominator).max().item()) if difference.numel() else 0.0
    return {
        "exact_equal": bool(torch.equal(left, right)),
        "shape_equal": True,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "shape": list(left.shape),
        "dtype": str(left.dtype).replace("torch.", ""),
    }


def _decode_bf16(location, payloads: dict[str, bytes], torch):
    weights = {}
    for part in ("gate_up_proj", "down_proj"):
        item = location.part(part)
        if item.dtype != "BF16":
            raise ValueError(f"single expert probe currently requires BF16, got {item.dtype}")
        data = payloads[part]
        expected = item.element_count * 2
        if len(data) != expected:
            raise ValueError(f"short {part} payload: expected {expected}, got {len(data)}")
        writable = bytearray(data)
        weights[part] = torch.frombuffer(writable, dtype=torch.bfloat16).clone().reshape(item.expert_shape)
    return weights


# [Main Dev]

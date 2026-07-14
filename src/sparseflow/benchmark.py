from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .cache import ExpertCache
from .loader import ShardReader, read_expert_bytes
from .locator import ExpertLocator
from .trace import RouteTrace, TraceRequest, load_route_trace, trace_sha256


def parse_capacities(value: str) -> list[int]:
    capacities = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        capacity = int(item)
        if capacity < 0:
            raise ValueError(f"cache capacity must be non-negative: {capacity}")
        capacities.append(capacity)
    if not capacities:
        raise ValueError("at least one cache capacity is required")
    return capacities


def parse_byte_budgets(value: str) -> list[int]:
    budgets = []
    multipliers = {"": 1, "b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2, "mib": 1024**2, "gb": 1024**3, "gib": 1024**3}
    for item in value.split(","):
        token = item.strip().lower()
        if not token:
            continue
        suffix = next((suffix for suffix in sorted(multipliers, key=len, reverse=True) if suffix and token.endswith(suffix)), "")
        number = token[: -len(suffix)] if suffix else token
        try:
            budget = int(float(number) * multipliers[suffix])
        except ValueError as exc:
            raise ValueError(f"invalid byte budget: {item}") from exc
        if budget < 0:
            raise ValueError(f"byte budget must be non-negative: {item}")
        budgets.append(budget)
    if not budgets:
        raise ValueError("at least one byte budget is required")
    return budgets


def parse_layers(value: str, num_layers: int) -> list[int]:
    layers: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid layer range: {item}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    if not layers:
        raise ValueError("at least one layer is required")
    result = sorted(layers)
    invalid = [layer for layer in result if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"layers outside model range [0, {num_layers}): {invalid}")
    return result


def generate_trace(
    layers: Iterable[int],
    num_experts: int,
    tokens: int,
    top_k: int,
    mode: str = "locality",
    seed: int = 1234,
) -> list[TraceRequest]:
    if tokens < 1:
        raise ValueError("tokens must be positive")
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}]")
    if mode not in {"uniform", "locality"}:
        raise ValueError(f"unsupported trace mode: {mode}")

    rng = random.Random(seed)
    layer_list = list(layers)
    hot_count = min(num_experts, max(top_k, min(32, num_experts // 8 or 1)))
    hot = {layer: rng.sample(range(num_experts), hot_count) for layer in layer_list}
    trace: list[TraceRequest] = []
    for _ in range(tokens):
        for layer in layer_list:
            population = hot[layer] if mode == "locality" else range(num_experts)
            if mode == "locality":
                selected = rng.sample(population, top_k)
            else:
                selected = rng.sample(range(num_experts), top_k)
            trace.extend((layer, expert_id) for expert_id in selected)
    return trace


def load_trace(path: str | Path) -> list[TraceRequest]:
    return load_route_trace(path).flat_requests


def run_expert_benchmark(
    model_dir: str | Path,
    capacities: Iterable[int],
    trace: list[TraceRequest] | RouteTrace,
    batch_union: bool = False,
    byte_budgets: Iterable[int] | None = None,
) -> dict[str, Any]:
    locator = ExpertLocator(model_dir)
    capacities = list(capacities)
    if byte_budgets is not None:
        specs = [(None, budget) for budget in byte_budgets]
    else:
        specs = [(capacity, None) for capacity in capacities]
    if not specs:
        raise ValueError("at least one cache capacity or byte budget is required")

    route_trace = trace if isinstance(trace, RouteTrace) else RouteTrace.from_flat(trace)
    replay_groups = route_trace.replay_groups(batch_union=batch_union)
    raw_trace = route_trace.flat_requests
    effective_events = [
        (group.phase, layer, expert)
        for group in replay_groups
        for layer, expert in group.requests
    ]
    effective_trace = [(layer, expert) for _, layer, expert in effective_events]
    if not effective_trace:
        raise ValueError("trace is empty")

    locations = {
        request: locator.locate(*request)
        for request in set(effective_trace)
    }
    results = []
    with ShardReader() as reader:
        for capacity, max_bytes in specs:
            cache = ExpertCache(capacity_per_layer=capacity, max_bytes=max_bytes)
            read_latencies: list[float] = []
            phase_metrics: dict[str, dict[str, int]] = defaultdict(
                lambda: {"requests": 0, "hits": 0, "misses": 0, "loaded_bytes": 0, "logical_bytes": 0}
            )
            reader_calls_before = reader.read_calls
            reader_bytes_before = reader.read_bytes
            io_before = _process_read_bytes()
            started = time.perf_counter()
            for phase, layer, expert_id in effective_events:
                phase_result = phase_metrics[phase]
                phase_result["requests"] += 1
                phase_result["logical_bytes"] += locations[(layer, expert_id)].nbytes
                hits_before = cache.stats.hits
                misses_before = cache.stats.misses
                loaded_before = cache.stats.loaded_bytes

                def load() -> dict[str, bytes]:
                    read_started = time.perf_counter()
                    payloads = read_expert_bytes(locations[(layer, expert_id)], reader)
                    read_latencies.append((time.perf_counter() - read_started) * 1000.0)
                    return payloads

                cache.get_or_load(layer, expert_id, load)
                phase_result["hits"] += cache.stats.hits - hits_before
                phase_result["misses"] += cache.stats.misses - misses_before
                phase_result["loaded_bytes"] += cache.stats.loaded_bytes - loaded_before
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            io_after = _process_read_bytes()
            stats = cache.stats_dict()
            for phase_result in phase_metrics.values():
                phase_result["hit_rate"] = (
                    phase_result["hits"] / phase_result["requests"]
                    if phase_result["requests"]
                    else 0.0
                )
            results.append(
                {
                    "capacity_per_layer": capacity,
                    "max_bytes": max_bytes,
                    **stats,
                    "logical_bytes": sum(
                        locations[(layer, expert_id)].nbytes
                        for layer, expert_id in effective_trace
                    ),
                    "phase_metrics": dict(phase_metrics),
                    "reader_read_calls": reader.read_calls - reader_calls_before,
                    "reader_read_bytes": reader.read_bytes - reader_bytes_before,
                    "process_read_bytes": (
                        io_after - io_before
                        if io_before is not None and io_after is not None
                        else None
                    ),
                    "wall_time_ms": elapsed_ms,
                    "read_latency_mean_ms": statistics.fmean(read_latencies) if read_latencies else 0.0,
                    "read_latency_p50_ms": _percentile(read_latencies, 50),
                    "read_latency_p95_ms": _percentile(read_latencies, 95),
                }
            )

    return {
        "schema_version": 1,
        "model": {
            "path": str(Path(model_dir).expanduser().resolve()),
            "num_experts": locator.num_experts,
        },
        "trace": {
            "schema_version": route_trace.schema_version,
            "requests": len(effective_trace),
            "raw_requests": len(raw_trace),
            "effective_requests": len(effective_trace),
            "batch_union_deduped_requests": len(raw_trace) - len(effective_trace),
            "batch_union": batch_union,
            "phases": route_trace.phases,
            "effective_phases": {
                phase: sum(1 for current_phase, _, _ in effective_events if current_phase == phase)
                for phase in sorted({current_phase for current_phase, _, _ in effective_events})
            },
            "groups": len(route_trace.groups),
            "layers": route_trace.layers,
            "sha256": route_trace.source_sha256 or trace_sha256(route_trace),
        },
        "results": results,
    }


def _process_read_bytes() -> int | None:
    """Return Linux process read_bytes, or None on platforms without procfs."""

    try:
        text = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "read_bytes":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile / 100 * len(ordered)) - 1))
    return ordered[index]


def format_expert_benchmark(result: dict[str, Any], format_bytes) -> str:
    lines = [
        f"SparseFlow expert-bench: {result['model']['path']}",
        f"trace       {result['trace']['raw_requests']} raw / {result['trace']['effective_requests']} effective requests",
        f"phases      {result['trace'].get('phases', {})}, batch-union {result['trace'].get('batch_union', False)}",
        f"trace sha   {result['trace']['sha256']}",
        "",
        "capacity  requests  hit-rate  loaded       proc-read   cached      wall-ms  read-p50  read-p95",
    ]
    for item in result["results"]:
        capacity_label = (
            str(item["capacity_per_layer"])
            if item["capacity_per_layer"] is not None
            else format_bytes(item["max_bytes"])
        )
        process_read = (
            "n/a"
            if item["process_read_bytes"] is None
            else format_bytes(item["process_read_bytes"])
        )
        lines.append(
            f"{capacity_label:>8}"
            f" {item['requests']:>9}"
            f" {item['hit_rate'] * 100:>8.2f}%"
            f" {format_bytes(item['loaded_bytes']):>10}"
            f" {process_read:>10}"
            f" {format_bytes(item['cached_bytes']):>10}"
            f" {item['wall_time_ms']:>8.2f}"
            f" {item['read_latency_p50_ms']:>9.2f}"
            f" {item['read_latency_p95_ms']:>9.2f}"
        )
    return "\n".join(lines)

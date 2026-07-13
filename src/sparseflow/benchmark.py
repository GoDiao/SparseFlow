from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from .cache import ExpertCache
from .loader import read_expert_bytes
from .locator import ExpertLocator

TraceRequest = tuple[int, int]


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
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("requests")
    if not isinstance(raw, list):
        raise ValueError(f"trace must be a JSON list or an object with requests: {source}")

    trace: list[TraceRequest] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            layer, expert = item.get("layer"), item.get("expert")
        elif isinstance(item, list) and len(item) == 2:
            layer, expert = item
        else:
            raise ValueError(f"invalid trace request at index {index}: {item!r}")
        if not isinstance(layer, int) or not isinstance(expert, int):
            raise ValueError(f"trace request must contain integer layer/expert at index {index}")
        trace.append((layer, expert))
    if not trace:
        raise ValueError(f"trace is empty: {source}")
    return trace


def run_expert_benchmark(
    model_dir: str | Path,
    capacities: Iterable[int],
    trace: list[TraceRequest],
) -> dict[str, Any]:
    locator = ExpertLocator(model_dir)
    capacities = list(capacities)
    if not capacities:
        raise ValueError("at least one cache capacity is required")

    locations = {
        request: locator.locate(*request)
        for request in set(trace)
    }

    trace_hash = hashlib.sha256(
        json.dumps([[layer, expert] for layer, expert in trace], separators=(",", ":")).encode()
    ).hexdigest()
    results = []
    for capacity in capacities:
        cache = ExpertCache(capacity)
        read_latencies: list[float] = []
        io_before = _process_read_bytes()
        started = time.perf_counter()
        for layer, expert_id in trace:
            def load() -> dict[str, bytes]:
                read_started = time.perf_counter()
                payloads = read_expert_bytes(locations[(layer, expert_id)])
                read_latencies.append((time.perf_counter() - read_started) * 1000.0)
                return payloads

            cache.get_or_load(layer, expert_id, load)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        io_after = _process_read_bytes()
        stats = cache.stats_dict()
        results.append(
            {
                "capacity_per_layer": capacity,
                **stats,
                "logical_bytes": sum(
                    locations[(layer, expert_id)].nbytes
                    for layer, expert_id in trace
                ),
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
            "requests": len(trace),
            "layers": sorted({layer for layer, _ in trace}),
            "sha256": trace_hash,
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
        f"trace       {result['trace']['requests']} requests, layers {result['trace']['layers']}",
        f"trace sha   {result['trace']['sha256']}",
        "",
        "capacity  requests  hit-rate  loaded       proc-read   cached      wall-ms  read-p50  read-p95",
    ]
    for item in result["results"]:
        process_read = (
            "n/a"
            if item["process_read_bytes"] is None
            else format_bytes(item["process_read_bytes"])
        )
        lines.append(
            f"{item['capacity_per_layer']:>8}"
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

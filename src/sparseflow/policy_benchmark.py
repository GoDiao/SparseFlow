from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .cache import ExpertCache
from .cache_policy import POLICY_VARIANTS, make_cache_policy
from .locator import ExpertLocator
from .trace import RouteTrace, load_route_trace, trace_sha256


def run_policy_replay(
    model_dir: str | Path,
    trace: RouteTrace,
    variant: str,
    max_bytes: int,
    hot_ratio: float = 0.25,
    prefetch_budget_ratio: float = 0.25,
    locator: ExpertLocator | None = None,
) -> dict[str, Any]:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"unknown Stage 7.3 policy variant: {variant}")
    if max_bytes <= 0:
        raise ValueError("policy replay max_bytes must be positive")
    locator = locator or ExpertLocator(model_dir)
    expert_bytes = {
        layer: locator.locate(layer, 0).nbytes
        for layer in trace.layers
    }
    typical_bytes = sorted(expert_bytes.values())[len(expert_bytes) // 2]
    max_hot_entries = int((max_bytes // typical_bytes) * hot_ratio)
    prefetch_budget = int(max_bytes * prefetch_budget_ratio)
    config = POLICY_VARIANTS[variant]
    policy = make_cache_policy(config["cache_policy"], max_hot_entries)
    cache = ExpertCache(max_bytes=max_bytes, policy=policy)
    groups = trace.replay_groups(batch_union=True)
    raw_by_forward = _raw_route_counts(trace)
    previous_decode_routes: dict[int, tuple[int, ...]] = {}
    prediction_history: list[dict[int, tuple[int, ...]]] = []
    ready: dict[tuple[int, int], int] = {}
    prefetch = {
        "submitted": 0,
        "read_bytes": 0,
        "hits": 0,
        "hit_bytes": 0,
        "wasted": 0,
        "wasted_bytes": 0,
    }
    phase_metrics: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "forwards": 0,
            "raw_requests": 0,
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "read_bytes": 0,
            "demand_read_bytes": 0,
            "prefetch_read_bytes": 0,
            "evictions": 0,
            "admission_rejections": 0,
        }
    )
    per_forward: list[dict[str, Any]] = []

    for group in groups:
        cache.begin_forward(group.forward, group.phase)
        route_counts = raw_by_forward[group.forward]
        for layer, counts in route_counts.items():
            cache.observe_routes(layer, counts)

        before = cache.stats_dict()
        read_before = cache.stats.loaded_bytes
        prefetch_read_before = prefetch["read_bytes"]
        if config["prefetch_policy"] == "previous-token":
            stable_routes = _stable_routes(previous_decode_routes, prediction_history)
            predicted = _merge_prediction(stable_routes, cache.policy.hot_keys())
            if previous_decode_routes:
                prediction_history.append(dict(previous_decode_routes))
                prediction_history = prediction_history[-2:]
            selected_bytes = 0
            for layer in sorted(predicted):
                for expert_id in predicted[layer]:
                    key = (layer, expert_id)
                    if cache.peek(layer, expert_id) is not None:
                        continue
                    nbytes = expert_bytes[layer]
                    if selected_bytes + nbytes > prefetch_budget:
                        continue
                    entry = cache.put_sized(layer, expert_id, nbytes, source="prefetch")
                    selected_bytes += nbytes
                    prefetch["submitted"] += 1
                    prefetch["read_bytes"] += nbytes
                    if cache.peek(layer, expert_id) is entry:
                        ready[key] = nbytes
                    _reconcile_ready(cache, ready, prefetch)

        demand_read_bytes = 0
        for layer, expert_id in group.requests:
            key = (layer, expert_id)
            nbytes = expert_bytes[layer]
            entry = cache.lookup(layer, expert_id, expected_nbytes=nbytes)
            if entry is not None:
                used = ready.pop(key, None)
                if used is not None:
                    prefetch["hits"] += 1
                    prefetch["hit_bytes"] += used
                continue
            cache.put_sized(layer, expert_id, nbytes, source="demand")
            demand_read_bytes += nbytes
            _reconcile_ready(cache, ready, prefetch)

        after = cache.stats_dict()
        phase = phase_metrics[group.phase]
        phase["forwards"] += 1
        phase["raw_requests"] += group.raw_requests
        phase["requests"] += len(group.requests)
        phase["hits"] += int(after["hits"] - before["hits"])
        phase["misses"] += int(after["misses"] - before["misses"])
        phase["read_bytes"] += int(cache.stats.loaded_bytes - read_before)
        phase["demand_read_bytes"] += demand_read_bytes
        phase["prefetch_read_bytes"] += prefetch["read_bytes"] - prefetch_read_before
        phase["evictions"] += int(after["evictions"] - before["evictions"])
        phase["admission_rejections"] += int(
            after["admission_rejections"] - before["admission_rejections"]
        )
        per_forward.append(
            {
                "forward": group.forward,
                "phase": group.phase,
                "raw_requests": group.raw_requests,
                "requests": len(group.requests),
                "unique_experts": len(set(group.requests)),
                "demand_read_bytes": demand_read_bytes,
                "total_loaded_bytes": int(cache.stats.loaded_bytes - read_before),
                "cached_bytes_after": cache.cached_bytes,
                "cache_entries_after": cache.entries,
            }
        )
        if group.phase == "decode":
            previous_decode_routes = _requests_by_layer(group.requests)

    _reconcile_ready(cache, ready, prefetch, final=True)
    stats = cache.stats_dict()
    decode = phase_metrics.get("decode", {})
    decode_forwards = int(decode.get("forwards", 0))
    return {
        "variant": variant,
        **config,
        "max_bytes": max_bytes,
        "hot_ratio": hot_ratio,
        "prefetch_budget_ratio": prefetch_budget_ratio,
        "prefetch_budget_bytes": prefetch_budget,
        "max_hot_entries": max_hot_entries,
        "trace_sha256": trace_sha256(trace),
        "trace_raw_requests": trace.raw_requests,
        "trace_effective_requests": sum(len(group.requests) for group in groups),
        "cache": stats,
        "prefetch": prefetch,
        "phase_metrics": dict(phase_metrics),
        "decode_read_bytes_per_forward": (
            int(decode.get("read_bytes", 0)) / decode_forwards if decode_forwards else 0.0
        ),
        "decode_demand_read_bytes_per_forward": (
            int(decode.get("demand_read_bytes", 0)) / decode_forwards
            if decode_forwards
            else 0.0
        ),
        "per_forward": per_forward,
        "reuse": _reuse_metrics(groups),
        "invariants": {
            "requests_equal_hits_plus_misses": stats["requests"] == stats["hits"] + stats["misses"],
            "budget_respected": cache.cached_bytes <= max_bytes,
            "prefetch_hits_bounded": prefetch["hits"] <= prefetch["submitted"],
            "prefetch_bytes_accounted": (
                prefetch["hit_bytes"] + prefetch["wasted_bytes"]
                <= prefetch["read_bytes"]
            ),
        },
    }


def run_policy_sweep(
    model_dir: str | Path,
    trace_paths: Iterable[str | Path],
    byte_budgets: Iterable[int],
    variants: Iterable[str] = tuple(POLICY_VARIANTS),
    hot_ratio: float = 0.25,
    prefetch_budget_ratio: float = 0.25,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in trace_paths)
    budgets = tuple(int(value) for value in byte_budgets)
    selected_variants = tuple(variants)
    if not paths:
        raise ValueError("policy sweep requires at least one trace")
    if not budgets or any(value <= 0 for value in budgets):
        raise ValueError("policy sweep requires positive byte budgets")
    runs = []
    locator = ExpertLocator(model_dir)
    for path in paths:
        trace = load_route_trace(path)
        for budget in budgets:
            for variant in selected_variants:
                replay = run_policy_replay(
                    model_dir,
                    trace,
                    variant,
                    budget,
                    hot_ratio=hot_ratio,
                    prefetch_budget_ratio=prefetch_budget_ratio,
                    locator=locator,
                )
                runs.append({"trace": str(path), **replay})
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_3_policy_sweep",
        "stage": "7.3",
        "agent": "Main Dev",
        "model": str(Path(model_dir).expanduser().resolve()),
        "trace_count": len(paths),
        "byte_budgets": list(budgets),
        "variants": list(selected_variants),
        "hot_ratio": hot_ratio,
        "prefetch_budget_ratio": prefetch_budget_ratio,
        "runs": runs,
        "summary": _summarize_runs(runs),
        "all_invariants_pass": all(
            all(run["invariants"].values()) for run in runs
        ),
        "boundary": "metadata-only replay using real route traces and real expert byte sizes",
    }


def discover_trace_paths(trace_dirs: Iterable[str | Path]) -> tuple[Path, ...]:
    result = []
    for directory in trace_dirs:
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise ValueError(f"trace directory does not exist: {root}")
        result.extend(sorted(root.glob("*.json")))
    return tuple(result)


def _raw_route_counts(trace: RouteTrace) -> dict[int, dict[int, Counter[int]]]:
    result: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    for group in trace.groups:
        for layer, expert_id in group.requests:
            result[group.forward][layer][expert_id] += 1
    return result


def _requests_by_layer(requests) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = defaultdict(list)
    for layer, expert_id in requests:
        if expert_id not in result[layer]:
            result[layer].append(expert_id)
    return {layer: tuple(experts) for layer, experts in result.items()}


def _merge_prediction(
    previous: dict[int, tuple[int, ...]],
    hot_keys: tuple[tuple[int, int], ...],
) -> dict[int, tuple[int, ...]]:
    merged: dict[int, list[int]] = defaultdict(list)
    for layer, expert_id in hot_keys:
        if expert_id not in merged[layer]:
            merged[layer].append(expert_id)
    for layer, experts in previous.items():
        for expert_id in experts:
            if expert_id not in merged[layer]:
                merged[layer].append(expert_id)
    return {layer: tuple(experts) for layer, experts in merged.items()}


def _stable_routes(
    current: dict[int, tuple[int, ...]],
    previous: list[dict[int, tuple[int, ...]]],
) -> dict[int, tuple[int, ...]]:
    if len(previous) < 2:
        return {}
    result = {}
    for layer, experts in current.items():
        prior = set(experts)
        for history in previous:
            prior.intersection_update(history.get(layer, ()))
        stable = tuple(expert_id for expert_id in experts if expert_id in prior)
        if stable:
            result[layer] = stable
    return result


def _reconcile_ready(
    cache: ExpertCache,
    ready: dict[tuple[int, int], int],
    prefetch: dict[str, int],
    final: bool = False,
) -> None:
    live = set(cache.cached_keys()) if not final else set()
    for key in tuple(ready):
        if key not in live:
            prefetch["wasted"] += 1
            prefetch["wasted_bytes"] += ready.pop(key)


def _reuse_metrics(groups) -> dict[str, float | int]:
    last_seen: dict[tuple[int, int], int] = {}
    distances = []
    request_index = 0
    for group in groups:
        for key in group.requests:
            if key in last_seen:
                distances.append(request_index - last_seen[key])
            last_seen[key] = request_index
            request_index += 1
    ordered = sorted(distances)
    return {
        "reused_requests": len(distances),
        "mean_distance": sum(distances) / len(distances) if distances else 0.0,
        "p50_distance": ordered[len(ordered) // 2] if ordered else 0,
        "p95_distance": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0,
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["variant"], run["max_bytes"])].append(run)
    result = []
    for (variant, budget), items in sorted(grouped.items()):
        requests = sum(item["cache"]["requests"] for item in items)
        hits = sum(item["cache"]["hits"] for item in items)
        decode_forwards = sum(
            item["phase_metrics"].get("decode", {}).get("forwards", 0)
            for item in items
        )
        decode_bytes = sum(
            item["phase_metrics"].get("decode", {}).get("read_bytes", 0)
            for item in items
        )
        decode_demand_bytes = sum(
            item["phase_metrics"].get("decode", {}).get("demand_read_bytes", 0)
            for item in items
        )
        result.append(
            {
                "variant": variant,
                "max_bytes": budget,
                "traces": len(items),
                "requests": requests,
                "hits": hits,
                "hit_rate": hits / requests if requests else 0.0,
                "decode_forwards": decode_forwards,
                "decode_read_bytes": decode_bytes,
                "decode_read_bytes_per_forward": (
                    decode_bytes / decode_forwards if decode_forwards else 0.0
                ),
                "decode_demand_read_bytes": decode_demand_bytes,
                "decode_demand_read_bytes_per_forward": (
                    decode_demand_bytes / decode_forwards if decode_forwards else 0.0
                ),
                "prefetch_read_bytes": sum(item["prefetch"]["read_bytes"] for item in items),
                "prefetch_hit_bytes": sum(item["prefetch"]["hit_bytes"] for item in items),
                "prefetch_wasted_bytes": sum(
                    item["prefetch"]["wasted_bytes"] for item in items
                ),
                "all_invariants_pass": all(
                    all(item["invariants"].values()) for item in items
                ),
            }
        )
    return result

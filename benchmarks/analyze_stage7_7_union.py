"""Analyze and optionally replay Stage 7.7 real route unions.

Metadata analysis is complete-model-free.  ``--replay-real`` additionally reads
canonical INT8 expert payloads through the production ExpertCache provider.

[Main Dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from sparseflow.benchmark import parse_byte_budgets
from sparseflow.multirequest_trace import MultiRequestTrace, analyze_schedule, load_multi_request_trace


def process_read_bytes() -> int | None:
    try:
        content = Path("/proc/self/io").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "read_bytes":
            return int(value.strip())
    return None


def replay_schedule(
    trace: MultiRequestTrace,
    schedule_id: str,
    int8_container: Path,
    budget: int,
    mode: str = "union",
    working_set_ratio: float = 1.0,
) -> dict[str, Any]:
    from sparseflow.cache import ExpertCache
    from sparseflow.int8_container import Int8ExpertIndex
    from sparseflow.loader import ShardReader
    from sparseflow.cohort_policy import SessionWorkingSet, partition_working_sets
    if not 0.0 < working_set_ratio <= 1.0:
        raise ValueError("working_set_ratio must be in (0, 1]")

    class RawInt8CacheReplay:
        """Replay storage/cache behavior without decoding or running kernels.

        Route-union analysis must measure only admission, eviction, and raw
        payload I/O.  The normal streaming provider intentionally prepares
        decoded/native tensors on a miss, which would make this a compute
        benchmark instead of a cache replay.
        """

        def __init__(self, root: Path, cache: ExpertCache, reader: ShardReader):
            self.index = Int8ExpertIndex.from_dir(root)
            self.cache = cache
            self.reader = reader

        def begin_forward(self, forward: int, phase: str) -> None:
            self.cache.begin_forward(forward, phase)

        def prepare(self, layer: int, expert_ids: tuple[int, ...]) -> None:
            for expert_id in expert_ids:
                self.index.locate(layer, int(expert_id))

        def get(self, layer: int, expert_id: int):
            location = self.index.locate(layer, int(expert_id))
            cached = self.cache.lookup(layer, int(expert_id), location.nbytes)
            if cached is not None:
                return cached
            payloads: dict[str, bytes] = {}
            for part in location.parts:
                payloads[f"{part.part}.data"] = self.reader.read(
                    location.file, part.data_offset, part.data_nbytes
                )
                payloads[f"{part.part}.scales"] = self.reader.read(
                    location.file, part.scale_offset, part.scale_nbytes
                )
            return self.cache.put_loaded(layer, int(expert_id), payloads)

        def finish_generation(self) -> None:
            pass

        def counters(self) -> dict[str, Any]:
            cache = self.cache.counters()
            return {
                "backend_id": "sparseflow-int8-raw-cache-replay",
                "reader_calls": self.reader.read_calls,
                "reader_bytes": self.reader.read_bytes,
                "requests": cache["requests"],
                "cache_hits": cache["hits"],
                "cache_misses": cache["misses"],
                "hit_rate": cache["hit_rate"],
                "cache_evictions": cache["evictions"],
                "cached_experts": cache["cache_entries"],
                "cached_bytes": cache["cached_bytes"],
                "loaded_bytes": cache["loaded_bytes"],
                "hit_bytes": cache["hit_bytes"],
                "miss_bytes": cache["miss_bytes"],
                "admission_rejections": cache["admission_rejections"],
                "pinned_entries": cache["pinned_entries"],
                "pinned_bytes": cache["pinned_bytes"],
            }

        def close(self) -> None:
            self.cache.clear()

    schedule = trace.schedule(schedule_id)
    sessions = trace.session_map()
    cache = ExpertCache(max_bytes=budget)
    reader = ShardReader()
    provider = RawInt8CacheReplay(int8_container, cache, reader)
    expert_nbytes = provider.index.locate(provider.index.layers[0], 0).nbytes
    assignments = 0
    union_requests = 0
    cohort_count = 0
    cohort_overflow = 0
    cohort_max_working_set = 0
    started = time.perf_counter()
    read_before = process_read_bytes()
    try:
        for step in schedule.steps:
            if mode == "union":
                layer_work = []
                for layer in range(40):
                    union: set[int] = set()
                    layer_assignments = 0
                    for session_id, forward in step.sessions:
                        group = sessions[session_id].group(forward)
                        selected = [
                            expert
                            for current_layer, expert in group.requests
                            if current_layer == layer
                        ]
                        layer_assignments += len(selected)
                        union.update(selected)
                    layer_work.append((layer, tuple(sorted(union)), layer_assignments))
            elif mode == "round-robin":
                layer_work = []
                for session_id, forward in step.sessions:
                    group = sessions[session_id].group(forward)
                    for layer in range(40):
                        selected = tuple(
                            expert
                            for current_layer, expert in group.requests
                            if current_layer == layer
                        )
                        layer_work.append((layer, selected, len(selected)))
            elif mode == "cache-aware":
                working_sets = []
                for session_id, forward in step.sessions:
                    group = sessions[session_id].group(forward)
                    items = frozenset(
                        (layer, expert)
                        for layer, expert in group.requests
                    )
                    working_sets.append(SessionWorkingSet(session_id, items))
                max_items = max(1, int(budget * working_set_ratio // expert_nbytes))
                cohorts = partition_working_sets(working_sets, max_items)
                cohort_count += len(cohorts)
                cohort_overflow += sum(1 for cohort in cohorts if cohort.overflow)
                cohort_max_working_set = max(
                    cohort_max_working_set,
                    max((len(cohort.working_set) for cohort in cohorts), default=0),
                )
                layer_work = []
                for cohort in cohorts:
                    active = set(cohort.session_ids)
                    for layer in range(40):
                        selected: list[int] = []
                        assignments_for_layer = 0
                        for session_id, forward in step.sessions:
                            if session_id not in active:
                                continue
                            group = sessions[session_id].group(forward)
                            current = [
                                expert
                                for current_layer, expert in group.requests
                                if current_layer == layer
                            ]
                            selected.extend(current)
                            assignments_for_layer += len(current)
                        layer_work.append((layer, tuple(sorted(set(selected))), assignments_for_layer))
            else:
                raise ValueError(f"unknown replay mode: {mode}")
            for layer, expert_ids, layer_assignments in layer_work:
                if not expert_ids:
                    continue
                provider.begin_forward(step.scheduler_step, "decode")
                provider.prepare(layer, tuple(sorted(set(expert_ids))))
                for expert_id in expert_ids:
                    provider.get(layer, expert_id)
                assignments += layer_assignments
                union_requests += len(expert_ids)
        provider.finish_generation()
        counters = provider.counters()
        counters["pinned_entries"] = cache.pinned_entries
        counters["pinned_bytes"] = cache.pinned_bytes
    finally:
        provider.close()
        reader.close()
    read_after = process_read_bytes()
    return {
        "schedule_id": schedule_id,
        "mode": mode,
        "raw_replay": True,
        "budget_bytes": budget,
        "assignments": assignments,
        "union_requests": union_requests,
        "union_reuse_assignments": assignments - union_requests,
        "union_compression": assignments / union_requests if union_requests else 0.0,
        "cohort_count": cohort_count,
        "cohort_overflow": cohort_overflow,
        "cohort_max_working_set": cohort_max_working_set,
        "working_set_ratio": working_set_ratio if mode == "cache-aware" else None,
        "cache": counters,
        "provider_read_bytes": int(counters.get("reader_bytes", 0)),
        "provider_read_calls": int(counters.get("reader_calls", 0)),
        "process_read_bytes": (
            read_after - read_before
            if read_before is not None and read_after is not None
            else None
        ),
        "wall_seconds": time.perf_counter() - started,
        "budget_respected": int(counters.get("cached_bytes", 0)) <= budget,
        "leases_released": int(counters.get("pinned_entries", 0)) == 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage 7.7 route union.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--int8-container")
    parser.add_argument("--budgets", default="4GiB,8GiB")
    parser.add_argument("--replay-real", action="store_true")
    parser.add_argument(
        "--replay-schedules",
        default="sync-b4,sync-b8",
        help="Comma-separated schedules for real pread replay; metadata always covers all schedules.",
    )
    parser.add_argument(
        "--replay-modes",
        default="union,round-robin",
        help="Comma-separated real replay modes.",
    )
    args = parser.parse_args(argv)
    if args.replay_real and not args.int8_container:
        parser.error("--replay-real requires --int8-container")

    trace = load_multi_request_trace(args.trace)
    output = Path(args.output).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_7_route_union",
        "agent": "Main Dev",
        "trace": {
            "path": str(Path(args.trace).expanduser().resolve()),
            "sha256": trace.as_dict()["trace_sha256"],
            "schema_version": trace.schema_version,
            "sessions": len(trace.sessions),
        },
        "metadata": [analyze_schedule(trace, schedule.schedule_id) for schedule in trace.schedules],
        "replay": [],
    }
    if args.replay_real:
        container = Path(args.int8_container).expanduser().resolve()
        selected = {
            item.strip()
            for item in args.replay_schedules.split(",")
            if item.strip()
        }
        available = {schedule.schedule_id for schedule in trace.schedules}
        unknown = selected - available
        if unknown:
            parser.error(f"unknown replay schedules: {sorted(unknown)}")
        for schedule in trace.schedules:
            if schedule.schedule_id not in selected:
                continue
            for mode in (item.strip() for item in args.replay_modes.split(",")):
                if not mode:
                    continue
                for budget in parse_byte_budgets(args.budgets):
                    result["replay"].append(
                        replay_schedule(trace, schedule.schedule_id, container, budget, mode)
                    )
    result["agent"] = "Main Dev"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "schedules": len(result["metadata"]),
        "replay_cells": len(result["replay"]),
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

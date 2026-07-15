from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    delta,
    evict_file_pages,
    git_snapshot,
    host_snapshot,
    model_snapshot,
    model_weight_files,
    percentile,
    process_snapshot,
    write_json,
)


def make_batches(layers: tuple[int, ...], count: int, seed: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if count < 1 or count % 8:
        raise ValueError("count must be a positive multiple of eight")
    rng = random.Random(seed)
    return tuple(
        (layers[index % len(layers)], tuple(rng.sample(range(256), 8)))
        for index in range(count // 8)
    )


def run_io_case(locator, batches, workers: int, cache_state: str) -> dict[str, Any]:
    from sparseflow.loader import ShardReader

    locations = tuple(
        tuple(locator.locate(layer, expert_id) for expert_id in expert_ids)
        for layer, expert_ids in batches
    )
    if cache_state == "workload-warm":
        with ShardReader() as warm_reader:
            for batch in locations:
                warm_reader.read_locations(batch)
    elif cache_state == "model-cold":
        control = evict_file_pages(model_weight_files(locator.model_dir))
        if control["failures"]:
            raise RuntimeError(f"page-cache eviction failed: {control['failures'][:3]}")
    else:
        raise ValueError(f"unknown I/O cache state: {cache_state}")

    before = process_snapshot()
    started = time.perf_counter()
    latencies = []
    useful_bytes = 0
    with ShardReader() as reader:
        def read_batch(batch):
            task_started = time.perf_counter()
            payloads = reader.read_locations(batch)
            elapsed = time.perf_counter() - task_started
            return elapsed, sum(len(part) for expert in payloads.values() for part in expert.values())

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for elapsed, nbytes in executor.map(read_batch, locations):
                latencies.append(elapsed)
                useful_bytes += nbytes
        reader_calls = reader.read_calls
        reader_bytes = reader.read_bytes
    wall_seconds = time.perf_counter() - started
    after = process_snapshot()
    return {
        "workers": workers,
        "cache_state": cache_state,
        "expert_requests": sum(len(batch) for batch in locations),
        "batch_requests": len(locations),
        "useful_bytes": useful_bytes,
        "reader_calls": reader_calls,
        "reader_bytes": reader_bytes,
        "wall_seconds": wall_seconds,
        "logical_gib_per_second": useful_bytes / 1024**3 / wall_seconds,
        "batch_latency_p50_seconds": percentile(latencies, 0.50),
        "batch_latency_p95_seconds": percentile(latencies, 0.95),
        "process_metrics_delta": delta(before, after),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.4 buffered expert I/O benchmark.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--threads", default="1,2,4,8")
    parser.add_argument("--states", default="model-cold,workload-warm")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    workers = tuple(int(value) for value in args.threads.split(","))
    states = tuple(args.states.split(","))
    if any(value < 1 for value in workers):
        parser.error("threads must contain positive values")

    from sparseflow.locator import ExpertLocator

    root = Path.cwd().resolve()
    model = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    locator = ExpertLocator(model)
    batches = make_batches(locator.layers, args.count, args.seed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_4_expert_io",
        "stage": "7.4",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "protocol": {
            "seed": args.seed,
            "expert_requests": args.count,
            "experts_per_batch": 8,
            "expert_bytes": locator.locate(locator.layers[0], 0).nbytes,
            "io_api": "buffered os.pread with persistent descriptors",
            "direct_io": "not measured; production Python loader uses buffered positional reads",
        },
        "cases": [],
    }
    output = Path(args.output).resolve()
    for state in states:
        for thread_count in workers:
            case = run_io_case(locator, batches, thread_count, state)
            result["cases"].append(case)
            write_json(output, result)
            print(
                f"state={state} threads={thread_count} "
                f"GiB/s={case['logical_gib_per_second']:.3f}",
                flush=True,
            )
    result["selected"] = {
        state: max(
            (case for case in result["cases"] if case["cache_state"] == state),
            key=lambda case: case["logical_gib_per_second"],
        )["workers"]
        for state in states
    }
    write_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

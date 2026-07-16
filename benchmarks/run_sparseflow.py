from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    delta,
    evict_file_pages,
    filesystem_snapshot,
    git_snapshot,
    host_snapshot,
    model_snapshot,
    model_weight_files,
    numeric_delta,
    parse_bytes,
    percentile,
    process_snapshot,
    sha256_text,
    write_json,
)
from .run_cpu import load_manifest


VARIANTS: dict[str, dict[str, Any]] = {
    "C3-R": {
        "mode": "resident",
        "cache_policy": "none",
        "prefetch_policy": "none",
        "prefetch_workers": 0,
    },
    "C3-S0": {
        "mode": "streaming",
        "cache_policy": "none",
        "prefetch_policy": "none",
        "prefetch_workers": 0,
    },
    "C3-S1": {
        "mode": "streaming",
        "cache_policy": "lru",
        "prefetch_policy": "none",
        "prefetch_workers": 0,
    },
    "C3-S2": {
        "mode": "streaming",
        "cache_policy": "hot",
        "prefetch_policy": "none",
        "prefetch_workers": 0,
    },
    "C3-S3": {
        "mode": "streaming",
        "cache_policy": "heat",
        "prefetch_policy": "none",
        "prefetch_workers": 0,
    },
    "C3-S4": {
        "mode": "streaming",
        "cache_policy": "heat",
        "prefetch_policy": "previous-token",
        "prefetch_workers": 2,
    },
}


def configure_threads(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if threads:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)


def validate_cache_state(cache_state: str, warmup: int, runs: int) -> None:
    if cache_state == "workload-warm" and warmup < 1:
        raise ValueError("workload-warm requires at least one warmup run")
    if cache_state in {"app-cold", "model-cold"} and (warmup != 0 or runs != 1):
        raise ValueError(f"{cache_state} requires --warmup 0 --runs 1")


def result_record(
    run_index: int,
    row: dict[str, Any],
    generated: dict[str, Any],
    before_process: dict[str, int],
    after_process: dict[str, int],
    before_provider: dict[str, Any] | None,
    after_provider: dict[str, Any] | None,
    wall_seconds: float,
) -> dict[str, Any]:
    decode_seconds = float(generated["decode_seconds"])
    decode_tokens = max(0, int(generated["generated_tokens"]) - 1)
    prefill_seconds = float(generated["prefill_seconds"])
    input_tokens = len(generated["input_ids"])
    decode_latencies = [float(value) for value in generated["decode_token_seconds"]]
    provider_delta = (
        numeric_delta(before_provider, after_provider)
        if before_provider is not None and after_provider is not None
        else None
    )
    return {
        "run_index": run_index,
        "prompt_id": row["id"],
        "group": row.get("group"),
        "category": row.get("category"),
        "input_tokens": input_tokens,
        "generated_tokens": int(generated["generated_tokens"]),
        "timing": {
            "time_to_first_token_seconds": prefill_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": input_tokens / prefill_seconds
            if prefill_seconds > 0
            else None,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_tokens / decode_seconds
            if decode_seconds > 0 and decode_tokens
            else None,
            "decode_token_seconds": decode_latencies,
            "per_token_latency_p50_seconds": percentile(decode_latencies, 0.50),
            "per_token_latency_p95_seconds": percentile(decode_latencies, 0.95),
            "prefetch_finalize_seconds": float(generated["prefetch_finalize_seconds"]),
            "end_to_end_seconds": wall_seconds,
        },
        "memory": {
            **generated["memory"],
            "process_after": after_process,
        },
        "process_metrics_delta": delta(before_process, after_process),
        "provider_delta": provider_delta,
        "cache_after": generated["cache"],
        "prefetch_after": generated["prefetch"],
        "telemetry": generated["telemetry"],
        "route_audit": generated["route_audit"],
        "runtime_identity": generated["runtime_identity"],
        "quality": {
            "generated_ids": generated["generated_ids"],
            "text": generated["text"],
            "logit_fingerprints": generated["logit_fingerprints"],
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    decode_speeds = [
        item["timing"]["decode_tokens_per_second"]
        for item in records
        if item["timing"]["decode_tokens_per_second"] is not None
    ]
    ttft = [item["timing"]["time_to_first_token_seconds"] for item in records]
    latencies = [
        value
        for item in records
        for value in item["timing"]["decode_token_seconds"]
    ]
    logical_reads = [
        int((item["provider_delta"] or {}).get("reader_bytes", 0))
        for item in records
    ]
    process_reads = [int(item["process_metrics_delta"].get("read_bytes", 0)) for item in records]
    return {
        "run_count": len(records),
        "median_ttft_seconds": statistics.median(ttft) if ttft else None,
        "median_decode_tokens_per_second": statistics.median(decode_speeds)
        if decode_speeds
        else None,
        "decode_latency_p50_seconds": percentile(latencies, 0.50),
        "decode_latency_p95_seconds": percentile(latencies, 0.95),
        "median_provider_read_bytes": statistics.median(logical_reads)
        if logical_reads
        else None,
        "median_process_read_bytes": statistics.median(process_reads)
        if process_reads
        else None,
        "peak_rss_bytes": max(
            int(item["memory"]["process_peak_rss"]) for item in records
        ),
        "generated_ids_exact_across_runs": len(
            {tuple(item["quality"]["generated_ids"]) for item in records}
        )
        == 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 7.4 C3-R/C3-S CPU benchmark."
    )
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--cache-bytes", default="4GiB")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prefetch-workers", type=int, default=None)
    parser.add_argument("--prefetch-budget-ratio", type=float, default=0.10)
    parser.add_argument("--hot-ratio", type=float, default=0.25)
    parser.add_argument("--coalesce-gap", type=int, default=0)
    parser.add_argument(
        "--cache-state",
        choices=("uncontrolled", "app-cold", "workload-warm", "model-cold"),
        default="workload-warm",
    )
    parser.add_argument("--telemetry-level", choices=("none", "summary", "layer"), default="summary")
    args = parser.parse_args(argv)

    if args.threads < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("threads/runs must be positive and warmup must be non-negative")
    if args.max_new_tokens < 2:
        parser.error("Stage 7.4 decode benchmark requires at least two generated tokens")
    try:
        validate_cache_state(args.cache_state, args.warmup, args.runs)
        cache_bytes = parse_bytes(args.cache_bytes)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path.cwd().resolve()
    model_dir = Path(args.model).expanduser()
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_dir = model_dir.resolve()
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    rows = load_manifest(manifest, args.limit or None)
    if not rows:
        parser.error("manifest contains no rows")

    configure_threads(args.threads)
    import torch
    import transformers

    from sparseflow.text_runtime import Qwen36TextRuntime

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    config = dict(VARIANTS[args.variant])
    if args.prefetch_workers is not None:
        config["prefetch_workers"] = args.prefetch_workers
    effective_cache_bytes = cache_bytes if config["mode"] == "streaming" else None
    output = Path(args.output).expanduser()
    result: dict[str, Any] = {
        "schema_version": 3,
        "kind": "sparseflow_stage7_4_benchmark",
        "stage": "7.4",
        "agent": "Main Dev",
        "experiment_id": (
            f"{args.variant.lower()}-{cache_bytes // 1024**2}m-"
            f"{args.cache_state}-t{args.threads}-o{args.max_new_tokens}"
        ),
        "backend": "sparseflow_cpu_resident"
        if config["mode"] == "resident"
        else "sparseflow_cpu_streaming",
        "variant": args.variant,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bf16",
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "load_mode": "memory-native",
            **config,
        },
        "host": host_snapshot(),
        "filesystem": filesystem_snapshot(model_dir),
        "git": git_snapshot(root),
        "workload": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_text(
                manifest.read_text(encoding="utf-8").splitlines(keepends=True)
            ),
            "prompt_ids": [row["id"] for row in rows],
            "rows": len(rows),
            "runs": args.runs,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": 1,
            "sampling": "greedy",
        },
        "storage_policy": {
            "cache_state": args.cache_state,
            "cache_bytes": effective_cache_bytes,
            "cache_policy": config["cache_policy"],
            "prefetch_policy": config["prefetch_policy"],
            "prefetch_workers": config["prefetch_workers"],
            "prefetch_budget_ratio": args.prefetch_budget_ratio,
            "hot_ratio": args.hot_ratio,
            "coalesce_gap": args.coalesce_gap,
        },
        "load": {},
        "page_cache_control": None,
        "warmup_records": [],
        "runs": [],
    }

    load_before = process_snapshot()
    load_started = time.perf_counter()
    runtime = Qwen36TextRuntime.from_pretrained(
        model_dir,
        mode=config["mode"],
        dtype="bf16",
        cache_slots=None,
        cache_bytes=effective_cache_bytes,
        prefetch_workers=config["prefetch_workers"],
        coalesce_gap=args.coalesce_gap,
        load_mode="memory-native",
        cache_policy=config["cache_policy"],
        prefetch_policy=config["prefetch_policy"],
        prefetch_budget_ratio=args.prefetch_budget_ratio,
        hot_ratio=args.hot_ratio,
        telemetry_level=args.telemetry_level,
        experts_implementation="eager",
    )
    result["load"] = {
        "seconds": time.perf_counter() - load_started,
        "metrics_delta": delta(load_before, process_snapshot()),
        "loader": runtime.loader_report,
    }
    write_json(output, result)

    try:
        if args.cache_state == "model-cold":
            result["page_cache_control"] = evict_file_pages(
                model_weight_files(model_dir)
            )

        for warmup_index in range(args.warmup):
            row = rows[warmup_index % len(rows)]
            started = time.perf_counter()
            warm = runtime.greedy_generate(
                row["text"],
                max_new_tokens=args.max_new_tokens,
                record_logit_fingerprints=False,
            )
            result["warmup_records"].append(
                {
                    "index": warmup_index,
                    "prompt_id": row["id"],
                    "seconds": time.perf_counter() - started,
                    "generated_ids": warm["generated_ids"],
                }
            )
            write_json(output, result)

        for run_index in range(args.runs):
            for row in rows:
                if runtime.telemetry is not None:
                    runtime.telemetry.reset()
                before_process = process_snapshot()
                before_provider = runtime.provider.counters() if runtime.provider else None
                started = time.perf_counter()
                generated = runtime.greedy_generate(
                    row["text"],
                    max_new_tokens=args.max_new_tokens,
                    record_logit_fingerprints=True,
                )
                wall_seconds = time.perf_counter() - started
                after_provider = runtime.provider.counters() if runtime.provider else None
                after_process = process_snapshot()
                result["runs"].append(
                    result_record(
                        run_index,
                        row,
                        generated,
                        before_process,
                        after_process,
                        before_provider,
                        after_provider,
                        wall_seconds,
                    )
                )
                write_json(output, result)
    finally:
        runtime.close()

    result["summary"] = summarize(result["runs"])
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"results={output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

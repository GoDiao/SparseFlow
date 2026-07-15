from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import git_snapshot, host_snapshot, model_snapshot, percentile, write_json
from .run_cpu import load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate Stage 7.4 C3-R CPU threads.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", default="1,5,10,20")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    thread_values = tuple(int(value) for value in args.threads.split(","))
    if not thread_values or any(value < 1 for value in thread_values):
        parser.error("threads must contain positive integers")
    if args.runs < 1 or args.max_new_tokens < 2:
        parser.error("runs must be positive and max-new-tokens must be at least two")

    root = Path.cwd().resolve()
    model = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    manifest = (
        (root / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest).resolve()
    )
    row = load_manifest(manifest, 1)[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(max(thread_values))
    os.environ["MKL_NUM_THREADS"] = str(max(thread_values))

    import torch
    import transformers
    from sparseflow.text_runtime import Qwen36TextRuntime

    torch.set_num_interop_threads(4)
    load_started = time.perf_counter()
    runtime = Qwen36TextRuntime.from_pretrained(
        model,
        mode="resident",
        dtype="bf16",
        cache_slots=None,
        cache_bytes=None,
        load_mode="memory-native",
        telemetry_level="none",
        experts_implementation="eager",
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_4_thread_calibration",
        "stage": "7.4",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "variant": "C3-R",
            "dtype": "bf16",
            "load_seconds": time.perf_counter() - load_started,
        },
        "workload": {
            "prompt_id": row["id"],
            "max_new_tokens": args.max_new_tokens,
            "warmup_per_thread": 1,
            "measured_runs_per_thread": args.runs,
        },
        "threads": [],
    }
    output = Path(args.output).resolve()
    write_json(output, result)
    try:
        reference_ids = None
        for threads in thread_values:
            torch.set_num_threads(threads)
            runtime.greedy_generate(row["text"], max_new_tokens=args.max_new_tokens)
            records = []
            for run_index in range(args.runs):
                generated = runtime.greedy_generate(
                    row["text"], max_new_tokens=args.max_new_tokens
                )
                decode_tokens = generated["generated_tokens"] - 1
                decode_seconds = generated["decode_seconds"]
                ids = generated["generated_ids"]
                if reference_ids is None:
                    reference_ids = ids
                records.append(
                    {
                        "run": run_index,
                        "prefill_seconds": generated["prefill_seconds"],
                        "decode_seconds": decode_seconds,
                        "decode_tokens_per_second": decode_tokens / decode_seconds,
                        "decode_token_seconds": generated["decode_token_seconds"],
                        "generated_ids": ids,
                        "generated_ids_exact": ids == reference_ids,
                    }
                )
            speeds = [record["decode_tokens_per_second"] for record in records]
            latencies = [
                value for record in records for value in record["decode_token_seconds"]
            ]
            result["threads"].append(
                {
                    "threads": threads,
                    "records": records,
                    "median_prefill_seconds": statistics.median(
                        record["prefill_seconds"] for record in records
                    ),
                    "median_decode_tokens_per_second": statistics.median(speeds),
                    "decode_latency_p50_seconds": percentile(latencies, 0.50),
                    "decode_latency_p95_seconds": percentile(latencies, 0.95),
                }
            )
            write_json(output, result)
    finally:
        runtime.close()

    winner = max(result["threads"], key=lambda item: item["median_decode_tokens_per_second"])
    result["selected_threads"] = winner["threads"]
    result["all_generated_ids_exact"] = all(
        record["generated_ids_exact"]
        for item in result["threads"]
        for record in item["records"]
    )
    write_json(output, result)
    print(json.dumps({"selected_threads": result["selected_threads"]}))
    return 0 if result["all_generated_ids_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

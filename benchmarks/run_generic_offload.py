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
    percentile,
    process_snapshot,
    sha256_text,
    write_json,
)
from .run_cpu import encode_chat, greedy_generate_timed, load_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 7.4 C2 Accelerate generic disk-offload baseline."
    )
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--offload-dir", default=".cache/stage7_4/generic-offload")
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--cache-state",
        choices=("uncontrolled", "workload-warm", "model-cold"),
        default="model-cold",
    )
    args = parser.parse_args(argv)

    if args.threads < 1 or args.runs < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("threads/runs/max-new-tokens must be positive; warmup may be zero")
    if args.cache_state == "workload-warm" and args.warmup < 1:
        parser.error("workload-warm requires at least one warmup")
    if args.cache_state == "model-cold" and (args.warmup != 0 or args.runs != 1):
        parser.error("model-cold requires --warmup 0 --runs 1")

    root = Path.cwd().resolve()
    model_dir = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    offload_dir = (
        (root / args.offload_dir).resolve()
        if not Path(args.offload_dir).is_absolute()
        else Path(args.offload_dir).resolve()
    )
    manifest = (
        (root / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest).resolve()
    )
    if not (offload_dir / "index.json").is_file():
        parser.error(
            "generic offload index is missing; run benchmarks.prepare_generic_offload first"
        )
    rows = load_manifest(manifest, args.limit or None)
    if not rows:
        parser.error("manifest contains no rows")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    import transformers
    from accelerate import disk_offload, init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    result: dict[str, Any] = {
        "schema_version": 3,
        "kind": "sparseflow_stage7_4_generic_offload",
        "stage": "7.4",
        "agent": "Main Dev",
        "experiment_id": f"c2-generic-{args.cache_state}-t{args.threads}-o{args.max_new_tokens}",
        "backend": "accelerate_cpu_generic_disk_offload",
        "variant": "C2",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bf16",
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "execution_device": "cpu",
            "offload_backend": "accelerate.disk_offload",
            "preload_module_classes": ["Qwen3_5MoeExperts"],
        },
        "host": host_snapshot(),
        "filesystem": filesystem_snapshot(offload_dir),
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
            "offload_dir": str(offload_dir),
            "granularity": "generic module/tensor hooks",
        },
        "load": {},
        "page_cache_control": None,
        "warmup_records": [],
        "runs": [],
    }
    output = Path(args.output).resolve()

    load_before = process_snapshot()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    # Checkpoint parameters live in Accelerate's disk map. Small deterministic
    # buffers such as rotary ``inv_freq`` are not checkpoint entries and must
    # be constructed normally on CPU rather than left unresolved on meta.
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config)
    model = disk_offload(
        model,
        offload_dir,
        execution_device=torch.device("cpu"),
        offload_buffers=False,
        preload_module_classes=["Qwen3_5MoeExperts"],
    ).eval()
    result["load"] = {
        "seconds": time.perf_counter() - load_started,
        "metrics_delta": delta(load_before, process_snapshot()),
    }
    write_json(output, result)
    encoded = {row["id"]: encode_chat(tokenizer, row["text"]) for row in rows}

    if args.cache_state == "model-cold":
        result["page_cache_control"] = evict_file_pages(
            sorted(offload_dir.glob("*.dat"))
        )

    for warmup_index in range(args.warmup):
        row = rows[warmup_index % len(rows)]
        started = time.perf_counter()
        with torch.inference_mode():
            timed = greedy_generate_timed(
                model, encoded[row["id"]], args.max_new_tokens, torch
            )
        result["warmup_records"].append(
            {
                "index": warmup_index,
                "prompt_id": row["id"],
                "seconds": time.perf_counter() - started,
                "output_token_ids": [
                    int(value)
                    for value in timed["output"][0, encoded[row["id"]]["input_ids"].shape[-1] :].tolist()
                ],
            }
        )
        write_json(output, result)

    for run_index in range(args.runs):
        for row in rows:
            inputs = encoded[row["id"]]
            input_tokens = int(inputs["input_ids"].shape[-1])
            before = process_snapshot()
            started = time.perf_counter()
            timed = greedy_generate_timed(model, inputs, args.max_new_tokens, torch)
            wall_seconds = time.perf_counter() - started
            after = process_snapshot()
            generated = timed["output"][0, input_tokens:]
            latencies = [float(value) for value in timed["decode_token_seconds"]]
            result["runs"].append(
                {
                    "run_index": run_index,
                    "prompt_id": row["id"],
                    "input_tokens": input_tokens,
                    "generated_tokens": int(generated.shape[-1]),
                    "timing": {
                        "time_to_first_token_seconds": float(timed["prefill_seconds"]),
                        "prefill_seconds": float(timed["prefill_seconds"]),
                        "decode_seconds": float(timed["decode_seconds"]),
                        "decode_tokens": int(timed["decode_tokens"]),
                        "decode_tokens_per_second": (
                            int(timed["decode_tokens"]) / float(timed["decode_seconds"])
                            if timed["decode_seconds"] and timed["decode_tokens"]
                            else None
                        ),
                        "decode_token_seconds": latencies,
                        "per_token_latency_p50_seconds": percentile(latencies, 0.50),
                        "per_token_latency_p95_seconds": percentile(latencies, 0.95),
                        "end_to_end_seconds": wall_seconds,
                    },
                    "process_metrics_delta": delta(before, after),
                    "memory": {"process_after": after},
                    "quality": {
                        "generated_ids": [int(value) for value in generated.tolist()],
                        "text": tokenizer.decode(generated, skip_special_tokens=True),
                    },
                }
            )
            write_json(output, result)

    decode_speeds = [
        item["timing"]["decode_tokens_per_second"]
        for item in result["runs"]
        if item["timing"]["decode_tokens_per_second"] is not None
    ]
    result["summary"] = {
        "run_count": len(result["runs"]),
        "median_ttft_seconds": statistics.median(
            item["timing"]["time_to_first_token_seconds"] for item in result["runs"]
        ),
        "median_decode_tokens_per_second": statistics.median(decode_speeds)
        if decode_speeds
        else None,
        "peak_rss_bytes": max(
            int(item["memory"]["process_after"]["peak_rss_bytes"])
            for item in result["runs"]
        ),
    }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

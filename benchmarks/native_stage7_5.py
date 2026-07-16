from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
from typing import Any

from .common import git_snapshot, host_snapshot, model_snapshot, parse_bytes, write_json


KERNELS = ("reference", "native")


def paired_order(repetition: int) -> tuple[str, str]:
    return KERNELS if repetition % 2 == 0 else tuple(reversed(KERNELS))


def extract_record(kernel: str, repetition: int, report: dict[str, Any]) -> dict[str, Any]:
    paths = {}
    for mode in ("resident", "streaming"):
        path = report[mode]
        decode_tokens = max(0, int(path["generated_tokens"]) - 1)
        decode_seconds = float(path["decode_seconds"])
        paths[mode] = {
            "load_seconds": float(path["load_seconds"]),
            "prefill_seconds": float(path["prefill_seconds"]),
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                decode_tokens / decode_seconds if decode_tokens and decode_seconds else None
            ),
            "rss_after_generation": int(path["memory"]["rss_after_generation"]),
            "generation_expert_io": path["generation_expert_io"],
            "provider_storage": path["provider_storage"],
            "generated_ids": path["generated_ids"],
            "logit_fingerprints": path["logit_fingerprints"],
        }
    return {
        "kernel": kernel,
        "repetition": repetition,
        "all_invariants_pass": bool(report["all_invariants_pass"]),
        "runtime_identity": report["runtime_identity"],
        "correctness": report["correctness"],
        "invariants": report["invariants"],
        "resident": paths["resident"],
        "streaming": paths["streaming"],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kernel in KERNELS:
        selected = [record for record in records if record["kernel"] == kernel]
        kernel_summary: dict[str, Any] = {"runs": len(selected)}
        for mode in ("resident", "streaming"):
            speeds = [
                record[mode]["decode_tokens_per_second"]
                for record in selected
                if record[mode]["decode_tokens_per_second"] is not None
            ]
            kernel_summary[mode] = {
                "median_decode_tokens_per_second": statistics.median(speeds),
                "median_prefill_seconds": statistics.median(
                    record[mode]["prefill_seconds"] for record in selected
                ),
                "median_load_seconds": statistics.median(
                    record[mode]["load_seconds"] for record in selected
                ),
                "max_rss_after_generation": max(
                    record[mode]["rss_after_generation"] for record in selected
                ),
            }
        result[kernel] = kernel_summary
    for mode in ("resident", "streaming"):
        result[f"native_{mode}_speedup"] = (
            result["native"][mode]["median_decode_tokens_per_second"]
            / result["reference"][mode]["median_decode_tokens_per_second"]
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 7.5.4 paired INT8 reference/native gate."
    )
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--prompt", default="请简要解释稀疏专家模型。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cache-bytes", default="4GiB")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2 or args.threads < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("tokens/threads/runs must be positive and max-new-tokens must be >= 2")
    try:
        cache_bytes = parse_bytes(args.cache_bytes)
    except ValueError as exc:
        parser.error(str(exc))

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    from sparseflow.text_runtime import compare_int8_reference_paths

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    root = Path.cwd().resolve()
    model = Path(args.model).expanduser()
    model = (root / model).resolve() if not model.is_absolute() else model.resolve()
    container = Path(args.int8_container).expanduser()
    container = (root / container).resolve() if not container.is_absolute() else container.resolve()
    output = Path(args.output).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_4_native_paired_benchmark",
        "stage": "7.5.4",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model),
        "int8_container": str(container),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "protocol": {
            "threads": args.threads,
            "max_new_tokens": args.max_new_tokens,
            "cache_bytes": cache_bytes,
            "runs": args.runs,
            "warmup": args.warmup,
            "order": "AB/BA paired W8A16-reference and W8A8-native",
            "cache_state": "workload-warm; OS page cache uncontrolled",
        },
        "warmups": [],
        "records": [],
        "summary": {},
        "acceptance": {},
    }
    write_json(output, result)

    def run(kernel: str) -> dict[str, Any]:
        storage = "int8-reference" if kernel == "reference" else "int8-native"
        return compare_int8_reference_paths(
            model,
            container,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            cache_bytes=cache_bytes,
            cache_policy="lru",
            telemetry_level="summary",
            expert_storage=storage,
        )

    for warmup in range(args.warmup):
        for kernel in paired_order(warmup):
            print(f"warmup={warmup + 1}/{args.warmup} kernel={kernel}", flush=True)
            report = run(kernel)
            result["warmups"].append(
                {
                    "kernel": kernel,
                    "repetition": warmup,
                    "all_invariants_pass": report["all_invariants_pass"],
                }
            )
            write_json(output, result)

    for repetition in range(args.runs):
        for kernel in paired_order(repetition):
            print(f"run={repetition + 1}/{args.runs} kernel={kernel}", flush=True)
            report = run(kernel)
            result["records"].append(extract_record(kernel, repetition, report))
            write_json(output, result)

    result["summary"] = summarize(result["records"])
    identities = {}
    for kernel in KERNELS:
        identities[kernel] = {
            (
                tuple(record[mode]["generated_ids"]),
                tuple(item["sha256"] for item in record[mode]["logit_fingerprints"]),
            )
            for record in result["records"]
            if record["kernel"] == kernel
            for mode in ("resident", "streaming")
        }
    result["acceptance"] = {
        "all_storage_invariants_pass": all(
            item["all_invariants_pass"] for item in result["records"]
        ),
        "reference_exact_across_runs_and_storage": len(identities["reference"]) == 1,
        "native_exact_across_runs_and_storage": len(identities["native"]) == 1,
        "resident_native_speedup_at_least_1_20x": (
            result["summary"]["native_resident_speedup"] >= 1.20
        ),
        "streaming_native_speedup_at_least_1_20x": (
            result["summary"]["native_streaming_speedup"] >= 1.20
        ),
    }
    result["acceptance"]["all_pass"] = all(result["acceptance"].values())
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(json.dumps(result["acceptance"], ensure_ascii=False))
    print(f"results={output}")
    return 0 if result["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

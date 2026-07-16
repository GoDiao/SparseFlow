from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from .common import git_snapshot, host_snapshot, model_snapshot, write_json


LEVELS = ("none", "summary", "layer")


def rotated_levels(repetition: int) -> tuple[str, ...]:
    offset = repetition % len(LEVELS)
    return LEVELS[offset:] + LEVELS[:offset]


def summarize(records: list[dict]) -> dict[str, dict]:
    result = {}
    for level in LEVELS:
        selected = [item for item in records if item["telemetry_level"] == level]
        result[level] = {
            "runs": len(selected),
            "median_prefill_seconds": statistics.median(
                item["prefill_seconds"] for item in selected
            ),
            "median_decode_tokens_per_second": statistics.median(
                item["decode_tokens_per_second"] for item in selected
            ),
            "median_wall_seconds": statistics.median(item["wall_seconds"] for item in selected),
            "median_observer_seconds": statistics.median(
                item["observer_seconds"] for item in selected
            ),
        }
        if level == "layer":
            closures = [item["critical_path_closure_ratio"] for item in selected]
            result[level]["median_critical_path_closure_ratio"] = statistics.median(
                closures
            )
    baseline = result["none"]
    for level in ("summary", "layer"):
        item = result[level]
        item["decode_throughput_delta_ratio_vs_none"] = (
            item["median_decode_tokens_per_second"]
            / baseline["median_decode_tokens_per_second"]
            - 1.0
        )
        item["wall_delta_ratio_vs_none"] = (
            item["median_wall_seconds"] / baseline["median_wall_seconds"] - 1.0
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Stage 7.5 telemetry observer effect.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--prompt", default="请简要解释稀疏专家模型为什么适合分层存储。")
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.runs < 1 or args.max_new_tokens < 2:
        parser.error("threads/runs must be positive and max-new-tokens must be >= 2")

    import torch

    from sparseflow.text_runtime import Qwen36TextRuntime

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    root = Path.cwd().resolve()
    model_dir = Path(args.model).expanduser()
    if not model_dir.is_absolute():
        model_dir = root / model_dir
    model_dir = model_dir.resolve()
    output = Path(args.output).expanduser().resolve()
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_observer_effect",
        "stage": "7.5.0",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "protocol": {
            "variant": "C3-R",
            "load_mode": "memory-native",
            "dtype": "bf16",
            "threads": args.threads,
            "runs_per_level": args.runs,
            "max_new_tokens": args.max_new_tokens,
            "levels": list(LEVELS),
            "order": "rotated Latin order within one loaded runtime",
        },
        "records": [],
        "summary": {},
        "acceptance": {},
    }
    write_json(output, result)

    runtime = Qwen36TextRuntime.from_pretrained(
        model_dir,
        mode="resident",
        dtype="bf16",
        load_mode="memory-native",
        # Install detailed-only router hooks once; each measured run still
        # switches the telemetry collector to its requested level.
        telemetry_level="layer",
        experts_implementation="eager",
    )
    try:
        runtime.greedy_generate(args.prompt, max_new_tokens=args.max_new_tokens)
        for repetition in range(args.runs):
            for level in rotated_levels(repetition):
                runtime.telemetry.level = level
                runtime.telemetry.reset()
                started = time.perf_counter()
                generated = runtime.greedy_generate(
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    record_logit_fingerprints=True,
                )
                wall_seconds = time.perf_counter() - started
                decode_tokens = generated["generated_tokens"] - 1
                telemetry = generated["telemetry"]
                record = {
                    "repetition": repetition,
                    "telemetry_level": level,
                    "wall_seconds": wall_seconds,
                    "prefill_seconds": generated["prefill_seconds"],
                    "decode_seconds": generated["decode_seconds"],
                    "decode_tokens_per_second": (
                        decode_tokens / generated["decode_seconds"]
                        if decode_tokens and generated["decode_seconds"] > 0
                        else None
                    ),
                    "generated_ids": generated["generated_ids"],
                    "logit_fingerprints": generated["logit_fingerprints"],
                    "observer_seconds": telemetry["observer_seconds"],
                    "telemetry_forwards": len(telemetry["forwards"]),
                    "telemetry_records": len(telemetry["records"]),
                }
                if level == "layer":
                    timings = telemetry["summary"]["timings_ms"]
                    layer_total = telemetry["summary"]["layer_total_ms"]
                    categorized = sum(
                        timings.get(key, 0.0)
                        for key in (
                            "dispatch",
                            "prepare",
                            "provider_get",
                            "expert_kernel",
                            "routing_accumulation",
                        )
                    )
                    record["timing_breakdown"] = {
                        "runtime_ms": timings,
                        "provider_ms": {
                            key: value
                            for key, value in telemetry["summary"]["provider"].items()
                            if key.startswith("timing_")
                        },
                        "layer_total_ms": layer_total,
                        "categorized_critical_path_ms": categorized,
                    }
                    record["critical_path_closure_ratio"] = (
                        categorized / layer_total if layer_total else 1.0
                    )
                result["records"].append(record)
                write_json(output, result)
    finally:
        runtime.close()

    result["summary"] = summarize(result["records"])
    identities = {
        (
            tuple(item["generated_ids"]),
            tuple(value["sha256"] for value in item["logit_fingerprints"]),
        )
        for item in result["records"]
    }
    summary_delta = abs(
        result["summary"]["summary"]["decode_throughput_delta_ratio_vs_none"]
    )
    result["acceptance"] = {
        "all_logits_and_ids_exact": len(identities) == 1,
        "summary_decode_delta_within_3_percent": summary_delta <= 0.03,
        "summary_records_are_aggregated": all(
            item["telemetry_records"] == 0
            for item in result["records"]
            if item["telemetry_level"] == "summary"
        ),
        "critical_path_closes_within_5_percent": (
            0.95
            <= result["summary"]["layer"]["median_critical_path_closure_ratio"]
            <= 1.05
        ),
        "timing_categories_present": all(
            {
                "timing_cache_lookup_ms_total",
                "timing_victim_selection_ms_total",
                "timing_allocation_reuse_ms_total",
                "timing_policy_maintenance_ms_total",
                "timing_tensor_decode_view_ms_total",
                "timing_pread_ms_total",
            }
            <= set(item["timing_breakdown"]["provider_ms"])
            and {
                "router",
                "dispatch",
                "prepare",
                "provider_get",
                "expert_kernel",
                "routing_accumulation",
            }
            <= set(item["timing_breakdown"]["runtime_ms"])
            for item in result["records"]
            if item["telemetry_level"] == "layer"
        ),
    }
    result["acceptance"]["all_pass"] = all(result["acceptance"].values())
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(json.dumps(result["acceptance"], ensure_ascii=False))
    print(f"results={output}")
    return 0 if result["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

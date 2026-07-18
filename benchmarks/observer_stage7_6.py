from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import git_snapshot, host_snapshot, model_snapshot, write_json
from .run_sparseflow import configure_threads


def _paired_order(repetition: int) -> tuple[str, str]:
    return ("none", "summary") if repetition % 2 == 0 else ("summary", "none")


def _route_digest(records: list[dict]) -> tuple[tuple[int, str], ...]:
    return tuple((int(item["layer"]), item["sha256"]) for item in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7.6 telemetry observer-effect gate")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument(
        "--prompt", default="请简要解释稀疏专家模型为什么适合分层存储。"
    )
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--profile-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if min(args.threads, args.pairs, args.profile_runs) < 1 or args.max_new_tokens < 2:
        parser.error("threads/runs must be positive and max-new-tokens must be >= 2")

    configure_threads(args.threads)
    import torch

    from sparseflow.native_int8 import set_native_profile
    from sparseflow.text_runtime import Qwen36TextRuntime

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    root = Path.cwd().resolve()
    model_dir = Path(args.model).expanduser().resolve()
    int8_container = Path(args.int8_container).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_6_observer_effect",
        "stage": "7.6.7",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "protocol": {
            "variant": "W8A8-native-hybrid-resident",
            "threads": args.threads,
            "pairs": args.pairs,
            "profile_runs": args.profile_runs,
            "max_new_tokens": args.max_new_tokens,
            "order": "AB/BA none-summary, then profile diagnostics",
        },
        "records": [],
    }
    write_json(output, result)
    runtime = Qwen36TextRuntime.from_pretrained(
        model_dir,
        mode="resident",
        dtype="bf16",
        load_mode="memory-native",
        telemetry_level="profile",
        experts_implementation="eager",
        expert_storage="int8-native",
        int8_container=int8_container,
        native_dispatch="hybrid",
    )
    try:
        set_native_profile(False)
        runtime.greedy_generate(args.prompt, max_new_tokens=args.max_new_tokens)
        schedule = [
            (repetition, level)
            for repetition in range(args.pairs)
            for level in _paired_order(repetition)
        ] + [(repetition, "profile") for repetition in range(args.profile_runs)]
        for repetition, level in schedule:
            runtime.telemetry.level = level
            runtime.telemetry.reset()
            set_native_profile(level == "profile")
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
                "decode_tokens_per_second": decode_tokens / generated["decode_seconds"],
                "generated_ids": generated["generated_ids"],
                "logit_sha256": [item["sha256"] for item in generated["logit_fingerprints"]],
                "route_digest": _route_digest(generated["route_audit"]),
                "observer_seconds": telemetry["observer_seconds"],
                "telemetry_summary": telemetry["summary"],
                "telemetry_records": len(telemetry["records"]),
            }
            result["records"].append(record)
            write_json(output, result)
    finally:
        set_native_profile(False)
        runtime.close()

    levels = {}
    for level in ("none", "summary", "profile"):
        selected = [item for item in result["records"] if item["telemetry_level"] == level]
        levels[level] = {
            "runs": len(selected),
            "median_ttft_seconds": statistics.median(
                item["prefill_seconds"] for item in selected
            ),
            "median_decode_tokens_per_second": statistics.median(
                item["decode_tokens_per_second"] for item in selected
            ),
            "median_wall_seconds": statistics.median(
                item["wall_seconds"] for item in selected
            ),
            "median_observer_seconds": statistics.median(
                item["observer_seconds"] for item in selected
            ),
        }
    paired_deltas = []
    for repetition in range(args.pairs):
        pair = {
            item["telemetry_level"]: item
            for item in result["records"]
            if item["repetition"] == repetition
            and item["telemetry_level"] in {"none", "summary"}
        }
        paired_deltas.append(
            pair["summary"]["decode_tokens_per_second"]
            / pair["none"]["decode_tokens_per_second"]
            - 1.0
        )
    profile_records = [
        item for item in result["records"] if item["telemetry_level"] == "profile"
    ]
    closure = []
    for item in profile_records:
        timing = item["telemetry_summary"]["timings_ms"]
        routed = float(timing["routed_experts"])
        categorized = sum(
            float(timing[key])
            for key in (
                "dispatch",
                "prepare",
                "provider_get",
                "expert_kernel",
                "routing_accumulation",
            )
        )
        closure.append(categorized / routed if routed else 1.0)
    identities = {
        (
            tuple(item["generated_ids"]),
            tuple(item["logit_sha256"]),
            tuple(item["route_digest"]),
        )
        for item in result["records"]
    }
    median_delta = statistics.median(paired_deltas)
    median_slowdown = max(0.0, -median_delta)
    acceptance = {
        "all_logits_routes_and_ids_exact": len(identities) == 1,
        "summary_decode_slowdown_within_1_percent": median_slowdown <= 0.01,
        "summary_has_no_layer_records": all(
            item["telemetry_records"] == 0
            for item in result["records"]
            if item["telemetry_level"] == "summary"
        ),
        "profile_critical_path_closes_within_5_percent": all(
            0.95 <= value <= 1.05 for value in closure
        ),
    }
    result["summary"] = {
        "levels": levels,
        "paired_summary_decode_deltas": paired_deltas,
        "median_summary_decode_delta": median_delta,
        "median_summary_decode_slowdown": median_slowdown,
        "absolute_delta_resolved_within_1_percent": abs(median_delta) <= 0.01,
        "profile_critical_path_closure": closure,
    }
    result["acceptance"] = {**acceptance, "all_pass": all(acceptance.values())}
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(json.dumps(result["acceptance"], ensure_ascii=False))
    print(f"results={output}")
    return 0 if result["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

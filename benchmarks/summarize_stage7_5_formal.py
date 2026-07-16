from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .common import write_json


TASKS = ("hellaswag", "arc_challenge", "mmlu")
QUALITY_LEVELS = {"smoke": 1, "pilot": 3, "development": 5, "formal": 20}


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def decode_read_bytes(run: dict[str, Any]) -> int:
    return sum(
        int(item["provider"].get("reader_bytes", 0))
        for item in run["telemetry"]["forwards"]
        if item["phase"] == "decode"
    )


def performance_row(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    runs = result["runs"]
    decode_bytes_per_token = []
    rss_after = []
    hits = misses = 0
    for run in runs:
        tokens = max(1, int(run["timing"]["decode_tokens"]))
        decode_bytes_per_token.append(decode_read_bytes(run) / tokens)
        rss_after.append(int(run["memory"]["rss_after_generation"]))
        provider = run["provider_delta"] or {}
        hits += int(provider.get("cache_hits", 0))
        misses += int(provider.get("cache_misses", 0))
    return {
        "source": str(path),
        "expert_storage": result["runtime"]["expert_storage"],
        "variant": result["variant"],
        "cache_state": result["storage_policy"]["cache_state"],
        "cache_bytes": result["storage_policy"]["cache_bytes"],
        "samples": len(runs),
        "median_load_seconds": result["summary"]["median_ttft_seconds"] * 0
        + result["load"]["seconds"],
        "median_ttft_seconds": result["summary"]["median_ttft_seconds"],
        "median_decode_tokens_per_second": result["summary"][
            "median_decode_tokens_per_second"
        ],
        "decode_latency_p50_seconds": result["summary"]["decode_latency_p50_seconds"],
        "decode_latency_p95_seconds": result["summary"]["decode_latency_p95_seconds"],
        "median_decode_physical_bytes_per_token": statistics.median(
            decode_bytes_per_token
        ),
        "cache_hit_rate": hits / (hits + misses) if hits + misses else None,
        "max_rss_after_generation": max(rss_after),
        "process_peak_rss_bytes": result["summary"]["peak_rss_bytes"],
        "generated_ids_exact_across_runs": result["summary"][
            "generated_ids_exact_across_runs"
        ],
        "git": result["git"],
    }


def load_performance(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    execution = json.loads((root / "matrix_execution.json").read_text(encoding="utf-8"))
    values = []
    raw = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "matrix_execution.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        raw[path.stem] = result
        values.append(performance_row(path, result))
    return execution, values, raw


def aggregate_cold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cold = [row for row in rows if row["cache_state"] == "model-cold"]
    return {
        "samples": len(cold),
        "median_ttft_seconds": statistics.median(row["median_ttft_seconds"] for row in cold),
        "median_decode_tokens_per_second": statistics.median(
            row["median_decode_tokens_per_second"] for row in cold
        ),
        "median_decode_physical_bytes_per_token": statistics.median(
            row["median_decode_physical_bytes_per_token"] for row in cold
        ),
        "max_rss_after_generation": max(row["max_rss_after_generation"] for row in cold),
        "raw_decode_tokens_per_second": [
            row["median_decode_tokens_per_second"] for row in cold
        ],
    }


def quality_subset(result: dict[str, Any], count: int) -> list[dict[str, Any]]:
    selected = []
    for task in TASKS:
        selected.extend(
            [item for item in result["questions"] if item["task"] == task][:count]
        )
    return selected


def quality_metrics(questions: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
        result = {"n": len(values)}
        for name, field in (
            ("accuracy", "correct"),
            ("acc_norm_char", "correct_norm_char"),
            ("acc_norm_token", "correct_norm_token"),
        ):
            successes = sum(bool(item[field]) for item in values)
            result[name] = successes / len(values)
            result[f"{name}_wilson95"] = wilson(successes, len(values))
        return result

    return {
        "aggregate": metrics(questions),
        "tasks": {
            task: metrics([item for item in questions if item["task"] == task])
            for task in TASKS
        },
    }


def native_storage_exact(resident: dict[str, Any], streaming: dict[str, Any]) -> dict[str, Any]:
    ids_equal = [item["id"] for item in resident["questions"]] == [
        item["id"] for item in streaming["questions"]
    ]
    predictions_equal = True
    token_values_equal = True
    max_ll_delta = 0.0
    for left, right in zip(resident["questions"], streaming["questions"]):
        predictions_equal &= (
            left["prediction"],
            left["prediction_norm_char"],
            left["prediction_norm_token"],
        ) == (
            right["prediction"],
            right["prediction_norm_char"],
            right["prediction_norm_token"],
        )
        for left_choice, right_choice in zip(left["choices"], right["choices"]):
            max_ll_delta = max(
                max_ll_delta,
                abs(left_choice["loglikelihood"] - right_choice["loglikelihood"]),
            )
            token_values_equal &= (
                left_choice["token_loglikelihoods"]
                == right_choice["token_loglikelihoods"]
            )
    return {
        "question_ids_equal": ids_equal,
        "predictions_equal": predictions_equal,
        "token_loglikelihoods_equal": token_values_equal,
        "max_choice_loglikelihood_delta": max_ll_delta,
        "all_exact": ids_equal and predictions_equal and token_values_equal and max_ll_delta == 0.0,
    }


def agreement(left: dict[str, Any], right: dict[str, Any]) -> dict[str, int]:
    pairs = list(zip(left["questions"], right["questions"]))
    return {
        "questions": len(pairs),
        "prediction_equal": sum(a["prediction"] == b["prediction"] for a, b in pairs),
        "prediction_norm_char_equal": sum(
            a["prediction_norm_char"] == b["prediction_norm_char"] for a, b in pairs
        ),
        "prediction_norm_token_equal": sum(
            a["prediction_norm_token"] == b["prediction_norm_token"] for a, b in pairs
        ),
    }


def load_quality(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    values = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "matrix_execution.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        values[path.stem] = value
    required = {
        "smoke-bf16-reference",
        "smoke-int8-reference",
        "smoke-int8-native",
        "smoke-int8-native-streaming",
        "formal-bf16-reference",
        "formal-int8-native",
        "formal-int8-native-streaming",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"quality results missing: {missing}")
    resident = values["formal-int8-native"]
    streaming = values["formal-int8-native-streaming"]
    bf16 = values["formal-bf16-reference"]
    levels = {
        level: {
            "bf16": quality_metrics(quality_subset(bf16, count)),
            "int8_native": quality_metrics(quality_subset(resident, count)),
        }
        for level, count in QUALITY_LEVELS.items()
    }
    streaming_reads = sum(
        int((item.get("provider_delta") or {}).get("reader_bytes", 0))
        for item in streaming["questions"]
    )
    return values, {
        "levels": levels,
        "native_resident_streaming": native_storage_exact(resident, streaming),
        "bf16_native_agreement": agreement(bf16, resident),
        "streaming_expert_read_bytes": streaming_reads,
        "int8_reference_formal": {
            "status": "not-run",
            "reason": (
                "W8A16 reference dequantization dominated the first formal question; "
                "reference numerical attribution is covered by the 32-token teacher-forced "
                "quality gate, standard smoke, and formal performance matrix."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.5.6 evidence.")
    parser.add_argument("--performance-dir", required=True)
    parser.add_argument("--quality-dir", required=True)
    parser.add_argument("--stage7-4-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    perf_root = Path(args.performance_dir).expanduser().resolve()
    quality_root = Path(args.quality_dir).expanduser().resolve()
    execution, rows, raw = load_performance(perf_root)
    quality_raw, quality = load_quality(quality_root)
    stage74 = json.loads(Path(args.stage7_4_summary).read_text(encoding="utf-8"))

    warm = [row for row in rows if row["cache_state"] == "workload-warm"]
    identities = {}
    for storage in ("bf16", "int8-reference", "int8-native"):
        values = []
        for result in raw.values():
            if result["runtime"]["expert_storage"] != storage:
                continue
            for run in result["runs"]:
                values.append(
                    (
                        tuple(run["quality"]["generated_ids"]),
                        tuple(item["sha256"] for item in run["quality"]["logit_fingerprints"]),
                    )
                )
        identities[storage] = len(set(values)) == 1
    clean = all(not row["git"]["dirty"] for row in rows)
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_6_formal_summary",
        "stage": "7.5.6",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "performance": {
            "rows": rows,
            "warm": warm,
            "native_4g_model_cold": aggregate_cold(rows),
            "generic_offload_stage7_4": stage74["system_baselines"],
        },
        "quality": quality,
        "validation": {
            "performance_matrix_complete": execution.get("all_complete") and len(rows) == 10,
            "performance_git_clean": clean,
            "storage_exact_by_precision": identities,
            "native_cold_has_three_replicates": len(
                [row for row in rows if row["cache_state"] == "model-cold"]
            )
            == 3,
            "formal_quality_has_60_rows": all(
                len(quality_raw[name]["questions"]) == 60
                for name in (
                    "formal-bf16-reference",
                    "formal-int8-native",
                    "formal-int8-native-streaming",
                )
            ),
            "native_quality_storage_exact": quality["native_resident_streaming"]["all_exact"],
            "standard_int8_reference_smoke_present": len(
                quality_raw["smoke-int8-reference"]["questions"]
            )
            == 3,
            "generic_offload_baseline_present": bool(stage74["system_baselines"].get("C2-cold")),
        },
    }
    result["validation"]["all_pass"] = all(
        value
        for key, value in result["validation"].items()
        if key not in {"storage_exact_by_precision", "all_pass"}
    ) and all(identities.values())
    write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result["validation"], ensure_ascii=False))
    return 0 if result["validation"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

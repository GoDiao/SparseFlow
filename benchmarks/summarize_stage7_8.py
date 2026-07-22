"""Summarize Stage 7.8 operator, resident, and streaming gates.

[Main Dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _by_batch(value: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["batch_size"]): item for item in value.get("batches", [])}


def summarize(
    grouped: dict[str, Any],
    cohort_results: list[dict[str, Any]],
    streaming: dict[str, Any],
    previous_stage7_7: dict[str, Any],
) -> dict[str, Any]:
    grouped_batches = _by_batch(grouped)
    operator_exact = all(
        bool(item["grouped_vs_fused"]["exact"])
        and bool(item["grouped_vs_fused"]["argmax_equal"])
        and bool(item["grouped"]["repeat_exact"])
        for item in grouped_batches.values()
    )
    b1 = grouped_batches.get(1, {})
    b4 = grouped_batches.get(4, {})
    b8 = grouped_batches.get(8, {})
    operator = {
        "exact_against_fused_all_batches": operator_exact,
        "b1_grouped_over_fused": float(b1.get("grouped_over_fused", 0.0)),
        "b1_no_regression": float(b1.get("grouped_over_fused", 0.0)) >= 0.97,
        "b4_grouped_speedup": float(b4.get("grouped_speedup", 0.0)),
        "b4_target_1_95": float(b4.get("grouped_speedup", 0.0)) >= 1.95,
        "b8_grouped_speedup": float(b8.get("grouped_speedup", 0.0)),
        "b8_nonzero": float(b8.get("grouped_speedup", 0.0)) > 0.0,
        "batches": [
            {
                "batch_size": size,
                "fused_speedup": float(item["fused_speedup"]),
                "grouped_speedup": float(item["grouped_speedup"]),
                "grouped_over_fused": float(item["grouped_over_fused"]),
                "exact": bool(item["grouped_vs_fused"]["exact"]),
                "route_groups": item["route_groups"],
            }
            for size, item in sorted(grouped_batches.items())
        ],
    }

    resident_cells = []
    resident_kernel_exact = True
    resident_behavior_exact = True
    resident_speed_go = True
    for item in cohort_results:
        grouped_vs_fused = item["grouped_vs_fused_batch"]
        behavior = item["correctness"]
        resident_kernel_exact = resident_kernel_exact and bool(grouped_vs_fused["all_equal"])
        resident_behavior_exact = resident_behavior_exact and bool(
            behavior["generated_ids_equal"]
            and behavior["texts_equal"]
            and behavior.get("logits", {}).get("argmax_equal", False)
        )
        resident_speed_go = resident_speed_go and float(item["aggregate_speedup"]) >= 1.25
        grouped_decode = float(item["grouped"]["decode_seconds"])
        fused_decode = float(item["fused_batch"]["decode_seconds"])
        resident_cells.append({
            "batch_size": int(item["batch_size"]),
            "independent_to_grouped_speedup": float(item["aggregate_speedup"]),
            "grouped_over_fused_batch_decode_speedup": float(
                item.get(
                    "grouped_over_fused_batch_decode_speedup",
                    fused_decode / grouped_decode if grouped_decode else 0.0,
                )
            ),
            "grouped_vs_fused_exact": bool(grouped_vs_fused["all_equal"]),
            "independent_ids_text_argmax_exact": bool(
                behavior["generated_ids_equal"]
                and behavior["texts_equal"]
                and behavior.get("logits", {}).get("argmax_equal", False)
            ),
            "independent_logit_fingerprints_exact": bool(
                behavior["logit_fingerprints_equal"]
            ),
            "independent_logit_error": behavior.get("logits"),
        })

    old_round_robin = {}
    for item in previous_stage7_7.get("replay_cells", []):
        if item.get("mode") == "round-robin":
            old_round_robin[(item["schedule_id"], int(item["budget_bytes"]))] = int(
                item["loaded_bytes"]
            )
    streaming_cells = []
    streaming_gate = True
    for item in streaming.get("replay", []):
        if item.get("mode") != "cache-aware":
            continue
        key = (item["schedule_id"], int(item["budget_bytes"]))
        baseline = old_round_robin.get(key)
        if baseline is None:
            raise ValueError(f"missing Stage 7.7 round-robin baseline for {key}")
        loaded = int(item["cache"]["loaded_bytes"])
        ratio = loaded / baseline if baseline else 0.0
        pass_gate = (
            ratio <= 1.0
            and int(item["cache"]["cached_bytes"]) <= int(item["budget_bytes"])
            and int(item.get("cohort_overflow", 0)) == 0
        )
        streaming_gate = streaming_gate and pass_gate
        streaming_cells.append({
            "schedule_id": item["schedule_id"],
            "budget_gib": int(item["budget_bytes"]) / 2**30,
            "working_set_ratio": item.get("working_set_ratio"),
            "cache_aware_loaded_gib": loaded / 2**30,
            "round_robin_loaded_gib": baseline / 2**30,
            "loaded_ratio": ratio,
            "cohorts": int(item.get("cohort_count", 0)),
            "budget_respected": int(item["cache"]["cached_bytes"]) <= int(item["budget_bytes"]),
            "leases_released": int(item.get("cache", {}).get("pinned_entries", 0)) == 0,
            "gate_pass": pass_gate,
        })

    no_go_reasons = []
    if not operator["exact_against_fused_all_batches"]:
        no_go_reasons.append("grouped operator is not exact against legacy fused output")
    if not operator["b1_no_regression"]:
        no_go_reasons.append("B=1 grouped path regressed by more than 3%")
    if not operator["b4_target_1_95"]:
        no_go_reasons.append("B=4 grouped operator missed the 1.95x target")
    if not resident_kernel_exact:
        no_go_reasons.append("resident grouped batch is not exact against fused batch")
    if not resident_behavior_exact:
        no_go_reasons.append("resident cohort changed session IDs, text, or argmax")
    if not resident_speed_go:
        no_go_reasons.append("resident cohort missed the 1.25x independent-session target")
    if not streaming_gate:
        no_go_reasons.append("cache-aware streaming loaded more bytes than round-robin in at least one cell")
    if any(not item["independent_logit_fingerprints_exact"] for item in resident_cells):
        no_go_reasons.append("independent-session full logits are not bit-exact under batched ATen execution")

    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_8_summary",
        "agent": "Main Dev",
        "operator": operator,
        "resident": {
            "cells": resident_cells,
            "kernel_exact": resident_kernel_exact,
            "behavior_exact": resident_behavior_exact,
            "speed_gate": resident_speed_go,
        },
        "streaming": {
            "cells": streaming_cells,
            "gate": streaming_gate,
        },
        "gates": {
            "operator": operator["exact_against_fused_all_batches"]
            and operator["b1_no_regression"]
            and operator["b4_target_1_95"],
            "resident_kernel": resident_kernel_exact,
            "resident_behavior": resident_behavior_exact,
            "resident_speed": resident_speed_go,
            "streaming": streaming_gate,
        },
        "all_pass": not no_go_reasons,
        "decision": (
            "stage7.8-complete-all-gates"
            if not no_go_reasons
            else "native-grouped-operator-go-resident-experimental-streaming-gated"
        ),
        "no_go_reasons": no_go_reasons,
        "next_scope": (
            "profile-gated native dispatcher and cache admission before making grouped the default"
            if no_go_reasons
            else "formal Stage 7.9 scheduler"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.8 results.")
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--cohort", action="append", required=True)
    parser.add_argument("--streaming", required=True)
    parser.add_argument("--previous-stage7-7", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(
        load_json(Path(args.grouped)),
        [load_json(Path(path)) for path in args.cohort],
        load_json(Path(args.streaming)),
        load_json(Path(args.previous_stage7_7)),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "all_pass": result["all_pass"],
        "decision": result["decision"],
        "no_go_reasons": result["no_go_reasons"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

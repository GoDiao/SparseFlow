"""Summarize the Stage 7.7 feasibility gates.

The raw route replay is intentionally the source of truth for storage results.
The older provider replay is not used because it included miss-time tensor
decoding.  A streaming NO-GO is a valid experimental outcome and is reported
separately from verifier/invariant failures.

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


def summarize(raw: dict[str, Any], grouped: dict[str, Any]) -> dict[str, Any]:
    replay = raw.get("replay")
    batches = grouped.get("batches")
    if not isinstance(replay, list) or not isinstance(batches, list):
        raise ValueError("missing replay or grouped batch results")

    cells: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in replay:
        schedule_id = str(item["schedule_id"])
        mode = str(item["mode"])
        budget = int(item["budget_bytes"])
        cache = item["cache"]
        requests = int(cache["requests"])
        hits = int(cache.get("hits", cache["cache_hits"]))
        misses = int(cache.get("misses", cache["cache_misses"]))
        loaded = int(cache["loaded_bytes"])
        reader_bytes = int(item["provider_read_bytes"])
        cell = {
            "schedule_id": schedule_id,
            "mode": mode,
            "budget_bytes": budget,
            "budget_gib": budget / 2**30,
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / requests if requests else 0.0,
            "loaded_bytes": loaded,
            "loaded_gib": loaded / 2**30,
            "provider_read_bytes": reader_bytes,
            "provider_read_gib": reader_bytes / 2**30,
            "provider_read_calls": int(item["provider_read_calls"]),
            "wall_seconds": float(item["wall_seconds"]),
            "budget_respected": bool(item["budget_respected"]),
            "leases_released": bool(item["leases_released"]),
            "demand_accounting_exact": loaded == reader_bytes,
            "raw_replay": True,
        }
        cells.append(cell)
        key = (schedule_id, mode, budget)
        if key in by_key:
            raise ValueError(f"duplicate replay cell: {key}")
        by_key[key] = cell

    comparisons: list[dict[str, Any]] = []
    streaming_invariants: list[str] = []
    core_schedules = ("sync-b4", "sync-b8")
    core_budgets = (4 * 2**30, 8 * 2**30)
    for schedule_id in core_schedules:
        for budget in core_budgets:
            union = by_key.get((schedule_id, "union", budget))
            round_robin = by_key.get((schedule_id, "round-robin", budget))
            if union is None or round_robin is None:
                raise ValueError(
                    f"missing core replay pair: {schedule_id}, {budget}"
                )
            ratio = (
                union["loaded_bytes"] / round_robin["loaded_bytes"]
                if round_robin["loaded_bytes"]
                else 0.0
            )
            pass_gate = union["loaded_bytes"] <= round_robin["loaded_bytes"]
            comparison = {
                "schedule_id": schedule_id,
                "budget_gib": budget / 2**30,
                "union_loaded_gib": union["loaded_gib"],
                "round_robin_loaded_gib": round_robin["loaded_gib"],
                "union_hit_rate": union["hit_rate"],
                "round_robin_hit_rate": round_robin["hit_rate"],
                "union_vs_round_robin_loaded_ratio": ratio,
                "normalized_loaded_bytes_not_higher": pass_gate,
                "budget_and_lease_invariants": all(
                    item["budget_respected"] and item["leases_released"]
                    and item["demand_accounting_exact"]
                    for item in (union, round_robin)
                ),
            }
            comparisons.append(comparison)
            if not comparison["budget_and_lease_invariants"]:
                streaming_invariants.append(
                    f"storage invariant failed for {schedule_id}/{budget // 2**30}GiB"
                )

    exact_batches = all(
        bool(item["comparison"]["exact"])
        and bool(item["comparison"]["argmax_equal"])
        and bool(item["canonical"]["repeat_exact"])
        and bool(item["fused"]["repeat_exact"])
        for item in batches
    )
    b4 = next((item for item in batches if int(item["batch_size"]) == 4), None)
    grouped_speedup = float(b4["aggregate_speedup"]) if b4 is not None else 0.0
    grouped_gate = exact_batches and grouped_speedup >= 1.5

    streaming_gate = not streaming_invariants and all(
        item["normalized_loaded_bytes_not_higher"] for item in comparisons
    )
    no_go_reasons: list[str] = []
    if not streaming_gate:
        failed = [
            f"{item['schedule_id']}/{item['budget_gib']:.0f}GiB"
            for item in comparisons
            if not item["normalized_loaded_bytes_not_higher"]
        ]
        if failed:
            no_go_reasons.append(
                "shared-cache union loaded more bytes than round-robin at "
                + ", ".join(failed)
            )
        no_go_reasons.extend(streaming_invariants)

    decision = (
        "resident-and-streaming-scheduler-go"
        if grouped_gate and streaming_gate
        else "resident-only-scheduler"
        if grouped_gate
        else "stop-scheduler-work"
    )
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_7_summary",
        "agent": "Main Dev",
        "trace": raw.get("trace", {}),
        "grouped_kernel": {
            "batches": [
                {
                    "batch_size": int(item["batch_size"]),
                    "aggregate_speedup": float(item["aggregate_speedup"]),
                    "exact": bool(item["comparison"]["exact"]),
                    "max_abs": float(item["comparison"]["max_abs"]),
                    "argmax_equal": bool(item["comparison"]["argmax_equal"]),
                }
                for item in batches
            ],
            "b4_speedup": grouped_speedup,
            "exact_all_batches": exact_batches,
            "gate_pass": grouped_gate,
            "threshold": 1.5,
        },
        "replay_cells": cells,
        "streaming_comparisons": comparisons,
        "gates": {
            "grouped_kernel": grouped_gate,
            "streaming_union": streaming_gate,
            "all_storage_invariants": not streaming_invariants,
        },
        "decision": decision,
        "all_pass": grouped_gate and streaming_gate,
        "no_go_reasons": no_go_reasons,
        "next_scope": (
            "Implement resident-only fixed-cohort scheduler; keep streaming "
            "scheduler gated until cache-aware union does not increase reads."
            if decision == "resident-only-scheduler"
            else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.7 gates.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = summarize(
        load_json(Path(args.replay).expanduser().resolve()),
        load_json(Path(args.grouped).expanduser().resolve()),
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

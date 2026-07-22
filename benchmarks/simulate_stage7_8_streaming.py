"""Replay the Stage 7.8 cache-aware sub-cohort streaming policy.

The replay compares the new bounded-working-set partition with the existing
round-robin baseline on the same real Stage 7.7 trace.  It measures raw INT8
payload reads only; no tensor decoding or expert kernel is run.

[Main Dev]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sparseflow.benchmark import parse_byte_budgets
from sparseflow.multirequest_trace import load_multi_request_trace

from analyze_stage7_7_union import replay_schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay Stage 7.8 cache-aware streaming.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="4GiB,8GiB")
    parser.add_argument("--schedules", default="sync-b4,sync-b8")
    parser.add_argument("--modes", default="cache-aware")
    parser.add_argument("--working-set-ratio", type=float, default=0.25)
    args = parser.parse_args(argv)
    trace = load_multi_request_trace(args.trace)
    container = Path(args.int8_container).expanduser().resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_8_streaming_subcohort",
        "agent": "Main Dev",
        "trace": trace.as_dict(),
        "replay": [],
    }
    for schedule_id in (item.strip() for item in args.schedules.split(",")):
        if not schedule_id:
            continue
        trace.schedule(schedule_id)
        for budget in parse_byte_budgets(args.budgets):
            for mode in (item.strip() for item in args.modes.split(",")):
                if not mode:
                    continue
                result["replay"].append(
                    replay_schedule(
                        trace,
                        schedule_id,
                        container,
                        budget,
                        mode,
                        args.working_set_ratio,
                    )
                )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "cells": [
            {
                "schedule": item["schedule_id"],
                "budget_gib": item["budget_bytes"] / 2**30,
                "mode": item["mode"],
                "loaded_gib": item["cache"]["loaded_bytes"] / 2**30,
                "cohorts": item["cohort_count"],
            }
            for item in result["replay"]
        ],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

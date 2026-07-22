"""Verify Stage 7.7 raw results and emit a structured decision.

This verifier treats a measured feasibility NO-GO as a successful validation
of the experiment.  It exits non-zero only when the result files violate
their storage, exactness, or declared gate contracts.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from summarize_stage7_7 import load_json, summarize


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def verify(
    replay_path: Path,
    grouped_path: Path,
    summary_path: Path,
    expected_trace_sha256: str | None = None,
) -> dict[str, Any]:
    replay = load_json(replay_path)
    grouped = load_json(grouped_path)
    summary = load_json(summary_path)
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    checks["summary_kind"] = summary.get("kind") == "sparseflow_stage7_7_summary"
    checks["summary_agent"] = summary.get("agent") == "Main Dev"
    trace = summary.get("trace", {})
    trace_sha = str(trace.get("sha256", ""))
    checks["trace_sha256_present"] = bool(trace_sha)
    if expected_trace_sha256 is not None:
        checks["trace_sha256_expected"] = trace_sha == expected_trace_sha256
    checks["replay_source_raw"] = all(
        str(item.get("cache", {}).get("backend_id", "")).endswith(
            "raw-cache-replay"
        )
        for item in replay.get("replay", [])
    )

    replay_cells = replay.get("replay", [])
    checks["replay_cells_complete"] = len(replay_cells) == 8
    checks["replay_budget_and_leases"] = all(
        bool(item.get("budget_respected"))
        and bool(item.get("leases_released"))
        and int(item["provider_read_bytes"]) == int(item["cache"]["loaded_bytes"])
        for item in replay_cells
    )
    grouped_batches = grouped.get("batches", [])
    checks["grouped_batches_complete"] = {int(item["batch_size"]) for item in grouped_batches} == {1, 2, 4, 8}
    checks["grouped_exact"] = all(
        bool(item["comparison"]["exact"])
        and bool(item["comparison"]["argmax_equal"])
        and bool(item["canonical"]["repeat_exact"])
        and bool(item["fused"]["repeat_exact"])
        for item in grouped_batches
    )

    recomputed = summarize(replay, grouped)
    checks["summary_matches_recomputed"] = (
        recomputed["all_pass"] == summary.get("all_pass")
        and recomputed["decision"] == summary.get("decision")
        and recomputed["no_go_reasons"] == summary.get("no_go_reasons")
    )
    checks["decision_is_explicit"] = summary.get("decision") in {
        "resident-and-streaming-scheduler-go",
        "resident-only-scheduler",
        "stop-scheduler-work",
    }
    checks["no_go_is_structured"] = bool(summary.get("all_pass")) or bool(
        summary.get("no_go_reasons")
    )
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)

    try:
        commit = git_value("rev-parse", "HEAD")
        clean = git_value("status", "--porcelain") == ""
    except (OSError, subprocess.CalledProcessError):
        commit = None
        clean = False
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_7_verification",
        "agent": "Main Dev",
        "inputs": {
            "replay": str(replay_path),
            "replay_sha256": sha256(replay_path),
            "grouped": str(grouped_path),
            "grouped_sha256": sha256(grouped_path),
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
        },
        "git": {"commit": commit, "clean": clean},
        "checks": checks,
        "verification_passed": not reasons,
        "all_pass": bool(summary.get("all_pass")),
        "decision": summary.get("decision"),
        "no_go_reasons": summary.get("no_go_reasons", []),
        "verification_failures": reasons,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 7.7 results.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-trace-sha256")
    args = parser.parse_args(argv)
    result = verify(
        Path(args.replay).expanduser().resolve(),
        Path(args.grouped).expanduser().resolve(),
        Path(args.summary).expanduser().resolve(),
        args.expected_trace_sha256,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "verification_passed": result["verification_passed"],
        "all_pass": result["all_pass"],
        "decision": result["decision"],
        "verification_failures": result["verification_failures"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if result["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

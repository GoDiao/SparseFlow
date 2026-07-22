"""Verify Stage 7.8 result structure and invariants.

An experimental NO-GO is valid when ``verification_passed`` is true and the
summary contains explicit no-go reasons.  This verifier does not turn a
failed performance gate into a false success.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def verify(summary: dict[str, Any], grouped: dict[str, Any], cohorts: list[dict[str, Any]], streaming: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {
        "summary_kind": summary.get("kind") == "sparseflow_stage7_8_summary",
        "summary_agent": summary.get("agent") == "Main Dev",
        "grouped_kind": grouped.get("kind") == "sparseflow_stage7_8_grouped_kernel",
        "grouped_agent": grouped.get("agent") == "Main Dev",
        "streaming_kind": streaming.get("kind") == "sparseflow_stage7_8_streaming_subcohort",
        "streaming_agent": streaming.get("agent") == "Main Dev",
        "summary_has_decision": isinstance(summary.get("decision"), str),
        "summary_has_reason_or_pass": bool(summary.get("all_pass")) or bool(summary.get("no_go_reasons")),
    }
    batches = grouped.get("batches", [])
    checks["grouped_batches_present"] = bool(batches)
    checks["grouped_repeat_exact"] = all(
        bool(item.get("grouped", {}).get("repeat_exact")) for item in batches
    )
    checks["grouped_vs_fused_exact"] = all(
        bool(item.get("grouped_vs_fused", {}).get("exact"))
        and bool(item.get("grouped_vs_fused", {}).get("argmax_equal"))
        for item in batches
    )
    checks["cohort_results_present"] = len(cohorts) >= 2
    checks["cohort_kernel_exact"] = all(
        bool(item.get("grouped_vs_fused_batch", {}).get("all_equal"))
        for item in cohorts
    )
    checks["cohort_ids_exact"] = all(
        bool(item.get("correctness", {}).get("generated_ids_equal"))
        for item in cohorts
    )
    checks["streaming_cells_present"] = bool(streaming.get("replay"))
    checks["streaming_accounting"] = all(
        int(item.get("cache", {}).get("cached_bytes", 0)) <= int(item.get("budget_bytes", 0))
        and int(item.get("cache", {}).get("pinned_entries", 0)) == 0
        for item in streaming.get("replay", [])
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
        "kind": "sparseflow_stage7_8_verification",
        "agent": "Main Dev",
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
    parser = argparse.ArgumentParser(description="Verify Stage 7.8 results.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--cohort", action="append", required=True)
    parser.add_argument("--streaming", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary_path = Path(args.summary).expanduser().resolve()
    grouped_path = Path(args.grouped).expanduser().resolve()
    cohort_paths = [Path(path).expanduser().resolve() for path in args.cohort]
    streaming_path = Path(args.streaming).expanduser().resolve()
    result = verify(
        load(summary_path),
        load(grouped_path),
        [load(path) for path in cohort_paths],
        load(streaming_path),
    )
    result["inputs"] = {
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "grouped": str(grouped_path),
        "grouped_sha256": sha256(grouped_path),
        "cohorts": [str(path) for path in cohort_paths],
        "streaming": str(streaming_path),
        "streaming_sha256": sha256(streaming_path),
    }
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

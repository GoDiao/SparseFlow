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


def verify(
    summary: dict[str, Any],
    grouped: dict[str, Any],
    cohorts: list[dict[str, Any]],
    streaming: dict[str, Any],
    formal: dict[str, Any],
    expected_commit: str,
    expected_formal_commit: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    checks = {
        "summary_kind": summary.get("kind") == "sparseflow_stage7_8_summary",
        "summary_agent": summary.get("agent") == "Main Dev",
        "grouped_kind": grouped.get("kind") == "sparseflow_stage7_8_grouped_kernel",
        "grouped_agent": grouped.get("agent") == "Main Dev",
        "streaming_kind": streaming.get("kind") == "sparseflow_stage7_8_streaming_subcohort",
        "streaming_agent": streaming.get("agent") == "Main Dev",
        "formal_kind": formal.get("kind") == "sparseflow_stage7_8_formal_resident_abba",
        "formal_agent": formal.get("agent") == "Main Dev",
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
    grouped_by_batch = {
        int(item.get("batch_size", -1)): item for item in batches
    }
    b1_operator = grouped_by_batch.get(1, {})
    b4_operator = grouped_by_batch.get(4, {})
    checks["operator_b1_no_regression"] = float(
        b1_operator.get("grouped_over_fused", 0.0)
    ) >= 0.97
    checks["operator_b4_threshold"] = float(
        b4_operator.get("grouped_speedup", 0.0)
    ) >= 1.95
    checks["operator_summary_matches_thresholds"] = (
        bool(summary.get("operator", {}).get("b1_no_regression"))
        == checks["operator_b1_no_regression"]
        and bool(summary.get("operator", {}).get("b4_target_1_95"))
        == checks["operator_b4_threshold"]
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
    formal_protocol = formal.get("protocol", {})
    checks["formal_protocol"] = (
        formal_protocol.get("batches") == [1, 4, 8]
        and int(formal_protocol.get("max_new_tokens", 0)) >= 32
        and int(formal_protocol.get("repeats", 0)) >= 3
        and formal_protocol.get("schedule") == "ABBA"
        and formal_protocol.get("quality_gate") == "equivalent-32-token-long-generation"
    )
    formal_batches = formal.get("batches", [])
    checks["formal_batches_present"] = len(formal_batches) == 3
    checks["formal_batch_sizes"] = [
        int(item.get("batch_size", -1)) for item in formal_batches
    ] == [1, 4, 8]
    checks["formal_runtime_identity"] = bool(formal.get("gates", {}).get("all_runtime_identity_exact")) and all(
        item.get("grouped", {}).get("runtime_identity", {}).get("mode") == "resident"
        and item.get("grouped", {}).get("runtime_identity", {}).get("load_mode") == "memory-native"
        and item.get("grouped", {}).get("runtime_identity", {}).get("expert_storage") == "int8-native"
        and item.get("grouped", {}).get("runtime_identity", {}).get("native_dispatch") == "grouped"
        and item.get("fused", {}).get("runtime_identity", {}).get("native_dispatch") == "hybrid"
        for item in formal_batches
    )
    checks["formal_repeated_exact"] = bool(formal.get("gates", {}).get("all_repeats_exact")) and all(
        bool(item.get("grouped", {}).get("repeat_exact"))
        and bool(item.get("fused", {}).get("repeat_exact"))
        for item in formal_batches
    )
    checks["formal_full_logits_exact"] = bool(
        formal.get("gates", {}).get("all_paired_full_logits_exact")
    ) and all(bool(item.get("paired_all_exact")) for item in formal_batches)
    checks["formal_behavior_exact"] = bool(
        formal.get("gates", {}).get("all_paired_behavior_exact")
    ) and all(bool(item.get("paired_behavior_exact")) for item in formal_batches)
    checks["formal_latency_present"] = all(
        item.get("grouped", {}).get("token_latency_p50_seconds") is not None
        and item.get("grouped", {}).get("token_latency_p95_seconds") is not None
        and item.get("fused", {}).get("token_latency_p50_seconds") is not None
        and item.get("fused", {}).get("token_latency_p95_seconds") is not None
        and float(item.get("grouped", {}).get("median_aggregate_decode_tok_per_second", 0.0)) > 0.0
        and float(item.get("fused", {}).get("median_aggregate_decode_tok_per_second", 0.0)) > 0.0
        for item in formal_batches
    )
    b1_formal = next((item for item in formal_batches if int(item.get("batch_size", -1)) == 1), {})
    b1_grouped_speed = float(
        b1_formal.get("grouped", {}).get("median_aggregate_decode_tok_per_second", 0.0)
    )
    b1_fused_speed = float(
        b1_formal.get("fused", {}).get("median_aggregate_decode_tok_per_second", 0.0)
    )
    checks["formal_b1_no_regression"] = (
        b1_fused_speed > 0.0 and b1_grouped_speed / b1_fused_speed >= 0.97
    )
    checks["formal_git_provenance"] = (
        formal.get("git", {}).get("commit") == expected_formal_commit
        and bool(formal.get("git", {}).get("clean_before_output"))
    )
    formal_summary = summary.get("formal_resident", {})
    checks["summary_formal_matches"] = (
        formal_summary.get("all_gates_pass") == bool(formal.get("all_gates_pass"))
        and formal_summary.get("commit") == formal.get("git", {}).get("commit")
        and formal_summary.get("full_logits_exact")
        == bool(formal.get("gates", {}).get("all_paired_full_logits_exact"))
        and formal_summary.get("behavior_exact")
        == bool(formal.get("gates", {}).get("all_paired_behavior_exact"))
        and formal_summary.get("repeated_exact")
        == bool(formal.get("gates", {}).get("all_repeats_exact"))
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
    checks["current_git_commit"] = commit == expected_commit
    checks["current_git_clean"] = clean
    if not checks["current_git_commit"]:
        reasons.append("current_git_commit")
    if not checks["current_git_clean"]:
        reasons.append("current_git_clean")
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
        "formal": {
            "commit": formal.get("git", {}).get("commit"),
            "expected_commit": expected_formal_commit,
            "batches": [int(item.get("batch_size", -1)) for item in formal_batches],
            "max_new_tokens": formal_protocol.get("max_new_tokens"),
            "repeats": formal_protocol.get("repeats"),
            "schedule": formal_protocol.get("schedule"),
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 7.8 results.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--grouped", required=True)
    parser.add_argument("--cohort", action="append", required=True)
    parser.add_argument("--streaming", required=True)
    parser.add_argument("--formal", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--formal-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    summary_path = Path(args.summary).expanduser().resolve()
    grouped_path = Path(args.grouped).expanduser().resolve()
    cohort_paths = [Path(path).expanduser().resolve() for path in args.cohort]
    streaming_path = Path(args.streaming).expanduser().resolve()
    formal_path = Path(args.formal).expanduser().resolve()
    result = verify(
        load(summary_path),
        load(grouped_path),
        [load(path) for path in cohort_paths],
        load(streaming_path),
        load(formal_path),
        args.expected_commit,
        args.formal_commit,
    )
    result["inputs"] = {
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "grouped": str(grouped_path),
        "grouped_sha256": sha256(grouped_path),
        "cohorts": [str(path) for path in cohort_paths],
        "streaming": str(streaming_path),
        "formal": str(formal_path),
        "formal_sha256": sha256(formal_path),
        "expected_commit": args.expected_commit,
        "formal_commit": args.formal_commit,
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

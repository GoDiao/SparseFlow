"""Verify compact Stage 7.11 laptop CLI artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(result: dict[str, Any], *, require_clean: bool = True) -> dict[str, Any]:
    samples = result.get("samples") or []
    protocol = result.get("protocol") or {}
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        cell = (str(sample.get("prompt_id")), int(sample.get("max_new_tokens") or 0))
        by_cell.setdefault(cell, []).append(sample)
    repeat_exact = bool(by_cell) and all(
        all(item.get("exit_code") == 0 for item in values)
        and len(
            {
                (
                    item.get("generated_ids_hash"),
                    json.dumps(item.get("logit_fingerprints"), sort_keys=True),
                    item.get("route_fingerprint"),
                )
                for item in values
            }
        )
        == 1
        for values in by_cell.values()
    )
    token_counts = protocol.get("token_counts") or []
    expected_cells = (
        int(protocol.get("prompt_count") or 0)
        * int(protocol.get("repeats") or 0)
        * len(token_counts)
        if token_counts
        else None
    )
    checks = {
        "kind": result.get("kind") == "sparseflow_stage7_11_laptop_cli",
        "agent": result.get("agent") == "Benchmark",
        "mode": protocol.get("mode") == "process-cold",
        "samples_present": bool(samples),
        "matrix_cardinality": expected_cells is None or len(samples) == expected_cells,
        "git_identity": bool(result.get("git", {}).get("commit")),
        "clean_commit": (not require_clean) or result.get("git", {}).get("dirty") is False,
        "model_identity": bool(result.get("model", {}).get("metadata_sha256")),
        "container_identity": bool(result.get("container", {}).get("metadata_sha256")),
        "all_processes_succeeded": all(item.get("exit_code") == 0 for item in samples),
        "repeat_correctness_exact": repeat_exact,
        "memory_nonzero": all(
            int(item.get("current_rss_bytes") or 0) > 0
            and int(item.get("peak_rss_bytes") or 0) >= int(item.get("current_rss_bytes") or 0)
            for item in samples
        ),
        "cache_budget_respected": all(
            item.get("cache_bytes") is None
            or int(item.get("cached_bytes") or 0) <= int(item["cache_bytes"])
            for item in samples
        ),
        "read_semantics_present": all(item.get("process_read_bytes_semantics") for item in samples),
        "leases_released": all(item.get("leases_after") == 0 for item in samples),
        "compact_only": all("route_audit" not in item and "captured_logits" not in item for item in samples),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_laptop_verification",
        "agent": "Benchmark",
        "checks": checks,
        "verification_passed": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 7.11 laptop CLI output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    verification = verify(result, require_clean=not args.allow_dirty)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))
    return 0 if verification["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

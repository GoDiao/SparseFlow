from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .common import write_json


def load_matrix(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    execution = json.loads((root / "matrix_execution.json").read_text(encoding="utf-8"))
    metadata = {Path(item["output"]).name: item for item in execution["cells"]}
    results = []
    for path in sorted(root.glob("*.json")):
        if path.name == "matrix_execution.json":
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stage") != "7.5.5":
            continue
        value["_path"] = str(path)
        value["_cell"] = metadata[path.name]
        results.append(value)
    return execution, results


def row_for(result: dict[str, Any]) -> dict[str, Any]:
    cell = result["_cell"]
    run = result["runs"][0]
    timing = run["timing"]
    provider = run["provider_delta"] or {}
    decode_tokens = max(1, int(timing["decode_tokens"]))
    hits = int(provider.get("cache_hits", 0))
    misses = int(provider.get("cache_misses", 0))
    requests = int(provider.get("demand_requests", 0))
    return {
        "cell_id": cell["cell_id"],
        "tag": cell["tag"],
        "variant": result["variant"],
        "cache_state": result["storage_policy"]["cache_state"],
        "cache_bytes": result["storage_policy"]["cache_bytes"],
        "max_new_tokens": result["workload"]["max_new_tokens"],
        "prefetch_workers": result["storage_policy"]["prefetch_workers"],
        "coalesce_gap": result["storage_policy"]["coalesce_gap"],
        "ttft_seconds": timing["time_to_first_token_seconds"],
        "decode_tokens_per_second": timing["decode_tokens_per_second"],
        "physical_bytes_per_decode_token": int(provider.get("reader_bytes", 0))
        / decode_tokens,
        "logical_bytes_per_decode_token": int(provider.get("loaded_bytes", 0))
        / decode_tokens,
        "cache_hit_rate": hits / (hits + misses) if hits + misses else 0.0,
        "demand_requests": requests,
        "demand_reuse_hits": int(provider.get("demand_reuse_hits", 0)),
        "demand_prefetch_served": int(provider.get("demand_prefetch_served", 0)),
        "demand_misses": int(provider.get("demand_misses", 0)),
        "prefetch_late": int(provider.get("prefetch_late", 0)),
        "prefetch_wasted_ready_bytes": int(
            provider.get("prefetch_wasted_ready_bytes", 0)
        ),
        "rss_after_generation": int(run["memory"]["rss_after_generation"]),
        "generated_ids": run["quality"]["generated_ids"],
        "logit_sha256": [
            item["sha256"] for item in run["quality"]["logit_fingerprints"]
        ],
    }


def validate(execution: dict[str, Any], results: list[dict[str, Any]], rows: list[dict[str, Any]]):
    issues = []
    if not execution.get("all_complete") or len(results) != 28:
        issues.append("matrix incomplete")
    commits = {item["git"]["commit"] for item in results}
    if len(commits) != 1:
        issues.append("multiple implementation commits")
    identities: dict[int, set[tuple[Any, ...]]] = {}
    for row in rows:
        identities.setdefault(row["max_new_tokens"], set()).add(
            (tuple(row["generated_ids"]), tuple(row["logit_sha256"]))
        )
        result = next(item for item in results if item["_cell"]["cell_id"] == row["cell_id"])
        run = result["runs"][0]
        provider = run["provider_delta"] or {}
        budget = result["storage_policy"]["cache_bytes"]
        cache_after = run.get("cache_after") or {}
        if budget is not None and int(cache_after.get("cached_bytes", 0)) > int(budget):
            issues.append(f"cache budget exceeded: {row['cell_id']}")
        requests = int(provider.get("demand_requests", 0))
        accounted = (
            int(provider.get("demand_reuse_hits", 0))
            + int(provider.get("demand_prefetch_served", 0))
            + int(provider.get("demand_misses", 0))
        )
        if requests != accounted:
            issues.append(f"demand accounting mismatch: {row['cell_id']}")
        prefetch = run.get("prefetch_after")
        if prefetch is not None and int(prefetch.get("failed", 0)):
            issues.append(f"prefetch failure: {row['cell_id']}")
        if result["runtime"].get("expert_storage") != "int8-native":
            issues.append(f"wrong expert storage: {row['cell_id']}")
    for tokens, values in identities.items():
        if len(values) != 1:
            issues.append(f"quality identity mismatch for {tokens} tokens")
    return {
        "matrix_complete": execution.get("all_complete") and len(results) == 28,
        "one_implementation_commit": len(commits) == 1,
        "exact_quality_by_output_length": all(len(values) == 1 for values in identities.values()),
        "cache_budget_and_demand_accounting": not any(
            "budget" in item or "accounting" in item for item in issues
        ),
        "prefetch_failure_free": not any("prefetch failure" in item for item in issues),
        "all_pass": not issues,
        "issues": issues,
    }


def recommendations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    core = [row for row in rows if row["tag"] == "core" and row["variant"] != "C3-S0"]
    by_budget = {}
    for budget in sorted({row["cache_bytes"] for row in core}):
        candidates = [row for row in core if row["cache_bytes"] == budget]
        winner = max(candidates, key=lambda item: item["decode_tokens_per_second"])
        by_budget[str(budget)] = {
            "variant": winner["variant"],
            "decode_tokens_per_second": winner["decode_tokens_per_second"],
            "physical_bytes_per_decode_token": winner["physical_bytes_per_decode_token"],
            "rss_after_generation": winner["rss_after_generation"],
        }
    cold = [row for row in rows if row["tag"] == "cold"]
    cold_winner = max(cold, key=lambda item: item["decode_tokens_per_second"])
    io_rows = [row for row in rows if row["tag"] == "io"]
    io_winner = max(io_rows, key=lambda item: item["decode_tokens_per_second"])
    length = {}
    for tokens in sorted({row["max_new_tokens"] for row in rows if row["tag"] == "length"}):
        candidates = [
            row for row in rows
            if row["tag"] == "length" and row["max_new_tokens"] == tokens
        ]
        winner = max(candidates, key=lambda item: item["decode_tokens_per_second"])
        length[str(tokens)] = {
            "variant": winner["variant"],
            "decode_tokens_per_second": winner["decode_tokens_per_second"],
        }
    return {
        "warm_by_budget": by_budget,
        "model_cold_4g": {
            "variant": cold_winner["variant"],
            "decode_tokens_per_second": cold_winner["decode_tokens_per_second"],
            "ttft_seconds": cold_winner["ttft_seconds"],
        },
        "best_s4_io_setting": {
            "prefetch_workers": io_winner["prefetch_workers"],
            "coalesce_gap": io_winner["coalesce_gap"],
            "decode_tokens_per_second": io_winner["decode_tokens_per_second"],
        },
        "short_output": length,
        "default": {
            "policy": "lru",
            "prefetch": "disabled",
            "reason": (
                "S1 won every measured 1-8 GiB warm budget, the 4 GiB model-cold cell, "
                "and both 8/16-token comparisons; S4 remains opt-in."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.5.5 cache calibration.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.input_dir).expanduser().resolve()
    execution, results = load_matrix(root)
    rows = [row_for(item) for item in results]
    validation = validate(execution, results, rows)
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_5_cache_summary",
        "stage": "7.5.5",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(root),
        "rows": rows,
        "recommendations": recommendations(rows),
        "validation": validation,
    }
    write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result["recommendations"], ensure_ascii=False))
    print(json.dumps(validation, ensure_ascii=False))
    return 0 if validation["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

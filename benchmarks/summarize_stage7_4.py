from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .common import percentile, write_json


def load_sparseflow_results(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    results = []
    for value in paths:
        path = Path(value)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") != "sparseflow_stage7_4_benchmark":
            continue
        if "summary" not in data:
            raise ValueError(f"incomplete benchmark result: {path}")
        data["_path"] = str(path)
        results.append(data)
    if not results:
        raise ValueError("no complete SparseFlow Stage 7.4 results found")
    return results


def decode_reader_bytes(record: dict[str, Any]) -> int:
    return sum(
        int(forward["provider"].get("reader_bytes", 0))
        for forward in record["telemetry"].get("forwards", [])
        if forward["phase"] == "decode"
    )


def validate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    reference_result = next(
        (item for item in results if item["variant"] == "C3-R"), None
    )
    if reference_result is None:
        raise ValueError("Stage 7.4 summary requires a C3-R reference")
    reference_record = reference_result["runs"][0]
    reference_identity = reference_record["runtime_identity"]
    reference_ids = reference_record["quality"]["generated_ids"]
    reference_logits = reference_record["quality"]["logit_fingerprints"]
    model_hashes = {
        (item["model"]["config_sha256"], item["model"]["index_sha256"])
        for item in results
    }
    git_commits = {item["git"]["commit"] for item in results}
    invariants = {
        "all_attributed": all(item.get("agent") == "Main Dev" for item in results),
        "single_model_revision": len(model_hashes) == 1,
        "single_git_commit": len(git_commits) == 1,
        "clean_worktrees": all(not item["git"]["dirty"] for item in results),
        "runtime_identity_exact": True,
        "generated_ids_exact": True,
        "logit_fingerprints_exact": True,
        "streaming_init_zero_expert_io": True,
        "cache_budgets_respected": True,
        "demand_accounting_exact": True,
        "prefetch_failures_zero": True,
    }
    for item in results:
        cache_budget = item["storage_policy"]["cache_bytes"]
        loader = item["load"]["loader"]
        if item["variant"] != "C3-R":
            invariants["streaming_init_zero_expert_io"] &= (
                loader["expert_reader_calls_after_init"] == 0
                and loader["expert_reader_bytes_after_init"] == 0
            )
        for record in item["runs"]:
            invariants["runtime_identity_exact"] &= (
                record["runtime_identity"] == reference_identity
            )
            invariants["generated_ids_exact"] &= (
                record["quality"]["generated_ids"] == reference_ids
            )
            invariants["logit_fingerprints_exact"] &= (
                record["quality"]["logit_fingerprints"] == reference_logits
            )
            if cache_budget is not None:
                cache = record["cache_after"]
                invariants["cache_budgets_respected"] &= (
                    cache is not None and int(cache["cached_bytes"]) <= cache_budget
                )
            provider = record["provider_delta"] or {}
            if item["variant"] != "C3-R":
                invariants["demand_accounting_exact"] &= int(
                    provider.get("demand_requests", 0)
                ) == (
                    int(provider.get("demand_reuse_hits", 0))
                    + int(provider.get("demand_prefetch_served", 0))
                    + int(provider.get("demand_misses", 0))
                )
                prefetch = record.get("prefetch_after") or {}
                invariants["prefetch_failures_zero"] &= int(
                    prefetch.get("failed", 0)
                ) == 0
    return {
        "reference_variant": "C3-R",
        "reference_generated_ids": reference_ids,
        "reference_runtime_identity": reference_identity,
        "model_hashes": [list(value) for value in sorted(model_hashes)],
        "git_commits": sorted(git_commits),
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
    }


def aggregate(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for result in results:
        budget = int(result["storage_policy"]["cache_bytes"] or 0)
        state = result["storage_policy"]["cache_state"]
        for record in result["runs"]:
            grouped[(result["variant"], budget, state)].append((result, record))
    rows = []
    for (variant, budget, state), items in sorted(grouped.items()):
        records = [record for _, record in items]
        decode_speeds = [record["timing"]["decode_tokens_per_second"] for record in records]
        token_latencies = [
            value
            for record in records
            for value in record["timing"]["decode_token_seconds"]
        ]
        decode_bytes = [decode_reader_bytes(record) for record in records]
        decode_tokens = [int(record["timing"]["decode_tokens"]) for record in records]
        provider = [record["provider_delta"] or {} for record in records]
        hits = sum(int(value.get("cache_hits", 0)) for value in provider)
        misses = sum(int(value.get("cache_misses", 0)) for value in provider)
        rows.append(
            {
                "variant": variant,
                "cache_bytes": budget,
                "cache_state": state,
                "samples": len(records),
                "median_load_seconds": statistics.median(
                    result["load"]["seconds"] for result, _ in items
                ),
                "median_ttft_seconds": statistics.median(
                    record["timing"]["time_to_first_token_seconds"] for record in records
                ),
                "median_decode_tokens_per_second": statistics.median(decode_speeds),
                "decode_latency_p50_seconds": percentile(token_latencies, 0.50),
                "decode_latency_p95_seconds": percentile(token_latencies, 0.95),
                "median_decode_reader_bytes": statistics.median(decode_bytes),
                "median_decode_reader_bytes_per_token": statistics.median(
                    bytes_value / tokens if tokens else 0.0
                    for bytes_value, tokens in zip(decode_bytes, decode_tokens)
                ),
                "cache_hit_rate": hits / (hits + misses) if hits + misses else None,
                "demand_reuse_hits": sum(
                    int(value.get("demand_reuse_hits", 0)) for value in provider
                ),
                "demand_prefetch_served": sum(
                    int(value.get("demand_prefetch_served", 0)) for value in provider
                ),
                "demand_misses": sum(int(value.get("demand_misses", 0)) for value in provider),
                "prefetch_wasted_ready_bytes": sum(
                    int(value.get("prefetch_wasted_ready_bytes", 0)) for value in provider
                ),
                "median_process_read_bytes": statistics.median(
                    int(record["process_metrics_delta"].get("read_bytes", 0))
                    for record in records
                ),
                "peak_rss_bytes": max(
                    int(record["memory"]["process_peak_rss"]) for record in records
                ),
                "source_files": sorted({result["_path"] for result, _ in items}),
            }
        )
    return rows


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.6 Stage 7.4 formal benchmark",
        "",
        "> Frozen BF16 C3-R/C3-S benchmark with controlled workload, thread count,",
        "> cache-state labels, three measured samples, raw JSON evidence, and exact",
        "> same-kernel correctness gates. [Main Dev]",
        "",
        "## Correctness gate",
        "",
        "```text",
    ]
    for key, value in summary["validation"]["invariants"].items():
        lines.append(f"{key:38} {value}")
    lines.extend(
        [
            "```",
            "",
            "## Performance matrix",
            "",
            "| Variant | Cache | State | n | TTFT s | Decode tok/s | p50 ms | p95 ms | Read MiB/token | Hit rate | Peak RSS GiB |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["rows"]:
        hit = "-" if row["cache_hit_rate"] is None else f"{row['cache_hit_rate'] * 100:.2f}%"
        lines.append(
            f"| {row['variant']} | {row['cache_bytes'] / 1024**3:.0f} GiB "
            f"| {row['cache_state']} | {row['samples']} "
            f"| {row['median_ttft_seconds']:.3f} "
            f"| {row['median_decode_tokens_per_second']:.4f} "
            f"| {row['decode_latency_p50_seconds'] * 1000:.1f} "
            f"| {row['decode_latency_p95_seconds'] * 1000:.1f} "
            f"| {row['median_decode_reader_bytes_per_token'] / 1024**2:.2f} "
            f"| {hit} | {row['peak_rss_bytes'] / 1024**3:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This report measures the frozen Python BF16 reference runtime on the current",
            "Cascade Lake CPU. It attributes storage policy with C3-R/C3-S same-kernel",
            "comparisons; it is not the Stage 7.5 INT8/native-kernel performance result.",
            "Model-cold means model-local `POSIX_FADV_DONTNEED` was requested and each",
            "sample ran in a fresh process. Workload-warm means one fixed warmup generation",
            "preceded three measured generations in the same process. [Main Dev]",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.4 C3 results.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    results = load_sparseflow_results(sorted(input_dir.glob("*.json")))
    validation = validate_results(results)
    summary = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_4_summary",
        "stage": "7.4",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
        "rows": aggregate(results),
        "raw_result_count": len(results),
    }
    write_json(args.output_json, summary)
    Path(args.output_md).write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False))
    return 0 if validation["all_invariants_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

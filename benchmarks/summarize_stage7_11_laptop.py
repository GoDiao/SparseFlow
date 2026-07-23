"""Summarize the Stage 7.11 CLI matrix without changing its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _stats(samples: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [float(item[field]) for item in samples if item.get(field) is not None]
    return {"p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}


def _cell_key(sample: dict[str, Any]) -> tuple[str, int]:
    return str(sample.get("prompt_id")), int(sample.get("max_new_tokens") or 0)


def summarize(result: dict[str, Any], *, max_peak_rss_gib: float = 11.0, min_32_tok_per_second: float | None = None) -> dict[str, Any]:
    samples = result.get("samples") or []
    protocol = result.get("protocol") or {}
    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for sample in samples:
        cells.setdefault(_cell_key(sample), []).append(sample)

    summaries: list[dict[str, Any]] = []
    for (prompt_id, token_count), values in sorted(cells.items()):
        decode_per_token = [
            float(item["decode_seconds"]) / max(1, int(item.get("generated_tokens") or token_count))
            for item in values
            if item.get("decode_seconds") is not None
        ]
        cell = {
            "prompt_id": prompt_id,
            "max_new_tokens": token_count,
            "sample_count": len(values),
            "successful_samples": sum(item.get("exit_code") == 0 for item in values),
            "load_seconds": _stats(values, "load_seconds"),
            "ttft_seconds": _stats(values, "ttft_seconds"),
            "decode_seconds": _stats(values, "decode_seconds"),
            "decode_seconds_per_token": {
                "p50": _percentile(decode_per_token, 0.50),
                "p95": _percentile(decode_per_token, 0.95),
            },
            "wall_seconds": _stats(values, "wall_seconds"),
            "peak_rss_bytes": _stats(values, "peak_rss_bytes"),
            "process_read_transfer_bytes": _stats(values, "process_read_transfer_bytes"),
            "logical_expert_read_bytes": _stats(values, "logical_expert_read_bytes"),
            "cached_bytes": _stats(values, "cached_bytes"),
            "cache_hits": _stats(values, "cache_hits"),
            "cache_misses": _stats(values, "cache_misses"),
            "cache_evictions": _stats(values, "cache_evictions"),
            "runtime_identities": sorted({json.dumps(item.get("runtime_identity") or {}, sort_keys=True) for item in values}),
        }
        summaries.append(cell)

    token_counts = [int(value) for value in protocol.get("token_counts", [])]
    if not token_counts and protocol.get("max_new_tokens") is not None:
        token_counts = [int(protocol["max_new_tokens"])]
    expected = int(protocol.get("prompt_count") or 0) * int(protocol.get("repeats") or 0) * len(token_counts)
    formal_shape = (
        int(protocol.get("prompt_count") or 0) == 3
        and int(protocol.get("repeats") or 0) == 2
        and sorted(token_counts) == [8, 16, 32]
    )
    all_success = bool(samples) and all(item.get("exit_code") == 0 for item in samples)
    repeat_exact = True
    for values in cells.values():
        fingerprints = {
            (
                item.get("generated_ids_hash"),
                json.dumps(item.get("logit_fingerprints"), sort_keys=True),
                item.get("route_fingerprint"),
            )
            for item in values
        }
        repeat_exact = repeat_exact and len(fingerprints) == 1 and all(item.get("exit_code") == 0 for item in values)
    peak_values = [float(item.get("peak_rss_bytes") or 0) for item in samples]
    peak_ok = bool(peak_values) and max(peak_values) <= max_peak_rss_gib * 1024**3
    threshold_cell = next((cell for cell in summaries if cell["max_new_tokens"] == 32), None)
    threshold_value = None
    if threshold_cell is not None:
        threshold_value = threshold_cell["decode_seconds_per_token"]["p50"]
        if threshold_value and threshold_value > 0:
            threshold_value = 1.0 / threshold_value
    threshold_configured = min_32_tok_per_second is not None
    threshold_ok = (
        threshold_configured
        and threshold_value is not None
        and threshold_value >= float(min_32_tok_per_second)
    )
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_laptop_cli_summary",
        "stage": "7.11.4",
        "agent": "Benchmark",
        "source": {
            "kind": result.get("kind"),
            "git": result.get("git"),
            "model": result.get("model"),
            "container": result.get("container"),
            "protocol": protocol,
            "sample_digest": hashlib.sha256(json.dumps(samples, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "cells": summaries,
        "gates": {
            "matrix_cardinality": expected > 0 and len(samples) == expected,
            "formal_matrix_shape": formal_shape,
            "all_processes_succeeded": all_success,
            "repeat_correctness_exact": repeat_exact,
            "peak_rss_within_budget": peak_ok,
            "performance_threshold_configured": threshold_configured,
            "32_token_threshold": threshold_ok,
            "passed": all((
                formal_shape,
                (expected > 0 and len(samples) == expected),
                all_success,
                repeat_exact,
                peak_ok,
                threshold_configured,
                threshold_ok,
            )),
        },
        "performance_threshold": {
            "max_peak_rss_gib": max_peak_rss_gib,
            "min_32_tok_per_second": min_32_tok_per_second,
            "observed_32_tok_per_second_p50": threshold_value,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage 7.11 laptop CLI samples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-peak-rss-gib", type=float, default=11.0)
    parser.add_argument("--min-32-tok-per-second", type=float)
    args = parser.parse_args(argv)
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    summary = summarize(
        result,
        max_peak_rss_gib=args.max_peak_rss_gib,
        min_32_tok_per_second=args.min_32_tok_per_second,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": summary["gates"]["passed"]}))
    return 0 if summary["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

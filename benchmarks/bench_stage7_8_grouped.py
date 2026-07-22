"""Benchmark the Stage 7.8 true per-expert grouped native operator.

The benchmark keeps the Stage 7.7 real hidden/routes fixture and compares the
old canonical and fused paths with the new grouped operator.  It is a routed
MoE operator gate, not a full Qwen generation benchmark.

[Main Dev]
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from sparseflow.moe_probe import run_routed_experts
from sparseflow.multirequest_moe import combine_layer_rows
from sparseflow.native_moe import run_fused_native_moe, run_grouped_native_moe
from sparseflow.int8_provider import Int8ResidentExpertProvider


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixture(path: Path) -> dict[str, Any]:
    fixture = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(fixture, dict) or fixture.get("kind") != "sparseflow_stage7_7_real_decode_hidden_fixture":
        raise ValueError("invalid Stage 7.7 hidden fixture")
    if len(fixture.get("sessions", [])) < 2:
        raise ValueError("fixture needs at least two sessions")
    return fixture


def layer_batch(records: list[dict[str, Any]], layer: int):
    return combine_layer_rows(records, layer, torch)


def route_group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    multiplicity: Counter[str] = Counter()
    total_assignments = 0
    total_groups = 0
    repeated_assignments = 0
    per_layer: list[dict[str, Any]] = []
    for layer in range(40):
        _hidden, selected, _routing = layer_batch(records, layer)
        counts = Counter(int(value) for value in selected.reshape(-1).tolist())
        total_assignments += sum(counts.values())
        total_groups += len(counts)
        repeated_assignments += sum(count for count in counts.values() if count > 1)
        multiplicity.update(str(count) for count in counts.values())
        per_layer.append({
            "layer": layer,
            "assignments": sum(counts.values()),
            "unique_experts": len(counts),
            "multiplicity": dict(sorted(
                (str(key), int(value)) for key, value in Counter(counts.values()).items()
            )),
        })
    return {
        "rows": len(records),
        "assignments": total_assignments,
        "groups": total_groups,
        "groups_with_reuse": sum(value for key, value in multiplicity.items() if int(key) > 1),
        "assignment_fraction_in_reused_groups": (
            repeated_assignments / total_assignments if total_assignments else 0.0
        ),
        "multiplicity": dict(sorted(
            ((key, int(value)) for key, value in multiplicity.items()),
            key=lambda item: int(item[0]),
        )),
        "per_layer": per_layer,
    }


def _timing_callback(store: dict[str, float]):
    def record(name: str, milliseconds: float) -> None:
        store[name] = store.get(name, 0.0) + float(milliseconds)

    return record


def run_case(
    records: list[dict[str, Any]],
    provider: Int8ResidentExpertProvider,
    variant: str,
    repeats: int,
) -> dict[str, Any]:
    outputs: list[torch.Tensor] = []
    timing: dict[str, float] = {}
    workspace = None
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            layer_outputs = []
            for layer in range(40):
                hidden, selected, routing = layer_batch(records, layer)
                callback = _timing_callback(timing)
                if variant == "canonical-batch-one":
                    rows = []
                    for row in range(hidden.shape[0]):
                        row_output = run_routed_experts(
                            hidden[row : row + 1],
                            selected[row : row + 1],
                            routing[row : row + 1],
                            lambda expert_id, current_layer=layer: provider.get(current_layer, expert_id),
                            prepare_routed=lambda expert_ids, current_layer=layer: provider.prepare(
                                current_layer, expert_ids
                            ),
                            timing_callback=callback,
                        )
                        rows.append(row_output)
                    output = torch.cat(rows, dim=0)
                elif variant == "fused-multi-row":
                    output = run_fused_native_moe(
                        hidden, selected, routing, provider, layer, timing_callback=callback
                    )
                elif variant == "true-grouped":
                    output, workspace = run_grouped_native_moe(
                        hidden,
                        selected,
                        routing,
                        provider,
                        layer,
                        workspace=workspace,
                        timing_callback=callback,
                    )
                else:
                    raise ValueError(f"unknown variant: {variant}")
                layer_outputs.append(output.clone())
            outputs.append(torch.cat(layer_outputs, dim=0))
    elapsed = time.perf_counter() - started
    reference = outputs[0]
    repeat_exact = all(torch.equal(reference, output) for output in outputs[1:])
    return {
        "variant": variant,
        "sessions": len(records),
        "rows_per_layer": len(records),
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "iterations_per_second": repeats / elapsed if elapsed else 0.0,
        "rows_per_second": repeats * len(records) / elapsed if elapsed else 0.0,
        "output_shape": list(reference.shape),
        "repeat_exact": repeat_exact,
        "timing_ms": timing,
        "workspace_bytes": workspace.allocated_bytes() if workspace is not None else 0,
        "output": reference,
    }


def compare_outputs(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left.float() - right.float()).abs()
    return {
        "exact": bool(torch.equal(left, right)),
        "max_abs": float(difference.max().item()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean().item()) if difference.numel() else 0.0,
        "argmax_equal": bool(torch.equal(left.argmax(dim=-1), right.argmax(dim=-1))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Stage 7.8 grouped routed MoE.")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    torch.set_num_threads(args.threads)
    fixture_path = Path(args.fixture).expanduser().resolve()
    fixture = load_fixture(fixture_path)
    all_records = fixture["sessions"]
    batch_sizes = [int(item) for item in args.batches.split(",") if item.strip()]
    if any(size < 1 or size > len(all_records) for size in batch_sizes):
        parser.error("batch size exceeds fixture session count")

    provider = Int8ResidentExpertProvider(
        Path(args.int8_container).expanduser().resolve(), torch, native=True
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_8_grouped_kernel",
        "agent": "Main Dev",
        "fixture": {
            "path": str(fixture_path),
            "sha256": file_sha256(fixture_path),
            "sessions": len(all_records),
        },
        "batches": [],
    }
    try:
        for batch_size in batch_sizes:
            records = all_records[:batch_size]
            canonical = run_case(records, provider, "canonical-batch-one", args.repeats)
            fused = run_case(records, provider, "fused-multi-row", args.repeats)
            grouped = run_case(records, provider, "true-grouped", args.repeats)
            result["batches"].append({
                "batch_size": batch_size,
                "route_groups": route_group_stats(records),
                "canonical": {key: value for key, value in canonical.items() if key != "output"},
                "fused": {key: value for key, value in fused.items() if key != "output"},
                "grouped": {key: value for key, value in grouped.items() if key != "output"},
                "fused_vs_canonical": compare_outputs(canonical["output"], fused["output"]),
                "grouped_vs_canonical": compare_outputs(canonical["output"], grouped["output"]),
                "grouped_vs_fused": compare_outputs(fused["output"], grouped["output"]),
                "fused_speedup": (
                    fused["rows_per_second"] / canonical["rows_per_second"]
                    if canonical["rows_per_second"] else 0.0
                ),
                "grouped_speedup": (
                    grouped["rows_per_second"] / canonical["rows_per_second"]
                    if canonical["rows_per_second"] else 0.0
                ),
                "grouped_over_fused": (
                    grouped["rows_per_second"] / fused["rows_per_second"]
                    if fused["rows_per_second"] else 0.0
                ),
            })
    finally:
        provider.close()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "batches": [
            {
                "batch_size": item["batch_size"],
                "fused_speedup": item["fused_speedup"],
                "grouped_speedup": item["grouped_speedup"],
                "grouped_over_fused": item["grouped_over_fused"],
                "grouped_exact": item["grouped_vs_fused"]["exact"],
            }
            for item in result["batches"]
        ],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

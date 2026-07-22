"""Benchmark real hidden-state multi-row routed execution for Stage 7.7.

This is a routed-MoE gate, not a full generation benchmark.  It compares the
existing canonical batch-one dispatch with the existing fused multi-row native
operator using the same captured hidden rows and routes.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from sparseflow.moe_probe import run_routed_experts
from sparseflow.multirequest_moe import combine_layer_rows
from sparseflow.native_moe import run_fused_native_moe
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


def run_case(
    records: list[dict[str, Any]],
    provider: Int8ResidentExpertProvider,
    variant: str,
    repeats: int,
) -> dict[str, Any]:
    outputs: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for _ in range(repeats):
            layer_outputs = []
            for layer in range(40):
                hidden, selected, routing = layer_batch(records, layer)
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
                        )
                        rows.append(row_output)
                    output = torch.cat(rows, dim=0)
                elif variant == "fused-multi-row":
                    output = run_fused_native_moe(
                        hidden,
                        selected,
                        routing,
                        provider,
                        layer,
                    )
                else:
                    raise ValueError(f"unknown variant: {variant}")
                layer_outputs.append(output)
            outputs.append(torch.cat(layer_outputs, dim=0))
    elapsed = time.perf_counter() - started
    reference = outputs[0]
    exact = all(torch.equal(reference, output) for output in outputs[1:])
    return {
        "variant": variant,
        "sessions": len(records),
        "rows_per_layer": len(records),
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "iterations_per_second": repeats / elapsed if elapsed else 0.0,
        "output_shape": list(reference.shape),
        "repeat_exact": exact,
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
    parser = argparse.ArgumentParser(description="Benchmark Stage 7.7 grouped routed MoE.")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--repeats", type=int, default=2)
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
        Path(args.int8_container).expanduser().resolve(),
        torch,
        native=True,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_7_grouped_kernel",
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
            comparison = compare_outputs(canonical["output"], fused["output"])
            canonical.pop("output")
            fused.pop("output")
            fused_tps = fused["iterations_per_second"] * batch_size
            canonical_tps = canonical["iterations_per_second"] * batch_size
            result["batches"].append({
                "batch_size": batch_size,
                "canonical": canonical,
                "fused": fused,
                "comparison": comparison,
                "aggregate_speedup": (
                    fused_tps / canonical_tps if canonical_tps else 0.0
                ),
                "aggregate_canonical_rows_per_second": canonical_tps,
                "aggregate_fused_rows_per_second": fused_tps,
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
                "speedup": item["aggregate_speedup"],
                "exact": item["comparison"]["exact"],
                "max_abs": item["comparison"]["max_abs"],
            }
            for item in result["batches"]
        ],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

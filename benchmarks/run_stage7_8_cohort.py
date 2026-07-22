"""Run the Stage 7.8 resident fixed-cohort Qwen gate.

This loads one memory-native INT8 resident runtime for the grouped cohort and
one equivalent hybrid runtime for independent-session comparison.  The two
models are loaded sequentially to avoid doubling the resident footprint.

[Main Dev]
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from sparseflow.fixed_cohort import (
    compare_batched_cohort_results,
    compare_fixed_cohort_results,
    generate_fixed_cohort,
    run_independent_fixed_cohort,
)
from sparseflow.text_runtime import Qwen36TextRuntime


DEFAULT_PROMPTS = (
    "用一句话解释缓存命中率。",
    "Explain why sparse expert models can save memory.",
    "Write a Python function that returns the square of an integer.",
    "计算 17 * 19，并只输出结果。",
    "什么是 NVMe？",
    "What is a routing expert in an MoE model?",
    "给出一个简单的数学等式。",
    "Describe a deterministic benchmark in one sentence.",
)


def load_runtime(
    model: Path, container: Path, dispatch: str, telemetry: str
) -> Qwen36TextRuntime:
    return Qwen36TextRuntime.from_pretrained(
        model,
        mode="resident",
        dtype="bf16",
        cache_slots=None,
        cache_bytes=None,
        telemetry_level=telemetry,
        load_mode="memory-native",
        expert_storage="int8-native",
        int8_container=container,
        native_dispatch=dispatch,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.8 resident fixed cohort.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--telemetry", choices=("summary", "profile"), default="summary")
    parser.add_argument("--prompt", action="append", dest="prompts")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.batch_size > len(DEFAULT_PROMPTS):
        parser.error(f"batch-size must be between 1 and {len(DEFAULT_PROMPTS)}")
    if args.max_new_tokens < 2:
        parser.error("max-new-tokens must be at least 2")
    torch.set_num_threads(args.threads)
    prompts = tuple((args.prompts or DEFAULT_PROMPTS)[: args.batch_size])
    if len(prompts) != args.batch_size:
        parser.error("the number of --prompt values must cover batch-size")
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()

    grouped_runtime = load_runtime(model, container, "grouped", args.telemetry)
    try:
        grouped = generate_fixed_cohort(
        grouped_runtime,
        prompts,
        max_new_tokens=args.max_new_tokens,
        stop_on_eos=False,
        capture_logits=True,
        )
    finally:
        grouped_runtime.close()
    gc.collect()

    fused_runtime = load_runtime(model, container, "hybrid", args.telemetry)
    try:
        fused_batch = generate_fixed_cohort(
            fused_runtime,
            prompts,
            max_new_tokens=args.max_new_tokens,
            stop_on_eos=False,
            capture_logits=True,
        )
        independent = run_independent_fixed_cohort(
            fused_runtime,
            prompts,
            max_new_tokens=args.max_new_tokens,
            capture_logits=True,
        )
    finally:
        fused_runtime.close()
    gc.collect()

    correctness = compare_fixed_cohort_results(grouped, independent)
    grouped_vs_fused = compare_batched_cohort_results(grouped, fused_batch)
    grouped["captured_logits"] = None
    fused_batch["captured_logits"] = None
    for item in independent:
        item["captured_logits"] = None
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_8_resident_fixed_cohort",
        "agent": "Main Dev",
        "model": str(model),
        "int8_container": str(container),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "threads": args.threads,
        "prompts": list(prompts),
        "grouped": grouped,
        "fused_batch": fused_batch,
        "independent": independent,
        "correctness": correctness,
        "grouped_vs_fused_batch": grouped_vs_fused,
        "grouped_over_fused_batch_decode_speedup": (
            float(fused_batch["decode_seconds"]) / float(grouped["decode_seconds"])
            if float(grouped["decode_seconds"]) > 0
            else 0.0
        ),
        "aggregate_speedup": None,
    }
    independent_decode = sum(float(item["decode_seconds"]) for item in independent)
    grouped_decode = float(grouped["decode_seconds"])
    if grouped_decode > 0 and independent_decode > 0:
        result["aggregate_speedup"] = independent_decode / grouped_decode
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "batch_size": args.batch_size,
        "aggregate_speedup": result["aggregate_speedup"],
        "correctness": result["correctness"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if result["correctness"]["generated_ids_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

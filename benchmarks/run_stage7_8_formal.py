"""Run the formal Stage 7.8 resident grouped-vs-fused acceptance gate.

The harness keeps both memory-native INT8 resident runtimes in one process and
executes three ``A B B A`` blocks for each requested cohort size.  Full logits
are compared in memory and only compact fingerprints, timings, routes, and
gate decisions are written to the result file.

[Main Dev]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

from sparseflow.fixed_cohort import generate_fixed_cohort
from sparseflow.text_runtime import Qwen36TextRuntime


BASE_PROMPTS = (
    "用一句话解释缓存命中率。",
    "Explain why sparse expert models can save memory.",
    "Write a Python function that returns the square of an integer.",
    "计算 17 * 19，并只输出结果。",
    "什么是 NVMe？",
    "What is a routing expert in an MoE model?",
    "给出一个简单的数学等式。",
    "Describe a deterministic benchmark in one sentence.",
)


def select_equal_prompts(model: Path) -> tuple[tuple[str, ...], list[int]]:
    """Build eight distinct prompts with one shared chat-template length."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=True,
        use_fast=True,
    )

    def token_length(text: str) -> int:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return int(encoded["input_ids"].shape[-1])

    candidate_maps: list[dict[int, str]] = []
    for base in BASE_PROMPTS:
        candidates: dict[int, str] = {}
        for padding_words in range(0, 128):
            padding = " ".join(["benchmark"] * padding_words)
            candidate = base if not padding else f"{base} {padding}"
            candidates[token_length(candidate)] = candidate
        candidate_maps.append(candidates)
    common_lengths = set(candidate_maps[0])
    for candidates in candidate_maps[1:]:
        common_lengths.intersection_update(candidates)
    target = next((value for value in sorted(common_lengths) if value >= 32), None)
    if target is None:
        raise RuntimeError("could not construct eight equal-length formal prompts")
    prompts = tuple(candidates[target] for candidates in candidate_maps)
    lengths = [token_length(prompt) for prompt in prompts]
    if len(set(lengths)) != 1:
        raise AssertionError(f"formal prompt lengths are not equal: {lengths}")
    return prompts, lengths


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_runtime(
    model: Path,
    container: Path,
    dispatch: str,
    threads: int,
) -> Qwen36TextRuntime:
    runtime = Qwen36TextRuntime.from_pretrained(
        model,
        mode="resident",
        dtype="bf16",
        cache_slots=None,
        cache_bytes=None,
        telemetry_level="summary",
        load_mode="memory-native",
        expert_storage="int8-native",
        int8_container=container,
        native_dispatch=dispatch,
    )
    expected = {
        "mode": "resident",
        "load_mode": "memory-native",
        "expert_storage": "int8-native",
        "native_dispatch": dispatch,
        "backend_id": "sparseflow-int8-native-resident",
        "provider_native": True,
    }
    actual = runtime_identity(runtime)
    if actual != expected:
        runtime.close()
        raise RuntimeError(
            f"runtime identity mismatch for {dispatch}: expected={expected!r} actual={actual!r}"
        )
    del threads
    return runtime


def runtime_identity(runtime: Qwen36TextRuntime) -> dict[str, Any]:
    provider = runtime.provider
    return {
        "mode": runtime.mode,
        "load_mode": runtime.load_mode,
        "expert_storage": runtime.expert_storage,
        "native_dispatch": runtime.native_dispatch,
        "backend_id": getattr(provider, "backend_id", None),
        "provider_native": bool(getattr(provider, "native", False)),
    }


def route_fingerprint(records: Any) -> str:
    return json_hash(records)


def run_observation(
    runtime: Qwen36TextRuntime,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    label: str,
    repetition: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    generated = generate_fixed_cohort(
        runtime,
        prompts,
        max_new_tokens=max_new_tokens,
        stop_on_eos=False,
        capture_logits=True,
    )
    wall_seconds = time.perf_counter() - started
    records = generated.get("route_audit") or []
    return {
        "label": label,
        "repetition": repetition,
        "runtime_identity": runtime_identity(runtime),
        "batch_size": int(generated["batch_size"]),
        "generated_tokens": int(generated["generated_tokens"]),
        "generated_ids": generated["generated_ids"],
        "texts": generated["texts"],
        "logit_fingerprints": generated["logit_fingerprints"],
        "route_fingerprint": route_fingerprint(records),
        "route_record_count": len(records),
        "prefill_seconds": float(generated["prefill_seconds"]),
        "decode_seconds": float(generated["decode_seconds"]),
        "decode_token_seconds": [
            float(value) for value in generated["decode_token_seconds"]
        ],
        "aggregate_decode_tok_per_second": float(
            generated["aggregate_decode_tok_per_second"]
        ),
        "session_decode_tok_per_second": float(
            generated["session_decode_tok_per_second"]
        ),
        "captured_logits": generated["captured_logits"],
        "wall_seconds": wall_seconds,
    }


def compare_observations(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_logits = left["captured_logits"]
    right_logits = right["captured_logits"]
    if len(left_logits) != len(right_logits):
        raise ValueError("logit step count differs")
    exact_logits = True
    argmax_equal = True
    max_abs = 0.0
    mean_abs_sum = 0.0
    value_count = 0
    for left_step, right_step in zip(left_logits, right_logits, strict=True):
        if len(left_step) != len(right_step):
            raise ValueError("logit batch row count differs")
        for left_row, right_row in zip(left_step, right_step, strict=True):
            exact_logits = exact_logits and bool(torch.equal(left_row, right_row))
            difference = (left_row - right_row).abs()
            max_abs = max(max_abs, float(difference.max().item()))
            mean_abs_sum += float(difference.sum().item())
            value_count += int(difference.numel())
            argmax_equal = argmax_equal and bool(
                torch.equal(left_row.argmax(dim=-1), right_row.argmax(dim=-1))
            )
    return {
        "generated_ids_equal": left["generated_ids"] == right["generated_ids"],
        "texts_equal": left["texts"] == right["texts"],
        "logit_fingerprints_equal": left["logit_fingerprints"] == right["logit_fingerprints"],
        "routes_equal": left["route_fingerprint"] == right["route_fingerprint"],
        "logits_exact": exact_logits,
        "argmax_equal": argmax_equal,
        "max_abs_logit_error": max_abs,
        "mean_abs_logit_error": mean_abs_sum / value_count if value_count else 0.0,
    }


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in observation.items()
        if key != "captured_logits"
    }


def summarize_dispatch(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        raise ValueError("missing dispatch observations")
    anchor = observations[0]
    repeat_exact = all(
        item["generated_ids"] == anchor["generated_ids"]
        and item["texts"] == anchor["texts"]
        and item["logit_fingerprints"] == anchor["logit_fingerprints"]
        and item["route_fingerprint"] == anchor["route_fingerprint"]
        for item in observations[1:]
    )
    token_latencies = [
        value
        for item in observations
        for value in item["decode_token_seconds"]
    ]
    decode_seconds = [item["decode_seconds"] for item in observations]
    return {
        "samples": len(observations),
        "repeat_exact": repeat_exact,
        "runtime_identity_exact": all(
            item["runtime_identity"] == anchor["runtime_identity"]
            for item in observations
        ),
        "generated_ids": anchor["generated_ids"],
        "texts": anchor["texts"],
        "route_fingerprint": anchor["route_fingerprint"],
        "median_decode_seconds": statistics.median(decode_seconds),
        "median_aggregate_decode_tok_per_second": statistics.median(
            item["aggregate_decode_tok_per_second"] for item in observations
        ),
        "token_latency_p50_seconds": percentile(token_latencies, 0.50),
        "token_latency_p95_seconds": percentile(token_latencies, 0.95),
        "token_latency_samples": len(token_latencies),
        "runtime_identity": anchor["runtime_identity"],
    }


def run_batch(
    grouped_runtime: Qwen36TextRuntime,
    fused_runtime: Qwen36TextRuntime,
    prompts: tuple[str, ...],
    max_new_tokens: int,
    repeats: int,
) -> dict[str, Any]:
    grouped_observations: list[dict[str, Any]] = []
    fused_observations: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for repetition in range(1, repeats + 1):
        a1 = run_observation(grouped_runtime, prompts, max_new_tokens, "A", repetition)
        b1 = run_observation(fused_runtime, prompts, max_new_tokens, "B", repetition)
        b2 = run_observation(fused_runtime, prompts, max_new_tokens, "B", repetition)
        a2 = run_observation(grouped_runtime, prompts, max_new_tokens, "A", repetition)
        grouped_observations.extend((a1, a2))
        fused_observations.extend((b1, b2))
        first_pair = compare_observations(a1, b1)
        second_pair = compare_observations(a2, b2)
        paired.extend((first_pair, second_pair))
        blocks.append({
            "repetition": repetition,
            "schedule": ["A(grouped)", "B(hybrid)", "B(hybrid)", "A(grouped)"],
            "first_pair": first_pair,
            "second_pair": second_pair,
        })
        del a1, b1, b2, a2
        gc.collect()
    grouped_summary = summarize_dispatch(grouped_observations)
    fused_summary = summarize_dispatch(fused_observations)
    return {
        "batch_size": len(prompts),
        "prompts": list(prompts),
        "max_new_tokens": max_new_tokens,
        "repeats": repeats,
        "schedule": "ABBA",
        "same_process": True,
        "grouped": grouped_summary,
        "fused": fused_summary,
        "paired_comparisons": paired,
        "blocks": blocks,
        "paired_all_exact": all(
            item["generated_ids_equal"]
            and item["texts_equal"]
            and item["logit_fingerprints_equal"]
            and item["routes_equal"]
            and item["logits_exact"]
            and item["argmax_equal"]
            for item in paired
        ),
        "paired_behavior_exact": all(
            item["generated_ids_equal"]
            and item["texts_equal"]
            and item["logit_fingerprints_equal"]
            and item["routes_equal"]
            and item["argmax_equal"]
            for item in paired
        ),
        "max_abs_logit_error": max(
            (item["max_abs_logit_error"] for item in paired),
            default=0.0,
        ),
        "mean_abs_logit_error": max(
            (item["mean_abs_logit_error"] for item in paired),
            default=0.0,
        ),
        "observations": {
            "grouped": [compact_observation(item) for item in grouped_observations],
            "fused": [compact_observation(item) for item in fused_observations],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run formal Stage 7.8 resident gate.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("formal gate requires at least three repeats")
    if args.max_new_tokens < 32:
        parser.error("formal gate requires at least 32 generated tokens")
    batch_sizes = tuple(int(item) for item in args.batches.split(",") if item.strip())
    if batch_sizes != (1, 4, 8):
        parser.error("formal gate requires batches exactly 1,4,8")
    if args.threads < 1:
        parser.error("threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()
    base_prompts = BASE_PROMPTS
    prompts, prompt_token_lengths = select_equal_prompts(model)
    started_utc = time.time()
    commit = git_value("rev-parse", "HEAD")
    clean_before_output = git_value("status", "--porcelain") == ""
    grouped_runtime = None
    fused_runtime = None
    batches: list[dict[str, Any]] = []
    try:
        grouped_runtime = load_runtime(model, container, "grouped", args.threads)
        fused_runtime = load_runtime(model, container, "hybrid", args.threads)
        for batch_size in batch_sizes:
            print(f"stage7.8-formal batch={batch_size} schedule=ABBA", flush=True)
            batches.append(
                run_batch(
                    grouped_runtime,
                    fused_runtime,
                    tuple(prompts[:batch_size]),
                    args.max_new_tokens,
                    args.repeats,
                )
            )
    finally:
        if grouped_runtime is not None:
            grouped_runtime.close()
        if fused_runtime is not None:
            fused_runtime.close()
        gc.collect()
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_8_formal_resident_abba",
        "agent": "Main Dev",
        "protocol": {
            "batches": list(batch_sizes),
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "schedule": "ABBA",
            "threads": args.threads,
            "quality_gate": "equivalent-32-token-long-generation",
        },
        "model": str(model),
        "int8_container": str(container),
        "base_prompts": list(base_prompts),
        "prompts": list(prompts),
        "prompt_token_lengths": prompt_token_lengths,
        "git": {
            "commit": commit,
            "clean_before_output": clean_before_output,
        },
        "process": {"pid": os.getpid(), "same_process": True},
        "batches": batches,
        "gates": {
            "protocol_exact": args.max_new_tokens >= 32 and args.repeats >= 3,
            "all_runtime_identity_exact": all(
                batch["grouped"]["runtime_identity_exact"]
                and batch["fused"]["runtime_identity_exact"]
                for batch in batches
            ),
            "all_paired_full_logits_exact": all(
                batch["paired_all_exact"] for batch in batches
            ),
            "all_paired_behavior_exact": all(
                batch["paired_behavior_exact"] for batch in batches
            ),
            "all_repeats_exact": all(
                batch["grouped"]["repeat_exact"]
                and batch["fused"]["repeat_exact"]
                for batch in batches
            ),
        },
        "finished_epoch_seconds": time.time() - started_utc,
    }
    result["all_gates_pass"] = all(result["gates"].values())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "all_gates_pass": result["all_gates_pass"],
        "git": result["git"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

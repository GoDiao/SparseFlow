"""Run the Stage 7.9 resident/low-memory validation ladder.

Full logits are captured only in process for the comparison.  The result file
stores compact fingerprints, routes, generated IDs, and accounting instead of
serializing the vocabulary logits for every token.

[Benchmark]
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

from sparseflow.release import apply_preset, container_identity, model_identity
from sparseflow.text_runtime import Qwen36TextRuntime


LEVEL_TOKENS = {"smoke": 32, "standard": 128, "endurance": 512}


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def run_path(
    model: Path,
    container: Path,
    preset: str,
    prompt: str,
    tokens: int,
) -> dict[str, Any]:
    config = apply_preset(preset)
    started = time.perf_counter()
    runtime = Qwen36TextRuntime.from_pretrained(
        model,
        mode=config["mode"],
        dtype="bf16",
        cache_slots=None if config["cache_bytes"] is not None else 16,
        cache_bytes=config["cache_bytes"],
        prefetch_workers=config["prefetch_workers"],
        coalesce_gap=0,
        cache_policy=config["cache_policy"],
        prefetch_policy=config["prefetch_policy"],
        telemetry_level="summary",
        experts_implementation="eager",
        load_mode=config["load_mode"],
        expert_storage=config["expert_storage"],
        int8_container=container,
        native_dispatch=config["native_dispatch"],
    )
    load_seconds = time.perf_counter() - started
    try:
        result = runtime.greedy_generate(
            prompt,
            max_new_tokens=tokens,
            stop_on_eos=False,
            record_logit_fingerprints=True,
            capture_logits=True,
        )
    finally:
        runtime.close()
    captured = result.pop("captured_logits")
    return {
        "preset": config,
        "load_seconds": load_seconds,
        "runtime_identity": result["runtime_identity"],
        "generated_ids": result["generated_ids"],
        "text": result["text"],
        "generated_tokens": result["generated_tokens"],
        "logit_fingerprints": result["logit_fingerprints"],
        "route_audit": result["route_audit"],
        "prefill_seconds": result["prefill_seconds"],
        "decode_seconds": result["decode_seconds"],
        "decode_token_seconds": result["decode_token_seconds"],
        "cache": result["cache"],
        "provider_storage": result["provider_storage"],
        "memory": result["memory"],
        "telemetry": result["telemetry"],
        "captured_logits_in_process": captured,
    }


def compare_paths(resident: dict[str, Any], streaming: dict[str, Any], level: str) -> dict[str, Any]:
    selected_steps = set(range(len(resident["captured_logits_in_process"])))
    if level == "standard":
        selected_steps = {
            step for step in selected_steps if step == 0 or step % 16 == 0
        }
    elif level == "endurance":
        selected_steps = {
            step for step in selected_steps if step == 0 or step % 32 == 0
        }
    logits_exact = True
    max_abs = 0.0
    mean_sum = 0.0
    count = 0
    for step in sorted(selected_steps):
        left = resident["captured_logits_in_process"][step]
        right = streaming["captured_logits_in_process"][step]
        difference = (left - right).abs()
        logits_exact = logits_exact and bool(torch.equal(left, right))
        max_abs = max(max_abs, float(difference.max().item()))
        mean_sum += float(difference.sum().item())
        count += int(difference.numel())
    resident_routes = resident["route_audit"]
    streaming_routes = streaming["route_audit"]
    return {
        "generated_ids_equal": resident["generated_ids"] == streaming["generated_ids"],
        "text_equal": resident["text"] == streaming["text"],
        "fingerprints_equal": resident["logit_fingerprints"] == streaming["logit_fingerprints"],
        "routes_equal": resident_routes == streaming_routes,
        "selected_logit_steps": sorted(selected_steps),
        "selected_logits_exact": logits_exact,
        "max_abs_logit_error": max_abs,
        "mean_abs_logit_error": mean_sum / count if count else 0.0,
        "all_equal": (
            resident["generated_ids"] == streaming["generated_ids"]
            and resident["text"] == streaming["text"]
            and resident["logit_fingerprints"] == streaming["logit_fingerprints"]
            and resident_routes == streaming_routes
            and logits_exact
        ),
    }


def compact_path(result: dict[str, Any]) -> dict[str, Any]:
    route_audit = result.get("route_audit") or []
    normalized_routes = [
        {key: value for key, value in record.items() if key != "expert_ids"}
        for record in route_audit
    ]
    route_payload = json.dumps(
        normalized_routes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    compact = {
        key: value for key, value in result.items()
        if key not in {"captured_logits_in_process", "route_audit"}
    }
    compact["route_audit"] = normalized_routes
    compact["route_fingerprint"] = hashlib.sha256(route_payload).hexdigest()
    return compact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 7.9 validation ladder.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--level", choices=tuple(LEVEL_TOKENS), default="smoke")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("repeats must be positive")
    tokens = LEVEL_TOKENS[args.level]
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()
    os.environ.setdefault("OMP_NUM_THREADS", "10")
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    try:
        torch.set_num_interop_threads(max(1, min(torch.get_num_threads(), 4)))
    except RuntimeError:
        pass

    samples = []
    for repetition in range(1, args.repeats + 1):
        for prompt_index, prompt in enumerate(args.prompt):
            print(
                f"stage7.9 level={args.level} repeat={repetition} prompt={prompt_index} "
                "resident+streaming",
                flush=True,
            )
            resident = run_path(model, container, "stable", prompt, tokens)
            gc.collect()
            streaming = run_path(model, container, "low-memory", prompt, tokens)
            comparison = compare_paths(resident, streaming, args.level)
            samples.append(
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "tokens": tokens,
                    "comparison": comparison,
                    "resident": compact_path(resident),
                    "streaming": compact_path(streaming),
                }
            )
            del resident, streaming
            gc.collect()

    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_validation",
        "stage": "7.9",
        "agent": "Benchmark",
        "protocol": {
            "level": args.level,
            "tokens": tokens,
            "repeats": args.repeats,
            "prompt_count": len(args.prompt),
            "full_logits_persisted": False,
            "comparison": "resident hybrid vs single-request streaming hybrid S1",
        },
        "model": model_identity(model),
        "container": container_identity(container),
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "clean_before_output": git_value("status", "--porcelain") == "",
        },
        "samples": samples,
        "gates": {
            "samples_present": bool(samples),
            "all_exact": all(item["comparison"]["all_equal"] for item in samples),
            "all_runtime_identity_present": all(
                item["resident"]["runtime_identity"]
                and item["streaming"]["runtime_identity"]
                for item in samples
            ),
            "streaming_cache_budget_respected": all(
                item["streaming"]["cache"] is not None
                and item["streaming"]["cache"]["cached_bytes"] <= 4 * 1024**3
                for item in samples
            ),
            "no_full_logits_persisted": all(
                "captured_logits_in_process" not in item
                for sample in samples
                for item in (sample["resident"], sample["streaming"])
            ),
            "routes_compact": all(
                "expert_ids" not in record
                for sample in samples
                for side in ("resident", "streaming")
                for record in sample[side]["route_audit"]
            ),
        },
    }
    result["all_gates_pass"] = all(result["gates"].values())
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_gates_pass": result["all_gates_pass"]}))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Benchmark]

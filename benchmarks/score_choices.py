from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .common import (
    delta,
    git_snapshot,
    host_snapshot,
    model_snapshot,
    process_snapshot,
    sha256_text,
    write_json,
)
from .run_cpu import materialize_cpu_resident


def load_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"ctx", "choices", "gold"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("choice rows require ctx, choices, and gold")
    return rows[:limit] if limit else rows


def token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"]
    return values[0] if values and isinstance(values[0], list) else values


def split_continuation(tokenizer: Any, context: str, choice: str) -> tuple[list[int], list[int]]:
    context_ids = token_ids(tokenizer, context)
    full_ids = token_ids(tokenizer, context + choice)
    prefix = len(context_ids)
    while prefix > 0 and (prefix > len(full_ids) or full_ids[:prefix] != context_ids[:prefix]):
        prefix -= 1
    continuation = full_ids[prefix:]
    if not continuation:
        continuation = token_ids(tokenizer, choice)
        full_ids = context_ids + continuation
        prefix = len(context_ids)
    if prefix < 1 or not continuation:
        raise ValueError(f"could not construct a continuation for context={context!r}")
    return full_ids, continuation


def score_one(model: Any, tokenizer: Any, context: str, choice: str, torch: Any) -> dict[str, Any]:
    full_ids, continuation = split_continuation(tokenizer, context, choice)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cpu")
    attention_mask = torch.ones_like(input_ids)
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[0]
    elapsed = time.perf_counter() - start
    positions = torch.arange(len(full_ids) - len(continuation), len(full_ids))
    token_logits = logits[positions - 1]
    target = torch.tensor(continuation, dtype=torch.long)
    log_probs = torch.log_softmax(token_logits.float(), dim=-1)
    selected = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    values = [float(value) for value in selected.tolist()]
    return {
        "loglikelihood": sum(values),
        "token_loglikelihoods": values,
        "continuation_tokens": len(continuation),
        "continuation_chars": max(1, len(choice)),
        "forward_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Colibri-style CPU choice scoring.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--data", default="benchmarks/manifests/colibri_smoke.jsonl")
    parser.add_argument("--output", default="benchmarks/results/colibri-smoke.json")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(max(1, min(args.threads, 4)))
        except RuntimeError:
            pass

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    root = Path.cwd().resolve()
    model_dir = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    data_path = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data).resolve()
    rows = load_rows(data_path, args.limit or None)
    if not rows:
        parser.error("choice data contains no rows")

    started = process_snapshot()
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).eval()
    materialize_seconds = materialize_cpu_resident(model, torch)
    load_seconds = time.perf_counter() - load_start
    loaded = process_snapshot()

    result: dict[str, Any] = {
        "schema_version": 1,
        "backend": "transformers_cpu_choice_score",
        "model": model_snapshot(model_dir),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": args.dtype,
            "threads": torch.get_num_threads(),
        },
        "host": host_snapshot(),
        "git": git_snapshot(root),
        "data": {
            "path": str(data_path),
            "rows": len(rows),
            "sha256": sha256_text(data_path.read_text(encoding="utf-8").splitlines(True)),
        },
        "load": {
            "seconds": load_seconds + materialize_seconds,
            "model_load_seconds": load_seconds,
            "materialize_seconds": materialize_seconds,
            "metrics": loaded,
            "physical_resident": True,
        },
        "questions": [],
    }

    for index, row in enumerate(rows):
        choice_scores = [
            score_one(model, tokenizer, row["ctx"], choice, torch)
            for choice in row["choices"]
        ]
        best = max(range(len(choice_scores)), key=lambda item: choice_scores[item]["loglikelihood"])
        best_char = max(
            range(len(choice_scores)),
            key=lambda item: choice_scores[item]["loglikelihood"]
            / choice_scores[item]["continuation_chars"],
        )
        best_token = max(
            range(len(choice_scores)),
            key=lambda item: choice_scores[item]["loglikelihood"]
            / choice_scores[item]["continuation_tokens"],
        )
        result["questions"].append(
            {
                "id": row.get("id", f"question-{index}"),
                "task": row.get("task"),
                "gold": int(row["gold"]),
                "prediction": best,
                "prediction_norm_char": best_char,
                "prediction_norm_token": best_token,
                "correct": best == int(row["gold"]),
                "correct_norm_char": best_char == int(row["gold"]),
                "correct_norm_token": best_token == int(row["gold"]),
                "choices": choice_scores,
            }
        )
        write_json(args.output, result)
        print(f"question={index + 1}/{len(rows)} id={result['questions'][-1]['id']}", flush=True)

    questions = result["questions"]
    result["summary"] = {
        "n": len(questions),
        "accuracy": sum(item["correct"] for item in questions) / len(questions),
        "acc_norm_char": sum(item["correct_norm_char"] for item in questions) / len(questions),
        "acc_norm_token": sum(item["correct_norm_token"] for item in questions) / len(questions),
        "forward_seconds": sum(
            choice["forward_seconds"]
            for item in questions
            for choice in item["choices"]
        ),
    }
    result["finished_metrics"] = delta(started, process_snapshot())
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

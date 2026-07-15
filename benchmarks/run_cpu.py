from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    delta,
    git_snapshot,
    host_snapshot,
    model_snapshot,
    now,
    percentile,
    process_snapshot,
    sha256_text,
    write_json,
)


def load_manifest(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not {"id", "text"}.issubset(row):
            raise ValueError("manifest rows require id and text")
        rows.append(row)
    return rows[:limit] if limit else rows


def encode_chat(tokenizer: Any, text: str) -> dict[str, Any]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {
        key: value.to("cpu") if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def materialize_cpu_resident(model: Any, torch: Any) -> float:
    """Copy mmap-backed CPU tensors into ordinary anonymous CPU memory."""
    start = now()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.device.type != "cpu":
                raise RuntimeError(f"expected CPU parameter, got {parameter.device}")
            parameter.data = parameter.detach().clone().contiguous()
        for module in model.modules():
            for name, buffer in module._buffers.items():
                if buffer is not None:
                    if buffer.device.type != "cpu":
                        raise RuntimeError(f"expected CPU buffer, got {buffer.device}")
                    module._buffers[name] = buffer.detach().clone().contiguous()
    return now() - start


def greedy_generate_timed(
    model: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
    torch: Any,
) -> dict[str, Any]:
    """Run greedy generation with prefill and decode timed separately.

    The first forward pass consumes the prompt and produces the first new
    token, so it is reported as prefill/TTFT. Only subsequent one-token
    forwards are reported as decode throughput. This avoids labeling the
    end-to-end ``generate`` duration as steady-state decode speed.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    cache_position = torch.arange(input_ids.shape[-1], device=input_ids.device)
    prefill_start = now()
    with torch.inference_mode():
        first = _generation_forward(
            model,
            input_ids,
            attention_mask,
            torch,
            past_key_values=None,
            cache_position=cache_position,
        )
        next_token = first.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    prefill_seconds = now() - prefill_start

    generated_tokens = [next_token]
    past_key_values = getattr(first, "past_key_values", None)
    if max_new_tokens > 1 and past_key_values is None:
        raise RuntimeError("model did not return past_key_values for decode timing")

    decode_token_seconds: list[float] = []
    decode_start = now()
    for _ in range(1, max_new_tokens):
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )
        cache_position = cache_position[-1:] + 1
        token_start = now()
        with torch.inference_mode():
            output = _generation_forward(
                model,
                next_token,
                attention_mask,
                torch,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        decode_token_seconds.append(now() - token_start)
        generated_tokens.append(next_token)
        past_key_values = getattr(output, "past_key_values", None)

    decode_seconds = now() - decode_start
    generated = torch.cat(generated_tokens, dim=-1)
    decode_tokens = max(0, int(generated.shape[-1]) - 1)
    return {
        "output": torch.cat([input_ids, generated], dim=-1),
        "prefill_seconds": prefill_seconds,
        "time_to_first_token_seconds": prefill_seconds,
        "decode_seconds": decode_seconds if decode_tokens else 0.0,
        "decode_tokens": decode_tokens,
        "decode_token_seconds": decode_token_seconds,
        "end_to_end_seconds": prefill_seconds + decode_seconds,
    }


def _generation_forward(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    torch: Any,
    past_key_values: Any,
    cache_position: Any,
) -> Any:
    """Use the model's generation input contract when it exposes one.

    Multimodal Qwen wrappers require request-local ``cache_position`` and
    position preparation even for text-only incremental decoding. Calling the
    raw forward method directly can reuse stale mRoPE state after a warmup.
    """

    prepare = getattr(model, "prepare_inputs_for_generation", None)
    if prepare is not None:
        model_inputs = prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        return model(**model_inputs)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a reproducible CPU Full Resident Transformers benchmark."
    )
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--manifest", default="benchmarks/manifests/cpu_dev.jsonl")
    parser.add_argument("--output", default="benchmarks/results/cpu-resident.json")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--measure-prefill",
        action="store_true",
        help="Compatibility flag; prefill is always timed separately now.",
    )
    parser.add_argument(
        "--no-materialize",
        action="store_true",
        help="Keep safetensors CPU mappings demand-paged; for mmap comparison only.",
    )
    args = parser.parse_args(argv)

    if args.threads < 0:
        parser.error("--threads must be non-negative")
    if args.runs < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        parser.error("runs must be positive; warmup may be zero; max-new-tokens must be positive")

    root = Path.cwd().resolve()
    model_dir = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    manifest_path = (
        root / args.manifest
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    ).resolve()
    rows = load_manifest(manifest_path, args.limit or None)
    if not rows:
        parser.error("manifest contains no rows")

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

    dtype = dtype_from_name(torch, args.dtype)
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines(keepends=True)
    result: dict[str, Any] = {
        "schema_version": 2,
        "stage": "7.4",
        "agent": "Main Dev",
        "workstream_owner": "Benchmark",
        "experiment_id": f"cpu-resident-{args.dtype}",
        "backend": "transformers_cpu_resident",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": args.dtype,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_resident": not args.no_materialize,
        },
        "host": host_snapshot(),
        "git": git_snapshot(root),
        "workload": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_text(manifest_lines),
            "rows": len(rows),
            "runs": args.runs,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "prefill_always_measured": True,
            "measure_prefill_compat_flag": args.measure_prefill,
            "batch_size": 1,
            "cache_state": "resident-workload-warm" if args.warmup else "resident-app-cold",
        },
        "load": {},
        "runs": [],
    }

    print(f"backend={result['backend']}")
    print(f"model={model_dir}")
    print(f"dtype={args.dtype} threads={torch.get_num_threads()}")
    print("loading CPU-resident model...", flush=True)

    import time as _time

    load_before = process_snapshot()
    load_start = _time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=dtype,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    materialize_seconds = 0.0
    if not args.no_materialize:
        print("materializing CPU-resident parameter storage...", flush=True)
        materialize_seconds = materialize_cpu_resident(model, torch)
    load_seconds = _time.perf_counter() - load_start
    load_after = process_snapshot()
    result["load"] = {
        "seconds": load_seconds,
        "materialize_seconds": materialize_seconds,
        "metrics_delta": delta(load_before, load_after),
    }
    print(f"loaded_seconds={load_seconds:.3f}", flush=True)

    encoded = {row["id"]: encode_chat(tokenizer, row["text"]) for row in rows}

    for warmup_index in range(args.warmup):
        row = rows[warmup_index % len(rows)]
        with torch.inference_mode():
            greedy_generate_timed(
                model,
                encoded[row["id"]],
                args.max_new_tokens,
                torch,
            )
        print(f"warmup={warmup_index + 1}/{args.warmup}", flush=True)

    for run_index in range(args.runs):
        for row in rows:
            inputs = encoded[row["id"]]
            input_tokens = int(inputs["input_ids"].shape[-1])
            before = process_snapshot()
            timed = greedy_generate_timed(
                model, inputs, args.max_new_tokens, torch
            )
            output = timed["output"]
            after = process_snapshot()

            new_tokens = output[0, input_tokens:]
            generated = int(new_tokens.shape[-1])
            decode_tokens = int(timed["decode_tokens"])
            decode_seconds = float(timed["decode_seconds"])
            end_to_end_seconds = float(timed["end_to_end_seconds"])
            record = {
                "run_index": run_index,
                "prompt_id": row["id"],
                "group": row.get("group"),
                "category": row.get("category"),
                "input_tokens": input_tokens,
                "generated_tokens": generated,
                "prefill_seconds": float(timed["prefill_seconds"]),
                "time_to_first_token_seconds": float(
                    timed["time_to_first_token_seconds"]
                ),
                "decode_seconds": decode_seconds,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": decode_tokens / decode_seconds
                if decode_seconds > 0 and decode_tokens > 0
                else 0.0,
                "decode_token_seconds": timed["decode_token_seconds"],
                "end_to_end_seconds": end_to_end_seconds,
                "end_to_end_tokens_per_second": generated / end_to_end_seconds
                if end_to_end_seconds > 0
                else 0.0,
                "metrics_delta": delta(before, after),
                "output_token_ids": [int(value) for value in new_tokens.tolist()],
                "decoded_output": tokenizer.decode(new_tokens, skip_special_tokens=False),
            }
            result["runs"].append(record)
            write_json(args.output, result)
            print(
                f"run={run_index + 1}/{args.runs} prompt={row['id']} "
                f"input={input_tokens} output={generated} "
                f"decode_tok_s={record['decode_tokens_per_second']:.4f} "
                f"e2e_tok_s={record['end_to_end_tokens_per_second']:.4f}",
                flush=True,
            )

    decode_speeds = [
        float(item["decode_tokens_per_second"])
        for item in result["runs"]
        if int(item["decode_tokens"]) > 0
    ]
    end_to_end_speeds = [
        float(item["end_to_end_tokens_per_second"]) for item in result["runs"]
    ]
    result["summary"] = {
        "run_count": len(result["runs"]),
        "median_decode_tokens_per_second": statistics.median(decode_speeds)
        if decode_speeds
        else None,
        "min_decode_tokens_per_second": min(decode_speeds)
        if decode_speeds
        else None,
        "max_decode_tokens_per_second": max(decode_speeds)
        if decode_speeds
        else None,
        "median_end_to_end_tokens_per_second": statistics.median(end_to_end_speeds),
        "decode_latency_p50_seconds": percentile(
            [
                value
                for item in result["runs"]
                for value in item["decode_token_seconds"]
            ],
            0.50,
        ),
        "decode_latency_p95_seconds": percentile(
            [
                value
                for item in result["runs"]
                for value in item["decode_token_seconds"]
            ],
            0.95,
        ),
        "generated_ids_exact_across_runs": len(
            {tuple(item["output_token_ids"]) for item in result["runs"]}
        )
        == 1,
        "peak_rss_bytes": max(
            [result["load"]["metrics_delta"].get("peak_rss_bytes", 0)]
            + [item["metrics_delta"].get("peak_rss_bytes", 0) for item in result["runs"]]
        ),
    }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    print(f"results={Path(args.output).resolve()}")
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Stage 7.4 result attribution and percentile metrics: [Main Dev]

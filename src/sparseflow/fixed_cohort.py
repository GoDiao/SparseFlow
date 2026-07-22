"""Fixed-cohort resident generation harness for Stage 7.8.

The harness deliberately has no admission, cancellation, or finished-row
compaction.  It exists to measure whether real batched Qwen decoder rows can
benefit from the grouped routed-MoE operator while each session keeps its own
generation state.

[Main Dev]
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from .text_runtime import Qwen36TextRuntime, _logit_fingerprint


def _left_pad_batch(runtime: Qwen36TextRuntime, prompts: Sequence[str]):
    if not prompts:
        raise ValueError("fixed cohort requires at least one prompt")
    encoded = [runtime.encode_chat(prompt) for prompt in prompts]
    torch = runtime.torch
    lengths = [int(item["input_ids"].shape[-1]) for item in encoded]
    if len(set(lengths)) != 1:
        raise ValueError(
            "Stage 7.8 fixed cohort requires equal encoded prompt lengths; "
            f"got {lengths}"
        )
    max_length = max(lengths)
    pad_id = runtime.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = runtime.tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("fixed cohort tokenizer requires pad_token_id or eos_token_id")
    input_ids = torch.full(
        (len(prompts), max_length),
        int(pad_id),
        dtype=torch.long,
        device=runtime.device,
    )
    attention_mask = torch.zeros(
        (len(prompts), max_length), dtype=torch.long, device=runtime.device
    )
    for row, item in enumerate(encoded):
        ids = item["input_ids"].to(runtime.device)
        length = ids.shape[-1]
        input_ids[row, max_length - length :] = ids[0]
        attention_mask[row, max_length - length :] = 1
    return input_ids, attention_mask, lengths


def generate_fixed_cohort(
    runtime: Qwen36TextRuntime,
    prompts: Sequence[str],
    max_new_tokens: int = 8,
    stop_on_eos: bool = False,
    capture_logits: bool = False,
) -> dict[str, Any]:
    """Generate equal-length output rows with one batched model forward."""

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    torch = runtime.torch
    input_ids, attention_mask, lengths = _left_pad_batch(runtime, prompts)
    batch_size = int(input_ids.shape[0])
    route_start = len(runtime.route_audit.records) if runtime.route_audit is not None else 0
    runtime._begin_forward(0, "prefill", int(input_ids.shape[-1]) - 1)
    prefill_started = time.perf_counter()
    with torch.inference_mode():
        first = runtime.prefill({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
    prefill_seconds = time.perf_counter() - prefill_started
    runtime.telemetry.add_timing("model_forward", prefill_seconds * 1000.0)
    last_indices = torch.tensor(
        [length - 1 + (int(input_ids.shape[-1]) - length) for length in lengths],
        dtype=torch.long,
        device=runtime.device,
    )
    row_indices = torch.arange(batch_size, dtype=torch.long, device=runtime.device)
    next_token = first.logits[row_indices, last_indices].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    logit_fingerprints = [[
        _logit_fingerprint(first.logits[row : row + 1, last_indices[row]], torch)
        for row in range(batch_size)
    ]]
    captured_logits = ([
        [
            first.logits[row : row + 1, last_indices[row]]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
            for row in range(batch_size)
        ]
    ] if capture_logits else None)
    past = getattr(first, "past_key_values", None)
    eos_id = getattr(runtime.tokenizer, "eos_token_id", None)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=runtime.device)
    decode_durations: list[float] = []
    previous_decode_routes: dict[int, tuple[int, ...]] = {}

    for forward_index in range(1, max_new_tokens):
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=-1
        )
        runtime._begin_forward(
            forward_index,
            "decode",
            int(input_ids.shape[-1]) + forward_index - 1,
        )
        if runtime.provider is not None:
            runtime.provider.predict(previous_decode_routes)
        route_start_decode = (
            len(runtime.route_audit.records) if runtime.route_audit is not None else 0
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = runtime.decode(next_token, attention_mask, past)
        elapsed = time.perf_counter() - started
        decode_durations.append(elapsed)
        runtime.telemetry.add_timing("model_forward", elapsed * 1000.0)
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        logit_fingerprints.append([
            _logit_fingerprint(output.logits[row : row + 1, -1], torch)
            for row in range(batch_size)
        ])
        if captured_logits is not None:
            captured_logits.append([
                output.logits[row : row + 1, -1]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clone()
                for row in range(batch_size)
            ])
        past = getattr(output, "past_key_values", None)
        if runtime.route_audit is not None:
            previous_decode_routes = runtime.route_audit.routes_since(route_start_decode)
        if eos_id is not None:
            finished |= next_token.squeeze(-1).eq(eos_id)
        if stop_on_eos and bool(finished.all()):
            break

    generated_ids = torch.cat(generated, dim=-1)
    texts = [
        runtime.tokenizer.decode(
            row.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for row in generated_ids
    ]
    if runtime.provider is not None:
        runtime.provider.finish_generation()
    route_records = (
        runtime.route_audit.records[route_start:]
        if runtime.route_audit is not None
        else None
    )
    decode_seconds = sum(decode_durations)
    return {
        "agent": "Main Dev",
        "batch_size": batch_size,
        "prompts": list(prompts),
        "input_lengths": lengths,
        "generated_ids": generated_ids.detach().cpu().tolist(),
        "texts": texts,
        "generated_tokens": int(generated_ids.shape[-1]),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_token_seconds": decode_durations,
        "aggregate_decode_tok_per_second": (
            batch_size * max(0, int(generated_ids.shape[-1]) - 1) / decode_seconds
            if decode_seconds else 0.0
        ),
        "session_decode_tok_per_second": (
            (max(0, int(generated_ids.shape[-1]) - 1) / decode_seconds)
            if decode_seconds else 0.0
        ),
        "logit_fingerprints": logit_fingerprints,
        "captured_logits": captured_logits,
        "finished": finished.detach().cpu().tolist(),
        "route_audit": route_records,
        "provider": runtime.provider.counters() if runtime.provider is not None else None,
        "telemetry": runtime.telemetry.as_dict(),
    }


def run_independent_fixed_cohort(
    runtime: Qwen36TextRuntime,
    prompts: Sequence[str],
    max_new_tokens: int = 8,
    capture_logits: bool = False,
) -> list[dict[str, Any]]:
    """Run the same requests independently using one loaded runtime."""

    return [
        runtime.greedy_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_on_eos=False,
            record_logit_fingerprints=True,
            capture_logits=capture_logits,
        )
        for prompt in prompts
    ]


def compare_fixed_cohort_results(
    grouped: dict[str, Any], independent: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if len(independent) != int(grouped["batch_size"]):
        raise ValueError("independent result count does not match cohort size")
    grouped_ids = grouped["generated_ids"]
    independent_ids = [item["generated_ids"] for item in independent]
    grouped_fingerprints = grouped["logit_fingerprints"]
    independent_fingerprints = [item["logit_fingerprints"] for item in independent]
    ids_equal = grouped_ids == independent_ids
    fingerprints_equal = all(
        grouped_fingerprints[step][row] == independent_fingerprints[row][step]
        for step in range(len(grouped_fingerprints))
        for row in range(len(independent))
    )
    result = {
        "generated_ids_equal": ids_equal,
        "logit_fingerprints_equal": fingerprints_equal,
        "texts_equal": grouped["texts"] == [item["text"] for item in independent],
        "all_equal": ids_equal and fingerprints_equal and grouped["texts"] == [item["text"] for item in independent],
    }
    grouped_logits = grouped.get("captured_logits")
    independent_logits = [item.get("captured_logits") for item in independent]
    if grouped_logits is not None and all(item is not None for item in independent_logits):
        max_abs = 0.0
        mean_abs_sum = 0.0
        value_count = 0
        argmax_equal = True
        for step, rows in enumerate(grouped_logits):
            for row, values in enumerate(rows):
                other = independent_logits[row][step]
                difference = (values - other).abs()
                max_abs = max(max_abs, float(difference.max().item()))
                mean_abs_sum += float(difference.sum().item())
                value_count += int(difference.numel())
                argmax_equal = argmax_equal and bool(
                    torch_equal(values.argmax(dim=-1), other.argmax(dim=-1))
                )
        result["logits"] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs_sum / value_count if value_count else 0.0,
            "argmax_equal": argmax_equal,
            "allclose_atol_1e-3": max_abs <= 1e-3,
            "allclose_atol_1e-2": max_abs <= 1e-2,
        }
    return result


def compare_batched_cohort_results(
    grouped: dict[str, Any], fused: dict[str, Any]
) -> dict[str, Any]:
    """Compare grouped and legacy fused runs with identical batch inputs."""

    result = {
        "generated_ids_equal": grouped["generated_ids"] == fused["generated_ids"],
        "logit_fingerprints_equal": grouped["logit_fingerprints"] == fused["logit_fingerprints"],
        "texts_equal": grouped["texts"] == fused["texts"],
    }
    grouped_logits = grouped.get("captured_logits")
    fused_logits = fused.get("captured_logits")
    if grouped_logits is not None and fused_logits is not None:
        max_abs = 0.0
        mean_abs_sum = 0.0
        value_count = 0
        argmax_equal = True
        for grouped_step, fused_step in zip(grouped_logits, fused_logits, strict=True):
            for grouped_row, fused_row in zip(grouped_step, fused_step, strict=True):
                difference = (grouped_row - fused_row).abs()
                max_abs = max(max_abs, float(difference.max().item()))
                mean_abs_sum += float(difference.sum().item())
                value_count += int(difference.numel())
                argmax_equal = argmax_equal and torch_equal(
                    grouped_row.argmax(dim=-1), fused_row.argmax(dim=-1)
                )
        result["logits"] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs_sum / value_count if value_count else 0.0,
            "argmax_equal": argmax_equal,
            "allclose_atol_1e-3": max_abs <= 1e-3,
            "allclose_atol_1e-2": max_abs <= 1e-2,
        }
    result["all_equal"] = all(result.values())
    return result


def torch_equal(left, right) -> bool:
    return bool(left.equal(right))


# [Main Dev]

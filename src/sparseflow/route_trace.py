from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .trace import RouteTrace, TraceGroup, trace_sha256


_LAYER_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")


def capture_route_trace(
    model_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    """Run a text-only Qwen3.5-MoE generation and capture selected experts.

    The hook records the router's actual ``selected_experts`` tensor. The
    resulting JSON can be replayed by ``expert-bench --trace``.
    """

    root = Path(model_dir).expanduser().resolve()
    torch, transformers, tokenizer, model = _load_model(root)
    return _capture_with_model(root, prompt, max_new_tokens, torch, transformers, tokenizer, model)


def capture_route_traces(
    model_dir: str | Path,
    prompts: list[dict[str, Any]],
    max_new_tokens: int = 8,
) -> list[dict[str, Any]]:
    """Capture multiple prompts while loading the large model only once."""

    root = Path(model_dir).expanduser().resolve()
    torch, transformers, tokenizer, model = _load_model(root)
    return [
        _capture_with_model(root, str(item["text"]), max_new_tokens, torch, transformers, tokenizer, model, item.get("id"))
        for item in prompts
    ]


def load_prompt_manifest(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("prompt manifest rows require a text field")
        prompts.append(item)
        if limit and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"prompt manifest is empty: {path}")
    return prompts


def _load_model(root: Path):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        root,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.eval()
    return torch, transformers, tokenizer, model


def _capture_with_model(
    root: Path,
    prompt: str,
    max_new_tokens: int,
    torch,
    transformers,
    tokenizer,
    model,
    prompt_id: str | None = None,
) -> dict[str, Any]:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    encoded = tokenizer(prompt, return_tensors="pt")
    selected_by_forward: dict[int, dict[int, dict[int, tuple[int, ...]]]] = {}
    state = {"forward_index": -1, "layer_zero_seen": False}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) < 3:
                raise RuntimeError(f"router output for layer {layer} has no selected_experts tuple")
            selected = output[2]
            if not hasattr(selected, "detach"):
                raise RuntimeError(f"router output for layer {layer} has invalid selected_experts")
            if layer == 0:
                state["forward_index"] += 1
                state["layer_zero_seen"] = True
            values = selected.detach().to("cpu").tolist()
            by_layer = selected_by_forward.setdefault(int(state["forward_index"]), {}).setdefault(layer, {})
            for row_index, expert_ids in enumerate(values):
                by_layer[row_index] = tuple(int(expert_id) for expert_id in expert_ids)

        return hook

    for name, module in model.named_modules():
        match = _LAYER_GATE_RE.search(name)
        if match:
            handles.append(module.register_forward_hook(make_hook(int(match.group(1)))))
    if not handles:
        raise RuntimeError("could not find Qwen3.5 MoE router modules named *.layers.<n>.mlp.gate")

    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    if not state["layer_zero_seen"]:
        raise RuntimeError("router hooks ran but layer 0 was not observed")

    input_ids = encoded["input_ids"][0].tolist()
    generated_ids = generated[0].tolist()
    groups: list[TraceGroup] = []
    for forward in sorted(selected_by_forward):
        by_layer = selected_by_forward[forward]
        row_ids = sorted({row for rows in by_layer.values() for row in rows})
        phase = "prefill" if forward == 0 else "decode"
        for row in row_ids:
            token_position = row if forward == 0 else len(input_ids) + forward - 1 + row
            if token_position < len(input_ids):
                token_id = input_ids[token_position]
            elif token_position < len(generated_ids):
                token_id = generated_ids[token_position]
            else:
                token_id = None
            requests = tuple(
                (layer, expert)
                for layer in sorted(by_layer)
                for expert in by_layer[layer].get(row, ())
            )
            groups.append(
                TraceGroup(
                    forward=forward,
                    phase=phase,
                    row=row,
                    token_position=token_position,
                    token_id=token_id,
                    requests=requests,
                )
            )
    structured = {
        "schema_version": 2,
        "kind": "qwen3_5_moe_route_trace",
        "forwards": _serialize_groups(groups),
    }
    return {
        **structured,
        "kind": "qwen3_5_moe_route_trace",
        "model": {
            "path": str(root),
            "model_type": getattr(model.config, "model_type", None),
        },
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "workload": {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "input_tokens": len(input_ids),
            "max_new_tokens": max_new_tokens,
            "generated_tokens": len(generated_ids) - len(input_ids),
            "input_ids": input_ids,
            "generated_ids": generated_ids,
            "forward_calls": state["forward_index"] + 1,
        },
        "trace_sha256": trace_sha256(RouteTrace(groups=tuple(groups))),
    }


def _serialize_groups(groups: list[TraceGroup]) -> list[dict[str, Any]]:
    forwards: dict[int, dict[str, Any]] = {}
    for group in groups:
        forward = forwards.setdefault(
            group.forward,
            {"forward": group.forward, "phase": group.phase, "rows": []},
        )
        by_layer: dict[int, list[int]] = {}
        for layer, expert in group.requests:
            by_layer.setdefault(layer, []).append(expert)
        forward["rows"].append(
            {
                "row": group.row,
                "token_position": group.token_position,
                "token_id": group.token_id,
                "layers": [
                    {"layer": layer, "expert_ids": experts}
                    for layer, experts in sorted(by_layer.items())
                ],
            }
        )
    return [forwards[key] for key in sorted(forwards)]


def write_route_trace(path: str | Path, trace: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

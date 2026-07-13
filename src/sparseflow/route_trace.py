from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


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

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    root = Path(model_dir).expanduser().resolve()
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

    encoded = tokenizer(prompt, return_tensors="pt")
    requests: list[dict[str, int]] = []
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
            for row_index, expert_ids in enumerate(values):
                for expert_id in expert_ids:
                    requests.append(
                        {
                            "forward": int(state["forward_index"]),
                            "row": int(row_index),
                            "layer": layer,
                            "expert": int(expert_id),
                        }
                    )

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

    canonical = json.dumps(requests, separators=(",", ":")).encode("utf-8")
    input_ids = encoded["input_ids"][0].tolist()
    generated_ids = generated[0].tolist()
    return {
        "schema_version": 1,
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
            "prompt": prompt,
            "input_tokens": len(input_ids),
            "max_new_tokens": max_new_tokens,
            "generated_tokens": len(generated_ids) - len(input_ids),
            "input_ids": input_ids,
            "generated_ids": generated_ids,
            "forward_calls": state["forward_index"] + 1,
        },
        "requests": requests,
        "trace_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_route_trace(path: str | Path, trace: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

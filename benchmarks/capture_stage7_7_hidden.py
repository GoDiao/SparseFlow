"""Capture real decode hidden rows and routes for grouped-kernel benchmarks.

The capture uses the production memory-native INT8 runtime.  One independent
decode row is saved for each prompt; the grouped benchmark combines only these
real rows and never fabricates router selections.

[Main Dev]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from sparseflow.route_trace import load_prompt_manifest
from sparseflow.text_runtime import Qwen36TextRuntime, SparseFlowQwenExperts


def capture(
    model_dir: Path,
    int8_container: Path,
    prompts: list[dict[str, Any]],
    output: Path,
    threads: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    runtime = Qwen36TextRuntime.from_pretrained(
        model_dir,
        mode="resident",
        dtype="bf16",
        load_mode="memory-native",
        expert_storage="int8-native",
        int8_container=int8_container,
        native_dispatch="hybrid",
        telemetry_level="none",
        cache_policy="lru",
        prefetch_workers=0,
    )
    experts = [
        module
        for module in runtime.model.modules()
        if isinstance(module, SparseFlowQwenExperts)
    ]
    if len(experts) != 40:
        runtime.close()
        raise RuntimeError(f"expected 40 SparseFlow expert modules, got {len(experts)}")

    state: dict[str, Any] = {"forward": -1, "active": False, "layers": {}}
    handles = []

    def make_hook(module: SparseFlowQwenExperts):
        def hook(_module, inputs):
            if module.layer == 0:
                state["forward"] += 1
            if not state["active"] or state["forward"] != 1:
                return
            hidden, selected, routing = inputs[:3]
            state["layers"][module.layer] = {
                "hidden_states": hidden.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                "selected_experts": selected.detach().to("cpu", dtype=torch.long).contiguous(),
                "routing_weights": routing.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
            }

        return hook

    for module in experts:
        handles.append(module.register_forward_pre_hook(make_hook(module)))
    records: list[dict[str, Any]] = []
    try:
        for index, prompt in enumerate(prompts):
            state["forward"] = -1
            state["active"] = True
            state["layers"] = {}
            generated = runtime.greedy_generate(
                str(prompt["text"]),
                max_new_tokens=2,
                record_logit_fingerprints=True,
            )
            state["active"] = False
            if sorted(state["layers"]) != list(range(40)):
                raise RuntimeError(
                    f"prompt {prompt.get('id', index)} did not capture all layers: "
                    f"{sorted(state['layers'])}"
                )
            records.append(
                {
                    "session_id": f"session-{prompt.get('id', index)}",
                    "prompt_id": str(prompt.get("id", index)),
                    "generated_ids": generated["generated_ids"],
                    "layers": state["layers"],
                }
            )
    finally:
        for handle in handles:
            handle.remove()
        runtime.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_7_real_decode_hidden_fixture",
        "agent": "Main Dev",
        "model": str(model_dir),
        "int8_container": str(int8_container),
        "sessions": records,
    }
    torch.save(payload, output)
    return {
        "agent": "Main Dev",
        "sessions": len(records),
        "layers_per_session": 40,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture real Stage 7.7 decode hidden rows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    prompts = load_prompt_manifest(args.manifest, args.limit)
    if len(prompts) < 2:
        parser.error("at least two prompts are required")
    result = capture(
        Path(args.model).expanduser().resolve(),
        Path(args.int8_container).expanduser().resolve(),
        prompts,
        Path(args.output).expanduser().resolve(),
        args.threads,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .safetensors import TensorSpan


@dataclass(frozen=True)
class TensorClass:
    category: str
    layer: int | None = None
    expert_part: str | None = None


class TensorClassifier:
    def classify(self, tensor: TensorSpan) -> TensorClass:
        name = tensor.name
        if "embed_tokens.weight" in name or name == "lm_head.weight":
            return TensorClass("embed_lm_head")
        if ".mlp.gate.weight" in name or ".mlp.shared_expert_gate.weight" in name:
            return TensorClass("routers")
        if ".mlp.shared_expert" in name:
            return TensorClass("shared_experts")
        if ".self_attn." in name or ".linear_attn." in name:
            return TensorClass("attention_or_linear_attention")
        if ".visual." in name or "vision" in name.lower():
            return TensorClass("vision")
        return TensorClass("other_dense")


class Qwen36MoeClassifier(TensorClassifier):
    routed_re = re.compile(
        r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$"
    )

    def classify(self, tensor: TensorSpan) -> TensorClass:
        match = self.routed_re.fullmatch(tensor.name)
        if match:
            return TensorClass("routed_experts", int(match.group(1)), match.group(2))
        return super().classify(tensor)


def classifier_for_model(model_dir: str | Path, model_type: str | None) -> TensorClassifier:
    if model_type == "qwen3_5_moe":
        return Qwen36MoeClassifier()
    return TensorClassifier()

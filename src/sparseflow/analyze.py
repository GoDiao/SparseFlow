from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .classifier import classifier_for_model
from .safetensors import ShardIndex

CATEGORIES = (
    "routed_experts",
    "shared_experts",
    "routers",
    "attention_or_linear_attention",
    "vision",
    "embed_lm_head",
    "other_dense",
)


@dataclass(frozen=True)
class ModelConfig:
    model_type: str | None
    text_config: dict[str, Any]
    raw: dict[str, Any]


def load_config(model_dir: str | Path) -> ModelConfig:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise ValueError(f"missing config.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    text_config = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else raw
    return ModelConfig(
        model_type=raw.get("model_type") or text_config.get("model_type"),
        text_config=text_config,
        raw=raw,
    )


def analyze_model(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).expanduser().resolve()
    config = load_config(root)
    index = ShardIndex.from_dir(root)
    classifier = classifier_for_model(root, config.model_type)

    category_bytes = {category: 0 for category in CATEGORIES}
    layer_expert_parts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    shard_bytes = sum(path.stat().st_size for path in root.glob("*.safetensors"))

    for tensor in index:
        tensor_class = classifier.classify(tensor)
        category_bytes[tensor_class.category] += tensor.nbytes
        if tensor_class.category == "routed_experts" and tensor_class.layer is not None:
            layer_expert_parts[tensor_class.layer][tensor_class.expert_part or "unknown"] += tensor.nbytes

    per_layer_expert_bytes = {
        layer: sum(parts.values()) for layer, parts in sorted(layer_expert_parts.items())
    }
    text_config = config.text_config
    top_k = int(text_config.get("num_experts_per_tok", 0) or 0)
    num_experts = int(text_config.get("num_experts", text_config.get("n_routed_experts", 0)) or 0)
    num_layers = int(text_config.get("num_hidden_layers", 0) or 0)
    per_expert_bytes = _estimate_per_expert_bytes(per_layer_expert_bytes, num_experts)
    per_cache_slot_set_bytes = per_expert_bytes * len(per_layer_expert_bytes)
    cold_read_per_token = per_expert_bytes * top_k * len(per_layer_expert_bytes)

    dense_resident_bytes = sum(
        value for key, value in category_bytes.items() if key != "routed_experts"
    )

    return {
        "schema_version": 1,
        "model": {
            "path": str(root),
            "model_type": config.model_type,
            "tensors": len(index),
            "shards": len(list(root.glob("*.safetensors"))),
            "safetensors_bytes": shard_bytes,
            "num_hidden_layers": num_layers,
            "hidden_size": int(text_config.get("hidden_size", 0) or 0),
            "num_experts": num_experts,
            "num_experts_per_tok": top_k,
            "context_length": int(text_config.get("max_position_embeddings", 0) or 0),
        },
        "footprint": {
            "category_bytes": category_bytes,
            "dense_resident_bytes": dense_resident_bytes,
            "routed_expert_bytes": category_bytes["routed_experts"],
            "per_layer_expert_bytes": per_layer_expert_bytes,
            "typical_layer_total_expert_bytes": _median(per_layer_expert_bytes.values()),
            "typical_expert_bytes": per_expert_bytes,
            "per_layer_cache_slot_set_bytes": per_cache_slot_set_bytes,
            "cold_expert_read_per_token_bytes": cold_read_per_token,
        },
        "expert_layout": {
            "layers": len(per_layer_expert_bytes),
            "parts_by_layer": {
                str(layer): dict(parts) for layer, parts in sorted(layer_expert_parts.items())
            },
            "fused": config.model_type == "qwen3_5_moe",
        },
    }


def _median(values) -> int:
    values = list(values)
    return int(statistics.median(values)) if values else 0


def _estimate_per_expert_bytes(per_layer: dict[int, int], num_experts: int) -> int:
    if not per_layer or num_experts <= 0:
        return 0
    return int(statistics.median(per_layer.values()) // num_experts)

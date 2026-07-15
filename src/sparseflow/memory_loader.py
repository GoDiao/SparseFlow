from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .analyze import load_config
from .safetensors import ShardIndex, TensorSpan


LoadAction = Literal["resident", "stream", "skip"]


@dataclass(frozen=True)
class MemoryLoadEntry:
    source_name: str
    target_name: str | None
    action: LoadAction
    reason: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "action": self.action,
            "reason": self.reason,
            "shard": str(self.shard),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class MemoryLoadPlan:
    model_dir: Path
    adapter: str
    entries: tuple[MemoryLoadEntry, ...]
    num_hidden_layers: int

    def entries_for(self, action: LoadAction) -> tuple[MemoryLoadEntry, ...]:
        return tuple(entry for entry in self.entries if entry.action == action)

    @property
    def resident_entries(self) -> tuple[MemoryLoadEntry, ...]:
        return self.entries_for("resident")

    @property
    def stream_entries(self) -> tuple[MemoryLoadEntry, ...]:
        return self.entries_for("stream")

    @property
    def skip_entries(self) -> tuple[MemoryLoadEntry, ...]:
        return self.entries_for("skip")

    def bytes_for(self, action: LoadAction) -> int:
        return sum(entry.nbytes for entry in self.entries_for(action))

    def as_dict(self, include_entries: bool = True) -> dict[str, Any]:
        action_counts = Counter(entry.action for entry in self.entries)
        reason_counts = Counter(entry.reason for entry in self.entries)
        action_bytes = {
            action: self.bytes_for(action)
            for action in ("resident", "stream", "skip")
        }
        reason_bytes: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            reason_bytes[entry.reason] += entry.nbytes
        result: dict[str, Any] = {
            "schema_version": 1,
            "kind": "memory_native_load_plan",
            "agent": "Main Dev",
            "model": str(self.model_dir),
            "adapter": self.adapter,
            "header_only": True,
            "payload_bytes_read": 0,
            "num_hidden_layers": self.num_hidden_layers,
            "tensor_counts": dict(sorted(action_counts.items())),
            "tensor_bytes": action_bytes,
            "reason_counts": dict(sorted(reason_counts.items())),
            "reason_bytes": dict(sorted(reason_bytes.items())),
        }
        if include_entries:
            result["entries"] = [entry.as_dict() for entry in self.entries]
        return result


def build_memory_load_plan(model_dir: str | Path) -> MemoryLoadPlan:
    root = Path(model_dir).expanduser().resolve()
    config = load_config(root)
    if config.model_type != "qwen3_5_moe":
        raise ValueError(
            f"memory-native adapter does not support model_type={config.model_type!r}"
        )
    index = ShardIndex.from_dir(root)
    num_layers = int(config.text_config.get("num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        raise ValueError("Qwen3.6 text config does not define num_hidden_layers")

    entries = tuple(
        _classify_qwen36_entry(tensor)
        for tensor in sorted(index, key=lambda item: item.name)
    )
    plan = MemoryLoadPlan(
        model_dir=root,
        adapter="qwen3_5_moe_text",
        entries=entries,
        num_hidden_layers=num_layers,
    )
    _validate_qwen36_plan(plan)
    return plan


def _classify_qwen36_entry(tensor: TensorSpan) -> MemoryLoadEntry:
    name = tensor.name
    target_name: str | None
    action: LoadAction
    reason: str
    if name.startswith("mtp."):
        target_name = None
        action = "skip"
        reason = "mtp"
    elif name.startswith("model.visual."):
        target_name = None
        action = "skip"
        reason = "vision"
    elif name.startswith("model.language_model."):
        target_name = "model." + name.removeprefix("model.language_model.")
        if ".mlp.experts." in name:
            action = "stream"
            reason = "routed_expert"
        else:
            action = "resident"
            reason = "text_resident"
    elif name == "lm_head.weight":
        target_name = name
        action = "resident"
        reason = "text_resident"
    else:
        raise ValueError(f"unclassified Qwen3.6 checkpoint tensor: {name}")
    return MemoryLoadEntry(
        source_name=name,
        target_name=target_name,
        action=action,
        reason=reason,
        shard=tensor.shard,
        dtype=tensor.dtype,
        shape=tensor.shape,
        nbytes=tensor.nbytes,
    )


def _validate_qwen36_plan(plan: MemoryLoadPlan) -> None:
    targets = [entry.target_name for entry in plan.entries if entry.target_name is not None]
    duplicates = sorted(name for name, count in Counter(targets).items() if count > 1)
    if duplicates:
        raise ValueError(f"memory-native key mapping has duplicate targets: {duplicates[:5]}")

    layer_parts: dict[int, set[str]] = defaultdict(set)
    prefix = "model.language_model.layers."
    for entry in plan.stream_entries:
        if not entry.source_name.startswith(prefix):
            raise ValueError(f"streamed tensor is outside language layers: {entry.source_name}")
        suffix = entry.source_name[len(prefix) :]
        layer_text, separator, rest = suffix.partition(".")
        if not separator or not layer_text.isdigit():
            raise ValueError(f"cannot parse streamed expert layer: {entry.source_name}")
        layer = int(layer_text)
        part = rest.removeprefix("mlp.experts.")
        if part not in {"gate_up_proj", "down_proj"}:
            raise ValueError(f"unsupported streamed expert part: {entry.source_name}")
        layer_parts[layer].add(part)

    expected_layers = set(range(plan.num_hidden_layers))
    if set(layer_parts) != expected_layers:
        missing = sorted(expected_layers - set(layer_parts))
        extra = sorted(set(layer_parts) - expected_layers)
        raise ValueError(f"streamed expert layers mismatch: missing={missing}, extra={extra}")
    incomplete = {
        layer: sorted({"gate_up_proj", "down_proj"} - parts)
        for layer, parts in layer_parts.items()
        if parts != {"gate_up_proj", "down_proj"}
    }
    if incomplete:
        raise ValueError(f"streamed expert layers have missing parts: {incomplete}")


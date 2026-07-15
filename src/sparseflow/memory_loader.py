from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Literal

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


@dataclass
class MetaTextModelBuild:
    model: Any
    plan: MemoryLoadPlan
    state_keys: tuple[str, ...]
    meta_parameter_names: tuple[str, ...]
    meta_buffer_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "memory_native_meta_build",
            "agent": "Main Dev",
            "model": str(self.plan.model_dir),
            "adapter": self.plan.adapter,
            "payload_bytes_read": 0,
            "state_tensors": len(self.state_keys),
            "meta_parameters": len(self.meta_parameter_names),
            "meta_buffers": len(self.meta_buffer_names),
            "routed_expert_parameters": sum(
                ".mlp.experts." in name for name in self.state_keys
            ),
            "resident_plan_tensors": len(self.plan.resident_entries),
            "resident_plan_bytes": self.plan.bytes_for("resident"),
            "streamed_plan_bytes": self.plan.bytes_for("stream"),
            "skipped_plan_bytes": self.plan.bytes_for("skip"),
        }


@dataclass
class MaterializedTextModel:
    model: Any
    plan: MemoryLoadPlan
    loaded_source_names: tuple[str, ...]
    loaded_target_names: tuple[str, ...]
    source_payload_bytes_read: int
    materialized_bytes: int
    load_seconds: float

    def as_dict(self) -> dict[str, Any]:
        remaining_meta_parameters = tuple(
            name for name, parameter in self.model.named_parameters() if parameter.is_meta
        )
        remaining_meta_buffers = tuple(
            name for name, buffer in self.model.named_buffers() if buffer.is_meta
        )
        return {
            "schema_version": 1,
            "kind": "memory_native_selective_load",
            "agent": "Main Dev",
            "model": str(self.plan.model_dir),
            "adapter": self.plan.adapter,
            "loaded_tensors": len(self.loaded_source_names),
            "source_payload_bytes_read": self.source_payload_bytes_read,
            "materialized_bytes": self.materialized_bytes,
            "expert_payload_bytes_during_init": 0,
            "streamed_expert_bytes_skipped": self.plan.bytes_for("stream"),
            "non_text_bytes_skipped": self.plan.bytes_for("skip"),
            "remaining_meta_parameters": len(remaining_meta_parameters),
            "remaining_meta_buffers": len(remaining_meta_buffers),
            "routed_expert_parameters": sum(
                ".mlp.experts." in name for name, _ in self.model.named_parameters()
            ),
            "load_seconds": self.load_seconds,
        }


def build_qwen36_meta_text_model(
    model_dir: str | Path,
    expert_factory: Callable[[int], Any] | None = None,
) -> MetaTextModelBuild:
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
    )

    plan = build_memory_load_plan(model_dir)
    config = AutoConfig.from_pretrained(
        plan.model_dir,
        local_files_only=True,
    )
    text_config = getattr(config, "text_config", None)
    if text_config is None or text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("checkpoint does not expose a Qwen3.6 text config")
    with torch.device("meta"):
        model = Qwen3_5MoeForCausalLM(text_config)
    model.eval()
    model.set_experts_implementation("eager")
    if expert_factory is None:
        expert_factory = lambda layer: _MetaExpertPlaceholder(layer)
    return prepare_qwen36_meta_text_model(model, plan, expert_factory)


def materialize_qwen36_text_model(
    build: MetaTextModelBuild,
    dtype: str = "bf16",
) -> MaterializedTextModel:
    materialized = materialize_meta_text_model(build, dtype=dtype)
    _materialize_qwen36_runtime_buffers(materialized.model)
    _audit_materialized_model(materialized)
    return materialized


def materialize_meta_text_model(
    build: MetaTextModelBuild,
    dtype: str = "bf16",
) -> MaterializedTextModel:
    import torch
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    dtype_value = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(dtype.lower())
    if dtype_value is None:
        raise ValueError("dtype must be bf16, fp16, or fp32")

    resident_by_shard: dict[Path, list[MemoryLoadEntry]] = defaultdict(list)
    for entry in build.plan.resident_entries:
        resident_by_shard[entry.shard].append(entry)

    loaded_sources: list[str] = []
    loaded_targets: list[str] = []
    source_bytes = 0
    materialized_bytes = 0
    started = time.perf_counter()
    for shard, entries in sorted(resident_by_shard.items(), key=lambda item: str(item[0])):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for entry in sorted(entries, key=lambda item: item.source_name):
                if entry.source_name not in available:
                    raise ValueError(
                        f"resident tensor is missing from planned shard: {entry.source_name}"
                    )
                value = handle.get_tensor(entry.source_name)
                if tuple(value.shape) != entry.shape:
                    raise ValueError(
                        f"checkpoint shape changed for {entry.source_name}: "
                        f"plan={entry.shape}, payload={tuple(value.shape)}"
                    )
                if value.is_floating_point():
                    owned = value.clone() if value.dtype == dtype_value else value.to(dtype_value)
                else:
                    owned = value.clone()
                assert entry.target_name is not None
                set_module_tensor_to_device(
                    build.model,
                    entry.target_name,
                    "cpu",
                    value=owned,
                    dtype=owned.dtype,
                    clear_cache=False,
                )
                loaded_sources.append(entry.source_name)
                loaded_targets.append(entry.target_name)
                source_bytes += entry.nbytes
                materialized_bytes += owned.numel() * owned.element_size()
                del value, owned

    expected_sources = {entry.source_name for entry in build.plan.resident_entries}
    expected_targets = {
        entry.target_name for entry in build.plan.resident_entries if entry.target_name
    }
    if set(loaded_sources) != expected_sources or set(loaded_targets) != expected_targets:
        raise ValueError("selective loader did not materialize the complete resident plan")
    return MaterializedTextModel(
        model=build.model,
        plan=build.plan,
        loaded_source_names=tuple(loaded_sources),
        loaded_target_names=tuple(loaded_targets),
        source_payload_bytes_read=source_bytes,
        materialized_bytes=materialized_bytes,
        load_seconds=time.perf_counter() - started,
    )


def _materialize_qwen36_runtime_buffers(model: Any) -> None:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
    )

    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(model.config, device="cpu")


def _audit_materialized_model(materialized: MaterializedTextModel) -> None:
    model = materialized.model
    remaining_meta_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    remaining_meta_buffers = [
        name for name, buffer in model.named_buffers() if buffer.is_meta
    ]
    if remaining_meta_parameters or remaining_meta_buffers:
        raise ValueError(
            "selective load left unresolved meta tensors: "
            f"parameters={remaining_meta_parameters[:8]}, buffers={remaining_meta_buffers[:8]}"
        )
    expert_parameters = [
        name for name, _ in model.named_parameters() if ".mlp.experts." in name
    ]
    if expert_parameters:
        raise ValueError(
            f"materialized model contains routed expert parameters: {expert_parameters[:5]}"
        )
    if materialized.source_payload_bytes_read != materialized.plan.bytes_for("resident"):
        raise ValueError("resident source-byte count does not match memory load plan")


def prepare_qwen36_meta_text_model(
    model: Any,
    plan: MemoryLoadPlan,
    expert_factory: Callable[[int], Any],
) -> MetaTextModelBuild:
    layers = _qwen36_text_layers(model)
    if len(layers) != plan.num_hidden_layers:
        raise ValueError(
            f"meta model layer count mismatch: model={len(layers)}, plan={plan.num_hidden_layers}"
        )
    for layer_index, decoder_layer in enumerate(layers):
        decoder_layer.mlp.experts = expert_factory(layer_index)

    state_keys = tuple(sorted(model.state_dict().keys()))
    resident_targets = tuple(
        sorted(entry.target_name for entry in plan.resident_entries if entry.target_name)
    )
    missing = sorted(set(state_keys) - set(resident_targets))
    unexpected = sorted(set(resident_targets) - set(state_keys))
    if missing or unexpected:
        raise ValueError(
            "meta model state does not match resident load plan: "
            f"missing_sources={missing[:8]}, unexpected_targets={unexpected[:8]}"
        )
    expert_parameters = [name for name in state_keys if ".mlp.experts." in name]
    if expert_parameters:
        raise ValueError(
            f"meta model still contains routed expert parameters: {expert_parameters[:5]}"
        )
    non_meta_parameters = [
        name for name, parameter in model.named_parameters() if not parameter.is_meta
    ]
    non_meta_buffers = [
        name for name, buffer in model.named_buffers() if not buffer.is_meta
    ]
    if non_meta_parameters or non_meta_buffers:
        raise ValueError(
            "meta model allocated real tensors before checkpoint loading: "
            f"parameters={non_meta_parameters[:5]}, buffers={non_meta_buffers[:5]}"
        )
    return MetaTextModelBuild(
        model=model,
        plan=plan,
        state_keys=state_keys,
        meta_parameter_names=tuple(name for name, _ in model.named_parameters()),
        meta_buffer_names=tuple(name for name, _ in model.named_buffers()),
    )


def _qwen36_text_layers(model: Any):
    try:
        return model.model.layers
    except AttributeError as exc:
        raise ValueError("model does not expose Qwen3.6 text decoder layers") from exc


class _MetaExpertPlaceholder:
    """Created lazily to avoid importing torch for header-only planning."""

    def __new__(cls, layer: int):
        from torch import nn

        class Placeholder(nn.Module):
            def __init__(self, layer_index: int):
                super().__init__()
                self.layer = layer_index

            def forward(self, *_args, **_kwargs):
                raise RuntimeError("meta expert placeholder cannot execute")

        return Placeholder(layer)


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

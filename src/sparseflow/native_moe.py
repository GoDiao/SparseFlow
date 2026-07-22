from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cache import ExpertLease
from .native_int8 import load_native_int8


@dataclass
class NativeExpertBatch:
    """Stable native views and leases for one routed MoE layer call."""

    expert_ids: tuple[int, ...]
    weights: list[dict[str, Any]]
    leases: list[ExpertLease] = field(default_factory=list)

    def close(self) -> None:
        for lease in reversed(self.leases):
            lease.release()
        self.leases.clear()

    def __enter__(self) -> "NativeExpertBatch":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def prepare_native_expert_batch(provider, layer: int, expert_ids: tuple[int, ...]) -> NativeExpertBatch:
    provider.prepare(layer, expert_ids)
    batch = NativeExpertBatch(expert_ids=expert_ids, weights=[])
    try:
        for expert_id in expert_ids:
            result = provider.get(layer, expert_id)
            native = result.get("gate_up_proj")
            if not isinstance(native, dict) or not native.get("native_int8"):
                raise TypeError("fused native MoE requires canonical native INT8 expert views")
            batch.weights.append(native)
            cache = getattr(provider, "cache", None)
            decoded = getattr(provider, "_decoded", {}).get((layer, expert_id))
            if cache is not None and decoded is not None and decoded[1] is result:
                batch.leases.append(cache.lease(decoded[0], references=(result, native)))
        return batch
    except Exception:
        batch.close()
        raise


def run_fused_native_moe(
    hidden_states,
    selected_experts,
    routing_weights,
    provider,
    layer: int,
    timing_callback=None,
):
    import time
    import torch

    load_native_int8()
    started = time.perf_counter() if timing_callback is not None else None
    expert_ids = tuple(int(value) for value in selected_experts.unique(sorted=True).tolist())
    if started is not None:
        timing_callback("dispatch", (time.perf_counter() - started) * 1000.0)
    started = time.perf_counter() if timing_callback is not None else None
    with prepare_native_expert_batch(provider, layer, expert_ids) as batch:
        if started is not None:
            timing_callback("provider_get", (time.perf_counter() - started) * 1000.0)
        roots = batch.weights
        expert_tensor = torch.tensor(expert_ids, dtype=torch.long, device="cpu")
        kernel_started = time.perf_counter() if timing_callback is not None else None
        result = torch.ops.sparseflow_native.fused_moe(
            hidden_states.contiguous(),
            selected_experts.contiguous(),
            routing_weights.contiguous(),
            expert_tensor,
            [item["gate_up_proj"]["weight"] for item in roots],
            [item["gate_up_proj"]["scales"] for item in roots],
            [item["gate_up_proj"]["row_sums"] for item in roots],
            [item["down_proj"]["weight"] for item in roots],
            [item["down_proj"]["scales"] for item in roots],
            [item["down_proj"]["row_sums"] for item in roots],
        )
        if kernel_started is not None:
            timing_callback("expert_kernel", (time.perf_counter() - kernel_started) * 1000.0)
        return result


# [Main Dev]


@dataclass
class GroupedMoEWorkspace:
    """Reusable CPU buffers for one shape family of grouped MoE calls."""

    max_rows: int
    max_assignments: int
    hidden: int
    gate_rows: int
    max_experts: int
    plan_counts: Any
    plan_offsets: Any
    plan_assignments: Any
    plan_rows: Any
    plan_slots: Any
    plan_token_order: Any
    hidden_quantized: Any
    hidden_scales: Any
    hidden_zero_points: Any
    projected: Any
    activated: Any
    activated_quantized: Any
    activated_scales: Any
    activated_zero_points: Any
    contributions: Any
    output: Any

    @classmethod
    def create(
        cls,
        max_rows: int,
        hidden: int,
        gate_rows: int,
        max_experts: int,
        torch_module: Any,
    ) -> "GroupedMoEWorkspace":
        if max_rows < 1 or hidden < 1 or gate_rows < 2 or gate_rows % 2:
            raise ValueError("invalid grouped workspace shape")
        max_assignments = max_rows * 8
        intermediate = gate_rows // 2
        return cls(
            max_rows=max_rows,
            max_assignments=max_assignments,
            hidden=hidden,
            gate_rows=gate_rows,
            max_experts=max_experts,
            plan_counts=torch_module.empty(max_experts, dtype=torch_module.long),
            plan_offsets=torch_module.empty(max_experts + 1, dtype=torch_module.long),
            plan_assignments=torch_module.empty(max_assignments, dtype=torch_module.long),
            plan_rows=torch_module.empty(max_assignments, dtype=torch_module.long),
            plan_slots=torch_module.empty(max_assignments, dtype=torch_module.long),
            plan_token_order=torch_module.empty(max_assignments, dtype=torch_module.long),
            hidden_quantized=torch_module.empty((max_rows, hidden), dtype=torch_module.uint8),
            hidden_scales=torch_module.empty(max_rows, dtype=torch_module.float32),
            hidden_zero_points=torch_module.empty(max_rows, dtype=torch_module.int32),
            projected=torch_module.empty((max_assignments, gate_rows), dtype=torch_module.float32),
            activated=torch_module.empty((max_assignments, intermediate), dtype=torch_module.float32),
            activated_quantized=torch_module.empty((max_assignments, intermediate), dtype=torch_module.uint8),
            activated_scales=torch_module.empty(max_assignments, dtype=torch_module.float32),
            activated_zero_points=torch_module.empty(max_assignments, dtype=torch_module.int32),
            contributions=torch_module.empty((max_assignments, hidden), dtype=torch_module.float32),
            output=torch_module.empty((max_rows, hidden), dtype=torch_module.bfloat16),
        )

    def compatible(self, rows: int, assignments: int, hidden: int, gate_rows: int) -> bool:
        return (
            rows <= self.max_rows
            and assignments <= self.max_assignments
            and hidden == self.hidden
            and gate_rows == self.gate_rows
        )

    def tensors(self) -> tuple[Any, ...]:
        return (
            self.plan_counts,
            self.plan_offsets,
            self.plan_assignments,
            self.plan_rows,
            self.plan_slots,
            self.plan_token_order,
            self.hidden_quantized,
            self.hidden_scales,
            self.hidden_zero_points,
            self.projected,
            self.activated,
            self.activated_quantized,
            self.activated_scales,
            self.activated_zero_points,
            self.contributions,
            self.output,
        )

    def allocated_bytes(self) -> int:
        return sum(int(t.numel() * t.element_size()) for t in self.tensors())


def _grouped_expert_ids(selected_experts, torch_module) -> tuple[int, ...]:
    return tuple(int(value) for value in selected_experts.unique(sorted=True).tolist())


def run_grouped_native_moe(
    hidden_states,
    selected_experts,
    routing_weights,
    provider,
    layer: int,
    workspace: GroupedMoEWorkspace | None = None,
    timing_callback=None,
):
    """Run one layer through the Stage 7.8 per-expert grouped operator."""

    import time
    import torch

    load_native_int8()
    hidden_states = hidden_states.contiguous()
    selected_experts = selected_experts.contiguous()
    routing_weights = routing_weights.contiguous()
    rows = int(hidden_states.shape[0])
    assignments = rows * int(selected_experts.shape[1])
    expert_ids = _grouped_expert_ids(selected_experts, torch)
    started = time.perf_counter() if timing_callback is not None else None
    if timing_callback is not None:
        timing_callback("group_dispatch", 0.0)
    provider.prepare(layer, expert_ids)
    batch = NativeExpertBatch(expert_ids=expert_ids, weights=[])
    try:
        for expert_id in expert_ids:
            result = provider.get(layer, expert_id)
            native = result.get("gate_up_proj")
            if not isinstance(native, dict) or not native.get("native_int8"):
                raise TypeError("grouped native MoE requires canonical native INT8 expert views")
            batch.weights.append(native)
            cache = getattr(provider, "cache", None)
            decoded = getattr(provider, "_decoded", {}).get((layer, expert_id))
            if cache is not None and decoded is not None and decoded[1] is result:
                batch.leases.append(cache.lease(decoded[0], references=(result, native)))
        if timing_callback is not None and started is not None:
            timing_callback("group_provider_get", (time.perf_counter() - started) * 1000.0)
        root = batch.weights[0]
        gate_rows = int(root["gate_up_proj"]["weight"].shape[0])
        hidden = int(hidden_states.shape[1])
        if workspace is None or not workspace.compatible(rows, assignments, hidden, gate_rows):
            workspace = GroupedMoEWorkspace.create(rows, hidden, gate_rows, max(256, len(expert_ids)), torch)
        expert_tensor = torch.tensor(expert_ids, dtype=torch.long)
        kernel_started = time.perf_counter() if timing_callback is not None else None
        result = torch.ops.sparseflow_native.grouped_moe(
            hidden_states,
            selected_experts,
            routing_weights,
            expert_tensor,
            [item["gate_up_proj"]["weight"] for item in batch.weights],
            [item["gate_up_proj"]["scales"] for item in batch.weights],
            [item["gate_up_proj"]["row_sums"] for item in batch.weights],
            [item["down_proj"]["weight"] for item in batch.weights],
            [item["down_proj"]["scales"] for item in batch.weights],
            [item["down_proj"]["row_sums"] for item in batch.weights],
            *workspace.tensors(),
        )
        if timing_callback is not None and kernel_started is not None:
            timing_callback("grouped_expert_kernel", (time.perf_counter() - kernel_started) * 1000.0)
        return result, workspace
    finally:
        batch.close()


# [Main Dev]

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

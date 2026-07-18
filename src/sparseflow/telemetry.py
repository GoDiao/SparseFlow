from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


_COUNTER_KEYS = (
    "reader_calls",
    "reader_bytes",
    "requests",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "loaded_bytes",
    "hit_bytes",
    "miss_bytes",
    "admission_rejections",
    "demand_read_calls",
    "demand_read_bytes",
    "demand_read_ms_total",
    "demand_requests",
    "demand_reuse_hits",
    "demand_prefetch_served",
    "demand_misses",
    "prefetch_submitted",
    "prefetch_hits",
    "prefetch_late",
    "prefetch_hit_bytes",
    "prefetch_wasted_ready_bytes",
    "timing_cache_lookup_ms_total",
    "timing_victim_selection_ms_total",
    "timing_allocation_reuse_ms_total",
    "timing_policy_maintenance_ms_total",
    "timing_tensor_decode_view_ms_total",
    "timing_pread_ms_total",
)

_TIMING_CATEGORIES = (
    "model_forward",
    "decoder_layer",
    "attention",
    "linear_attention",
    "input_layernorm",
    "post_attention_layernorm",
    "moe_block",
    "router",
    "shared_expert",
    "shared_expert_gate",
    "routed_experts",
    "dispatch",
    "prepare",
    "provider_get",
    "expert_kernel",
    "routing_accumulation",
    "final_norm",
    "lm_head",
    "argmax",
    "token_loop_overhead",
    "native_row_sums",
    "native_activation_quantization",
    "native_gemv",
    "native_dynamic_linear",
)


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | float]:
    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in _COUNTER_KEYS
    }


def _zero_provider() -> dict[str, int | float]:
    return {key: 0 for key in _COUNTER_KEYS}


@dataclass
class RuntimeTelemetry:
    """Runtime counters with an O(1) summary path and opt-in layer records."""

    level: str = "summary"
    records: list[dict[str, Any]] = field(default_factory=list)
    _forwards: list[dict[str, Any]] = field(default_factory=list)
    _current: dict[str, Any] | None = None
    _observer_seconds: float = 0.0
    _pending_timings_ms: dict[str, float] = field(default_factory=dict)
    _provider_total: dict[str, int | float] = field(default_factory=_zero_provider)

    def __post_init__(self) -> None:
        if self.level not in {"none", "summary", "profile", "layer"}:
            raise ValueError(f"unknown telemetry level: {self.level}")

    def reset(self) -> None:
        self.records.clear()
        self._forwards.clear()
        self._current = None
        self._observer_seconds = 0.0
        self._pending_timings_ms.clear()
        self._provider_total = _zero_provider()

    def begin_forward(
        self,
        forward: int,
        phase: str,
        token_position: int | None,
    ) -> None:
        if self.level == "none":
            return
        self._flush_current()
        self._current = {
            "forward": forward,
            "phase": phase,
            "token_position": token_position,
            "layers": 0,
            "route_requests": 0,
            "unique_experts_sum": 0 if self.level == "layer" else None,
            "layer_total_ms": 0.0,
            "provider": _zero_provider(),
            "timings_ms": {key: 0.0 for key in _TIMING_CATEGORIES},
            "cached_bytes_after": 0,
            "cache_entries_after": 0,
        }
        self._pending_timings_ms = {key: 0.0 for key in _TIMING_CATEGORIES}

    def record_summary_layer(self, selected_experts) -> None:
        """Update fixed-cost route counters without provider snapshots."""

        if self.level != "summary":
            return
        if self._current is None:
            self.begin_forward(-1, "unknown", None)
        assert self._current is not None
        self._current["layers"] += 1
        self._current["route_requests"] += int(selected_experts.numel())

    def set_provider_total(
        self,
        provider_before: dict[str, Any] | None,
        provider_after: dict[str, Any] | None,
    ) -> None:
        if (
            self.level == "summary"
            and provider_before is not None
            and provider_after is not None
        ):
            self._provider_total = counter_delta(provider_before, provider_after)

    def add_timing(self, category: str, elapsed_ms: float) -> None:
        if self.level not in {"profile", "layer"}:
            return
        if category not in self._pending_timings_ms:
            self._pending_timings_ms[category] = 0.0
            assert self._current is not None
            self._current["timings_ms"][category] = 0.0
        self._pending_timings_ms[category] += elapsed_ms
        assert self._current is not None
        self._current["timings_ms"][category] += elapsed_ms

    def record_layer(
        self,
        layer: int,
        selected_experts,
        before: dict[str, Any],
        after: dict[str, Any],
        elapsed_ms: float,
    ) -> None:
        if self.level == "none":
            return
        observer_started = time.perf_counter()
        if self._current is None:
            self.begin_forward(-1, "unknown", None)
        assert self._current is not None
        provider = counter_delta(before, after)
        route_requests = int(selected_experts.numel())
        current = self._current
        current["layers"] += 1
        current["route_requests"] += route_requests
        current["layer_total_ms"] += elapsed_ms
        for key, value in provider.items():
            current["provider"][key] += value
        current["cached_bytes_after"] = int(after.get("cached_bytes", 0))
        current["cache_entries_after"] = int(after.get("cached_experts", 0))

        if self.level == "layer":
            unique_experts = int(selected_experts.unique().numel())
            current["unique_experts_sum"] += unique_experts
            self.records.append(
                {
                    "forward": current["forward"],
                    "phase": current["phase"],
                    "token_position": current["token_position"],
                    "layer": layer,
                    "rows": int(selected_experts.shape[0]),
                    "route_requests": route_requests,
                    "unique_experts": unique_experts,
                    "layer_total_ms": elapsed_ms,
                    "provider": provider,
                    "timings_ms": dict(self._pending_timings_ms),
                    "cached_bytes_after": current["cached_bytes_after"],
                    "cache_entries_after": current["cache_entries_after"],
                }
            )
            self._pending_timings_ms = {
                key: 0.0 for key in self._pending_timings_ms
            }
        self._observer_seconds += time.perf_counter() - observer_started

    def as_dict(self) -> dict[str, Any]:
        if self.level == "none":
            return {
                "level": "none",
                "observer_seconds": 0.0,
                "summary": {},
                "forwards": [],
                "records": [],
            }
        self._flush_current()
        summary = {
            "forwards": len(self._forwards),
            "layers": sum(item["layers"] for item in self._forwards),
            "prefill_forwards": sum(item["phase"] == "prefill" for item in self._forwards),
            "decode_forwards": sum(item["phase"] == "decode" for item in self._forwards),
            "route_requests": sum(item["route_requests"] for item in self._forwards),
            "unique_experts_sum": (
                sum(item["unique_experts_sum"] for item in self._forwards)
                if self.level == "layer"
                else None
            ),
            "layer_total_ms": sum(item["layer_total_ms"] for item in self._forwards),
            "provider": (
                dict(self._provider_total)
                if self.level == "summary"
                else _sum_provider(self._forwards)
            ),
            "timings_ms": _sum_timings(self._forwards),
        }
        return {
            "level": self.level,
            "observer_seconds": self._observer_seconds,
            "summary": summary,
            "forwards": list(self._forwards),
            "records": list(self.records) if self.level == "layer" else [],
        }

    def _flush_current(self) -> None:
        if self._current is not None:
            self._forwards.append(self._current)
            self._current = None


def _sum_provider(forwards: list[dict[str, Any]]) -> dict[str, int | float]:
    result = _zero_provider()
    for item in forwards:
        for key, value in item["provider"].items():
            result[key] += value
    return result


def _sum_timings(forwards: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in forwards:
        for key, value in item["timings_ms"].items():
            result[key] = result.get(key, 0.0) + value
    return result


# [Main Dev]

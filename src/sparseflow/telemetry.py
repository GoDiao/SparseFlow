from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
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
)


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key in _COUNTER_KEYS:
        left = before.get(key, 0)
        right = after.get(key, 0)
        result[key] = right - left
    return result


@dataclass
class RuntimeTelemetry:
    level: str = "summary"
    records: list[dict[str, Any]] = field(default_factory=list)
    _forward: int = -1
    _phase: str = "unknown"
    _token_position: int | None = None

    def __post_init__(self) -> None:
        if self.level not in {"none", "summary", "layer"}:
            raise ValueError(f"unknown telemetry level: {self.level}")

    def begin_forward(
        self,
        forward: int,
        phase: str,
        token_position: int | None,
    ) -> None:
        self._forward = forward
        self._phase = phase
        self._token_position = token_position

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
        unique = selected_experts.detach().to(device="cpu").unique()
        self.records.append(
            {
                "forward": self._forward,
                "phase": self._phase,
                "token_position": self._token_position,
                "layer": layer,
                "rows": int(selected_experts.shape[0]),
                "route_requests": int(selected_experts.numel()),
                "unique_experts": int(unique.numel()),
                "layer_total_ms": elapsed_ms,
                "provider": counter_delta(before, after),
                "cached_bytes_after": int(after.get("cached_bytes", 0)),
                "cache_entries_after": int(after.get("cached_experts", 0)),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        if self.level == "none":
            return {"level": "none", "summary": {}, "forwards": [], "records": []}
        by_forward: dict[tuple[int, str, int | None], list[dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            key = (record["forward"], record["phase"], record["token_position"])
            by_forward[key].append(record)
        forwards = []
        for (forward, phase, token_position), records in sorted(by_forward.items()):
            provider = _sum_provider(records)
            forwards.append(
                {
                    "forward": forward,
                    "phase": phase,
                    "token_position": token_position,
                    "layers": len(records),
                    "route_requests": sum(item["route_requests"] for item in records),
                    "unique_experts_sum": sum(item["unique_experts"] for item in records),
                    "layer_total_ms": sum(item["layer_total_ms"] for item in records),
                    "provider": provider,
                    "cached_bytes_after": records[-1]["cached_bytes_after"],
                    "cache_entries_after": records[-1]["cache_entries_after"],
                }
            )
        summary = {
            "forwards": len(forwards),
            "layers": len(self.records),
            "prefill_forwards": sum(item["phase"] == "prefill" for item in forwards),
            "decode_forwards": sum(item["phase"] == "decode" for item in forwards),
            "route_requests": sum(item["route_requests"] for item in self.records),
            "unique_experts_sum": sum(item["unique_experts"] for item in self.records),
            "layer_total_ms": sum(item["layer_total_ms"] for item in self.records),
            "provider": _sum_provider(self.records),
        }
        return {
            "level": self.level,
            "summary": summary,
            "forwards": forwards,
            "records": self.records if self.level == "layer" else [],
        }


def _sum_provider(records: list[dict[str, Any]]) -> dict[str, int | float]:
    result: dict[str, int | float] = {key: 0 for key in _COUNTER_KEYS}
    for record in records:
        for key, value in record["provider"].items():
            result[key] += value
    return result

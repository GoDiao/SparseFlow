from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TraceRequest = tuple[int, int]


@dataclass(frozen=True)
class TraceGroup:
    """Expert selections for one token row in one model forward."""

    forward: int
    phase: str
    row: int
    token_position: int | None
    token_id: int | None
    requests: tuple[TraceRequest, ...]


@dataclass(frozen=True)
class ReplayGroup:
    forward: int
    phase: str
    raw_requests: int
    requests: tuple[TraceRequest, ...]


@dataclass(frozen=True)
class RouteTrace:
    """Structured route trace preserving forward, row, and phase boundaries."""

    groups: tuple[TraceGroup, ...]
    source_sha256: str | None = None
    schema_version: int = 2

    @property
    def flat_requests(self) -> list[TraceRequest]:
        return [request for group in self.groups for request in group.requests]

    @property
    def raw_requests(self) -> int:
        return sum(len(group.requests) for group in self.groups)

    @property
    def phases(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group in self.groups:
            result[group.phase] = result.get(group.phase, 0) + len(group.requests)
        return result

    @property
    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.flat_requests})

    def batch_union_requests(self) -> list[TraceRequest]:
        """Deduplicate experts within each forward and layer.

        This models loading an expert once for a prefill/batch group even when
        several token rows selected the same expert.
        """

        return [request for group in self.replay_groups(batch_union=True) for request in group.requests]

    def replay_groups(self, batch_union: bool = False) -> tuple[ReplayGroup, ...]:
        if not batch_union:
            return tuple(
                ReplayGroup(
                    forward=group.forward,
                    phase=group.phase,
                    raw_requests=len(group.requests),
                    requests=group.requests,
                )
                for group in self.groups
            )

        by_forward: dict[int, list[TraceGroup]] = {}
        for group in self.groups:
            by_forward.setdefault(group.forward, []).append(group)

        result: list[ReplayGroup] = []
        for forward in sorted(by_forward):
            groups = by_forward[forward]
            seen_by_layer: dict[int, set[int]] = {}
            requests: list[TraceRequest] = []
            for group in groups:
                for layer, expert in group.requests:
                    seen = seen_by_layer.setdefault(layer, set())
                    if expert in seen:
                        continue
                    seen.add(expert)
                    requests.append((layer, expert))
            result.append(
                ReplayGroup(
                    forward=forward,
                    phase=groups[0].phase,
                    raw_requests=sum(len(group.requests) for group in groups),
                    requests=tuple(requests),
                )
            )
        return tuple(result)

    @classmethod
    def from_flat(cls, requests: Iterable[TraceRequest]) -> "RouteTrace":
        groups = tuple(
            TraceGroup(
                forward=index,
                phase="synthetic",
                row=0,
                token_position=index,
                token_id=None,
                requests=(request,),
            )
            for index, request in enumerate(requests)
        )
        return cls(groups=groups)


def load_route_trace(path: str | Path) -> RouteTrace:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("forwards"), list):
        trace = _parse_grouped_trace(raw)
    elif isinstance(raw, dict) and isinstance(raw.get("requests"), list):
        trace = _parse_flat_trace(raw["requests"], raw)
    elif isinstance(raw, list):
        trace = _parse_flat_trace(raw, {})
    else:
        raise ValueError(f"trace must contain forwards or requests: {source}")
    if not trace.groups:
        raise ValueError(f"trace is empty: {source}")
    return trace


def _parse_grouped_trace(raw: dict[str, Any]) -> RouteTrace:
    groups: list[TraceGroup] = []
    for forward_item in raw["forwards"]:
        if not isinstance(forward_item, dict):
            raise ValueError("each forwards item must be an object")
        forward = _as_int(forward_item.get("forward"), "forward")
        phase = str(forward_item.get("phase", "unknown"))
        rows = forward_item.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"forward {forward} must contain a rows list")
        for row_item in rows:
            if not isinstance(row_item, dict):
                raise ValueError(f"forward {forward} contains an invalid row")
            row = _as_int(row_item.get("row"), "row")
            token_position = _optional_int(row_item.get("token_position"))
            token_id = _optional_int(row_item.get("token_id"))
            layers = row_item.get("layers")
            if not isinstance(layers, list):
                raise ValueError(f"forward {forward}, row {row} must contain layers")
            requests: list[TraceRequest] = []
            for layer_item in layers:
                if not isinstance(layer_item, dict):
                    raise ValueError("each layer item must be an object")
                layer = _as_int(layer_item.get("layer"), "layer")
                experts = layer_item.get("expert_ids", layer_item.get("experts"))
                if not isinstance(experts, list) or not all(isinstance(e, int) for e in experts):
                    raise ValueError(f"layer {layer} must contain integer expert_ids")
                requests.extend((layer, expert) for expert in experts)
            groups.append(
                TraceGroup(
                    forward=forward,
                    phase=phase,
                    row=row,
                    token_position=token_position,
                    token_id=token_id,
                    requests=tuple(requests),
                )
            )
    return RouteTrace(
        groups=tuple(groups),
        source_sha256=raw.get("trace_sha256"),
        schema_version=int(raw.get("schema_version", 2)),
    )


def _parse_flat_trace(items: list[Any], raw: dict[str, Any]) -> RouteTrace:
    grouped: dict[tuple[int, int], list[TraceRequest]] = {}
    metadata: dict[tuple[int, int], tuple[str, int | None, int | None]] = {}
    for index, item in enumerate(items):
        if isinstance(item, dict):
            layer, expert = item.get("layer"), item.get("expert")
            forward = int(item.get("forward", index))
            row = int(item.get("row", 0))
            phase = str(item.get("phase", "unknown"))
            token_position = _optional_int(item.get("token_position"))
            token_id = _optional_int(item.get("token_id"))
        elif isinstance(item, list) and len(item) == 2:
            layer, expert = item
            forward, row, phase = index, 0, "synthetic"
            token_position, token_id = index, None
        else:
            raise ValueError(f"invalid trace request at index {index}: {item!r}")
        if not isinstance(layer, int) or not isinstance(expert, int):
            raise ValueError(f"trace request must contain integer layer/expert at index {index}")
        key = (forward, row)
        grouped.setdefault(key, []).append((layer, expert))
        metadata.setdefault(key, (phase, token_position, token_id))

    groups = tuple(
        TraceGroup(
            forward=forward,
            phase=metadata[(forward, row)][0],
            row=row,
            token_position=metadata[(forward, row)][1],
            token_id=metadata[(forward, row)][2],
            requests=tuple(grouped[(forward, row)]),
        )
        for forward, row in sorted(grouped)
    )
    return RouteTrace(
        groups=groups,
        source_sha256=raw.get("trace_sha256"),
        schema_version=int(raw.get("schema_version", 1)),
    )


def _as_int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def trace_sha256(trace: RouteTrace) -> str:
    canonical = [
        {
            "forward": group.forward,
            "phase": group.phase,
            "row": group.row,
            "token_position": group.token_position,
            "token_id": group.token_id,
            "requests": [[layer, expert] for layer, expert in group.requests],
        }
        for group in trace.groups
    ]
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

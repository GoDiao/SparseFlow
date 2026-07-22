"""Session-aware route traces for Stage 7.7 multi-request experiments.

The v2 route format remains the source of truth for one request.  This module
adds only the session and scheduler dimensions needed to combine independent
real traces; it never invents router selections.

[Main Dev]
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .trace import RouteTrace, TraceGroup, load_route_trace


@dataclass(frozen=True)
class SessionRoute:
    """One independent request captured from a real model forward."""

    session_id: str
    prompt_id: str
    prompt: str
    groups: tuple[TraceGroup, ...]
    source_sha256: str
    input_tokens: int | None = None
    generated_ids: tuple[int, ...] = ()

    def group(self, forward: int, row: int = 0) -> TraceGroup:
        for item in self.groups:
            if item.forward == forward and item.row == row:
                return item
        raise KeyError(
            f"session={self.session_id} has no forward={forward}, row={row}"
        )

    @property
    def decode_forwards(self) -> tuple[int, ...]:
        return tuple(sorted({item.forward for item in self.groups if item.phase == "decode"}))

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        session_id: str | None = None,
        prompt_id: str | None = None,
    ) -> "SessionRoute":
        trace = load_route_trace_from_raw(raw)
        workload = raw.get("workload") if isinstance(raw.get("workload"), Mapping) else {}
        resolved_prompt_id = str(
            prompt_id or workload.get("prompt_id") or session_id or "session"
        )
        resolved_session_id = str(session_id or resolved_prompt_id)
        prompt = str(workload.get("prompt", ""))
        source = str(raw.get("trace_sha256") or route_trace_digest(trace))
        generated = workload.get("generated_ids", [])
        generated_ids = tuple(int(value) for value in generated) if isinstance(generated, list) else ()
        input_tokens = workload.get("input_tokens")
        return cls(
            session_id=resolved_session_id,
            prompt_id=resolved_prompt_id,
            prompt=prompt,
            groups=trace.groups,
            source_sha256=source,
            input_tokens=int(input_tokens) if isinstance(input_tokens, int) else None,
            generated_ids=generated_ids,
        )

    @classmethod
    def from_path(cls, path: str | Path, session_id: str | None = None) -> "SessionRoute":
        source = Path(path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"route trace must be a JSON object: {source}")
        return cls.from_raw(raw, session_id=session_id)


@dataclass(frozen=True)
class ScheduleStep:
    scheduler_step: int
    sessions: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scheduler_step": self.scheduler_step,
            "sessions": [
                {"session_id": session_id, "forward": forward}
                for session_id, forward in self.sessions
            ],
        }


@dataclass(frozen=True)
class SessionSchedule:
    schedule_id: str
    kind: str
    steps: tuple[ScheduleStep, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "kind": self.kind,
            "steps": [step.as_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MultiRequestTrace:
    sessions: tuple[SessionRoute, ...]
    schedules: tuple[SessionSchedule, ...]
    schema_version: int = 3
    model: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        ids = [session.session_id for session in self.sessions]
        if len(ids) != len(set(ids)):
            raise ValueError("session_id values must be unique")
        prompt_ids = [session.prompt_id for session in self.sessions]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("prompt_id values must be unique")
        known = set(ids)
        for schedule in self.schedules:
            for step in schedule.steps:
                for session_id, forward in step.sessions:
                    if session_id not in known:
                        raise ValueError(f"schedule references unknown session: {session_id}")
                    if forward < 1:
                        raise ValueError("multi-request schedules must reference decode forwards")

    def session_map(self) -> dict[str, SessionRoute]:
        return {session.session_id: session for session in self.sessions}

    def schedule(self, schedule_id: str) -> SessionSchedule:
        for schedule in self.schedules:
            if schedule.schedule_id == schedule_id:
                return schedule
        raise KeyError(f"unknown schedule: {schedule_id}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "sparseflow_multi_request_route_trace",
            "model": dict(self.model or {}),
            "sessions": [
                {
                    "session_id": session.session_id,
                    "prompt_id": session.prompt_id,
                    "prompt": session.prompt,
                    "source_sha256": session.source_sha256,
                    "input_tokens": session.input_tokens,
                    "generated_ids": list(session.generated_ids),
                    "forwards": _serialize_groups(session.groups),
                }
                for session in self.sessions
            ],
            "schedules": [schedule.as_dict() for schedule in self.schedules],
            "trace_sha256": multi_request_digest(self),
        }

    def write(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_route_trace_from_raw(raw: Mapping[str, Any]) -> RouteTrace:
    """Load a v1/v2 trace object without creating a temporary file."""

    if isinstance(raw.get("forwards"), list):
        groups: list[TraceGroup] = []
        for forward_item in raw["forwards"]:
            if not isinstance(forward_item, Mapping):
                raise ValueError("forwards entries must be objects")
            forward = int(forward_item["forward"])
            phase = str(forward_item.get("phase", "unknown"))
            rows = forward_item.get("rows")
            if not isinstance(rows, list):
                raise ValueError(f"forward {forward} must contain rows")
            for row_item in rows:
                if not isinstance(row_item, Mapping):
                    raise ValueError("row entries must be objects")
                requests: list[tuple[int, int]] = []
                for layer_item in row_item.get("layers", []):
                    if not isinstance(layer_item, Mapping):
                        raise ValueError("layer entries must be objects")
                    layer = int(layer_item["layer"])
                    experts = layer_item.get("expert_ids", layer_item.get("experts"))
                    if not isinstance(experts, list):
                        raise ValueError("layer expert_ids must be a list")
                    requests.extend((layer, int(expert)) for expert in experts)
                groups.append(
                    TraceGroup(
                        forward=forward,
                        phase=phase,
                        row=int(row_item.get("row", 0)),
                        token_position=(
                            int(row_item["token_position"])
                            if isinstance(row_item.get("token_position"), int)
                            else None
                        ),
                        token_id=(
                            int(row_item["token_id"])
                            if isinstance(row_item.get("token_id"), int)
                            else None
                        ),
                        requests=tuple(requests),
                    )
                )
        return RouteTrace(
            groups=tuple(groups),
            source_sha256=(str(raw["trace_sha256"]) if raw.get("trace_sha256") else None),
            schema_version=int(raw.get("schema_version", 2)),
        )
    raise ValueError("route trace object must contain forwards")


def make_schedule(
    sessions: Iterable[SessionRoute],
    schedule_id: str,
    kind: str = "synchronous",
    max_steps: int | None = None,
    arrival_offsets: Mapping[str, int] | None = None,
) -> SessionSchedule:
    session_list = tuple(sessions)
    if not session_list:
        raise ValueError("at least one session is required")
    offsets = dict(arrival_offsets or {})
    if any(int(value) < 0 for value in offsets.values()):
        raise ValueError("arrival offsets must be non-negative")
    available = {
        session.session_id: len(session.decode_forwards)
        for session in session_list
    }
    step_count = max_steps or max(available.values())
    steps: list[ScheduleStep] = []
    for scheduler_step in range(step_count):
        active: list[tuple[str, int]] = []
        for session in session_list:
            offset = int(offsets.get(session.session_id, 0))
            relative = scheduler_step - offset
            if relative < 0 or relative >= available[session.session_id]:
                continue
            active.append((session.session_id, session.decode_forwards[relative]))
        if active:
            steps.append(ScheduleStep(scheduler_step=scheduler_step, sessions=tuple(active)))
    if not steps:
        raise ValueError("schedule contains no active session steps")
    return SessionSchedule(schedule_id=schedule_id, kind=kind, steps=tuple(steps))


def analyze_schedule(
    trace: MultiRequestTrace,
    schedule_id: str,
) -> dict[str, Any]:
    """Calculate union statistics without reading model weights."""

    schedule = trace.schedule(schedule_id)
    sessions = trace.session_map()
    step_results: list[dict[str, Any]] = []
    total_assignments = 0
    total_unique = 0
    per_layer: dict[int, dict[str, int]] = {}

    for step in schedule.steps:
        layer_results: list[dict[str, Any]] = []
        for layer in range(40):
            per_session: dict[str, set[int]] = {}
            assignments = 0
            for session_id, forward in step.sessions:
                group = sessions[session_id].group(forward)
                selected = [expert for current_layer, expert in group.requests if current_layer == layer]
                if selected:
                    per_session[session_id] = set(selected)
                    assignments += len(selected)
            union = set().union(*per_session.values()) if per_session else set()
            unique = len(union)
            overlaps = assignments - unique
            multiplicity = Counter(
                expert
                for selected in per_session.values()
                for expert in selected
            )
            pairwise: list[float] = []
            session_sets = list(per_session.values())
            for left_index, left in enumerate(session_sets):
                for right in session_sets[left_index + 1 :]:
                    denominator = len(left | right)
                    pairwise.append(len(left & right) / denominator if denominator else 1.0)
            layer_results.append(
                {
                    "layer": layer,
                    "sessions": len(per_session),
                    "assignments": assignments,
                    "unique_experts": unique,
                    "overlap_assignments": overlaps,
                    "union_ratio": assignments / unique if unique else 0.0,
                    "mean_pairwise_jaccard": sum(pairwise) / len(pairwise) if pairwise else 0.0,
                    "multiplicity": dict(sorted(Counter(multiplicity.values()).items())),
                }
            )
            total_assignments += assignments
            total_unique += unique
            aggregate = per_layer.setdefault(layer, {"assignments": 0, "unique_experts": 0})
            aggregate["assignments"] += assignments
            aggregate["unique_experts"] += unique
        step_results.append(
            {
                "scheduler_step": step.scheduler_step,
                "active_sessions": [session_id for session_id, _ in step.sessions],
                "layers": layer_results,
                "assignments": sum(item["assignments"] for item in layer_results),
                "unique_experts": sum(item["unique_experts"] for item in layer_results),
            }
        )
    return {
        "schedule_id": schedule_id,
        "schedule_kind": schedule.kind,
        "sessions": max((len(step.sessions) for step in schedule.steps), default=0),
        "trace_sessions": len(trace.sessions),
        "steps": len(schedule.steps),
        "assignments": total_assignments,
        "unique_experts": total_unique,
        "overlap_assignments": total_assignments - total_unique,
        "union_ratio": total_assignments / total_unique if total_unique else 0.0,
        "per_layer": [
            {"layer": layer, **values, "union_ratio": values["assignments"] / values["unique_experts"] if values["unique_experts"] else 0.0}
            for layer, values in sorted(per_layer.items())
        ],
        "steps_detail": step_results,
    }


def route_trace_digest(trace: RouteTrace) -> str:
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


def multi_request_digest(trace: MultiRequestTrace) -> str:
    canonical = {
        "sessions": [
            {
                "session_id": session.session_id,
                "prompt_id": session.prompt_id,
                "source_sha256": session.source_sha256,
                "groups": [
                    {
                        "forward": group.forward,
                        "phase": group.phase,
                        "row": group.row,
                        "requests": [[layer, expert] for layer, expert in group.requests],
                    }
                    for group in session.groups
                ],
            }
            for session in trace.sessions
        ],
        "schedules": [schedule.as_dict() for schedule in trace.schedules],
    }
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _serialize_groups(groups: Iterable[TraceGroup]) -> list[dict[str, Any]]:
    forwards: dict[int, dict[str, Any]] = {}
    for group in groups:
        forward = forwards.setdefault(
            group.forward,
            {"forward": group.forward, "phase": group.phase, "rows": []},
        )
        by_layer: dict[int, list[int]] = {}
        for layer, expert in group.requests:
            by_layer.setdefault(layer, []).append(expert)
        forward["rows"].append(
            {
                "row": group.row,
                "token_position": group.token_position,
                "token_id": group.token_id,
                "layers": [
                    {"layer": layer, "expert_ids": experts}
                    for layer, experts in sorted(by_layer.items())
                ],
            }
        )
    return [forwards[key] for key in sorted(forwards)]


def load_multi_request_trace(path: str | Path) -> MultiRequestTrace:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", 0)) != 3:
        raise ValueError("expected a schema_version=3 multi-request trace")
    sessions: list[SessionRoute] = []
    for item in raw.get("sessions", []):
        if not isinstance(item, Mapping):
            raise ValueError("session entries must be objects")
        session_raw = {
            "schema_version": 2,
            "forwards": item.get("forwards", []),
            "trace_sha256": item.get("source_sha256"),
            "workload": {
                "prompt_id": item.get("prompt_id"),
                "prompt": item.get("prompt", ""),
                "input_tokens": item.get("input_tokens"),
                "generated_ids": item.get("generated_ids", []),
            },
        }
        sessions.append(SessionRoute.from_raw(session_raw, session_id=str(item["session_id"])))
    schedules: list[SessionSchedule] = []
    for item in raw.get("schedules", []):
        steps = tuple(
            ScheduleStep(
                scheduler_step=int(step["scheduler_step"]),
                sessions=tuple(
                    (str(session["session_id"]), int(session["forward"]))
                    for session in step.get("sessions", [])
                ),
            )
            for step in item.get("steps", [])
        )
        schedules.append(
            SessionSchedule(
                schedule_id=str(item["schedule_id"]),
                kind=str(item.get("kind", "unknown")),
                steps=steps,
            )
        )
    return MultiRequestTrace(
        sessions=tuple(sessions),
        schedules=tuple(schedules),
        schema_version=3,
        model=raw.get("model") if isinstance(raw.get("model"), Mapping) else {},
    )


# [Main Dev]

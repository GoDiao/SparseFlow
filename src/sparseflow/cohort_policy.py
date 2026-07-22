"""Cache-aware fixed-cohort partitioning helpers.

The policy groups real sessions by the growth they add to a bounded
layer/expert working set.  It does not synthesize routes and does not own a
runtime scheduler.

[Main Dev]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Hashable, Iterable, Sequence


@dataclass(frozen=True)
class SessionWorkingSet:
    session_id: str
    items: FrozenSet[Hashable]


@dataclass(frozen=True)
class SessionCohort:
    session_ids: tuple[str, ...]
    working_set: FrozenSet[Hashable]
    overflow: bool


def partition_working_sets(
    sessions: Sequence[SessionWorkingSet],
    max_items: int,
) -> tuple[SessionCohort, ...]:
    """Greedily form cohorts whose union stays below ``max_items``.

    Candidates are assigned to the existing cohort that gains the fewest new
    items.  Ties preserve input order.  A single session larger than the
    bound is retained as an overflow cohort so the policy cannot drop a real
    request; callers report that condition as a failed budget fit.
    """

    if max_items < 1:
        raise ValueError("max_items must be positive")
    groups: list[dict[str, object]] = []
    for session in sessions:
        if not session.session_id:
            raise ValueError("session_id must not be empty")
        chosen = None
        chosen_growth = None
        for index, group in enumerate(groups):
            current = group["working_set"]
            assert isinstance(current, set)
            growth = len(current | set(session.items)) - len(current)
            if len(current) + growth <= max_items and (
                chosen_growth is None or growth < chosen_growth
            ):
                chosen = index
                chosen_growth = growth
        if chosen is None:
            groups.append({
                "session_ids": [session.session_id],
                "working_set": set(session.items),
            })
        else:
            group = groups[chosen]
            ids = group["session_ids"]
            current = group["working_set"]
            assert isinstance(ids, list) and isinstance(current, set)
            ids.append(session.session_id)
            current.update(session.items)
    result = []
    for group in groups:
        ids = group["session_ids"]
        current = group["working_set"]
        assert isinstance(ids, list) and isinstance(current, set)
        result.append(SessionCohort(
            session_ids=tuple(str(item) for item in ids),
            working_set=frozenset(current),
            overflow=len(current) > max_items,
        ))
    return tuple(result)


# [Main Dev]

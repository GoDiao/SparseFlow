from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Protocol, runtime_checkable


CacheKey = tuple[int, int]

POLICY_VARIANTS = {
    "S0": {"cache_policy": "none", "prefetch_policy": "none"},
    "S1": {"cache_policy": "lru", "prefetch_policy": "none"},
    "S2": {"cache_policy": "hot", "prefetch_policy": "none"},
    "S3": {"cache_policy": "heat", "prefetch_policy": "none"},
    "S4": {"cache_policy": "heat", "prefetch_policy": "previous-token"},
}


@runtime_checkable
class CachePolicy(Protocol):
    policy_id: str

    def begin_forward(self, forward: int, phase: str) -> None: ...

    def observe_routes(
        self,
        layer: int,
        expert_counts: Mapping[int, int],
        phase: str,
    ) -> None: ...

    def should_admit(self, key: CacheKey, phase: str, source: str) -> bool: ...

    def choose_victim(self, keys_lru_to_mru: Iterable[CacheKey]) -> CacheKey: ...

    def on_insert(self, key: CacheKey, nbytes: int) -> None: ...

    def on_evict(self, key: CacheKey, nbytes: int) -> None: ...

    def hot_keys(self) -> tuple[CacheKey, ...]: ...

    def snapshot(self) -> dict[str, object]: ...


class LRUPolicy:
    policy_id = "lru"

    def __init__(self):
        self._forward = -1
        self._phase = "unknown"

    def begin_forward(self, forward: int, phase: str) -> None:
        self._forward = forward
        self._phase = phase

    def observe_routes(
        self,
        layer: int,
        expert_counts: Mapping[int, int],
        phase: str,
    ) -> None:
        del layer, expert_counts, phase

    def should_admit(self, key: CacheKey, phase: str, source: str) -> bool:
        del key, phase, source
        return True

    def choose_victim(self, keys_lru_to_mru: Iterable[CacheKey]) -> CacheKey:
        try:
            return next(iter(keys_lru_to_mru))
        except StopIteration as exc:
            raise RuntimeError("cannot choose a cache victim from an empty set") from exc

    def on_insert(self, key: CacheKey, nbytes: int) -> None:
        del key, nbytes

    def on_evict(self, key: CacheKey, nbytes: int) -> None:
        del key, nbytes

    def hot_keys(self) -> tuple[CacheKey, ...]:
        return ()

    def snapshot(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "forward": self._forward,
            "phase": self._phase,
            "hot_entries": 0,
            "promotions": 0,
            "demotions": 0,
            "admission_rejections": 0,
        }


class NoCachePolicy(LRUPolicy):
    policy_id = "none"

    def __init__(self):
        super().__init__()
        self._rejections = 0

    def should_admit(self, key: CacheKey, phase: str, source: str) -> bool:
        del key, phase, source
        self._rejections += 1
        return False

    def snapshot(self) -> dict[str, object]:
        return {**super().snapshot(), "admission_rejections": self._rejections}


class HeatPolicy(LRUPolicy):
    """LRU with route heat, a bounded hot tier, decay, and hysteresis."""

    def __init__(
        self,
        policy_id: str,
        max_hot_entries: int,
        decay: float,
        promote_threshold: float,
        demote_threshold: float,
        swap_margin: float,
        second_touch_prefill: bool,
    ):
        super().__init__()
        if max_hot_entries < 0:
            raise ValueError("max_hot_entries must be non-negative")
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if promote_threshold < demote_threshold:
            raise ValueError("promotion threshold must be >= demotion threshold")
        if swap_margin < 0:
            raise ValueError("swap_margin must be non-negative")
        self.policy_id = policy_id
        self.max_hot_entries = max_hot_entries
        self.decay = decay
        self.promote_threshold = promote_threshold
        self.demote_threshold = demote_threshold
        self.swap_margin = swap_margin
        self.second_touch_prefill = second_touch_prefill
        self._heat: dict[CacheKey, float] = defaultdict(float)
        self._seen: dict[CacheKey, int] = defaultdict(int)
        self._hot: set[CacheKey] = set()
        self._promotions = 0
        self._demotions = 0
        self._rejections = 0

    def begin_forward(self, forward: int, phase: str) -> None:
        if forward != self._forward and self._forward >= 0 and self.decay < 1.0:
            for key in tuple(self._heat):
                value = self._heat[key] * self.decay
                if value < 1e-6:
                    self._heat.pop(key, None)
                else:
                    self._heat[key] = value
            for key in tuple(self._hot):
                if self._heat.get(key, 0.0) < self.demote_threshold:
                    self._hot.remove(key)
                    self._demotions += 1
        super().begin_forward(forward, phase)

    def observe_routes(
        self,
        layer: int,
        expert_counts: Mapping[int, int],
        phase: str,
    ) -> None:
        del phase
        for expert_id, count in expert_counts.items():
            key = (layer, int(expert_id))
            count = int(count)
            if count <= 0:
                continue
            self._seen[key] += count
            self._heat[key] += count
            self._maybe_promote(key)

    def _maybe_promote(self, key: CacheKey) -> None:
        if key in self._hot or self.max_hot_entries == 0:
            return
        heat = self._heat[key]
        if heat < self.promote_threshold:
            return
        if len(self._hot) < self.max_hot_entries:
            self._hot.add(key)
            self._promotions += 1
            return
        coldest = min(self._hot, key=lambda item: (self._heat.get(item, 0.0), item))
        if heat >= self._heat.get(coldest, 0.0) + self.swap_margin:
            self._hot.remove(coldest)
            self._hot.add(key)
            self._demotions += 1
            self._promotions += 1

    def should_admit(self, key: CacheKey, phase: str, source: str) -> bool:
        if source == "prefetch" or key in self._hot:
            return True
        if self.second_touch_prefill and phase == "prefill" and self._seen.get(key, 0) < 2:
            self._rejections += 1
            return False
        return True

    def choose_victim(self, keys_lru_to_mru: Iterable[CacheKey]) -> CacheKey:
        keys = tuple(keys_lru_to_mru)
        if not keys:
            raise RuntimeError("cannot choose a cache victim from an empty set")
        for key in keys:
            if key not in self._hot:
                return key
        order = {key: index for index, key in enumerate(keys)}
        return min(keys, key=lambda item: (self._heat.get(item, 0.0), order[item]))

    def on_evict(self, key: CacheKey, nbytes: int) -> None:
        del nbytes
        if key in self._hot:
            self._hot.remove(key)
            self._demotions += 1

    def hot_keys(self) -> tuple[CacheKey, ...]:
        return tuple(sorted(self._hot))

    def snapshot(self) -> dict[str, object]:
        top = sorted(self._heat.items(), key=lambda item: (-item[1], item[0]))[:16]
        return {
            "policy_id": self.policy_id,
            "forward": self._forward,
            "phase": self._phase,
            "hot_entries": len(self._hot),
            "max_hot_entries": self.max_hot_entries,
            "promotions": self._promotions,
            "demotions": self._demotions,
            "admission_rejections": self._rejections,
            "decay": self.decay,
            "promote_threshold": self.promote_threshold,
            "demote_threshold": self.demote_threshold,
            "swap_margin": self.swap_margin,
            "second_touch_prefill": self.second_touch_prefill,
            "top_heat": [
                {"layer": key[0], "expert": key[1], "heat": value}
                for key, value in top
            ],
        }


def make_cache_policy(name: str, max_hot_entries: int = 0) -> CachePolicy:
    if name == "none":
        return NoCachePolicy()
    if name == "lru":
        return LRUPolicy()
    if name == "hot":
        return HeatPolicy(
            policy_id="hot",
            max_hot_entries=max_hot_entries,
            decay=1.0,
            promote_threshold=2.0,
            demote_threshold=0.0,
            swap_margin=0.0,
            second_touch_prefill=False,
        )
    if name == "heat":
        return HeatPolicy(
            policy_id="heat-hysteresis",
            max_hot_entries=max_hot_entries,
            decay=0.90,
            promote_threshold=3.0,
            demote_threshold=1.0,
            swap_margin=1.0,
            second_touch_prefill=True,
        )
    raise ValueError(f"unknown cache policy: {name}")

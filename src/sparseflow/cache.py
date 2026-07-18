from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Any, Callable, Mapping, TypeAlias

from .cache_policy import CachePolicy, LRUPolicy


BytePayload: TypeAlias = bytes | bytearray | memoryview


@dataclass(frozen=True)
class CachedExpert:
    layer: int
    expert_id: int
    parts: Mapping[str, BytePayload]
    size_override: int | None = None

    @property
    def nbytes(self) -> int:
        return (
            self.size_override
            if self.size_override is not None
            else sum(len(data) for data in self.parts.values())
        )


class ExpertLease:
    """Keep one cache entry and its backing buffers stable during native use."""

    def __init__(
        self,
        cache: "ExpertCache | None",
        entry: CachedExpert,
        references: tuple[Any, ...] = (),
    ):
        self.cache = cache
        self.entry = entry
        self.references = references
        self._released = False

    def __enter__(self) -> CachedExpert:
        return self.entry

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.cache is not None:
            self.cache._release_lease(self.entry)


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loaded_bytes: int = 0
    hit_bytes: int = 0
    miss_bytes: int = 0
    admission_rejections: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def as_dict(self, cached_bytes: int, entries: int) -> dict[str, int | float]:
        weighted_total = self.hit_bytes + self.miss_bytes
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "loaded_bytes": self.loaded_bytes,
            "hit_bytes": self.hit_bytes,
            "miss_bytes": self.miss_bytes,
            "byte_weighted_hit_rate": self.hit_bytes / weighted_total if weighted_total else 0.0,
            "admission_rejections": self.admission_rejections,
            "cached_bytes": cached_bytes,
            "cache_entries": entries,
        }


class ExpertCache:
    """Per-layer LRU cache for raw routed-expert payloads.

    ``capacity_per_layer`` counts logical experts, not tensor parts. One cache
    entry contains all parts of one expert, currently ``gate_up_proj`` and
    ``down_proj`` for Qwen3.6.
    """

    def __init__(
        self,
        capacity_per_layer: int | None = None,
        max_bytes: int | None = None,
        policy: CachePolicy | None = None,
        collect_timings: bool = False,
    ):
        if capacity_per_layer is None and max_bytes is None:
            raise ValueError("capacity_per_layer or max_bytes is required")
        if capacity_per_layer is not None and capacity_per_layer < 0:
            raise ValueError("capacity_per_layer must be non-negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.capacity_per_layer = capacity_per_layer
        self.max_bytes = max_bytes
        self.policy = policy or LRUPolicy()
        self.collect_timings = collect_timings
        self.stats = CacheStats()
        self._layers: dict[int, OrderedDict[int, CachedExpert]] = {}
        self._global_lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._cached_bytes = 0
        self._entries = 0
        self._pin_counts: dict[int, int] = {}
        self._pinned_entries: dict[int, CachedExpert] = {}
        self._eviction_listeners: list[Callable[[CachedExpert], None]] = []
        self._phase = "unknown"
        self._forward = -1
        self._timings_ms = {
            "cache_lookup": 0.0,
            "victim_selection": 0.0,
            "allocation_reuse": 0.0,
            "policy_maintenance": 0.0,
        }

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def entries(self) -> int:
        return self._entries

    @property
    def pinned_entries(self) -> int:
        return len(self._pinned_entries)

    @property
    def pinned_bytes(self) -> int:
        return sum(entry.nbytes for entry in self._pinned_entries.values())

    def lease(
        self,
        entry: CachedExpert,
        references: tuple[Any, ...] = (),
    ) -> ExpertLease:
        """Pin an admitted entry until the returned lease is released."""

        live = self.peek(entry.layer, entry.expert_id)
        if live is not entry:
            return ExpertLease(None, entry, references)
        identity = id(entry)
        self._pin_counts[identity] = self._pin_counts.get(identity, 0) + 1
        self._pinned_entries[identity] = entry
        return ExpertLease(self, entry, references)

    def _release_lease(self, entry: CachedExpert) -> None:
        identity = id(entry)
        count = self._pin_counts.get(identity, 0)
        if count <= 0:
            raise RuntimeError("expert lease released after cache ownership ended")
        if count == 1:
            self._pin_counts.pop(identity, None)
            self._pinned_entries.pop(identity, None)
        else:
            self._pin_counts[identity] = count - 1

    def _is_pinned(self, entry: CachedExpert) -> bool:
        return self._pin_counts.get(id(entry), 0) > 0

    def add_eviction_listener(self, listener: Callable[[CachedExpert], None]) -> None:
        if listener not in self._eviction_listeners:
            self._eviction_listeners.append(listener)

    def remove_eviction_listener(self, listener: Callable[[CachedExpert], None]) -> None:
        try:
            self._eviction_listeners.remove(listener)
        except ValueError:
            pass

    def begin_forward(self, forward: int, phase: str) -> None:
        self._forward = forward
        self._phase = phase
        started = time.perf_counter() if self.collect_timings else None
        self.policy.begin_forward(forward, phase)
        self._record_timing("policy_maintenance", started)

    def observe_routes(self, layer: int, expert_counts: Mapping[int, int]) -> None:
        started = time.perf_counter() if self.collect_timings else None
        self.policy.observe_routes(layer, expert_counts, self._phase)
        self._record_timing("policy_maintenance", started)

    def lookup(
        self,
        layer: int,
        expert_id: int,
        expected_nbytes: int | None = None,
    ) -> CachedExpert | None:
        started = time.perf_counter() if self.collect_timings else None
        self.stats.requests += 1
        layer_cache = self._layers.get(layer)
        if layer_cache is None or expert_id not in layer_cache:
            self.stats.misses += 1
            if expected_nbytes is not None:
                self.stats.miss_bytes += expected_nbytes
            self._record_timing("cache_lookup", started)
            return None
        entry = layer_cache.pop(expert_id)
        layer_cache[expert_id] = entry
        key = (layer, expert_id)
        self._global_lru.pop(key, None)
        self._global_lru[key] = None
        self.stats.hits += 1
        self.stats.hit_bytes += entry.nbytes
        self._record_timing("cache_lookup", started)
        return entry

    def peek(self, layer: int, expert_id: int) -> CachedExpert | None:
        """Inspect an entry without changing LRU order or request stats."""

        layer_cache = self._layers.get(layer)
        return layer_cache.get(expert_id) if layer_cache is not None else None

    def get_or_load(
        self,
        layer: int,
        expert_id: int,
        loader: Callable[[], Mapping[str, BytePayload]],
    ) -> CachedExpert:
        cached = self.lookup(layer, expert_id)
        if cached is not None:
            return cached

        return self.put_loaded(layer, expert_id, loader())

    def put_loaded(
        self,
        layer: int,
        expert_id: int,
        payloads: Mapping[str, BytePayload],
        source: str = "demand",
    ) -> CachedExpert:
        """Insert already-read payloads without incrementing request stats."""

        existing = self.peek(layer, expert_id)
        if existing is not None:
            return existing
        # Keep one writable backing buffer per part so tensor views can be
        # created without a second copy at the runtime adapter boundary.
        allocation_started = time.perf_counter() if self.collect_timings else None
        normalized = {
            part: data if isinstance(data, bytearray) else bytearray(data)
            for part, data in payloads.items()
        }
        self._record_timing("allocation_reuse", allocation_started)
        if not normalized:
            raise ValueError(f"loader returned no payloads for layer={layer}, expert={expert_id}")
        entry = CachedExpert(layer=layer, expert_id=expert_id, parts=normalized)
        self._insert(entry, source=source)
        self.stats.loaded_bytes += entry.nbytes
        return entry

    def put_sized(
        self,
        layer: int,
        expert_id: int,
        nbytes: int,
        source: str = "demand",
    ) -> CachedExpert:
        """Insert a metadata-only entry for policy/route replay."""

        if nbytes <= 0:
            raise ValueError("metadata cache entry size must be positive")
        existing = self.peek(layer, expert_id)
        if existing is not None:
            return existing
        entry = CachedExpert(
            layer=layer,
            expert_id=expert_id,
            parts={},
            size_override=nbytes,
        )
        self._insert(entry, source=source)
        self.stats.loaded_bytes += nbytes
        return entry

    def _insert(self, entry: CachedExpert, source: str) -> None:
        layer_cache = self._layers.setdefault(entry.layer, OrderedDict())
        previous = layer_cache.pop(entry.expert_id, None)
        if previous is not None:
            self._cached_bytes -= previous.nbytes
            self._entries -= 1
            self._global_lru.pop((entry.layer, entry.expert_id), None)
            self._notify_evicted(previous)

        if self.capacity_per_layer == 0:
            return
        if self.max_bytes == 0 or (self.max_bytes is not None and entry.nbytes > self.max_bytes):
            return
        policy_started = time.perf_counter() if self.collect_timings else None
        should_admit = self.policy.should_admit(
            (entry.layer, entry.expert_id),
            self._phase,
            source,
        )
        self._record_timing("policy_maintenance", policy_started)
        if not should_admit:
            self.stats.admission_rejections += 1
            return

        while self.capacity_per_layer is not None and len(layer_cache) >= self.capacity_per_layer:
            victim_started = time.perf_counter() if self.collect_timings else None
            candidates = tuple(
                (entry.layer, expert)
                for expert, candidate in layer_cache.items()
                if not self._is_pinned(candidate)
            )
            if not candidates:
                self.stats.admission_rejections += 1
                return
            victim = candidates[0] if type(self.policy) is LRUPolicy else self.policy.choose_victim(candidates)
            self._record_timing("victim_selection", victim_started)
            self._evict_key(victim)

        layer_cache[entry.expert_id] = entry
        self._cached_bytes += entry.nbytes
        self._entries += 1
        self._global_lru[(entry.layer, entry.expert_id)] = None
        policy_started = time.perf_counter() if self.collect_timings else None
        self.policy.on_insert((entry.layer, entry.expert_id), entry.nbytes)
        self._record_timing("policy_maintenance", policy_started)

        while self.max_bytes is not None and self._cached_bytes > self.max_bytes:
            victim_started = time.perf_counter() if self.collect_timings else None
            candidates = tuple(
                key
                for key in self._global_lru
                if not self._is_pinned(self._layers[key[0]][key[1]])
            )
            if not candidates:
                raise RuntimeError("cache budget exceeded with no evictable expert")
            victim = candidates[0] if type(self.policy) is LRUPolicy else self.policy.choose_victim(candidates)
            self._record_timing("victim_selection", victim_started)
            if victim == (entry.layer, entry.expert_id):
                self._remove_key(victim, notify=False, count_eviction=False)
                self.stats.admission_rejections += 1
                return
            self._evict_key(victim)

    def _evict_key(self, key: tuple[int, int]) -> None:
        entry = self._layers[key[0]][key[1]]
        if self._is_pinned(entry):
            raise RuntimeError(f"cannot evict leased expert: {key}")
        self._remove_key(key, notify=True, count_eviction=True)

    def _remove_key(
        self,
        key: tuple[int, int],
        notify: bool,
        count_eviction: bool,
    ) -> None:
        old_layer, old_expert = key
        old_cache = self._layers[old_layer]
        old_entry = old_cache.pop(old_expert)
        self._global_lru.pop(key, None)
        self._cached_bytes -= old_entry.nbytes
        self._entries -= 1
        if count_eviction:
            self.stats.evictions += 1
        policy_started = time.perf_counter() if self.collect_timings else None
        self.policy.on_evict(key, old_entry.nbytes)
        self._record_timing("policy_maintenance", policy_started)
        if notify:
            self._notify_evicted(old_entry)

    def _notify_evicted(self, entry: CachedExpert) -> None:
        for listener in tuple(self._eviction_listeners):
            listener(entry)

    def clear(self) -> None:
        if self._pin_counts:
            raise RuntimeError("cannot clear ExpertCache while expert leases are active")
        for layer_cache in self._layers.values():
            for entry in layer_cache.values():
                self.policy.on_evict((entry.layer, entry.expert_id), entry.nbytes)
                self._notify_evicted(entry)
        self._layers.clear()
        self._global_lru.clear()
        self._cached_bytes = 0
        self._entries = 0

    def cached_keys(self) -> tuple[tuple[int, int], ...]:
        """Return cached keys without changing LRU order or statistics."""

        return tuple(self._global_lru.keys())

    def counters(self) -> dict[str, int | float]:
        """Return fixed-cost counters without materializing policy diagnostics."""

        return {
            **self.stats.as_dict(self.cached_bytes, self.entries),
            "pinned_entries": self.pinned_entries,
            "pinned_bytes": self.pinned_bytes,
            **{f"timing_{key}_ms_total": value for key, value in self._timings_ms.items()},
        }

    def _record_timing(self, category: str, started: float | None) -> None:
        if started is not None:
            self._timings_ms[category] += (time.perf_counter() - started) * 1000.0

    def stats_dict(self, include_policy: bool = True) -> dict[str, object]:
        result: dict[str, object] = dict(self.counters())
        if include_policy:
            result["policy"] = self.policy.snapshot()
        return result


# [Main Dev]

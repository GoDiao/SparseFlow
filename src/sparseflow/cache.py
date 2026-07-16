from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping, TypeAlias

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
        self.stats = CacheStats()
        self._layers: dict[int, OrderedDict[int, CachedExpert]] = {}
        self._global_lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._cached_bytes = 0
        self._entries = 0
        self._eviction_listeners: list[Callable[[CachedExpert], None]] = []
        self._phase = "unknown"
        self._forward = -1

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def entries(self) -> int:
        return self._entries

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
        self.policy.begin_forward(forward, phase)

    def observe_routes(self, layer: int, expert_counts: Mapping[int, int]) -> None:
        self.policy.observe_routes(layer, expert_counts, self._phase)

    def lookup(
        self,
        layer: int,
        expert_id: int,
        expected_nbytes: int | None = None,
    ) -> CachedExpert | None:
        self.stats.requests += 1
        layer_cache = self._layers.get(layer)
        if layer_cache is None or expert_id not in layer_cache:
            self.stats.misses += 1
            if expected_nbytes is not None:
                self.stats.miss_bytes += expected_nbytes
            return None
        entry = layer_cache.pop(expert_id)
        layer_cache[expert_id] = entry
        key = (layer, expert_id)
        self._global_lru.pop(key, None)
        self._global_lru[key] = None
        self.stats.hits += 1
        self.stats.hit_bytes += entry.nbytes
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
        normalized = {
            part: data if isinstance(data, bytearray) else bytearray(data)
            for part, data in payloads.items()
        }
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
        if not self.policy.should_admit(
            (entry.layer, entry.expert_id),
            self._phase,
            source,
        ):
            self.stats.admission_rejections += 1
            return

        while self.capacity_per_layer is not None and len(layer_cache) >= self.capacity_per_layer:
            if type(self.policy) is LRUPolicy:
                victim = (entry.layer, next(iter(layer_cache)))
            else:
                candidates = tuple((entry.layer, expert) for expert in layer_cache)
                victim = self.policy.choose_victim(candidates)
            self._evict_key(victim)

        layer_cache[entry.expert_id] = entry
        self._cached_bytes += entry.nbytes
        self._entries += 1
        self._global_lru[(entry.layer, entry.expert_id)] = None
        self.policy.on_insert((entry.layer, entry.expert_id), entry.nbytes)

        while self.max_bytes is not None and self._cached_bytes > self.max_bytes:
            if type(self.policy) is LRUPolicy:
                victim = next(iter(self._global_lru))
            else:
                victim = self.policy.choose_victim(tuple(self._global_lru))
            self._evict_key(victim)

    def _evict_key(self, key: tuple[int, int]) -> None:
        old_layer, old_expert = key
        old_cache = self._layers[old_layer]
        old_entry = old_cache.pop(old_expert)
        self._global_lru.pop(key, None)
        self._cached_bytes -= old_entry.nbytes
        self._entries -= 1
        self.stats.evictions += 1
        self.policy.on_evict(key, old_entry.nbytes)
        self._notify_evicted(old_entry)

    def _notify_evicted(self, entry: CachedExpert) -> None:
        for listener in tuple(self._eviction_listeners):
            listener(entry)

    def clear(self) -> None:
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

        return self.stats.as_dict(self.cached_bytes, self.entries)

    def stats_dict(self, include_policy: bool = True) -> dict[str, object]:
        result: dict[str, object] = dict(self.counters())
        if include_policy:
            result["policy"] = self.policy.snapshot()
        return result


# [Main Dev]

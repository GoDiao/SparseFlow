from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Mapping, TypeAlias


BytePayload: TypeAlias = bytes | bytearray | memoryview


@dataclass(frozen=True)
class CachedExpert:
    layer: int
    expert_id: int
    parts: Mapping[str, BytePayload]

    @property
    def nbytes(self) -> int:
        return sum(len(data) for data in self.parts.values())


@dataclass
class CacheStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loaded_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def as_dict(self, cached_bytes: int, entries: int) -> dict[str, int | float]:
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "evictions": self.evictions,
            "loaded_bytes": self.loaded_bytes,
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
    ):
        if capacity_per_layer is None and max_bytes is None:
            raise ValueError("capacity_per_layer or max_bytes is required")
        if capacity_per_layer is not None and capacity_per_layer < 0:
            raise ValueError("capacity_per_layer must be non-negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        self.capacity_per_layer = capacity_per_layer
        self.max_bytes = max_bytes
        self.stats = CacheStats()
        self._layers: dict[int, OrderedDict[int, CachedExpert]] = {}
        self._global_lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self._cached_bytes = 0

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def entries(self) -> int:
        return sum(len(layer_cache) for layer_cache in self._layers.values())

    def lookup(self, layer: int, expert_id: int) -> CachedExpert | None:
        self.stats.requests += 1
        layer_cache = self._layers.get(layer)
        if layer_cache is None or expert_id not in layer_cache:
            self.stats.misses += 1
            return None
        entry = layer_cache.pop(expert_id)
        layer_cache[expert_id] = entry
        key = (layer, expert_id)
        self._global_lru.pop(key, None)
        self._global_lru[key] = None
        self.stats.hits += 1
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
    ) -> CachedExpert:
        """Insert already-read payloads without incrementing request stats."""

        existing = self.peek(layer, expert_id)
        if existing is not None:
            return existing
        # Keep one writable backing buffer per part so tensor views can be
        # created without a second copy at the runtime adapter boundary.
        normalized = {part: bytearray(data) for part, data in payloads.items()}
        if not normalized:
            raise ValueError(f"loader returned no payloads for layer={layer}, expert={expert_id}")
        entry = CachedExpert(layer=layer, expert_id=expert_id, parts=normalized)
        self._insert(entry)
        self.stats.loaded_bytes += entry.nbytes
        return entry

    def _insert(self, entry: CachedExpert) -> None:
        layer_cache = self._layers.setdefault(entry.layer, OrderedDict())
        previous = layer_cache.pop(entry.expert_id, None)
        if previous is not None:
            self._cached_bytes -= previous.nbytes
            self._global_lru.pop((entry.layer, entry.expert_id), None)

        if self.capacity_per_layer == 0:
            return
        if self.max_bytes == 0 or (self.max_bytes is not None and entry.nbytes > self.max_bytes):
            return

        while self.capacity_per_layer is not None and len(layer_cache) >= self.capacity_per_layer:
            _, evicted = layer_cache.popitem(last=False)
            self._cached_bytes -= evicted.nbytes
            self._global_lru.pop((evicted.layer, evicted.expert_id), None)
            self.stats.evictions += 1

        layer_cache[entry.expert_id] = entry
        self._cached_bytes += entry.nbytes
        self._global_lru[(entry.layer, entry.expert_id)] = None

        while self.max_bytes is not None and self._cached_bytes > self.max_bytes:
            key, _ = self._global_lru.popitem(last=False)
            old_layer, old_expert = key
            old_cache = self._layers[old_layer]
            old_entry = old_cache.pop(old_expert)
            self._cached_bytes -= old_entry.nbytes
            self.stats.evictions += 1

    def clear(self) -> None:
        self._layers.clear()
        self._global_lru.clear()
        self._cached_bytes = 0

    def cached_keys(self) -> tuple[tuple[int, int], ...]:
        """Return cached keys without changing LRU order or statistics."""

        return tuple(self._global_lru.keys())

    def stats_dict(self) -> dict[str, int | float]:
        return self.stats.as_dict(self.cached_bytes, self.entries)

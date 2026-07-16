from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class BufferPoolStats:
    allocations: int = 0
    reuses: int = 0
    releases: int = 0
    dropped: int = 0


class BufferPool:
    """Bounded size-class pool for writable expert payload buffers."""

    def __init__(self, max_cached_bytes: int, max_per_size: int = 8):
        if max_cached_bytes < 0:
            raise ValueError("max_cached_bytes must be non-negative")
        if max_per_size < 1:
            raise ValueError("max_per_size must be positive")
        self.max_cached_bytes = max_cached_bytes
        self.max_per_size = max_per_size
        self.stats = BufferPoolStats()
        self._available: dict[int, list[bytearray]] = defaultdict(list)
        self._cached_bytes = 0

    @property
    def cached_bytes(self) -> int:
        return self._cached_bytes

    @property
    def buffers(self) -> int:
        return sum(len(items) for items in self._available.values())

    def acquire(self, nbytes: int) -> bytearray:
        if nbytes <= 0:
            raise ValueError("buffer size must be positive")
        available = self._available.get(nbytes)
        if available:
            buffer = available.pop()
            self._cached_bytes -= nbytes
            self.stats.reuses += 1
            return buffer
        self.stats.allocations += 1
        return bytearray(nbytes)

    def release(self, buffer: bytearray) -> None:
        nbytes = len(buffer)
        available = self._available[nbytes]
        if (
            nbytes > self.max_cached_bytes
            or self._cached_bytes + nbytes > self.max_cached_bytes
            or len(available) >= self.max_per_size
        ):
            self.stats.dropped += 1
            return
        available.append(buffer)
        self._cached_bytes += nbytes
        self.stats.releases += 1

    def clear(self) -> None:
        self._available.clear()
        self._cached_bytes = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "allocations": self.stats.allocations,
            "reuses": self.stats.reuses,
            "releases": self.stats.releases,
            "dropped": self.stats.dropped,
            "cached_bytes": self.cached_bytes,
            "buffers": self.buffers,
            "max_cached_bytes": self.max_cached_bytes,
            "max_per_size": self.max_per_size,
        }


# [Main Dev]

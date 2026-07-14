from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .locator import ExpertLocation, ExpertLocator


@dataclass(frozen=True)
class LocatedPart:
    """One expert tensor part participating in a batch read."""

    key: tuple[int, int]
    part: str
    location: Any


@dataclass(frozen=True)
class CoalescedRange:
    """A physical shard range containing one or more logical tensor parts."""

    shard: Path
    start: int
    end: int
    parts: tuple[LocatedPart, ...]

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def useful_bytes(self) -> int:
        return sum(item.location.nbytes for item in self.parts)

    @property
    def wasted_bytes(self) -> int:
        return max(0, self.nbytes - self.useful_bytes)


@dataclass(frozen=True)
class BatchReadStats:
    """Metrics for one logical expert batch read."""

    ranges: int
    logical_ranges: int
    useful_bytes: int
    physical_bytes: int
    read_calls: int

    @property
    def wasted_bytes(self) -> int:
        return max(0, self.physical_bytes - self.useful_bytes)

    @property
    def coalescing_ratio(self) -> float:
        return self.read_calls / self.logical_ranges if self.logical_ranges else 1.0


class BatchReadResult(dict):
    """Dictionary-compatible batch payloads carrying physical read metrics."""

    def __init__(self, payloads: dict[tuple[int, int], dict[str, bytes]], stats: BatchReadStats):
        super().__init__(payloads)
        self.stats = stats


def coalesce_locations(
    locations: Iterable[ExpertLocation],
    max_gap: int = 0,
) -> tuple[CoalescedRange, ...]:
    """Group expert slices by shard and merge ranges within ``max_gap``.

    The returned ranges retain enough metadata to split one physical read
    back into ``(layer, expert_id, part)`` payloads.  ``max_gap`` is explicit
    because reading a small hole can reduce syscalls, while a large hole would
    defeat the purpose of expert streaming.
    """

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    pending: list[LocatedPart] = []
    seen: set[tuple[int, int]] = set()
    for location in locations:
        key = (location.layer, location.expert_id)
        if key in seen:
            raise ValueError(f"duplicate expert location in batch: {key}")
        seen.add(key)
        pending.extend(
            LocatedPart(key=key, part=item.part, location=item)
            for item in location.parts
        )
    pending.sort(key=lambda item: (str(item.location.shard), item.location.file_offset))

    ranges: list[CoalescedRange] = []
    for item in pending:
        shard = item.location.shard
        start = item.location.file_offset
        end = start + item.location.nbytes
        if ranges and ranges[-1].shard == shard and start <= ranges[-1].end + max_gap:
            previous = ranges[-1]
            ranges[-1] = CoalescedRange(
                shard=previous.shard,
                start=previous.start,
                end=max(previous.end, end),
                parts=previous.parts + (item,),
            )
        else:
            ranges.append(
                CoalescedRange(
                    shard=shard,
                    start=start,
                    end=end,
                    parts=(item,),
                )
            )
    return tuple(ranges)


class ShardReader:
    """Persistent descriptor pool with positional reads for safetensor shards."""

    def __init__(self, max_open_files: int = 64):
        if max_open_files < 1:
            raise ValueError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self._fds: OrderedDict[Path, int] = OrderedDict()
        self._lock = threading.Lock()
        self.read_calls = 0
        self.read_bytes = 0
        self.last_batch_stats = BatchReadStats(0, 0, 0, 0, 0)

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _fd_for(self, path: Path) -> int:
        path = path.resolve()
        with self._lock:
            fd = self._fds.pop(path, None)
            if fd is None:
                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
                fd = os.open(path, flags)
            self._fds[path] = fd
            while len(self._fds) > self.max_open_files:
                _old_path, old_fd = self._fds.popitem(last=False)
                os.close(old_fd)
            return fd

    def read(self, path: Path, offset: int, nbytes: int) -> bytes:
        if offset < 0 or nbytes < 0:
            raise ValueError("offset and nbytes must be non-negative")
        fd = self._fd_for(path)
        if hasattr(os, "pread"):
            data = os.pread(fd, nbytes, offset)
        else:  # Windows/Python fallback; the lock protects the shared file position.
            with self._lock:
                os.lseek(fd, offset, os.SEEK_SET)
                data = os.read(fd, nbytes)
        with self._lock:
            self.read_calls += 1
            self.read_bytes += len(data)
        if len(data) != nbytes:
            raise OSError(
                f"short expert read from {path}: expected {nbytes} bytes, got {len(data)}"
            )
        return data

    def read_slices(self, location: ExpertLocation) -> dict[str, bytes]:
        """Read one expert through the general batch range implementation."""

        return self.read_locations([location])[ (location.layer, location.expert_id) ]

    def read_locations(
        self,
        locations: Iterable[ExpertLocation],
        max_gap: int = 0,
    ) -> BatchReadResult:
        """Read multiple experts with shard-range coalescing.

        Per-batch metrics are available through :attr:`last_batch_stats`.
        Logical payloads are returned separately for every expert, so callers
        do not need to know whether a physical read was coalesced.
        """

        locations = list(locations)
        ranges = coalesce_locations(locations, max_gap=max_gap)
        result: dict[tuple[int, int], dict[str, bytes]] = {
            (location.layer, location.expert_id): {}
            for location in locations
        }
        useful_bytes = 0
        physical_bytes = 0
        logical_ranges = sum(len(location.parts) for location in locations)
        calls_before = self.read_calls
        for batch in ranges:
            data = self.read(batch.shard, batch.start, batch.nbytes)
            physical_bytes += len(data)
            useful_bytes += batch.useful_bytes
            for item in batch.parts:
                offset = item.location.file_offset - batch.start
                result[item.key][item.part] = data[offset : offset + item.location.nbytes]
        stats = BatchReadStats(
            ranges=len(ranges),
            logical_ranges=logical_ranges,
            useful_bytes=useful_bytes,
            physical_bytes=physical_bytes,
            read_calls=self.read_calls - calls_before,
        )
        self.last_batch_stats = stats
        return BatchReadResult(result, stats)

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def read_expert_bytes(
    location: ExpertLocation,
    reader: ShardReader | None = None,
) -> dict[str, bytes]:
    """Read each located expert part as raw bytes using positional I/O."""

    if reader is not None:
        return reader.read_slices(location)
    with ShardReader() as owned_reader:
        return owned_reader.read_slices(location)


def load_expert_raw(
    model_dir: str | Path,
    layer: int,
    expert_id: int,
) -> dict[str, Any]:
    """Read one expert's raw tensor slices and return verification metadata.

    This is deliberately a storage probe, not a model loader: bytes are read
    from the exact ranges returned by ``ExpertLocator`` and are not converted
    to a framework tensor or retained after hashing.
    """

    location = ExpertLocator(model_dir).locate(layer, expert_id)
    parts = []
    total_bytes = 0
    combined = hashlib.sha256()
    payloads = read_expert_bytes(location)

    for item in location:
        data = payloads[item.part]
        digest = hashlib.sha256(data).hexdigest()
        combined.update(data)
        total_bytes += len(data)
        parts.append(
            {
                **item.as_dict(),
                "bytes_read": len(data),
                "sha256": digest,
            }
        )

    return {
        "schema_version": 1,
        "layer": location.layer,
        "expert_id": location.expert_id,
        "total_bytes": total_bytes,
        "sha256": combined.hexdigest(),
        "parts": parts,
    }

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .locator import ExpertLocation, ExpertLocator


def read_expert_bytes(location: ExpertLocation) -> dict[str, bytes]:
    """Read each located expert part as raw bytes."""

    return {
        item.part: _read_exact(item.shard, item.file_offset, item.nbytes)
        for item in location
    }


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


def _read_exact(path: Path, offset: int, nbytes: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(nbytes)
    if len(data) != nbytes:
        raise OSError(
            f"short expert read from {path}: expected {nbytes} bytes, got {len(data)}"
        )
    return data

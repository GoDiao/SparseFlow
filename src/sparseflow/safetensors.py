from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TensorSpan:
    name: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offset: int
    nbytes: int

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


class ShardIndex:
    def __init__(self, model_dir: Path, tensors: dict[str, TensorSpan]):
        self.model_dir = model_dir
        self.tensors = tensors

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "ShardIndex":
        root = Path(model_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"model directory does not exist: {root}")

        tensors: dict[str, TensorSpan] = {}
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"no safetensors shards found: {root}")

        for shard in shards:
            tensors.update(_read_shard_header(shard))
        return cls(root, tensors)

    def __iter__(self) -> Iterable[TensorSpan]:
        return iter(self.tensors.values())

    def __len__(self) -> int:
        return len(self.tensors)

    def find(self, name: str) -> TensorSpan:
        return self.tensors[name]


def _read_shard_header(path: Path) -> dict[str, TensorSpan]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"short safetensors header: {path}")
        header_len = int.from_bytes(raw_len, "little")
        if header_len < 2 or header_len > file_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(stream.read(header_len))

    payload_start = 8 + header_len
    result: dict[str, TensorSpan] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not 0 <= start <= end <= file_size - payload_start:
            raise ValueError(f"invalid tensor offsets for {name}: {path}")
        result[name] = TensorSpan(
            name=name,
            shard=path,
            dtype=meta["dtype"],
            shape=tuple(meta.get("shape", ())),
            data_offset=payload_start + start,
            nbytes=end - start,
        )
    return result

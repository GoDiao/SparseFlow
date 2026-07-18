from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mmap
from pathlib import Path
import struct
from typing import Any, Callable, Iterable

from .loader import ShardReader
from .locator import ExpertLocator
from .memory_loader import peak_rss_bytes


FORMAT_VERSION = 1
FORMAT_ID = "canonical-int8-v1"
MAGIC = b"SPFINT8\0"
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIII")
PARTS = ("gate_up_proj", "down_proj")
EXEC_FORMAT_ID = "canonical-int8-exec-v1"
EXEC_FORMAT_VERSION = 1
EXEC_INDEX_NAME = "execution-index.json"
EXEC_MANIFEST_NAME = "execution-manifest.json"
EXEC_MAGIC = b"SPFEXEC\0"
EXEC_HEADER = struct.Struct("<8sIIII")


@dataclass(frozen=True)
class Int8PartLocation:
    part: str
    shape: tuple[int, int]
    quant_axis: int
    data_offset: int
    data_nbytes: int
    scale_offset: int
    scale_nbytes: int
    data_sha256: str
    scale_sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Int8PartLocation":
        return cls(
            part=value["part"],
            shape=tuple(value["shape"]),
            quant_axis=int(value["quant_axis"]),
            data_offset=int(value["data_offset"]),
            data_nbytes=int(value["data_nbytes"]),
            scale_offset=int(value["scale_offset"]),
            scale_nbytes=int(value["scale_nbytes"]),
            data_sha256=value["data_sha256"],
            scale_sha256=value["scale_sha256"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "part": self.part,
            "shape": list(self.shape),
            "quant_axis": self.quant_axis,
            "data_dtype": "I8",
            "scale_dtype": "F16",
            "data_offset": self.data_offset,
            "data_nbytes": self.data_nbytes,
            "scale_offset": self.scale_offset,
            "scale_nbytes": self.scale_nbytes,
            "data_sha256": self.data_sha256,
            "scale_sha256": self.scale_sha256,
        }


@dataclass(frozen=True)
class Int8ExpertLocation:
    layer: int
    expert_id: int
    file: Path
    parts: tuple[Int8PartLocation, ...]

    @property
    def nbytes(self) -> int:
        return sum(item.data_nbytes + item.scale_nbytes for item in self.parts)

    def part(self, name: str) -> Int8PartLocation:
        for item in self.parts:
            if item.part == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class Int8RowSumLocation:
    layer: int
    expert_id: int
    part: str
    file: Path
    offset: int
    nbytes: int
    sha256: str
    rows: int

    @classmethod
    def from_dict(cls, root: Path, value: dict[str, Any]) -> "Int8RowSumLocation":
        return cls(
            layer=int(value["layer"]),
            expert_id=int(value["expert_id"]),
            part=value["part"],
            file=root / value["file"],
            offset=int(value["offset"]),
            nbytes=int(value["nbytes"]),
            sha256=value["sha256"],
            rows=int(value["rows"]),
        )


class Int8ExpertIndex:
    def __init__(self, root: Path, manifest: dict[str, Any], entries: Iterable[dict[str, Any]]):
        self.root = root
        self.manifest = manifest
        self._entries: dict[tuple[int, int], Int8ExpertLocation] = {}
        for entry in entries:
            key = (int(entry["layer"]), int(entry["expert_id"]))
            if key in self._entries:
                raise ValueError(f"duplicate INT8 expert entry: {key}")
            self._entries[key] = Int8ExpertLocation(
                layer=key[0],
                expert_id=key[1],
                file=root / entry["file"],
                parts=tuple(Int8PartLocation.from_dict(item) for item in entry["parts"]),
            )
        self.execution_manifest: dict[str, Any] | None = None
        self._row_sums: dict[tuple[int, int, str], Int8RowSumLocation] = {}

    @classmethod
    def from_dir(cls, root: str | Path) -> "Int8ExpertIndex":
        path = Path(root).expanduser().resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        if manifest.get("format_id") != FORMAT_ID or manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported INT8 container format: {manifest.get('format_id')}")
        if _sha256_file(path / "index.json") != manifest.get("index_sha256"):
            raise ValueError("INT8 container index checksum mismatch")
        result = cls(path, manifest, index["entries"])
        for layer in result.layers:
            result._validate_layer_header(result.locate(layer, 0).file, layer)
        result._load_execution_metadata()
        return result

    @property
    def has_offline_row_sums(self) -> bool:
        return bool(self._row_sums)

    @property
    def execution_files(self) -> tuple[Path, ...]:
        return tuple(sorted({location.file for location in self._row_sums.values()}))

    def row_sum_location(
        self,
        layer: int,
        expert_id: int,
        part: str,
    ) -> Int8RowSumLocation | None:
        return self._row_sums.get((layer, expert_id, part))

    def read_row_sums(self, layer: int, expert_id: int, part: str) -> bytes | None:
        location = self.row_sum_location(layer, expert_id, part)
        if location is None:
            return None
        with location.file.open("rb", buffering=0) as stream:
            stream.seek(location.offset)
            payload = stream.read(location.nbytes)
        if len(payload) != location.nbytes:
            raise OSError(f"short INT8 row-sum read: {location.file}")
        if hashlib.sha256(payload).hexdigest() != location.sha256:
            raise ValueError(
                f"INT8 row-sum checksum mismatch: layer={layer}, expert={expert_id}, part={part}"
            )
        return payload

    def _load_execution_metadata(self) -> None:
        manifest_path = self.root / EXEC_MANIFEST_NAME
        index_path = self.root / EXEC_INDEX_NAME
        if not manifest_path.exists() and not index_path.exists():
            return
        if not manifest_path.is_file() or not index_path.is_file():
            raise ValueError("incomplete INT8 execution metadata")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format_id") != EXEC_FORMAT_ID
            or int(manifest.get("format_version", -1)) != EXEC_FORMAT_VERSION
        ):
            raise ValueError("unsupported INT8 execution metadata format")
        if manifest.get("weight_index_sha256") != self.manifest.get("index_sha256"):
            raise ValueError("INT8 execution metadata belongs to another weight index")
        if _sha256_file(index_path) != manifest.get("index_sha256"):
            raise ValueError("INT8 execution metadata index checksum mismatch")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        for value in payload.get("row_sums", []):
            location = Int8RowSumLocation.from_dict(self.root, value)
            key = (location.layer, location.expert_id, location.part)
            if key in self._row_sums:
                raise ValueError(f"duplicate INT8 row-sum entry: {key}")
            self._row_sums[key] = location
        expected = len(self._entries) * len(PARTS)
        if len(self._row_sums) != expected:
            raise ValueError(
                f"incomplete INT8 row-sum index: expected {expected}, got {len(self._row_sums)}"
            )
        self.execution_manifest = manifest

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({key[0] for key in self._entries}))

    @property
    def num_experts(self) -> int:
        return int(self.manifest["num_experts"])

    def locate(self, layer: int, expert_id: int) -> Int8ExpertLocation:
        try:
            return self._entries[(layer, expert_id)]
        except KeyError as exc:
            raise ValueError(f"INT8 expert is not indexed: layer={layer}, expert={expert_id}") from exc

    def read(
        self,
        layer: int,
        expert_id: int,
        verify: bool = True,
    ) -> dict[str, dict[str, bytes]]:
        location = self.locate(layer, expert_id)
        result = {}
        with location.file.open("rb") as stream:
            for part in location.parts:
                stream.seek(part.data_offset)
                data = stream.read(part.data_nbytes)
                stream.seek(part.scale_offset)
                scales = stream.read(part.scale_nbytes)
                if len(data) != part.data_nbytes or len(scales) != part.scale_nbytes:
                    raise OSError(f"short INT8 container read: {location.file}")
                if verify and (
                    hashlib.sha256(data).hexdigest() != part.data_sha256
                    or hashlib.sha256(scales).hexdigest() != part.scale_sha256
                ):
                    raise ValueError(
                        f"INT8 expert checksum mismatch: layer={layer}, expert={expert_id}, "
                        f"part={part.part}"
                    )
                result[part.part] = {"data": data, "scales": scales}
        return result

    def _validate_layer_header(self, path: Path, expected_layer: int) -> None:
        with path.open("rb") as stream:
            raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"short INT8 layer header: {path}")
        magic, version, layer, num_experts, alignment = HEADER.unpack(raw)
        if (
            magic != MAGIC
            or version != FORMAT_VERSION
            or layer != expected_layer
            or num_experts != self.num_experts
            or alignment != ALIGNMENT
        ):
            raise ValueError(f"invalid INT8 layer header: {path}")


def convert_experts_int8(
    model_dir: str | Path,
    output_dir: str | Path,
    layers: Iterable[int] | None = None,
    resume: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import torch

    model = Path(model_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    layer_dir = output / "layers"
    layer_dir.mkdir(exist_ok=True)
    locator = ExpertLocator(model)
    selected_layers = locator.layers if layers is None else tuple(sorted(set(int(x) for x in layers)))
    unknown = sorted(set(selected_layers) - set(locator.layers))
    if unknown:
        raise ValueError(f"unknown routed expert layers: {unknown}")

    source = _source_identity(model)
    all_entries: list[dict[str, Any]] = []
    converted_layers = 0
    resumed_layers = 0
    logical_bytes = 0
    started = datetime.now(timezone.utc)
    with ShardReader() as reader:
        for layer in selected_layers:
            final_data = layer_dir / f"layer-{layer:03d}.sfi"
            final_index = layer_dir / f"layer-{layer:03d}.json"
            if resume and final_data.is_file() and final_index.is_file():
                layer_meta = json.loads(final_index.read_text(encoding="utf-8"))
                _validate_layer_metadata(layer_meta, source, layer, locator.num_experts, final_data)
                entries = layer_meta["entries"]
                resumed_layers += 1
            else:
                entries = _convert_layer(
                    locator,
                    reader,
                    layer,
                    final_data,
                    final_index,
                    source,
                    torch,
                )
                converted_layers += 1
            all_entries.extend(entries)
            logical_bytes += sum(
                int(part["data_nbytes"]) + int(part["scale_nbytes"])
                for entry in entries
                for part in entry["parts"]
            )
            if progress is not None:
                progress(
                    {
                        "layer": layer,
                        "layers_complete": len({item["layer"] for item in all_entries}),
                        "layers_total": len(selected_layers),
                        "experts_complete": len(all_entries),
                        "resumed": resume and final_index.is_file() and converted_layers == 0,
                    }
                )

    entries_expected = len(selected_layers) * locator.num_experts
    if len(all_entries) != entries_expected:
        raise RuntimeError(f"INT8 index incomplete: expected {entries_expected}, got {len(all_entries)}")
    index = {
        "schema_version": 1,
        "format_id": FORMAT_ID,
        "entries": all_entries,
    }
    _write_json_atomic(output / "index.json", index)
    physical_bytes = sum(path.stat().st_size for path in sorted(layer_dir.glob("*.sfi")))
    manifest = {
        "schema_version": 1,
        "format_id": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "quantization": {
            "weights": "per-output-channel symmetric int8",
            "data_dtype": "I8",
            "scale_dtype": "F16",
            "quant_axis": 0,
            "rounding": "torch.round",
            "clamp": [-127, 127],
            "zero_row_scale": 1.0,
        },
        "layout": "expert-major canonical row-major",
        "alignment": ALIGNMENT,
        "layers": list(selected_layers),
        "num_layers": len(selected_layers),
        "num_experts": locator.num_experts,
        "entries": len(all_entries),
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "index_sha256": _sha256_file(output / "index.json"),
    }
    _write_json_atomic(output / "manifest.json", manifest)
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "schema_version": 1,
        "kind": "sparseflow_int8_conversion",
        "agent": "Main Dev",
        "output": str(output),
        "format_id": FORMAT_ID,
        "layers": len(selected_layers),
        "experts": len(all_entries),
        "converted_layers": converted_layers,
        "resumed_layers": resumed_layers,
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "source_bf16_expert_bytes": sum(
            locator.locate(layer, expert_id).nbytes
            for layer in selected_layers
            for expert_id in range(locator.num_experts)
        ),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
        "manifest": manifest,
    }


def build_int8_execution_metadata(
    container_dir: str | Path,
    resume: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build optional offline row sums without rewriting canonical INT8 weights."""

    import torch

    root = Path(container_dir).expanduser().resolve()
    index = Int8ExpertIndex.from_dir(root)
    output = root / "execution"
    output.mkdir(exist_ok=True)
    all_entries: list[dict[str, Any]] = []
    converted_layers = 0
    resumed_layers = 0
    started = datetime.now(timezone.utc)
    for layer in index.layers:
        data_path = output / f"layer-{layer:03d}.sfx"
        meta_path = output / f"layer-{layer:03d}.json"
        if resume and data_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            _validate_execution_layer_metadata(
                meta,
                index.manifest["index_sha256"],
                layer,
                index.num_experts,
                data_path,
            )
            entries = meta["row_sums"]
            resumed_layers += 1
        else:
            entries = _build_execution_layer(index, layer, data_path, meta_path, torch)
            converted_layers += 1
        all_entries.extend(entries)
        if progress is not None:
            progress(
                {
                    "layer": layer,
                    "layers_complete": converted_layers + resumed_layers,
                    "layers_total": len(index.layers),
                    "row_sums_complete": len(all_entries),
                }
            )

    expected = len(index.layers) * index.num_experts * len(PARTS)
    if len(all_entries) != expected:
        raise RuntimeError(
            f"INT8 execution index incomplete: expected {expected}, got {len(all_entries)}"
        )
    execution_index = {
        "schema_version": 1,
        "format_id": EXEC_FORMAT_ID,
        "row_sums": all_entries,
    }
    _write_json_atomic(root / EXEC_INDEX_NAME, execution_index)
    physical_bytes = sum(path.stat().st_size for path in output.glob("*.sfx"))
    manifest = {
        "schema_version": 1,
        "format_id": EXEC_FORMAT_ID,
        "format_version": EXEC_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "weight_format_id": index.manifest["format_id"],
        "weight_index_sha256": index.manifest["index_sha256"],
        "features": ["offline-row-sums"],
        "row_sum_dtype": "I32",
        "alignment": ALIGNMENT,
        "layers": list(index.layers),
        "num_experts": index.num_experts,
        "entries": len(all_entries),
        "physical_bytes": physical_bytes,
        "index_sha256": _sha256_file(root / EXEC_INDEX_NAME),
    }
    _write_json_atomic(root / EXEC_MANIFEST_NAME, manifest)
    return {
        "schema_version": 1,
        "kind": "sparseflow_int8_execution_metadata",
        "agent": "Main Dev",
        "container": str(root),
        "format_id": EXEC_FORMAT_ID,
        "converted_layers": converted_layers,
        "resumed_layers": resumed_layers,
        "entries": len(all_entries),
        "physical_bytes": physical_bytes,
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "peak_rss_bytes": peak_rss_bytes(),
        "manifest": manifest,
    }


def _build_execution_layer(index, layer, final_data, final_index, torch):
    temp_data = final_data.with_suffix(final_data.suffix + ".tmp")
    source_file = index.locate(layer, 0).file
    entries = []
    with source_file.open("rb") as source, temp_data.open("wb") as stream:
        mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            header = EXEC_HEADER.pack(
                EXEC_MAGIC,
                EXEC_FORMAT_VERSION,
                layer,
                index.num_experts,
                ALIGNMENT,
            )
            stream.write(header)
            stream.write(b"\0" * (ALIGNMENT - len(header)))
            for expert_id in range(index.num_experts):
                location = index.locate(layer, expert_id)
                for part in location.parts:
                    view = memoryview(mapped)[
                        part.data_offset : part.data_offset + part.data_nbytes
                    ]
                    weight = torch.frombuffer(view, dtype=torch.int8).reshape(part.shape)
                    payload = (
                        weight.sum(dim=1, dtype=torch.int32)
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
                    del weight, view
                    _align_stream(stream)
                    offset = stream.tell()
                    stream.write(payload)
                    entries.append(
                        {
                            "layer": layer,
                            "expert_id": expert_id,
                            "part": part.part,
                            "file": f"execution/{final_data.name}",
                            "offset": offset,
                            "nbytes": len(payload),
                            "rows": part.shape[0],
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    )
            _align_stream(stream)
            stream.flush()
        finally:
            mapped.close()
    temp_data.replace(final_data)
    meta = {
        "schema_version": 1,
        "format_id": EXEC_FORMAT_ID,
        "format_version": EXEC_FORMAT_VERSION,
        "weight_index_sha256": index.manifest["index_sha256"],
        "layer": layer,
        "num_experts": index.num_experts,
        "file": final_data.name,
        "file_bytes": final_data.stat().st_size,
        "file_sha256": _sha256_file(final_data),
        "row_sums": entries,
    }
    _write_json_atomic(final_index, meta)
    return entries


def _validate_execution_layer_metadata(
    meta,
    weight_index_sha256,
    layer,
    num_experts,
    data_file,
) -> None:
    if (
        meta.get("format_id") != EXEC_FORMAT_ID
        or int(meta.get("format_version", -1)) != EXEC_FORMAT_VERSION
        or meta.get("weight_index_sha256") != weight_index_sha256
    ):
        raise ValueError(f"incompatible resumable execution metadata: {data_file}")
    if int(meta.get("layer", -1)) != layer or int(meta.get("num_experts", -1)) != num_experts:
        raise ValueError(f"execution metadata layer mismatch: {data_file}")
    if data_file.stat().st_size != int(meta.get("file_bytes", -1)):
        raise ValueError(f"execution metadata size mismatch: {data_file}")
    if _sha256_file(data_file) != meta.get("file_sha256"):
        raise ValueError(f"execution metadata checksum mismatch: {data_file}")


def dequantize_part(location: Int8PartLocation, payload: dict[str, bytes], torch):
    quantized = torch.frombuffer(bytearray(payload["data"]), dtype=torch.int8).reshape(
        location.shape
    )
    scales = torch.frombuffer(bytearray(payload["scales"]), dtype=torch.float16)
    return quantized.float() * scales.float().unsqueeze(1)


def _convert_layer(locator, reader, layer, final_data, final_index, source, torch):
    temp_data = final_data.with_suffix(final_data.suffix + ".tmp")
    entries = []
    with temp_data.open("wb") as stream:
        header = HEADER.pack(MAGIC, FORMAT_VERSION, layer, locator.num_experts, ALIGNMENT)
        stream.write(header)
        stream.write(b"\0" * (ALIGNMENT - len(header)))
        for expert_id in range(locator.num_experts):
            location = locator.locate(layer, expert_id)
            payloads = reader.read_expert_into(location)
            parts = []
            for part_name in PARTS:
                source_part = location.part(part_name)
                if source_part.dtype != "BF16" or len(source_part.expert_shape) != 2:
                    raise ValueError(
                        f"INT8 converter requires 2D BF16 experts: {source_part.tensor_name}"
                    )
                data, scales = _quantize_bf16(
                    payloads[part_name], source_part.expert_shape, torch
                )
                _align_stream(stream)
                data_offset = stream.tell()
                stream.write(data)
                _align_stream(stream)
                scale_offset = stream.tell()
                stream.write(scales)
                parts.append(
                    Int8PartLocation(
                        part=part_name,
                        shape=source_part.expert_shape,
                        quant_axis=0,
                        data_offset=data_offset,
                        data_nbytes=len(data),
                        scale_offset=scale_offset,
                        scale_nbytes=len(scales),
                        data_sha256=hashlib.sha256(data).hexdigest(),
                        scale_sha256=hashlib.sha256(scales).hexdigest(),
                    ).as_dict()
                )
            entries.append(
                {
                    "layer": layer,
                    "expert_id": expert_id,
                    "file": f"layers/{final_data.name}",
                    "parts": parts,
                }
            )
        _align_stream(stream)
        stream.flush()
    temp_data.replace(final_data)
    layer_meta = {
        "schema_version": 1,
        "format_id": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "source": source,
        "layer": layer,
        "num_experts": locator.num_experts,
        "file": final_data.name,
        "file_bytes": final_data.stat().st_size,
        "file_sha256": _sha256_file(final_data),
        "entries": entries,
    }
    _write_json_atomic(final_index, layer_meta)
    return entries


def _quantize_bf16(payload: bytes | bytearray, shape: tuple[int, int], torch):
    expected = shape[0] * shape[1] * 2
    if len(payload) != expected:
        raise ValueError(f"BF16 payload size mismatch: expected {expected}, got {len(payload)}")
    weight = torch.frombuffer(payload, dtype=torch.bfloat16).reshape(shape).float()
    max_abs = weight.abs().amax(dim=1)
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 127.0)
    quantized = torch.round(weight / scales.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
    return (
        quantized.contiguous().numpy().tobytes(),
        scales.to(torch.float16).contiguous().numpy().tobytes(),
    )


def _align_stream(stream) -> None:
    padding = (-stream.tell()) % ALIGNMENT
    if padding:
        stream.write(b"\0" * padding)


def _source_identity(model: Path) -> dict[str, Any]:
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    return {
        "model_name": model.name,
        "config_sha256": _sha256_file(config),
        "index_sha256": _sha256_file(index) if index.is_file() else None,
    }


def _validate_layer_metadata(meta, source, layer, num_experts, data_file) -> None:
    if meta.get("format_id") != FORMAT_ID or meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"incompatible resumable layer metadata: {data_file}")
    if meta.get("source") != source or int(meta.get("layer", -1)) != layer:
        raise ValueError(f"resumable layer source mismatch: {data_file}")
    if int(meta.get("num_experts", -1)) != num_experts:
        raise ValueError(f"resumable layer expert count mismatch: {data_file}")
    if data_file.stat().st_size != int(meta.get("file_bytes", -1)):
        raise ValueError(f"resumable layer size mismatch: {data_file}")
    if _sha256_file(data_file) != meta.get("file_sha256"):
        raise ValueError(f"resumable layer checksum mismatch: {data_file}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


# [Main Dev]

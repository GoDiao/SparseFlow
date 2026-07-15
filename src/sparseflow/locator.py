from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .analyze import load_config
from .classifier import classifier_for_model
from .safetensors import ShardIndex, TensorSpan


class ExpertLocatorError(ValueError):
    """Raised when an expert cannot be mapped to a contiguous tensor slice."""


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def dtype_nbytes(dtype: str) -> int:
    try:
        return _DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ExpertLocatorError(f"unsupported safetensors dtype for slicing: {dtype}") from exc


@dataclass(frozen=True)
class ExpertSlice:
    """One contiguous expert slice inside a safetensors tensor."""

    layer: int
    expert_id: int
    part: str
    tensor_name: str
    shard: Path
    dtype: str
    tensor_shape: tuple[int, ...]
    expert_shape: tuple[int, ...]
    expert_axis: int
    element_offset: int
    element_count: int
    file_offset: int
    nbytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "expert_id": self.expert_id,
            "part": self.part,
            "tensor_name": self.tensor_name,
            "shard": str(self.shard),
            "dtype": self.dtype,
            "tensor_shape": list(self.tensor_shape),
            "expert_shape": list(self.expert_shape),
            "expert_axis": self.expert_axis,
            "element_offset": self.element_offset,
            "element_count": self.element_count,
            "file_offset": self.file_offset,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class ExpertLocation:
    """All routed-weight slices needed for one logical expert."""

    layer: int
    expert_id: int
    parts: tuple[ExpertSlice, ...]

    @property
    def nbytes(self) -> int:
        return sum(item.nbytes for item in self.parts)

    def part(self, name: str) -> ExpertSlice:
        for item in self.parts:
            if item.part == name:
                return item
        raise KeyError(name)

    def __iter__(self) -> Iterable[ExpertSlice]:
        return iter(self.parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "expert_id": self.expert_id,
            "parts": [item.as_dict() for item in self.parts],
        }


class ExpertLocator:
    """Map ``(layer, expert_id)`` to contiguous fused-tensor byte ranges.

    The first implementation supports the Qwen3.6 fused layout exposed by
    :class:`Qwen36MoeClassifier`. It deliberately returns metadata only; a
    loader can later use ``file_offset`` and ``nbytes`` with ``pread`` without
    loading the surrounding tensor.
    """

    def __init__(self, model_dir: str | Path, index: ShardIndex | None = None):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.config = load_config(self.model_dir)
        self.index = index or ShardIndex.from_dir(self.model_dir)
        self.classifier = classifier_for_model(self.model_dir, self.config.model_type)
        self.num_experts = self._read_num_experts()
        self._parts: dict[int, dict[str, TensorSpan]] = {}
        self._index_routed_parts()

    def _read_num_experts(self) -> int:
        config = self.config.text_config
        value = config.get("num_experts", config.get("n_routed_experts", 0))
        try:
            result = int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ExpertLocatorError(f"invalid expert count in config: {value!r}") from exc
        if result <= 0:
            raise ExpertLocatorError("model config does not define a positive expert count")
        return result

    def _index_routed_parts(self) -> None:
        for tensor in self.index:
            tensor_class = self.classifier.classify(tensor)
            if tensor_class.category != "routed_experts" or tensor_class.layer is None:
                continue
            parts = self._parts.setdefault(tensor_class.layer, {})
            part = tensor_class.expert_part or tensor.name
            if part in parts:
                previous = parts[part]
                raise ExpertLocatorError(
                    f"duplicate routed expert part {part!r} at layer {tensor_class.layer}: "
                    f"{previous.name} and {tensor.name}"
                )
            parts[part] = tensor

    @property
    def layers(self) -> tuple[int, ...]:
        """Return layers with indexed routed-expert tensors."""

        return tuple(sorted(self._parts))

    def fused_parts(self, layer: int) -> Mapping[str, TensorSpan]:
        """Expose the fused routed tensors backing one layer as read-only metadata."""

        try:
            return dict(self._parts[layer])
        except KeyError as exc:
            raise ExpertLocatorError(f"no routed expert tensors found for layer {layer}") from exc

    def fused_part(self, layer: int, part: str) -> TensorSpan:
        """Return one fused routed tensor used by the resident provider."""

        try:
            return self._parts[layer][part]
        except KeyError as exc:
            raise ExpertLocatorError(
                f"no routed expert part {part!r} found for layer {layer}"
            ) from exc

    def locate(self, layer: int, expert_id: int) -> ExpertLocation:
        if layer < 0:
            raise ExpertLocatorError(f"layer must be non-negative: {layer}")
        if layer not in self._parts:
            raise ExpertLocatorError(f"no routed expert tensors found for layer {layer}")
        if not 0 <= expert_id < self.num_experts:
            raise ExpertLocatorError(
                f"expert id {expert_id} is outside [0, {self.num_experts})"
            )

        slices = tuple(
            self._locate_part(layer, expert_id, part, tensor)
            for part, tensor in sorted(self._parts[layer].items())
        )
        if not slices:
            raise ExpertLocatorError(f"no routed expert parts found for layer {layer}")
        return ExpertLocation(layer=layer, expert_id=expert_id, parts=slices)

    def _locate_part(
        self,
        layer: int,
        expert_id: int,
        part: str,
        tensor: TensorSpan,
    ) -> ExpertSlice:
        if not tensor.shape:
            raise ExpertLocatorError(f"routed tensor is scalar and cannot contain experts: {tensor.name}")
        if tensor.numel <= 0:
            raise ExpertLocatorError(f"routed tensor has no elements: {tensor.name}")

        axes = [axis for axis, size in enumerate(tensor.shape) if size == self.num_experts]
        if len(axes) != 1:
            raise ExpertLocatorError(
                f"cannot infer a unique expert axis for {tensor.name}: "
                f"shape={tensor.shape}, num_experts={self.num_experts}"
            )
        expert_axis = axes[0]
        if expert_axis != 0:
            raise ExpertLocatorError(
                f"non-contiguous expert layout is not supported yet for {tensor.name}: "
                f"expert axis={expert_axis}, shape={tensor.shape}"
            )

        expert_shape = tuple(tensor.shape[1:])
        element_count = math.prod(expert_shape) if expert_shape else 1
        element_offset = expert_id * element_count
        item_bytes = dtype_nbytes(tensor.dtype)
        expected_nbytes = tensor.numel * item_bytes
        if expected_nbytes != tensor.nbytes:
            raise ExpertLocatorError(
                f"tensor byte size does not match dtype/shape for {tensor.name}: "
                f"header={tensor.nbytes}, expected={expected_nbytes}"
            )

        file_offset = tensor.data_offset + element_offset * item_bytes
        nbytes = element_count * item_bytes
        payload_end = tensor.data_offset + tensor.nbytes
        if file_offset < tensor.data_offset or file_offset + nbytes > payload_end:
            raise ExpertLocatorError(
                f"calculated expert slice is outside tensor payload for {tensor.name}: "
                f"offset={file_offset}, bytes={nbytes}, payload=[{tensor.data_offset}, {payload_end})"
            )

        return ExpertSlice(
            layer=layer,
            expert_id=expert_id,
            part=part,
            tensor_name=tensor.name,
            shard=tensor.shard,
            dtype=tensor.dtype,
            tensor_shape=tensor.shape,
            expert_shape=expert_shape,
            expert_axis=expert_axis,
            element_offset=element_offset,
            element_count=element_count,
            file_offset=file_offset,
            nbytes=nbytes,
        )

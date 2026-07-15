from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import model_snapshot, write_json


def expected_dat_bytes(dtype: str, shape: list[int]) -> int:
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    widths = {
        "BF16": 2,
        "F16": 2,
        "F32": 4,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    try:
        return elements * widths[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype for generic offload: {dtype}") from exc


def prepare_offload(model_dir: str | Path, offload_dir: str | Path) -> dict[str, Any]:
    from accelerate.utils import offload_weight, save_offload_index
    from safetensors import safe_open

    model = Path(model_dir).resolve()
    output = Path(offload_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_index_path = model / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = source_index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    index: dict[str, dict[str, Any]] = {}
    resumed = 0
    written = 0
    written_bytes = 0
    started = time.perf_counter()
    for shard_name in sorted(by_shard):
        shard_path = model / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as tensors:
            for name in sorted(by_shard[shard_name]):
                span = tensors.get_slice(name)
                dtype = span.get_dtype()
                shape = list(span.get_shape())
                nbytes = expected_dat_bytes(dtype, shape)
                target = output / f"{name}.dat"
                accelerate_dtype = "bfloat16" if dtype == "BF16" else {
                    "F16": "float16",
                    "F32": "float32",
                    "I64": "int64",
                    "I32": "int32",
                    "I16": "int16",
                    "I8": "int8",
                    "U8": "uint8",
                    "BOOL": "bool",
                }[dtype]
                index[name] = {"dtype": accelerate_dtype, "shape": shape}
                if target.is_file() and target.stat().st_size == nbytes:
                    resumed += 1
                    written_bytes += nbytes
                    continue
                tensor = tensors.get_tensor(name)
                offload_weight(tensor, name, output)
                del tensor
                if target.stat().st_size != nbytes:
                    raise RuntimeError(
                        f"generic offload size mismatch for {name}: "
                        f"{target.stat().st_size} != {nbytes}"
                    )
                written += 1
                written_bytes += nbytes
                if written % 20 == 0:
                    print(
                        f"converted={written + resumed}/{len(weight_map)} "
                        f"bytes={written_bytes}",
                        flush=True,
                    )
    save_offload_index(index, output)
    report = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_4_generic_offload_layout",
        "stage": "7.4",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model),
        "offload_dir": str(output),
        "tensors": len(index),
        "bytes": written_bytes,
        "written_tensors": written,
        "resumed_tensors": resumed,
        "seconds": time.perf_counter() - started,
        "index_complete": len(index) == len(weight_map),
    }
    write_json(output / "sparseflow_prepare_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert safetensors into Accelerate generic disk-offload files."
    )
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--offload-dir", default=".cache/stage7_4/generic-offload")
    args = parser.parse_args(argv)
    report = prepare_offload(args.model, args.offload_dir)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["index_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

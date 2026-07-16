from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .common import git_snapshot, host_snapshot, model_snapshot, write_json


SAMPLES = ((0, 0), (0, 255), (19, 127), (39, 0), (39, 255))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a SparseFlow INT8 expert container.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--conversion-report", required=True)
    parser.add_argument("--resume-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import torch

    from sparseflow.int8_container import ALIGNMENT, Int8ExpertIndex, dequantize_part
    from sparseflow.loader import ShardReader
    from sparseflow.locator import ExpertLocator

    root = Path.cwd().resolve()
    model = Path(args.model).expanduser().resolve()
    container = Path(args.container).expanduser().resolve()
    conversion = json.loads(Path(args.conversion_report).read_text(encoding="utf-8"))
    resume = json.loads(Path(args.resume_report).read_text(encoding="utf-8"))
    index = Int8ExpertIndex.from_dir(container)
    locator = ExpertLocator(model)

    entries = [index.locate(layer, expert) for layer in index.layers for expert in range(index.num_experts)]
    alignment_exact = all(
        part.data_offset % ALIGNMENT == 0 and part.scale_offset % ALIGNMENT == 0
        for entry in entries
        for part in entry.parts
    )
    sample_records = []
    max_abs_error = 0.0
    mean_abs_errors = []
    with ShardReader() as reader:
        for layer, expert in SAMPLES:
            int8_location = index.locate(layer, expert)
            payload = index.read(layer, expert, verify=True)
            source_location = locator.locate(layer, expert)
            source_payload = reader.read_expert_into(source_location)
            parts = {}
            for part_name in ("gate_up_proj", "down_proj"):
                source_part = source_location.part(part_name)
                expected = torch.frombuffer(
                    source_payload[part_name], dtype=torch.bfloat16
                ).reshape(source_part.expert_shape).float()
                actual = dequantize_part(
                    int8_location.part(part_name), payload[part_name], torch
                )
                difference = (actual - expected).abs()
                maximum = float(difference.max())
                mean = float(difference.mean())
                max_abs_error = max(max_abs_error, maximum)
                mean_abs_errors.append(mean)
                parts[part_name] = {
                    "shape": list(actual.shape),
                    "max_abs_error": maximum,
                    "mean_abs_error": mean,
                    "finite": bool(torch.isfinite(actual).all()),
                }
            sample_records.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "logical_bytes": int8_location.nbytes,
                    "parts": parts,
                }
            )

    source_bytes = int(conversion["source_bf16_expert_bytes"])
    physical_bytes = int(conversion["physical_bytes"])
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_int8_container_audit",
        "stage": "7.5.2",
        "agent": "Main Dev",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "model": model_snapshot(model),
        "container": {
            "path": str(container),
            "manifest": index.manifest,
            "layer_files": len(list((container / "layers").glob("*.sfi"))),
            "indexed_entries": len(entries),
            "physical_to_bf16_ratio": physical_bytes / source_bytes,
        },
        "conversion": conversion,
        "resume": resume,
        "samples": sample_records,
        "sample_summary": {
            "max_abs_error": max_abs_error,
            "mean_abs_error": sum(mean_abs_errors) / len(mean_abs_errors),
        },
        "acceptance": {
            "format_exact": index.manifest["format_id"] == "canonical-int8-v1",
            "layers_exact": len(index.layers) == 40,
            "entries_exact": len(entries) == 40 * 256,
            "layer_files_exact": len(list((container / "layers").glob("*.sfi"))) == 40,
            "alignment_exact": alignment_exact,
            "storage_at_most_51_percent_of_bf16": physical_bytes <= source_bytes * 0.51,
            "conversion_peak_rss_below_1_gib": int(conversion["peak_rss_bytes"]) < 1024**3,
            "resume_verified_all_layers": (
                int(resume["converted_layers"]) == 0 and int(resume["resumed_layers"]) == 40
            ),
            "sample_checksums_and_finite": all(
                part["finite"]
                for sample in sample_records
                for part in sample["parts"].values()
            ),
            "sample_max_abs_error_below_0_01": max_abs_error < 0.01,
        },
    }
    result["acceptance"]["all_pass"] = all(result["acceptance"].values())
    write_json(Path(args.output).expanduser(), result)
    print(json.dumps(result["sample_summary"], ensure_ascii=False))
    print(json.dumps(result["acceptance"], ensure_ascii=False))
    print(f"results={Path(args.output).expanduser().resolve()}")
    return 0 if result["acceptance"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

"""Capture the reproducible, dependency-light Stage 7.11 environment record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

from benchmarks.common import git_snapshot
from sparseflow.process_metrics import host_snapshot, process_snapshot
from sparseflow.release import container_identity, cpu_features, model_identity


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment(
    *,
    root: Path,
    model: Path | None = None,
    container: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_environment",
        "stage": "7.11.0",
        "agent": "Benchmark",
        "host": host_snapshot(),
        "process": process_snapshot(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "sparseflow": _package_version("sparseflow"),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "safetensors": _package_version("safetensors"),
            "accelerate": _package_version("accelerate"),
        },
        "cpu": cpu_features(),
        "git": git_snapshot(root),
        "repository": {"root": str(root.resolve())},
        "scope": {
            "model_payloads_included": False,
            "container_payloads_included": False,
            "formal_result": False,
        },
    }
    disk = shutil.disk_usage(root)
    result["disk"] = {
        "path": str(root.resolve()),
        "free_bytes": int(disk.free),
        "total_bytes": int(disk.total),
    }
    if model is not None:
        result["model"] = model_identity(model)
    if container is not None:
        result["container"] = container_identity(container)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Stage 7.11 environment metadata.")
    parser.add_argument("--model")
    parser.add_argument("--int8-container")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result = capture_environment(
        root=root,
        model=Path(args.model).expanduser().resolve() if args.model else None,
        container=Path(args.int8_container).expanduser().resolve() if args.int8_container else None,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "commit": result["git"].get("commit")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

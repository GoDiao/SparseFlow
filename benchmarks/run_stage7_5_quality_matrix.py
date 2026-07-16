from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .common import git_snapshot, write_json


BACKENDS = (
    "bf16-reference",
    "int8-reference",
    "int8-native",
    "int8-native-streaming",
)
LEVELS = ("smoke", "pilot", "formal")


def cells() -> tuple[tuple[str, str], ...]:
    return tuple((level, backend) for level in LEVELS for backend in BACKENDS)


def command_for(
    level: str,
    backend: str,
    model: Path,
    container: Path,
    manifest_dir: Path,
    output: Path,
    threads: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.score_choices",
        "--model",
        str(model),
        "--data",
        str(manifest_dir / f"quality_{level}_v1.jsonl"),
        "--output",
        str(output),
        "--backend",
        backend,
        "--threads",
        str(threads),
        "--choice-execution",
        "batch",
    ]
    if backend.startswith("int8-"):
        command.extend(("--int8-container", str(container)))
    if backend == "int8-native-streaming":
        command.extend(("--cache-bytes", "4GiB"))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.5.6 quality ladder.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest-dir", default="benchmarks/manifests")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--levels", help="Optional comma-separated level filter.")
    parser.add_argument("--backends", help="Optional comma-separated backend filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    model = Path(args.model).expanduser()
    model = (root / model).resolve() if not model.is_absolute() else model.resolve()
    container = Path(args.int8_container).expanduser()
    container = (root / container).resolve() if not container.is_absolute() else container.resolve()
    manifest_dir = Path(args.manifest_dir).expanduser()
    manifest_dir = (
        (root / manifest_dir).resolve() if not manifest_dir.is_absolute() else manifest_dir.resolve()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_6_quality_matrix_execution",
        "stage": "7.5.6",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(root),
        "model": str(model),
        "int8_container": str(container),
        "manifest_dir": str(manifest_dir),
        "threads": args.threads,
        "choice_execution": "batch",
        "cells": [],
    }
    execution = output_dir / "matrix_execution.json"
    env = dict(os.environ)
    source = str(root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    selected_levels = set(args.levels.split(",")) if args.levels else set(LEVELS)
    selected_backends = set(args.backends.split(",")) if args.backends else set(BACKENDS)
    unknown_levels = selected_levels - set(LEVELS)
    unknown_backends = selected_backends - set(BACKENDS)
    if unknown_levels or unknown_backends:
        parser.error(
            f"unknown levels/backends: {sorted(unknown_levels)}, {sorted(unknown_backends)}"
        )
    matrix = tuple(
        (level, backend)
        for level, backend in cells()
        if level in selected_levels and backend in selected_backends
    )
    for index, (level, backend) in enumerate(matrix, 1):
        output = output_dir / f"{level}-{backend}.json"
        command = command_for(
            level,
            backend,
            model,
            container,
            manifest_dir,
            output,
            args.threads,
        )
        record: dict[str, Any] = {
            "level": level,
            "backend": backend,
            "output": str(output),
            "command": command,
        }
        if args.resume and output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("summary", {}).get("n"):
                    record["status"] = "skipped-complete"
                    result["cells"].append(record)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if args.dry_run:
            record["status"] = "dry-run"
            result["cells"].append(record)
            print(" ".join(command))
            continue

        print(f"cell={index}/{len(matrix)} level={level} backend={backend}", flush=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_seconds,
                check=False,
            )
            record["seconds"] = time.perf_counter() - started
            record["returncode"] = completed.returncode
            record["log_tail"] = completed.stdout[-8000:]
            record["status"] = "complete" if completed.returncode == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            record["seconds"] = time.perf_counter() - started
            record["returncode"] = None
            text = exc.stdout or ""
            record["log_tail"] = text[-8000:] if isinstance(text, str) else ""
            record["status"] = "timeout"
        result["cells"].append(record)
        write_json(execution, result)

    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    result["all_complete"] = all(
        item["status"] in {"complete", "skipped-complete", "dry-run"}
        for item in result["cells"]
    )
    write_json(execution, result)
    return 0 if result["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

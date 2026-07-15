from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import git_snapshot, write_json


GIB = 1024**3


@dataclass(frozen=True)
class MatrixCell:
    variant: str
    cache_bytes: int
    cache_state: str
    warmup: int
    runs: int
    replicate: int | None = None

    @property
    def cell_id(self) -> str:
        budget = f"{self.cache_bytes // GIB}g" if self.cache_bytes else "0g"
        suffix = f"-r{self.replicate}" if self.replicate is not None else ""
        return f"{self.variant.lower()}-{budget}-{self.cache_state}{suffix}"


def core_matrix() -> tuple[MatrixCell, ...]:
    cells = [
        MatrixCell("C3-R", 0, "workload-warm", warmup=1, runs=3),
        MatrixCell("C3-S0", 0, "workload-warm", warmup=1, runs=3),
    ]
    for variant in ("C3-S1", "C3-S2", "C3-S3", "C3-S4"):
        for budget in (1, 2, 4, 8):
            cells.append(
                MatrixCell(variant, budget * GIB, "workload-warm", warmup=1, runs=3)
            )
    for variant, budget in (("C3-S0", 0), ("C3-S3", 4 * GIB), ("C3-S4", 4 * GIB)):
        for replicate in range(3):
            cells.append(
                MatrixCell(
                    variant,
                    budget,
                    "model-cold",
                    warmup=0,
                    runs=1,
                    replicate=replicate,
                )
            )
    return tuple(cells)


def command_for(
    cell: MatrixCell,
    root: Path,
    model: Path,
    manifest: Path,
    output: Path,
    threads: int,
    max_new_tokens: int,
    telemetry_level: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.run_sparseflow",
        "--model",
        str(model),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--variant",
        cell.variant,
        "--cache-bytes",
        str(cell.cache_bytes),
        "--cache-state",
        cell.cache_state,
        "--warmup",
        str(cell.warmup),
        "--runs",
        str(cell.runs),
        "--threads",
        str(threads),
        "--max-new-tokens",
        str(max_new_tokens),
        "--limit",
        "1",
        "--telemetry-level",
        telemetry_level,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 7.4 C3 matrix.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--telemetry-level", choices=("none", "summary", "layer"), default="summary")
    parser.add_argument("--variants", help="Optional comma-separated variant filter.")
    parser.add_argument("--states", help="Optional comma-separated cache-state filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    model = (root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model).resolve()
    manifest = (
        (root / args.manifest).resolve()
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest).resolve()
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = set(args.variants.split(",")) if args.variants else None
    states = set(args.states.split(",")) if args.states else None
    cells = tuple(
        cell
        for cell in core_matrix()
        if (variants is None or cell.variant in variants)
        and (states is None or cell.cache_state in states)
    )
    if not cells:
        parser.error("matrix filters selected no cells")

    manifest_result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_4_matrix_execution",
        "stage": "7.4",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(root),
        "model": str(model),
        "workload_manifest": str(manifest),
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "timeout_seconds": args.timeout_seconds,
        "cells": [],
    }
    execution_path = output_dir / "matrix_execution.json"

    env = dict(os.environ)
    source = str(root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for index, cell in enumerate(cells, 1):
        output = output_dir / f"{cell.cell_id}.json"
        command = command_for(
            cell,
            root,
            model,
            manifest,
            output,
            args.threads,
            args.max_new_tokens,
            args.telemetry_level,
        )
        record: dict[str, Any] = {
            **asdict(cell),
            "cell_id": cell.cell_id,
            "output": str(output),
            "command": command,
        }
        if args.resume and output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if "summary" in existing:
                    record["status"] = "skipped-complete"
                    manifest_result["cells"].append(record)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if args.dry_run:
            record["status"] = "dry-run"
            manifest_result["cells"].append(record)
            print(" ".join(command))
            continue

        print(f"cell={index}/{len(cells)} id={cell.cell_id}", flush=True)
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
            output_text = exc.stdout or ""
            record["log_tail"] = output_text[-8000:] if isinstance(output_text, str) else ""
            record["status"] = "timeout"
        manifest_result["cells"].append(record)
        write_json(execution_path, manifest_result)

    manifest_result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_result["all_complete"] = all(
        cell["status"] in {"complete", "skipped-complete", "dry-run"}
        for cell in manifest_result["cells"]
    )
    write_json(execution_path, manifest_result)
    return 0 if manifest_result["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

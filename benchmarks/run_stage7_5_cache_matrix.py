from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from .common import git_snapshot, write_json


GIB = 1024**3


@dataclass(frozen=True)
class CacheCell:
    variant: str
    cache_bytes: int
    cache_state: str = "workload-warm"
    warmup: int = 1
    runs: int = 1
    max_new_tokens: int = 32
    prefetch_workers: int | None = None
    coalesce_gap: int = 0
    tag: str = "core"

    @property
    def cell_id(self) -> str:
        budget = f"{self.cache_bytes // 1024**2}m"
        workers = f"-w{self.prefetch_workers}" if self.prefetch_workers is not None else ""
        gap = f"-gap{self.coalesce_gap}" if self.coalesce_gap else ""
        return (
            f"{self.tag}-{self.variant.lower()}-{budget}-{self.cache_state}"
            f"-o{self.max_new_tokens}{workers}{gap}"
        )


def calibration_matrix() -> tuple[CacheCell, ...]:
    cells = [CacheCell("C3-S0", 0)]
    for variant in ("C3-S1", "C3-S3", "C3-S4"):
        for budget_mib in (512, 1024, 2048, 4096, 8192):
            cells.append(CacheCell(variant, budget_mib * 1024**2))
    for variant, budget in (
        ("C3-S0", 0),
        ("C3-S1", 4 * GIB),
        ("C3-S3", 4 * GIB),
        ("C3-S4", 4 * GIB),
    ):
        cells.append(
            CacheCell(
                variant,
                budget,
                cache_state="model-cold",
                warmup=0,
                tag="cold",
            )
        )
    cells.extend(
        (
            CacheCell("C3-S4", GIB, prefetch_workers=1, tag="io"),
            CacheCell("C3-S4", GIB, prefetch_workers=4, tag="io"),
            CacheCell("C3-S4", GIB, prefetch_workers=2, coalesce_gap=4096, tag="io"),
            CacheCell("C3-S4", GIB, prefetch_workers=2, coalesce_gap=65536, tag="io"),
        )
    )
    for tokens in (8, 16):
        cells.extend(
            (
                CacheCell("C3-S1", GIB, max_new_tokens=tokens, tag="length"),
                CacheCell("C3-S4", GIB, max_new_tokens=tokens, tag="length"),
            )
        )
    return tuple(cells)


def command_for(
    cell: CacheCell,
    model: Path,
    container: Path,
    manifest: Path,
    output: Path,
    threads: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_sparseflow",
        "--model",
        str(model),
        "--int8-container",
        str(container),
        "--expert-storage",
        "int8-native",
        "--stage",
        "7.5.5",
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
        str(cell.max_new_tokens),
        "--limit",
        "1",
        "--coalesce-gap",
        str(cell.coalesce_gap),
        "--telemetry-level",
        "summary",
    ]
    if cell.prefetch_workers is not None:
        command.extend(("--prefetch-workers", str(cell.prefetch_workers)))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.5.5 INT8 cache calibration.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--tags", help="Optional comma-separated core/cold/io/length filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.timeout_seconds < 1:
        parser.error("threads and timeout must be positive")

    root = Path.cwd().resolve()
    model = Path(args.model).expanduser()
    model = (root / model).resolve() if not model.is_absolute() else model.resolve()
    container = Path(args.int8_container).expanduser()
    container = (root / container).resolve() if not container.is_absolute() else container.resolve()
    manifest = Path(args.manifest).expanduser()
    manifest = (root / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = set(args.tags.split(",")) if args.tags else None
    cells = tuple(cell for cell in calibration_matrix() if tags is None or cell.tag in tags)
    if not cells:
        parser.error("matrix filters selected no cells")

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_5_cache_matrix_execution",
        "stage": "7.5.5",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(root),
        "model": str(model),
        "int8_container": str(container),
        "manifest": str(manifest),
        "threads": args.threads,
        "cells": [],
    }
    execution = output_dir / "matrix_execution.json"
    env = dict(os.environ)
    source = str(root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for index, cell in enumerate(cells, 1):
        output = output_dir / f"{cell.cell_id}.json"
        command = command_for(cell, model, container, manifest, output, args.threads)
        record: dict[str, Any] = {
            **asdict(cell),
            "cell_id": cell.cell_id,
            "output": str(output),
            "command": command,
        }
        if args.resume and output.is_file():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if existing.get("summary", {}).get("run_count"):
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

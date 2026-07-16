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
class FormalCell:
    expert_storage: str
    variant: str
    cache_bytes: int
    cache_state: str
    warmup: int
    runs: int
    replicate: int | None = None

    @property
    def cell_id(self) -> str:
        storage = self.expert_storage.replace("int8-", "i8-")
        budget = f"{self.cache_bytes // GIB}g" if self.cache_bytes else "0g"
        replicate = f"-r{self.replicate}" if self.replicate is not None else ""
        return f"{storage}-{self.variant.lower()}-{budget}-{self.cache_state}{replicate}"


def formal_matrix() -> tuple[FormalCell, ...]:
    cells = [
        FormalCell("bf16", "C3-R", 0, "workload-warm", 1, 3),
        FormalCell("bf16", "C3-S1", 8 * GIB, "workload-warm", 1, 3),
        FormalCell("int8-reference", "C3-R", 0, "workload-warm", 1, 3),
        FormalCell("int8-reference", "C3-S1", 4 * GIB, "workload-warm", 1, 3),
        FormalCell("int8-native", "C3-R", 0, "workload-warm", 1, 3),
        FormalCell("int8-native", "C3-S1", 4 * GIB, "workload-warm", 1, 3),
        FormalCell("int8-native", "C3-S1", 8 * GIB, "workload-warm", 1, 3),
    ]
    cells.extend(
        FormalCell(
            "int8-native",
            "C3-S1",
            4 * GIB,
            "model-cold",
            0,
            1,
            replicate,
        )
        for replicate in range(3)
    )
    return tuple(cells)


def command_for(
    cell: FormalCell,
    model: Path,
    container: Path,
    manifest: Path,
    output: Path,
    threads: int,
    max_new_tokens: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.run_sparseflow",
        "--model",
        str(model),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--expert-storage",
        cell.expert_storage,
        "--stage",
        "7.5.6",
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
        "summary",
    ]
    if cell.expert_storage.startswith("int8-"):
        command.extend(("--int8-container", str(container)))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Stage 7.5.6 performance matrix.")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path.cwd().resolve()
    model = Path(args.model).expanduser()
    model = (root / model).resolve() if not model.is_absolute() else model.resolve()
    container = Path(args.int8_container).expanduser()
    container = (root / container).resolve() if not container.is_absolute() else container.resolve()
    manifest = Path(args.manifest).expanduser()
    manifest = (root / manifest).resolve() if not manifest.is_absolute() else manifest.resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = formal_matrix()
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_5_6_formal_matrix_execution",
        "stage": "7.5.6",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_snapshot(root),
        "model": str(model),
        "int8_container": str(container),
        "manifest": str(manifest),
        "threads": args.threads,
        "max_new_tokens": args.max_new_tokens,
        "cells": [],
    }
    execution = output_dir / "matrix_execution.json"
    env = dict(os.environ)
    source = str(root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for index, cell in enumerate(cells, 1):
        output = output_dir / f"{cell.cell_id}.json"
        command = command_for(
            cell,
            model,
            container,
            manifest,
            output,
            args.threads,
            args.max_new_tokens,
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

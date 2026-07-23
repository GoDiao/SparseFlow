"""Run the read-only Stage 7.11 cache/context admission pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmarks.common import git_snapshot
from sparseflow.process_metrics import host_snapshot
from sparseflow.release import container_identity, model_identity


def _parse_bytes(value: str) -> int:
    text = value.strip().lower()
    suffixes = {
        "gib": 1024**3,
        "mib": 1024**2,
        "kib": 1024,
        "gb": 1000**3,
        "mb": 1000**2,
        "kb": 1000,
        "b": 1,
    }
    for suffix, multiplier in suffixes.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(float(text))


def run_pilot(
    *,
    model: Path,
    container: Path,
    cache_values: list[str],
    context_tokens: int,
    available_ram: str | None = None,
    check_native: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(root / "src")
    for cache in cache_values:
        command = [
            sys.executable,
            "-m",
            "sparseflow",
            "doctor",
            str(model),
            "--preset",
            "laptop-16gb",
            "--int8-container",
            str(container),
            "--cache-bytes",
            cache,
            "--ctx",
            str(context_tokens),
            "--batch-size",
            "1",
            "--json",
        ]
        if available_ram:
            command.extend(["--available-ram", available_ram])
        if check_native:
            command.append("--check-native")
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"doctor did not emit JSON for cache={cache}: {completed.stdout[-1000:]}"
            ) from exc
        memory = report.get("memory") or {}
        components = memory.get("components") or {}
        rows.append(
            {
                "cache_bytes": _parse_bytes(cache),
                "exit_code": completed.returncode,
                "ready": bool(report.get("ready")),
                "memory_status": memory.get("status"),
                "available_ram_bytes": int(memory.get("available_ram_bytes") or 0),
                "required_ram_bytes": int(memory.get("required_ram_bytes") or 0),
                "recommended_ram_bytes": int(memory.get("recommended_ram_bytes") or 0),
                "headroom_bytes": int(memory.get("headroom_bytes") or 0),
                "context_tokens": int(memory.get("context_tokens") or context_tokens),
                "batch_size": int(memory.get("batch_size") or 0),
                "streaming_cache_bytes": int(components.get("streaming_cache_bytes") or 0),
                "resident_int8_expert_bytes": int(components.get("resident_int8_expert_bytes") or 0),
                "cpu_avx512_vnni": bool((report.get("cpu") or {}).get("avx512_vnni")),
                "native_extension_status": next(
                    (
                        item.get("status")
                        for item in report.get("checks", [])
                        if item.get("id") == "native_extension"
                    ),
                    None,
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_cache_context_pilot",
        "stage": "7.11.3",
        "agent": "Benchmark",
        "protocol": {
            "preset": "laptop-16gb",
            "mode": "streaming",
            "context_tokens": context_tokens,
            "batch_size": 1,
            "cache_values": [_parse_bytes(value) for value in cache_values],
            "read_only": True,
            "check_native": check_native,
        },
        "host": host_snapshot(),
        "git": git_snapshot(root),
        "model": model_identity(model),
        "container": container_identity(container),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.11 cache/context Doctor pilot.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--cache-bytes", action="append")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--available-ram")
    parser.add_argument("--check-native", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cache_values = args.cache_bytes or ["128MiB", "256MiB", "512MiB"]
    if args.context_tokens < 1 or any(_parse_bytes(value) < 1 for value in cache_values):
        parser.error("context-tokens and cache-bytes must be positive")
    result = run_pilot(
        model=Path(args.model).expanduser().resolve(),
        container=Path(args.int8_container).expanduser().resolve(),
        cache_values=cache_values,
        context_tokens=args.context_tokens,
        available_ram=args.available_ram,
        check_native=args.check_native,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(result["rows"]), "ready": [row["ready"] for row in result["rows"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

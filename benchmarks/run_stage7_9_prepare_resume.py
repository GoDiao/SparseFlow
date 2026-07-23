"""Re-run the one-layer prepare/resume audit with file-level evidence.

The audit uses a fresh ignored output directory.  It records the completed
layer's digest and mtime immediately before and after the resume invocation,
plus the converter reports and temporary-file inventory.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": digest(path),
        }
    return result


def run(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resumable one-layer INT8 preparation.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    python = str(Path(args.python).expanduser().resolve())
    model = str(Path(args.model).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    evidence_path = Path(args.evidence).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    first_report = output_dir.parent / f"{output_dir.name}-first.json"
    resume_report = output_dir.parent / f"{output_dir.name}-resume.json"
    command = [python, "-m", "sparseflow", "prepare-int8", model, "--output", str(output_dir), "--layers", "0", "--json"]
    first_run = run(command + ["--report", str(first_report)])
    first_data = json.loads(first_report.read_text(encoding="utf-8")) if first_report.exists() else {}
    before_resume = inventory(output_dir)
    second_run = run(command + ["--report", str(resume_report)])
    resume_data = json.loads(resume_report.read_text(encoding="utf-8")) if resume_report.exists() else {}
    after_resume = inventory(output_dir)
    temporary_files = [
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tmp", ".partial", ".part"}
    ]
    gates = {
        "first_returncode_zero": first_run["returncode"] == 0,
        "resume_returncode_zero": second_run["returncode"] == 0,
        "first_converted_one_layer": first_data.get("conversion", {}).get("converted_layers") == 1,
        "resume_converted_zero": resume_data.get("conversion", {}).get("converted_layers") == 0,
        "resume_detected_one_layer": resume_data.get("conversion", {}).get("resumed_layers") == 1,
        "completed_layer_hash_and_mtime_unchanged": before_resume == after_resume,
        "weight_manifest_unchanged": first_data.get("conversion", {}).get("manifest", {}).get("index_sha256") == resume_data.get("conversion", {}).get("manifest", {}).get("index_sha256"),
        "execution_manifest_unchanged": first_data.get("execution", {}).get("manifest", {}).get("index_sha256") == resume_data.get("execution", {}).get("manifest", {}).get("index_sha256"),
        "disk_preflight_pass": bool(first_data.get("disk", {}).get("pass")) and bool(resume_data.get("disk", {}).get("pass")),
        "no_temporary_files": not temporary_files,
    }
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_prepare_resume_evidence",
        "stage": "7.9-evidence-closure",
        "agent": "Main Dev",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_clean": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() == "",
        "model": model,
        "output_dir": str(output_dir),
        "first_report": str(first_report),
        "resume_report": str(resume_report),
        "before_resume": before_resume,
        "after_resume": after_resume,
        "temporary_files": temporary_files,
        "first_run": {k: v for k, v in first_run.items() if k not in {"stdout", "stderr"}},
        "resume_run": {k: v for k, v in second_run.items() if k not in {"stdout", "stderr"}},
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(evidence_path), "all_gates_pass": result["all_gates_pass"], "gates": gates}))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build and verify the native extension from a new cache directory.

This turns the former empty-cache statement into a subprocess artifact with a
before/after file inventory and return-code gate.

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


def inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    if not root.exists():
        return result
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SparseFlow native extension from an empty cache.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    python = str(Path(args.python).expanduser().resolve())
    model = str(Path(args.model).expanduser().resolve())
    container = str(Path(args.int8_container).expanduser().resolve())
    cache = Path(args.cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    before = inventory(cache)
    command = [
        python,
        "-m",
        "sparseflow",
        "doctor",
        model,
        "--preset",
        "low-memory",
        "--int8-container",
        container,
        "--check-native",
        "--json",
    ]
    env = os.environ.copy()
    env["SPARSEFLOW_NATIVE_CACHE"] = str(cache)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=Path.cwd(), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    after = inventory(cache)
    gates = {
        "cache_empty_before": len(before) == 0,
        "returncode_success": completed.returncode == 0,
        "native_shared_object_present": any(item["path"].endswith(".so") for item in after),
        "cache_populated_after": len(after) > 0,
        "no_traceback": "Traceback (most recent call last)" not in completed.stdout + completed.stderr,
    }
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_native_build_evidence",
        "stage": "7.9-evidence-closure",
        "agent": "Main Dev",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_clean": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() == "",
        "cache": str(cache),
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
        "before": before,
        "after": after,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_gates_pass": result["all_gates_pass"], "gates": gates}))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

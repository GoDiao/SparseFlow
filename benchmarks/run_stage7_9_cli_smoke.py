"""Record reproducible base-install CLI smoke evidence for Stage 7.9.

Each command is executed as a subprocess and the gate is derived from its
return code and captured output.  This intentionally uses the no-Torch base
environment; runtime commands must fail cleanly at the optional dependency
boundary rather than produce a traceback.

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


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_command(command: list[str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        timed_out = True
    combined = stdout + "\n" + stderr
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout_sha256": digest(stdout),
        "stderr_sha256": digest(stderr),
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "traceback": "Traceback (most recent call last)" in combined,
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record Stage 7.9 base CLI smoke evidence.")
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    python = str(Path(args.python).expanduser().resolve())
    model = str(Path(args.model).expanduser().resolve())
    container = str(Path(args.int8_container).expanduser().resolve())
    base = [python, "-m", "sparseflow"]
    torch_probe = subprocess.run(
        [python, "-c", "import importlib.util; print(importlib.util.find_spec('torch') is not None)"],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    torch_visible = torch_probe.returncode == 0 and torch_probe.stdout.strip() == "True"
    commands = {
        "preset": base + ["preset", "--json"],
        "inspect": base + ["inspect", model, "--json"],
        "plan": base + ["plan", model, "--ram", "96", "--ctx", "4096", "--json"],
        "doctor_low_memory": base + [
            "doctor", model, "--preset", "low-memory", "--int8-container", container, "--json"
        ],
        "run_clean_optional_dependency_error": base + [
            "run", model, "--preset", "low-memory", "--int8-container", container,
            "--prompt", "test", "--max-new-tokens", "1", "--json"
        ],
    }
    records = {name: run_command(command, 120) for name, command in commands.items()}
    gates = {
        "preset": records["preset"]["returncode"] == 0 and not records["preset"]["traceback"],
        "inspect": records["inspect"]["returncode"] == 0 and not records["inspect"]["traceback"],
        "plan": records["plan"]["returncode"] in (0, 1) and not records["plan"]["traceback"],
        "doctor_low_memory": records["doctor_low_memory"]["returncode"] in (0, 1) and not records["doctor_low_memory"]["traceback"],
        "run_optional_error": records["run_clean_optional_dependency_error"]["returncode"] == 2
        and not records["run_clean_optional_dependency_error"]["traceback"],
        "torch_hidden": not torch_visible,
    }
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_cli_smoke_evidence",
        "stage": "7.9-evidence-closure",
        "agent": "Main Dev",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_clean": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() == "",
        "environment": {
            "python": python,
            "torch_visible": torch_visible,
            "install": "uv pip install -e . --no-deps",
        },
        "model": model,
        "int8_container": container,
        "commands": records,
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

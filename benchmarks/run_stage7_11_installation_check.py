"""Record the Stage 7.11 Windows installation replay checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from benchmarks.common import git_snapshot


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_installation(root: Path, *, clean_replay: bool, run_tests: bool = False) -> dict[str, Any]:
    python_path = Path(sys.executable).resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    preset = subprocess.run(
        [sys.executable, "-m", "sparseflow", "preset", "laptop-16gb", "--json"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    preset_ok = False
    preset_value: dict[str, Any] | None = None
    try:
        preset_value = json.loads(preset.stdout)
        preset_ok = preset.returncode == 0 and preset_value.get("preset", {}).get("name") == "laptop-16gb"
    except json.JSONDecodeError:
        pass

    torch_cuda = None
    torch_import_ok = False
    try:
        import torch

        torch_import_ok = True
        torch_cuda = bool(torch.cuda.is_available())
    except (ImportError, OSError):
        pass

    scripts = {
        name: (root / "scripts" / name).is_file()
        for name in ("bootstrap_uv_windows.ps1", "setup_windows.ps1", "use_runtime_windows.ps1")
    }
    cache_values = {
        name: os.environ.get(name)
        for name in ("UV_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "TEMP", "TMP", "SPARSEFLOW_NATIVE_CACHE")
        if os.environ.get(name)
    }
    cache_on_project_drive = all(_under(Path(value), root) for value in cache_values.values())
    test_result: dict[str, Any] | None = None
    if run_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        test_result = {
            "return_code": completed.returncode,
            "passed": completed.returncode == 0,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    checks = {
        "python_on_project_drive": _under(python_path, root),
        "runtime_import": torch_import_ok and _version("transformers") is not None,
        "cpu_torch": torch_import_ok and torch_cuda is False,
        "preset_cli": preset_ok,
        "setup_scripts": all(scripts.values()),
        "cache_paths_on_project_drive": cache_on_project_drive,
        "clean_replay_declared": clean_replay,
    }
    if run_tests:
        checks["full_test_suite"] = bool(test_result and test_result["passed"])
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_installation_verification",
        "stage": "7.11.7",
        "agent": "Board",
        "git": git_snapshot(root),
        "python": {
            "executable": str(python_path),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": {
            "torch": _version("torch"),
            "transformers": _version("transformers"),
            "safetensors": _version("safetensors"),
            "accelerate": _version("accelerate"),
        },
        "preset": preset_value,
        "torch_cuda_available": torch_cuda,
        "cache_paths": cache_values,
        "setup_scripts": scripts,
        "checks": checks,
        "clean_replay": clean_replay,
        "test_suite": test_result,
        "passed": all(checks.values()),
        "notes": [
            "This artifact verifies the isolated Python/setup replay, not model RAM admission.",
            "Doctor and real generation remain separate Stage 7.11 gates.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record Stage 7.11 installation checks.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clean-replay", action="store_true")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    result = check_installation(root, clean_replay=args.clean_replay, run_tests=args.run_tests)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

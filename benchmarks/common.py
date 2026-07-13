from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def now() -> float:
    return time.perf_counter()


def read_proc_file(path: str) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if not _:
                continue
            value = value.strip().split()[0] if value.strip() else "0"
            try:
                result[key] = int(value)
            except ValueError:
                continue
    except OSError:
        pass
    return result


def process_snapshot() -> dict[str, int]:
    status = read_proc_file("/proc/self/status")
    io = read_proc_file("/proc/self/io")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "rss_bytes": status.get("VmRSS", 0) * 1024,
        "peak_rss_bytes": status.get("VmHWM", 0) * 1024,
        "read_bytes": io.get("read_bytes", 0),
        "read_chars": io.get("rchar", 0),
        "read_syscalls": io.get("syscr", 0),
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
        "voluntary_context_switches": int(usage.ru_nvcsw),
        "involuntary_context_switches": int(usage.ru_nivcsw),
    }


def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in after.items():
        old = before.get(key, 0)
        result[key] = value - old
    return result


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def git_snapshot(root: str | Path) -> dict[str, Any]:
    root = str(root)

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def model_snapshot(model_dir: str | Path) -> dict[str, Any]:
    root = Path(model_dir).resolve()
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    return {
        "path": str(root),
        "config_sha256": sha256_file(config) if config.is_file() else None,
        "index_sha256": sha256_file(index) if index.is_file() else None,
        "config": json.loads(config.read_text(encoding="utf-8")) if config.is_file() else None,
    }


def host_snapshot() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

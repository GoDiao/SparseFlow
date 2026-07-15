from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import shutil
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
    memory = read_proc_file("/proc/meminfo")
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
        "memory_total_bytes": memory.get("MemTotal", 0) * 1024,
        "memory_available_bytes": memory.get("MemAvailable", 0) * 1024,
        "numa_nodes": _read_text("/sys/devices/system/node/online"),
        "pid": os.getpid(),
    }


def filesystem_snapshot(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    usage = shutil.disk_usage(root)
    return {
        "path": str(root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def numeric_delta(before: Any, after: Any) -> Any:
    """Subtract nested numeric counters while preserving after-state metadata."""

    if isinstance(before, dict) and isinstance(after, dict):
        return {
            key: numeric_delta(before.get(key), value)
            for key, value in after.items()
        }
    if (
        not isinstance(before, bool)
        and not isinstance(after, bool)
        and isinstance(before, (int, float))
        and isinstance(after, (int, float))
    ):
        return after - before
    return after


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_bytes(value: str) -> int:
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
            number = text[: -len(suffix)].strip()
            break
    else:
        number = text
        multiplier = 1
    try:
        result = int(float(number) * multiplier)
    except ValueError as exc:
        raise ValueError(f"invalid byte size: {value}") from exc
    if result < 0:
        raise ValueError("byte size must be non-negative")
    return result


def evict_file_pages(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Best-effort model-local page-cache eviction using POSIX_FADV_DONTNEED."""

    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advice is None or not hasattr(os, "posix_fadvise"):
        return {"supported": False, "files": 0, "bytes": 0, "failures": []}
    files = 0
    total_bytes = 0
    failures = []
    for value in paths:
        path = Path(value)
        try:
            with path.open("rb", buffering=0) as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, advice)
            files += 1
            total_bytes += path.stat().st_size
        except OSError as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return {
        "supported": True,
        "files": files,
        "bytes": total_bytes,
        "failures": failures,
    }


def model_weight_files(model_dir: str | Path) -> tuple[Path, ...]:
    return tuple(sorted(Path(model_dir).resolve().glob("*.safetensors")))


def _read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# Stage 7.4 benchmark utilities: [Main Dev]

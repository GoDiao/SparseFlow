"""Capture independent model-cold I/O evidence for Stage 7.9.

The parent process launches one fresh ``sparseflow run`` process per sample and
polls that child through procfs.  ``provider_demand_read_bytes`` is the logical
SparseFlow counter; ``process_physical_read_bytes`` is Linux ``read_bytes``
from ``/proc/<pid>/io`` and is kept separate.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Any


POSIX_FADV_DONTNEED = 4


def parse_proc_file(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if not value:
                continue
            token = value.strip().split()[0]
            try:
                values[key] = int(token)
            except ValueError:
                continue
    except (FileNotFoundError, PermissionError):
        pass
    return values


def proc_io(pid: int) -> dict[str, int] | None:
    path = Path(f"/proc/{pid}/io")
    values = parse_proc_file(path)
    return values or None


def proc_status(pid: int) -> dict[str, int] | None:
    values = parse_proc_file(Path(f"/proc/{pid}/status"))
    if not values:
        return None
    return {
        "rss_bytes": values.get("VmRSS", 0) * 1024,
        "hwm_bytes": values.get("VmHWM", 0) * 1024,
    }


def command_text(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def block_devices() -> list[dict[str, Any]]:
    value = command_text("lsblk", "-ndo", "NAME,ROTA,TYPE")
    devices = []
    for line in value.splitlines():
        fields = line.split(None, 2)
        if not fields:
            continue
        name = fields[0]
        devices.append({
            "name": name,
            "model": command_text("lsblk", "-ndo", "MODEL", f"/dev/{name}") or None,
            "rotational": int(fields[1]) if len(fields) > 1 and fields[1] in {"0", "1"} else None,
            "type": fields[2] if len(fields) > 2 else None,
        })
    return devices


def storage_identity(path: Path) -> dict[str, Any]:
    mount = command_text("findmnt", "-T", str(path), "-no", "SOURCE,FSTYPE,TARGET")
    fields = mount.split()
    source = fields[0] if fields else None
    filesystem = fields[1] if len(fields) > 1 else None
    target = fields[2] if len(fields) > 2 else None
    block_device = source if source and source.startswith("/") else None
    block_model = None
    rotational = None
    if block_device:
        block_model = command_text("lsblk", "-no", "MODEL", block_device) or None
        rota = command_text("lsblk", "-no", "ROTA", block_device)
        rotational = int(rota) if rota in {"0", "1"} else None
    return {
        "path": str(path),
        "mount_source": source,
        "filesystem": filesystem,
        "mount_target": target,
        "block_device": block_device,
        "block_device_model": block_model,
        "rotational": rotational,
        "statfs_type": command_text("stat", "-f", "-c", "%T", str(path)) or None,
        "block_devices_visible": block_devices(),
    }


def evict_file_pages(root: Path) -> dict[str, int]:
    files = 0
    bytes_seen = 0
    failures = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files += 1
        try:
            bytes_seen += path.stat().st_size
            fd = os.open(path, os.O_RDONLY)
            try:
                result = os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
                if result not in (None, 0):
                    failures += 1
            finally:
                os.close(fd)
        except (OSError, ValueError):
            failures += 1
    return {"files": files, "bytes": bytes_seen, "failures": failures}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample(
    index: int,
    model: Path,
    container: Path,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
    python: str,
    threads: int,
) -> dict[str, Any]:
    eviction = {
        "model": evict_file_pages(model),
        "container": evict_file_pages(container),
    }
    output = output_dir / f"run_{index}.json"
    log_path = output_dir / f"run_{index}.log"
    command = [
        python,
        "-m",
        "sparseflow",
        "run",
        str(model),
        "--preset",
        "low-memory",
        "--int8-container",
        str(container),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--output",
        str(output),
        "--json",
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "SPARSEFLOW_NATIVE_CACHE": "/root/workspace/cache/release-native",
    })
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        io_before = proc_io(process.pid) or {}
        io_last = dict(io_before)
        peak_rss = 0
        peak_hwm = 0
        while process.poll() is None:
            current_io = proc_io(process.pid)
            if current_io:
                io_last = current_io
            status = proc_status(process.pid) or {}
            peak_rss = max(peak_rss, status.get("rss_bytes", 0))
            peak_hwm = max(peak_hwm, status.get("hwm_bytes", 0))
            time.sleep(0.05)
        status = proc_status(process.pid) or {}
        peak_rss = max(peak_rss, status.get("rss_bytes", 0))
        peak_hwm = max(peak_hwm, status.get("hwm_bytes", 0))
        # procfs entries disappear as soon as the child exits.  Use the last
        # sampled counters instead of silently turning a completed run into a
        # missing physical-I/O measurement.
        io_after = proc_io(process.pid) or io_last
    elapsed = time.perf_counter() - started
    result = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    run = result.get("result", {})
    provider = run.get("provider_storage") or {}
    log_digest = sha256_file(log_path) if log_path.exists() else None
    return {
        "index": index,
        "fresh_process": True,
        "pid": process.pid,
        "command": command,
        "returncode": process.returncode,
        "elapsed_seconds": elapsed,
        "page_cache_eviction": eviction,
        "proc_io_before": io_before,
        "proc_io_after": io_after,
        "process_physical_read_bytes": (
            io_after.get("read_bytes", 0) - io_before.get("read_bytes", 0)
            if io_before and io_after
            else None
        ),
        "process_rchar_bytes": (
            io_after.get("rchar", 0) - io_before.get("rchar", 0)
            if io_before and io_after
            else None
        ),
        "process_read_syscalls": (
            io_after.get("syscr", 0) - io_before.get("syscr", 0)
            if io_before and io_after
            else None
        ),
        "peak_rss_bytes_sampled": peak_rss or None,
        "peak_hwm_bytes_sampled": peak_hwm or None,
        "log_sha256": log_digest,
        "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "output": {
            "returncode": process.returncode,
            "prefill_seconds": run.get("prefill_seconds"),
            "decode_seconds": run.get("decode_seconds"),
            "decode_token_seconds": run.get("decode_token_seconds", []),
            "generated_tokens": run.get("generated_tokens"),
            "provider_logical_read_bytes": provider.get("demand_read_bytes"),
            "provider_loaded_bytes": provider.get("loaded_bytes"),
            "cache": run.get("cache"),
            "memory": run.get("memory"),
            "runtime_identity": run.get("runtime_identity"),
            "model_identity": result.get("model"),
            "container_identity": result.get("container"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Stage 7.9 model-cold physical I/O.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=".venv-release/bin/python")
    parser.add_argument("--prompt", default="Explain sparse expert routing in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        sample(i, model, container, output_dir, args.prompt, args.max_new_tokens, args.python, args.threads)
        for i in range(1, args.samples + 1)
    ]
    storage = {"model": storage_identity(model), "container": storage_identity(container)}
    result = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_cold_io_evidence",
        "stage": "7.9-evidence-closure",
        "agent": "Main Dev",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_clean": subprocess.check_output(["git", "status", "--porcelain"], text=True).strip() == "",
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpus": os.cpu_count(),
            "threads": args.threads,
        },
        "model": str(model),
        "int8_container": str(container),
        "protocol": {
            "independent_processes": args.samples,
            "fresh_process_per_sample": True,
            "page_cache_eviction": "POSIX_FADV_DONTNEED on every regular model/container file before each child",
            "physical_counter": "Linux /proc/<pid>/io read_bytes delta",
            "logical_counter": "SparseFlow provider demand_read_bytes",
            "max_new_tokens": args.max_new_tokens,
        },
        "storage": storage,
        "samples": samples,
        "gates": {
            "sample_count": len(samples) == args.samples,
            "all_fresh_processes": all(x["fresh_process"] for x in samples),
            "all_returncode_zero": all(x["returncode"] == 0 for x in samples),
            "all_fadvise_calls_ok": all(
                section["failures"] == 0
                for sample_item in samples
                for section in sample_item["page_cache_eviction"].values()
            ),
            "physical_read_captured": all(x["process_physical_read_bytes"] is not None for x in samples),
            "storage_identity_captured": all(
                storage[name].get("filesystem") or storage[name].get("statfs_type")
                for name in ("model", "container")
            ),
        },
    }
    result["all_gates_pass"] = all(result["gates"].values())
    output = output_dir / "cold_io_matrix.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_gates_pass": result["all_gates_pass"], "gates": result["gates"]}))
    return 0 if result["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

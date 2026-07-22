"""Public-alpha release contracts and read-only readiness checks.

This module deliberately does not load model weights.  It validates the files,
CPU capabilities, INT8 metadata, and preset contract before the runtime starts.

[Main Dev]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
from typing import Any

from .analyze import analyze_model, load_config
from .int8_container import Int8ExpertIndex
from .safetensors import ShardIndex


@dataclass(frozen=True)
class RuntimePreset:
    name: str
    description: str
    mode: str
    load_mode: str
    expert_storage: str
    native_dispatch: str
    cache_policy: str
    cache_bytes: int | None
    prefetch_workers: int
    prefetch_policy: str
    batch_mode: str
    public_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, RuntimePreset] = {
    "stable": RuntimePreset(
        name="stable",
        description="INT8 native resident Qwen3.6 hybrid runtime.",
        mode="resident",
        load_mode="memory-native",
        expert_storage="int8-native",
        native_dispatch="hybrid",
        cache_policy="none",
        cache_bytes=None,
        prefetch_workers=0,
        prefetch_policy="none",
        batch_mode="single",
        public_status="stable",
    ),
    "low-memory": RuntimePreset(
        name="low-memory",
        description="INT8 native single-request streaming with S1 LRU.",
        mode="streaming",
        load_mode="memory-native",
        expert_storage="int8-native",
        native_dispatch="hybrid",
        cache_policy="lru",
        cache_bytes=4 * 1024**3,
        prefetch_workers=0,
        prefetch_policy="none",
        batch_mode="single",
        public_status="stable-low-memory",
    ),
    "experimental-batch": RuntimePreset(
        name="experimental-batch",
        description="INT8 native resident grouped fixed-cohort runtime.",
        mode="resident",
        load_mode="memory-native",
        expert_storage="int8-native",
        native_dispatch="grouped",
        cache_policy="none",
        cache_bytes=None,
        prefetch_workers=0,
        prefetch_policy="none",
        batch_mode="fixed-cohort",
        public_status="experimental-opt-in",
    ),
}


def get_preset(name: str) -> RuntimePreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {choices}") from exc


def apply_preset(
    name: str,
    *,
    cache_bytes: int | None = None,
    telemetry_level: str = "summary",
) -> dict[str, Any]:
    """Return the serializable runtime configuration used by a public preset."""

    preset = get_preset(name)
    config = preset.as_dict()
    if cache_bytes is not None:
        if preset.mode != "streaming":
            raise ValueError("--cache-bytes is only valid for the low-memory preset")
        if cache_bytes <= 0:
            raise ValueError("cache bytes must be positive")
        config["cache_bytes"] = int(cache_bytes)
    config["telemetry_level"] = telemetry_level
    config["prefetch_enabled"] = config["prefetch_workers"] > 0
    config["shared_streaming_batching"] = False
    return config


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def model_identity(model_dir: str | Path, *, full_payload_hash: bool = False) -> dict[str, Any]:
    """Return a fast model identity; payload hashing is explicit and expensive."""

    root = Path(model_dir).expanduser().resolve()
    metadata_names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    metadata_files = [root / name for name in metadata_names if (root / name).is_file()]
    shards = sorted(root.glob("*.safetensors"))
    identity_files = metadata_files + shards
    result: dict[str, Any] = {
        "path": str(root),
        "metadata_sha256": _metadata_digest(root, metadata_files),
        "files": len(identity_files),
        "shards": len(shards),
        "shard_bytes": sum(path.stat().st_size for path in shards),
        "payload_hash_mode": "full" if full_payload_hash else "size-only",
    }
    if full_payload_hash:
        result["payload_sha256"] = _metadata_digest(root, shards)
    else:
        digest = hashlib.sha256()
        for path in shards:
            digest.update(path.name.encode("utf-8"))
            digest.update(str(path.stat().st_size).encode("ascii"))
        result["payload_size_sha256"] = digest.hexdigest()
    return result


def container_identity(container_dir: str | Path) -> dict[str, Any]:
    root = Path(container_dir).expanduser().resolve()
    metadata = [
        path
        for path in (
            root / "manifest.json",
            root / "index.json",
            root / "execution_manifest.json",
            root / "execution_index.json",
        )
        if path.is_file()
    ]
    if not metadata:
        raise ValueError(f"missing SparseFlow INT8 metadata: {root}")
    return {
        "path": str(root),
        "metadata_sha256": _metadata_digest(root, metadata),
        "metadata_files": sorted(path.relative_to(root).as_posix() for path in metadata),
        "weight_bytes": sum(path.stat().st_size for path in root.glob("layers/*.sfi")),
        "execution_bytes": sum(path.stat().st_size for path in root.glob("execution/*.sfx")),
    }


def prepare_disk_check(model_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Estimate conversion workspace without reading model payloads."""

    model = Path(model_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    analysis = analyze_model(model)
    source_routed = int(analysis["footprint"]["routed_expert_bytes"])
    # Canonical INT8 is approximately half the BF16 routed payload.  Include
    # a conservative temporary-file and manifest reserve for resumable writes.
    target_estimate = source_routed // 2
    existing = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) if output.is_dir() else 0
    required_new = max(0, target_estimate - existing) + 2 * 1024**3
    free = shutil.disk_usage(output.parent if output.parent.exists() else model).free
    return {
        "free_bytes": free,
        "target_estimate_bytes": target_estimate,
        "existing_output_bytes": existing,
        "required_new_bytes": required_new,
        "pass": free >= required_new,
    }


def cpu_features() -> dict[str, Any]:
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                flags.update(line.split(":", 1)[1].strip().split())
                break
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "flags_source": str(cpuinfo) if cpuinfo.is_file() else "unavailable",
    }


def _check_native_extension() -> dict[str, Any]:
    try:
        from .native_int8 import load_native_int8

        load_native_int8()
        return {"status": "pass", "available": True, "error": None}
    except Exception as exc:  # doctor must report build failures, not hide them
        return {
            "status": "fail",
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def doctor(
    model_dir: str | Path,
    *,
    preset: str = "stable",
    int8_container_dir: str | Path | None = None,
    cache_bytes: int | None = None,
    check_native: bool = False,
    full_payload_hash: bool = False,
) -> dict[str, Any]:
    """Run all preflight checks without constructing a model runtime."""

    root = Path(model_dir).expanduser().resolve()
    config = get_preset(preset)
    effective_config = apply_preset(
        preset,
        cache_bytes=cache_bytes,
    )
    checks: list[dict[str, Any]] = []

    def add(check_id: str, status: str, detail: Any) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    add("model_directory", "pass" if root.is_dir() else "fail", str(root))
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    add("required_files", "pass" if not missing else "fail", {"missing": missing})

    analysis: dict[str, Any] | None = None
    identity: dict[str, Any] | None = None
    try:
        load_config(root)
        ShardIndex.from_dir(root)
        analysis = analyze_model(root)
        identity = model_identity(root, full_payload_hash=full_payload_hash)
        add("safetensors_headers", "pass", {"shards": analysis["model"]["shards"]})
        add("model_config", "pass", analysis["model"])
    except Exception as exc:
        add("model_structure", "fail", f"{type(exc).__name__}: {exc}")

    free_bytes = shutil.disk_usage(root if root.is_dir() else root.parent).free
    runtime_reserve = 2 * 1024**3 + int(effective_config["cache_bytes"] or 0)
    add(
        "disk_space",
        "pass" if free_bytes >= runtime_reserve else "fail",
        {"free_bytes": free_bytes, "runtime_reserve_bytes": runtime_reserve},
    )

    cpu = cpu_features()
    needs_vnni = config.expert_storage == "int8-native"
    add(
        "cpu_isa",
        "pass" if not needs_vnni or cpu["avx512_vnni"] else "fail",
        {"required": "avx512_vnni" if needs_vnni else None, **cpu},
    )

    container: dict[str, Any] | None = None
    if config.expert_storage == "int8-native":
        if int8_container_dir is None:
            add("int8_container", "fail", "--int8-container is required for this preset")
        else:
            try:
                index = Int8ExpertIndex.from_dir(int8_container_dir)
                container = container_identity(int8_container_dir)
                expected_layers = int(analysis["model"]["num_hidden_layers"]) if analysis else 0
                complete_layers = tuple(index.layers) == tuple(range(expected_layers))
                add(
                    "int8_container",
                    "pass" if complete_layers else "fail",
                    {
                        **container,
                        "format_id": index.manifest.get("format_id"),
                        "layers": list(index.layers),
                        "expected_layers": expected_layers,
                        "complete_layers": complete_layers,
                        "offline_row_sums": index.has_offline_row_sums,
                    },
                )
                add(
                    "offline_row_sums",
                    "pass" if index.has_offline_row_sums else "warn",
                    {"required_for_native": True},
                )
            except Exception as exc:
                add("int8_container", "fail", f"{type(exc).__name__}: {exc}")

    native = None
    if check_native and cpu["avx512_vnni"]:
        native = _check_native_extension()
        add("native_extension", native["status"], native)
    else:
        add(
            "native_extension",
            "skip" if not check_native else "fail",
            "not compiled; pass --check-native on an AVX-512 VNNI host",
        )

    errors = [item for item in checks if item["status"] == "fail"]
    warnings = [item for item in checks if item["status"] == "warn"]
    return {
        "schema_version": 1,
        "kind": "sparseflow_public_alpha_doctor",
        "agent": "Main Dev",
        "preset": effective_config,
        "model": identity,
        "analysis": analysis,
        "container": container,
        "cpu": cpu,
        "checks": checks,
        "errors": len(errors),
        "warnings": len(warnings),
        "ready": not errors,
        "native_extension": native,
        "notes": [
            "shared streaming batching is disabled by policy",
            "pure fused decode is diagnostics-only",
            "model payload hash is size-only unless --full-payload-hash is used",
        ],
    }


# [Main Dev]

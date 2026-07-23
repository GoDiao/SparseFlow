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
from typing import Any, Callable

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
    default_context_tokens: int
    default_max_completion_tokens: int

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
        default_context_tokens=4096,
        default_max_completion_tokens=256,
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
        default_context_tokens=4096,
        default_max_completion_tokens=256,
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
        default_context_tokens=4096,
        default_max_completion_tokens=32,
    ),
    "laptop-16gb": RuntimePreset(
        name="laptop-16gb",
        description="Experimental INT8 streaming profile for hosts with about 16 GiB RAM.",
        mode="streaming",
        load_mode="memory-native",
        expert_storage="int8-native",
        native_dispatch="hybrid",
        cache_policy="lru",
        cache_bytes=256 * 1024**2,
        prefetch_workers=0,
        prefetch_policy="none",
        batch_mode="single",
        public_status="experimental-laptop",
        default_context_tokens=2048,
        default_max_completion_tokens=128,
    ),
}

GIB = 1024**3
_RUNTIME_PAGE_CACHE_RESERVE = 2 * GIB
_MIN_RECOMMENDED_HEADROOM = 2 * GIB


def _read_memory_value(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _proc_mem_available(path: Path = Path("/proc/meminfo")) -> int:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip() == "MemAvailable":
            fields = value.strip().split()
            if fields and fields[0].isdigit():
                # /proc/meminfo reports kB, not bytes.
                return int(fields[0]) * 1024
    return 0


def _windows_memory_available() -> int:
    if os.name != "nt":
        return 0
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return int(status.ullAvailPhys)
    except (AttributeError, OSError, ValueError):
        return 0


def memory_snapshot(
    *,
    override_bytes: int | None = None,
    proc_meminfo_path: str | Path = "/proc/meminfo",
    cgroup_root: str | Path = "/sys/fs/cgroup",
    windows_memory_provider: Callable[[], int] | None = None,
) -> dict[str, Any]:
    """Read usable memory, respecting cgroup limits when available.

    ``MemTotal`` is intentionally not used.  A process can only safely admit
    work against MemAvailable, or against the smaller remaining cgroup limit.
    The paths are injectable so the precedence rules can be tested without
    changing the host or container configuration.
    """

    host_available = _proc_mem_available(Path(proc_meminfo_path))
    host_source = "proc-meminfo" if host_available else "unknown"
    if not host_available:
        provider = windows_memory_provider
        if provider is None and os.name == "nt":
            provider = _windows_memory_available
        if provider is not None:
            try:
                host_available = max(0, int(provider()))
            except (OSError, TypeError, ValueError):
                host_available = 0
            if host_available:
                host_source = "windows-global-memory-status"
    root = Path(cgroup_root)
    cgroup_limit = None
    cgroup_current = None
    for candidate in (root, root / "memory"):
        limit_path = candidate / "memory.max"
        current_path = candidate / "memory.current"
        if not limit_path.is_file():
            limit_path = candidate / "memory.limit_in_bytes"
            current_path = candidate / "memory.usage_in_bytes"
        limit = _read_memory_value(limit_path)
        if limit is None:
            continue
        # cgroup v1 uses a very large sentinel for unlimited memory.
        if limit >= (1 << 60):
            continue
        cgroup_limit = limit
        cgroup_current = _read_memory_value(current_path)
        break

    cgroup_available = None
    if cgroup_limit is not None:
        cgroup_available = max(
            0,
            cgroup_limit - (cgroup_current or 0),
        )

    if override_bytes is not None:
        available = max(0, int(override_bytes))
        source = "override"
    elif cgroup_available is not None:
        available = min(host_available, cgroup_available) if host_available else cgroup_available
        source = "cgroup"
    elif host_available:
        available = host_available
        source = host_source
    else:
        available = 0
        source = "unknown"

    return {
        "available_ram_bytes": available,
        "source": source,
        "host_available_bytes": host_available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_current_bytes": cgroup_current,
        "cgroup_available_bytes": cgroup_available,
    }


def _estimate_state_bytes(analysis: dict[str, Any], ctx: int, batch_size: int) -> dict[str, int]:
    """Estimate KV and recurrent state for the supported text runtime."""

    model = analysis.get("model", {})
    text = analysis.get("model", {})
    # The analyzer intentionally exposes the architecture values needed by the
    # planner.  Older/synthetic fixtures may omit some values; zero is safer
    # than inventing a large state requirement in that case.
    hidden = int(model.get("hidden_size") or 0)
    layers = int(model.get("num_hidden_layers") or 0)
    # Qwen3.6 fields are available in the analyzer's model record when present.
    config_values = analysis.get("_text_config", text)
    if "_text_config" not in analysis:
        # The public analyzer keeps its output compact; read config.json here
        # only for the small set of state-shape fields used by admission.
        model_dir = analysis.get("model", {}).get("path")
        if model_dir:
            try:
                config_values = load_config(model_dir).text_config
            except (OSError, ValueError, KeyError):
                config_values = text
    kv_heads = int(config_values.get("num_key_value_heads") or 0)
    head_dim = int(config_values.get("head_dim") or 0)
    linear_key_heads = int(config_values.get("linear_num_key_heads") or 0)
    linear_value_heads = int(config_values.get("linear_num_value_heads") or 0)
    linear_key_dim = int(config_values.get("linear_key_head_dim") or 0)
    linear_value_dim = int(config_values.get("linear_value_head_dim") or 0)
    bytes_per_value = 2  # BF16 state in the current Qwen host runtime.
    context = max(1, int(ctx))
    batch = max(1, int(batch_size))
    attention_kv = (
        2 * layers * kv_heads * head_dim * context * bytes_per_value * batch
    )
    deltanet = (
        layers
        * linear_key_heads
        * linear_value_heads
        * linear_key_dim
        * linear_value_dim
        * bytes_per_value
        * batch
    )
    # Keep a small fallback for synthetic fixtures and future configs that do
    # not expose the DeltaNet dimensions in the public analyzer record.
    if not attention_kv and hidden and layers:
        attention_kv = layers * hidden * context * bytes_per_value // 16
    return {
        "attention_kv_state_bytes": attention_kv,
        "deltanet_state_bytes": deltanet,
        "kv_deltanet_state_bytes": attention_kv + deltanet,
    }


def memory_budget(
    analysis: dict[str, Any] | None,
    *,
    preset: str,
    effective_config: dict[str, Any],
    container: dict[str, Any] | None = None,
    model_dir: str | Path | None = None,
    ctx: int = 4096,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Calculate the RAM admission contract for a public preset."""

    if analysis is None:
        return {
            "available_ram_bytes": 0,
            "required_ram_bytes": 0,
            "recommended_ram_bytes": 0,
            "headroom_bytes": 0,
            "source": "unknown",
            "status": "warn",
            "reason": "model analysis unavailable; RAM admission is unknown",
            "components": {},
        }

    footprint = analysis.get("footprint", {})
    routed_bf16 = int(footprint.get("routed_expert_bytes") or 0)
    dense = int(footprint.get("dense_resident_bytes") or 0)
    resident_plan_bytes = None
    if model_dir is not None:
        try:
            from .memory_loader import build_memory_load_plan

            resident_plan_bytes = build_memory_load_plan(model_dir).bytes_for("resident")
        except (OSError, ValueError, KeyError):
            resident_plan_bytes = None
    if resident_plan_bytes is not None and resident_plan_bytes > 0:
        dense = int(resident_plan_bytes)

    container_weights = int((container or {}).get("weight_bytes") or 0)
    container_execution = int((container or {}).get("execution_bytes") or 0)
    int8_experts = container_weights or (routed_bf16 + 1) // 2
    row_sums = container_execution or (routed_bf16 // 512 if routed_bf16 else 0)
    # Keep the canonical component as one session's state.  A fixed cohort
    # accounts for the additional rows separately below; this prevents the
    # batch multiplier from being counted twice in the total.
    state = _estimate_state_bytes(analysis, ctx, 1)

    cache_bytes = int(effective_config.get("cache_bytes") or 0)
    streaming_mode = effective_config.get("mode") == "streaming"
    batch_state = 0
    batch_workspace = 0
    if preset == "experimental-batch":
        # Fixed-cohort keeps one additional KV/DeltaNet state row per extra
        # session, plus the shared grouped workspace.
        batch_state = max(0, int(batch_size) - 1) * state["kv_deltanet_state_bytes"]
        batch_workspace = (1 * GIB) + max(0, int(batch_size) - 1) * (256 * 1024**2)

    components = {
        "dense_resident_bytes": dense,
        "resident_int8_expert_bytes": int8_experts if not streaming_mode else 0,
        "streaming_cache_bytes": cache_bytes if streaming_mode else 0,
        "execution_row_sum_sidecar_bytes": row_sums,
        "kv_deltanet_state_bytes": state["kv_deltanet_state_bytes"],
        "activation_workspace_bytes": 1 * GIB,
        "experimental_batch_state_bytes": batch_state,
        "experimental_batch_workspace_bytes": batch_workspace,
        "runtime_page_cache_reserve_bytes": _RUNTIME_PAGE_CACHE_RESERVE,
    }
    required = sum(components.values())
    recommended = required + max(_MIN_RECOMMENDED_HEADROOM, required // 10)
    return {
        "required_ram_bytes": required,
        "recommended_ram_bytes": recommended,
        "components": components,
        "context_tokens": int(ctx),
        "batch_size": int(batch_size),
        "estimates": {
            "dense_source": "memory-native-plan" if resident_plan_bytes else "model-footprint",
            "int8_expert_source": "container" if container_weights else "bf16-half-estimate",
            "row_sum_source": "container" if container_execution else "bf16-ratio-estimate",
        },
    }


def evaluate_memory_admission(
    analysis: dict[str, Any] | None,
    *,
    preset: str,
    effective_config: dict[str, Any],
    container: dict[str, Any] | None = None,
    model_dir: str | Path | None = None,
    ctx: int = 4096,
    batch_size: int = 4,
    available_ram_bytes: int | None = None,
) -> dict[str, Any]:
    budget = memory_budget(
        analysis,
        preset=preset,
        effective_config=effective_config,
        container=container,
        model_dir=model_dir,
        ctx=ctx,
        batch_size=batch_size,
    )
    snapshot = memory_snapshot(override_bytes=available_ram_bytes)
    available = snapshot["available_ram_bytes"]
    required = int(budget.get("required_ram_bytes", 0))
    recommended = int(budget.get("recommended_ram_bytes", 0))
    if budget.get("status") == "warn" and budget.get("reason"):
        status = "warn"
    elif available <= 0:
        status = "warn"
    elif available < required:
        status = "fail"
    elif available < recommended:
        status = "warn"
    else:
        status = "pass"
    return {
        **snapshot,
        **budget,
        "available_ram_bytes": available,
        "headroom_bytes": available - required if available else 0,
        "status": status,
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
            raise ValueError("--cache-bytes is only valid for streaming presets")
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


def cpu_features(
    *,
    proc_cpuinfo_path: str | Path = "/proc/cpuinfo",
    cpuinfo_provider: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    flags: set[str] = set()
    cpuinfo = Path(proc_cpuinfo_path)
    flags_source = "unavailable"
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                flags.update(item.lower() for item in line.split(":", 1)[1].strip().split())
                flags_source = str(cpuinfo)
                break
    else:
        provider = cpuinfo_provider
        if provider is None:
            try:
                from cpuinfo import get_cpu_info
            except ImportError:
                get_cpu_info = None
            provider = get_cpu_info
        if provider is not None:
            try:
                info = provider() or {}
                flags.update(str(item).lower() for item in info.get("flags", ()))
                flags_source = "py-cpuinfo"
            except Exception:
                pass
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 1,
        "avx512f": "avx512f" in flags,
        "avx512_vnni": bool({"avx512_vnni", "avx512vnni"} & flags),
        "flags_source": flags_source,
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
    available_ram_bytes: int | None = None,
    ctx: int = 4096,
    batch_size: int = 4,
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

    memory = evaluate_memory_admission(
        analysis,
        preset=preset,
        effective_config=effective_config,
        container=container,
        model_dir=root,
        ctx=ctx,
        batch_size=batch_size,
        available_ram_bytes=available_ram_bytes,
    )
    add("memory_admission", memory["status"], memory)

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
        "memory": memory,
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

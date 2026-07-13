from __future__ import annotations

import ctypes
import re
import shutil
from pathlib import Path
from typing import Any

from .analyze import analyze_model
from .bytes import gb_to_bytes

GB = 1_000_000_000


def available_memory_bytes() -> int:
    if not hasattr(ctypes, "windll"):
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
        except OSError:
            return 0
        match = re.search(r"MemAvailable:\s+(\d+)", text)
        return int(match.group(1)) * 1024 if match else 0

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
    status.dwLength = ctypes.sizeof(status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return 0


def build_plan(
    model_dir: str | Path,
    ram_gb: float | None = None,
    ctx: int = 4096,
    reserve_gb: float = 2.5,
    available_memory: int | None = None,
) -> dict[str, Any]:
    analysis = analyze_model(model_dir)
    model_path = Path(model_dir).expanduser().resolve()
    disk = shutil.disk_usage(model_path)
    available_memory = available_memory_bytes() if available_memory is None else available_memory

    ram_budget = gb_to_bytes(ram_gb) if ram_gb is not None else int(available_memory * 0.88)
    reserve_bytes = gb_to_bytes(reserve_gb)
    dense = int(analysis["footprint"]["dense_resident_bytes"])
    typical_expert = int(analysis["footprint"]["typical_expert_bytes"])
    per_slot_set = int(analysis["footprint"]["per_layer_cache_slot_set_bytes"])
    expert_layers = int(analysis["expert_layout"]["layers"])

    # Conservative first-pass runtime reserve. This is deliberately explicit so it
    # can be replaced by backend-specific profiling later.
    kv_bytes = _estimate_kv_bytes(analysis, ctx)
    working_set_bytes = typical_expert * 64
    activation_bytes = gb_to_bytes(1.0)
    runtime_bytes = reserve_bytes + kv_bytes + working_set_bytes + activation_bytes
    cache_bytes = max(0, ram_budget - dense - runtime_bytes)
    slots_per_layer = int(cache_bytes // per_slot_set) if per_slot_set else 0
    configured_experts = int(analysis["model"]["num_experts"] or 0)
    if configured_experts:
        slots_per_layer = min(slots_per_layer, configured_experts)

    warnings: list[str] = []
    if ram_budget > available_memory and available_memory:
        warnings.append("planned RAM budget exceeds currently available physical memory")
    if slots_per_layer < 1 and expert_layers:
        warnings.append("RAM budget cannot hold one routed-expert slot per sparse layer")
    if disk.free < gb_to_bytes(1):
        warnings.append("less than 1 GB free beside the model directory")

    return {
        "schema_version": 1,
        "model": analysis["model"],
        "tiers": {
            "disk": {
                "role": "backing_store",
                "model_bytes": analysis["model"]["safetensors_bytes"],
                "available_bytes": disk.free,
            },
            "ram": {
                "role": "resident_dense_plus_expert_cache",
                "available_bytes": available_memory,
                "budget_bytes": ram_budget,
                "dense_resident_bytes": dense,
                "runtime_reserve_bytes": runtime_bytes,
                "runtime_breakdown": {
                    "page_cache_reserve_bytes": reserve_bytes,
                    "kv_estimate_bytes": kv_bytes,
                    "expert_working_set_bytes": working_set_bytes,
                    "activation_reserve_bytes": activation_bytes,
                },
                "expert_cache_bytes": cache_bytes,
                "cache_slots_per_layer": slots_per_layer,
            },
            "vram": {
                "role": "future_hot_expert_tier",
                "budget_bytes": 0,
                "expert_capacity": 0,
            },
        },
        "analysis": analysis["footprint"],
        "warnings": warnings,
    }


def _estimate_kv_bytes(analysis: dict[str, Any], ctx: int) -> int:
    model = analysis["model"]
    hidden = int(model.get("hidden_size") or 2048)
    layers = int(model.get("num_hidden_layers") or 0)
    # Placeholder until adapters expose exact KV state. Keep it intentionally
    # modest for linear-attention models, but visible in the plan output.
    return layers * ctx * hidden * 2

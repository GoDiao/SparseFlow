from __future__ import annotations

import os
from pathlib import Path
import threading


_LOAD_LOCK = threading.Lock()
_LOADED = False
_PROFILE_ANCHOR = None
_PROFILE_FIELDS = (
    "native_row_sums_ns",
    "native_row_sums_calls",
    "native_activation_quantization_ns",
    "native_activation_quantization_calls",
    "native_gemv_ns",
    "native_gemv_calls",
    "native_dynamic_linear_ns",
    "native_dynamic_linear_calls",
)


def load_native_int8() -> None:
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:
            return
        _require_avx512_vnni()
        import torch
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parents[2]
        source = root / "native" / "int8_vnni.cpp"
        build = Path(
            os.environ.get(
                "SPARSEFLOW_NATIVE_CACHE",
                root / ".cache" / "native" / "int8_vnni",
            )
        ).expanduser()
        build.mkdir(parents=True, exist_ok=True)
        load(
            name="sparseflow_int8_vnni",
            sources=[str(source), str(root / "native" / "moe_dispatch.cpp")],
            extra_cflags=[
                "-O3",
                "-std=c++17",
                "-mavx512f",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
            build_directory=str(build),
            is_python_module=False,
            verbose=False,
        )
        if not hasattr(torch.ops.sparseflow_native, "dynamic_linear") or not hasattr(
            torch.ops.sparseflow_native, "fused_moe"
        ):
            raise RuntimeError("SparseFlow native INT8 operators failed to register")
        _LOADED = True


def prepare_native_weights(location, payloads, torch):
    load_native_int8()
    result = {"native_int8": True}
    for part in location.parts:
        data = payloads[f"{part.part}.data"]
        scales = payloads[f"{part.part}.scales"]
        weight = torch.frombuffer(data, dtype=torch.int8).reshape(part.shape)
        scale_tensor = torch.frombuffer(scales, dtype=torch.float16).float().contiguous()
        row_sum_payload = payloads.get(f"{part.part}.row_sums")
        row_sums = (
            torch.frombuffer(row_sum_payload, dtype=torch.int32).contiguous()
            if row_sum_payload is not None
            else torch.ops.sparseflow_native.row_sums(weight)
        )
        result[part.part] = {
            "weight": weight,
            "scales": scale_tensor,
            "row_sums": row_sums,
        }
    return result


def set_native_profile(enabled: bool) -> None:
    global _PROFILE_ANCHOR
    load_native_int8()
    import torch

    if _PROFILE_ANCHOR is None:
        _PROFILE_ANCHOR = torch.empty(0)
    torch.ops.sparseflow_native.set_profile_enabled(_PROFILE_ANCHOR, bool(enabled))
    if enabled:
        torch.ops.sparseflow_native.reset_profile(_PROFILE_ANCHOR)


def native_profile_snapshot() -> dict[str, int]:
    load_native_int8()
    import torch

    global _PROFILE_ANCHOR
    if _PROFILE_ANCHOR is None:
        _PROFILE_ANCHOR = torch.empty(0)
    values = torch.ops.sparseflow_native.profile_snapshot(_PROFILE_ANCHOR).tolist()
    return dict(zip(_PROFILE_FIELDS, (int(value) for value in values), strict=True))


def native_profile_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, float]:
    return {
        "native_row_sums": (after["native_row_sums_ns"] - before["native_row_sums_ns"]) / 1e6,
        "native_activation_quantization": (
            after["native_activation_quantization_ns"]
            - before["native_activation_quantization_ns"]
        )
        / 1e6,
        "native_gemv": (after["native_gemv_ns"] - before["native_gemv_ns"]) / 1e6,
        "native_dynamic_linear": (
            after["native_dynamic_linear_ns"] - before["native_dynamic_linear_ns"]
        )
        / 1e6,
    }


def run_native_expert(hidden_states, weights, torch):
    load_native_int8()
    gate_up = weights["gate_up_proj"]
    projected = torch.ops.sparseflow_native.dynamic_linear(
        hidden_states.float().contiguous(),
        gate_up["weight"],
        gate_up["scales"],
        gate_up["row_sums"],
    )
    gate, up = projected.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    down = weights["down_proj"]
    output = torch.ops.sparseflow_native.dynamic_linear(
        activated.contiguous(),
        down["weight"],
        down["scales"],
        down["row_sums"],
    )
    return output.to(hidden_states.dtype)


def reference_dynamic_linear(input_tensor, weight, scales, row_sums, torch):
    values = input_tensor.float().contiguous()
    rows = []
    for row in values:
        low = min(0.0, float(row.min()))
        high = max(0.0, float(row.max()))
        scale = (high - low) / 255.0 if high > low else 1.0
        zero_point = max(0, min(255, round(-low / scale)))
        quantized = torch.round(row / scale).to(torch.int32).add_(zero_point).clamp_(0, 255)
        accumulator = quantized @ weight.to(torch.int32).transpose(0, 1)
        accumulator -= zero_point * row_sums
        rows.append(accumulator.float() * scale * scales)
    return torch.stack(rows)


def _require_avx512_vnni() -> None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file() and "avx512_vnni" not in cpuinfo.read_text(encoding="utf-8"):
        raise RuntimeError("SparseFlow native INT8 kernel requires AVX-512 VNNI")


# [Main Dev]

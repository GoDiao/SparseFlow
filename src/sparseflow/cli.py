from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .analyze import analyze_model
from .benchmark import (
    format_expert_benchmark,
    generate_trace,
    load_trace,
    parse_capacities,
    parse_byte_budgets,
    parse_layers,
    run_expert_benchmark,
)
from .bytes import format_bytes
from .cache_policy import POLICY_VARIANTS
from .loader import load_expert_raw
from .int8_container import build_int8_execution_metadata, convert_experts_int8
from .locator import ExpertLocator
from .memory_loader import (
    build_memory_load_plan,
    build_qwen36_meta_text_model,
    materialize_qwen36_text_model,
)
from .moe_probe import compare_expert_paths, compare_moe_cache_paths, compare_moe_paths
from .moe_runtime import compare_multilayer_moe_paths
from .plan import build_plan
from .policy_benchmark import discover_trace_paths, run_policy_sweep
from .release import (
    PRESETS,
    apply_preset,
    container_identity,
    doctor,
    get_preset,
    model_identity,
    prepare_disk_check,
)
from .route_trace import (
    capture_route_trace,
    capture_route_traces,
    load_prompt_manifest,
    write_route_trace,
)
from .trace import load_route_trace


class RuntimeExtrasError(ValueError):
    """Raised when a command needs the optional torch runtime."""


def _load_text_runtime(command: str):
    """Import runtime code only for commands that actually execute a model."""

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import safetensors  # noqa: F401
        import accelerate  # noqa: F401
        from . import text_runtime
    except (ImportError, OSError) as exc:
        raise RuntimeExtrasError(
            f"{command} requires the optional runtime dependencies (torch, "
            "transformers, safetensors, accelerate); install them with "
            "`pip install -e '.[runtime]'`"
        ) from exc
    return text_runtime


# Keep these names as lazy compatibility shims for callers that patch the CLI
# boundary in tests.  They do not import torch until the function is called.
def compare_text_paths(*args: Any, **kwargs: Any):
    return _load_text_runtime("text-check").compare_text_paths(*args, **kwargs)


def compare_sparseflow_runtime_paths(*args: Any, **kwargs: Any):
    return _load_text_runtime("runtime-check").compare_sparseflow_runtime_paths(*args, **kwargs)


def compare_int8_reference_paths(*args: Any, **kwargs: Any):
    return _load_text_runtime("int8-reference-check").compare_int8_reference_paths(*args, **kwargs)


def compare_int8_native_paths(*args: Any, **kwargs: Any):
    return _load_text_runtime("int8-reference-check").compare_int8_native_paths(*args, **kwargs)


def compare_bf16_int8_quantization(*args: Any, **kwargs: Any):
    return _load_text_runtime("int8-quality-check").compare_bf16_int8_quantization(*args, **kwargs)


def compare_int8_native_quantization(*args: Any, **kwargs: Any):
    return _load_text_runtime("int8-quality-check").compare_int8_native_quantization(*args, **kwargs)


def compare_sparseflow_policy_paths(*args: Any, **kwargs: Any):
    return _load_text_runtime("policy-check").compare_sparseflow_policy_paths(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparseflow")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect model safetensors without loading tensor payloads.")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    plan_p = sub.add_parser("plan", help="Create a Disk/RAM/VRAM placement plan.")
    plan_p.add_argument("model")
    plan_p.add_argument("--ram", type=float, default=None, help="RAM budget in decimal GB.")
    plan_p.add_argument("--available-ram", help="Override available RAM used by the planner, e.g. 16GiB.")
    plan_p.add_argument("--ctx", type=int, default=4096, help="Context length for memory projection.")
    plan_p.add_argument("--reserve", type=float, default=2.5, help="Page-cache/runtime reserve in decimal GB.")
    plan_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    preset_p = sub.add_parser(
        "preset",
        help="Show the supported Public Alpha runtime presets.",
    )
    preset_p.add_argument("name", choices=tuple(sorted(PRESETS)), nargs="?", default=None)
    preset_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    doctor_p = sub.add_parser(
        "doctor",
        help="Run read-only model, INT8, disk, CPU, and runtime readiness checks.",
    )
    doctor_p.add_argument("model")
    doctor_p.add_argument("--preset", choices=tuple(sorted(PRESETS)), default="stable")
    doctor_p.add_argument("--int8-container")
    doctor_p.add_argument("--cache-bytes", help="Override low-memory cache budget, e.g. 4GiB.")
    doctor_p.add_argument("--available-ram", help="Override available RAM for admission, e.g. 16GiB.")
    doctor_p.add_argument("--ctx", type=int, help="Context length for RAM admission; defaults to the preset.")
    doctor_p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Fixed cohort size used by experimental-batch RAM admission.",
    )
    doctor_p.add_argument("--check-native", action="store_true", help="Build and load the native extension.")
    doctor_p.add_argument("--full-payload-hash", action="store_true", help="Hash all model payload bytes; expensive.")
    doctor_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    prepare_p = sub.add_parser(
        "prepare-int8",
        help="Create/resume a complete INT8 expert container and execution metadata.",
    )
    prepare_p.add_argument("model")
    prepare_p.add_argument("--output", required=True)
    prepare_p.add_argument("--layers", help="Layer list/ranges, e.g. 0-3 or 0,2,4.")
    prepare_p.add_argument("--report")
    prepare_p.add_argument("--no-resume", action="store_true")
    prepare_p.add_argument("--json", action="store_true")

    run_p = sub.add_parser(
        "run",
        help="Run a Public Alpha Qwen3.6 preset.",
    )
    run_p.add_argument("model")
    run_p.add_argument("--preset", choices=tuple(sorted(PRESETS)), default="stable")
    run_p.add_argument("--int8-container", required=True)
    run_p.add_argument("--prompt", action="append", required=True, help="Prompt; repeat for experimental-batch.")
    run_p.add_argument("--max-new-tokens", type=int, default=32)
    run_p.add_argument("--ctx", type=int, help="Context admission limit for the text runtime.")
    run_p.add_argument("--cache-bytes", help="Override the low-memory preset budget.")
    run_p.add_argument("--telemetry-level", choices=("none", "summary", "profile", "layer"), default="summary")
    run_p.add_argument("--output")
    run_p.add_argument("--json", action="store_true")

    serve_p = sub.add_parser(
        "serve",
        help="Start the local OpenAI-compatible SparseFlow server.",
    )
    serve_p.add_argument("model")
    serve_p.add_argument("--preset", choices=("stable", "low-memory", "laptop-16gb"), default="low-memory")
    serve_p.add_argument("--int8-container", required=True)
    serve_p.add_argument("--cache-bytes")
    serve_p.add_argument("--ctx", type=int, help="Context limit; defaults to the preset.")
    serve_p.add_argument("--max-completion-tokens", type=int, help="Server output cap; defaults to the preset.")
    serve_p.add_argument("--telemetry-level", choices=("none", "summary", "profile", "layer"), default="summary")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--model-id", default="qwen3.6-35b-a3b-sparseflow")
    serve_p.add_argument("--api-key")
    serve_p.add_argument("--cors-origin", action="append", dest="cors_origins")
    serve_p.add_argument("--max-queue", type=int, default=8)
    serve_p.add_argument("--queue-timeout", type=float, default=300.0)
    serve_p.add_argument("--keepalive-seconds", type=float, default=10.0)
    serve_p.add_argument("--allow-unauthenticated-remote", action="store_true")

    native_plan_p = sub.add_parser(
        "native-plan",
        help="Plan a text-only memory-native checkpoint load without reading payloads.",
    )
    native_plan_p.add_argument("model")
    native_plan_p.add_argument("--entries", action="store_true", help="Include every tensor mapping.")
    native_plan_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    native_meta_p = sub.add_parser(
        "native-meta",
        help="Build the Qwen3.6 text runtime on meta device with experts removed.",
    )
    native_meta_p.add_argument("model")
    native_meta_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    native_load_p = sub.add_parser(
        "native-load",
        help="Selectively materialize Qwen3.6 text weights while skipping experts.",
    )
    native_load_p.add_argument("model")
    native_load_p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    native_load_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    int8_convert_p = sub.add_parser(
        "int8-convert",
        help="Convert routed BF16 experts into a versioned SparseFlow INT8 container.",
    )
    int8_convert_p.add_argument("model")
    int8_convert_p.add_argument("--output", required=True)
    int8_convert_p.add_argument("--layers", help="Layer list/ranges, e.g. 0-3 or 0,2,4.")
    int8_convert_p.add_argument("--threads", type=int, default=10)
    int8_convert_p.add_argument("--no-resume", action="store_true")
    int8_convert_p.add_argument("--report", help="Write the conversion result JSON.")
    int8_convert_p.add_argument("--json", action="store_true")

    int8_meta_p = sub.add_parser(
        "int8-exec-meta",
        help="Build optional offline execution metadata for an INT8 container.",
    )
    int8_meta_p.add_argument("container")
    int8_meta_p.add_argument("--threads", type=int, default=10)
    int8_meta_p.add_argument("--no-resume", action="store_true")
    int8_meta_p.add_argument("--report")
    int8_meta_p.add_argument("--json", action="store_true")

    for command, help_text in (
        ("expert-stat", "Locate one fused expert without reading its payload."),
        ("expert-load", "Read and checksum one expert's raw tensor slices."),
    ):
        expert_p = sub.add_parser(command, help=help_text)
        expert_p.add_argument("model")
        expert_p.add_argument("--layer", type=int, required=True)
        expert_p.add_argument("--expert", type=int, required=True)
        expert_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    bench_p = sub.add_parser(
        "expert-bench",
        help="Benchmark per-layer expert-cache hit rate and raw read overhead.",
    )
    bench_p.add_argument("model")
    bench_p.add_argument("--capacities", default="1,2,4,8", help="LRU slots per layer, comma-separated.")
    bench_p.add_argument("--byte-budgets", help="Global byte budgets, e.g. 512MiB,1GiB,2GiB.")
    bench_p.add_argument("--layers", default="0", help="Layer list/ranges, e.g. 0 or 0-39.")
    bench_p.add_argument("--tokens", type=int, default=4, help="Generated trace tokens when --trace is absent.")
    bench_p.add_argument("--topk", type=int, default=8, help="Experts selected per layer in generated traces.")
    bench_p.add_argument(
        "--mode",
        choices=("uniform", "locality"),
        default="locality",
        help="Generated trace distribution.",
    )
    bench_p.add_argument("--seed", type=int, default=1234)
    bench_p.add_argument("--trace", help="JSON trace file; overrides generated trace options.")
    bench_p.add_argument("--batch-union", action="store_true", help="Deduplicate experts within each forward/layer group.")
    bench_p.add_argument("--output", help="Write the machine-readable result JSON to this path.")
    bench_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    policy_sweep_p = sub.add_parser(
        "policy-sweep",
        help="Replay real route traces across Stage 7.3 cache/prefetch policies.",
    )
    policy_sweep_p.add_argument("model")
    policy_sweep_p.add_argument(
        "--trace-dir",
        action="append",
        required=True,
        help="Directory containing route-trace v2 JSON; repeat for multiple lengths.",
    )
    policy_sweep_p.add_argument("--byte-budgets", default="1GiB,2GiB,4GiB,8GiB")
    policy_sweep_p.add_argument("--variants", default=",".join(POLICY_VARIANTS))
    policy_sweep_p.add_argument("--hot-ratio", type=float, default=0.25)
    policy_sweep_p.add_argument("--prefetch-budget-ratio", type=float, default=0.25)
    policy_sweep_p.add_argument("--output", help="Write sweep JSON to this path.")
    policy_sweep_p.add_argument("--json", action="store_true")

    route_p = sub.add_parser(
        "route-trace",
        help="Capture actual Qwen3.5 MoE router expert selections from generation.",
    )
    route_p.add_argument("model")
    route_p.add_argument("--prompt", required=True)
    route_p.add_argument("--max-new-tokens", type=int, default=8)
    route_p.add_argument("--output", required=True)

    route_batch_p = sub.add_parser(
        "route-trace-batch",
        help="Capture multiple prompt route traces while loading the model once.",
    )
    route_batch_p.add_argument("model")
    route_batch_p.add_argument("--manifest", required=True, help="JSONL rows with id and text fields.")
    route_batch_p.add_argument("--limit", type=int, default=0)
    route_batch_p.add_argument("--max-new-tokens", type=int, default=8)
    route_batch_p.add_argument("--output-dir", required=True)

    check_p = sub.add_parser(
        "expert-kernel-check",
        help="Compare resident and streaming execution for one routed expert.",
    )
    check_p.add_argument("model")
    check_p.add_argument("--layer", type=int, required=True)
    check_p.add_argument("--expert", type=int, required=True)
    check_p.add_argument("--rows", type=int, default=2)
    check_p.add_argument("--seed", type=int, default=1234)
    check_p.add_argument("--json", action="store_true")

    moe_check_p = sub.add_parser(
        "expert-moe-check",
        help="Compare resident and streaming execution for one complete Qwen3.6 MoE layer.",
    )
    moe_check_p.add_argument("model")
    moe_check_p.add_argument("--layer", type=int, default=0)
    moe_check_p.add_argument("--rows", type=int, default=1)
    moe_check_p.add_argument("--seed", type=int, default=1234)
    moe_check_p.add_argument("--json", action="store_true")

    moe_cache_p = sub.add_parser(
        "expert-moe-cache-check",
        help="Compare repeated resident and ExpertCache-backed streaming MoE forwards.",
    )
    moe_cache_p.add_argument("model")
    moe_cache_p.add_argument("--layer", type=int, default=0)
    moe_cache_p.add_argument("--rows", type=int, default=8)
    moe_cache_p.add_argument("--forwards", type=int, default=4)
    moe_cache_p.add_argument("--repeats", type=int, default=2)
    moe_cache_p.add_argument(
        "--cache-slots",
        type=int,
        default=None,
        help="Per-layer LRU capacity; defaults to 4 when no byte budget is given.",
    )
    moe_cache_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 48MiB.")
    moe_cache_p.add_argument("--seed", type=int, default=1234)
    moe_cache_p.add_argument("--json", action="store_true")

    multi_p = sub.add_parser(
        "moe-multi-check",
        help="Compare resident and streaming execution across multiple Qwen3.6 MoE layers.",
    )
    multi_p.add_argument("model")
    multi_p.add_argument("--layers", default="0-1", help="Layer list/ranges, e.g. 0-1 or 0,2,4.")
    multi_p.add_argument("--rows", type=int, default=2)
    multi_p.add_argument("--cache-slots", type=int, default=16)
    multi_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 48MiB.")
    multi_p.add_argument("--prefetch-workers", type=int, default=0)
    multi_p.add_argument("--coalesce-gap", type=int, default=0)
    multi_p.add_argument("--seed", type=int, default=1234)
    multi_p.add_argument("--json", action="store_true")

    text_p = sub.add_parser(
        "text-generate",
        help="Run the full Qwen3.6 text-only Python reference runtime.",
    )
    text_p.add_argument("model")
    text_p.add_argument("--prompt", required=True)
    text_p.add_argument("--mode", choices=("resident", "streaming"), default="streaming")
    text_p.add_argument(
        "--expert-backend",
        choices=("sparseflow-resident", "sparseflow-streaming"),
        help="Select a Stage 7.2 memory-native SparseFlow expert backend.",
    )
    text_p.add_argument(
        "--load-mode",
        choices=("transformers", "memory-native"),
        default="transformers",
    )
    text_p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    text_p.add_argument(
        "--experts-implementation",
        choices=("eager", "grouped_mm"),
        default=None,
        help="Resident expert kernel; SparseFlow streaming currently requires eager.",
    )
    text_p.add_argument("--max-new-tokens", type=int, default=8)
    text_p.add_argument("--cache-slots", type=int, default=16)
    text_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 4GiB.")
    text_p.add_argument("--prefetch-workers", type=int, default=0)
    text_p.add_argument("--coalesce-gap", type=int, default=0)
    text_p.add_argument("--cache-policy", choices=("none", "lru", "hot", "heat"), default="lru")
    text_p.add_argument(
        "--prefetch-policy",
        choices=("none", "current-route", "previous-token", "hot-set"),
        default="current-route",
    )
    text_p.add_argument("--hot-ratio", type=float, default=0.25)
    text_p.add_argument("--prefetch-budget-ratio", type=float, default=0.25)
    text_p.add_argument(
        "--telemetry-level",
        choices=("none", "summary", "profile", "layer"),
        default="summary",
    )
    text_p.add_argument("--telemetry-output", help="Write forward/layer telemetry as JSONL.")
    text_p.add_argument("--output", help="Write generation JSON to this path.")
    text_p.add_argument("--json", action="store_true")

    text_check_p = sub.add_parser(
        "text-check",
        help="Compare resident and SparseFlow streaming text generation.",
    )
    text_check_p.add_argument("model")
    text_check_p.add_argument("--prompt", required=True)
    text_check_p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    text_check_p.add_argument(
        "--streaming-loader",
        choices=("transformers", "memory-native"),
        default="transformers",
    )
    text_check_p.add_argument("--max-new-tokens", type=int, default=8)
    text_check_p.add_argument("--cache-slots", type=int, default=16)
    text_check_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 4GiB.")
    text_check_p.add_argument("--prefetch-workers", type=int, default=0)
    text_check_p.add_argument("--coalesce-gap", type=int, default=0)
    text_check_p.add_argument(
        "--cache-policy",
        choices=("none", "lru", "hot", "heat"),
        default="lru",
    )
    text_check_p.add_argument(
        "--prefetch-policy",
        choices=("none", "current-route", "previous-token", "hot-set"),
        default="current-route",
    )
    text_check_p.add_argument("--hot-ratio", type=float, default=0.25)
    text_check_p.add_argument("--prefetch-budget-ratio", type=float, default=0.25)
    text_check_p.add_argument(
        "--telemetry-level",
        choices=("none", "summary", "profile", "layer"),
        default="summary",
    )
    text_check_p.add_argument("--output", help="Write comparison JSON to this path.")
    text_check_p.add_argument("--json", action="store_true")

    runtime_check_p = sub.add_parser(
        "runtime-check",
        help="Compare memory-native C3-R and C3-S with the same SparseFlow kernel.",
    )
    runtime_check_p.add_argument("model")
    runtime_check_p.add_argument("--prompt", required=True)
    runtime_check_p.add_argument("--dtype", choices=("bf16",), default="bf16")
    runtime_check_p.add_argument("--max-new-tokens", type=int, default=4)
    runtime_check_p.add_argument("--cache-slots", type=int, default=16)
    runtime_check_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 4GiB.")
    runtime_check_p.add_argument("--prefetch-workers", type=int, default=0)
    runtime_check_p.add_argument("--coalesce-gap", type=int, default=0)
    runtime_check_p.add_argument(
        "--cache-policy",
        choices=("none", "lru", "hot", "heat"),
        default="lru",
    )
    runtime_check_p.add_argument(
        "--prefetch-policy",
        choices=("none", "current-route", "previous-token", "hot-set"),
        default="current-route",
    )
    runtime_check_p.add_argument("--hot-ratio", type=float, default=0.25)
    runtime_check_p.add_argument("--prefetch-budget-ratio", type=float, default=0.25)
    runtime_check_p.add_argument(
        "--telemetry-level",
        choices=("none", "summary", "profile", "layer"),
        default="summary",
    )
    runtime_check_p.add_argument("--output", help="Write comparison JSON to this path.")
    runtime_check_p.add_argument("--json", action="store_true")

    int8_check_p = sub.add_parser(
        "int8-reference-check",
        help="Compare INT8 resident and streaming providers with one reference kernel.",
    )
    int8_check_p.add_argument("model")
    int8_check_p.add_argument("--int8-container", required=True)
    int8_check_p.add_argument("--prompt", required=True)
    int8_check_p.add_argument("--max-new-tokens", type=int, default=4)
    int8_check_p.add_argument("--kernel", choices=("reference", "native"), default="reference")
    int8_check_p.add_argument("--cache-bytes", default="4GiB")
    int8_check_p.add_argument(
        "--cache-policy",
        choices=("none", "lru", "hot", "heat"),
        default="lru",
    )
    int8_check_p.add_argument(
        "--telemetry-level",
        choices=("none", "summary", "profile", "layer"),
        default="summary",
    )
    int8_check_p.add_argument("--prefetch-workers", type=int, default=0)
    int8_check_p.add_argument(
        "--prefetch-policy",
        choices=("none", "current-route", "previous-token", "hot-set"),
        default="none",
    )
    int8_check_p.add_argument("--prefetch-budget-ratio", type=float, default=0.10)
    int8_check_p.add_argument("--coalesce-gap", type=int, default=0)
    int8_check_p.add_argument(
        "--native-dispatch", choices=("legacy", "fused", "hybrid", "grouped"), default="legacy"
    )
    int8_check_p.add_argument("--deterministic-io-pipeline", action="store_true")
    int8_check_p.add_argument("--fuse-deltanet-projections", action="store_true")
    int8_check_p.add_argument("--output")
    int8_check_p.add_argument("--json", action="store_true")

    int8_quality_p = sub.add_parser(
        "int8-quality-check",
        help="Compare BF16 and INT8 logits on one fixed teacher-forced token path.",
    )
    int8_quality_p.add_argument("model")
    int8_quality_p.add_argument("--int8-container", required=True)
    int8_quality_p.add_argument("--prompt", required=True)
    int8_quality_p.add_argument("--max-new-tokens", type=int, default=32)
    int8_quality_p.add_argument("--top-k", type=int, default=10)
    int8_quality_p.add_argument(
        "--kernel",
        choices=("reference", "native"),
        default="reference",
        help="reference compares BF16/W8A16; native compares W8A16/W8A8.",
    )
    int8_quality_p.add_argument("--output")
    int8_quality_p.add_argument("--json", action="store_true")

    policy_check_p = sub.add_parser(
        "policy-check",
        help="Compare Stage 7.3 S0-S4 policies against one memory-native C3-R run.",
    )
    policy_check_p.add_argument("model")
    policy_check_p.add_argument("--prompt", required=True)
    policy_check_p.add_argument("--max-new-tokens", type=int, default=8)
    policy_check_p.add_argument("--cache-bytes", default="4GiB")
    policy_check_p.add_argument("--variants", default=",".join(POLICY_VARIANTS))
    policy_check_p.add_argument("--prefetch-workers", type=int, default=2)
    policy_check_p.add_argument("--prefetch-budget-ratio", type=float, default=0.10)
    policy_check_p.add_argument("--hot-ratio", type=float, default=0.25)
    policy_check_p.add_argument(
        "--telemetry-level",
        choices=("none", "summary", "profile", "layer"),
        default="summary",
    )
    policy_check_p.add_argument("--output")
    policy_check_p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "preset":
            if args.name is None:
                result = {
                    "schema_version": 1,
                    "kind": "sparseflow_public_alpha_presets",
                    "agent": "Main Dev",
                    "presets": [PRESETS[name].as_dict() for name in sorted(PRESETS)],
                }
            else:
                result = {
                    "schema_version": 1,
                    "kind": "sparseflow_public_alpha_preset",
                    "agent": "Main Dev",
                    "preset": get_preset(args.name).as_dict(),
                }
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_preset(result))
            return 0
        if args.command == "doctor":
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            available_ram = (
                _parse_single_byte_budget(args.available_ram, flag="--available-ram")
                if args.available_ram
                else None
            )
            context_tokens = (
                args.ctx
                if args.ctx is not None
                else get_preset(args.preset).default_context_tokens
            )
            if context_tokens <= 0:
                raise ValueError("--ctx must be positive")
            if args.batch_size <= 0:
                raise ValueError("--batch-size must be positive")
            batch_size = args.batch_size if args.preset == "experimental-batch" else 1
            result = doctor(
                args.model,
                preset=args.preset,
                int8_container_dir=args.int8_container,
                cache_bytes=cache_bytes,
                check_native=args.check_native,
                full_payload_hash=args.full_payload_hash,
                available_ram_bytes=available_ram,
                ctx=context_tokens,
                batch_size=batch_size,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_doctor(result))
            return 0 if result["ready"] else 1
        if args.command == "prepare-int8":
            layers = None
            if args.layers:
                locator = ExpertLocator(args.model)
                layers = parse_layers(args.layers, max(locator.layers) + 1)
            resume = not args.no_resume
            disk = prepare_disk_check(args.model, args.output)
            if not disk["pass"]:
                raise ValueError(
                    "insufficient disk space for resumable INT8 preparation: "
                    f"free={disk['free_bytes']} required={disk['required_new_bytes']}"
                )
            converted = convert_experts_int8(
                args.model,
                args.output,
                layers=layers,
                resume=resume,
            )
            execution = build_int8_execution_metadata(args.output, resume=resume)
            result = {
                "schema_version": 1,
                "kind": "sparseflow_public_alpha_int8_prepare",
                "agent": "Main Dev",
                "model": model_identity(args.model),
                "container": container_identity(args.output),
                "disk": disk,
                "conversion": converted,
                "execution": execution,
            }
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.report:
                report = Path(args.report).expanduser()
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_prepare_int8(result))
            return 0
        if args.command == "run":
            runtime_api = _load_text_runtime("run")
            Qwen36TextRuntime = runtime_api.Qwen36TextRuntime
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            config = apply_preset(
                args.preset,
                cache_bytes=cache_bytes,
                telemetry_level=args.telemetry_level,
            )
            prompts = tuple(args.prompt)
            context_tokens = (
                args.ctx
                if args.ctx is not None
                else config["default_context_tokens"]
            )
            if context_tokens <= 0:
                raise ValueError("--ctx must be positive")
            if config["batch_mode"] == "fixed-cohort":
                if len(prompts) < 2:
                    raise ValueError("experimental-batch requires at least two --prompt values")
                from .fixed_cohort import generate_fixed_cohort

                runtime = Qwen36TextRuntime.from_pretrained(
                    args.model,
                    mode=config["mode"],
                    dtype="bf16",
                    cache_slots=None,
                    cache_bytes=None,
                    prefetch_workers=0,
                    coalesce_gap=0,
                    cache_policy=config["cache_policy"],
                    prefetch_policy=config["prefetch_policy"],
                    telemetry_level=args.telemetry_level,
                    experts_implementation="eager",
                    load_mode=config["load_mode"],
                    expert_storage=config["expert_storage"],
                    int8_container=args.int8_container,
                    native_dispatch=config["native_dispatch"],
                )
                try:
                    result = generate_fixed_cohort(
                        runtime,
                        prompts,
                        max_new_tokens=args.max_new_tokens,
                        stop_on_eos=False,
                        capture_logits=False,
                    )
                finally:
                    runtime.close()
            else:
                runtime = Qwen36TextRuntime.from_pretrained(
                    args.model,
                    mode=config["mode"],
                    dtype="bf16",
                    cache_slots=None if config["cache_bytes"] is not None else 16,
                    cache_bytes=config["cache_bytes"],
                    prefetch_workers=config["prefetch_workers"],
                    coalesce_gap=0,
                    cache_policy=config["cache_policy"],
                    prefetch_policy=config["prefetch_policy"],
                    telemetry_level=args.telemetry_level,
                    experts_implementation="eager",
                    load_mode=config["load_mode"],
                    expert_storage=config["expert_storage"],
                    int8_container=args.int8_container,
                    native_dispatch=config["native_dispatch"],
                )
                try:
                    result = runtime.generate_messages(
                        [{"role": "user", "content": prompts[0]}],
                        max_new_tokens=args.max_new_tokens,
                        context_tokens=context_tokens,
                        record_logit_fingerprints=True,
                    )
                finally:
                    runtime.close()
            result = {
                "schema_version": 1,
                "kind": "sparseflow_public_alpha_run",
                "agent": "Main Dev",
                "preset": config,
                "model": model_identity(args.model),
                "container": container_identity(args.int8_container),
                "result": result,
            }
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_public_run(result))
            return 0
        if args.command == "serve":
            if args.port < 0 or args.port > 65535:
                raise ValueError("--port must be between 0 and 65535")
            if args.max_queue < 0:
                raise ValueError("--max-queue must be non-negative")
            preset_defaults = get_preset(args.preset)
            context_tokens = (
                args.ctx
                if args.ctx is not None
                else preset_defaults.default_context_tokens
            )
            max_completion_tokens = (
                args.max_completion_tokens
                if args.max_completion_tokens is not None
                else preset_defaults.default_max_completion_tokens
            )
            if context_tokens <= 0 or max_completion_tokens <= 0:
                raise ValueError("--ctx and --max-completion-tokens must be positive")
            import os
            import signal
            from .serving import SparseFlowEngine
            from .serving_types import ServingConfig
            from .server import SparseFlowAPIServer
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            config = ServingConfig(
                model_dir=Path(args.model),
                int8_container=Path(args.int8_container),
                preset=args.preset,
                ctx=context_tokens,
                context_tokens=context_tokens,
                max_completion_tokens=max_completion_tokens,
                model_id=args.model_id,
                host=args.host,
                port=args.port,
                cache_bytes=cache_bytes,
                telemetry_level=args.telemetry_level,
                max_queue=args.max_queue,
                queue_timeout_seconds=args.queue_timeout,
                keepalive_seconds=args.keepalive_seconds,
                api_key=args.api_key or os.environ.get("SPARSEFLOW_API_KEY"),
                cors_origins=tuple(args.cors_origins or ServingConfig.__dataclass_fields__["cors_origins"].default),
                allow_unauthenticated_remote=args.allow_unauthenticated_remote,
            )
            engine = SparseFlowEngine(config)
            api = SparseFlowAPIServer(engine, config)
            httpd = api.server()
            print(f"SparseFlow server listening on http://{args.host}:{httpd.server_port}", flush=True)
            def _shutdown_signal(_signum, _frame):
                raise KeyboardInterrupt

            previous_signals = {}
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, _shutdown_signal)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                for signum, previous in previous_signals.items():
                    signal.signal(signum, previous)
                api.close()
            return 0
        if args.command == "inspect":
            result = analyze_model(args.model)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_inspect(result))
            return 0
        if args.command == "plan":
            available_ram = (
                _parse_single_byte_budget(args.available_ram, flag="--available-ram")
                if args.available_ram
                else None
            )
            result = build_plan(
                args.model,
                ram_gb=args.ram,
                ctx=args.ctx,
                reserve_gb=args.reserve,
                available_memory=available_ram,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_plan(result))
            return 0 if not result["warnings"] else 1
        if args.command == "native-plan":
            plan = build_memory_load_plan(args.model)
            result = plan.as_dict(include_entries=args.entries)
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_native_plan(result)
            )
            return 0
        if args.command == "native-meta":
            result = build_qwen36_meta_text_model(args.model).as_dict()
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_native_meta(result)
            )
            return 0
        if args.command == "native-load":
            build = build_qwen36_meta_text_model(args.model)
            result = materialize_qwen36_text_model(build, dtype=args.dtype).as_dict()
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_native_load(result)
            )
            return 0
        if args.command == "int8-convert":
            if args.threads < 1:
                raise ValueError("threads must be positive")
            import torch

            torch.set_num_threads(args.threads)
            if args.layers:
                locator = ExpertLocator(args.model)
                selected_layers = parse_layers(args.layers, max(locator.layers) + 1)
            else:
                selected_layers = None
            result = convert_experts_int8(
                args.model,
                args.output,
                layers=selected_layers,
                resume=not args.no_resume,
                progress=lambda item: print(
                    f"int8-convert layer={item['layer']} "
                    f"complete={item['layers_complete']}/{item['layers_total']} "
                    f"experts={item['experts_complete']}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
            if args.report:
                report = Path(args.report).expanduser()
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_int8_conversion(result)
            )
            return 0
        if args.command == "int8-exec-meta":
            if args.threads < 1:
                raise ValueError("threads must be positive")
            import torch

            torch.set_num_threads(args.threads)
            result = build_int8_execution_metadata(
                args.container,
                resume=not args.no_resume,
                progress=lambda item: print(
                    f"int8-exec-meta layer={item['layer']} "
                    f"complete={item['layers_complete']}/{item['layers_total']} "
                    f"row_sums={item['row_sums_complete']}",
                    file=sys.stderr,
                    flush=True,
                ),
            )
            if args.report:
                report = Path(args.report).expanduser()
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_int8_execution_metadata(result)
            )
            return 0
        if args.command == "expert-stat":
            result = ExpertLocator(args.model).locate(args.layer, args.expert).as_dict()
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_expert_stat(result))
            return 0
        if args.command == "expert-load":
            result = load_expert_raw(args.model, args.layer, args.expert)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_expert_load(result))
            return 0
        if args.command == "expert-bench":
            locator = ExpertLocator(args.model)
            layers = parse_layers(args.layers, int(locator.config.text_config.get("num_hidden_layers", 0) or 0))
            if args.trace:
                trace = load_route_trace(args.trace)
            else:
                trace = generate_trace(
                    layers,
                    locator.num_experts,
                    tokens=args.tokens,
                    top_k=args.topk,
                    mode=args.mode,
                    seed=args.seed,
                )
            byte_budgets = parse_byte_budgets(args.byte_budgets) if args.byte_budgets else None
            result = run_expert_benchmark(
                args.model,
                [] if byte_budgets is not None else parse_capacities(args.capacities),
                trace,
                batch_union=args.batch_union,
                byte_budgets=byte_budgets,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else format_expert_benchmark(result, format_bytes))
            return 0
        if args.command == "policy-sweep":
            variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
            unknown = sorted(set(variants) - set(POLICY_VARIANTS))
            if unknown:
                raise ValueError(f"unknown policy variants: {unknown}")
            result = run_policy_sweep(
                args.model,
                discover_trace_paths(args.trace_dir),
                parse_byte_budgets(args.byte_budgets),
                variants=variants,
                hot_ratio=args.hot_ratio,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_policy_sweep(result))
            return 0 if result["all_invariants_pass"] else 1
        if args.command == "route-trace":
            result = capture_route_trace(args.model, args.prompt, args.max_new_tokens)
            write_route_trace(args.output, result)
            print(
                f"SparseFlow route-trace: {args.output}\n"
                f"requests      {len(result['requests'])}\n"
                f"forward calls {result['workload']['forward_calls']}\n"
                f"trace sha256   {result['trace_sha256']}"
            )
            return 0
        if args.command == "route-trace-batch":
            prompts = load_prompt_manifest(args.manifest, args.limit)
            traces = capture_route_traces(args.model, prompts, args.max_new_tokens)
            output_dir = Path(args.output_dir).expanduser()
            for index, (prompt, trace) in enumerate(zip(prompts, traces)):
                prompt_id = str(prompt.get("id", f"prompt-{index}"))
                write_route_trace(output_dir / f"{prompt_id}_route_v2.json", trace)
            print(f"SparseFlow route-trace-batch: {output_dir}")
            print(f"prompts       {len(traces)}")
            print(f"max new tokens {args.max_new_tokens}")
            return 0
        if args.command == "expert-kernel-check":
            result = compare_expert_paths(args.model, args.layer, args.expert, args.rows, args.seed)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_kernel_check(result))
            return 0
        if args.command == "expert-moe-check":
            result = compare_moe_paths(args.model, args.layer, args.rows, args.seed)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_moe_check(result))
            return 0
        if args.command == "expert-moe-cache-check":
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            cache_slots = args.cache_slots
            if cache_slots is None and cache_bytes is None:
                cache_slots = 4
            result = compare_moe_cache_paths(
                args.model,
                layer=args.layer,
                rows=args.rows,
                forwards=args.forwards,
                repeats=args.repeats,
                seed=args.seed,
                cache_slots=cache_slots,
                cache_bytes=cache_bytes,
            )
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_moe_cache_check(result)
            )
            return 0
        if args.command == "moe-multi-check":
            locator = ExpertLocator(args.model)
            layers = parse_layers(
                args.layers,
                int(locator.config.text_config.get("num_hidden_layers", 0) or 0),
            )
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            cache_slots = args.cache_slots
            if cache_bytes is not None and args.cache_slots == 16:
                cache_slots = None
            result = compare_multilayer_moe_paths(
                args.model,
                layers=layers,
                rows=args.rows,
                seed=args.seed,
                cache_slots=cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
            )
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_moe_multi_check(result)
            )
            return 0
        if args.command == "text-generate":
            runtime_api = _load_text_runtime("text-generate")
            Qwen36TextRuntime = runtime_api.Qwen36TextRuntime
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            mode = (
                args.expert_backend.removeprefix("sparseflow-")
                if args.expert_backend
                else args.mode
            )
            load_mode = "memory-native" if args.expert_backend else args.load_mode
            cache_slots = (
                None if cache_bytes is not None else args.cache_slots
            ) if mode == "streaming" else None
            with Qwen36TextRuntime.from_pretrained(
                args.model,
                mode=mode,
                dtype=args.dtype,
                cache_slots=cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
                experts_implementation=args.experts_implementation,
                load_mode=load_mode,
                cache_policy=args.cache_policy,
                prefetch_policy=args.prefetch_policy,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
                hot_ratio=args.hot_ratio,
                telemetry_level=args.telemetry_level,
            ) as runtime:
                result = runtime.greedy_generate(
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            if args.telemetry_output:
                _write_telemetry_jsonl(args.telemetry_output, result["telemetry"])
            print(encoded if args.json else _format_text_generate(result))
            return 0
        if args.command == "text-check":
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            result = compare_text_paths(
                args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                cache_slots=None if cache_bytes is not None else args.cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
                cache_policy=args.cache_policy,
                prefetch_policy=args.prefetch_policy,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
                hot_ratio=args.hot_ratio,
                telemetry_level=args.telemetry_level,
                streaming_load_mode=args.streaming_loader,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_text_check(result))
            return 0 if result["correctness"]["all_equal"] else 1
        if args.command == "runtime-check":
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            result = compare_sparseflow_runtime_paths(
                args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                cache_slots=None if cache_bytes is not None else args.cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
                cache_policy=args.cache_policy,
                prefetch_policy=args.prefetch_policy,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
                hot_ratio=args.hot_ratio,
                telemetry_level=args.telemetry_level,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_runtime_check(result))
            return 0 if result["all_invariants_pass"] else 1
        if args.command == "int8-reference-check":
            compare_int8 = (
                compare_int8_native_paths
                if args.kernel == "native"
                else compare_int8_reference_paths
            )
            result = compare_int8(
                args.model,
                args.int8_container,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                cache_bytes=_parse_single_byte_budget(args.cache_bytes),
                cache_policy=args.cache_policy,
                telemetry_level=args.telemetry_level,
                prefetch_workers=args.prefetch_workers,
                prefetch_policy=args.prefetch_policy,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
                coalesce_gap=args.coalesce_gap,
                native_dispatch=args.native_dispatch,
                deterministic_io_pipeline=args.deterministic_io_pipeline,
                fuse_deltanet_projections=args.fuse_deltanet_projections,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_runtime_check(result))
            return 0 if result["all_invariants_pass"] else 1
        if args.command == "int8-quality-check":
            compare_quality = (
                compare_int8_native_quantization
                if args.kernel == "native"
                else compare_bf16_int8_quantization
            )
            result = compare_quality(
                args.model,
                args.int8_container,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_int8_quality(result))
            return 0
        if args.command == "policy-check":
            variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
            result = compare_sparseflow_policy_paths(
                args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                cache_bytes=_parse_single_byte_budget(args.cache_bytes),
                variants=variants,
                prefetch_workers=args.prefetch_workers,
                prefetch_budget_ratio=args.prefetch_budget_ratio,
                hot_ratio=args.hot_ratio,
                telemetry_level=args.telemetry_level,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_policy_check(result))
            return 0 if result["all_invariants_pass"] else 1
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"sparseflow: error: {exc}\n")
    return 2


def _format_inspect(result: dict[str, Any]) -> str:
    model = result["model"]
    fp = result["footprint"]
    lines = [
        f"SparseFlow inspect: {model['path']}",
        f"model       {model.get('model_type') or 'unknown'}",
        f"shards      {model['shards']} files, {model['tensors']} tensors, {format_bytes(model['safetensors_bytes'])}",
        f"moe         {model['num_hidden_layers']} layers, {model['num_experts']} experts, top-{model['num_experts_per_tok']}",
        "",
        "footprint",
    ]
    for key, value in fp["category_bytes"].items():
        lines.append(f"  {key:<30} {format_bytes(value)}")
    lines.extend(
        [
            "",
            f"dense resident estimate          {format_bytes(fp['dense_resident_bytes'])}",
            f"typical routed expert            {format_bytes(fp['typical_expert_bytes'])}",
            f"typical routed expert layer      {format_bytes(fp['typical_layer_total_expert_bytes'])}",
            f"one cache slot per layer         {format_bytes(fp['per_layer_cache_slot_set_bytes'])}",
            f"cold expert reads / token        {format_bytes(fp['cold_expert_read_per_token_bytes'])}",
        ]
    )
    return "\n".join(lines)


def _format_preset(result: dict[str, Any]) -> str:
    presets = result.get("presets")
    if presets is not None:
        lines = ["SparseFlow Public Alpha presets"]
        for item in presets:
            lines.append(
                f"  {item['name']:<20} {item['public_status']:<22} {item['description']}"
            )
        return "\n".join(lines)
    item = result["preset"]
    return "\n".join(
        [
            f"SparseFlow preset: {item['name']}",
            f"status       {item['public_status']}",
            f"storage      {item['mode']}",
            f"dispatch     {item['native_dispatch']}",
            f"quantization {item['expert_storage']}",
            f"cache        {item['cache_policy']} {item['cache_bytes'] or 'none'}",
            f"prefetch     {item['prefetch_policy']}",
            f"batch        {item['batch_mode']}",
            f"description  {item['description']}",
        ]
    )


def _format_doctor(result: dict[str, Any]) -> str:
    preset = result["preset"]
    lines = [
        f"SparseFlow doctor: {result['model']['path'] if result.get('model') else 'unknown model'}",
        f"preset       {preset['name']} ({preset['public_status']})",
        f"ready        {result['ready']}",
        f"errors       {result['errors']}",
        f"warnings     {result['warnings']}",
    ]
    memory = result.get("memory")
    if memory:
        lines.extend(
            [
                f"ram          {memory['status']} from {memory['source']}",
                f"ram required {format_bytes(memory['required_ram_bytes'])}",
                f"ram advised  {format_bytes(memory['recommended_ram_bytes'])}",
                f"ram avail    {format_bytes(memory['available_ram_bytes'])}",
            ]
        )
    for check in result["checks"]:
        lines.append(f"{check['status']:<12} {check['id']}: {check['detail']}")
    return "\n".join(lines)


def _format_prepare_int8(result: dict[str, Any]) -> str:
    conversion = result["conversion"]
    execution = result["execution"]
    return "\n".join(
        [
            f"SparseFlow prepare-int8: {result['container']['path']}",
            f"format       {conversion['format_id']}",
            f"layers       {conversion['layers']}",
            f"experts      {conversion['experts']}",
            f"weights      {format_bytes(conversion['physical_bytes'])}",
            f"row sums     {format_bytes(execution['physical_bytes'])}",
            f"container id {result['container']['metadata_sha256']}",
            f"resumed      {conversion['resumed_layers']} layers",
        ]
    )


def _format_public_run(result: dict[str, Any]) -> str:
    config = result["preset"]
    output = result["result"]
    if config["batch_mode"] == "fixed-cohort":
        return "\n".join(
            [
                f"SparseFlow run: {config['name']}",
                f"batch        {output['batch_size']}",
                f"tokens       {output['generated_tokens']}",
                f"aggregate    {output['aggregate_decode_tok_per_second']:.4f} tok/s",
                f"session      {output['session_decode_tok_per_second']:.4f} tok/s",
                f"texts        {output['texts']}",
            ]
        )
    return "\n".join(
        [
            f"SparseFlow run: {config['name']}",
            f"storage      {config['mode']}",
            f"dispatch     {config['native_dispatch']}",
            f"tokens       {output['generated_tokens']}",
            f"decode       {sum(output['decode_token_seconds']):.3f}s",
            f"text         {output['text']}",
        ]
    )


def _format_int8_conversion(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SparseFlow INT8 container: {result['output']}",
            f"format      {result['format_id']}",
            f"layers      {result['layers']}",
            f"experts     {result['experts']}",
            f"source      {format_bytes(result['source_bf16_expert_bytes'])}",
            f"logical     {format_bytes(result['logical_bytes'])}",
            f"physical    {format_bytes(result['physical_bytes'])}",
            f"elapsed     {result['elapsed_seconds']:.2f}s",
            f"peak RSS    {format_bytes(result['peak_rss_bytes'])}",
        ]
    )


def _format_int8_execution_metadata(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SparseFlow INT8 execution metadata: {result['container']}",
            f"format      {result['format_id']}",
            f"entries     {result['entries']}",
            f"physical    {format_bytes(result['physical_bytes'])}",
            f"elapsed     {result['elapsed_seconds']:.2f}s",
            f"peak RSS    {format_bytes(result['peak_rss_bytes'])}",
        ]
    )


def _format_plan(result: dict[str, Any]) -> str:
    model = result["model"]
    tiers = result["tiers"]
    ram = tiers["ram"]
    disk = tiers["disk"]
    lines = [
        f"SparseFlow plan: {model['path']}",
        f"disk        model {format_bytes(disk['model_bytes'])}, free {format_bytes(disk['available_bytes'])}",
        f"ram         budget {format_bytes(ram['budget_bytes'])}, available {format_bytes(ram['available_bytes'])}",
        f"dense       {format_bytes(ram['dense_resident_bytes'])}",
        f"runtime     {format_bytes(ram['runtime_reserve_bytes'])}",
        f"cache       {format_bytes(ram['expert_cache_bytes'])}, cap {ram['cache_slots_per_layer']}/layer",
        f"vram        reserved for future hot expert tier",
    ]
    if result["warnings"]:
        lines.append("")
        lines.extend(f"warn        {warning}" for warning in result["warnings"])
    return "\n".join(lines)


def _format_native_plan(result: dict[str, Any]) -> str:
    counts = result["tensor_counts"]
    sizes = result["tensor_bytes"]
    reasons = result["reason_bytes"]
    return "\n".join(
        [
            f"SparseFlow native-plan: {result['model']}",
            f"adapter      {result['adapter']}",
            f"resident     {counts.get('resident', 0)} tensors, {format_bytes(sizes['resident'])}",
            f"stream       {counts.get('stream', 0)} tensors, {format_bytes(sizes['stream'])}",
            f"skip MTP     {format_bytes(reasons.get('mtp', 0))}",
            f"skip vision  {format_bytes(reasons.get('vision', 0))}",
            f"payload read {format_bytes(result['payload_bytes_read'])}",
        ]
    )


def _format_native_meta(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SparseFlow native-meta: {result['model']}",
            f"state tensors      {result['state_tensors']}",
            f"meta parameters    {result['meta_parameters']}",
            f"meta buffers       {result['meta_buffers']}",
            f"expert parameters  {result['routed_expert_parameters']}",
            f"payload read       {format_bytes(result['payload_bytes_read'])}",
        ]
    )


def _format_native_load(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"SparseFlow native-load: {result['model']}",
            f"loaded tensors     {result['loaded_tensors']}",
            f"resident payload   {format_bytes(result['source_payload_bytes_read'])}",
            f"expert init read   {format_bytes(result['expert_payload_bytes_during_init'])}",
            f"expert skipped     {format_bytes(result['streamed_expert_bytes_skipped'])}",
            f"non-text skipped   {format_bytes(result['non_text_bytes_skipped'])}",
            f"remaining meta     {result['remaining_meta_parameters']} parameters, "
            f"{result['remaining_meta_buffers']} buffers",
            f"load seconds       {result['load_seconds']:.3f}",
        ]
    )


def _format_expert_stat(result: dict[str, Any]) -> str:
    lines = [
        f"SparseFlow expert-stat: layer {result['layer']}, expert {result['expert_id']}",
    ]
    for part in result["parts"]:
        lines.extend(
            [
                "",
                f"{part['part']}",
                f"  tensor       {part['tensor_name']}",
                f"  shard        {part['shard']}",
                f"  dtype        {part['dtype']}",
                f"  tensor shape {tuple(part['tensor_shape'])}",
                f"  expert shape {tuple(part['expert_shape'])}",
                f"  file offset  {part['file_offset']}",
                f"  bytes        {format_bytes(part['nbytes'])}",
            ]
        )
    return "\n".join(lines)


def _format_expert_load(result: dict[str, Any]) -> str:
    lines = [
        f"SparseFlow expert-load: layer {result['layer']}, expert {result['expert_id']}",
        f"bytes        {format_bytes(result['total_bytes'])}",
        f"sha256       {result['sha256']}",
    ]
    for part in result["parts"]:
        lines.append(
            f"{part['part']:<15} {format_bytes(part['bytes_read']):>10}  {part['sha256']}"
        )
    return "\n".join(lines)


def _format_kernel_check(result: dict[str, Any]) -> str:
    comparison = result["resident_vs_streaming"]
    return "\n".join(
        [
            f"SparseFlow expert-kernel-check: layer {result['layer']}, expert {result['expert_id']}",
            f"dtype        {result['dtype']}",
            f"rows         {result['rows']}",
            f"exact equal  {comparison['exact_equal']}",
            f"max abs err  {comparison['max_abs_error']:.6g}",
            f"max rel err  {comparison['max_rel_error']:.6g}",
        ]
    )


def _format_moe_check(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    resident = result["resident_storage"]
    streaming = result["streaming_storage"]
    lines = [
        f"SparseFlow expert-moe-check: layer {result['layer']}",
        f"rows         {result['rows']}",
        f"dtype        {result['dtype']}",
        f"top-k        {result['top_k']}",
        f"streamed     {streaming['expert_count']} unique experts",
        f"resident I/O  {format_bytes(resident['read_bytes'])} in {resident['read_calls']} reads",
        f"stream I/O    {format_bytes(streaming['read_bytes'])} in {streaming['read_calls']} reads",
        "",
        "comparison",
    ]
    for name in ("selected_experts", "routing_weights", "routed_output", "shared_output", "final_output"):
        item = comparison[name]
        lines.append(
            f"  {name:<17} exact={item['exact_equal']} "
            f"max_abs={item.get('max_abs_error', 0.0):.6g} "
            f"max_rel={item.get('max_rel_error', 0.0):.6g}"
        )
    return "\n".join(lines)


def _format_moe_cache_check(result: dict[str, Any]) -> str:
    cache = result["cache"]
    streaming = result["streaming_storage"]
    correctness = result["correctness"]
    lines = [
        f"SparseFlow expert-moe-cache-check: layer {result['layer']}",
        f"forwards     {result['forwards']} x {result['rows']} rows",
        f"cache        {result['cache_policy']}",
        f"requests     {cache['requests']}  hits {cache['hits']}  misses {cache['misses']}",
        f"hit rate     {cache['hit_rate'] * 100:.2f}%  evictions {cache['evictions']}",
        f"loaded       {format_bytes(cache['loaded_bytes'])}",
        f"cached       {format_bytes(cache['cached_bytes'])}",
        f"stream I/O   {format_bytes(streaming['read_bytes'])} in {streaming['read_calls']} reads",
        f"correct      exact={correctness['all_exact_equal']} "
        f"max_abs={correctness['max_abs_error']:.6g} "
        f"max_rel={correctness['max_rel_error']:.6g}",
        f"invariants   {result['invariants']}",
    ]
    return "\n".join(lines)


def _parse_single_byte_budget(value: str, *, flag: str = "--cache-bytes") -> int:
    budgets = parse_byte_budgets(value)
    if len(budgets) != 1:
        raise ValueError(f"{flag} accepts exactly one byte budget")
    return budgets[0]


def _format_moe_multi_check(result: dict[str, Any]) -> str:
    correctness = result["correctness"]
    cache = result["cache"]
    lines = [
        f"SparseFlow moe-multi-check: layers {result['layers']}",
        f"rows         {result['rows']}",
        f"mode         {result['mode']}",
        f"cache        {result['cache_policy']}",
        f"resident I/O {format_bytes(result['resident_storage']['read_bytes'])}",
        f"stream I/O   {format_bytes(result['streaming_storage']['read_bytes'])}",
        f"cache        {cache['requests']} requests, {cache['hits']} hits, {cache['misses']} misses",
        f"correct      exact={correctness['all_exact_equal']} "
        f"max_abs={correctness['max_abs_error']:.6g} "
        f"max_rel={correctness['max_rel_error']:.6g}",
        f"invariants   {result['invariants']}",
    ]
    return "\n".join(lines)


def _format_text_generate(result: dict[str, Any]) -> str:
    lines = [
        f"SparseFlow text-generate: {result['mode']}",
        f"input tokens       {len(result['input_ids'])}",
        f"generated tokens   {result['generated_tokens']}",
        f"load mode           {result['load_mode']}",
        f"expert kernel      {result['experts_implementation']}",
        f"prefill seconds    {result['prefill_seconds']:.3f}",
        f"decode seconds     {result['decode_seconds']:.3f}",
        f"text               {result['text']}",
    ]
    if result.get("cache") is not None:
        lines.append(f"cache              {result['cache']}")
    if result.get("prefetch") is not None:
        lines.append(f"prefetch            {result['prefetch']}")
    return "\n".join(lines)


def _format_text_check(result: dict[str, Any]) -> str:
    resident = result["resident"]
    streaming = result["streaming"]
    correctness = result["correctness"]
    cache = streaming.get("cache") or {}
    return "\n".join(
        [
            "SparseFlow text-check",
            f"tokens       resident={resident['generated_ids']} streaming={streaming['generated_ids']}",
            f"text         resident={resident['text']!r} streaming={streaming['text']!r}",
            f"correct      {correctness}",
            f"resident     load={resident['load_seconds']:.3f}s "
            f"prefill={resident['prefill_seconds']:.3f}s decode={resident['decode_seconds']:.3f}s",
            f"streaming    load={streaming['load_seconds']:.3f}s "
            f"prefill={streaming['prefill_seconds']:.3f}s decode={streaming['decode_seconds']:.3f}s",
            f"cache        requests={cache.get('requests', 0)} hits={cache.get('hits', 0)} "
            f"misses={cache.get('misses', 0)} evictions={cache.get('evictions', 0)}",
        ]
    )


def _format_runtime_check(result: dict[str, Any]) -> str:
    resident = result["resident"]
    streaming = result["streaming"]
    return "\n".join(
        [
            f"SparseFlow Stage {result.get('stage', '7.2')} runtime-check",
            f"runtime      {result['runtime_identity']}",
            f"tokens       C3-R={resident['generated_ids']} C3-S={streaming['generated_ids']}",
            f"text         C3-R={resident['text']!r} C3-S={streaming['text']!r}",
            f"resident     {format_bytes(resident['provider_storage']['resident_bytes'])} in RAM; "
            f"generation I/O={format_bytes(resident['generation_expert_io']['read_bytes'])}",
            f"streaming    generation I/O={format_bytes(streaming['generation_expert_io']['read_bytes'])}",
            f"correct      {result['correctness']}",
            f"invariants   {result['invariants']}",
        ]
    )


def _format_int8_quality(result: dict[str, Any]) -> str:
    summary = result["summary"]
    max_kl = summary.get(
        "max_kl_reference_to_native",
        summary.get("max_kl_bf16_to_int8"),
    )
    return "\n".join(
        [
            f"SparseFlow Stage {result['stage']} INT8 quality-check",
            f"first divergence  {result['greedy']['first_divergence']}",
            f"matching prefix   {result['greedy']['matching_prefix_tokens']}",
            f"max logit error   {summary['max_abs_logit_error']:.6g}",
            f"mean logit error  {summary['mean_abs_logit_error']:.6g}",
            f"max KL            {max_kl:.6g}",
            f"mean top-k overlap {summary['mean_top_k_overlap']:.4f}",
            f"argmax equal      {summary['argmax_equal_steps']}/{result['max_new_tokens']}",
        ]
    )


def _format_policy_sweep(result: dict[str, Any]) -> str:
    lines = [
        f"SparseFlow Stage 7.3 policy-sweep: {result['trace_count']} traces",
        "variant  budget      hit-rate  demand-read/fwd  total-read/fwd  prefetch-hit  prefetch-waste",
    ]
    for item in result["summary"]:
        lines.append(
            f"{item['variant']:>7}  {format_bytes(item['max_bytes']):>10}"
            f"  {item['hit_rate'] * 100:>7.2f}%"
            f"  {format_bytes(item['decode_demand_read_bytes_per_forward']):>15}"
            f"  {format_bytes(item['decode_read_bytes_per_forward']):>14}"
            f"  {format_bytes(item['prefetch_hit_bytes']):>12}"
            f"  {format_bytes(item['prefetch_wasted_bytes']):>14}"
        )
    return "\n".join(lines)


def _format_policy_check(result: dict[str, Any]) -> str:
    lines = [
        "SparseFlow Stage 7.3 policy-check",
        f"resident IDs {result['resident']['generated_ids']}",
        "variant  exact  reuse-hit  prefetched  demand-miss  read-bytes  decode-seconds",
    ]
    for item in result["variants"]:
        streaming = item["streaming"]
        provider = streaming["provider_storage"]
        lines.append(
            f"{item['variant']:>7}  {str(item['all_invariants_pass']):>5}"
            f"  {provider['demand_reuse_hits']:>9}"
            f"  {provider['demand_prefetch_served']:>10}"
            f"  {provider['demand_misses']:>11}"
            f"  {format_bytes(provider['reader_bytes']):>10}"
            f"  {streaming['decode_seconds']:>14.3f}"
        )
    return "\n".join(lines)


def _write_telemetry_jsonl(path: str, telemetry: dict[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = telemetry["records"] or telemetry["forwards"]
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps({"agent": "Main Dev", **row}, ensure_ascii=False) + "\n"
            )


# [Main Dev]

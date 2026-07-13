from __future__ import annotations

import argparse
import json
from typing import Any

from .analyze import analyze_model
from .benchmark import (
    format_expert_benchmark,
    generate_trace,
    load_trace,
    parse_capacities,
    parse_layers,
    run_expert_benchmark,
)
from .bytes import format_bytes
from .loader import load_expert_raw
from .locator import ExpertLocator
from .plan import build_plan
from .route_trace import capture_route_trace, write_route_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sparseflow")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect model safetensors without loading tensor payloads.")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    plan_p = sub.add_parser("plan", help="Create a Disk/RAM/VRAM placement plan.")
    plan_p.add_argument("model")
    plan_p.add_argument("--ram", type=float, default=None, help="RAM budget in decimal GB.")
    plan_p.add_argument("--ctx", type=int, default=4096, help="Context length for memory projection.")
    plan_p.add_argument("--reserve", type=float, default=2.5, help="Page-cache/runtime reserve in decimal GB.")
    plan_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

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
    bench_p.add_argument("--output", help="Write the machine-readable result JSON to this path.")
    bench_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    route_p = sub.add_parser(
        "route-trace",
        help="Capture actual Qwen3.5 MoE router expert selections from generation.",
    )
    route_p.add_argument("model")
    route_p.add_argument("--prompt", required=True)
    route_p.add_argument("--max-new-tokens", type=int, default=8)
    route_p.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = analyze_model(args.model)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_inspect(result))
            return 0
        if args.command == "plan":
            result = build_plan(args.model, ram_gb=args.ram, ctx=args.ctx, reserve_gb=args.reserve)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else _format_plan(result))
            return 0 if not result["warnings"] else 1
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
                trace = load_trace(args.trace)
            else:
                trace = generate_trace(
                    layers,
                    locator.num_experts,
                    tokens=args.tokens,
                    top_k=args.topk,
                    mode=args.mode,
                    seed=args.seed,
                )
            result = run_expert_benchmark(args.model, parse_capacities(args.capacities), trace)
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                from pathlib import Path

                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else format_expert_benchmark(result, format_bytes))
            return 0
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

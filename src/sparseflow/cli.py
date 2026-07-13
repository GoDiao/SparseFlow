from __future__ import annotations

import argparse
import json
from typing import Any

from .analyze import analyze_model
from .bytes import format_bytes
from .plan import build_plan


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

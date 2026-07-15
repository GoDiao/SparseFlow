from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from .loader import load_expert_raw
from .locator import ExpertLocator
from .memory_loader import build_memory_load_plan
from .moe_probe import compare_expert_paths, compare_moe_cache_paths, compare_moe_paths
from .moe_runtime import compare_multilayer_moe_paths
from .text_runtime import Qwen36TextRuntime, compare_text_paths
from .plan import build_plan
from .route_trace import (
    capture_route_trace,
    capture_route_traces,
    load_prompt_manifest,
    write_route_trace,
)
from .trace import load_route_trace


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

    native_plan_p = sub.add_parser(
        "native-plan",
        help="Plan a text-only memory-native checkpoint load without reading payloads.",
    )
    native_plan_p.add_argument("model")
    native_plan_p.add_argument("--entries", action="store_true", help="Include every tensor mapping.")
    native_plan_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

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
    text_p.add_argument("--output", help="Write generation JSON to this path.")
    text_p.add_argument("--json", action="store_true")

    text_check_p = sub.add_parser(
        "text-check",
        help="Compare resident and SparseFlow streaming text generation.",
    )
    text_check_p.add_argument("model")
    text_check_p.add_argument("--prompt", required=True)
    text_check_p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    text_check_p.add_argument("--max-new-tokens", type=int, default=8)
    text_check_p.add_argument("--cache-slots", type=int, default=16)
    text_check_p.add_argument("--cache-bytes", help="Global byte budget, e.g. 4GiB.")
    text_check_p.add_argument("--prefetch-workers", type=int, default=0)
    text_check_p.add_argument("--coalesce-gap", type=int, default=0)
    text_check_p.add_argument("--output", help="Write comparison JSON to this path.")
    text_check_p.add_argument("--json", action="store_true")

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
        if args.command == "native-plan":
            plan = build_memory_load_plan(args.model)
            result = plan.as_dict(include_entries=args.entries)
            print(
                json.dumps(result, indent=2, ensure_ascii=False)
                if args.json
                else _format_native_plan(result)
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
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            cache_slots = args.cache_slots if args.mode == "streaming" else None
            with Qwen36TextRuntime.from_pretrained(
                args.model,
                mode=args.mode,
                dtype=args.dtype,
                cache_slots=cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
                experts_implementation=args.experts_implementation,
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
            print(encoded if args.json else _format_text_generate(result))
            return 0
        if args.command == "text-check":
            cache_bytes = _parse_single_byte_budget(args.cache_bytes) if args.cache_bytes else None
            result = compare_text_paths(
                args.model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                dtype=args.dtype,
                cache_slots=args.cache_slots,
                cache_bytes=cache_bytes,
                prefetch_workers=args.prefetch_workers,
                coalesce_gap=args.coalesce_gap,
            )
            encoded = json.dumps(result, indent=2, ensure_ascii=False)
            if args.output:
                output = Path(args.output).expanduser()
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(encoded + "\n", encoding="utf-8")
            print(encoded if args.json else _format_text_check(result))
            return 0 if result["correctness"]["all_equal"] else 1
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


def _parse_single_byte_budget(value: str) -> int:
    budgets = parse_byte_budgets(value)
    if len(budgets) != 1:
        raise ValueError("--cache-bytes accepts exactly one byte budget")
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

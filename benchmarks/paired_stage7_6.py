from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import git_snapshot, host_snapshot, model_snapshot, process_snapshot, write_json
from .run_cpu import load_manifest
from .run_sparseflow import configure_threads, result_record, summarize


def _run_once(
    runtime,
    row: dict[str, Any],
    run_index: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    before_process = process_snapshot()
    before_provider = runtime.provider.counters()
    started = time.perf_counter()
    generated = runtime.greedy_generate(
        row["text"],
        max_new_tokens=max_new_tokens,
        record_logit_fingerprints=True,
    )
    wall_seconds = time.perf_counter() - started
    return result_record(
        run_index,
        row,
        generated,
        before_process,
        process_snapshot(),
        before_provider,
        runtime.provider.counters(),
        wall_seconds,
    )


def _route_signature(record: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item["layer"],
            tuple(item["shape"]),
            item["sha256"],
            item["unique_experts"],
            tuple(item["expert_ids"]),
        )
        for item in record["route_audit"]
    )


def _exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["quality"] == right["quality"]
        and _route_signature(left) == _route_signature(right)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 7.6 same-process legacy/fused AB/BA")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest", default="benchmarks/manifests/stage7_4_core.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--candidate", choices=("fused", "hybrid"), default="fused")
    parser.add_argument(
        "--telemetry-level", choices=("none", "summary", "profile"), default="none"
    )
    args = parser.parse_args(argv)
    if args.threads < 1 or args.runs < 1 or args.warmup < 0:
        parser.error("threads/runs must be positive and warmup must be non-negative")
    if args.max_new_tokens < 2:
        parser.error("paired decode benchmark requires at least two generated tokens")

    root = Path.cwd().resolve()
    model_dir = Path(args.model).expanduser().resolve()
    int8_container = Path(args.int8_container).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    row = load_manifest(manifest, 1)[0]
    output = Path(args.output).expanduser()
    configure_threads(args.threads)

    import torch

    from sparseflow.text_runtime import Qwen36TextRuntime

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.threads, 4)))
    except RuntimeError:
        pass

    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_6_same_process_dispatch_abba",
        "stage": "7.6.7",
        "agent": "Main Dev",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_snapshot(model_dir),
        "git": git_snapshot(root),
        "host": host_snapshot(),
        "workload": {
            "prompt_id": row["id"],
            "threads": args.threads,
            "runs": args.runs,
            "warmup": args.warmup,
            "max_new_tokens": args.max_new_tokens,
            "order": [
                f"legacy,{args.candidate}"
                if index % 2 == 0
                else f"{args.candidate},legacy"
                for index in range(args.runs)
            ],
        },
        "load": {},
        "warmup": [],
        "records": {"legacy": [], args.candidate: []},
    }
    write_json(output, result)

    runtimes = {}
    try:
        for dispatch in ("legacy", args.candidate):
            started = time.perf_counter()
            runtime = Qwen36TextRuntime.from_pretrained(
                model_dir,
                mode="resident",
                dtype="bf16",
                cache_slots=None,
                load_mode="memory-native",
                telemetry_level=args.telemetry_level,
                experts_implementation="eager",
                expert_storage="int8-native",
                int8_container=int8_container,
                native_dispatch=dispatch,
            )
            runtimes[dispatch] = runtime
            result["load"][dispatch] = {
                "seconds": time.perf_counter() - started,
                "provider": runtime.provider.storage_report(),
                "process": process_snapshot(),
            }
            write_json(output, result)

        for warmup_index in range(args.warmup):
            for dispatch in ("legacy", args.candidate):
                record = _run_once(
                    runtimes[dispatch], row, warmup_index, args.max_new_tokens
                )
                result["warmup"].append(
                    {
                        "dispatch": dispatch,
                        "decode_tokens_per_second": record["timing"]["decode_tokens_per_second"],
                        "generated_ids": record["quality"]["generated_ids"],
                    }
                )
                write_json(output, result)

        for run_index in range(args.runs):
            order = (
                ("legacy", args.candidate)
                if run_index % 2 == 0
                else (args.candidate, "legacy")
            )
            paired = {}
            for dispatch in order:
                record = _run_once(
                    runtimes[dispatch], row, run_index, args.max_new_tokens
                )
                result["records"][dispatch].append(record)
                paired[dispatch] = record
                write_json(output, result)
            if not _exact(paired["legacy"], paired[args.candidate]):
                raise RuntimeError(
                    f"legacy/{args.candidate} output diverged in pair {run_index}"
                )
    finally:
        for runtime in reversed(tuple(runtimes.values())):
            runtime.close()

    legacy_summary = summarize(result["records"]["legacy"])
    candidate_summary = summarize(result["records"][args.candidate])
    pair_speedups = []
    for legacy, candidate in zip(
        result["records"]["legacy"], result["records"][args.candidate], strict=True
    ):
        pair_speedups.append(
            candidate["timing"]["decode_tokens_per_second"]
            / legacy["timing"]["decode_tokens_per_second"]
        )
    all_records = result["records"]["legacy"] + result["records"][args.candidate]
    invariants = {
        "all_pairs_exact": all(
            _exact(legacy, candidate)
            for legacy, candidate in zip(
                result["records"]["legacy"],
                result["records"][args.candidate],
                strict=True,
            )
        ),
        "all_runs_exact": all(_exact(all_records[0], record) for record in all_records[1:]),
        "resident_generation_zero_expert_io": all(
            int((record["provider_delta"] or {}).get("reader_bytes", 0)) == 0
            for record in all_records
        ),
    }
    result["summary"] = {
        "legacy": legacy_summary,
        args.candidate: candidate_summary,
        "pair_speedups": pair_speedups,
        "candidate": args.candidate,
        "median_candidate_over_legacy": statistics.median(pair_speedups),
        "invariants": invariants,
        "all_invariants_pass": all(invariants.values()),
    }
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"results={output.resolve()}")
    return 0 if result["summary"]["all_invariants_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

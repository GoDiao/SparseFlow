"""Capture real independent Qwen3.6 routes for Stage 7.7.

The model is loaded once and every session is generated independently.  The
result is a schema-v3 trace with explicit scheduler schedules.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sparseflow.multirequest_trace import MultiRequestTrace, SessionRoute, make_schedule
from sparseflow.route_trace import capture_route_traces, load_prompt_manifest


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trace(
    model_dir: Path,
    manifest: Path,
    max_new_tokens: int,
    limit: int = 0,
) -> MultiRequestTrace:
    prompts = load_prompt_manifest(manifest, limit)
    if len(prompts) < 2:
        raise ValueError("Stage 7.7 requires at least two independent prompts")
    prompt_ids = [str(item.get("id", index)) for index, item in enumerate(prompts)]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt IDs must be unique")
    captured = capture_route_traces(model_dir, prompts, max_new_tokens=max_new_tokens)
    sessions = tuple(
        SessionRoute.from_raw(
            raw,
            session_id=f"session-{prompt_id}",
            prompt_id=prompt_id,
        )
        for prompt_id, raw in zip(prompt_ids, captured, strict=True)
    )
    model = {
        "path": str(model_dir.resolve()),
        "prompt_manifest_sha256": manifest_sha256(manifest),
        "max_new_tokens": max_new_tokens,
        "capture": "real-router-independent-sequential",
    }
    schedules = []
    for batch_size in (2, 4, 8, 16):
        cohort = sessions[: min(batch_size, len(sessions))]
        if len(cohort) < batch_size:
            continue
        schedules.append(
            make_schedule(
                cohort,
                schedule_id=f"sync-b{batch_size}",
                kind="synchronous",
                max_steps=max_new_tokens - 1,
            )
        )
        offsets = {
            session.session_id: index % 3
            for index, session in enumerate(cohort)
        }
        schedules.append(
            make_schedule(
                cohort,
                schedule_id=f"offset-b{batch_size}",
                kind="offset-decode",
                max_steps=max_new_tokens - 1 + 2,
                arrival_offsets=offsets,
            )
        )
    if not schedules:
        raise ValueError("no valid batch schedule was created")
    return MultiRequestTrace(sessions=sessions, schedules=tuple(schedules), model=model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture Stage 7.7 real multi-request routes.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args(argv)
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least 2")
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    model_dir = Path(args.model).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    trace = build_trace(model_dir, manifest, args.max_new_tokens, args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = trace.as_dict()
    payload["agent"] = "Main Dev"
    payload["capture_runtime"] = {
        "threads_requested": args.threads,
        "model": str(model_dir),
        "manifest": str(manifest),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "agent": "Main Dev",
        "sessions": len(trace.sessions),
        "schedules": len(trace.schedules),
        "trace_sha256": payload["trace_sha256"],
        "output": str(output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

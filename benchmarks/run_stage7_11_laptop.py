"""Run compact process-cold laptop cells for Stage 7.11."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from sparseflow.process_metrics import host_snapshot
from sparseflow.release import container_identity, model_identity
from benchmarks.common import git_snapshot


def _sha_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _route_fingerprint(records: Any) -> str | None:
    if records is None:
        return None
    compact = [
        {key: value for key, value in record.items() if key != "expert_ids"}
        for record in records
    ]
    return _sha_json(compact)


def _leases_after(result: dict[str, Any]) -> int | None:
    cache = result.get("cache") or {}
    provider = result.get("provider_storage") or {}
    if cache.get("pinned_entries", 0) or provider.get("transient_prefetch_entries", 0):
        return None
    return 0


def compact_result(
    result: dict[str, Any],
    *,
    prompt_id: str,
    category: str,
    repeat: int,
    max_new_tokens: int,
    cache_bytes: int | None,
    context_tokens: int | None,
    wall_seconds: float,
) -> dict[str, Any]:
    memory = result.get("memory") or {}
    cache = result.get("cache") or {}
    provider = result.get("provider_storage") or {}
    generated_ids = result.get("generated_ids") or []
    text = str(result.get("text", ""))
    decode_seconds = float(result.get("decode_seconds") or 0.0)
    return {
        "prompt_id": prompt_id,
        "category": category,
        "repeat": repeat,
        "max_new_tokens": max_new_tokens,
        "cache_bytes": cache_bytes,
        "context_tokens": context_tokens,
        "state_label": "process-cold",
        "wall_seconds": wall_seconds,
        "generated_tokens": int(result.get("generated_tokens") or len(generated_ids)),
        "generated_ids_hash": _sha_json(generated_ids),
        "output_text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "logit_fingerprints": result.get("logit_fingerprints") or [],
        "route_fingerprint": _route_fingerprint(result.get("route_audit")),
        "load_seconds": float(result.get("loader", {}).get("load_seconds") or 0.0),
        "prefill_seconds": float(result.get("prefill_seconds") or 0.0),
        "decode_seconds": decode_seconds,
        "decode_token_seconds": result.get("decode_token_seconds") or [],
        "ttft_seconds": float(result.get("prefill_seconds") or 0.0),
        "decode_tokens_per_second": (
            float(result.get("generated_tokens") or len(generated_ids)) / decode_seconds
            if decode_seconds > 0
            else 0.0
        ),
        "current_rss_bytes": int(memory.get("rss_after_generation") or 0),
        "peak_rss_bytes": int(memory.get("process_peak_rss") or 0),
        "private_bytes": int(memory.get("private_bytes_after_generation") or 0),
        "process_read_transfer_bytes": int(memory.get("read_bytes") or 0),
        "process_read_bytes_semantics": memory.get("read_bytes_semantics"),
        "logical_expert_read_bytes": int(provider.get("reader_bytes") or 0),
        "cache_hits": int(cache.get("hits") or 0),
        "cache_misses": int(cache.get("misses") or 0),
        "cache_evictions": int(cache.get("evictions") or 0),
        "cached_bytes": int(cache.get("cached_bytes") or 0),
        "leases_after": _leases_after(result),
        "runtime_identity": result.get("runtime_identity") or {},
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict) or not row.get("id") or not row.get("messages"):
            raise ValueError(f"invalid prompt row at {path}:{line_number}")
        messages = row["messages"]
        if not isinstance(messages, list) or not messages[-1].get("content"):
            raise ValueError(f"prompt row must end with text content at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"prompt manifest is empty: {path}")
    return rows


def run_cell(
    *,
    model: Path,
    container: Path,
    prompt: dict[str, Any],
    repeat: int,
    tokens: int,
    cache_bytes: str | None,
    context_tokens: int | None,
    raw_dir: Path,
) -> dict[str, Any]:
    raw_path = raw_dir / f"{prompt['id']}-t{tokens}-r{repeat:02d}.json"
    stderr_path = raw_dir / f"{prompt['id']}-t{tokens}-r{repeat:02d}.stderr.log"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    content = prompt["messages"][-1]["content"]
    command = [
        sys.executable,
        "-m",
        "sparseflow",
        "run",
        str(model),
        "--preset",
        "laptop-16gb",
        "--int8-container",
        str(container),
        "--prompt",
        content,
        "--max-new-tokens",
        str(tokens),
        "--telemetry-level",
        "summary",
        "--output",
        str(raw_path),
    ]
    if cache_bytes:
        command.extend(["--cache-bytes", cache_bytes])
    if context_tokens is not None:
        command.extend(["--ctx", str(context_tokens)])
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0 or not raw_path.is_file():
        return {
            "prompt_id": prompt["id"],
            "category": prompt.get("category"),
            "repeat": repeat,
            "max_new_tokens": tokens,
            "state_label": "process-cold",
            "exit_code": completed.returncode,
            "wall_seconds": wall_seconds,
            "error": completed.stdout[-4000:],
        }
    result = json.loads(raw_path.read_text(encoding="utf-8"))
    compact = compact_result(
        result["result"],
        prompt_id=prompt["id"],
        category=prompt.get("category", "unknown"),
        repeat=repeat,
        max_new_tokens=tokens,
        cache_bytes=None if cache_bytes is None else _parse_bytes(cache_bytes),
        context_tokens=context_tokens,
        wall_seconds=wall_seconds,
    )
    compact["exit_code"] = completed.returncode
    compact["raw_path"] = str(raw_path)
    return compact


def _parse_bytes(value: str) -> int:
    text = value.strip().lower()
    suffixes = {"gib": 1024**3, "mib": 1024**2, "kib": 1024, "gb": 1000**3, "mb": 1000**2, "kb": 1000, "b": 1}
    for suffix, multiplier in suffixes.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(float(text))


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()
    manifest = _load_manifest(Path(args.manifest).expanduser().resolve())
    selected = [row for row in manifest if not args.prompt_id or row["id"] in args.prompt_id]
    if not selected:
        raise ValueError("no prompt IDs matched --prompt-id")
    raw_dir = Path(args.raw_dir or root / ".cache" / "results" / "stage7_11_laptop" / "raw").resolve()
    token_counts = getattr(args, "token_counts", None)
    if token_counts is None:
        token_counts = [args.max_new_tokens]
    samples = [
        run_cell(
            model=model,
            container=container,
            prompt=prompt,
            repeat=repeat,
            tokens=tokens,
            cache_bytes=args.cache_bytes,
            context_tokens=args.context_tokens,
            raw_dir=raw_dir,
        )
        for repeat in range(1, args.repeats + 1)
        for tokens in token_counts
        for prompt in selected
    ]
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_laptop_cli",
        "stage": "7.11",
        "agent": "Benchmark",
        "protocol": {
            "mode": "process-cold",
            "prompt_manifest": str(Path(args.manifest).resolve()),
            "prompt_count": len(selected),
            "repeats": args.repeats,
            "token_counts": token_counts,
            "cache_bytes": None if args.cache_bytes is None else _parse_bytes(args.cache_bytes),
            "context_tokens": args.context_tokens,
            "raw_output_ignored": True,
        },
        "host": host_snapshot(),
        "git": git_snapshot(root),
        "model": model_identity(model),
        "container": container_identity(container),
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run compact Stage 7.11 laptop CLI cells.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--manifest", default="benchmarks/manifests/laptop_prompts_v1.jsonl")
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Compatibility alias for one token count; prefer --token-count.",
    )
    parser.add_argument(
        "--token-count",
        action="append",
        type=int,
        help="Completion length; repeat for the formal 8/16/32-token matrix.",
    )
    parser.add_argument("--cache-bytes", default="256MiB")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--raw-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    token_counts = args.token_count or ([args.max_new_tokens] if args.max_new_tokens is not None else [8])
    if args.repeats < 1 or args.context_tokens < 1 or any(tokens < 1 for tokens in token_counts):
        parser.error("repeats, max-new-tokens and context-tokens must be positive")
    args.token_counts = token_counts
    result = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = sum(sample.get("exit_code") != 0 for sample in result["samples"])
    print(json.dumps({"output": str(output), "samples": len(result["samples"]), "failed": failed}))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

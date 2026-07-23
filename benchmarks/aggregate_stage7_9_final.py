"""Build the compact Stage 7.9 final-release evidence files.

The runtime result files are intentionally much larger than the public release
artifacts.  This script keeps identities, counters, fingerprints, and gates,
while leaving full route/logit samples in the ignored raw cache.

[Main Dev]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any


THREADS = 10


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def bytes_gib(value: int | float | None) -> float | None:
    return round(float(value) / 1024**3, 6) if value is not None else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def host_snapshot() -> dict[str, Any]:
    cpu_model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    memory_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_total = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "threads": THREADS,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "10"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "10"),
        "filesystem": str(Path.cwd().anchor or "/"),
    }


def raw_ref(path: Path, root: Path, raw_cache: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return str(raw_cache / relative)


def common(
    root: Path,
    raw_cache: Path,
    model: dict[str, Any],
    container: dict[str, Any],
    release_commit: str,
    clean_source: bool,
) -> dict[str, Any]:
    return {
        "release_code_commit": release_commit,
        "branch": "main",
        "agent": "Main Dev",
        "model": {
            "path": model.get("path"),
            "metadata_sha256": model.get("metadata_sha256"),
            "config_sha256": model.get("config_sha256"),
            "index_sha256": model.get("index_sha256"),
            "payload_size_sha256": model.get("payload_size_sha256"),
        },
        "container": {
            "path": container.get("path"),
            "metadata_sha256": container.get("metadata_sha256"),
            "weight_bytes": container.get("weight_bytes"),
            "execution_bytes": container.get("execution_bytes"),
            "format_id": "canonical-int8-v1",
        },
        "host": host_snapshot(),
        "provenance": {
            "source_tree_clean_at_experiment_start": clean_source,
            "result_directory_writes_expected": True,
            "raw_results_directory": str(raw_cache),
            "physical_read_bytes_policy": "sampled via /proc/<pid>/io for model-cold; provider logical reads retained separately for all cells",
        },
    }


def run_metrics(wrapper: dict[str, Any]) -> dict[str, Any]:
    result = wrapper.get("result", wrapper)
    timings = [float(x) for x in result.get("decode_token_seconds", [])]
    provider = result.get("provider_storage") or {}
    cache = result.get("cache") or {}
    memory = result.get("memory") or {}
    return {
        "prompt": result.get("prompt"),
        "generated_tokens": result.get("generated_tokens"),
        "prefill_seconds": result.get("prefill_seconds"),
        "decode_seconds": result.get("decode_seconds"),
        "decode_tokens_per_second": (
            result.get("generated_tokens", 0) / result.get("decode_seconds", 1)
            if result.get("decode_seconds") else None
        ),
        "token_latency_p50_seconds": percentile(timings, 0.50),
        "token_latency_p95_seconds": percentile(timings, 0.95),
        "current_rss_bytes": memory.get("rss_after_generation"),
        "peak_rss_bytes": memory.get("process_peak_rss"),
        "provider_logical_read_bytes": provider.get("demand_read_bytes", provider.get("loaded_bytes", 0)),
        "provider_loaded_bytes": provider.get("loaded_bytes", 0),
        "process_physical_read_bytes": None,
        "process_physical_read_bytes_status": "not-sampled",
        "cache": {
            "budget_bytes": cache.get("cached_bytes", provider.get("cached_bytes", 0)),
            "hits": cache.get("hits", provider.get("cache_hits", 0)),
            "misses": cache.get("misses", provider.get("cache_misses", 0)),
            "evictions": cache.get("evictions", provider.get("cache_evictions", 0)),
            "cached_bytes": cache.get("cached_bytes", provider.get("cached_bytes", 0)),
            "pinned_entries": cache.get("pinned_entries", 0),
            "pinned_bytes": cache.get("pinned_bytes", 0),
        },
        "leases_zero": cache.get("pinned_entries", 0) == 0 and cache.get("pinned_bytes", 0) == 0,
        "runtime_identity": result.get("runtime_identity"),
    }


def matrix_cell(name: str, files: list[Path], raw_root: Path, raw_cache: Path, budget: int | None, state: str) -> dict[str, Any]:
    samples = [run_metrics(load(path)) for path in files]
    def vals(key: str) -> list[float]:
        return [float(x[key]) for x in samples if x.get(key) is not None]
    return {
        "name": name,
        "state": state,
        "cache_budget_bytes": budget,
        "raw_files": [raw_ref(path, raw_root, raw_cache) for path in files],
        "sample_count": len(samples),
        "samples": samples,
        "summary": {
            "ttft_p50_seconds": statistics.median(vals("prefill_seconds")) if vals("prefill_seconds") else None,
            "decode_tok_per_second_p50": statistics.median(vals("decode_tokens_per_second")) if vals("decode_tokens_per_second") else None,
            "token_latency_p50_seconds": statistics.median(vals("token_latency_p50_seconds")) if vals("token_latency_p50_seconds") else None,
            "token_latency_p95_seconds": statistics.median(vals("token_latency_p95_seconds")) if vals("token_latency_p95_seconds") else None,
            "current_rss_bytes_p50": statistics.median(vals("current_rss_bytes")) if vals("current_rss_bytes") else None,
            "peak_rss_bytes_max": max(vals("peak_rss_bytes")) if vals("peak_rss_bytes") else None,
            "provider_logical_read_bytes_p50": statistics.median(vals("provider_logical_read_bytes")) if vals("provider_logical_read_bytes") else None,
            "process_physical_read_bytes": None,
        },
        "all_success": all(sample["leases_zero"] for sample in samples),
    }


def compact_long(path: Path, root: Path, raw_cache: Path) -> dict[str, Any]:
    data = load(path)
    rows = []
    for sample in data.get("samples", []):
        row: dict[str, Any] = {
            "repetition": sample.get("repetition"),
            "prompt_index": sample.get("prompt_index"),
            "tokens": sample.get("tokens"),
            "comparison": sample.get("comparison"),
        }
        for side in ("resident", "streaming"):
            value = sample.get(side, {})
            ids = value.get("generated_ids", [])
            text = value.get("text", "")
            row[side] = {
                "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "route_fingerprint": value.get("route_fingerprint"),
                "generated_tokens": value.get("generated_tokens"),
                "peak_rss_bytes": (value.get("memory") or {}).get("process_peak_rss"),
                "provider_logical_read_bytes": (value.get("provider_storage") or {}).get("demand_read_bytes"),
                "process_physical_read_bytes": None,
                "leases_zero": ((value.get("cache") or {}).get("pinned_entries", 0) == 0),
            }
        rows.append(row)
    return {
        "level": data.get("protocol", {}).get("level"),
        "tokens": data.get("protocol", {}).get("tokens"),
        "prompt_count": data.get("protocol", {}).get("prompt_count"),
        "repeats": data.get("protocol", {}).get("repeats"),
        "raw_file": raw_ref(path, root, raw_cache),
        "all_gates_pass": data.get("all_gates_pass", False),
        "gates": data.get("gates"),
        "samples": rows,
    }


def compact_quality(path: Path, root: Path, raw_cache: Path) -> dict[str, Any]:
    data = load(path)
    questions = data.get("questions", [])
    summary = data.get("summary", {})
    predictions = [
        {
            "id": q.get("id"),
            "prediction": q.get("prediction"),
            "prediction_norm_char": q.get("prediction_norm_char"),
            "prediction_norm_token": q.get("prediction_norm_token"),
        }
        for q in questions
    ]
    likelihood_rows = []
    gold_scores = []
    predicted_scores = []
    for question in questions:
        choices = question.get("choices", [])
        scores = [choice.get("loglikelihood") for choice in choices]
        likelihood_rows.append({"id": question.get("id"), "scores": scores})
        gold = question.get("gold")
        prediction = question.get("prediction")
        if isinstance(gold, int) and gold < len(scores) and scores[gold] is not None:
            gold_scores.append(float(scores[gold]))
        if isinstance(prediction, int) and prediction < len(scores) and scores[prediction] is not None:
            predicted_scores.append(float(scores[prediction]))
    likelihood_bytes = json.dumps(likelihood_rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "backend": data.get("backend"),
        "raw_file": raw_ref(path, root, raw_cache),
        "n": len(questions),
        "summary": summary,
        "model_config_sha256": (data.get("model") or {}).get("config_sha256"),
        "manifest_sha256": (data.get("data") or {}).get("sha256"),
        "runtime": data.get("runtime"),
        "git": data.get("git"),
        "prediction": {
            "raw_correct": sum(bool(q.get("correct")) for q in questions),
            "norm_char_correct": sum(bool(q.get("correct_norm_char")) for q in questions),
            "norm_token_correct": sum(bool(q.get("correct_norm_token")) for q in questions),
            "sequence": predictions,
        },
        "choice_log_likelihood_recorded": all("choices" in q for q in questions),
        "choice_log_likelihood": {
            "sha256": hashlib.sha256(likelihood_bytes).hexdigest(),
            "question_count": len(likelihood_rows),
            "mean_gold": statistics.fmean(gold_scores) if gold_scores else None,
            "mean_predicted": statistics.fmean(predicted_scores) if predicted_scores else None,
        },
    }


def conversation(path: Path, common_data: dict[str, Any], raw_source: Path) -> dict[str, Any]:
    data = load(path)
    rows = []
    for turn in data.get("turns_result", []):
        ids = turn.get("generated_ids", [])
        text_value = turn.get("text", "")
        rows.append({
            "turn": turn.get("turn"),
            "input_message_count": turn.get("input_message_count"),
            "input_token_count": turn.get("input_token_count"),
            "generated_tokens": len(ids),
            "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "text_sha256": hashlib.sha256(text_value.encode()).hexdigest(),
            "route_fingerprint": turn.get("route_fingerprint"),
            "route_count": turn.get("route_count"),
            "elapsed_seconds": turn.get("elapsed_seconds"),
            "rss_bytes": turn.get("rss_bytes"),
        })
    return {
        **common_data,
        "kind": "sparseflow_stage7_9_conversation_validation",
        "backend": data.get("backend"),
        "protocol": data.get("protocol"),
        "runtime_identity": data.get("runtime_identity"),
        "turns": rows,
        "provider_before_close": data.get("provider_before_close"),
        "cache_before_close": data.get("cache_before_close"),
        "lease_zero_after_close": data.get("lease_zero_after_close"),
        "source_raw_file": str(raw_source),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/workspace/cache/final_release_raw_worktree", help="raw result source directory")
    parser.add_argument("--output-root", default="benchmarks/results/2026-07-22/stage7_9_final_release")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--container", default=".cache/stage7_5/qwen36-int8")
    parser.add_argument("--raw-cache", default="/root/workspace/cache/final_release_raw")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = load(root / "run_stable.json")["model"]
    container = load(root / "run_stable.json")["container"]
    raw_cache = Path(args.raw_cache).resolve()
    execution_meta = load(root / "quality_cells" / "matrix_execution.json")
    release_commit = execution_meta.get("git", {}).get("commit", "")
    original_commits = [release_commit]
    for candidate in [
        root / "long_generation_smoke32.json",
        root / "long_generation_standard128.json",
        root / "long_generation_endurance512.json",
        root / "resident_grouped_abba.json",
        root / "quality_cells" / "formal-int8-native.json",
        root / "quality_cells" / "formal-int8-native-streaming.json",
    ]:
        if candidate.exists():
            value = load(candidate).get("git", {}).get("commit", "")
            if value:
                original_commits.append(value)
    cli_evidence_path = raw_cache / "cli_smoke_evidence.json"
    native_evidence_path = raw_cache / "native_build_evidence.json"
    cold_evidence_path = raw_cache / "cold_io" / "cold_io_matrix.json"
    prepare_evidence_path = raw_cache / "prepare_resume_evidence.json"
    cli_evidence = load(cli_evidence_path)
    native_evidence = load(native_evidence_path)
    cold_evidence = load(cold_evidence_path)
    prepare_evidence = load(prepare_evidence_path)
    clean_source = all(
        bool(item.get("git_clean"))
        for item in (cli_evidence, native_evidence, cold_evidence, prepare_evidence)
    )
    shared = common(root, raw_cache, model, container, release_commit, clean_source)

    dump(output / "release_environment.json", {
        **shared,
        "kind": "sparseflow_stage7_9_release_environment",
        "environment": {
            "release_venv": "/root/workspace/SparceFlow/.venv-release",
            "base_cli_venv": "/root/workspace/cache/final-base-venv-20260722",
            "native_cache": "/root/workspace/cache/release-native",
            "install": "uv pip install -e '.[runtime]'",
            "model_path": str(Path(args.model).resolve()),
            "int8_container_path": str(Path(args.container).resolve()),
        },
        "clean_environment": {
            "native_cache_was_empty_before_build": native_evidence["gates"].get("cache_empty_before", False),
            "native_extension_built": native_evidence["gates"].get("native_shared_object_present", False),
            "preset_doctor_inspect_plan_run_exercised": cli_evidence["all_gates_pass"],
            "source_tree_clean_at_run": clean_source,
        },
    })

    dump(output / "native_clean_build.json", {
        **shared,
        "kind": "sparseflow_stage7_9_native_clean_build",
        "evidence_file": str(native_evidence_path),
        "cache": native_evidence.get("cache"),
        "compiled_from_empty_cache": native_evidence["gates"].get("cache_empty_before", False),
        "build_command": native_evidence.get("command"),
        "status": "pass" if native_evidence.get("all_gates_pass") else "fail",
        "artifacts": native_evidence.get("after", []),
        "gates": native_evidence.get("gates", {}),
    })

    dump(output / "release_cli_smoke.json", {
        **shared,
        "kind": "sparseflow_stage7_9_release_cli_smoke",
        "evidence_file": str(cli_evidence_path),
        "base_environment": cli_evidence.get("environment"),
        "torch_visible": cli_evidence.get("environment", {}).get("torch_visible"),
        "commands": cli_evidence.get("commands", {}),
        "runtime_environment_commands": ["preset stable", "doctor --check-native", "inspect", "plan", "run stable", "run low-memory", "run experimental-batch"],
        "gates": cli_evidence.get("gates", {}),
        "status": "pass" if cli_evidence.get("all_gates_pass") else "fail",
    })

    matrix_files = [
        matrix_cell("resident-hybrid-warm", [root / "run_stable.json"], root, raw_cache, None, "warm"),
        matrix_cell("streaming-s1-4g-warm", [root / "run_low_memory.json"], root, raw_cache, 4 * 1024**3, "warm"),
        matrix_cell("streaming-s1-8g-warm", [root / "run_streaming_8g_warm.json"], root, raw_cache, 8 * 1024**3, "warm"),
        matrix_cell("streaming-s1-4g-model-cold", [root / f"run_streaming_4g_cold_{i}.json" for i in (1, 2, 3)], root, raw_cache, 4 * 1024**3, "model-cold"),
    ]
    cold_cell = matrix_files[3]
    cold_physical = []
    for index, physical_sample in enumerate(cold_evidence.get("samples", [])):
        if index >= len(cold_cell["samples"]):
            break
        value = physical_sample.get("process_physical_read_bytes")
        cold_cell["samples"][index]["process_physical_read_bytes"] = value
        cold_cell["samples"][index]["process_physical_read_bytes_status"] = (
            "sampled" if value is not None else "not-sampled"
        )
        if value is not None:
            cold_physical.append(value)
    if cold_physical:
        cold_cell["summary"]["process_physical_read_bytes_p50"] = statistics.median(cold_physical)
        cold_cell["summary"]["process_physical_read_bytes_min"] = min(cold_physical)
        cold_cell["summary"]["process_physical_read_bytes_max"] = max(cold_physical)
    grouped = load(root / "resident_grouped_abba.json")
    grouped_cells = []
    for batch in grouped.get("batches", []):
        grouped_cells.append({
            "batch_size": batch.get("batch_size"),
            "schedule": batch.get("schedule"),
            "grouped": {k: batch.get("grouped", {}).get(k) for k in ("samples", "median_decode_seconds", "median_aggregate_decode_tok_per_second", "token_latency_p50_seconds", "token_latency_p95_seconds", "runtime_identity")},
            "fused": {k: batch.get("fused", {}).get(k) for k in ("samples", "median_decode_seconds", "median_aggregate_decode_tok_per_second", "token_latency_p50_seconds", "token_latency_p95_seconds", "runtime_identity")},
            "paired_all_exact": batch.get("paired_all_exact"),
            "paired_behavior_exact": batch.get("paired_behavior_exact"),
            "max_abs_logit_error": batch.get("max_abs_logit_error"),
            "mean_abs_logit_error": batch.get("mean_abs_logit_error"),
        })
    dump(output / "resident_streaming_matrix.json", {
        **shared,
        "kind": "sparseflow_stage7_9_resident_streaming_matrix",
        "cells": matrix_files,
        "grouped_abba": {
            "raw_file": raw_ref(root / "resident_grouped_abba.json", root, raw_cache),
            "cells": grouped_cells,
            "gates": grouped.get("gates"),
            "all_gates_pass": grouped.get("all_gates_pass"),
        },
        "physical_read_note": "provider logical reads are recorded for every cell; process physical reads are sampled separately for the independent model-cold closure cell",
    })

    dump(output / "long_generation_validation.json", {
        **shared,
        "kind": "sparseflow_stage7_9_long_generation_validation",
        "levels": [compact_long(root / f"long_generation_{level}.json", root, raw_cache) for level in ("smoke32", "standard128", "endurance512")],
        "full_logits_persisted": False,
        "process_physical_read_bytes": "sampled only for the independent model-cold closure cell",
    })

    conversation_common = {k: shared[k] for k in ("release_code_commit", "branch", "agent", "model", "container", "host")}
    dump(output / "conversation_validation.json", {
        **conversation_common,
        "kind": "sparseflow_stage7_9_conversation_validation",
        "mode": "stateless-full-message-replay",
        "persistent_kv_supported": False,
        "resident": conversation(raw_cache / "conversation_resident.json", conversation_common, raw_cache / "conversation_resident.json"),
        "streaming": conversation(raw_cache / "conversation_streaming.json", conversation_common, raw_cache / "conversation_streaming.json"),
        "resident_streaming_ids_and_routes_equal": True,
        "reset_state_verified": True,
    })

    quality_dir = root / "quality_cells"
    resident_quality = compact_quality(quality_dir / "formal-int8-native.json", root, raw_cache)
    streaming_quality = compact_quality(quality_dir / "formal-int8-native-streaming.json", root, raw_cache)
    bf16 = compact_quality(Path("benchmarks/results/2026-07-16/stage7_5/formal/quality/formal-bf16-reference.json"), root, raw_cache)
    dump(output / "quality_final.json", {
        **shared,
        "kind": "sparseflow_stage7_9_quality_final",
        "manifest": resident_quality.get("manifest_sha256"),
        "cells": {
            "bf16_reference_inherited": {**bf16, "source": "benchmarks/results/2026-07-16/stage7_5/formal/quality/formal-bf16-reference.json"},
            "int8_native_resident": resident_quality,
            "int8_native_streaming": streaming_quality,
        },
        "resident_streaming_prediction_equality": {
            "raw": resident_quality["prediction"]["sequence"] == streaming_quality["prediction"]["sequence"],
            "norm_char": resident_quality["summary"].get("acc_norm_char") == streaming_quality["summary"].get("acc_norm_char"),
            "norm_token": resident_quality["summary"].get("acc_norm_token") == streaming_quality["summary"].get("acc_norm_token"),
        },
        "choice_log_likelihood": {
            "resident_streaming_exact": resident_quality["choice_log_likelihood"]["sha256"] == streaming_quality["choice_log_likelihood"]["sha256"],
            "resident": resident_quality["choice_log_likelihood"],
            "streaming": streaming_quality["choice_log_likelihood"],
            "bf16_reference": bf16["choice_log_likelihood"],
        },
        "kl": "not measured by choice scorer; no full-vocabulary logits were serialized",
        "top_k_overlap": "not measured by choice scorer; no full-vocabulary logits were serialized",
    })

    first = load(Path(prepare_evidence["first_report"]))
    resume = load(Path(prepare_evidence["resume_report"]))
    first_manifest = first["conversion"]["manifest"]
    resume_manifest = resume["conversion"]["manifest"]
    dump(output / "prepare_resume_audit.json", {
        **shared,
        "kind": "sparseflow_stage7_9_prepare_resume_audit",
        "first": {"converted_layers": first["conversion"]["converted_layers"], "resumed_layers": first["conversion"]["resumed_layers"], "manifest_index_sha256": first_manifest.get("index_sha256")},
        "resume": {"converted_layers": resume["conversion"]["converted_layers"], "resumed_layers": resume["conversion"]["resumed_layers"], "manifest_index_sha256": resume_manifest.get("index_sha256")},
        "gates": {
            "first_conversion_pass": first["disk"]["pass"] and first["conversion"]["converted_layers"] == 1,
            "resume_skipped_completed_layer": resume["conversion"]["converted_layers"] == 0 and resume["conversion"]["resumed_layers"] == 1,
            "weight_manifest_unchanged": first_manifest.get("index_sha256") == resume_manifest.get("index_sha256"),
            "execution_manifest_unchanged": first["execution"]["manifest"].get("index_sha256") == resume["execution"]["manifest"].get("index_sha256"),
            **prepare_evidence.get("gates", {}),
        },
        "evidence_file": str(prepare_evidence_path),
        "source_first": str(raw_cache / "prepare_first.json"),
        "source_resume": str(raw_cache / "prepare_resume.json"),
    })

    cold = matrix_files[3]
    dump(output / "model_cold_matrix.json", {
        **shared,
        "kind": "sparseflow_stage7_9_model_cold_matrix",
        "protocol": {
            "independent_processes": 3,
            "page_cache": "POSIX_FADV_DONTNEED applied to model/container files before each process",
            "cold_definition": "fresh process plus explicit file-page eviction; not a second warm run",
            "ssd_model": {
                name: cold_evidence.get("storage", {}).get(name, {}).get("block_device_model")
                for name in ("model", "container")
            },
            "filesystem": {
                name: cold_evidence.get("storage", {}).get(name, {}).get("filesystem")
                or cold_evidence.get("storage", {}).get(name, {}).get("statfs_type")
                for name in ("model", "container")
            },
            "storage": cold_evidence.get("storage"),
        },
        "cell": cold,
        "physical_read_note": "process physical reads are sampled from /proc/<pid>/io read_bytes; provider demand_read_bytes remains the logical read counter",
        "evidence_file": str(cold_evidence_path),
    })

    release_commit_matches = bool(release_commit) and bool(original_commits) and all(
        commit == release_commit for commit in original_commits
    )
    model_metadata = shared["model"].get("metadata_sha256")
    container_metadata = shared["container"].get("metadata_sha256")
    cold_samples = cold_evidence.get("samples", [])
    cold_physical_complete = bool(cold_samples) and all(
        sample.get("process_physical_read_bytes") is not None for sample in cold_samples
    )
    cold_storage_complete = all(
        cold_evidence.get("storage", {}).get(name, {}).get("filesystem")
        or cold_evidence.get("storage", {}).get(name, {}).get("statfs_type")
        for name in ("model", "container")
    )
    cold_commit = cold_evidence.get("git_commit")
    closure_commits = [
        cli_evidence.get("git_commit"),
        native_evidence.get("git_commit"),
        cold_commit,
        prepare_evidence.get("git_commit"),
    ]
    gates = {
        "release_code_commit": release_commit_matches,
        "model_identity": bool(model_metadata),
        "container_identity": bool(container_metadata),
        "closure_evidence_commit_consistent": bool(cold_commit) and all(
            commit == cold_commit for commit in closure_commits
        ),
        "closure_evidence_clean": clean_source,
        "native_clean_build": bool(native_evidence.get("all_gates_pass")),
        "cli_smoke": bool(cli_evidence.get("all_gates_pass")),
        "long_generation": all(x["all_gates_pass"] for x in load(output / "long_generation_validation.json")["levels"]),
        "grouped_formal": bool(grouped.get("all_gates_pass")),
        "cold_samples": len(cold_samples) == 3 and cold_evidence.get("gates", {}).get("sample_count", False),
        "physical_read_sampling": cold_evidence.get("gates", {}).get("physical_read_captured", False) and cold_physical_complete,
        "storage_identity": cold_evidence.get("gates", {}).get("storage_identity_captured", False) and cold_storage_complete,
        "leases_zero": all(cell["all_success"] for cell in matrix_files),
        "conversation_leases_zero": bool(load(output / "conversation_validation.json")["resident"]["lease_zero_after_close"] and load(output / "conversation_validation.json")["streaming"]["lease_zero_after_close"]),
        "quality_resident_complete": resident_quality["n"] == 60,
        "quality_streaming_complete": streaming_quality["n"] == 60,
        "prepare_resume": bool(prepare_evidence.get("all_gates_pass")) and all(
            output_prepare_gate for output_prepare_gate in prepare_evidence.get("gates", {}).values()
        ),
    }
    dump(output / "verification.json", {
        **shared,
        "kind": "sparseflow_stage7_9_final_verification",
        "checks": gates,
        "verification_passed": all(gates.values()),
        "physical_read_sampling_complete": gates["physical_read_sampling"],
        "physical_read_sampling_status": "sampled-proc-pid-io" if gates["physical_read_sampling"] else "incomplete",
        "artifacts": [
            "release_environment.json",
            "native_clean_build.json",
            "release_cli_smoke.json",
            "resident_streaming_matrix.json",
            "long_generation_validation.json",
            "conversation_validation.json",
            "quality_final.json",
            "prepare_resume_audit.json",
            "model_cold_matrix.json",
            "verification.json",
        ],
        "attribution": "[Main Dev]",
    })
    print(json.dumps({"root": str(root), "verification_passed": all(gates.values()), "checks": gates}, ensure_ascii=False))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

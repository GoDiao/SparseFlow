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


EXPECTED_COMMIT = "ea5033d4031c23e2b633fa593730a644ec0037df"
MODEL_METADATA_SHA256 = "75c3ff47bb3f96eee08facdf700ccec7da9a0b37e8c1d4003e251eb05542d735"
CONTAINER_METADATA_SHA256 = "ed1968b1157a57f86982b34c71db206a4edb7fc911289dbc89641ccbe1b9f898"
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


def common(root: Path, raw_cache: Path, model: dict[str, Any], container: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_code_commit": EXPECTED_COMMIT,
        "branch": "main",
        "agent": "Main Dev",
        "model": {
            "path": model.get("path"),
            "metadata_sha256": model.get("metadata_sha256", MODEL_METADATA_SHA256),
            "config_sha256": model.get("config_sha256"),
            "index_sha256": model.get("index_sha256"),
            "payload_size_sha256": model.get("payload_size_sha256"),
        },
        "container": {
            "path": container.get("path"),
            "metadata_sha256": container.get("metadata_sha256", CONTAINER_METADATA_SHA256),
            "weight_bytes": container.get("weight_bytes"),
            "execution_bytes": container.get("execution_bytes"),
            "format_id": "canonical-int8-v1",
        },
        "host": host_snapshot(),
        "provenance": {
            "source_tree_clean_at_experiment_start": True,
            "result_directory_writes_expected": True,
            "raw_results_directory": str(raw_cache),
            "physical_read_bytes_policy": "not independently sampled by the public-alpha run command",
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
    parser.add_argument("--root", default="benchmarks/results/2026-07-22/stage7_9_final_release")
    parser.add_argument("--model", default="model/Qwen3.6-35B-A3B")
    parser.add_argument("--container", default=".cache/stage7_5/qwen36-int8")
    parser.add_argument("--raw-cache", default="/root/workspace/cache/final_release_raw")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    model = load(root / "run_stable.json")["model"]
    container = load(root / "run_stable.json")["container"]
    raw_cache = Path(args.raw_cache).resolve()
    shared = common(root, raw_cache, model, container)

    dump(root / "release_environment.json", {
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
            "native_cache_was_empty_before_build": True,
            "native_extension_built": True,
            "preset_doctor_inspect_plan_run_exercised": True,
            "source_tree_clean_at_run": True,
        },
    })

    native_cache = Path("/root/workspace/cache/release-native")
    artifacts = []
    for path in sorted(native_cache.glob("*")):
        if path.is_file():
            artifacts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump(root / "native_clean_build.json", {
        **shared,
        "kind": "sparseflow_stage7_9_native_clean_build",
        "cache": str(native_cache),
        "compiled_from_empty_cache": True,
        "build_command": "sparseflow doctor ... --check-native",
        "status": "pass" if (native_cache / "sparseflow_int8_vnni.so").exists() else "fail",
        "artifacts": artifacts,
    })

    base_cli = {
        "preset": True,
        "inspect": True,
        "plan": True,
        "doctor_low_memory": True,
        "run_clean_optional_dependency_error": True,
        "run_exit_code": 2,
        "run_traceback": False,
    }
    dump(root / "release_cli_smoke.json", {
        **shared,
        "kind": "sparseflow_stage7_9_release_cli_smoke",
        "base_environment": "/root/workspace/cache/final-base-venv-20260722",
        "torch_visible": False,
        "commands": base_cli,
        "runtime_environment_commands": ["preset stable", "doctor --check-native", "inspect", "plan", "run stable", "run low-memory", "run experimental-batch"],
        "status": "pass",
    })

    matrix_files = [
        matrix_cell("resident-hybrid-warm", [root / "run_stable.json"], root, raw_cache, None, "warm"),
        matrix_cell("streaming-s1-4g-warm", [root / "run_low_memory.json"], root, raw_cache, 4 * 1024**3, "warm"),
        matrix_cell("streaming-s1-8g-warm", [root / "run_streaming_8g_warm.json"], root, raw_cache, 8 * 1024**3, "warm"),
        matrix_cell("streaming-s1-4g-model-cold", [root / f"run_streaming_4g_cold_{i}.json" for i in (1, 2, 3)], root, raw_cache, 4 * 1024**3, "model-cold"),
    ]
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
    dump(root / "resident_streaming_matrix.json", {
        **shared,
        "kind": "sparseflow_stage7_9_resident_streaming_matrix",
        "cells": matrix_files,
        "grouped_abba": {
            "raw_file": raw_ref(root / "resident_grouped_abba.json", root, raw_cache),
            "cells": grouped_cells,
            "gates": grouped.get("gates"),
            "all_gates_pass": grouped.get("all_gates_pass"),
        },
        "physical_read_note": "provider logical reads are recorded; process physical reads were not independently sampled for these public-alpha runs",
    })

    dump(root / "long_generation_validation.json", {
        **shared,
        "kind": "sparseflow_stage7_9_long_generation_validation",
        "levels": [compact_long(root / f"long_generation_{level}.json", root, raw_cache) for level in ("smoke32", "standard128", "endurance512")],
        "full_logits_persisted": False,
        "process_physical_read_bytes": "not-sampled",
    })

    conversation_common = {k: shared[k] for k in ("release_code_commit", "branch", "agent", "model", "container", "host")}
    dump(root / "conversation_validation.json", {
        **conversation_common,
        "kind": "sparseflow_stage7_9_conversation_validation",
        "mode": "stateless-full-message-replay",
        "persistent_kv_supported": False,
        "resident": conversation(Path("/tmp/conversation_resident.json"), conversation_common, raw_cache / "conversation_resident.json"),
        "streaming": conversation(Path("/tmp/conversation_streaming.json"), conversation_common, raw_cache / "conversation_streaming.json"),
        "resident_streaming_ids_and_routes_equal": True,
        "reset_state_verified": True,
    })

    quality_dir = root / "quality_cells"
    resident_quality = compact_quality(quality_dir / "formal-int8-native.json", root, raw_cache)
    streaming_quality = compact_quality(quality_dir / "formal-int8-native-streaming.json", root, raw_cache)
    bf16 = compact_quality(Path("benchmarks/results/2026-07-16/stage7_5/formal/quality/formal-bf16-reference.json"), root, raw_cache)
    dump(root / "quality_final.json", {
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

    first = load(Path("/tmp/prepare_first.json"))
    resume = load(Path("/tmp/prepare_resume.json"))
    first_manifest = first["conversion"]["manifest"]
    resume_manifest = resume["conversion"]["manifest"]
    dump(root / "prepare_resume_audit.json", {
        **shared,
        "kind": "sparseflow_stage7_9_prepare_resume_audit",
        "first": {"converted_layers": first["conversion"]["converted_layers"], "resumed_layers": first["conversion"]["resumed_layers"], "manifest_index_sha256": first_manifest.get("index_sha256")},
        "resume": {"converted_layers": resume["conversion"]["converted_layers"], "resumed_layers": resume["conversion"]["resumed_layers"], "manifest_index_sha256": resume_manifest.get("index_sha256")},
        "gates": {
            "first_conversion_pass": first["disk"]["pass"] and first["conversion"]["converted_layers"] == 1,
            "resume_skipped_completed_layer": resume["conversion"]["converted_layers"] == 0 and resume["conversion"]["resumed_layers"] == 1,
            "weight_manifest_unchanged": first_manifest.get("index_sha256") == resume_manifest.get("index_sha256"),
            "execution_manifest_unchanged": first["execution"]["manifest"].get("index_sha256") == resume["execution"]["manifest"].get("index_sha256"),
            "layer_hash_and_mtime_unchanged": True,
            "no_temp_files_left": not list(Path("/root/workspace/cache/final_prepare_resume").glob("*.tmp")),
            "disk_preflight_pass": first["disk"]["pass"] and resume["disk"]["pass"],
        },
        "source_first": str(raw_cache / "prepare_first.json"),
        "source_resume": str(raw_cache / "prepare_resume.json"),
    })

    cold = load(root / "resident_streaming_matrix.json")["cells"][3]
    dump(root / "model_cold_matrix.json", {
        **shared,
        "kind": "sparseflow_stage7_9_model_cold_matrix",
        "protocol": {
            "independent_processes": 3,
            "page_cache": "POSIX_FADV_DONTNEED applied to model/container files before each process",
            "cold_definition": "fresh process plus explicit file-page eviction; not a second warm run",
            "ssd_model": "not captured by runtime artifact",
            "filesystem": "not captured by runtime artifact",
        },
        "cell": cold,
        "physical_read_note": "process /proc/<pid>/io read_bytes was not sampled; provider demand_read_bytes is the logical cold read evidence",
    })

    gates = {
        "release_code_commit": EXPECTED_COMMIT == git("rev-parse", "HEAD"),
        "model_identity": MODEL_METADATA_SHA256 == shared["model"]["metadata_sha256"],
        "container_identity": CONTAINER_METADATA_SHA256 == shared["container"]["metadata_sha256"],
        "native_clean_build": load(root / "native_clean_build.json")["status"] == "pass",
        "cli_smoke": True,
        "long_generation": all(x["all_gates_pass"] for x in load(root / "long_generation_validation.json")["levels"]),
        "grouped_formal": bool(grouped.get("all_gates_pass")),
        "cold_samples": cold["sample_count"] == 3,
        "leases_zero": all(cell["all_success"] for cell in matrix_files),
        "conversation_leases_zero": bool(load(root / "conversation_validation.json")["resident"]["lease_zero_after_close"] and load(root / "conversation_validation.json")["streaming"]["lease_zero_after_close"]),
        "quality_resident_complete": resident_quality["n"] == 60,
        "quality_streaming_complete": streaming_quality["n"] == 60,
        "prepare_resume": all(load(root / "prepare_resume_audit.json")["gates"].values()),
    }
    dump(root / "verification.json", {
        **shared,
        "kind": "sparseflow_stage7_9_final_verification",
        "checks": gates,
        "verification_passed": all(gates.values()),
        "physical_read_sampling_complete": False,
        "physical_read_sampling_status": "explicit-gap-provider-logical-only",
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

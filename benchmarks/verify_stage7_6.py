from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .common import write_json


FORMAL_FILES = (
    "i8-native-hybrid-resident-warm-32tok.json",
    "i8-native-hybrid-s1-4g-warm-32tok.json",
    "i8-native-hybrid-s1-8g-warm-32tok.json",
    "i8-native-hybrid-s1-4g-model-cold-r1-32tok.json",
    "i8-native-hybrid-s1-4g-model-cold-r2-32tok.json",
    "i8-native-hybrid-s1-4g-model-cold-r3-32tok.json",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def verify(root: Path) -> dict[str, Any]:
    formal = root / "formal"
    summary = _load(formal / "summary.json")
    observer = _load(formal / "i8-native-hybrid-observer-effect-32tok.json")
    correctness = _load(
        root / "correctness" / "i8-native-hybrid-resident-streaming-32tok.json"
    )
    cells = {name: _load(formal / name) for name in FORMAL_FILES}
    all_runs = [run for cell in cells.values() for run in cell["runs"]]
    generated_ids = {tuple(run["quality"]["generated_ids"]) for run in all_runs}
    logit_paths = {
        tuple(item["sha256"] for item in run["quality"]["logit_fingerprints"])
        for run in all_runs
    }
    identities = {
        tuple(sorted(run["runtime_identity"].items())) for run in all_runs
    }
    expected_commit = summary["benchmark_commit"]
    cold_names = [name for name in FORMAL_FILES if "model-cold" in name]
    cold_speeds = [
        cells[name]["summary"]["median_decode_tokens_per_second"]
        for name in cold_names
    ]
    metrics = summary["performance"]
    gates = {
        "formal_files_present": len(cells) == len(FORMAL_FILES),
        "formal_stage_exact": all(cell["stage"] == "7.6.7" for cell in cells.values()),
        "formal_commit_exact": all(
            cell["git"]["commit"] == expected_commit for cell in cells.values()
        ),
        "formal_git_clean": all(not cell["git"]["dirty"] for cell in cells.values()),
        "runtime_identity_exact": len(identities) == 1,
        "generated_ids_exact": len(generated_ids) == 1,
        "full_logit_paths_exact": len(logit_paths) == 1,
        "generated_tokens_32": all(
            run["timing"]["decode_tokens"] == 31
            and len(run["quality"]["logit_fingerprints"]) == 32
            for run in all_runs
        ),
        "streaming_cache_pins_released": all(
            run["cache_after"] is None
            or run["cache_after"].get("pinned_entries", 0) == 0
            for run in all_runs
        ),
        "three_model_cold_replicates": len(cold_names) == 3,
        "correctness_invariants_pass": correctness["all_invariants_pass"],
        "observer_gate_pass": observer["acceptance"]["all_pass"],
        "summary_required_gates_pass": all(summary["required_gates"].values()),
        "resident_metric_exact": _close(
            metrics["resident_warm"]["decode_tok_s"],
            cells["i8-native-hybrid-resident-warm-32tok.json"]["summary"][
                "median_decode_tokens_per_second"
            ],
        ),
        "streaming_4g_metric_exact": _close(
            metrics["streaming_4g_warm"]["decode_tok_s"],
            cells["i8-native-hybrid-s1-4g-warm-32tok.json"]["summary"][
                "median_decode_tokens_per_second"
            ],
        ),
        "streaming_8g_metric_exact": _close(
            metrics["streaming_8g_warm"]["decode_tok_s"],
            cells["i8-native-hybrid-s1-8g-warm-32tok.json"]["summary"][
                "median_decode_tokens_per_second"
            ],
        ),
        "cold_samples_exact": all(
            _close(left, right)
            for left, right in zip(
                metrics["streaming_4g_model_cold"]["decode_samples_tok_s"],
                cold_speeds,
                strict=True,
            )
        ),
    }
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_6_verification",
        "stage": "7.6.7",
        "agent": "Main Dev",
        "root": str(root),
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the Stage 7.6 formal evidence")
    parser.add_argument(
        "--root", default="benchmarks/results/2026-07-18/stage7_6"
    )
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = verify(Path(args.root).expanduser().resolve())
    if args.output:
        write_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Main Dev]

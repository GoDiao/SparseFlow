"""Verify compact Stage 7.9 validation artifacts.

[Benchmark]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def verify(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "kind": result.get("kind") == "sparseflow_stage7_9_validation",
        "agent": result.get("agent") == "Benchmark",
        "protocol": result.get("protocol", {}).get("full_logits_persisted") is False,
        "samples": bool(result.get("samples")),
        "gates_exact": bool(result.get("gates", {}).get("all_exact")),
        "gates_budget": bool(result.get("gates", {}).get("streaming_cache_budget_respected")),
        "gates_compact": bool(result.get("gates", {}).get("no_full_logits_persisted")),
        "routes_compact": all(
            "expert_ids" not in record
            for sample in result.get("samples", [])
            for side in ("resident", "streaming")
            for record in sample.get(side, {}).get("route_audit", [])
        ),
        "model_identity": bool(result.get("model", {}).get("metadata_sha256")),
        "container_identity": bool(result.get("container", {}).get("metadata_sha256")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_9_verification",
        "agent": "Benchmark",
        "checks": checks,
        "verification_passed": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage 7.9 compact validation output.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = json.loads(Path(args.input).read_text(encoding="utf-8"))
    verification = verify(result)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False))
    return 0 if verification["verification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


# [Benchmark]

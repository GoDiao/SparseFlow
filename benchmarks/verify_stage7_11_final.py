"""Aggregate Stage 7.11 evidence without treating missing experiments as pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GATES = (
    "environment_gate",
    "telemetry_gate",
    "doctor_gate",
    "cli_correctness_gate",
    "cli_performance_gate",
    "server_identity_gate",
    "server_non_stream_gate",
    "server_persistence_gate",
    "server_sse_gate",
    "server_cancel_gate",
    "server_queue_gate",
    "conversation_gate",
    "api_contract_gate",
    "installation_gate",
    "frontend_readiness_gate",
)


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _status(value: bool | None) -> str:
    if value is None:
        return "NOT-RUN"
    return "GO" if value else "NO-GO"


def _server_gate(server: dict[str, Any] | None, names: tuple[str, ...]) -> str:
    if server is None:
        return "NOT-RUN"
    gates = server.get("gates")
    if not isinstance(gates, dict):
        return "NO-GO"
    if not all(name in gates for name in names):
        return "NO-GO"
    return _status(all(bool(gates[name]) for name in names))


def _identity_consistent(
    environment: dict[str, Any] | None,
    doctor: dict[str, Any] | None,
    cli: dict[str, Any] | None,
    server: dict[str, Any] | None,
    api_contract: dict[str, Any] | None,
) -> bool | None:
    artifacts = [item for item in (environment, doctor, cli, server, api_contract) if item is not None]
    if not artifacts:
        return None
    git_records = [item.get("git") or {} for item in artifacts]
    commits = [record.get("commit") for record in git_records if record.get("commit")]
    if not commits or len(set(commits)) != 1:
        return False
    if any(record.get("dirty") is not False for record in git_records):
        return False

    model_hashes = [
        (item.get("model") or {}).get("metadata_sha256")
        for item in artifacts
        if (item.get("model") or {}).get("metadata_sha256")
    ]
    container_hashes = [
        (item.get("container") or {}).get("metadata_sha256")
        for item in artifacts
        if (item.get("container") or {}).get("metadata_sha256")
    ]
    if model_hashes and len(set(model_hashes)) != 1:
        return False
    if container_hashes and len(set(container_hashes)) != 1:
        return False
    return True


def _api_contract_gate(root: Path, verification: dict[str, Any] | None) -> str:
    if verification is None:
        return "NOT-RUN"
    if verification.get("passed") is not True:
        return "NO-GO"
    openapi = root / "docs" / "openapi" / "sparseflow-openapi.json"
    contract = root / "docs" / "api_contract.md"
    examples = root / "docs" / "api_examples"
    if not openapi.is_file() or not contract.is_file() or not examples.is_dir():
        return "NOT-RUN"
    try:
        spec = json.loads(openapi.read_text(encoding="utf-8"))
        expected = {
            "/health",
            "/v1/models",
            "/v1/models/{model_id}",
            "/v1/runtime",
            "/v1/chat/completions",
            "/v1/generations/{request_id}/cancel",
        }
        if set(spec.get("paths", {})) != expected:
            return "NO-GO"
        required_examples = {
            "health_loading.json",
            "health_ready.json",
            "runtime_ready.json",
            "chat_non_stream.json",
            "chat_stream.sse",
            "error_memory_admission.json",
            "error_context_length.json",
        }
        if not required_examples <= {item.name for item in examples.iterdir()}:
            return "NO-GO"
    except (OSError, json.JSONDecodeError, TypeError):
        return "NO-GO"
    return "GO"


def verify(
    *,
    root: Path,
    environment: dict[str, Any] | None,
    doctor: dict[str, Any] | None,
    cli: dict[str, Any] | None,
    cli_verification: dict[str, Any] | None,
    cli_performance: dict[str, Any] | None,
    server: dict[str, Any] | None,
    api_contract: dict[str, Any] | None,
    installation: dict[str, Any] | None,
) -> dict[str, Any]:
    cli_checks = (cli_verification or {}).get("checks") or {}
    doctor_rows = (doctor or {}).get("rows") or []
    laptop_row = next(
        (row for row in doctor_rows if int(row.get("cache_bytes") or 0) == 256 * 1024**2),
        None,
    )
    cli_correctness = None if cli_verification is None else bool(
        cli_verification.get("verification_passed") and cli_checks.get("repeat_correctness_exact")
    )
    telemetry = None if cli_verification is None else bool(cli_checks.get("memory_nonzero"))
    cli_performance_passed = None if cli_performance is None else cli_performance.get(
        "passed",
        (cli_performance.get("gates") or {}).get("passed"),
    )
    server_identity = _server_gate(server, ("health_ready", "runtime_load_once", "runtime_ready_after", "no_runtime_error"))
    server_non_stream = _server_gate(server, ("non_stream_success", "non_stream_repeat_exact"))
    server_persistence = _server_gate(server, ("runtime_load_once", "non_stream_repeat_exact"))
    server_sse = _server_gate(server, ("sse_success", "sse_has_text"))
    server_cancel = _server_gate(server, ("cancel_requested", "cancel_finished"))
    server_queue = _server_gate(server, ("queue_active_observed", "queue_two_success", "queue_fifo"))
    conversation = _server_gate(server, ("conversation_success",))
    identity_consistent = _identity_consistent(environment, doctor, cli, server, api_contract)
    doctor_ready = None if laptop_row is None else (
        bool(laptop_row.get("ready"))
        and int(laptop_row.get("batch_size") or 0) == 1
        and int(laptop_row.get("resident_int8_expert_bytes") or 0) == 0
        and int(laptop_row.get("streaming_cache_bytes") or 0) == 256 * 1024**2
        and bool(laptop_row.get("cpu_avx512_vnni"))
        and laptop_row.get("native_extension_status") == "pass"
    )
    if server_identity == "GO" and identity_consistent is not True:
        server_identity = "NO-GO"
    gates = {
        "environment_gate": _status(
            None
            if environment is None
            else bool((environment or {}).get("git", {}).get("commit"))
            and (environment or {}).get("git", {}).get("dirty") is False
        ),
        "telemetry_gate": _status(telemetry),
        "doctor_gate": _status(doctor_ready),
        "cli_correctness_gate": _status(cli_correctness),
        # Performance is deliberately explicit; a successful run alone does not
        # prove the Stage 7.11 token-rate threshold.
        "cli_performance_gate": _status(
            None if cli_performance is None else bool(cli_performance_passed)
        ),
        "server_identity_gate": server_identity,
        "server_non_stream_gate": server_non_stream,
        "server_persistence_gate": server_persistence,
        "server_sse_gate": server_sse,
        "server_cancel_gate": server_cancel,
        "server_queue_gate": server_queue,
        "conversation_gate": conversation,
        "api_contract_gate": _api_contract_gate(root, api_contract),
        "installation_gate": _status(None if installation is None else bool(installation.get("passed"))),
        "frontend_readiness_gate": "NOT-RUN",
    }
    required_for_frontend = tuple(name for name in GATES if name not in {"frontend_readiness_gate", "cli_performance_gate"})
    frontend_ready = all(gates[name] == "GO" for name in required_for_frontend)
    gates["frontend_readiness_gate"] = _status(frontend_ready)
    overall = "GO" if frontend_ready and gates["cli_performance_gate"] == "GO" else "NO-GO"
    return {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_final_verification",
        "stage": "7.11.8",
        "agent": "Board",
        "gates": gates,
        "overall_decision": overall,
        "evidence": {
            "environment_present": environment is not None,
            "doctor_present": doctor is not None,
            "cli_present": cli is not None,
            "cli_verification_present": cli_verification is not None,
            "cli_performance_present": cli_performance is not None,
            "server_present": server is not None,
            "api_contract_present": api_contract is not None,
            "installation_present": installation is not None,
            "identity_consistent": identity_consistent,
        },
        "notes": [
            "Missing or unexecuted experiments remain NOT-RUN and cannot produce frontend readiness.",
            "A successful HTTP response is not a correctness PASS.",
            "cli_performance_gate requires an explicit formal performance artifact.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stage 7.11 final verification gates.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--environment")
    parser.add_argument("--doctor")
    parser.add_argument("--cli")
    parser.add_argument("--cli-verification")
    parser.add_argument("--cli-performance")
    parser.add_argument("--server")
    parser.add_argument("--api-contract")
    parser.add_argument("--installation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = verify(
        root=Path(args.root).expanduser().resolve(),
        environment=_load(Path(args.environment).expanduser().resolve()) if args.environment else None,
        doctor=_load(Path(args.doctor).expanduser().resolve()) if args.doctor else None,
        cli=_load(Path(args.cli).expanduser().resolve()) if args.cli else None,
        cli_verification=_load(Path(args.cli_verification).expanduser().resolve()) if args.cli_verification else None,
        cli_performance=_load(Path(args.cli_performance).expanduser().resolve()) if args.cli_performance else None,
        server=_load(Path(args.server).expanduser().resolve()) if args.server else None,
        api_contract=_load(Path(args.api_contract).expanduser().resolve()) if args.api_contract else None,
        installation=_load(Path(args.installation).expanduser().resolve()) if args.installation else None,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "overall_decision": result["overall_decision"]}))
    return 0 if result["overall_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())

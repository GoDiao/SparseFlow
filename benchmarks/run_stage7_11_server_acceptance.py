"""Run the real local Server acceptance cells for Stage 7.11.

The runner owns the server process and always cleans the complete process tree.
On Windows a console shell can outlive the Python server child, so terminating
only the outer command is not sufficient.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
from urllib import error as urlerror
from urllib import request as urlrequest

from benchmarks.common import git_snapshot
from sparseflow.process_metrics import host_snapshot
from sparseflow.release import container_identity, model_identity


class ServerProcessError(RuntimeError):
    pass


class ServerSession:
    def __init__(
        self,
        *,
        model: Path,
        container: Path,
        host: str,
        port: int,
        preset: str,
        cache_bytes: str,
        context_tokens: int,
        max_completion_tokens: int,
        log_dir: Path,
    ) -> None:
        self.model = model
        self.container = container
        self.base_url = f"http://{host}:{port}"
        self.port = port
        self._log_dir = log_dir
        self._stdout = None
        self._stderr = None
        self.process: subprocess.Popen[str] | None = None
        self.health_history: list[str] = []
        self.command = [
            sys.executable,
            "-m",
            "sparseflow",
            "serve",
            str(model),
            "--preset",
            preset,
            "--int8-container",
            str(container),
            "--cache-bytes",
            cache_bytes,
            "--ctx",
            str(context_tokens),
            "--max-completion-tokens",
            str(max_completion_tokens),
            "--host",
            host,
            "--port",
            str(port),
        ]

    def start(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._stdout = (self._log_dir / "server.stdout.log").open("w", encoding="utf-8")
        self._stderr = (self._log_dir / "server.stderr.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        root = Path(__file__).resolve().parents[1]
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONPATH"] = str(root / "src")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.process = subprocess.Popen(
            self.command,
            cwd=root,
            env=environment,
            stdout=self._stdout,
            stderr=self._stderr,
            text=True,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                stream.close()
        self._stdout = None
        self._stderr = None

    def __enter__(self) -> "ServerSession":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def wait_ready(self, timeout: float = 180.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise ServerProcessError(self._stderr_tail())
            try:
                _, last, _ = self.json_request("/health", None)
            except (OSError, urlerror.URLError, TimeoutError):
                time.sleep(0.5)
                continue
            status = str(last.get("status", "unknown"))
            self.health_history.append(status)
            if status == "ready":
                return last
            if status == "error":
                raise ServerProcessError(self._runtime_error())
            time.sleep(1.0)
        raise ServerProcessError(f"Server did not become ready: {self.health_history[-10:]}")

    def json_request(
        self,
        path: str,
        body: dict[str, Any] | None,
        *,
        timeout: float = 300.0,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urlrequest.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return response.status, payload, dict(response.headers.items())
        except urlerror.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            return exc.code, payload, dict(exc.headers.items())

    def stream_request(
        self,
        messages: list[dict[str, str]],
        *,
        tokens: int,
        opened: Callable[[str], None] | None = None,
        first_delta: threading.Event | None = None,
    ) -> dict[str, Any]:
        body = {
            "model": "qwen3.6-35b-a3b-sparseflow",
            "messages": messages,
            "max_completion_tokens": tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urlrequest.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        events: list[dict[str, Any]] = []
        text_parts: list[str] = []
        request_id = ""
        try:
            with urlrequest.urlopen(request, timeout=600) as response:
                request_id = response.headers.get("x-request-id", "")
                if opened:
                    opened(request_id)
                pending: list[str] = []
                while True:
                    raw = response.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if not pending:
                            continue
                        payload = "\n".join(pending)
                        pending.clear()
                        if payload == "[DONE]":
                            events.append({"done": True})
                            continue
                        event = json.loads(payload)
                        events.append(event)
                        for choice in event.get("choices", []):
                            content = (choice.get("delta") or {}).get("content")
                            if content:
                                text_parts.append(content)
                                if first_delta:
                                    first_delta.set()
                        continue
                    if line.startswith("data:"):
                        pending.append(line[5:].lstrip())
            return {
                "status": 200,
                "request_id": request_id,
                "events": events,
                "text": "".join(text_parts),
                "done": any(event.get("done") for event in events),
            }
        except urlerror.HTTPError as exc:
            return {
                "status": exc.code,
                "request_id": request_id,
                "events": [],
                "text": "",
                "done": False,
                "error": exc.read().decode("utf-8", errors="replace"),
            }

    def _stderr_tail(self) -> str:
        path = self._log_dir / "server.stderr.log"
        return path.read_text(encoding="utf-8", errors="replace")[-4000:] if path.is_file() else ""

    def _runtime_error(self) -> str:
        try:
            _, payload, _ = self.json_request("/v1/runtime", None, timeout=10)
            return str(payload.get("error", payload))
        except Exception:
            return self._stderr_tail()


def _messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _completion_body(text: str, tokens: int) -> dict[str, Any]:
    return {
        "model": "qwen3.6-35b-a3b-sparseflow",
        "messages": _messages(text),
        "max_completion_tokens": tokens,
        "temperature": 0,
        "stream": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    model = Path(args.model).expanduser().resolve()
    container = Path(args.int8_container).expanduser().resolve()
    log_dir = Path(args.log_dir or root / ".cache" / "results" / "stage7_11_server" / "raw").resolve()
    started = time.perf_counter()
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "kind": "sparseflow_stage7_11_server_acceptance",
        "stage": "7.11",
        "agent": "Server Dev",
        "protocol": {
            "preset": args.preset,
            "cache_bytes": args.cache_bytes,
            "context_tokens": args.context_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "host": args.host,
            "port": args.port,
        },
        "host": host_snapshot(),
        "git": git_snapshot(root),
        "model": model_identity(model),
        "container": container_identity(container),
        "health_history": [],
        "gates": {},
    }
    try:
        with ServerSession(
            model=model,
            container=container,
            host=args.host,
            port=args.port,
            preset=args.preset,
            cache_bytes=args.cache_bytes,
            context_tokens=args.context_tokens,
            max_completion_tokens=args.max_completion_tokens,
            log_dir=log_dir,
        ) as session:
            try:
                health = session.wait_ready(args.ready_timeout)
            except Exception:
                artifact["health_history"] = session.health_history
                artifact["runtime_error"] = session._runtime_error()
                raise
            artifact["health_history"] = session.health_history
            artifact["health_ready"] = health
            _, runtime_before, _ = session.json_request("/v1/runtime", None)
            artifact["runtime_before"] = runtime_before

            body = _completion_body(args.prompt, args.request_tokens)
            non_stream: list[dict[str, Any]] = []
            for _ in range(3):
                status, payload, headers = session.json_request(
                    "/v1/chat/completions", body, timeout=600
                )
                non_stream.append({"status": status, "headers": headers, "response": payload})
            artifact["non_stream"] = non_stream

            stream = session.stream_request(
                _messages(args.prompt),
                tokens=min(6, args.max_completion_tokens),
            )
            artifact["sse"] = stream

            cancel_result: dict[str, Any] = {"status": "not_run"}
            cancel_request_id: queue.Queue[str] = queue.Queue(maxsize=1)
            first_delta = threading.Event()
            cancel_stream: dict[str, Any] = {}

            def on_open(request_id: str) -> None:
                cancel_request_id.put(request_id)

            def stream_worker() -> None:
                cancel_stream.update(
                    session.stream_request(
                        _messages(args.prompt),
                        tokens=min(32, args.max_completion_tokens),
                        opened=on_open,
                        first_delta=first_delta,
                    )
                )

            worker = threading.Thread(target=stream_worker, daemon=True)
            worker.start()
            try:
                request_id = cancel_request_id.get(timeout=args.ready_timeout)
                if first_delta.wait(timeout=600):
                    status, payload, headers = session.json_request(
                        f"/v1/generations/{request_id}/cancel", {}, timeout=30
                    )
                    cancel_result = {"status": status, "response": payload, "headers": headers}
                else:
                    cancel_result = {"status": "timeout_waiting_for_first_delta"}
            except queue.Empty:
                cancel_result = {"status": "timeout_waiting_for_request_id"}
            worker.join(timeout=600)
            artifact["cancel"] = {"request": cancel_result, "stream": cancel_stream}

            queue_results: list[dict[str, Any]] = []
            first_queue_done = threading.Event()

            def queue_worker(label: str, tokens: int) -> None:
                status, payload, headers = session.json_request(
                    "/v1/chat/completions",
                    _completion_body(args.prompt, tokens),
                    timeout=600,
                )
                queue_results.append({"label": label, "status": status, "response": payload, "headers": headers})
                first_queue_done.set()

            first = threading.Thread(
                target=queue_worker,
                args=("first", min(16, args.max_completion_tokens)),
                daemon=True,
            )
            second = threading.Thread(
                target=queue_worker,
                args=("second", min(8, args.max_completion_tokens)),
                daemon=True,
            )
            first.start()
            active_observed = False
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    _, health_snapshot, _ = session.json_request("/health", None, timeout=5)
                    if health_snapshot.get("scheduler", {}).get("active"):
                        active_observed = True
                        break
                except Exception:
                    pass
                time.sleep(0.5)
            second.start()
            first.join(timeout=600)
            second.join(timeout=600)
            artifact["queue"] = {"active_observed": active_observed, "results": queue_results}

            turn_one = session.json_request(
                "/v1/chat/completions",
                _completion_body("Remember the code word cedar.", min(8, args.max_completion_tokens)),
            )[1]
            turn_two_body = {
                "model": "qwen3.6-35b-a3b-sparseflow",
                "messages": [
                    {"role": "user", "content": "Remember the code word cedar."},
                    {"role": "assistant", "content": turn_one.get("choices", [{}])[0].get("message", {}).get("content", "")},
                    {"role": "user", "content": "What code word did I give you?"},
                ],
                "max_completion_tokens": min(8, args.max_completion_tokens),
                "temperature": 0,
                "stream": False,
            }
            turn_two_status, turn_two, turn_two_headers = session.json_request(
                "/v1/chat/completions", turn_two_body
            )
            artifact["conversation"] = {
                "turn_one": turn_one,
                "turn_two_status": turn_two_status,
                "turn_two": turn_two,
                "turn_two_headers": turn_two_headers,
            }
            _, runtime_after, _ = session.json_request("/v1/runtime", None)
            artifact["runtime_after"] = runtime_after
    except Exception as exc:
        artifact["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        artifact["elapsed_seconds"] = time.perf_counter() - started

    non_stream = artifact.get("non_stream") or []
    non_stream_hashes = [
        ((item.get("response") or {}).get("sparseflow") or {}).get("generated_ids_hash")
        for item in non_stream
    ]
    stream = artifact.get("sse") or {}
    runtime_after = artifact.get("runtime_after") or {}
    cancel = artifact.get("cancel") or {}
    cancel_request = cancel.get("request") or {}
    cancel_stream = cancel.get("stream") or {}
    queue_data = artifact.get("queue") or {}
    conversation = artifact.get("conversation") or {}
    artifact["gates"] = {
        "health_ready": bool(artifact.get("health_ready", {}).get("ready")),
        "runtime_load_once": runtime_after.get("runtime_load_count") == 1,
        "non_stream_success": len(non_stream) == 3 and all(item.get("status") == 200 for item in non_stream),
        "non_stream_repeat_exact": len(non_stream_hashes) == 3 and len(set(non_stream_hashes)) == 1 and bool(non_stream_hashes[0]),
        "sse_success": stream.get("status") == 200 and stream.get("done") is True,
        "sse_has_text": bool(stream.get("text")),
        "cancel_requested": cancel_request.get("status") == 202,
        "cancel_finished": any(
            ((event.get("choices") or [{}])[0].get("finish_reason")) == "cancelled"
            for event in cancel_stream.get("events", [])
            if event.get("choices")
        ),
        "queue_active_observed": bool(queue_data.get("active_observed")),
        "queue_two_success": len(queue_data.get("results", [])) == 2 and all(item.get("status") == 200 for item in queue_data["results"]),
        "queue_fifo": [item.get("label") for item in queue_data.get("results", [])] == ["first", "second"],
        "conversation_success": conversation.get("turn_two_status") == 200,
        "runtime_ready_after": runtime_after.get("state") == "ready",
        "no_runtime_error": "error" not in artifact,
    }
    artifact["all_gates_pass"] = all(artifact["gates"].values())
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage 7.11 real Server acceptance.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-container", required=True)
    parser.add_argument("--preset", default="low-memory", choices=("low-memory", "stable", "laptop-16gb"))
    parser.add_argument("--cache-bytes", default="256MiB")
    parser.add_argument("--context-tokens", type=int, default=2048)
    parser.add_argument("--max-completion-tokens", type=int, default=32)
    parser.add_argument("--request-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Explain sparse expert routing briefly.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--log-dir")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.context_tokens <= args.max_completion_tokens or args.request_tokens < 1:
        parser.error("context-tokens must exceed max-completion-tokens and request-tokens must be positive")
    result = run(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "all_gates_pass": result.get("all_gates_pass", False)}))
    return 0 if result.get("all_gates_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())

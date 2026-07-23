"""Small standard-library OpenAI-compatible gateway for SparseFlow.

The HTTP layer deliberately knows nothing about Torch, Transformers, or model
weights.  Runtime-specific behavior lives behind ``GenerationEngine``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import select
import secrets
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

from .serving_types import (
    GenerationCancelled,
    ContextLengthExceeded,
    GenerationEngine,
    GenerationRequest,
    GenerationResult,
    RuntimeState,
    ServingConfig,
)


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error" if self.status < 500 else "server_error",
                "param": self.param,
                "code": self.code,
            }
        }


class ClientCancelled(GenerationCancelled):
    pass


def error_object(error: APIError) -> dict[str, Any]:
    return error.body()


def _request_id(prefix: str = "req") -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _hash_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_text(content: Any, param: str = "messages") -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise APIError(400, "unsupported_content_type", "Only text message content is supported.", param)
    if not content:
        raise APIError(400, "invalid_request", "Content parts must not be empty.", param)
    parts: list[str] = []
    for index, part in enumerate(content):
        part_param = f"{param}[{index}]"
        if not isinstance(part, dict):
            raise APIError(400, "invalid_request", "Each content part must be an object.", part_param)
        if part.get("type") not in {"text", "input_text"}:
            raise APIError(400, "unsupported_content_type", "Only text content parts are supported.", part_param)
        if set(part) - {"type", "text"}:
            raise APIError(400, "unsupported_content_type", "Only text content parts are supported.", part_param)
        text = part.get("text")
        if not isinstance(text, str):
            raise APIError(400, "invalid_request", "Text content parts require a string text field.", f"{part_param}.text")
        parts.append(text)
    return "".join(parts)


def normalize_messages(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise APIError(400, "invalid_request", "messages must be a non-empty array.", "messages")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise APIError(400, "invalid_request", "Each message must be an object.", f"messages[{index}]")
        role = item.get("role")
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant"}:
            raise APIError(400, "invalid_request", "Unsupported message role.", f"messages[{index}].role")
        if "content" not in item:
            raise APIError(400, "invalid_request", "Message content is required.", f"messages[{index}].content")
        normalized.append({"role": role, "content": content_text(item["content"], f"messages[{index}].content")})
    return tuple(normalized)


def generation_options(body: dict[str, Any], max_cap: int) -> tuple[int, bool, bool]:
    if "max_completion_tokens" in body and "max_tokens" in body:
        raise APIError(400, "unsupported_parameter", "Use only one token limit field.", "max_tokens")
    token_value = body.get("max_completion_tokens", body.get("max_tokens", min(256, max_cap)))
    if not isinstance(token_value, int) or isinstance(token_value, bool) or not 1 <= token_value <= max_cap:
        raise APIError(400, "invalid_request", f"max completion tokens must be between 1 and {max_cap}.", "max_completion_tokens")
    temperature = body.get("temperature", 0)
    top_p = body.get("top_p", 1)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise APIError(400, "invalid_request", "temperature must be a number.", "temperature")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise APIError(400, "invalid_request", "top_p must be a number.", "top_p")
    if temperature not in (0, 0.0) or top_p not in (1, 1.0):
        raise APIError(400, "sampling_not_supported", "Only greedy decoding is currently supported.", "temperature")
    if not isinstance(body.get("n", 1), int) or isinstance(body.get("n", 1), bool):
        raise APIError(400, "invalid_request", "n must be an integer.", "n")
    if body.get("n", 1) != 1:
        raise APIError(400, "unsupported_value", "Only n=1 is supported.", "n")
    for key in ("stop", "logprobs", "seed", "frequency_penalty", "presence_penalty", "tools", "functions", "tool_choice", "cache_slot"):
        if key in body and body[key] not in (None, False, 0):
            code = "tools_not_supported" if key in {"tools", "functions", "tool_choice"} else "persistent_session_not_supported" if key == "cache_slot" else "unsupported_parameter"
            raise APIError(400, code, f"The {key} parameter is not supported.", key)
    response_format = body.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise APIError(400, "unsupported_parameter", "Only text response format is supported.", "response_format")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise APIError(400, "invalid_request", "stream must be boolean.", "stream")
    options = body.get("stream_options", {})
    if not isinstance(options, dict):
        raise APIError(400, "invalid_request", "stream_options must be an object.", "stream_options")
    unknown_options = sorted(set(options) - {"include_usage"})
    if unknown_options:
        raise APIError(400, "unsupported_parameter", f"Unsupported stream option: {unknown_options[0]}.", "stream_options")
    include_usage = options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise APIError(400, "invalid_request", "stream_options.include_usage must be boolean.", "stream_options.include_usage")
    return token_value, stream, include_usage


def model_object(model_id: str, created: int | None = None) -> dict[str, Any]:
    return {"id": model_id, "object": "model", "created": created or int(time.time()), "owned_by": "sparseflow"}


@dataclass
class _Ticket:
    request: GenerationRequest
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    deltas: deque[str] = field(default_factory=deque)
    result: GenerationResult | None = None
    error: BaseException | None = None
    queue_wait_ms: float = 0.0
    enqueued_at: float = field(default_factory=time.monotonic)
    delta_condition: threading.Condition = field(default_factory=threading.Condition)
    terminal_status: str | None = None

    def put_delta(self, value: str) -> None:
        with self.delta_condition:
            self.deltas.append(value)
            self.delta_condition.notify_all()


class GenerationScheduler:
    """One active generation plus bounded FIFO admission."""

    def __init__(self, engine: GenerationEngine, max_queue: int = 8, queue_timeout_seconds: float = 300.0):
        self.engine = engine
        self.max_queue = max_queue
        self.queue_timeout_seconds = queue_timeout_seconds
        self._queue: deque[_Ticket] = deque()
        self._active: _Ticket | None = None
        self._tickets: dict[str, _Ticket] = {}
        self._finished: dict[str, tuple[float, str]] = {}
        self._finished_ttl_seconds = max(60.0, queue_timeout_seconds)
        self._finished_limit = 1024
        self._closed = False
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="sparseflow-generation", daemon=True)
        self._stats = {"admitted": 0, "completed": 0, "failed": 0, "rejected": 0, "timed_out": 0, "cancelled": 0}
        self._worker.start()

    def _prune_finished_locked(self) -> None:
        cutoff = time.monotonic() - self._finished_ttl_seconds
        expired = [request_id for request_id, (finished_at, _) in self._finished.items() if finished_at < cutoff]
        for request_id in expired:
            self._finished.pop(request_id, None)
        while len(self._finished) > self._finished_limit:
            self._finished.pop(next(iter(self._finished)))

    def _finish_locked(self, ticket: _Ticket, status: str) -> None:
        ticket.terminal_status = status
        ticket.done.set()
        with ticket.delta_condition:
            ticket.delta_condition.notify_all()
        self._tickets.pop(ticket.request.request_id, None)
        self._finished[ticket.request.request_id] = (time.monotonic(), status)
        self._prune_finished_locked()

    def submit(self, request: GenerationRequest, client_disconnected: Callable[[], bool] | None = None) -> _Ticket:
        ticket = _Ticket(request)
        with self._condition:
            if self._closed:
                raise APIError(503, "scheduler_closed", "The scheduler is shutting down.")
            self._prune_finished_locked()
            if (self.max_queue > 0 and len(self._queue) >= self.max_queue) or (
                self.max_queue == 0 and self._active is not None
            ):
                self._stats["rejected"] += 1
                raise APIError(429, "queue_full", "The generation queue is full.")
            self._tickets[request.request_id] = ticket
            self._queue.append(ticket)
            self._stats["admitted"] += 1
            self._condition.notify_all()
        deadline = time.monotonic() + self.queue_timeout_seconds
        while not ticket.started.is_set() and not ticket.done.is_set():
            if client_disconnected is not None and client_disconnected():
                with self._condition:
                    if not ticket.started.is_set():
                        try:
                            self._queue.remove(ticket)
                        except ValueError:
                            pass
                        ticket.cancel_event.set()
                        ticket.result = GenerationResult("", finish_reason="cancelled")
                        self._stats["cancelled"] += 1
                        self._finish_locked(ticket, "cancelled")
                        raise ClientCancelled("client disconnected while queued")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._condition:
                    try:
                        self._queue.remove(ticket)
                    except ValueError:
                        if ticket.started.is_set() or ticket.done.is_set():
                            break
                        raise APIError(429, "queue_timeout", "The request waited too long for the runtime.")
                    ticket.cancel_event.set()
                    ticket.result = GenerationResult("", finish_reason="cancelled")
                    self._stats["timed_out"] += 1
                    self._finish_locked(ticket, "timed_out")
                raise APIError(429, "queue_timeout", "The request waited too long for the runtime.")
            ticket.started.wait(min(remaining, 0.05))
            with self._condition:
                if ticket.terminal_status == "scheduler_closed":
                    raise APIError(503, "scheduler_closed", "The generation scheduler is shutting down.")
        ticket.queue_wait_ms = (time.monotonic() - ticket.enqueued_at) * 1000.0
        return ticket

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if self._closed and not self._queue:
                    return
                ticket = self._queue.popleft()
                self._active = ticket
                ticket.started.set()
            try:
                if ticket.cancel_event.is_set():
                    ticket.result = GenerationResult("", finish_reason="cancelled")
                else:
                    ticket.result = self.engine.generate(ticket.request, ticket.put_delta, ticket.cancel_event.is_set)
                if ticket.result.finish_reason == "cancelled":
                    self._stats["cancelled"] += 1
                else:
                    self._stats["completed"] += 1
            except GenerationCancelled as exc:
                ticket.error = exc
                ticket.result = GenerationResult("", finish_reason="cancelled")
                self._stats["cancelled"] += 1
            except BaseException as exc:  # surface cleanly through the API
                ticket.error = exc
                self._stats["failed"] += 1
            finally:
                with self._condition:
                    self._active = None
                    status = "cancelled" if ticket.result and ticket.result.finish_reason == "cancelled" else "failed" if ticket.error else "completed"
                    self._finish_locked(ticket, status)
                    self._condition.notify_all()

    def cancel(self, request_id: str) -> str:
        with self._condition:
            self._prune_finished_locked()
            ticket = self._tickets.get(request_id)
            if ticket is None:
                if request_id in self._finished:
                    raise APIError(409, "generation_finished", "Generation has already finished.")
                raise APIError(404, "generation_not_found", "Generation request was not found.")
            if ticket.done.is_set():
                raise APIError(409, "generation_finished", "Generation has already finished.")
            if ticket is not self._active:
                try:
                    self._queue.remove(ticket)
                except ValueError:
                    pass
                ticket.cancel_event.set()
                ticket.result = GenerationResult("", finish_reason="cancelled")
                self._stats["cancelled"] += 1
                self._finish_locked(ticket, "cancelled")
                self._condition.notify_all()
                return "cancellation_requested"
            ticket.cancel_event.set()
            return "cancellation_requested"

    def close(self) -> None:
        with self._condition:
            self._closed = True
            queued = list(self._queue)
            self._queue.clear()
            for ticket in queued:
                ticket.cancel_event.set()
                self._finish_locked(ticket, "scheduler_closed")
            if self._active:
                self._active.cancel_event.set()
            self._condition.notify_all()
        self._worker.join(timeout=30)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            active = self._active
            self._prune_finished_locked()
            return {"active": active is not None, "active_request_id": active.request.request_id if active else None, "queued": len(self._queue), "max_queue": self.max_queue, "queue_timeout_seconds": self.queue_timeout_seconds, **self._stats}


class SSEWriter:
    def __init__(self, handler: BaseHTTPRequestHandler, request_id: str, model_id: str, created: int, cors_headers: dict[str, str]):
        self.handler = handler
        self.request_id = request_id
        self.model_id = model_id
        self.created = created
        self.lock = threading.Lock()
        self.cors_headers = cors_headers

    def headers(self, queue_wait_ms: float) -> None:
        self.handler.send_response(200)
        self.handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.handler.send_header("Cache-Control", "no-cache")
        self.handler.send_header("Connection", "close")
        self.handler.send_header("X-Accel-Buffering", "no")
        self.handler.send_header("x-request-id", self.request_id)
        self.handler.send_header("x-sparseflow-queue-wait-ms", f"{queue_wait_ms:.3f}")
        for key, value in self.cors_headers.items():
            self.handler.send_header(key, value)
        self.handler.end_headers()
        self.handler.close_connection = True

    def write(self, payload: str) -> None:
        with self.lock:
            self.handler.wfile.write((f"data: {payload}\n\n").encode("utf-8"))
            self.handler.wfile.flush()

    def comment(self, value: str) -> None:
        with self.lock:
            self.handler.wfile.write((f": {value}\n\n").encode("utf-8"))
            self.handler.wfile.flush()


class _SparseFlowHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


class SparseFlowAPIServer:
    def __init__(self, engine: GenerationEngine, config: ServingConfig):
        self.engine = engine
        self.config = config
        self.scheduler = GenerationScheduler(engine, config.max_queue, config.queue_timeout_seconds)
        self._http: ThreadingHTTPServer | None = None

    def snapshot(self) -> dict[str, Any]:
        snapshot = dict(self.engine.snapshot())
        snapshot["scheduler"] = self.scheduler.snapshot()
        return snapshot

    def server(self) -> ThreadingHTTPServer:
        owner = self

        class Handler(SparseFlowAPIHandler):
            api = owner

        http = _SparseFlowHTTPServer((self.config.host, self.config.port), Handler)
        http.daemon_threads = True
        self._http = http
        return http

    def close(self) -> None:
        self.scheduler.close()
        self.engine.close()
        if self._http:
            self._http.shutdown()
            self._http.server_close()


class SparseFlowAPIHandler(BaseHTTPRequestHandler):
    api: SparseFlowAPIServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:
        return

    def _origin_headers(self) -> dict[str, str]:
        origin = self.headers.get("Origin")
        if origin and origin in self.api.config.cors_origins:
            return {"Access-Control-Allow-Origin": origin, "Access-Control-Allow-Credentials": "true", "Vary": "Origin"}
        return {}

    def _auth(self, required: bool = True) -> None:
        if not required:
            return
        expected = self.api.config.api_key
        if expected is None:
            return
        authorization = self.headers.get("Authorization", "")
        if authorization != f"Bearer {expected}":
            raise APIError(401, "invalid_api_key", "Invalid API key.")

    def _send_json(self, status: int, body: dict[str, Any], request_id: str | None = None, headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        if request_id:
            self.send_header("x-request-id", request_id)
        for key, value in {**self._origin_headers(), **(headers or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, error: APIError, request_id: str | None = None) -> None:
        headers = {"Retry-After": "1"} if error.status == 429 else {}
        try:
            self._send_json(error.status, error.body(), request_id, headers)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for key, value in self._origin_headers().items():
            self.send_header(key, value)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        request_id = _request_id()
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/health":
                snapshot = self.api.snapshot()
                status = snapshot.get("state", "ready")
                self._send_json(200, {"status": status, "ready": status == RuntimeState.READY.value, "model": self.api.config.model_id, "preset": self.api.config.preset, "scheduler": snapshot.get("scheduler", {}), "runtime": {"session_mode": "stateless-full-message-replay", "persistent_kv_supported": False}})
                return
            self._auth()
            if path == "/v1/models":
                self._send_json(200, {"object": "list", "data": [model_object(self.api.config.model_id)]}, request_id)
                return
            if path == "/v1/runtime":
                self._send_json(200, self.api.snapshot(), request_id)
                return
            if path == f"/v1/models/{self.api.config.model_id}":
                self._send_json(200, model_object(self.api.config.model_id), request_id)
                return
            if path.startswith("/v1/models/"):
                raise APIError(404, "model_not_found", "The requested model is not available.", "model")
            raise APIError(404, "not_found", "The requested endpoint was not found.")
        except APIError as error:
            self._error(error, request_id)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "invalid_request", "Invalid Content-Length header.")
        if length <= 0 or length > 4 * 1024 * 1024:
            raise APIError(400, "invalid_request", "Request body must be between 1 byte and 4 MiB.")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise APIError(400, "invalid_json", "Request body is not valid JSON.")
        if not isinstance(body, dict):
            raise APIError(400, "invalid_request", "Request body must be an object.")
        return body

    def _client_disconnected(self) -> bool:
        """Probe a queued HTTP socket without consuming request data."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, ConnectionResetError, OSError):
            return True

    @staticmethod
    def _generation_error(exc: BaseException) -> APIError:
        if isinstance(exc, ContextLengthExceeded):
            return APIError(400, "context_length_exceeded", str(exc), "max_completion_tokens")
        if isinstance(exc, GenerationCancelled):
            return APIError(499, "client_cancelled", "The client cancelled generation.")
        return APIError(500, "engine_error", "Generation failed.")

    def do_POST(self) -> None:
        request_id = _request_id()
        try:
            self._auth()
            path = urlparse(self.path).path.rstrip("/")
            if path.startswith("/v1/generations/") and path.endswith("/cancel"):
                target = path[len("/v1/generations/"):-len("/cancel")]
                status = self.api.scheduler.cancel(target)
                self._send_json(202, {"id": target, "status": status}, request_id)
                return
            if path != "/v1/chat/completions":
                raise APIError(404, "not_found", "The requested endpoint was not found.")
            body = self._read_json()
            if body.get("model") != self.api.config.model_id:
                raise APIError(404, "model_not_found", "The requested model is not available.", "model")
            tokens, stream, include_usage = generation_options(body, self.api.config.max_completion_tokens)
            messages = normalize_messages(body.get("messages"))
            snapshot = self.api.snapshot()
            if snapshot.get("state") not in {RuntimeState.READY.value, RuntimeState.BUSY.value}:
                raise APIError(503, "runtime_loading" if snapshot.get("state") == RuntimeState.LOADING.value else "runtime_unavailable", "The runtime is not ready.")
            created = int(time.time())
            request = GenerationRequest(request_id, self.api.config.model_id, messages, tokens, stream, include_usage, created)
            validator = getattr(self.api.engine, "validate_request", None)
            if validator is not None:
                try:
                    validator(request)
                except ContextLengthExceeded:
                    raise
                except RuntimeError as exc:
                    raise APIError(503, "runtime_unavailable", "The runtime is not ready.") from exc
            ticket = self.api.scheduler.submit(request, self._client_disconnected)
            if stream:
                self._stream(ticket)
            else:
                ticket.done.wait()
                if ticket.error and not (ticket.result and ticket.result.finish_reason == "cancelled"):
                    raise self._generation_error(ticket.error)
                assert ticket.result is not None
                self._send_json(200, self._completion(request, ticket.result, ticket.queue_wait_ms), request_id)
        except APIError as error:
            self._error(error, request_id)
        except ContextLengthExceeded as error:
            self._error(APIError(400, "context_length_exceeded", str(error), "max_completion_tokens"), request_id)
        except ClientCancelled:
            return
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.api.scheduler.cancel(request_id)
            except APIError:
                pass

    def _completion(self, request: GenerationRequest, result: GenerationResult, queue_wait_ms: float) -> dict[str, Any]:
        return {"id": f"chatcmpl-{request.request_id[4:]}", "object": "chat.completion", "created": request.created, "model": request.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text, "refusal": None}, "logprobs": None, "finish_reason": result.finish_reason}], "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "total_tokens": result.prompt_tokens + result.completion_tokens}, "sparseflow": {"request_id": request.request_id, "queue_wait_ms": queue_wait_ms, "session_mode": "stateless-full-message-replay", **result.telemetry}}

    def _stream(self, ticket: _Ticket) -> None:
        request = ticket.request
        writer = SSEWriter(self, request.request_id, request.model, request.created, self._origin_headers())
        try:
            writer.headers(ticket.queue_wait_ms)
            prefix = {"id": f"chatcmpl-{request.request_id[4:]}", "object": "chat.completion.chunk", "created": request.created, "model": request.model}
            writer.write(json.dumps({**prefix, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": None, "finish_reason": None}]}, ensure_ascii=False, separators=(",", ":")))
            while not ticket.done.is_set() or ticket.deltas:
                with ticket.delta_condition:
                    if not ticket.deltas and not ticket.done.is_set():
                        ticket.delta_condition.wait(timeout=self.api.config.keepalive_seconds)
                    deltas = list(ticket.deltas)
                    ticket.deltas.clear()
                if not deltas and not ticket.done.is_set():
                    writer.comment("sparseflow-keepalive")
                for delta in deltas:
                    writer.write(json.dumps({**prefix, "choices": [{"index": 0, "delta": {"content": delta}, "logprobs": None, "finish_reason": None}]}, ensure_ascii=False, separators=(",", ":")))
            if ticket.error and not (ticket.result and ticket.result.finish_reason == "cancelled"):
                error = self._generation_error(ticket.error)
                writer.write(json.dumps(error.body(), ensure_ascii=False, separators=(",", ":")))
            else:
                result = ticket.result or GenerationResult("", finish_reason="cancelled")
                writer.write(json.dumps({**prefix, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": result.finish_reason}]}))
                if request.include_usage:
                    writer.write(json.dumps({**prefix, "choices": [], "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "total_tokens": result.prompt_tokens + result.completion_tokens}}))
            writer.write("[DONE]")
        except (BrokenPipeError, ConnectionResetError, OSError):
            ticket.cancel_event.set()


def serve_http(engine: GenerationEngine, config: ServingConfig) -> SparseFlowAPIServer:
    """Create and bind a server; callers own its lifetime and close it."""
    api = SparseFlowAPIServer(engine, config)
    api.server()
    return api

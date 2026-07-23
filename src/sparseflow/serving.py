"""Runtime adapter and lifecycle management for the local server."""

from __future__ import annotations

import hashlib
import inspect
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .release import apply_preset, container_identity, doctor, model_identity
from .process_metrics import process_snapshot
from .serving_types import ContextLengthExceeded, GenerationCancelled, GenerationRequest, GenerationResult, RuntimeState, ServingConfig


class RuntimeLoadError(RuntimeError):
    pass


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_generation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    text = str(result.get("text", ""))
    ids = result.get("generated_ids") or []
    decode_seconds = float(result.get("decode_seconds") or 0.0)
    return {
        "prompt_tokens": max(0, len(result.get("input_ids") or []) - 1),
        "completion_tokens": int(result.get("generated_tokens") or max(0, len(ids) - 1)),
        "prefill_seconds": float(result.get("prefill_seconds") or 0.0),
        "decode_seconds": decode_seconds,
        "decode_tok_per_second": (len(ids) / decode_seconds) if decode_seconds else 0.0,
        "memory": result.get("memory"),
        "cache": result.get("cache"),
        "provider_storage": result.get("provider_storage"),
        "runtime_identity": result.get("runtime_identity", {}),
        "generated_ids_hash": _sha(",".join(map(str, ids))) if ids else None,
        "output_text_hash": _sha(text),
        "route_fingerprint": _sha(repr(result.get("route_audit"))) if result.get("route_audit") is not None else None,
    }


def create_public_runtime(config: ServingConfig):
    """Construct the same preset runtime used by the Stage 7.9 CLI.

    This function is intentionally the first place that imports the optional
    Torch/Transformers runtime.
    """
    try:
        from .text_runtime import Qwen36TextRuntime
    except (ImportError, OSError) as exc:
        raise RuntimeLoadError("SparseFlow server requires optional runtime dependencies; install sparseflow[runtime].") from exc
    preset = apply_preset(config.preset, cache_bytes=config.cache_bytes, telemetry_level=config.telemetry_level)
    return Qwen36TextRuntime.from_pretrained(
        config.model_dir,
        mode=preset["mode"],
        dtype="bf16",
        cache_slots=None if preset["cache_bytes"] is not None else 16,
        cache_bytes=preset["cache_bytes"],
        prefetch_workers=preset["prefetch_workers"],
        coalesce_gap=0,
        cache_policy=preset["cache_policy"],
        prefetch_policy=preset["prefetch_policy"],
        telemetry_level=config.telemetry_level,
        experts_implementation="eager",
        load_mode=preset["load_mode"],
        expert_storage=preset["expert_storage"],
        int8_container=config.int8_container,
        native_dispatch=preset["native_dispatch"],
    )


def _safe_identity(fn: Callable[..., dict[str, Any]], path: Path) -> dict[str, Any] | None:
    try:
        return fn(path)
    except (OSError, ValueError, KeyError):
        return None


class SparseFlowEngine:
    """Own one persistent Qwen runtime and serialize generation requests."""

    def __init__(self, config: ServingConfig, *, load_async: bool = True):
        self.config = config
        self.model_id = config.model_id
        self.preset = config.preset
        self._state = RuntimeState.LOADING
        self._error: str | None = None
        self._runtime: Any = None
        self._last_generation: dict[str, Any] | None = None
        self._runtime_load_count = 0
        self._runtime_loaded_at: float | None = None
        self._runtime_load_seconds = 0.0
        self._generation_count = 0
        self._last_request_id: str | None = None
        self._active_request_id: str | None = None
        self._active_cancel: threading.Event | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._loader: threading.Thread | None = None
        self._generation_done = threading.Event()
        self._generation_done.set()
        if load_async:
            self._loader = threading.Thread(target=self._load, name="sparseflow-runtime-loader", daemon=True)
            self._loader.start()

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            check = doctor(
                self.config.model_dir,
                preset=self.config.preset,
                int8_container_dir=self.config.int8_container,
                cache_bytes=self.config.cache_bytes,
                ctx=self.config.ctx,
            )
            if not check.get("ready"):
                reasons = "; ".join(str(item.get("message", item.get("id", "failed"))) for item in check.get("checks", []) if item.get("status") == "fail")
                raise RuntimeLoadError(f"doctor preflight failed: {reasons or 'runtime is not ready'}")
            runtime = create_public_runtime(self.config)
            with self._lock:
                if self._closed:
                    runtime.close()
                else:
                    self._runtime = runtime
                    self._runtime_load_count += 1
                    self._runtime_loaded_at = time.time()
                    self._runtime_load_seconds = time.perf_counter() - started
                    self._state = RuntimeState.READY
        except Exception as exc:
            with self._lock:
                self._runtime_load_seconds = time.perf_counter() - started
                self._error = str(exc)
                self._state = RuntimeState.ERROR

    def wait_ready(self, timeout: float | None = None) -> bool:
        if self._loader:
            self._loader.join(timeout)
        with self._lock:
            return self._state == RuntimeState.READY

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = self._state.value
            error = self._error
            runtime = self._runtime
            last = dict(self._last_generation) if self._last_generation else None
            runtime_load_count = self._runtime_load_count
            runtime_loaded_at = self._runtime_loaded_at
            runtime_load_seconds = self._runtime_load_seconds
            generation_count = self._generation_count
            last_request_id = self._last_request_id
        effective_config = apply_preset(
            self.config.preset,
            cache_bytes=self.config.cache_bytes,
            telemetry_level=self.config.telemetry_level,
        )
        effective_config.update(
            {
                "context_tokens": self.config.context_tokens,
                "max_completion_tokens": self.config.max_completion_tokens,
                "model_id": self.model_id,
            }
        )
        process = process_snapshot()
        process_memory = {
            "rss_bytes": int(process.get("rss_bytes", 0)),
            "peak_rss_bytes": int(process.get("peak_rss_bytes", 0)),
            "private_bytes": int(process.get("private_bytes", 0)),
            "read_bytes": int(process.get("read_bytes", 0)),
            "read_bytes_semantics": process.get("read_bytes_semantics"),
            "platform_source": process.get("platform_source"),
        }
        result: dict[str, Any] = {
            "schema_version": 1,
            "state": state,
            "model_id": self.model_id,
            "preset": self.preset,
            "session_mode": "stateless-full-message-replay",
            "persistent_kv_supported": False,
            "runtime_identity": getattr(runtime, "runtime_identity", {}) if runtime is not None else {},
            "public_status": effective_config["public_status"],
            "effective_config": effective_config,
            "runtime_load_count": runtime_load_count,
            "runtime_loaded_at": runtime_loaded_at,
            "runtime_load_seconds": runtime_load_seconds,
            "generation_count": generation_count,
            "last_request_id": last_request_id,
            "process_memory": process_memory,
            "last_generation": last,
        }
        if error:
            result["error"] = error
        model = _safe_identity(model_identity, self.config.model_dir)
        container = _safe_identity(container_identity, self.config.int8_container)
        if model is not None:
            result["model"] = model
        if container is not None:
            result["container"] = container
        return result

    def generate(self, request: GenerationRequest, on_delta: Callable[[str], None] | None = None, is_cancelled: Callable[[], bool] | None = None) -> GenerationResult:
        with self._lock:
            if self._runtime is None or self._state not in {RuntimeState.READY, RuntimeState.BUSY}:
                raise RuntimeLoadError("runtime is not ready")
            self._state = RuntimeState.BUSY
            self._active_request_id = request.request_id
            cancel_event = threading.Event()
            self._active_cancel = cancel_event
            runtime = self._runtime
            self._generation_done.clear()

        def cancelled() -> bool:
            return cancel_event.is_set() or bool(is_cancelled and is_cancelled())

        try:
            generate_messages = runtime.generate_messages
            try:
                parameters = inspect.signature(generate_messages).parameters
                accepts_context = "context_tokens" in parameters or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
            except (TypeError, ValueError):
                accepts_context = True
            kwargs = {
                "max_new_tokens": request.max_completion_tokens,
                "stop_on_eos": True,
                "on_text_delta": on_delta,
                "is_cancelled": cancelled,
            }
            if accepts_context:
                kwargs["context_tokens"] = self.config.context_tokens
            raw = generate_messages(list(request.messages), **kwargs)
            metrics = compact_generation_metrics(raw)
            result = GenerationResult(
                text=str(raw.get("text", "")),
                prompt_tokens=int(metrics["prompt_tokens"]),
                completion_tokens=int(metrics["completion_tokens"]),
                finish_reason=str(raw.get("finish_reason", "stop")),
                prefill_seconds=float(metrics["prefill_seconds"]),
                decode_seconds=float(metrics["decode_seconds"]),
                telemetry=metrics,
                runtime_identity=raw.get("runtime_identity", {}),
                generated_ids_hash=metrics["generated_ids_hash"],
                output_text_hash=metrics["output_text_hash"],
                route_fingerprint=metrics["route_fingerprint"],
            )
            with self._lock:
                metrics["request_id"] = request.request_id
                self._last_generation = metrics
            return result
        finally:
            with self._lock:
                self._generation_count += 1
                self._last_request_id = request.request_id
                self._active_request_id = None
                self._active_cancel = None
                if not self._closed and self._state != RuntimeState.ERROR:
                    self._state = RuntimeState.READY
                self._generation_done.set()

    def validate_request(self, request: GenerationRequest) -> None:
        """Validate tokenized context before scheduler admission when possible."""
        with self._lock:
            runtime = self._runtime
            if runtime is None or self._state not in {RuntimeState.READY, RuntimeState.BUSY}:
                raise RuntimeLoadError("runtime is not ready")
        validator = getattr(runtime, "validate_context", None)
        if validator is not None:
            validator(list(request.messages), request.max_completion_tokens, self.config.context_tokens)

    def cancel(self, request_id: str) -> str:
        with self._lock:
            if request_id != self._active_request_id or self._active_cancel is None:
                raise RuntimeLoadError("generation request was not active")
            self._active_cancel.set()
            return "cancellation_requested"

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._state = RuntimeState.STOPPING
            if self._active_cancel:
                self._active_cancel.set()
            runtime = self._runtime
            self._runtime = None
        if self._loader and self._loader.is_alive():
            self._loader.join(timeout=30)
        self._generation_done.wait(timeout=30)
        if runtime is not None:
            runtime.close()
        with self._lock:
            self._state = RuntimeState.STOPPED

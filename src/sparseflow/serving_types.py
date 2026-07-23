"""Dependency-free contracts shared by the local HTTP gateway and runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


class RuntimeState(str, Enum):
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"


class GenerationCancelled(RuntimeError):
    """Raised when a runtime observes cancellation at a token boundary."""


class ContextLengthExceeded(ValueError):
    """Raised before generation when prompt plus requested output exceeds context."""

    def __init__(self, prompt_tokens: int, max_completion_tokens: int, context_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.context_tokens = context_tokens
        super().__init__(
            "prompt tokens plus max completion tokens exceed the configured context "
            f"({prompt_tokens} + {max_completion_tokens} > {context_tokens})"
        )


@dataclass(frozen=True)
class ServingConfig:
    model_dir: Path
    int8_container: Path
    preset: str = "low-memory"
    cache_bytes: int | None = None
    telemetry_level: str = "summary"
    ctx: int = 4096
    context_tokens: int | None = None
    max_completion_tokens: int = 256
    host: str = "127.0.0.1"
    port: int = 8000
    model_id: str = "qwen3.6-35b-a3b-sparseflow"
    api_key: str | None = None
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://tauri.localhost",
        "tauri://localhost",
    )
    max_queue: int = 8
    queue_timeout_seconds: float = 300.0
    keepalive_seconds: float = 10.0
    allow_unauthenticated_remote: bool = False

    def __post_init__(self) -> None:
        if self.context_tokens is not None and self.ctx != 4096 and self.ctx != self.context_tokens:
            raise ValueError("ctx and context_tokens must match when both are provided")
        effective_context = self.context_tokens if self.context_tokens is not None else self.ctx
        object.__setattr__(self, "ctx", effective_context)
        object.__setattr__(self, "context_tokens", effective_context)
        object.__setattr__(self, "model_dir", Path(self.model_dir).expanduser().resolve())
        object.__setattr__(self, "int8_container", Path(self.int8_container).expanduser().resolve())
        if self.preset not in {"stable", "low-memory", "laptop-16gb"}:
            raise ValueError("server supports only stable, low-memory and laptop-16gb presets")
        if effective_context <= 0 or self.max_completion_tokens <= 0:
            raise ValueError("ctx and max_completion_tokens must be positive")
        if effective_context <= self.max_completion_tokens:
            raise ValueError("context_tokens must be greater than max_completion_tokens")
        if self.max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if self.queue_timeout_seconds <= 0 or self.keepalive_seconds <= 0:
            raise ValueError("queue timeout and keepalive must be positive")
        if isinstance(self.port, bool) or not 0 <= self.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if self.host not in {"127.0.0.1", "localhost", "::1"} and not (
            self.api_key or self.allow_unauthenticated_remote
        ):
            raise ValueError("remote bind requires api_key or explicit unauthenticated override")


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    model: str
    messages: tuple[dict[str, str], ...]
    max_completion_tokens: int
    stream: bool = False
    include_usage: bool = False
    created: int = 0


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)
    runtime_identity: dict[str, Any] = field(default_factory=dict)
    generated_ids_hash: str | None = None
    output_text_hash: str | None = None
    route_fingerprint: str | None = None


class GenerationEngine(Protocol):
    model_id: str
    preset: str

    def snapshot(self) -> dict[str, Any]: ...

    def generate(
        self,
        request: GenerationRequest,
        on_delta: Callable[[str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> GenerationResult: ...

    def cancel(self, request_id: str) -> str: ...

    def close(self) -> None: ...

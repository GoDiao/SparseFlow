# SparseFlow Server API Contract

This is the frontend-facing contract for the local SparseFlow server. The
machine-readable schema is [sparseflow-openapi.json](openapi/sparseflow-openapi.json).
The server is OpenAI-compatible only within the capability boundary documented
here; it is not a general OpenAI API implementation.

## Base URL and lifecycle

The default bind address is `http://127.0.0.1:8000`. A frontend should poll
`GET /health` before enabling chat input.

| State | Meaning | Chat submission |
| --- | --- | --- |
| `loading` | Model/runtime is being loaded. | Disable temporarily. |
| `ready` | Runtime can accept a request. | Enable. |
| `busy` | One request is running; later requests may wait in the bounded queue. | Enable, but show queued state when applicable. |
| `error` | Runtime failed to load or failed permanently. | Disable and show the error from `/v1/runtime`. |
| `stopping` | Server is shutting down. | Disable. |
| `stopped` | Server is closed. | Disable. |

`/health` is intentionally small and does not expose model hashes or detailed
telemetry. Use `/v1/runtime` for diagnostics and the status panel.

## Endpoints

### `GET /health`

Anonymous readiness probe. Required fields:

```json
{
  "status": "ready",
  "ready": true,
  "model": "qwen3.6-35b-a3b-sparseflow",
  "preset": "laptop-16gb",
  "scheduler": {
    "active": false,
    "active_request_id": null,
    "queued": 0,
    "max_queue": 8
  },
  "runtime": {
    "session_mode": "stateless-full-message-replay",
    "persistent_kv_supported": false
  }
}
```

### `GET /v1/models`

Returns the one configured model. The frontend should use its `id` as the
`model` request field rather than hard-coding a second model name.

### `GET /v1/models/{model_id}`

Returns the configured model when the path id matches. Unknown ids return the
standard error object with `code=model_not_found`.

### `GET /v1/runtime`

Returns the runtime snapshot. The following fields are stable frontend fields:

```text
schema_version, state, model_id, preset, public_status
effective_config.cache_bytes
effective_config.context_tokens
effective_config.max_completion_tokens
runtime_load_count, runtime_load_seconds, runtime_identity
model.metadata_sha256
container.metadata_sha256, container.weight_bytes, container.execution_bytes
process_memory.rss_bytes, process_memory.peak_rss_bytes
last_generation
```

`last_generation` is `null` before the first successful or cancelled request.
When present, only compact counters and hashes are exposed. A frontend must
not depend on Python internals or benchmark raw artifacts.

### `POST /v1/chat/completions`

Request body:

```json
{
  "model": "qwen3.6-35b-a3b-sparseflow",
  "messages": [
    {"role": "user", "content": "Explain sparse expert routing."}
  ],
  "max_completion_tokens": 32,
  "stream": false
}
```

Supported message roles are `system`, `developer` (normalized to `system`),
`user`, and `assistant`. Content must be text or text content parts.
`max_tokens` is accepted as a compatibility alias when
`max_completion_tokens` is absent. The configured preset is the upper bound.

Only deterministic greedy decoding is supported:

- `temperature` must be `0` or omitted.
- `top_p` must be `1` or omitted.
- `n` must be `1` or omitted.
- `response_format: {"type":"text"}` is accepted.

Requests with sampling, tools, multimodal content, `cache_slot`, or other
unsupported options receive a structured `400` error. Multi-turn chat is
stateless: send the complete message history on every request. Persistent
KV/DeltaNet sessions are not part of this contract.

### `POST /v1/generations/{request_id}/cancel`

Requests cancellation of an active or queued generation:

```json
{"id":"req_0123456789abcdef","status":"cancellation_requested"}
```

Cancellation is cooperative at runtime token boundaries. A cancelled request
must not be used as evidence of a correctness `PASS`; the frontend should show
it as `cancelled`.

## Streaming

Set `stream: true` to receive `Content-Type: text/event-stream`. Each event is
`data: <JSON>`, followed by a blank line. The first chunk contains the assistant
role, later chunks contain text deltas, and the terminal event contains a
`finish_reason`. The stream ends with:

```text
data: [DONE]
```

Keepalive comments beginning with `: sparseflow-keepalive` are not JSON data
events and should be ignored. When `stream_options.include_usage` is true, a
final empty-choice usage chunk is sent before `[DONE]`.

## Errors

All JSON errors use this shape:

```json
{
  "error": {
    "message": "The runtime is not ready.",
    "type": "server_error",
    "param": null,
    "code": "runtime_loading"
  }
}
```

Memory admission failures during server startup surface as
`runtime_unavailable`; the detailed Doctor reason is available from the
runtime lifecycle/error state rather than being exposed as a second HTTP code.
Useful frontend codes include `runtime_loading`, `runtime_unavailable`,
`context_length_exceeded`, `queue_full`, `queue_timeout`,
`client_cancelled`, `model_not_found`, and `sampling_not_supported`.

## Evidence labels

Status and benchmark panels must distinguish:

- `MEASURED`: emitted by a real SparseFlow runtime artifact.
- `SIMULATED`: static/demo data only.
- `UNKNOWN`: no source is available.
- `PASS`/`FAIL`: verifier gate results, not HTTP success.
- `NOT_RUN`: the check has not executed.

An HTTP 200 response, generated text, or a running process alone is not a
SparseFlow correctness `PASS`.

## Capability boundary

The current public contract does not include sampling, tool calling, vision
inputs, MTP, persistent KV/DeltaNet sessions, or continuous batching. The
server is local-first and should bind to `127.0.0.1` unless an API key or an
explicit unauthenticated remote override is configured.

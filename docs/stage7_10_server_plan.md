# Stage 7.10: SparseFlow Local Server

**Decision owner:** Board
**Implementation owner:** Server Dev
**Runtime reviewer:** Main Dev
**Benchmark reviewer:** Benchmark
**Primary consumer:** Frontend
**Status:** planned
**Recorded:** 2026-07-23

> Stage 7.11 extends this original server scope with the experimental
> `laptop-16gb` preset and its dedicated acceptance protocol. The stable and
> low-memory-only statements below describe the Stage 7.10 freeze; the current
> frontend-facing contract is [docs/api_contract.md](api_contract.md).

## 1. 文档用途

本文是 Stage 7.10 的完整执行规范。实现者必须按本文顺序工作，不得根据个人偏好替换协议、扩大范围或改变已冻结的 Qwen3.6 runtime 默认路径。

本文特意写得比普通 roadmap 更详细。每个步骤都包含：

- 要修改或新增的文件；
- 要实现的类、函数和数据字段；
- 可以从 Colibri 迁移的代码；
- 不得从 Colibri 复制的模型专用逻辑；
- 本地单元测试；
- 实验机真实 runtime 验收；
- GO/NO-GO 条件；
- 完成后需要提交的证据。

实现者遇到本文未定义的协议决策时，必须暂停并交给 Board 决策。不得自行增加依赖、改默认 preset、开放未验证的能力或声称已有 persistent KV。

## 2. Stage 目标

实现一个本地常驻的 SparseFlow HTTP server，使一个已经通过 `doctor` 的 Qwen3.6-35B-A3B runtime 只加载一次，并通过 OpenAI-compatible Chat Completions API 为 Web UI、Tauri desktop 和普通 API client 提供文本生成。

目标命令：

```bash
PYTHONPATH=src python -m sparseflow serve "$MODEL" \
  --preset low-memory \
  --int8-container "$INT8" \
  --host 127.0.0.1 \
  --port 8000 \
  --model-id qwen3.6-35b-a3b-sparseflow
```

目标请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b-sparseflow",
    "messages": [{"role": "user", "content": "Explain sparse expert routing."}],
    "max_completion_tokens": 32,
    "temperature": 0,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Stage 完成时必须同时满足：

1. Server 在没有请求时只持有一个 Qwen runtime。
2. 同一 server 进程连续处理多个请求时不重复加载模型。
3. `stable` 和 `low-memory` 与 Stage 7.9 CLI 路径使用完全相同的 runtime factory 和 preset 配置。
4. 单请求执行保持串行；排队请求使用 bounded FIFO。
5. 非流式和 SSE 流式结果与直接 runtime 结果一致。
6. 多轮对话明确使用 stateless full-message replay。
7. 浏览器断开或显式取消后，generation 在 token 边界停止，provider/cache lease 最终归零。
8. Server module 的导入不导致 `preset`、`inspect`、`plan`、`doctor` 提前导入 Torch。
9. 本地 FakeEngine 测试可在不安装 runtime extras 的环境运行。
10. 真实 Qwen 集成测试在 Linux AVX-512 VNNI 实验机运行并生成正式证据。

## 3. 已冻结的事实

以下事实来自 Stage 7.9，Stage 7.10 不得重新解释：

| 项目 | 冻结状态 |
|---|---|
| Stable preset | INT8 native resident hybrid |
| Low-memory preset | INT8 native single-request streaming, S1 LRU, 4 GiB default cache |
| Experimental batch | resident fixed cohort，仅 benchmark/opt-in，不进入 server |
| Shared streaming batching | disabled |
| Sampling | 当前 runtime 未实现，只支持 greedy |
| Persistent KV session | 未实现 |
| Conversation | stateless full-message replay |
| Vision | 不在 Public Alpha 范围 |
| Tool calling | 未验证，不在 Stage 7.10 范围 |
| Runtime host | Transformers/PyTorch 继续拥有 attention、DeltaNet、KV state 和 tokenizer |

Stage 7.10 是 serving 层，不是新的 expert kernel stage。除非修复 server 集成所暴露的正确性问题，否则不得修改 native kernel、quantization container、cache policy 或 dispatch 默认值。

## 4. 范围

### 4.1 必须实现

- `sparseflow serve` CLI。
- 模型启动前的 `doctor` preflight。
- Server 启动后的 runtime lifecycle state。
- 一个常驻 `Qwen36TextRuntime`。
- `GET /health`。
- `GET /v1/models`。
- `GET /v1/models/{model_id}`。
- `GET /v1/runtime`。
- `POST /v1/chat/completions` 非流式响应。
- `POST /v1/chat/completions` SSE 响应。
- `POST /v1/generations/{request_id}/cancel`。
- Bearer API key。
- 精确 CORS allowlist。
- bounded FIFO admission queue。
- queue full 与 queue timeout 错误。
- runtime loading、ready、busy、error、stopping、stopped 状态。
- request ID、queue wait、usage、TTFT、decode throughput、RSS、cache summary。
- client disconnect cancellation。
- long-prefill SSE keepalive。
- FakeEngine unit tests。
- 实验机真实 Qwen acceptance harness。
- README 与 API contract 文档。

### 4.2 明确不实现

- 动态切换 model path。
- 一个 server 同时加载多个模型。
- OpenAI legacy `/v1/completions`。
- Responses API。
- continuous batching。
- server 内启用 `experimental-batch`。
- persistent KV 或 DeltaNet session。
- `cache_slot`。
- tool calling。
- image、audio 或其他 multimodal content。
- temperature sampling、top-p sampling、top-k sampling。
- custom stop sequence。
- logprobs。
- frequency penalty、presence penalty。
- per-request seed。
- JSON schema constrained decoding。
- Web UI 与 Tauri packaging。
- remote hosted multi-tenant security。
- GPU tier。

上述字段如果出现在请求中，必须返回明确的 OpenAI-shaped 4xx error。不得忽略字段后继续生成。

## 5. Colibri 复用边界

参考代码：`_reference/colibri/c/openai_server.py`，Apache-2.0。

### 5.1 可以迁移并改名

| Colibri 组件 | SparseFlow 处理 |
|---|---|
| `APIError` | 迁移，保留 OpenAI error object |
| `ClientCancelled` | 迁移，统一 queued 与 active cancellation |
| `GenerationScheduler` | 迁移，增加 failed 与 active request id 统计 |
| `content_text` | 迁移，错误文案改为 SparseFlow text-only |
| `generation_options` 的校验框架 | 迁移，但只允许 greedy 参数 |
| `APIServer` | 迁移，engine 类型改为 protocol |
| `APIHandler` 基础 HTTP 逻辑 | 迁移并拆分较长函数 |
| CORS | 迁移，header 名称改为 SparseFlow |
| Bearer auth | 迁移，增加 remote bind 安全 gate |
| `/health` | 迁移并接 RuntimeManager snapshot |
| `/v1/models` | 迁移并修改 owner/model metadata |
| SSE event framing | 迁移，keepalive 改为 SSE comment |
| `client_disconnected` | 迁移，并连接 active cancel event |
| HTTP 与 scheduler tests | 迁移到 `tests/test_server.py` |

### 5.2 不得复制

| Colibri 组件 | 原因 |
|---|---|
| `render_chat` | GLM-5.2 prompt markers 专用 |
| `parse_tool_calls` | GLM tool XML 专用 |
| tool salvage/de-mangler | 未验证且会改写模型输出 |
| `Engine` stdin/stdout sentinel protocol | SparseFlow runtime 是 Python object |
| `READY`、`END`、`STAT` | C engine 私有协议 |
| `cache_slot` 与 KV slot | SparseFlow 尚无 persistent session |
| `enable_thinking` GLM 处理 | Qwen chat template 尚未形成 server contract |
| `COLI_*` environment variables | 品牌和配置命名错误 |
| fake `reasoning_content` keepalive delta | 会污染标准模型输出 |

### 5.3 必须改名

- `colibri` -> `sparseflow`
- `x-colibri-queue-wait-ms` -> `x-sparseflow-queue-wait-ms`
- `COLI_API_KEY` -> `SPARSEFLOW_API_KEY`
- `COLI_MAX_QUEUE` -> `SPARSEFLOW_MAX_QUEUE`
- `COLI_QUEUE_TIMEOUT` -> `SPARSEFLOW_QUEUE_TIMEOUT`
- server version -> `sparseflow`
- model `owned_by` -> `sparseflow`
- engine error 文案不得包含 Colibri 或 GLM。

## 6. License 与 attribution gate

SparseFlow 当前没有 `LICENSE`。在复制 Colibri 的实质代码前，维护者必须完成 license 决策。

推荐方案：

1. SparseFlow 采用 Apache-2.0。
2. 仓库根目录新增 Apache-2.0 `LICENSE`。
3. 新增 `NOTICE`，注明 server protocol code adapted from `JustVugg/colibri`。
4. `src/sparseflow/server.py` 顶部保留简短 attribution 与 Apache-2.0 来源。
5. README 的 acknowledgements 记录 Colibri。

如果维护者没有批准 Apache-2.0，则实现者不得逐段复制 `openai_server.py`，只能依据公开 API 行为重新实现，并由维护者单独确认最终 license。本文不构成法律意见。

## 7. 目标架构

```text
Browser / Tauri / OpenAI client
                |
                | HTTP JSON + SSE
                v
src/sparseflow/server.py
  API validation, auth, CORS, queue, SSE, errors
                |
                | GenerationEngine protocol
                v
src/sparseflow/serving_types.py
  dependency-free config, request/result types, protocol, state enum
                |
                v
src/sparseflow/serving.py
  lifecycle, doctor, runtime factory, cancellation, snapshots
                |
                | messages + callback + cancellation
                v
src/sparseflow/text_runtime.py
  Qwen tokenizer template, prefill, decode, provider/cache
                |
                v
SparseFlow expert provider / native W8A8 backend
```

关键约束：

- `server.py` 必须只依赖 Python standard library 与轻量 SparseFlow types。
- `serving_types.py` 必须只依赖 Python standard library。
- `server.py` import 时不得 import Torch、Transformers、Safetensors 或 Accelerate。
- `serving.py` 只在 runtime loader thread 内导入 `text_runtime`。
- HTTP handler 不得直接访问 `Qwen36TextRuntime.model`、tokenizer 或 provider。
- `text_runtime.py` 不得包含 HTTP、SSE、CORS 或 API key 逻辑。
- CLI 与 server 必须调用同一个 public runtime factory。

## 8. 文件级实施清单

### 8.1 新增 `src/sparseflow/server.py`

职责：

- 定义 OpenAI-shaped error。
- 定义 HTTP request validation。
- 定义 bounded FIFO scheduler。
- 定义 HTTP server 与 handler。
- 定义 SSE writer。
- 定义 CORS 与 auth。
- 调用 `GenerationEngine` protocol，不接触 Qwen 实现。

必须包含的主要符号：

- `APIError`
- `ClientCancelled`
- `error_object(error)`
- `GenerationScheduler`
- `content_text(content, param)`
- `normalize_messages(messages)`
- `generation_options(body, max_tokens)`
- `model_object(model_id, created)`
- `SSEWriter`
- `SparseFlowAPIServer`
- `SparseFlowAPIHandler`
- `serve_http(engine, config)`

### 8.2 新增 `src/sparseflow/serving_types.py`

职责：

- 保存 HTTP gateway 与 runtime adapter 共同使用的 dependency-free types。
- 避免 `server.py` 为了类型标注 import `serving.py`。
- 让 FakeEngine tests 在没有 Torch 的环境 import 完整 protocol。

必须包含的主要符号：

- `ServingConfig`
- `GenerationRequest`
- `GenerationResult`
- `GenerationCancelled`
- `GenerationEngine` protocol
- `RuntimeState`

`server.py` 和 `serving.py` 都从本文件 import contract。`serving_types.py` 不得 import `server.py` 或 `serving.py`。

### 8.3 新增 `src/sparseflow/serving.py`

职责：

- 定义 server 与 runtime 之间的稳定 contract。
- 运行 `doctor`。
- 根据 public preset 构造 runtime。
- 管理 lifecycle state。
- 保证 runtime 只加载一次。
- 管理 request cancellation event。
- 把 runtime 大结果压缩成 API telemetry。
- shutdown 时关闭 provider、reader 和 native profile。

必须包含的主要符号：

- `RuntimeLoadError`
- `SparseFlowEngine`
- `create_public_runtime(config)`
- `compact_generation_metrics(result)`

### 8.4 修改 `src/sparseflow/text_runtime.py`

新增：

- `encode_messages(messages)`
- `generate_messages(messages, max_new_tokens, stop_on_eos, on_text_delta, is_cancelled)`
- 内部共享 generation loop。
- callback text streamer。
- cancellation 检查。
- success、cancel、exception 三条路径统一 cleanup。

保留：

- `encode_chat(prompt)`，改为调用 `encode_messages([user message])`。
- `greedy_generate(prompt)`，改为调用共享 generation loop。
- 原有返回字段和 Stage 7.9 CLI 行为。

### 8.5 修改 `src/sparseflow/cli.py`

新增 `serve` parser 与 handler。

约束：

- 顶层不得导入 `serving.py` 或 `text_runtime.py`。
- 只有进入 `serve` handler 后才加载 runtime extras。
- `preset`、`inspect`、`plan`、`doctor` 的 no-Torch regression tests 必须继续通过。
- `serve` 只允许 `stable` 和 `low-memory`。
- `experimental-batch` 必须在参数解析或配置校验阶段失败。

### 8.6 新增测试

- `tests/test_server.py`
- `tests/test_serving.py`
- 在 `tests/test_text_runtime.py` 增加 message/callback/cancel tests。
- 在 `tests/test_release.py` 增加 no-Torch server import boundary tests。

### 8.7 新增 benchmark acceptance runner

- `benchmarks/run_stage7_10_server_acceptance.py`
- 输出到 `benchmarks/results/<date>/stage7_10_server/`
- raw 大文件继续放实验机 E-drive 对应目录或 `/root/workspace/cache/`，不得提交模型 payload。

### 8.8 更新文档

- `README.md`
- `docs/stage7_10_server_plan.md`
- `docs/results/qwen36_stage7_10_server_<date>.md`
- `.handoff.md` 只在正式验收后由对应 owner 更新。

## 9. Python contract

### 9.1 `ServingConfig`

字段必须固定为：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_dir` | `Path` | Qwen model directory |
| `int8_container` | `Path` | canonical INT8 container |
| `preset` | `str` | `stable` 或 `low-memory` |
| `cache_bytes` | `int or None` | low-memory override |
| `telemetry_level` | `str` | 默认 `summary` |
| `context_tokens` | `int` | doctor 与 request admission 上限 |
| `max_completion_tokens` | `int` | server 全局 generation cap |
| `model_id` | `str` | API model id |
| `host` | `str` | bind host |
| `port` | `int` | bind port |
| `api_key` | `str or None` | Bearer credential |
| `cors_origins` | `tuple[str, ...]` | exact origin allowlist |
| `max_queue` | `int` | 等待队列容量 |
| `queue_timeout_seconds` | `float` | 排队超时 |
| `keepalive_seconds` | `float` | SSE idle keepalive interval |

构造后必须验证：

- model 与 container 使用绝对路径。
- port 在 `1..65535`。
- max queue 不小于 0。
- queue timeout 大于 0。
- keepalive 大于 0。
- max completion tokens 大于 0。
- context tokens 大于 max completion tokens。
- remote bind 无 API key 时拒绝启动，除非用户显式传入 `--allow-unauthenticated-remote`。

### 9.2 `GenerationRequest`

字段：

| 字段 | 类型 |
|---|---|
| `request_id` | `str` |
| `model` | `str` |
| `messages` | normalized message tuple |
| `max_completion_tokens` | `int` |
| `stream` | `bool` |
| `include_usage` | `bool` |
| `temperature` | `float`，只能为 `0.0` |
| `top_p` | `float`，只能为 `1.0` |

### 9.3 `GenerationResult`

字段：

| 字段 | 类型 | 来源 |
|---|---|---|
| `text` | `str` | runtime final decode |
| `prompt_tokens` | `int` | input ids length |
| `completion_tokens` | `int` | generated ids length |
| `finish_reason` | `str` | `stop`、`length` 或 `cancelled` |
| `generated_ids_sha256` | `str` | compact correctness evidence |
| `text_sha256` | `str` | compact correctness evidence |
| `route_fingerprint` | `str or None` | compact route audit |
| `prefill_seconds` | `float` | runtime result |
| `decode_seconds` | `float` | runtime result |
| `decode_tokens_per_second` | `float or None` | derived metric |
| `memory` | compact dict | current/peak RSS |
| `cache` | compact dict or `None` | hit/miss/eviction/bytes/leases |
| `runtime_identity` | dict | Stage 7.9 identity |

不得在普通 API response 中返回：

- full logits；
- raw route records；
- model tensor names；
- local stack trace；
- API key；
- environment secrets。

### 9.4 `GenerationEngine` protocol

Protocol 方法必须固定为：

| 方法 | 返回 | 语义 |
|---|---|---|
| `start()` | `None` | 启动唯一 background loader；只允许一次 |
| `snapshot()` | detached `dict` | 返回 lifecycle/runtime compact snapshot |
| `generate(request, on_delta, client_disconnected)` | `GenerationResult` | 串行执行一次 generation |
| `cancel(request_id)` | status string | 设置 active request cancel event |
| `close()` | `None` | 幂等 shutdown |

`generate()` 内部将 explicit cancel event 与 `client_disconnected()` 合并成一个 `is_cancelled()` callback，再传给 `Qwen36TextRuntime.generate_messages()`。HTTP 层不得直接访问 cancel event。

`snapshot()` 返回的新 dict 不得共享 engine 内部 mutable object。FakeEngine 必须实现同一 protocol，HTTP tests 不允许使用 Qwen runtime。

## 10. Runtime lifecycle

`RuntimeState` 必须只允许以下状态：

```text
created -> loading -> ready -> busy -> ready
                    -> error
ready or error -> stopping -> stopped
```

规则：

1. `SparseFlowEngine.start()` 只能调用一次。
2. `start()` 创建一个 loader thread，不阻塞 HTTP server 开始监听。
3. loader thread 首先运行 `doctor(..., check_native=True)`。
4. `doctor.ready == false` 时进入 `error`，不得构造 runtime。
5. `doctor.ready == true` 时调用 `create_public_runtime()`。
6. runtime 构造成功后进入 `ready`。
7. scheduler admission 成功且 generation 开始时进入 `busy`。
8. generation 正常、取消或失败后，只要 runtime 仍可用，都回到 `ready`。
9. 不可恢复的 runtime error 进入 `error`。
10. `close()` 设置所有 active cancel event，关闭 scheduler，等待 active generation 返回，然后只调用一次 `runtime.close()`。
11. `close()` 必须幂等。

`GET /health` 在 loading 阶段必须可用。Chat endpoint 在非 ready/busy 状态收到请求时返回 503，不等待无限期加载。

## 11. Runtime factory

当前 `cli.py` 的 `run` handler 内联构造 `Qwen36TextRuntime`。Stage 7.10 必须把 single-request runtime 构造抽取到 `serving.py:create_public_runtime()`，然后 CLI `run` 与 server 同时调用它。

Factory 必须执行：

1. `apply_preset(preset, cache_bytes, telemetry_level)`。
2. 验证 `batch_mode == "single"`。
3. 调用 `Qwen36TextRuntime.from_pretrained()`。
4. 参数与 Stage 7.9 保持一致：
   - dtype `bf16`；
   - load mode 来自 preset；
   - expert storage 来自 preset；
   - native dispatch 来自 preset；
   - prefetch workers 来自 preset；
   - cache policy 来自 preset；
   - prefetch policy 来自 preset；
   - experts implementation `eager`；
   - coalesce gap `0`。
5. resident preset 使用 16 cache slots 的旧参数语义时，不得创建 streaming cache。
6. low-memory preset 使用 byte budget 时，`cache_slots=None`。

不得在 CLI 和 server 中维护两份参数映射。

## 12. Message normalization

`normalize_messages()` 只接收非空 JSON array。

允许 role：

- `system`
- `developer`
- `user`
- `assistant`

规则：

1. 每个 message 必须是 object。
2. `content` 可以是 string。
3. `content` 可以是 text parts array；part type 只允许 `text` 或 `input_text`。
4. image、audio、file、tool content 返回 `unsupported_content_type`。
5. `developer` 在传入 tokenizer 前规范化为 `system`，直到 Qwen3.6 template 的 developer role 被正式验证。
6. assistant content 在 Stage 7.10 必须是 string；纯 tool call assistant message 不支持。
7. 不得手工拼接 Qwen special token。
8. 规范化后的 messages 必须传给 tokenizer 的 `apply_chat_template()`。

`Qwen36TextRuntime.encode_messages()` 必须使用：

- `tokenize=True`
- `add_generation_prompt=True`
- `return_tensors="pt"`
- `return_dict=True`

`encode_chat(prompt)` 必须等价于单条 user message 的 `encode_messages()`。现有 CLI 32/128/512-token correctness 不得因为该重构发生变化。

完成 tokenization 后必须检查：

```text
prompt_tokens + max_completion_tokens <= context_tokens
```

超出时抛出 typed context-length error，由 HTTP 层映射为 400 `context_length_exceeded`。不得依靠底层 attention 或 position index 报错后返回 500。

## 13. Generation loop 重构

当前 `greedy_generate()` 将 prompt encoding、prefill、decode、telemetry 和 final decode 放在一个函数内。重构必须保留算子顺序和 argmax 行为。

推荐结构：

1. `encode_chat(prompt)` 生成 inputs。
2. `encode_messages(messages)` 生成 inputs。
3. `_greedy_generate_inputs(inputs, request_metadata, max_new_tokens, stop_on_eos, on_text_delta, is_cancelled)` 执行唯一 generation loop。
4. `greedy_generate(prompt)` 调用 1 和 3。
5. `generate_messages(messages)` 调用 2 和 3。

不得复制两份 decode loop。

### 13.1 Persistent runtime 的请求级重置

Stage 7.9 每次 CLI generation 都会新建 runtime，Stage 7.10 则会复用同一个 runtime。开始每个 generation 前必须显式执行：

1. `self.telemetry.reset()`，避免 timing/forward/provider summary 跨请求累计。
2. 清空 `self.route_audit.records`，避免 route records 随 server 生命周期无限增长。
3. 确认 provider 没有上一请求遗留的 active generation/prefetch state。
4. 记录 provider/cache 的 generation-start counters，用 delta 计算本请求 metrics。
5. 不清空 low-memory LRU cache；expert cache 跨请求保留是 server 常驻的预期收益。

请求结束后 `serving.py` 只能保存 compact hashes/counters，不能把完整 runtime result、full routes 或 captured logits 保存在 `last_generation`。

### 13.2 Callback 顺序

1. Prefill 完成。
2. 选出第一个 token。
3. 把第一个 token 交给 text streamer。
4. 每次 decode 后选出 token。
5. 把 token 交给 text streamer。
6. EOS 或 max token 结束后 flush streamer。
7. final text 仍由全部 generated ids decode 得到。

### 13.3 Text delta

不要对每个 token 单独调用 `tokenizer.decode([token])`。这会破坏 byte fallback、Unicode 和前导空格。

实现一个 callback wrapper，复用 Transformers `TextStreamer` 的 token cache 与 flush 行为：

- generation loop 每产生一个 token 调用 streamer `put()`；
- streamer 的 finalized text callback 调用 `on_text_delta(text)`；
- generation 结束调用 streamer `end()`；
- 未提供 callback 时不构造 streamer；
- final result text 仍使用一次完整 decode，作为 correctness source of truth。

必须测试：

- ASCII；
- 中文；
- token boundary 中的前导空格；
- Unicode replacement 不重复；
- 所有 delta 拼接后等于 final text。

### 13.4 Cancellation

`is_cancelled()` 检查点必须放在：

- prefill 开始前；
- prefill 返回后；
- 每次 decode forward 开始前；
- 每次 decode forward 返回后；
- callback 写入失败后。

当前 Stage 不要求强行中断正在运行的单个 PyTorch forward。因此 cold prefill 中的取消延迟可能等于一次 prefill 时间，必须在 API 文档中说明。

无论 success、cancel 或 exception，必须在 `finally` 中保证：

- `provider.finish_generation()` 最多调用一次；
- outstanding prefetch 被清理；
- pinned lease 归零；
- text streamer 被结束；
- telemetry 状态不会污染下一请求。

## 14. OpenAI-compatible request contract

### 14.1 支持字段

| 字段 | 规则 |
|---|---|
| `model` | 必须等于 server model id |
| `messages` | 必须非空且通过 text-only normalization |
| `max_completion_tokens` | `1..server_cap` |
| `max_tokens` | 作为兼容 alias；不能与前者冲突 |
| `temperature` | 缺省或 `0` |
| `top_p` | 缺省或 `1` |
| `n` | 缺省或 `1` |
| `stream` | boolean |
| `stream_options.include_usage` | boolean |
| `response_format` | 缺省或 `{"type":"text"}` |

### 14.2 必须拒绝

| 字段 | error code |
|---|---|
| nonzero `temperature` | `sampling_not_supported` |
| `top_p != 1` | `sampling_not_supported` |
| `n != 1` | `unsupported_value` |
| `stop` | `unsupported_parameter` |
| `logprobs` | `unsupported_parameter` |
| penalties | `unsupported_parameter` |
| `seed` | `unsupported_parameter` |
| `tools`、`functions`、`tool_choice` | `tools_not_supported` |
| multimodal content | `unsupported_content_type` |
| `cache_slot` | `persistent_session_not_supported` |
| non-text response format | `unsupported_parameter` |

默认值必须反映真实 runtime：

- `temperature = 0.0`
- `top_p = 1.0`
- `n = 1`
- `max_completion_tokens = min(256, server_cap)`

不得复制 Colibri 的 `temperature=0.7` 和 `top_p=0.9` 默认值。

## 15. HTTP endpoints

### 15.1 `GET /health`

无需 API key，返回最小健康信息：

```json
{
  "status": "ready",
  "ready": true,
  "model": "qwen3.6-35b-a3b-sparseflow",
  "preset": "low-memory",
  "scheduler": {
    "active": false,
    "active_request_id": null,
    "queued": 0,
    "max_queue": 8,
    "queue_timeout_seconds": 300.0,
    "admitted": 0,
    "completed": 0,
    "failed": 0,
    "rejected": 0,
    "timed_out": 0,
    "cancelled": 0
  },
  "runtime": {
    "session_mode": "stateless-full-message-replay",
    "persistent_kv_supported": false
  }
}
```

`status` 允许值：`loading`、`ready`、`busy`、`error`、`stopping`、`stopped`。

### 15.2 `GET /v1/models`

需要 auth。返回一个 model：

```json
{
  "object": "list",
  "data": [{
    "id": "qwen3.6-35b-a3b-sparseflow",
    "object": "model",
    "created": 1784770000,
    "owned_by": "sparseflow"
  }]
}
```

### 15.3 `GET /v1/models/{model_id}`

需要 auth。model id 不匹配返回 404 `model_not_found`。

### 15.4 `GET /v1/runtime`

需要 auth。返回可供 UI 展示的状态，不返回大数组：

```json
{
  "schema_version": 1,
  "state": "ready",
  "preset": "low-memory",
  "model_id": "qwen3.6-35b-a3b-sparseflow",
  "session_mode": "stateless-full-message-replay",
  "persistent_kv_supported": false,
  "runtime_identity": {
    "runtime_id": "qwen36-text-memory-native-v1",
    "expert_module_id": "SparseFlowQwenExperts-v1",
    "dispatch_id": "qwen36-native-grouped-prefill-canonical-decode-v1",
    "kernel_id": "int8-w8a8-avx512-vnni-linear-silu-linear-v1"
  },
  "model": {
    "metadata_sha256": "75c3ff47bb3f96eee08facdf700ccec7da9a0b37e8c1d4003e251eb05542d735"
  },
  "container": {
    "format_id": "canonical-int8-v1",
    "metadata_sha256": "ed1968b1157a57f86982b34c71db206a4edb7fc911289dbc89641ccbe1b9f898"
  },
  "last_generation": null
}
```

`last_generation` 只保留 compact metrics。不得返回 prompt 原文或完整 message history。

### 15.5 `POST /v1/chat/completions`

需要 auth。runtime 非 ready/busy 时返回 503。

非流式成功响应：

```json
{
  "id": "chatcmpl-0123456789abcdef",
  "object": "chat.completion",
  "created": 1784770000,
  "model": "qwen3.6-35b-a3b-sparseflow",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Sparse expert routing activates only a subset of experts.",
      "refusal": null
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 17,
    "completion_tokens": 10,
    "total_tokens": 27
  },
  "sparseflow": {
    "request_id": "req_0123456789abcdef",
    "queue_wait_ms": 0,
    "session_mode": "stateless-full-message-replay"
  }
}
```

### 15.6 `POST /v1/generations/{request_id}/cancel`

需要 auth。

- 已经向客户端返回 `x-request-id` 的 active streaming request：设置 cancel event，返回 202。
- 已完成 request：返回 409 `generation_finished`。
- 未知 request：返回 404 `generation_not_found`。
- cancel 操作必须幂等；重复取消同一 active request 仍返回 202。

OpenAI chat request 在排队期间尚未返回 response headers，因此客户端还不知道 server-generated request id。Queued request 通过关闭原 HTTP connection 取消，由 scheduler 的 disconnect probe 移除 ticket；Stage 7.10 不引入 client-supplied request id。

返回：

```json
{
  "id": "req_0123456789abcdef",
  "status": "cancellation_requested"
}
```

## 16. SSE contract

响应 headers：

- `Content-Type: text/event-stream`
- `Cache-Control: no-cache`
- `X-Accel-Buffering: no`
- `x-request-id: <request_id>`
- `x-sparseflow-queue-wait-ms: <milliseconds>`
- CORS headers

发送顺序：

1. assistant role chunk；
2. 零个或多个 content delta；
3. finish chunk；
4. 可选 usage chunk；
5. `data: [DONE]`；
6. flush 并关闭本次 HTTP connection。

初始 chunk：

```text
data: {"id":"chatcmpl-0123456789abcdef","object":"chat.completion.chunk","created":1784770000,"model":"qwen3.6-35b-a3b-sparseflow","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

```

Content chunk：

```text
data: {"id":"chatcmpl-0123456789abcdef","object":"chat.completion.chunk","created":1784770000,"model":"qwen3.6-35b-a3b-sparseflow","choices":[{"index":0,"delta":{"content":"Sparse"},"logprobs":null,"finish_reason":null}]}

```

Keepalive 必须使用 SSE comment，不得伪造 model delta：

```text
: sparseflow-keepalive

```

规则：

- keepalive 默认每 10 秒检查一次 idle gap。
- content 正常输出时不得额外发送 keepalive。
- keepalive 与 content 写入必须共享同一个 lock。
- 第一次 write failure 设置 request cancel event。
- generation 结束必须 stop 并 join keepalive thread。
- `stream_options.include_usage=true` 时，usage chunk 使用空 choices。
- server-side exception 若发生在 headers 发送前，返回普通 JSON error。
- exception 若发生在 SSE headers 发送后，发送一个带 `error` object 的 data frame，然后发送 `[DONE]`，不得尝试改 HTTP status。

## 17. Scheduler contract

继续使用单 active request 的 bounded FIFO。

规则：

1. 完成 JSON schema、auth、model 和 runtime state 校验后才进入 scheduler。
2. queue 满时在 SSE headers 发送前返回 429。
3. queue timeout 在 SSE headers 发送前返回 429。
4. queued client disconnect 时从 deque 移除 ticket。
5. active request 结束后必须在 `finally` 中释放 active slot。
6. runtime generation exception 计入 `failed`，不能计入 `completed`。
7. cancellation 单独计入 `cancelled`。
8. scheduler close 唤醒所有 waiter，并返回 503 `scheduler_closed`。
9. scheduler 不得创建第二个 runtime。
10. scheduler 不得把请求合并成 grouped batch。

`max_queue=0` 表示 active request 存在时立即拒绝新的请求。

## 18. Auth、CORS 与网络安全

默认：

- host `127.0.0.1`
- port `8000`
- CORS origins：
  - `http://127.0.0.1:5173`
  - `http://localhost:5173`
  - `http://tauri.localhost`
  - `tauri://localhost`

API key 来源优先级：

1. `--api-key`
2. `SPARSEFLOW_API_KEY`
3. `None`

安全规则：

- loopback bind 可以无 key。
- 非 loopback bind 必须有 key。
- 只有显式 `--allow-unauthenticated-remote` 可以绕过该启动 gate。
- API key 不得写入 health、runtime、日志或 exception。
- `/health` 可匿名访问，但只返回最小状态。
- `/v1/*` 与 cancel endpoint 必须 auth。
- CORS 未命中的 origin 不返回 allow-origin。
- `*` origin 只允许用户显式传入。
- request body 上限固定为 4 MiB。
- HTTP error 不返回 traceback。
- 完整 traceback 只进入 server stderr 或结构化本地日志。

## 19. CLI contract

新增参数：

```text
sparseflow serve MODEL
  --preset {stable,low-memory}
  --int8-container PATH
  --cache-bytes SIZE
  --ctx TOKENS
  --max-completion-tokens TOKENS
  --telemetry-level {none,summary,profile,layer}
  --host HOST
  --port PORT
  --model-id ID
  --api-key KEY
  --cors-origin ORIGIN
  --max-queue COUNT
  --queue-timeout SECONDS
  --keepalive-seconds SECONDS
  --allow-unauthenticated-remote
```

默认值：

| 参数 | 默认 |
|---|---|
| preset | `low-memory` |
| ctx | `4096` |
| max completion tokens | `256` |
| telemetry | `summary` |
| host | `127.0.0.1` |
| port | `8000` |
| model id | `qwen3.6-35b-a3b-sparseflow` |
| max queue | `8` |
| queue timeout | `300` |
| keepalive | `10` |

启动顺序：

1. parse arguments；
2. build validated `ServingConfig`；
3. construct `SparseFlowEngine`，此时不导入 runtime extras；
4. bind HTTP socket；
5. start runtime loader thread；
6. print listening URL 和 state；
7. serve forever；
8. SIGINT/SIGTERM 时 graceful shutdown。

如果 runtime extras 缺失，HTTP server 可以进入 error state并在 `/health` 报告不可用，CLI stderr 必须给出安装 `.[runtime]` 的明确说明。不得出现 import traceback。

## 20. Error contract

Error body：

```json
{
  "error": {
    "message": "The requested model is not available.",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

必须实现：

| HTTP | code | 场景 |
|---:|---|---|
| 400 | `invalid_json` | JSON 解析失败 |
| 400 | `invalid_request` | body 不是 object |
| 400 | `unsupported_parameter` | 不支持字段 |
| 400 | `sampling_not_supported` | sampling 请求 |
| 400 | `unsupported_content_type` | multimodal content |
| 400 | `persistent_session_not_supported` | cache slot |
| 400 | `context_length_exceeded` | prompt 与 completion cap 超过 context |
| 401 | `invalid_api_key` | auth 失败 |
| 404 | `not_found` | path 不存在 |
| 404 | `model_not_found` | model id 不匹配 |
| 404 | `generation_not_found` | cancel 未知 request |
| 409 | `generation_finished` | cancel 已完成 request |
| 429 | `queue_full` | queue 饱和 |
| 429 | `queue_timeout` | 等待超时 |
| 503 | `runtime_loading` | runtime 仍在加载 |
| 503 | `runtime_unavailable` | doctor/load 失败 |
| 503 | `scheduler_closed` | shutdown |
| 500 | `engine_error` | generation 内部错误 |

所有 response 带 `x-request-id`。429 response 带 `Retry-After: 1`。

## 21. Telemetry contract

### 21.1 每请求必须记录

- request id；
- model id；
- preset；
- enqueue timestamp；
- queue wait；
- generation start/end；
- TTFT；
- prompt token count；
- completion token count；
- decode seconds；
- decode tok/s；
- finish reason；
- current RSS；
- process peak RSS；
- cache hit/miss/eviction；
- cache bytes；
- provider logical read bytes；
- pinned entries/bytes；
- runtime identity；
- generated ids hash；
- output text hash；
- compact route fingerprint；
- error code 或 cancellation state。

### 21.2 不得记录

- API key；
- Authorization header；
- 默认情况下的 prompt/message 原文；
- full logits；
- full routes；
- model payload。

### 21.3 `/v1/runtime`

只保留最后一次 generation compact metrics 和累计 scheduler/cache summary。所有 snapshot 在 lock 下复制，HTTP handler 不得持有内部 mutable dict 引用。

## 22. 详细实施顺序

### 22.0 License gate 与 baseline

文件：

- `LICENSE`
- `NOTICE`
- `README.md`

动作：

1. 由维护者确认 SparseFlow license。
2. 记录 Colibri attribution。
3. 在改代码前运行当前 test suite。
4. 保存 baseline commit、Python 版本和 test result。
5. 确认工作区干净。

验收：

- License decision 有明确 commit。
- Stage 7.9 release tests 未回归。
- 未安装任何 server framework dependency。

建议 commit：`Document server attribution and Stage 7.10 license boundary`

### 22.1 Generic HTTP gateway

文件：

- 新增 `src/sparseflow/server.py`
- 新增 `src/sparseflow/serving_types.py`
- 新增 `tests/test_server.py`

动作：

1. 从 Colibri 迁移 `APIError`、scheduler、auth、CORS、body parsing 和 model routes。
2. 删除所有 GLM template、tool parser、sentinel engine code。
3. 在 `serving_types.py` 定义完整 dependency-free `GenerationEngine` contract。
4. 使用 FakeEngine 实现非流式 chat test。
5. 实现 SSE role/content/finish/usage/`[DONE]`。
6. keepalive 使用 SSE comment。
7. 实现 queue full、timeout、FIFO 和 shutdown。
8. 实现 loading/error engine fake state。
9. 实现 cancel registry 与 cancel endpoint。

本步骤的 `server.py` 与 `serving_types.py` 不得 import `serving.py`、Torch 或 Transformers。

验收：

- Colibri 可迁移的 19 类测试都有 SparseFlow 对应测试。
- `python -S` 环境可 import `sparseflow.server`。
- fake SSE 的 delta 拼接等于 fake final text。
- queue 并发测试稳定重复 20 次。
- test 结束没有残留 server thread。

建议 commit：`Add dependency-free OpenAI HTTP gateway`

### 22.2 Qwen message 与 streaming generation API

文件：

- 修改 `src/sparseflow/text_runtime.py`
- 修改 `tests/test_text_runtime.py`

动作：

1. 新增 `encode_messages()`。
2. 让 `encode_chat()` 委托给它。
3. 抽取唯一 `_greedy_generate_inputs()`。
4. 让原 `greedy_generate()` 委托给共享 loop。
5. 新增 `generate_messages()`。
6. 增加 callback text streamer。
7. 增加 cancellation checkpoints。
8. 每个 request 开始时 reset telemetry 和 route audit。
9. 把 provider finalization 放入严格 `try/finally`。
10. 保持 generated ids、text、route 与 logits 顺序不变。

验收：

- 原有 text runtime tests 全部通过。
- 单 user message 与旧 prompt API token ids exact。
- callback 拼接文本与 final text exact。
- cancel 前、prefill 后、decode 中三类测试通过。
- exception path lease 归零。
- 连续两次 generation 的 telemetry/route records 不跨请求累计。
- 连续两次 generation 保留 expert LRU cache。
- Stage 7.9 CLI API 返回字段没有删除或改名。

建议 commit：`Add message-aware cancellable streaming generation`

### 22.3 Persistent runtime adapter

文件：

- 新增 `src/sparseflow/serving.py`
- 新增 `tests/test_serving.py`
- 修改 `src/sparseflow/cli.py` 的 `run` handler 使用 shared factory

动作：

1. 实现 dataclasses 与 protocol。
2. 实现 lifecycle state machine。
3. 实现 doctor preflight。
4. 抽取 `create_public_runtime()`。
5. CLI `run` 改用 shared factory。
6. 实现 background loader。
7. 实现 generate/cancel/close。
8. 实现 compact metrics。
9. 使用 fake doctor/factory 验证失败与 shutdown。

验收：

- runtime factory 在连续请求中只调用一次。
- CLI `run` preset 参数映射与修改前 exact。
- doctor fail 时 runtime factory 调用次数为 0。
- close 调用两次不会 double-close provider。
- active cancel 后 state 回到 ready。
- fatal load error 保持 error state。

建议 commit：`Add persistent SparseFlow serving engine`

### 22.4 CLI `serve` integration

文件：

- 修改 `src/sparseflow/cli.py`
- 修改 `tests/test_release.py`
- 修改 `README.md`

动作：

1. 添加 parser。
2. 添加 config validation。
3. handler 内 lazy import serving/server。
4. 绑定 socket 后启动 background load。
5. 实现 signal shutdown。
6. 输出启动 URL、model id、preset 和 health URL。
7. README 添加 curl 示例。

验收：

- `sparseflow serve --help` 不导入 Torch。
- no-Torch 环境运行 serve 后进入结构化 error state，不输出 traceback。
- `preset`、`inspect`、`plan`、`doctor` no-Torch tests 继续通过。
- experimental batch 被拒绝。
- remote bind 无 key 被拒绝。
- port 0 可用于 test server。

建议 commit：`Expose Qwen local server through CLI`

### 22.5 Local protocol acceptance

环境：本机，不加载模型。

动作：

1. FakeEngine 启动真实 TCP server。
2. 使用 `urllib.request` 测试 endpoint。
3. 使用碎片化 Unicode delta 测试 SSE。
4. 运行并发 queue tests。
5. 测试 client disconnect。
6. 测试 cancel endpoint。
7. 测试 keepalive comment。
8. 测试 auth 与 CORS。
9. 测试所有 unsupported fields。
10. 检查 thread/socket cleanup。

验收：

- server unit tests 全部通过。
- test suite 不需要新增 pip dependency。
- repeated run 没有 flaky queue order。
- `git diff --check` 通过。

建议 commit：`Complete local server protocol acceptance`

### 22.6 Experiment-host Qwen integration

环境：Linux AVX-512 VNNI 实验机，使用 frozen model/container identity。

必须分别启动两个独立 server process：

1. `stable` server。
2. `low-memory` server。

每个 process 执行：

1. 启动后立即轮询 `/health`，保存 loading -> ready transition。
2. 调用 `/v1/models`。
3. 调用 `/v1/runtime`。
4. 单 user prompt 非流式生成 32 tokens。
5. 相同 prompt 流式生成 32 tokens。
6. 将 SSE delta 拼接并与非流式 text 对比。
7. 与 frozen CLI 同 prompt generated ids hash/text hash 对比。
8. 运行中文、英文、代码、数学四类 prompt。
9. 运行四轮 stateless full-message replay。
10. 发起两个并发请求，验证第二个进入 queue，runtime 未重复加载。
11. stream 首 token 后取消，验证 generation 停止和 leases zero。
12. cold prefill 验证 keepalive，客户端连接不因 idle timeout 断开。
13. 连续运行 10 个短请求，验证 RSS 没有持续无界增长。
14. graceful shutdown 后检查 provider/cache/native resources。

不得在一个 process 中先加载 resident 再测 low-memory RSS。

建议 commit：`Record Stage 7.10 Qwen server acceptance`

### 22.7 Release freeze

文件：

- `docs/results/qwen36_stage7_10_server_<date>.md`
- compact JSON artifacts
- `README.md`
- `.handoff.md`

动作：

1. 锁定 final code commit。
2. 重新运行 local tests。
3. 重新运行 experiment smoke matrix。
4. 记录 model/container/runtime identity。
5. 记录所有已支持与未支持字段。
6. 更新 frontend API contract。
7. Board 进行 GO/NO-GO review。

建议 commit：`Complete Stage 7.10 local server validation`

## 23. 本地测试矩阵

### 23.1 Scheduler

- 空闲时立即 admission。
- 两个 waiter FIFO。
- `max_queue=0` 返回 429。
- queue timeout 返回 429。
- queued disconnect 移除 ticket。
- queued HTTP disconnect 触发取消。
- close 唤醒 waiter。
- success 计 completed。
- exception 计 failed。
- cancel 计 cancelled。

### 23.2 HTTP

- health loading、ready、busy、error。
- model list 与 model get。
- auth success/failure。
- CORS exact allowlist。
- CORS rejected origin。
- OPTIONS。
- invalid content length。
- body 过大。
- invalid JSON。
- wrong model。
- unknown route。
- non-stream chat。
- stream chat。
- usage stream。
- queue full HTTP response。
- loading 503。
- engine 500 无 traceback。
- cancel 202/404/409。

### 23.3 Request validation

- empty messages。
- invalid role。
- string content。
- text parts。
- image rejection。
- `temperature=0`。
- nonzero temperature rejection。
- `top_p=1`。
- non-default top-p rejection。
- max token lower/upper boundary。
- context length boundary。
- conflicting max token aliases。
- invalid stream type。
- invalid stream options。
- tools rejection。
- cache slot rejection。

### 23.4 SSE

- initial role chunk。
- multiple content chunks。
- Unicode split safety。
- keepalive comment。
- finish reason stop。
- finish reason length。
- usage chunk。
- `[DONE]`。
- write failure cancellation。
- keepalive thread exit。
- connection close。

### 23.5 Runtime adapter

- doctor pass -> factory once -> ready。
- doctor fail -> error。
- missing runtime extras -> error without traceback response。
- two sequential requests reuse runtime。
- cancellation state recovery。
- fatal load error。
- close idempotence。
- compact telemetry strips prompt and logits。

### 23.6 Import boundary

以下命令必须在无 Torch site-packages 环境运行：

```bash
PYTHONPATH=src python -S -m sparseflow preset stable --json
PYTHONPATH=src python -S -m sparseflow inspect "$MODEL" --json
PYTHONPATH=src python -S -m sparseflow plan "$MODEL" --ram 16 --json
PYTHONPATH=src python -S -c "import sparseflow.server"
```

## 24. 实验机正式验收矩阵

| Cell | Preset | API | Stream | Tokens | Repeats |
|---|---|---|---:|---:|---:|
| S1 | stable | chat | false | 32 | 2 |
| S2 | stable | chat | true | 32 | 2 |
| L1 | low-memory 4 GiB | chat | false | 32 | 2 |
| L2 | low-memory 4 GiB | chat | true | 32 | 2 |
| C1 | low-memory 4 GiB | four-turn replay | true | 16/turn | 1 |
| Q1 | low-memory 4 GiB | two concurrent requests | true | 16 | 3 |
| X1 | low-memory 4 GiB | cancellation | true | 64 cap | 3 |
| K1 | low-memory 4 GiB cold | keepalive | true | 4 | 3 processes |

每个 cell 记录：

- code commit；
- git clean；
- model hash；
- container hash；
- runtime identity；
- Python/Torch/Transformers versions；
- CPU 与 threads；
- server preset/config；
- request body hash；
- prompt tokens；
- completion tokens；
- queue wait；
- TTFT；
- decode tok/s；
- current/peak RSS；
- logical read bytes；
- cache counters；
- leases zero；
- HTTP status；
- SSE event count；
- generated ids hash；
- text hash；
- route fingerprint；
- shutdown status。

## 25. 性能 gate

Server 不是新的计算优化，因此性能目标是“不明显增加额外开销”。

Gate：

1. 同 preset、prompt、tokens、threads 下，server direct engine 的 decode tok/s 不低于 CLI baseline 的 95%。
2. HTTP/SSE orchestration 的非模型 CPU 时间平均不超过 10 ms/token。
3. 非流式与流式 decode tok/s 差异不超过 5%。
4. 第二次请求不得重复产生 model load RSS 峰值。
5. 10 个连续短请求后，ready-state RSS 相对第 2 个请求结束后增长不超过 512 MiB；若 allocator 保留导致超出，必须提供可重复证据和稳定 plateau，不能直接判定 pass。
6. queue wait 必须从 TTFT 中单独报告。
7. cold keepalive 不得改变生成 token、route 或 cache accounting。

## 26. 正确性 gate

1. 单 user message 的 encoded ids 与旧 `encode_chat(prompt)` exact。
2. Server stable generated ids/text/routes 与 CLI stable exact。
3. Server low-memory generated ids/text/routes 与 CLI low-memory exact。
4. Stable 与 low-memory 在相同 arithmetic path 下 exact。
5. SSE delta 拼接 text 与 non-stream text exact。
6. 32-token 中文、英文、代码、数学全部 exact。
7. 四轮 replay 中 resident/streaming 每轮 ids/text/routes exact。
8. Cancellation 后下一请求与 clean server 对照 exact。
9. Cancel/exception 后 pinned entries 与 pinned bytes 为 0。
10. Server 不改变 Stage 7.9 default dispatch。

## 27. 发布 gate

Stage 7.10 只有在以下项目全部通过后才能标记 complete：

- License/attribution 已处理。
- No-Torch import boundary 通过。
- 本地 server unit tests 通过。
- 所有 unsupported fields 返回明确错误。
- Server 默认只绑定 loopback。
- Remote unauthenticated bind 有硬 gate。
- Runtime 只加载一次。
- Stable 真实 Qwen API exact。
- Low-memory 真实 Qwen API exact。
- SSE 与 non-stream exact。
- Queue、429、timeout 验证通过。
- Disconnect/cancel cleanup 通过。
- Cold keepalive 验证通过。
- RSS 连续请求验证通过。
- `/health` 与 `/v1/runtime` schema 冻结。
- README 命令在 clean environment 可执行。
- 所有结果包含 clean commit 和 runtime identity。
- Experimental batch 未通过 server 暴露。
- 文档明确声明 stateless conversation、greedy-only、text-only。

## 28. 必须生成的正式产物

```text
benchmarks/results/<date>/stage7_10_server/
  environment.json
  local_protocol_tests.json
  stable_server_matrix.json
  low_memory_server_matrix.json
  streaming_equivalence.json
  conversation_replay.json
  queue_validation.json
  cancellation_validation.json
  cold_keepalive.json
  rss_stability.json
  verification.json
```

`verification.json` 的 gate 必须从以上 artifact 推导，不得写死 `True`。每个 gate 必须能追溯到具体 command、return code、hash 或 measurement。

## 29. 实现者禁止事项

- 不得安装 FastAPI、Flask、Uvicorn、aiohttp 或其他 server framework。
- 不得把 runtime 放进每个 request handler 重新构造。
- 不得让 ThreadingHTTPServer 并发执行两个 Qwen forward。
- 不得使用 grouped fixed cohort 冒充 continuous batching。
- 不得支持非零 temperature 后仍执行 greedy。
- 不得手写 Qwen special-token prompt。
- 不得声称 persistent KV。
- 不得返回 `cache_slot` 能力。
- 不得复制 GLM tool tags。
- 不得使用 fake reasoning delta 作为 keepalive。
- 不得吞掉不支持的 OpenAI 参数。
- 不得在 API response 暴露 traceback、API key 或完整本地环境。
- 不得破坏 no-Torch base CLI。
- 不得修改 Stage 7.9 preset 默认值。
- 不得在没有真实实验机结果时标记 Stage complete。
- 不得把本机 Windows PyTorch DLL 问题误判为 server protocol failure。
- 不得把 cache/model/venv 安装到 C 盘；本地缓存继续放在 E 盘项目目录。

## 30. 低能力实现代理的工作规则

1. 一次只实现一个 22.x 步骤。
2. 每完成一个步骤立即运行该步骤列出的 tests。
3. tests 失败时只修复当前步骤涉及的文件，不重构 native runtime。
4. 不确定字段语义时查本文，不根据 Colibri 名称直接复制。
5. 任何 Torch import 都要检查是否发生在 lazy runtime boundary 之后。
6. 任何并发代码都必须有 deterministic unit test。
7. 任何 state transition 都必须在 lock 下完成。
8. 任何 background thread 都必须有 shutdown 与 join test。
9. 任何 SSE write 都必须经过单一 writer lock。
10. 任何新 API 字段都必须同时更新 schema test 和 README。
11. 任何正式结果都必须记录 commit、git clean、runtime identity。
12. 不得在同一个 commit 同时做 server protocol、runtime kernel 和 frontend UI。

## 31. Review checklist

Reviewer 按顺序检查：

1. Diff 是否只覆盖当前 22.x 步骤。
2. 是否存在顶层 Torch import regression。
3. 是否复用了 shared runtime factory。
4. 是否仍是单 active generation。
5. 是否显式拒绝 sampling/tools/KV slots。
6. message 是否经过 Qwen tokenizer template。
7. SSE 是否按顺序结束。
8. keepalive 是否为 comment。
9. disconnect 是否触发 cancellation。
10. cleanup 是否在 exception path 执行。
11. telemetry 是否不含 prompt 与 secrets。
12. 测试是否覆盖真实 HTTP socket。
13. 实验结果是否来自独立 stable/low-memory process。
14. 性能与正确性 gate 是否由 artifact 推导。
15. README 是否没有超出真实能力的声明。

## 32. Stage 结束后的下一步

Stage 7.10 完成后，Frontend team 可以只依赖冻结 API contract 开发：

- model/server connection；
- loading/ready/error state；
- Chat；
- SSE output；
- stop generation；
- queue state；
- preset/runtime/cache/RSS telemetry。

以下能力进入后续 stage：

- dynamic model management；
- persistent session/KV；
- sampling；
- tool calling；
- multi-request grouped serving；
- desktop process supervision；
- static frontend asset hosting；
- second-model adapter validation。

<!-- [Board] -->

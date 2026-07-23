# Stage 7.11: Laptop Runtime Closure and Frontend Readiness

**Decision owner:** Board
**Runtime owner:** Main Dev
**Server owner:** Server Dev
**Benchmark owner:** Benchmark
**Release reviewer:** Board
**Next consumer:** Frontend team
**Status:** planned
**Recorded:** 2026-07-23

## 1. 文档用途

本文定义 Stage 7.11 的完整实施顺序。Stage 7.11 从当前已经可运行的
Qwen3.6 Windows laptop runtime 出发，一直覆盖到前端可以开始接入之前。
本文不包含前端页面、Tauri packaging 或 UI 视觉实现。

本文可能交给能力较弱的执行模型，因此执行者必须遵守以下规则：

1. 严格按章节顺序实施，不跳过前置验收。
2. 每完成一个步骤，先运行该步骤规定的测试，再进入下一步。
3. 不得根据个人偏好修改 frozen runtime、量化格式或 native kernel。
4. 不得把一次短 smoke 宣传成完整质量、稳定性或性能认证。
5. 遇到本文未定义的架构决策时，暂停并交给 Board，不得自行扩大范围。
6. 模型、INT8 container、venv、临时文件和下载缓存必须继续保留在 E 盘。
7. 不得将 `model/`、`.cache/`、`.tools/`、`.venv-runtime/` 提交到 Git。
8. 正式 benchmark 只能在 clean commit 上运行；dirty worktree 结果只能用于开发诊断。

## 2. Stage 目标

Stage 7.11 的核心目标是：

> 把“16 GiB Windows 笔记本可以运行 Qwen3.6-35B-A3B”从一次成功 smoke，提升为可复现、可观测、可通过 Server 使用，并且具有冻结前端合同的 Public Alpha 路径。

完成后必须具备：

1. Windows production runtime 能报告非零 current RSS 与 peak RSS。
2. 一套可重复执行的 laptop benchmark runner。
3. 中文、英文、代码 prompt 的 8/16/32-token 矩阵。
4. process-cold 与 persistent-server-warm 被清楚区分。
5. `laptop-16gb` preset 经过实测后冻结，默认仍为实验性 opt-in。
6. 真实 `/v1/chat/completions` 非流式请求通过。
7. 真实 SSE streaming、cancel、queue、连续请求和多轮 replay 通过。
8. 同一个 Server 进程只加载一次 dense model/runtime。
9. CLI 与 Server 对同一冻结请求产生相同的 generated IDs/logit fingerprint。
10. 前端所需的 runtime、health、completion、SSE 和错误合同已冻结并有示例 fixture。
11. Windows/Linux 安装文档、已知限制和 laptop 运行说明同步完成。
12. Stage 7.11 verification artifact 给出明确 GO 或 NO-GO。

## 3. 当前已验证基线

以下事实已经在本机真实运行中验证，执行者不得重复把它们描述为假设：

| 项目 | 当前结果 |
|---|---|
| Host | Lenovo laptop, Windows 10 build 19045 |
| CPU | Intel i5-11400H, 6C/12T |
| RAM | 15.85 GiB physical |
| CPU ISA | AVX-512 F + AVX-512 VNNI |
| Python | uv-managed CPython 3.12.13 |
| Torch | 2.9.1+cpu |
| Transformers | 5.14.1 |
| Source model | Qwen3.6-35B-A3B, about 67 GiB |
| INT8 container | complete 40 layers, about 30.23 GiB |
| Offline row sums | complete |
| Native operators | `dynamic_linear`, `fused_moe`, `grouped_moe` pass |
| Windows tests | 122/122 pass at the recorded baseline |
| Doctor, 256 MiB cache | `ready=true`, memory status `warn` |
| Available RAM during smoke | about 9.95 GiB |
| Required RAM | about 8.86 GiB |
| Recommended RAM | about 10.87 GiB |
| 1-token smoke | succeeded |
| 2-token smoke | succeeded, including one decode forward |
| 2-token model load | about 7.17 s |
| 2-token prefill | about 21.83 s |
| One decode token | about 2.23 s, around 0.45 tok/s |
| 2-token expert reads | about 7.77 GiB |
| Cache budget | 256 MiB, observed about 255.5 MiB resident |

当前结果证明技术可行性，但尚未证明：

- 32-token 稳定性；
- 多请求 persistent runtime；
- Server warm cache 收益；
- Windows peak RSS 正确记录；
- SSE/cancel/queue 的真实模型路径；
- 多轮 conversation 的真实模型路径；
- 通用 16 GiB laptop 兼容性；
- 模型质量相对 Stage 7.9 frozen quality evidence 没有回归。

## 4. Stage 边界

### 4.1 必须完成

- Windows production process telemetry。
- Laptop benchmark runner 与 verifier。
- 128/256/512 MiB cache admission smoke。
- 8/16/32-token laptop matrix。
- 中文、英文、代码 prompt。
- Process-cold 和 server-warm 两种状态。
- `laptop-16gb` preset。
- CLI context admission。
- 真实 Server 非流式和 SSE。
- 真实 Server 连续请求。
- 真实 Server cancel、queue、multi-turn replay。
- Runtime load count 和 persistent runtime identity。
- Compact frontend API contract。
- OpenAPI 或等价机器可读 schema。
- Installation/README/results/verification。

### 4.2 明确不完成

- React/Vite/Tauri 前端实现。
- 前端视觉调整。
- GPU hot tier。
- Vision encoder。
- MTP。
- Sampling、top-p、top-k、temperature。
- Tool calling。
- Persistent KV/DeltaNet conversation session。
- Continuous batching。
- Shared streaming batching。
- 新量化格式，包括 INT4。
- 新 native expert kernel。
- 安装 `flash-linear-attention` 或 `causal-conv1d` 作为默认依赖。
- 第二个 MoE model adapter。
- 对所有 16 GiB 笔记本作兼容性保证。

Stage 7.11 是 runtime closure、serving acceptance 和产品合同阶段，不是新的专项 kernel 优化阶段。

## 5. 术语与测量状态

执行者必须使用以下术语，不得混用：

### 5.1 `process-cold`

每个 cell 启动新的 Python/SparseFlow 进程，runtime 重新构造。Windows 文件页可能仍在 OS standby/page cache 中，因此这不是物理 NVMe cold。

### 5.2 `server-first-request`

Server 进程只启动一次。runtime 完成加载后处理的第一个 generation 请求。

### 5.3 `server-warm`

同一个 Server/runtime 进程中的第二次及之后请求。Dense model 不重新加载；low-memory expert cache 保留。

### 5.4 `model-cold`

只有在能明确清理 page cache 或使用 fresh-file/reboot 协议时才允许使用该名称。Stage 7.11 Windows laptop 不要求 model-cold，不得把 process-cold 写成 model-cold。

### 5.5 `correctness PASS`

只有满足冻结的 generated IDs、logit fingerprint、route fingerprint 或 Stage 7.9 quality gate 时才能标记 PASS。HTTP 200、生成了可读文本或进程未崩溃只能标记 runtime success，不能替代 correctness PASS。

## 6. 总体实施顺序

必须按以下顺序执行：

```text
7.11.0 Freeze current baseline and clean commit
        |
        v
7.11.1 Production Windows telemetry
        |
        v
7.11.2 Laptop benchmark harness and fixtures
        |
        v
7.11.3 Cache/context pilot and laptop preset decision
        |
        v
7.11.4 Formal CLI laptop matrix
        |
        v
7.11.5 Real persistent Server acceptance
        |
        v
7.11.6 API and frontend data contract freeze
        |
        v
7.11.7 Installation and release hardening
        |
        v
7.11.8 Final verification and frontend handoff
```

如果某一步是 NO-GO，后续步骤不得假装通过。允许修复该步骤后重跑，不允许降低正确性门槛来获得 GO。

## 7. Stage 7.11.0: 冻结当前基线

### 7.11.0.1 目的

先把 Stage 7.10 Server、本机环境支持、Windows native build、INT8 resume identity 和安装脚本形成一个可追踪基线。正式 benchmark 不得在当前未提交 worktree 上直接开始。

### 7.11.0.2 必须检查

```powershell
git status --short --branch
git diff --check
. .\scripts\use_runtime_windows.ps1
python -m unittest discover -s tests -p 'test_*.py'
```

### 7.11.0.3 必须确认

- `model/` ignored。
- `.cache/` ignored。
- `.tools/` ignored。
- `.venv-runtime/` ignored。
- `_reference/` ignored。
- 没有模型 shard、INT8 container 或临时 build output 出现在 Git status。
- `122/122` 或更高测试数通过。
- `git diff --check` 无 whitespace error。
- Stage 7.10 Server 文件已包含在基线 commit。

### 7.11.0.4 提交建议

建议按以下逻辑提交，不要求使用相同 message：

1. Stage 7.10 Server implementation。
2. Windows runtime and installation support。
3. Windows benchmark telemetry and INT8 resume identity fix。
4. Stage 7.11 plan。

不得把 30 GiB container 或 `.cache/results` 开发输出加入提交。

### 7.11.0.5 验收

- 正式 benchmark commit 是 clean。
- 记录 full commit SHA。
- `git status --porcelain` 为空。
- 如果不能 clean，Stage 7.11 只能继续开发，不得生成 formal result。

## 8. Stage 7.11.1: Production Windows Telemetry

### 8.1 当前问题

`benchmarks/common.py` 已经能在 Windows 返回真实 RSS 和 host memory，但 production runtime 仍通过 `src/sparseflow/memory_loader.py` 中的 Linux/macOS `resource` 实现读取 peak RSS。Windows 上以下字段目前为 `0`：

- `loader.rss_before_materialize`
- `loader.rss_after_materialize`
- `loader.process_peak_rss_at_materialize_end`
- `memory.rss_before_prefill`
- `memory.rss_after_generation`
- `memory.process_peak_rss`
- Server `last_generation.memory`

前端和发布报告不能依赖为零的字段。

### 8.2 新增文件

新增：

```text
src/sparseflow/process_metrics.py
tests/test_process_metrics.py
```

### 8.3 `process_metrics.py` 职责

该文件属于 production package，不得 import `benchmarks`。

必须提供：

```python
def process_snapshot() -> dict[str, int | float]
def current_rss_bytes() -> int
def peak_rss_bytes() -> int
def host_memory_snapshot() -> dict[str, int | str | None]
```

`process_snapshot()` 必须至少包含：

```text
rss_bytes
peak_rss_bytes
private_bytes
read_bytes
read_syscalls
read_bytes_semantics
user_seconds
system_seconds
page_faults
platform_source
```

### 8.4 Windows 实现

使用标准库 `ctypes`：

- `GetCurrentProcess`
- `GetProcessMemoryInfo`
- `GetProcessIoCounters`
- `GetProcessTimes`
- `GlobalMemoryStatusEx`

必须显式设置所有 Windows API 的 `argtypes` 和 `restype`。不得依赖 ctypes 默认 32-bit return type，否则 64-bit pseudo handle 会失效。

Windows 字段映射：

| SparseFlow 字段 | Windows 来源 |
|---|---|
| `rss_bytes` | `WorkingSetSize` |
| `peak_rss_bytes` | `PeakWorkingSetSize` |
| `private_bytes` | `PrivateUsage` |
| `read_bytes` | `ReadTransferCount` |
| `read_syscalls` | `ReadOperationCount` |
| `user_seconds` | `GetProcessTimes` user FILETIME |
| `system_seconds` | `GetProcessTimes` kernel FILETIME |
| `page_faults` | `PageFaultCount` |

Windows `ReadTransferCount` 表示进程 I/O read transfer bytes，可能包含由系统缓存满足的读取。不得把该字段命名或展示为 physical NVMe bytes。Linux `/proc/self/io/read_bytes` 可以单独标记为 storage-accounted bytes，但跨平台汇总必须使用中性名称，并附带 `read_bytes_semantics`。

### 8.5 Linux/macOS 实现

- Linux current RSS 优先 `/proc/self/status` `VmRSS`。
- Linux peak RSS 使用 `resource.getrusage()`，KiB 转 bytes。
- macOS `ru_maxrss` 直接按 bytes。
- 读取失败时返回 typed unavailable source，不得静默伪造有效值。
- Public numeric compatibility 可以使用 `0`，但必须同时返回 `platform_source="unavailable"` 或 availability flag。

### 8.6 修改文件

修改：

```text
src/sparseflow/memory_loader.py
src/sparseflow/text_runtime.py
src/sparseflow/serving.py
benchmarks/common.py
tests/test_memory_loader.py
tests/test_text_runtime.py
tests/test_serving.py
```

要求：

1. `memory_loader.py` 删除重复的 OS-specific RSS 实现，改为 import production helper。
2. `text_runtime.py` 在 load、prefill、generation end 捕获 snapshot。
3. `serving.py:compact_generation_metrics()` 保留 compact memory 字段。
4. `benchmarks/common.py` 复用 production helper，benchmark-specific host metadata 保留在 benchmarks 层。
5. 不得让 base CLI 因 import telemetry 而提前 import Torch。

### 8.7 测试

必须覆盖：

- 当前 RSS 大于 0。
- peak RSS 大于等于 current RSS。
- host total memory 大于 available memory。
- Windows fake API struct mapping。
- Linux `/proc` fixture。
- resource unavailable fallback。
- Server compact metrics 不丢 memory source。
- `preset/inspect/plan/doctor` no-Torch boundary 继续通过。

### 8.8 GO 门槛

- 真实 2-token Windows run 中所有 runtime RSS 字段非零。
- `process_peak_rss >= rss_after_generation`。
- production 与 benchmark 的 current RSS snapshot 在相邻采样时差异小于 64 MiB。
- 完整 unit test 通过。

## 9. Stage 7.11.2: Laptop Benchmark Harness

### 9.1 新增文件

```text
benchmarks/run_stage7_11_laptop.py
benchmarks/verify_stage7_11_laptop.py
benchmarks/manifests/laptop_prompts_v1.jsonl
tests/test_stage7_11_laptop.py
```

### 9.2 Prompt manifest

固定三个 prompt，不允许 benchmark 时临时改写：

```jsonl
{"id":"zh-explain","category":"zh","messages":[{"role":"user","content":"用简洁的中文解释稀疏专家路由的作用。"}]}
{"id":"en-explain","category":"en","messages":[{"role":"user","content":"Explain sparse expert routing and why it reduces active computation."}]}
{"id":"code-python","category":"code","messages":[{"role":"user","content":"Write a Python function that returns the top-k indices from a list of scores."}]}
```

Manifest 文件必须是 UTF-8。Windows PowerShell 查看时使用 `Get-Content -Encoding UTF8`。

### 9.3 Runner 参数

Runner 至少支持：

```text
--model
--int8-container
--output-dir
--cache-bytes
--context-tokens
--max-new-tokens
--prompt-id
--repeats
--mode process|server
--server-url
--api-key
--timeout
--json
```

### 9.4 Runner 环境规则

Windows 子进程必须继承或显式设置：

```text
PYTHONPATH=<repo>/src
UV_CACHE_DIR=<repo>/.cache/uv
UV_PYTHON_INSTALL_DIR=<repo>/.tools/python
XDG_CACHE_HOME=<repo>/.cache
HF_HOME=<repo>/.cache/huggingface
TORCH_HOME=<repo>/.cache/torch
TEMP=<repo>/.cache/tmp
TMP=<repo>/.cache/tmp
SPARSEFLOW_NATIVE_CACHE=<repo>/.cache/native/int8_vnni_windows
PYTHONUTF8=1
```

不得把 TEMP、Torch cache、HF cache 或 native cache 写到 C 盘。

### 9.5 Raw 与 compact artifact

Raw output 写入 ignored 路径：

```text
.cache/results/stage7_11_laptop/raw/
```

可提交 compact output 写入：

```text
benchmarks/results/<date>/stage7_11_laptop/
```

Compact cell 只保留：

```text
schema_version
kind
commit
git_clean
host_identity
runtime_identity
model_identity
container_identity
prompt_id
max_new_tokens
cache_bytes
context_tokens
state_label
repeat
exit_code
generated_tokens
generated_ids_hash
output_text_hash
logit_fingerprints
route_fingerprint
load_seconds
prefill_seconds
decode_seconds
decode_token_seconds
ttft_seconds
decode_tokens_per_second
current_rss_bytes
peak_rss_bytes
private_bytes
logical_expert_read_bytes
process_read_transfer_bytes
process_read_bytes_semantics
cache_hits
cache_misses
cache_evictions
cached_bytes
leases_after
error
```

不得提交 full logits、完整 route arrays 或数千行 provider trace。

### 9.6 Verifier

`verify_stage7_11_laptop.py` 必须检查：

- schema/version。
- commit 不为空。
- formal artifact `git_clean=true`。
- model/container identity 一致。
- runtime identity 一致。
- exit code 为 0。
- generated token 数符合请求或 EOS。
- memory 字段非零。
- cache budget 不突破。
- provider leases 为 0。
- 同一 prompt/repeat 的 fingerprint 可比较。
- Server/CLI correctness pair exact。
- 不允许 `SIMULATED` 数据标记为 PASS。

### 9.7 单元测试

使用 fixture JSON，不加载真实模型：

- compact 正常结果。
- 缺少 identity 失败。
- dirty formal result 失败。
- cache 超预算失败。
- zero RSS 失败。
- mismatched generated hash 失败。
- allowed EOS early stop。
- Windows UTF-8 prompt roundtrip。

## 10. Stage 7.11.3: Cache/Context Pilot and Laptop Preset

### 10.1 目的

先使用现有 `low-memory` preset 加显式 override 测试，再决定是否新增 `laptop-16gb`。不得先写 preset 后寻找支持数据。

### 10.2 Pilot 配置

固定英文 prompt `en-explain`，`max_new_tokens=8`，每个配置独立进程一次：

| Cell | Cache | Context |
|---|---:|---:|
| P1 | 128 MiB | 2048 |
| P2 | 256 MiB | 2048 |
| P3 | 512 MiB | 2048 |
| P4 | 256 MiB | 4096 |

每个 cell 先运行真实 Doctor。Doctor fail 时 cell 标记 `admission-fail`，不得强制运行。

### 10.3 记录指标

- required/recommended/available RAM。
- headroom。
- peak RSS/private bytes。
- TTFT/prefill/decode。
- logical expert read bytes。
- process read transfer bytes 及其平台语义。
- cache entries/hit/miss/eviction。
- generated IDs hash。
- cache budget violations。
- process exit status。

### 10.4 Preset 决策原则

推荐默认候选：

```text
name: laptop-16gb
mode: streaming
load_mode: memory-native
expert_storage: int8-native
native_dispatch: hybrid
cache_policy: lru
cache_bytes: 256 MiB
prefetch_workers: 0
prefetch_policy: none
batch_mode: single
default_context_tokens: 2048
default_max_completion_tokens: 128
public_status: experimental-laptop
```

只有以下条件全部满足才新增 preset：

1. P2 Doctor ready。
2. P2 8-token generation 成功。
3. P2 peak RSS 不超过 11.0 GiB。
4. P2 至少保留 512 MiB runtime headroom。
5. P2 generated hash 与 P4 相同。
6. P2 无 lease/cache accounting error。
7. P2 相比 P1 没有明显吞吐回退。
8. P3 若更快但内存余量不足，仍不得替代 P2 成为默认。

### 10.5 RuntimePreset schema 修改

如果 preset GO，修改：

```text
src/sparseflow/release.py
src/sparseflow/cli.py
src/sparseflow/serving_types.py
src/sparseflow/serving.py
tests/test_release.py
tests/test_serving.py
tests/test_server_contract.py
```

推荐在 `RuntimePreset` 增加：

```text
default_context_tokens
default_max_completion_tokens
```

现有 preset 必须显式填值，保持当前行为：

| Preset | Context | Max completion |
|---|---:|---:|
| stable | 4096 | 256 |
| low-memory | 4096 | 256 |
| experimental-batch | 4096 | 32 |
| laptop-16gb | 2048 | 128 |

### 10.6 CLI 行为

- `run` 新增 `--ctx`。
- `run --ctx` 必须调用 `generate_messages(..., context_tokens=ctx)`，不能继续绕过 context validation。
- `serve --ctx` 和 `--max-completion-tokens` 未指定时使用 preset 默认值。
- 显式 CLI flag 覆盖 preset default。
- `doctor` 未指定 ctx 时使用 preset default。
- `--cache-bytes` 对所有 streaming preset 可用，不能只写死 `low-memory`。
- `stable`/resident 使用 `--cache-bytes` 继续报错。

### 10.7 Server 行为

- Server 允许 `laptop-16gb`。
- `experimental-batch` 仍禁止进入 Server。
- `/v1/runtime` 返回 effective context、max completion、cache budget 和 public status。
- Frontend 不得根据 preset 名称自行推导这些值。

### 10.8 Preset 状态

Stage 7.11 完成时 `laptop-16gb` 只能是 `experimental-laptop`，不得标记 `stable`。至少需要第二台 16 GiB AVX-512 VNNI Windows host 复现后再讨论升级。

## 11. Stage 7.11.4: Formal CLI Laptop Matrix

### 11.1 前置条件

- Telemetry GO。
- Runner GO。
- Preset decision 已完成。
- Formal commit clean。
- 可用 RAM 达到 Doctor required。
- 测试期间关闭浏览器、IDE、微信和其他大内存应用。
- 禁止通过 `--available-ram` 伪造 admission。

### 11.2 矩阵

使用最终候选 laptop 配置：

| Category | Prompt | Output tokens | Repeats | Process state |
|---|---|---:|---:|---|
| zh | `zh-explain` | 8 | 2 | process-cold |
| zh | `zh-explain` | 16 | 2 | process-cold |
| zh | `zh-explain` | 32 | 2 | process-cold |
| en | `en-explain` | 8 | 2 | process-cold |
| en | `en-explain` | 16 | 2 | process-cold |
| en | `en-explain` | 32 | 2 | process-cold |
| code | `code-python` | 8 | 2 | process-cold |
| code | `code-python` | 16 | 2 | process-cold |
| code | `code-python` | 32 | 2 | process-cold |

总计 18 个独立进程 cell。

### 11.3 运行顺序

避免所有长任务集中在最后：

```text
zh-8-r0, en-8-r0, code-8-r0,
zh-16-r0, en-16-r0, code-16-r0,
zh-32-r0, en-32-r0, code-32-r0,
code-8-r1, en-8-r1, zh-8-r1,
code-16-r1, en-16-r1, zh-16-r1,
code-32-r1, en-32-r1, zh-32-r1
```

### 11.4 每个 cell 前检查

1. 读取实际 available RAM。
2. 运行 Doctor 或使用同参数 admission helper。
3. 如果 memory status fail，等待/释放内存，不启动 runtime。
4. 记录其他高内存进程摘要，但不得自动终止用户进程。
5. 确认 E 盘至少有 5 GiB free runtime reserve。

### 11.5 正确性门槛

- 同 prompt、同 output length 的两个 repeat generated IDs hash exact。
- Logit fingerprint sequence exact。
- Route fingerprint exact，或明确说明 route compact 算法版本。
- 中文 prompt UTF-8 roundtrip exact。
- 结果不是空文本，除非模型产生 EOS；EOS 必须记录 generated IDs。
- Provider lease count 为 0。

### 11.6 性能门槛

性能门槛是 release guard，不是营销目标：

| Metric | Initial GO threshold |
|---|---:|
| 8-token run total | <= 60 s |
| 16-token run total | <= 90 s |
| 32-token run total | <= 140 s |
| Prefill P50 | <= 35 s |
| Decode P50 | <= 3.0 s/token |
| Peak RSS | <= 11.0 GiB |
| Cache budget | <= configured budget |
| Crash/OOM | 0 |

如果所有 correctness 通过但性能略低于门槛，Stage 可以标记 runtime correctness GO、performance NO-GO，不得篡改结果。

### 11.7 汇总

输出：

```text
environment.json
doctor_matrix.json
cli_laptop_matrix.json
cli_laptop_summary.json
cli_laptop_verification.json
```

Summary 至少包含 P50/P95：

- load time。
- TTFT/prefill。
- decode seconds/token。
- total wall time。
- peak RSS。
- expert read bytes/token。
- process read transfer bytes/token，并显示平台语义。
- cache hit rate。

## 12. Stage 7.11.5: Real Persistent Server Acceptance

### 12.1 目标

验证 Stage 7.10 Server 在真实 laptop runtime 上工作，而不仅是 FakeEngine/unit test。

### 12.2 必须新增 runner

新增：

```text
benchmarks/run_stage7_11_server_acceptance.py
tests/test_stage7_11_server_acceptance.py
```

Runner 负责：

1. 启动一个 `sparseflow serve` 子进程。
2. 等待 `/health` 从 loading 进入 ready。
3. 记录 runtime load time。
4. 执行下列请求矩阵。
5. 请求结束后发送正常 shutdown 或终止子进程。
6. 确认没有遗留 Server/Python 进程。
7. 写 compact artifact。

### 12.3 Server telemetry 补充

修改 `SparseFlowEngine` snapshot，增加：

```text
runtime_load_count
runtime_loaded_at
runtime_load_seconds
generation_count
last_request_id
effective_config
process_memory
provider/cache compact state
```

约束：

- `runtime_load_count` 在一个正常 Server 生命周期中必须始终为 1。
- 不返回 prompt 原文。
- 不返回完整 message history。
- 不返回 full routes/logits。
- `last_generation` 只保存 compact hashes 和 counters。

### 12.4 Acceptance A: Health and identity

步骤：

1. 启动 Server。
2. 轮询 `/health`，间隔 500 ms，timeout 180 s。
3. 加载阶段必须返回 `loading` 而不是连接失败。
4. Ready 后请求 `/v1/runtime` 和 `/v1/models`。

验收：

- `/health.ready=true`。
- `/v1/runtime.state=ready`。
- model/container/runtime identity 与 CLI formal matrix 一致。
- `runtime_load_count=1`。
- effective cache/context 与 laptop preset 一致。

### 12.5 Acceptance B: Non-streaming correctness

请求：

```json
{
  "model": "qwen3.6-35b-a3b-sparseflow",
  "messages": [{"role":"user","content":"Explain sparse expert routing and why it reduces active computation."}],
  "max_completion_tokens": 8,
  "temperature": 0,
  "stream": false
}
```

验收：

- HTTP 200。
- OpenAI-shaped response。
- finish reason 正确。
- usage prompt/completion/total 一致。
- generated IDs hash 与同配置 CLI exact。
- runtime load count 仍为 1。
- completion 后 state 返回 ready。
- lease count 为 0。

### 12.6 Acceptance C: Persistent warm requests

同一 Server 连续发送相同 8-token 请求 3 次。

记录：

- 每次 TTFT/decode。
- 每次 cache hit/miss/read bytes。
- process RSS。
- runtime load count。

验收：

- 三次 generated hash exact。
- runtime load count 始终为 1。
- 第二、三次不重新读取 dense source payload。
- Server process 不退出。
- cache 保留但不突破 budget。
- warm 请求 expert read bytes 不高于 first request；如果更高，记录为异常并调查。

不得强制要求 warm cache 一定获得性能提升；先以真实数据决定。

### 12.7 Acceptance D: SSE streaming

步骤：

1. 发送 `stream=true`、16-token 请求。
2. 逐行读取 SSE。
3. 忽略 `:` 开头 keepalive comment。
4. 收集所有 `delta.content`。
5. 读取 final usage chunk 和 `[DONE]`。

验收：

- 首个 event role 为 assistant。
- 所有 delta 拼接等于最终 non-stream text。
- SSE 中没有乱码或重复 Unicode。
- `[DONE]` 恰好一次。
- include_usage=true 时 usage 恰好一次。
- Server 状态回到 ready。

### 12.8 Acceptance E: Active cancellation

步骤：

1. 启动 32-token SSE 请求。
2. 收到第一个非空 content delta 后记录时间。
3. 调用 `/v1/generations/{request_id}/cancel`。
4. 继续读取到 stream 正常结束。

验收：

- cancel endpoint 返回 `cancellation_requested` 或等价 documented status。
- finish reason 为 `cancelled`。
- 取消延迟不超过一次 decode forward 加 2 s；当前目标 <= 5 s。
- provider finish_generation 恰好一次。
- leases/transient prefetch 为 0。
- 下一请求仍能成功。

### 12.9 Acceptance F: Queue

步骤：

1. 第一个请求使用 16 tokens，占用 runtime。
2. 在第一个请求执行期间提交第二个 8-token 请求。
3. 检查 `/health.scheduler.queued` 至少观察到 1。
4. 两个请求都等待完成。

验收：

- 执行顺序为 FIFO。
- 同一时刻只有一个 active request。
- 第二个 response 带非零 queue wait header/metric。
- runtime load count 仍为 1。
- 不发生 shared batching。

### 12.10 Acceptance G: Multi-turn stateless replay

请求 1：

```json
{
  "messages": [
    {"role":"user","content":"Remember the code word cedar."}
  ],
  "max_completion_tokens": 8
}
```

请求 2 必须由 client 重发完整历史：

```json
{
  "messages": [
    {"role":"user","content":"Remember the code word cedar."},
    {"role":"assistant","content":"<request-1-text>"},
    {"role":"user","content":"What code word did I give you?"}
  ],
  "max_completion_tokens": 8
}
```

验收：

- Server 接收完整 messages。
- 第二次 prompt token 数大于第一次。
- runtime load count 仍为 1。
- API/runtime snapshot 明确 `session_mode=stateless-full-message-replay`。
- 不出现 `persistent_kv_supported=true`。
- 不以回答是否语义正确作为唯一 contract gate；记录文本并做 runtime correctness hash。

### 12.11 Server artifact

输出：

```text
server_environment.json
server_health_identity.json
server_non_stream.json
server_persistent_requests.json
server_sse.json
server_cancel.json
server_queue.json
server_conversation.json
server_verification.json
```

## 13. Stage 7.11.6: API and Frontend Data Contract Freeze

### 13.1 目标

前端开始工作前，必须冻结最小合同。前端不得读取 benchmark raw JSON 或猜测 Python 内部字段。

### 13.2 新增文档与 schema

```text
docs/api_contract.md
docs/openapi/sparseflow-openapi.json
docs/api_examples/health_loading.json
docs/api_examples/health_ready.json
docs/api_examples/runtime_ready.json
docs/api_examples/chat_non_stream.json
docs/api_examples/chat_stream.sse
docs/api_examples/error_memory_admission.json
docs/api_examples/error_context_length.json
tests/test_openapi_contract.py
```

### 13.3 Frontend 必需 endpoint

冻结：

```text
GET /health
GET /v1/models
GET /v1/models/{model_id}
GET /v1/runtime
POST /v1/chat/completions
POST /v1/generations/{request_id}/cancel
```

不新增前端私有 endpoint，除非现有合同无法表达必要状态。

### 13.4 `/health` 必需字段

```text
status
ready
model
preset
scheduler.active
scheduler.active_request_id
scheduler.queued
scheduler.max_queue
runtime.session_mode
runtime.persistent_kv_supported
```

### 13.5 `/v1/runtime` 必需字段

```text
schema_version
state
model_id
preset
public_status
effective_config.cache_bytes
effective_config.context_tokens
effective_config.max_completion_tokens
runtime_load_count
runtime_load_seconds
runtime_identity
model.metadata_sha256
container.metadata_sha256
container.weight_bytes
container.execution_bytes
process_memory.rss_bytes
process_memory.peak_rss_bytes
process_memory.private_bytes
last_generation
```

`last_generation` 只允许：

```text
request_id
prompt_tokens
completion_tokens
finish_reason
prefill_seconds
decode_seconds
decode_tok_per_second
generated_ids_hash
output_text_hash
route_fingerprint
memory
cache compact counters
provider read bytes
```

### 13.6 UI status semantics

合同必须定义：

| Runtime state | Frontend meaning |
|---|---|
| loading | 模型正在加载，Chat 暂不可提交 |
| ready | 可提交请求 |
| busy | 当前有请求，后续请求可能排队 |
| error | 启动或 runtime error，展示 server error |
| stopping | 正在关闭 |
| stopped | 已关闭 |

### 13.7 Evidence labels

API 和 example fixture 必须支持或文档化以下标签：

- `MEASURED`: 来自真实 SparseFlow runtime artifact。
- `SIMULATED`: UI demo fixture。
- `UNKNOWN`: 没有来源。
- `PASS`: 只有 verifier 通过时使用。
- `FAIL`: verifier 明确失败。
- `NOT_RUN`: 尚未执行。

前端不得把 `SIMULATED` 数据显示为 SparseFlow correctness PASS。

### 13.8 OpenAPI test

必须验证：

- OpenAPI JSON 可以解析。
- 所有冻结 endpoint 存在。
- Example response 满足 schema required fields。
- Error object 与 Server 实际响应一致。
- SSE contract 在文档中说明，不伪装成普通 JSON response。
- OpenAPI 不声明 sampling、tools、multimodal 或 persistent session 已支持。

## 14. Stage 7.11.7: Installation and Release Hardening

### 14.1 更新文件

```text
README.md
docs/installation.md
docs/stage7_11_laptop_runtime_plan.md
docs/results/qwen36_stage7_11_laptop_<date>.md
```

`.handoff.md` 只有在 Stage 正式验收后，由 owner 根据正式 artifact 更新。开发中的评价不得提前写入 handoff。

### 14.2 Windows 安装演练

在 clean environment 或删除后重建的 isolated environment 中执行：

```powershell
.\scripts\bootstrap_uv_windows.ps1
.\scripts\setup_windows.ps1
. .\scripts\use_runtime_windows.ps1
sparseflow preset laptop-16gb --json
sparseflow doctor $model --preset laptop-16gb --int8-container $int8 --check-native
sparseflow run $model --preset laptop-16gb --int8-container $int8 --prompt "Hello" --max-new-tokens 2
```

不得删除现有模型或 INT8 container 来演练安装。Python environment 可以重建，模型资产只做只读验证。

### 14.3 Setup script 验收

- Python 只安装到 `.tools/python`。
- `--no-bin --no-registry` 保留。
- venv 在 `.venv-runtime`。
- CPU Torch 正确。
- second run idempotent。
- 不重复创建 Windows launcher。
- 所有 cache/temp 指向项目盘。
- MSVC auto-discovery 成功。
- native build cache 可以删除后重新编译。

### 14.4 Doctor UX

Doctor 对 laptop preset 必须清楚输出：

- available RAM。
- required RAM。
- recommended RAM。
- headroom。
- cache budget。
- context tokens。
- model/container/native pass/fail。
- 当前是 pass、warn 还是 fail。

Memory warn 可以运行 smoke，但文档必须说明风险。Memory fail 时 Server 必须拒绝加载，不得提供 force flag 绕过。

### 14.5 README 声明边界

允许声明：

> SparseFlow has run Qwen3.6-35B-A3B on a 16 GiB Windows laptop using an INT8 streamed-expert path.

不得声明：

- 所有 16 GiB laptop 都支持。
- 达到实时聊天速度。
- GPU 不需要且永远无价值。
- 与 BF16 完全等价但没有引用 quality evidence。
- 32-token/long-context 已通过，除非 Stage 7.11 formal artifact 确实通过。

### 14.6 Known limitations

必须记录：

- AVX-512 VNNI required。
- 16 GiB 主机运行前需要释放内存。
- laptop preset 是 experimental。
- Transformers 当前使用 Torch fallback，而不是 optional DeltaNet fast path。
- 256 MiB cache 可能产生高 expert read amplification。
- Windows process-cold 不是物理 model-cold。
- 单请求串行。
- stateless full-message replay。
- no sampling/tools/vision/MTP。

## 15. Stage 7.11.8: Final Verification

### 15.1 必须存在的 artifact

```text
benchmarks/results/<date>/stage7_11_laptop/environment.json
benchmarks/results/<date>/stage7_11_laptop/doctor_matrix.json
benchmarks/results/<date>/stage7_11_laptop/cache_context_pilot.json
benchmarks/results/<date>/stage7_11_laptop/cli_laptop_matrix.json
benchmarks/results/<date>/stage7_11_laptop/cli_laptop_summary.json
benchmarks/results/<date>/stage7_11_laptop/server_health_identity.json
benchmarks/results/<date>/stage7_11_laptop/server_non_stream.json
benchmarks/results/<date>/stage7_11_laptop/server_persistent_requests.json
benchmarks/results/<date>/stage7_11_laptop/server_sse.json
benchmarks/results/<date>/stage7_11_laptop/server_cancel.json
benchmarks/results/<date>/stage7_11_laptop/server_queue.json
benchmarks/results/<date>/stage7_11_laptop/server_conversation.json
benchmarks/results/<date>/stage7_11_laptop/verification.json
docs/results/qwen36_stage7_11_laptop_<date>.md
```

### 15.2 Verification JSON

`verification.json` 必须包含独立 gate：

```text
environment_gate
telemetry_gate
doctor_gate
cli_correctness_gate
cli_performance_gate
server_identity_gate
server_non_stream_gate
server_persistence_gate
server_sse_gate
server_cancel_gate
server_queue_gate
conversation_gate
api_contract_gate
installation_gate
frontend_readiness_gate
overall_decision
```

每个 gate 只能是：

```text
GO
NO-GO
NOT-RUN
```

不得用布尔值掩盖未执行状态。

### 15.3 Overall GO 条件

以下全部为 GO 才能进入前端接入：

1. Telemetry 非零且可信。
2. Doctor laptop preset ready。
3. 18-cell CLI matrix 无 crash/OOM。
4. CLI repeat correctness exact。
5. Peak RSS <= 11.0 GiB。
6. 32-token cell 在初始性能门槛内，或 Board 明确接受 performance exception。
7. Server runtime load count 恒为 1。
8. Server non-stream 与 CLI exact。
9. SSE delta 拼接 exact。
10. Cancel 后 lease 归零，下一请求可用。
11. Queue FIFO 通过。
12. Multi-turn replay contract 通过。
13. OpenAPI/examples 与实际 Server 一致。
14. Installation replay 通过。
15. 所有 unit tests 通过。

### 15.4 Partial GO

如果 runtime correctness、Server 和 contract 通过，但 performance gate NO-GO：

- 可以进入内部前端集成；
- 不得发布“流畅聊天”宣传；
- UI 必须展示 experimental/laptop 状态；
- Stage 报告必须明确 performance limitation。

如果 correctness、memory safety、Server persistence 或 API contract 任一 NO-GO，则不得交给前端作为真实 runtime contract。

## 16. Frontend Handoff Package

Stage 7.11 完成后，交给前端团队的内容只能是：

```text
docs/api_contract.md
docs/openapi/sparseflow-openapi.json
docs/api_examples/*
docs/installation.md
docs/results/qwen36_stage7_11_laptop_<date>.md
verification.json
本地 server 启动命令
测试 API key/host/port 配置
```

前端团队不需要：

- import Python runtime。
- 读取 model/container 文件。
- 解析 benchmark raw artifact。
- 猜测 cache 或 memory 字段。
- 实现 persistent KV。
- 直接调用 Transformers。

前端的首个真实集成只能覆盖：

- Chat。
- Loading/Ready/Busy/Error 状态。
- SSE token delta。
- Cancel。
- Compact runtime/cache/memory metrics。
- Stateless multi-turn full-message replay。

Route Trace 和 Benchmarking 前端视图可以消费 compact evidence，但不能直接展示完整 route arrays 或把 simulated fixture 标记为 PASS。

## 17. 禁止事项

执行 Stage 7.11 时禁止：

1. 修改 INT8 quantization arithmetic。
2. 修改 native kernel 默认 dispatch。
3. 默认启用 pure fused decode。
4. 引入 INT4。
5. 引入 GPU tier。
6. 默认安装未验证的 DeltaNet fast-path dependency。
7. 用 pagefile 成功替代 physical memory admission。
8. 自动结束用户 Chrome、IDE 或其他进程。
9. 使用 `--available-ram` 伪造 formal Doctor pass。
10. 在 Windows 将 process-cold 称为 model-cold。
11. 提交 model、container、venv 或 cache。
12. 在结果中保存 API key。
13. 在 `/v1/runtime` 返回 prompt/history。
14. 把 HTTP 200 标记为 correctness PASS。
15. 在 Stage 结束前让前端依赖未冻结字段。

## 18. 执行者完成检查表

执行模型结束前逐项确认：

- [ ] 当前工作基于 clean commit。
- [ ] Windows production telemetry 非零。
- [ ] Telemetry unit tests 通过。
- [ ] Laptop runner 和 verifier 已实现。
- [ ] Prompt manifest UTF-8 正确。
- [ ] Cache/context pilot 已完成。
- [ ] `laptop-16gb` preset 有数据支持。
- [ ] `run --ctx` 和 Server context admission 一致。
- [ ] 18-cell CLI matrix 已完成。
- [ ] 真实 Server ready。
- [ ] Runtime load count 为 1。
- [ ] Non-stream correctness exact。
- [ ] Persistent warm requests 通过。
- [ ] SSE delta exact。
- [ ] Cancel cleanup 通过。
- [ ] Queue FIFO 通过。
- [ ] Multi-turn replay 通过。
- [ ] OpenAPI 与 example fixture 已冻结。
- [ ] Windows clean installation replay 通过。
- [ ] 完整 test suite 通过。
- [ ] Verification JSON 已生成。
- [ ] Stage report 明确 GO/NO-GO。
- [ ] `.handoff.md` 只在正式验收后同步。

## 19. Stage 完成定义

Stage 7.11 只有在以下状态下才算完成：

> 在 clean commit 上，Qwen3.6-35B-A3B 使用冻结的 laptop preset，在 16 GiB Windows laptop 上完成可复现 CLI matrix；真实 Server 在同一进程中只加载一次 runtime，并通过 non-stream、SSE、cancel、queue 和 stateless multi-turn；production telemetry、OpenAPI、安装说明和 frontend contract 均已冻结，最终 verification 明确允许前端接入。

如果只完成一次短 generation，不算 Stage 7.11 完成。

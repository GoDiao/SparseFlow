# SparseFlow Implementation Plan

SparseFlow 的第一阶段目标不是立刻手写完整推理器，而是先把 MoE 模型的资源结构、分层预算和 expert 访问边界做清楚。这样后续无论接 Qwen3.6、OLMoE、Mixtral，还是自研 runtime，都有一套稳定地基。

## Phase 1: Inspect / Plan

先做两个只读 CLI：

```bash
python -m sparseflow inspect model/Qwen3.6-35B-A3B
python -m sparseflow plan model/Qwen3.6-35B-A3B --ram 12 --ctx 4096
```

核心模块：

- `ShardIndex`
  - 只读取 safetensors header。
  - 建立 tensor name 到 shard、offset、size、dtype、shape 的索引。
  - 不读取 tensor payload。
- `TensorClassifier`
  - 将 tensor 分成 routed expert、shared expert、router、attention、embedding/lm head、vision、other dense。
  - Qwen3.6 先用 adapter 处理 fused expert：`mlp.experts.gate_up_proj` 和 `mlp.experts.down_proj`。
- `ModelFootprintAnalyzer`
  - 输出总权重、各类别权重、每层 routed expert 尺寸。
  - 输出每 token 冷读 expert 量的理论估算。
- `TierPlanner`
  - 根据 RAM/VRAM/磁盘和上下文长度给出计划。
  - 第一版只做 CPU/RAM 规划，VRAM 先保留字段。
  - 输出 versioned JSON，后续 CLI/API/UI 共用。

## Phase 2: Expert Locator / Lazy Loader

先不跑完整 forward，只验证从 safetensors 中定位和读取 expert。

目标 CLI：

```bash
python -m sparseflow expert-stat model/Qwen3.6-35B-A3B --layer 0 --expert 17
python -m sparseflow expert-load model/Qwen3.6-35B-A3B --layer 0 --expert 17
```

需要注意：Qwen3.6 的 expert 不是每个 expert 一个独立 tensor，而是每层 fused tensor 内按 expert 维度切片。`ExpertLocator` 必须理解 shape 和切片范围，不能照搬 colibri 的 GLM 文件布局。

## Phase 3: Cache / Hot Tier Simulation

在完整 runtime 前先模拟 MoE expert 访问。

目标：

- 统计 batch 内 unique experts 数量。
- 估算 per-layer LRU 在不同 cache size 下的命中率。
- 验证 pinned hot experts 的收益。
- 决定是否值得做 async prefetch、PIPE overlap、router lookahead。

第一版策略：

- per-layer LRU。
- RAM pinned tier。
- colibri 风格 heat decay + hysteresis swap。
- 暂不做 VRAM hot tier。

### Phase 3 status

基础阶段已经完成：ExpertCache、per-layer LRU、global byte budget、真实
route trace、batch-union 和 `expert-bench` 均已实现并有结果记录。Hot-tier
heat policy、predictive prefetch 和 VRAM tier 保留到后续优化阶段。

## Phase 4: Minimal Runtime Boundary

完整 Qwen3.6 包含 Gated DeltaNet、Gated Attention、Vision、MTP，早期不承诺完整生成。项目边界分成两层：

- SparseFlow core：shard index、model footprint、tier plan、expert locate、cache、prefetch。
- Model adapters：模型命名规则、shape、expert slice、router/dense 分类。

### Phase 4 status

单层 Qwen3.6 MoE 的 resident/cached-streaming runtime 已完成：
`CachedExpertProvider` 将正式 ExpertCache 接入 router、routed expert、
shared expert、shared gate 和 final output，并通过 synthetic 与真实
Qwen3.6 layer 0 的 9 项验收。结果见
`docs/results/qwen36_moe_cache_correctness_20260714.md`。

I/O coalescing、同层异步 prefetch 和多层 MoE-only runtime 已完成，结果见
`docs/results/qwen36_stage5_moe_runtime_20260714.md`。完整 Qwen3.6
attention、Gated DeltaNet、KV cache 和 generation 仍不属于本阶段。

这样 SparseFlow 不会被 Qwen3.6 的特殊结构绑死。

## Phase 5: I/O and MoE-only runtime

阶段 5 已完成：

- 批量 shard range coalescing；
- useful/physical/wasted byte 统计；
- bounded asynchronous prefetch 和 in-flight 去重；
- 两层及多层 MoE-only resident/streaming 对照；
- prefetch failure 清理、CLI 和真实 Qwen3.6 layer 0–1 结果。

结果：`docs/results/qwen36_stage5_moe_runtime_20260714.md`。

## Phase 6: Full Qwen3.6 text-only reference runtime

阶段 6 的目标是把已经验证过的 MoE storage path 接进完整的 Qwen3.6
文本模型流程，但先把 Python 版本定位为 correctness/reference runtime，
不把它误称为最终低内存或高性能推理器。

当前数据流是：

```text
tokenizer/chat template
  -> embedding
  -> attention / Gated DeltaNet
  -> router
  -> ExpertCache + ShardReader routed experts
  -> shared expert / residual / norm
  -> lm_head logits
  -> past_key_values KV cache
  -> one-token decode / greedy generation
```

### 阶段 6 的六个实现边界

1. **Expert matmul / SiLU**
   - resident 与 streaming 共用 `run_expert_kernel`，避免存储策略引入
     算术差异。
   - 参考路径使用 `linear(gate_up).chunk -> silu(gate) * up ->
     linear(down)`，再由 `run_routed_experts` 做 top-k 权重累加。
   - 后续 native kernel 才考虑 fused gate/up、SiLU、down，以及按
     batch-union 重排 token。

2. **BF16 / INT8 解码和矩阵乘**
   - 当前阶段先以 BF16 原始 slice 解码为正确性基线，验证 dtype、shape、
     expert axis 和 byte range 一致。
   - INT8/INT4 的量化格式、scale/zero-point、反量化融合和质量回归不在
     当前 smoke 的验收范围，必须在 BF16 reference 稳定后由 profiling
     和 Benchmark 共同定义。

3. **coalesced pread / async I/O**
   - 复用阶段 5 的持久 fd、`pread`、range coalescing、in-flight 去重和
     bounded worker pool。
   - 全模型 forward 中由当前层 router 产生 expert 请求，prefetch 在
     expert kernel 消费前提交同层批量读取；记录 useful/physical/wasted
     bytes、等待和失败。

4. **cache lookup / tensor view 生命周期**
   - `ExpertCache` 只保存模型无关的 raw bytes。
   - `CachedExpertProvider` 将 bytes 解码成 tensor view，并把 view 绑定到
     对应的 cache entry；entry 被 eviction 后禁止继续复用旧 view。
   - 同时支持 per-layer slots 和 global byte budget，所有 lookup、miss、
     eviction、loaded bytes 都必须进入结果 JSON。

5. **attention / DeltaNet**
   - 阶段 6 Python 参考实现直接复用 Transformers 官方 Qwen3.6 模块，
     覆盖 Gated Attention、Gated DeltaNet、norm、residual 和输出接口。
   - SparseFlow 暂不复制一套模型特化 attention kernel；只有在完整流程
     正确性和 profiling 后，才将热点下沉到 C/C++/Rust。

6. **KV cache 操作**
   - prefill 使用 `use_cache=True` 获取 Transformers `past_key_values`。
   - decode 每次只输入上一个 token，扩展 attention mask，并把更新后的
     cache 传回下一步。
   - 第一版验证 greedy token 回归和 cache 生命周期；paged KV、量化 KV、
     sliding-window/长上下文优化留到后续阶段。

### Phase 6 status

已实现 `Qwen36TextRuntime`、`SparseFlowQwenExperts`、`text-generate` 和
`text-check` CLI。`text-check` 顺序执行 resident/streaming，固定两侧为
相同的 eager expert 算术，并比较 input IDs、generated IDs、文本以及每个
prefill/decode 步骤的完整词表 logits SHA-256。

真实 Qwen3.6 已通过 4-token 验收：prefill 加连续三次 KV-cache decode 的
`248320` 维 BF16 logits 指纹逐步完全一致。带两个 prefetch workers 的
路径也通过同一门槛。Transformers 默认 `grouped_mm` 被保留为独立性能
backend，不与 eager streaming 混作 storage-policy 正确性比较。结果见
`docs/results/qwen36_stage6_text_runtime_20260714.md`。

当前关键限制是 `from_pretrained` 仍先由 Transformers 完整加载 checkpoint，
再替换 routed experts；因此这是完整流程的集成/正确性里程碑，还不是
Colibri 式低峰值内存加载器。

## Phase 7: Memory-native and performance runtime

阶段 7 再处理最终效果：meta-device/custom state-dict loader、dense 常驻、
expert 不全量 materialize、INT8/INT4 解码、fused expert matmul、平台专用
异步 I/O、hot expert tier、paged/quantized KV cache，以及 C++/Rust 下沉。
顺序必须由阶段 6 的 correctness 和 Benchmark profiling 驱动。

### Phase 7.1 status: memory-native loader complete

Qwen3.6 text-only memory-native 路径已完成：header-only load plan、meta
构建、加载前 expert 替换、逐 tensor resident materialization、强制 meta/
expert 审计、`--load-mode memory-native` 和完整 logits `text-check` 均已
实现。真实模型从未完整 materialize 60 GiB language experts，4-token
standalone 进程峰值 RSS 为 9.216 GiB，prefill 和三次 KV decode 与 eager
resident 的完整 BF16 logits 精确一致。结果见
`docs/results/qwen36_stage7_1_memory_native_20260715.md`。[Main Dev]

### Phase 7.2 status: C3-R / C3-S same-kernel runtime complete

已实现正式 `ExpertProvider` 边界、全量 RAM `ResidentExpertProvider` 和现有
ExpertCache streaming provider 的统一接口。memory-native C3-R/C3-S 现在
都安装相同的 `SparseFlowQwenExperts`，并共享相同的 routed dispatch、eager
BF16 expert kernel、attention/DeltaNet、KV cache 和 greedy loop；唯一变化是
expert 权重来自 resident fused buffer 或 SSD/ExpertCache。

真实 Qwen3.6 验收预载 40 层、10,240 个逻辑 expert、80 个 fused buffer，
总计恰好 60.000 GiB；C3-R 预载后的 generation expert I/O 为零。C3-R 与
C3-S 在 prefill 和三次 decode 的完整 logits、160 条 route digest、生成 IDs
和文本上全部精确一致。结果见
`docs/results/qwen36_stage7_2_c3_runtime_20260715.md`。[Main Dev]

下一步进入 Phase 7.3：在这个已冻结的同-kernel 边界上完善逐 token/forward
telemetry，处理 cache policy、hot tier、lookahead prefetch、重复读取和
I/O overlap。正式可复现性能矩阵仍留在 Phase 7.4。[Main Dev]

### Phase 7.3 status: telemetry, cache, and prefetch complete

已实现可插拔 S0-S4 cache/prefetch policy、global/per-layer budget、hot tier、
heat decay、hysteresis、prefill second-touch admission、逐 forward/layer
telemetry、current-route async prefetch、三次连续 route stability prediction、
bounded predictive budget、in-flight 去重和 generation-end accounting。

15 条真实 8/16/32-token traces、4 个 byte budgets、5 个 policy 的 300-run
metadata sweep 全部通过不变量。真实 Qwen3.6 8-token `policy-check` 中 S0-S4
均与 C3-R 保持完整 logits、320 条 route、IDs/text 和 runtime identity 精确
一致；4 GiB S3/S4 都把逻辑读取降到 15.79 GiB，S4 将 2693 个同步 miss 转为
prefetch-served 且没有重复读取。结果见
`docs/results/qwen36_stage7_3_cache_prefetch_20260715.md`。[Main Dev]

下一步进入 Phase 7.4：冻结线程数、workload、cold/warm page-cache 状态、
重复次数和统计口径，正式执行 C1/C2/C3-R/C3-S0..S4 Benchmark。Stage 7.3
的单次 wall time 仅作为开发观测，不作为正式性能结论。[Main Dev]

### Phase 7.4 status: formal BF16 Benchmark complete

已冻结 10 CPU threads、41-token 核心 prompt、32-token greedy decode、一次
warmup + 三次 measured、1/2/4/8 GiB budgets，以及 model-cold/workload-warm
状态。27-cell C3 矩阵运行约 2.41 小时；全部结果来自 clean worktree，并通过
同一模型 revision、同一 runtime/kernel identity、32-step 完整 logits、route、
generated IDs、cache budget、demand accounting 和 prefetch failure 门禁。

主要结果：C3-R 为 2.9618 tok/s、66.355 GiB RSS；当前 Python runtime 的最佳
warm 点是 C3-S1 LRU 8 GiB，为 0.8263 tok/s、13.651 GiB RSS、701.61 MiB
expert read/decode token。C3-S0 model-cold/workload-warm 分别为 0.3725 /
0.4752 tok/s。C3-S4 4 GiB cold 比 S3 快 1.17x，但复杂 policy/prefetch 在
warm Python 路径中仍不及基础 LRU。

C1 Transformers resident 为 2.9595 tok/s、67.074 GiB RSS。C2 Accelerate
generic offload 的两-token基线在 model-cold / workload-warm 下分别为
0.02567 / 0.02794 tok/s；C3-S1 8 GiB 比 warm C2 快 29.6x。三次 cold C2
的 TTFT 中位为 308.8 秒，单 decode 在 35.7–310.7 秒之间；C2 只运行两
token 是因为每个 generic forward 扫描完整 checkpoint。该限制在报告中
明确保留，未外推成伪造的 32-token 样本。

完整报告和 40 MiB raw evidence 位于
`benchmarks/results/2026-07-15/stage7_4/`。[Main Dev]

下一步进入 Phase 7.5：保留 Stage 7.4 的 workload/schema/correctness gate，
优先实现 INT8 expert storage/decode 与 AVX-512 VNNI native kernel，再重跑同一
resident/streaming 矩阵。不得把量化误差归因于 streaming。[Main Dev]

### Phase 7.5.0 status: observer-effect gate complete

Provider hot counters、policy diagnostics 和 telemetry 已拆分；summary 路径不再
排序 heat table、计算逐层 unique experts 或保留 layer records。真实 C3-R
配对实验中，summary 相对 none 的 decode 差异为 `-0.37%`，13 次完整
logits/IDs exact，详细关键路径闭合 `98.73%`。结果见
`docs/results/qwen36_stage7_5_0_observer_20260716.md`。[Main Dev]

## Early Non-goals

- 阶段 6 暂不支持 vision 输入、MTP/speculative decoding 和生产级 serving。
- 阶段 6 暂不承诺 INT8/INT4 质量、低峰值内存或正式吞吐结论。
- 阶段 7 之前不把 Python reference kernel 当作最终 native backend。
- 不默认下载模型或缓存到 C 盘。

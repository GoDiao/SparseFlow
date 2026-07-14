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

## Early Non-goals

- 阶段 6 暂不支持 vision 输入、MTP/speculative decoding 和生产级 serving。
- 阶段 6 暂不承诺 INT8/INT4 质量、低峰值内存或正式吞吐结论。
- 阶段 7 之前不把 Python reference kernel 当作最终 native backend。
- 不默认下载模型或缓存到 C 盘。

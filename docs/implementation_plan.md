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

## Phase 4: Minimal Runtime Boundary

完整 Qwen3.6 包含 Gated DeltaNet、Gated Attention、Vision、MTP，早期不承诺完整生成。项目边界分成两层：

- SparseFlow core：shard index、model footprint、tier plan、expert locate、cache、prefetch。
- Model adapters：模型命名规则、shape、expert slice、router/dense 分类。

这样 SparseFlow 不会被 Qwen3.6 的特殊结构绑死。

## Early Non-goals

- 不直接实现完整 Qwen3.6 forward。
- 不做 VRAM expert hot tier。
- 不做 MTP/speculative decoding。
- 不做 UI 直连 runtime。
- 不默认下载模型或缓存到 C 盘。

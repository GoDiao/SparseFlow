# Qwen3.6 单层 MoE resident/streaming 正确性

## 目的

验证 SparseFlow 的第一条完整 MoE block 闭环：

```text
router -> softmax + top-k -> routing weights
      -> routed expert weighted accumulation
      -> shared expert -> shared expert gate
      -> final MoE output
```

本次结果只覆盖 Qwen3.6 的单个 MoE 层，不代表完整 Transformer forward、Gated DeltaNet、视觉模块或生成服务已经接入。

## 实现范围

- `src/sparseflow/moe_probe.py`
  - `compare_moe_paths`：完整单层对照入口。
  - `run_moe_kernel`：resident/streaming 共用的 MoE 计算函数。
  - `_route_hidden_states`：复现 Qwen3.6 的 router softmax、top-k 和权重归一化。
  - `_qwen36_moe_spans`：定位 router、routed、shared 和 shared gate 权重。
- `src/sparseflow/cli.py`
  - 新增 `expert-moe-check`。
- `tests/test_moe_probe.py`
  - synthetic fused safetensors fixture。
  - 完整 MoE 对照和 CLI JSON 输出测试。

两条路径使用同一份 hidden states、同一套 Qwen3.6 算法和相同的 BF16 权重。差异只在 routed expert 权重的提供方式：

- resident：预加载当前层全部 256 个 routed experts。
- streaming：根据 router 结果，只读取本次 forward 中 unique selected experts，每个 expert 的两个 fused slice 只读一次。

## 真实模型配置

模型路径：`/root/workspace/SparceFlow/model/Qwen3.6-35B-A3B`

```text
layer                 0
rows                  2
hidden size           2048
num experts           256
top-k                 8
dtype                 BF16
```

本次 2-row forward 选择了 16 个 unique experts。完整 raw JSON 在 [`qwen36_moe_layer_correctness_20260714.json`](./qwen36_moe_layer_correctness_20260714.json)。

## 结果

### Correctness

| 对照项 | exact equal | max absolute error | max relative error |
|---|---:|---:|---:|
| selected experts | true | 0 | 0 |
| routing weights | true | 0 | 0 |
| router logits | true | 0 | 0 |
| routed output | true | 0 | 0 |
| shared output | true | 0 | 0 |
| final output | true | 0 | 0 |

### Transformers 独立对照

为避免 resident 与 streaming 两条自有路径共享同一个错误，另外把同一
layer 0 的权重挂载到 Transformers 官方
`Qwen3_5MoeSparseMoeBlock`，使用同一 seed、同一 hidden states 和同一
BF16 权重进行对照：

```text
exact_equal       true
max_abs_error     0
max_rel_error     0
```

原始记录在 [`qwen36_moe_transformers_reference_20260714.json`](./qwen36_moe_transformers_reference_20260714.json)。这个对照仍然只覆盖 MoE block，不覆盖完整 Qwen3.6 transformer 的 attention、linear attention 或 generation cache。[Main Dev]

### Storage policy

| 路径 | routed expert 数量 | read calls | routed bytes |
|---|---:|---:|---:|
| resident | 256 | 2 | 1,610,612,736（1.50 GiB） |
| streaming | 16 unique | 32 | 100,663,296（96 MiB） |

两条路径共用的 router、shared expert 和 shared gate 权重读取为 7,344,128 bytes。这个数字不包含在上表 routed bytes 中。

因此，在这个小 batch 上，streaming 路径把 routed expert 的读取量从 1.50 GiB 降到 96 MiB，约为 resident routed 读取量的 6.25%。这只是一次随机 hidden-state、2-row、单层 probe 的存储结果，不应直接外推为稳态生成吞吐。

## CLI

```bash
PYTHONPATH=src python -m sparseflow expert-moe-check \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --layer 0 --rows 2 --seed 1234 --json
```

## 边界与下一步

这次已经证明“完整单层 MoE 的 resident 与 streaming 权重路径可以得到完全一致的输出”。但当前实现仍是 correctness probe：

1. streaming 路径使用本次 forward 内的局部 unique-expert 字典，还没有接入长期 `ExpertCache`。
2. `ShardReader` 已支持持久 fd + `pread`，但同层 top-k 并发读取、`readv`/coalescing 和异步 prefetch 仍待实现。
3. 还没有把该 block 接到完整 Qwen3.6 attention/linear-attention 和 generation runtime。
4. 还需要增加不同 rows、不同 layer、冷页缓存条件的 correctness 和 I/O 测量。
5. 后续应把本 probe 接入正式 `ExpertCache`，并继续验证多层 runtime；当前 Transformers 独立 MoE block 对照已经完成。

本结果由 **Main Dev** 执行并记录。[Main Dev]

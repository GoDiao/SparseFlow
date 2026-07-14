# 阶段 5：I/O coalescing、异步 prefetch 与多层 MoE runtime

## 范围

本阶段实现并验证的是 **Qwen3.6 MoE-only runtime**：串联多个 MoE block，验证 expert 存储、cache、批量 I/O 和 prefetch 的组合行为。

它不包含完整 Qwen3.6 Transformer 的 attention、Gated DeltaNet、residual、KV cache、embedding、lm head 或 generation。

## 实现内容

### I/O coalescing

[`src/sparseflow/loader.py`](../../src/sparseflow/loader.py) 新增：

- `coalesce_locations()`：按 shard 和 file offset 合并多个 expert slice；
- `ShardReader.read_locations()`：一次 physical read 后拆分回各个 `(layer, expert, part)`；
- `BatchReadStats`：记录 logical ranges、physical ranges、useful bytes、physical bytes、wasted bytes。

`max_gap=0` 只合并连续范围；较大的 gap 可以减少 syscall，但会读取未使用的空洞，指标会明确暴露这个代价。

### 异步 prefetch

[`CachedExpertProvider`](../../src/sparseflow/moe_probe.py) 新增：

- bounded `ThreadPoolExecutor`；
- in-flight expert 去重；
- 一个 batch 的 coalesced read；
- `get()` 等待对应 Future 并一次性提交 batch payloads 到 cache；
- prefetch failure、wait、read time、useful/physical bytes 统计。

`run_moe_kernel()` 支持 `prepare_routed` callback：router 得到 unique experts 后先提交 prefetch，再计算 shared expert，随后等待 routed expert payload，从而形成同层 I/O/计算重叠的执行边界。

### 多层 MoE runtime

[`src/sparseflow/moe_runtime.py`](../../src/sparseflow/moe_runtime.py) 新增：

- `Qwen36MoeOnlyRuntime`：逐层执行 Qwen3.6 MoE block；
- `compare_multilayer_moe_paths()`：resident 与 streaming/prefetch 多层对照；
- 全局 cache 仍按 `(layer, expert_id)` 隔离；
- 每层保留 selected experts、routing、storage 和 cache 统计。

CLI：

```bash
PYTHONPATH=src python -m sparseflow moe-multi-check \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --layers 0-1 --rows 2 --cache-slots 16 \
  --prefetch-workers 2 --coalesce-gap 0 --seed 1234 --json
```

## 真实 Qwen3.6 layer 0–1 结果

硬件/模型：实验机、Qwen3.6 BF16、2 rows、layers `[0, 1]`、16 slots/layer。完整汇总见 [`qwen36_stage5_moe_runtime_20260714.json`](./qwen36_stage5_moe_runtime_20260714.json)。

### 无 prefetch，gap=0

```text
resident read calls       14
resident read bytes       3,235,913,728
stream routed read calls  64
stream routed read bytes  201,326,592
cache requests            32
cache hits/misses         0 / 32
correctness               exact equal
```

### prefetch，gap=0

```text
prefetch batches           2
prefetch submitted        32 experts
prefetch completed         2
prefetch waits             2
logical ranges            64
physical ranges           62
useful bytes               201,326,592
physical bytes             201,326,592
wasted bytes               0
stream routed read calls  62
cache hits/misses         30 / 2
correctness               exact equal
```

相对于无 prefetch，`gap=0` 的 coalescing 将 routed read calls 从 64 降到 62，同时没有增加 physical bytes。由于 page cache 未清空，wall time 只作为功能样本，不作为性能结论。

### prefetch，gap=6 MiB

```text
logical ranges            64
physical ranges           58
useful bytes               201,326,592
physical bytes             222,298,112
wasted bytes                20,971,520
stream routed read calls  58
cache hits/misses         30 / 2
correctness               exact equal
```

这组实验说明 coalescing 存在明确 trade-off：read calls 减少 4 次，但额外读取约 20 MiB 未使用数据。`max_gap` 不能盲目增大，后续应结合 SSD 带宽和 syscall latency 做选择。

### Prefetch 时间指标

`gap=0` 运行记录了：

```text
prefetch read time total   132.64 ms
prefetch wait time total    267.25 ms
prefetch failures           0
```

这些数据是单次、page-cache 未控制的功能样本；正式性能 benchmark 需要单独控制 cold/warm cache、重复次数和线程数。

## 验证结果

- synthetic batch coalescing：4 logical ranges 合并为 1 physical range，useful=physical=8 bytes；
- synthetic 两层 runtime：resident/prefetch exact equal，跨 layer cache key 隔离；
- synthetic prefetch：in-flight batch 只读取一次，后续 expert 走 cache hit；
- 真实 Qwen3.6 layer 0–1：无 prefetch、gap=0、gap=6 MiB 三组均 exact equal；
- 真实多层 cache invariant：`hits + misses == requests`、`loaded bytes == useful bytes`、`physical >= useful`、cache budget valid；
- 全部测试：20 个通过，包含 prefetch failure 清理测试。

## 阶段 5 边界

阶段 5 的 MoE-only runtime 已完成。下一阶段才接入完整 Qwen3.6 Transformer；prefetch 预测策略、hot expert policy、readv/平台专用异步 I/O 仍可在后续性能阶段继续优化。

本结果由 **Main Dev** 执行并记录。[Main Dev]

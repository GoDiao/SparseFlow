# Qwen3.6 ExpertCache 接入单层 MoE 验证

## 目标

把已经通过 resident/streaming correctness 的单层 MoE block 接入正式
`ExpertCache`，验证：

```text
router
→ 当前 forward 内按 layer/expert 去重
→ ExpertCache lookup
→ miss 时 ShardReader 读取 expert
→ cache entry 解码为 tensor view
→ routed weighted accumulation
→ shared expert + shared gate
→ final output
```

## 实现

- [`src/sparseflow/moe_probe.py`](../../src/sparseflow/moe_probe.py)
  - `CachedExpertProvider`：连接 `ExpertCache`、`ExpertLocator`、`ShardReader` 和 BF16 tensor 解码。
  - `compare_moe_cache_paths`：重复 forward 的 resident/cached-streaming 对照。
  - cache hit 时复用同一个 `CachedExpert` backing buffer 的 tensor view。
  - cache eviction 后清理 decoded view，避免被淘汰 expert 留在内存中。
- [`src/sparseflow/cache.py`](../../src/sparseflow/cache.py)
  - `CachedExpert` 支持 bytearray backing payload。
  - 增加不改变 LRU 顺序和统计的 `cached_keys()`。
- [`src/sparseflow/cli.py`](../../src/sparseflow/cli.py)
  - 新增 `expert-moe-cache-check`。
- [`tests/test_moe_probe.py`](../../tests/test_moe_probe.py)
  - synthetic hit/miss、eviction、global byte budget、跨 layer cache 和 CLI 测试。

## 统计口径

`raw route requests` 是 token×top-k 的原始请求数；`cache requests` 是当前
forward 内按 `(layer, expert)` 去重后真正调用 `ExpertCache` 的次数。

因此，8 行、top-8 的 forward 最多有 64 个 raw route requests，但实际可能
只有 59 个 unique expert requests。cache 统计使用后者，避免把 batch 内
重复 expert 错算成磁盘读取。

## 真实 Qwen3.6 结果

模型：`/root/workspace/SparceFlow/model/Qwen3.6-35B-A3B`，layer 0，BF16。
完整汇总 JSON 在
[`qwen36_moe_cache_correctness_20260714.json`](./qwen36_moe_cache_correctness_20260714.json)。

### 重复 forward、batch 去重和命中

配置：8 rows、4 forwards、每个 hidden batch 重复 2 次、64 slots/layer。

| 指标 | 结果 |
|---|---:|
| unique experts/forward | 59, 59, 54, 54 |
| cache requests | 226 |
| hits / misses | 123 / 103 |
| hit rate | 54.42% |
| evictions | 39 |
| loaded/read bytes | 648,019,968 |
| cached bytes | 402,653,184（384 MiB） |
| final correctness | exact equal |

重复 forward 的命中情况：

- forward 0：59 misses，读取 371,195,904 bytes；
- forward 1：59 hits，读取 0 bytes；
- forward 2：54 unique，其中 10 hits、44 misses；
- forward 3：54 hits，读取 0 bytes。

第一 forward 的 64 个 raw route requests 被去重为 59 个 expert requests，
实际读取 118 个 tensor parts。这验证了多行 token 对同一 expert 的复用。

### Per-layer LRU 驱逐

配置：2 rows、2 forwards、4 slots/layer。

```text
requests       30
hits            0
misses         30
evictions      26
cached bytes   25,165,824 (24 MiB)
loaded bytes  188,743,680
reader bytes  188,743,680
```

两次 forward 的 final output 都与 resident 路径 exact equal。

### Global byte budget

配置：2 rows、2 forwards、`max_bytes=6 MiB`。Qwen3.6 单个 expert 正好是
6 MiB。

```text
requests       30
hits            0
misses         30
evictions      29
cached bytes    6,291,456 (6 MiB)
loaded bytes  188,743,680
reader bytes  188,743,680
budget valid   true
correctness    exact equal
```

## 9 项验收条件

| # | 条件 | 证据 | 状态 |
|---:|---|---|---|
| 1 | streaming 不再使用临时 expert 字典 | `CachedExpertProvider` 持久连接 `ExpertCache` | PASS |
| 2 | streaming 实际调用 `ExpertCache` | 真实结果含 requests/hits/misses/loaded bytes | PASS |
| 3 | per-layer LRU 有真实命中/驱逐 | 4 slots/layer 结果：26 次 eviction；跨 layer synthetic 测试 | PASS |
| 4 | global byte budget 限制实际缓存大小 | 6 MiB 结果：cached bytes=6,291,456 | PASS |
| 5 | 多行 token 对同一 expert 只读取一次 | 8 rows：64 raw requests → 59 unique，118 part reads | PASS |
| 6 | 多次 forward 产生真实 cache hit | 重复 forward：59 hits、54 hits | PASS |
| 7 | cache correctness 与 resident 一致 | synthetic 和三组真实 Qwen3.6 结果全部 exact equal | PASS |
| 8 | 统计不变量一致 | 每组 `hits+misses=requests`、`loaded_bytes=reader_bytes`、budget valid | PASS |
| 9 | 结果与交接记录落盘 | 本文档、JSON 和 `.handoff.md` | PASS |

## CLI

```bash
PYTHONPATH=src python -m sparseflow expert-moe-cache-check \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --layer 0 --rows 8 --forwards 4 --repeats 2 \
  --cache-slots 64 --seed 1234 --json
```

全局 byte budget：

```bash
PYTHONPATH=src python -m sparseflow expert-moe-cache-check \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --layer 0 --rows 2 --forwards 2 --repeats 1 \
  --cache-bytes 6MiB --seed 1234 --json
```

## 当前边界

阶段 4 已完成的是单层 MoE 的正式 cache runtime。它还没有接入完整
Qwen3.6 attention、Gated DeltaNet、KV cache 或 generation；prefetch、
readv/coalescing 和多层 runtime 属于后续阶段。

本结果由 **Main Dev** 执行并记录。[Main Dev]

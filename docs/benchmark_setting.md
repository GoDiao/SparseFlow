# SparseFlow CPU Benchmark & Eval Setting

## 1. 文档目的

本文定义 SparseFlow 在 Qwen/Qwen3.6-35B-A3B 上的 CPU 实验矩阵、质量评测方法、运行规范和结果格式。

当前 Benchmark 轨道只负责：

- 设计可复现的性能实验；
- 建立 Transformers CPU baseline；
- 移植和扩展 Colibri 风格的质量 eval；
- 运行 eval、保存原始结果并形成对比报告；
- 向 Main Dev 提供 runtime 必须暴露的指标接口。

ExpertLocator、CPU runtime、expert cache 和 prefetch 的实现属于 Main Dev 轨道，不属于 Benchmark 轨道。

## 2. 核心实验问题

CPU 实验需要回答以下问题：

1. 完整模型全部驻留 RAM 时，CPU 推理的性能和内存基线是多少？
2. 在 RAM 不足时，普通 tensor/layer disk offload 的代价是多少？
3. SparseFlow 的 expert-aware streaming 是否能在更低 RAM 下运行，并优于普通 offload？
4. expert cache 从冷到热时，hit rate、磁盘读取量和 decode 速度如何变化？
5. streaming、cache、prefetch 和量化是否保持模型的数值正确性与任务质量？

主要假设：

- H1 — Feasibility：Full Resident 在低 RAM budget 下失败，而 SparseFlow 仍可运行。
- H2 — Expert awareness：相同 RAM budget 下，SparseFlow 比普通 offload 读取更少数据并获得更高吞吐。
- H3 — Warm convergence：随着 cache 增大和热度学习，SparseFlow 性能逐步接近同 runtime 的 resident 模式。
- H4 — Quality preservation：同精度下，streaming 不应改变 logits、greedy token 或标准任务答案。
- H5 — Quantization isolation：量化损失必须和 streaming 损失分开测量，不能混为一个结论。

## 3. 当前冻结事实

### 3.1 模型

~~~text
public model name      Qwen/Qwen3.6-35B-A3B
internal model_type    qwen3_5_moe
transformers class     Qwen3_5MoeForConditionalGeneration
weight dtype           BF16
shards                 26
tensors                1045
weight bytes           71,903,776,776
weight size            66.97 GiB
config SHA256          93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99
index SHA256           41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83
layers                 40
experts/layer          256
active routed experts  top-8
~~~

资源分析：

~~~text
routed experts                 60.00 GiB
dense resident estimate         6.97 GiB
typical routed expert           6.00 MiB
all experts in one layer        1.50 GiB
one cache slot per layer      240.00 MiB
theoretical cold read/token     1.88 GiB
~~~

“Qwen3.6”是公开发布名，Transformers 内部复用 qwen3_5_moe 架构标识。所有结果必须同时记录这两个名称。

### 3.2 当前实验机

~~~text
CPU          Intel Xeon Gold 6248R @ 3.00GHz
CPU topology 10 physical cores / 20 logical CPUs / 1 NUMA node
ISA          AVX2, AVX-512, AVX-512 VNNI
RAM          125 GiB physical, no swap
GPU          2 x Tesla V100S 32GB（CPU 正式实验不使用）
model path   /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B
filesystem   275.49 GiB total / 196.28 GiB free at observation time
~~~

当前 GPU+CPU-offload smoke test 只证明模型可生成，不属于正式 CPU baseline：

~~~text
input tokens       18
generated tokens    8
load time          21.198 s
generation time    18.475 s
decode throughput   0.433 tok/s
~~~

### 3.3 当前软件环境

~~~text
PyTorch       2.9.1+cu128
Transformers  5.14.0.dev0
source commit 11ed2ff4df5fdfb3117f0e3365ef6ad94081ba69
Accelerate    1.14.0
Safetensors   0.8.0
~~~

正式结果必须记录实际运行时版本；升级依赖后不得与旧结果直接混合。

## 4. CPU 实验组

### 4.1 主实验组

| ID | 模式 | 权重放置 | 目的 |
|---|---|---|---|
| C1 | CPU Full Resident | 全部权重常驻 RAM | 标准 CPU 容量与性能基线 |
| C2 | CPU Generic Offload | 普通 tensor/layer 级磁盘 offload | 现有通用低内存方案基线 |
| C3 | SparseFlow CPU | dense/router/shared 常驻，routed experts 按需读取和缓存 | 主要被测方案 |

### 4.2 内部归因对照

| ID | 模式 | 用途 |
|---|---|---|
| C3-R | SparseFlow Resident | 与 C3 使用相同 CPU kernel/runtime，但所有 experts 常驻 RAM |
| C3-S0 | Streaming only | 无 expert cache，测纯 streaming 下限 |
| C3-S1 | Per-layer LRU | 只启用基础 LRU |
| C3-S2 | LRU + hot expert | 加入长期热门 expert tier |
| C3-S3 | Heat decay + hysteresis | 加入热度衰减和防抖替换 |
| C3-S4 | S3 + prefetch | 加入 async/router-lookahead prefetch |

核心归因关系：

~~~text
C1 vs C2       全量 RAM 与普通 offload
C2 vs C3       普通 offload 与 expert-aware offload
C3-R vs C3     单独衡量 storage/cache 策略，不混入 kernel 差异
C3-S0..S4      衡量每项优化的独立贡献
~~~

若 C3 和 C1 使用不同计算 kernel，只能把 C1 vs C3 描述为“系统级对比”；算法归因必须使用 C3-R。

## 5. 精度矩阵

### 5.1 第一阶段：BF16 正确性矩阵

~~~text
C1 BF16 resident
C2 BF16 generic offload
C3-R BF16 resident
C3 BF16 streaming
~~~

目标是隔离存储策略。当前 Cascade Lake CPU 没有原生 BF16 指令，因此该阶段的绝对速度不是最终 CPU 性能结论。

### 5.2 第二阶段：实用量化矩阵

~~~text
INT8 resident   vs INT8 streaming
INT4 resident   vs INT4 streaming
BF16 resident   vs INT8 resident
BF16 resident   vs INT4 resident
~~~

当前 CPU 支持 AVX-512 VNNI，INT8 是优先实用路线。任何量化性能结果必须配套同量化 resident 对照。

禁止直接把 BF16 resident 与 INT4 streaming 的差异全部归因于 SparseFlow。

## 6. 性能 workload

### 6.1 CPU 线程校准

在正式实验前，以同一个 32-token prompt 和 8-token decode 做线程 sweep：

~~~text
1 / 5 / 10 / 20 threads
~~~

记录 prefill、decode、CPU 利用率和上下文切换。选出最佳线程数后，在同一批正式结果中固定不变。

### 6.2 Prefill workload

| 输入长度 | 输出长度 | Batch | 用途 |
|---:|---:|---:|---|
| 32 | 1 | 1 | 极短输入 |
| 128 | 1 | 1 | 短输入 |
| 512 | 1 | 1 | 中输入 |
| 1024 | 1 | 1 | 长输入 |
| 4096 | 1 | 1 | 长上下文压力测试 |

记录：prefill latency、prefill tok/s、TTFT、RSS、page faults、disk bytes。

### 6.3 Decode workload

固定约 32-token prompt：

| 输出长度 | Batch | 用途 |
|---:|---:|---|
| 8 | 1 | 快速开发回归 |
| 32 | 1 | 标准 decode |
| 64 | 1 | 稳态 decode |

所有正式 decode 使用 greedy：

~~~text
do_sample=false
temperature disabled
top_p disabled
fixed chat template
~~~

### 6.4 Route locality workload

建立两份固定 prompt manifest：

- locality：连续对话、相同主题、同一代码库续写，预期 expert 重用较高；
- diversity：中文、英文、数学、代码、知识问答交错，预期路由更分散。

按 token 记录：cache hit、unique experts、disk bytes/token、decode latency。结果必须展示从冷到热的时间序列，不能只报告最终平均值。

### 6.5 批量实验

主结论固定 batch=1，匹配本项目的低资源交互式目标。完成主实验后再增加 batch=2/4，研究 batch-union expert 读取；批量结果不与 batch=1 混合。

## 7. RAM 与 expert cache sweep

### 7.1 RAM budget

~~~text
16 / 24 / 32 / 48 / 64 / 96 / 120 GiB
~~~

C1/C2/C3 应在相同预算点运行或明确记录 OOM / unsupported。不能只展示 SparseFlow 能运行的点。

### 7.2 Cache capacity

Qwen3.6 每层一个 expert slot 的全模型成本约 240MiB：

| Slots/layer | Expert cache 总量 | 解释 |
|---:|---:|---|
| 0 | 0 | 纯 streaming |
| 8 | 1.88 GiB | 约等于单 token 的 top-8 活跃集 |
| 16 | 3.75 GiB | 小 cache |
| 32 | 7.50 GiB | 中 cache |
| 64 | 15.00 GiB | 大 cache |
| 128 | 30.00 GiB | 半量 experts |
| 256 | 60.00 GiB | routed experts 全驻留 |

每个点记录：peak RSS、hit rate、byte-weighted hit rate、evictions、disk bytes/token、tok/s。

主要结果应画成 Pareto 曲线：

~~~text
RAM / cache size -> decode tok/s
RAM / cache size -> disk bytes/token
RAM / cache size -> cache hit rate
~~~

SparseFlow 的成功不要求 cold streaming 比全驻留更快；主要目标是以远低于完整模型体积的 RAM 获得可接受速度，并在 cache 变暖后逼近 resident 性能。

## 8. I/O 微基准

借鉴 Colibri iobench，但使用 Qwen3.6 的实际 expert 粒度：

~~~text
random block size  6 MiB
read count         256
threads            1 / 2 / 4 / 8
mode               buffered / O_DIRECT
~~~

额外测试相邻 gate_up_proj 与 down_proj slice 的组合读取，以模拟一个完整 expert。

必须区分：

- engine/app cache cold；
- engine/app cache warm；
- OS page cache warm；
- O_DIRECT 或 per-file eviction 后的 disk-cold。

当前机器近期下载和加载过模型，Linux page cache 很可能包含大量权重。未控制 page cache 的读速不能写成 SSD 带宽。

## 9. Colibri 风格质量 eval

### 9.1 数据格式

复用 Colibri 的离线 JSONL：

~~~json
{"ctx": "question/context", "choices": [" answer A", " answer B"], "gold": 0}
~~~

默认任务：

~~~text
HellaSwag
ARC-Challenge
MMLU
~~~

扩展任务：

~~~text
ARC-Easy
WinoGrande
PIQA
OpenBookQA
~~~

所有抽样固定 seed=1234，并保存题目 ID/顺序 manifest，避免不同 backend 实际测到不同题目。

### 9.2 Log-likelihood 评分

对每个选项计算：

~~~text
LL(choice) = sum(log p(continuation_token_i | context, previous_tokens))
~~~

输出三项指标：

- acc：总 log-likelihood 最大的选项；
- acc_norm_char：按字符数归一化，兼容 Colibri；
- acc_norm_token：按 continuation token 数归一化，作为更标准的补充。

同时保存每道题、每个选项的原始 log-likelihood，不能只保存最终准确率。

### 9.3 Backend 接口

质量 harness 只依赖统一接口：

~~~text
score_continuations(context_tokens, continuation_token_lists)
    -> per-choice total_loglikelihood
    -> per-choice token_loglikelihoods
    -> runtime/resource metrics
~~~

需要接入：

~~~text
transformers_cpu_resident
transformers_cpu_generic_offload
sparseflow_cpu_resident
sparseflow_cpu_streaming
~~~

### 9.4 Eval 规模

| 阶段 | 样本量 | 用途 |
|---|---:|---|
| Smoke | 每项 1 题，共 3 题 | 检查 tokenization、边界与评分链路 |
| Pilot | 每项 3 题，共 9 题 | 比较 backend、线程数和 cache 状态，快速发现明显问题 |
| Development | 每项 5 题，共 15 题 | 开发期回归与消融实验，控制单轮实验耗时 |
| Formal | 每项 20 题，共 60 题 | 阶段性质量结论与置信区间；仅在实现稳定后运行 |

所有阶段都必须冻结题目 ID、顺序和 `seed=1234`，不能因为结果不理想而临时换题。当前不直接复刻 Colibri 的 n=40：对于我们的 CPU 级开发实验，15 题 development 集已经足够做方向筛选，60 题 formal 集再提供更稳定的阶段性结论。若后续需要与 Colibri 的 n=40 结果做严格横向对照，再单独增加 compatibility run，而不改变默认开发矩阵。

Colibri 报告的 n=40、三项 mean acc_norm=62.5% 只是一条初步数据。Qwen3.6 默认 thinking，0-shot multiple-choice log-likelihood 不适合作为模型绝对能力结论，但非常适合做同模型、同题目、不同 engine 的质量回归。

本轮质量评测采用上述分层规模，由 Benchmark 负责冻结 manifest、执行评分并保存原始结果。[Benchmark]

### 9.5 Stage 7.5 冻结质量协议

Stage 7.5.6 已冻结以下嵌套 manifest：[Main Dev]

```text
quality_smoke_v1.jsonl        3 rows   sha256 906a574d...
quality_pilot_v1.jsonl        9 rows   sha256 e62f3670...
quality_development_v1.jsonl 15 rows   sha256 de7fbe24...
quality_formal_v1.jsonl      60 rows   sha256 2beca7b5...
```

来源 revision、抽样 row、完整 hash 位于
`benchmarks/manifests/quality_manifest_v1.meta.json`。每项任务使用独立
`random.Random(f"1234:{task}")` 无放回抽样，较小阶段严格是 Formal 的
per-task 前缀。[Main Dev]

正式 scoring 使用 `choice_execution=batch`：同一问题的所有 choice 在一个
padded batch forward 中执行，以共享 batch-union expert I/O。Sequential 与
batch 的标准 Smoke 保持预测一致，但不得混用两者的原始 log-likelihood。
Pilot/Development 指标从同一 Formal raw result 的冻结前缀派生，不重复执行
相同题目。[Main Dev]

W8A16 reference 的标准 Smoke 保留；Formal task scoring 未继续，因为 Python
reference dequantization 成为 dominant observer。其数值归因使用独立的
teacher-forced quality gate，性能归因使用完整 Stage 7.5.6 matrix，不补写或
推测缺失的 Formal accuracy。[Main Dev]

## 10. 数值与生成正确性

### 10.1 Tiny architecture oracle

借鉴 Colibri 的 tiny-random oracle：建立保留真实 Qwen MoE 数据流的小模型，用 Transformers 产生 reference：

~~~text
teacher forcing predictions
greedy generated token IDs
selected logits
~~~

它用于快速验证 runtime plumbing，不衡量语言质量。

### 10.2 真实模型 teacher forcing

从固定 prompt manifest 抽取短序列，比较 C1、C3-R、C3：

~~~text
top-1 agreement
top-5 overlap
max/mean absolute logit error
KL divergence
baseline top1-top2 margin at mismatches
~~~

同 runtime、同精度的 C3-R 与 C3 只改变权重来源，原则上应得到相同结果。若出现差异，应先作为 correctness bug 调查。

### 10.3 Greedy generation regression

~~~text
20 fixed prompts
max_new_tokens=32
do_sample=false
~~~

记录完整 token IDs、完全一致率、首个分叉位置和分叉处 logits margin。不能只比较解码后的字符串。

## 11. 指标定义

### 11.1 通用性能指标

~~~text
model_load_seconds
input_tokens
generated_tokens
time_to_first_token_seconds
prefill_seconds
prefill_tokens_per_second
decode_seconds
decode_tokens_per_second
per_token_latency_p50/p95
end_to_end_seconds
peak_rss_bytes
steady_rss_bytes
minor/major_page_faults
cpu_user/system_seconds
~~~

### 11.2 I/O 与 cache 指标

~~~text
process_read_bytes
process_read_syscalls
disk_read_seconds
disk_bytes_per_token
cache_requests
cache_hits/misses
cache_hit_rate
byte_weighted_hit_rate
cache_evictions
prefetch_requests
prefetch_hits
prefetch_wasted_bytes
unique_experts_per_token
expert_reuse_distance
~~~

性能计时必须把 model load、prefill 和 decode 分开。Colibri 风格的单一 tok/s 不能替代这些拆分指标。

对于 CPU generation runner，首个 prompt forward 同时产生第一个新 token，计入
`prefill_seconds` 和 `time_to_first_token_seconds`；只有后续单 token forward
计入 `decode_seconds` 和 `decode_tokens_per_second`。`end_to_end_seconds` 才是
从 prompt forward 开始到全部生成 token 完成的总时间，不能将其标记为纯 decode。

Choice scoring 的 `load.seconds` 是 tokenizer、model load 和 materialization
区间的一次性总和；`materialize_seconds` 已经包含在其中，禁止重复相加。

## 12. 运行规范

每个正式 workload：

~~~text
warmup runs       1
measured runs     3
reported value    median，并保留全部 raw runs
batch             1
sampling          disabled
prompt order      fixed by manifest
~~~

正式运行前记录：

~~~text
git commit / dirty status
model config SHA256
model index SHA256
PyTorch/Transformers/runtime version
CPU/RAM/NUMA/filesystem/device information
thread and affinity settings
RAM/cache budget
precision and quantization metadata
~~~

同一对比表内不得混用不同软件版本、prompt manifest 或模型 revision。

## 13. 原始结果格式

每个 run 保存一条 JSON，至少包含：

~~~json
{
  "schema_version": 2,
  "experiment_id": "C3-cache32-decode32",
  "backend": "sparseflow_cpu_streaming",
  "model": {
    "public_name": "Qwen/Qwen3.6-35B-A3B",
    "model_type": "qwen3_5_moe",
    "config_sha256": "..."
  },
  "precision": "bf16",
  "ram_budget_gib": 32,
  "cache_slots_per_layer": 32,
  "workload": {
    "manifest_sha256": "...",
    "input_tokens": 32,
    "max_new_tokens": 32,
    "batch_size": 1
  },
  "timing": {},
  "memory": {},
  "io": {},
  "cache": {},
  "quality": {},
  "output_token_ids": []
}
~~~

建议目录：

~~~text
benchmarks/
  manifests/
  data/                    # large datasets remain outside Git
  results/<date>/<experiment-id>/  # raw results are tracked in Git
  reports/
~~~

模型、HF/ModelScope cache、offload 文件和虚拟环境继续保持 Git ignored。

## 14. 结果展示

主表：

| Mode | RAM | Load | TTFT | Prefill tok/s | Decode tok/s | Disk GiB/token | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 Resident | | | | | | 0 | |
| C2 Generic Offload | | | | | | | |
| C3 Streaming Cold | | | | | | | |
| C3 Streaming Warm | | | | | | | |

必须生成的曲线：

1. RAM budget -> decode tok/s；
2. cache slots -> hit rate 与 disk bytes/token；
3. token index -> cache warm-up 与 latency；
4. resident/offload/streaming -> accuracy 与 log-likelihood delta；
5. BF16/INT8/INT4 -> 性能和质量 Pareto frontier。

质量结果应报告置信区间；Smoke、Pilot 和 Development 结果只用于工程决策，必须标记为 preliminary，不得宣传为模型能力结论。Formal 的每项 20 题仍属于小规模阶段性评估；只有在项目需要公开能力结论时，才扩展到更大的标准数据集。

## 15. 判定原则

### Correctness gate

- C3-R 与 C3 同精度、同 kernel 时应保持 token/logit 一致；
- 任何无法解释的 mismatch 在修复前不得进入性能 headline；
- 量化 mismatch 必须单独归因，不得写成 streaming 损失。

### Performance claim

- SparseFlow 的主要对手是在相同 RAM budget 下的 C2 Generic Offload；
- C3 应报告 cold 和 warm 两种状态；
- 不能用 OS page-cache warm 结果冒充 disk-cold；
- 不能只挑选最快一次运行。

### Project success claim

项目成功的核心表述应是：

> 在保持模型数值行为和任务质量的前提下，SparseFlow 使 Qwen3.6-35B-A3B 在远低于完整 routed-expert 体积的 RAM budget 中运行，并通过 expert-aware cache/prefetch 显著减少相对普通 offload 的磁盘读取量和推理延迟。

## 16. 分阶段执行顺序

### Stage A — CPU baseline establishment

1. 完成 C1 CPU Full Resident 8-token smoke；
2. 进行 1/5/10/20 线程校准；
3. 跑短 prefill/decode development workload；
4. 固化 prompt manifest 和 raw result schema。

### Stage B — Colibri eval adaptation

1. 完成 HellaSwag、ARC-Challenge、MMLU 各 1 题的 offline smoke，共 3 题；
2. 接入 Transformers CPU continuation scoring；
3. 下载并冻结 HellaSwag/ARC/MMLU 数据 manifest；
4. 先跑每项 3 题的 Pilot，共 9 题；
5. 冻结后跑每项 5 题的 Development，共 15 题；
6. CPU baseline 和 SparseFlow 实现稳定后，再跑每项 20 题的 Formal，共 60 题。

### Stage C — Generic offload baseline

1. 在 16/24/32/48/64 GiB budget 测 C2；
2. 记录 OOM、load、TTFT、tok/s、disk bytes；
3. 与 C1 的 full-resident 结果对齐。

### Stage D — SparseFlow integration

1. Main Dev 提供 C3-R/C3 backend 和指标接口；
2. Benchmark 运行相同 manifests；
3. 完成 RAM/cache sweep 和 S0-S4 消融；
4. 完成 correctness gate 后再发布性能结果。

### Stage E — Formal report

1. 默认每项质量任务运行 20 题，共 60 题；若要发布模型能力结论，再扩展到更大的标准数据集；
2. 运行 BF16 与量化矩阵；
3. 输出原始 JSON、汇总表、曲线和实验限制；
4. 将结果写入公开 benchmark 报告。

## 17. 记录身份规范

新增实现、实验记录或设计决策时，记录正文末尾必须明确标注执行身份：

~~~text
[Main Dev]
[Benchmark]
~~~

Benchmark 负责的实验设计、运行、eval 和结果记录统一使用 `[Benchmark]`；
Main Dev 负责的 runtime、ExpertLocator、cache 和 kernel 实现统一使用
`[Main Dev]`。如果由多个身份共同完成，末尾列出全部身份，例如
`[Main Dev][Benchmark]`。

本规范变更由 [Benchmark] 执行。

## 18. Stage 7.4 observed closure

Stage 7.4 BF16 performance/correctness matrix 已由 Main Dev 在冻结 runtime 上
执行完成。正式证据入口为：

~~~text
benchmarks/results/2026-07-15/stage7_4/report.md
benchmarks/results/2026-07-15/stage7_4/system_summary.json
benchmarks/results/2026-07-15/stage7_4/c3_raw/
~~~

执行使用 10 threads、固定 core prompt、32-token C3 decode、warmup=1、
measured=3、1/2/4/8 GiB cache sweep，以及三次独立 model-cold S0/S3/S4
样本。C3 的 32-step logits、route、IDs 和 storage invariants 全部通过。

C2 generic offload 只完成 2-token cold/warm 基线：三次 cold TTFT 中位
308.8 秒，单 decode 在 35.7–310.7 秒之间，request 实际块读取为
64.56–128.43 GiB，因此没有伪造或外推 32-token raw sample。报告将该结果
明确标为短序列系统基线。Colibri-style task quality
ladder 仍属于独立 Benchmark workstream；BF16 streaming correctness 已由
同-kernel 完整 logits gate 覆盖，INT8/INT4 quality matrix 在 Stage 7.5 后
执行。[Main Dev]

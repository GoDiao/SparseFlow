# SparseFlow Benchmark Framework

本目录属于 `[Benchmark]` 轨道，负责可复现实验、质量 eval 和原始结果保存。
CPU runtime、ExpertLocator、cache 和 prefetch 实现属于 `[Main Dev]` 轨道。

## CPU Full Resident 性能基线

使用当前服务器已有的 PyTorch/Transformers 环境，不使用 GPU：

~~~bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 \
python3 -m benchmarks.run_cpu \
  --model model/Qwen3.6-35B-A3B \
  --manifest benchmarks/manifests/cpu_dev.jsonl \
  --dtype bf16 \
  --threads 10 \
  --limit 1 \
  --warmup 0 \
  --runs 1 \
  --max-new-tokens 8 \
  --output benchmarks/results/cpu-resident-smoke.json
~~~

首次实验建议从 `--limit 1 --max-new-tokens 8` 开始。确认 CPU forward/generate
正常后，再运行完整 manifest。完整 BF16 权重约 67GiB，加载时要预留系统和
runtime 内存。默认 runner 会 materialize 参数，使 C1 真正测量物理 CPU
resident；不要使用 `--no-materialize` 生成 C1 正式结果。

`--no-materialize` 只用于额外记录 safetensors mmap/按需 page-fault 对照，
不能和 Full Resident 结果混称。

prefill 会在正式 greedy generation 中自动单独计时；`--measure-prefill` 保留为
兼容参数，不再额外执行一次 forward：

~~~bash
python3 -m benchmarks.run_cpu \
  --model model/Qwen3.6-35B-A3B \
  --threads 10 --runs 3 --warmup 1 \
  --max-new-tokens 32 --measure-prefill \
  --output benchmarks/results/cpu-resident-decode32.json
~~~

结果文件会包含：

- model config/index SHA256；
- Git commit 和 dirty 状态；
- PyTorch/Transformers/dtype/线程数；
- tokenizer/model/materialization load 分项时间；
- prefill/TTFT、steady-state decode、端到端生成时间；
- input/output token 数和 token IDs；
- RSS、page fault、CPU time、process read bytes；
- 每个 prompt 的 raw 结果和汇总 median。

## Colibri 风格质量 smoke

`score_choices.py` 使用 Colibri 的离线 JSONL 结构，对每个选项计算
continuation log-likelihood，并输出 `accuracy`、`acc_norm_char` 和
`acc_norm_token`：

choice scoring 的 `load.seconds` 是一次性的总加载时间，已经包含
`materialize_seconds`；不能再把 materialization 加到总时间上。

~~~bash
CUDA_VISIBLE_DEVICES="" \
python3 -m benchmarks.score_choices \
  --model model/Qwen3.6-35B-A3B \
  --data benchmarks/manifests/colibri_smoke.jsonl \
  --limit 3 \
  --threads 10 \
  --output benchmarks/results/colibri-smoke.json
~~~

开发集和正式集必须在 `benchmarks/data/` 中冻结，并记录数据 manifest hash。

## Stage 7.4 SparseFlow 正式矩阵

Stage 7.4 使用独立的 C3 runner，不修改 Stage 7.3 已冻结的模型/kernel 路径：

~~~bash
PYTHONPATH=src python -m benchmarks.run_sparseflow \
  --model model/Qwen3.6-35B-A3B \
  --manifest benchmarks/manifests/stage7_4_core.jsonl \
  --variant C3-S4 --cache-bytes 4GiB \
  --cache-state workload-warm --warmup 1 --runs 3 \
  --threads 10 --max-new-tokens 32 \
  --output /tmp/stage7_4/c3-s4-4g-workload-warm.json
~~~

冻结的 C3 核心矩阵包含 C3-R、S0、S1-S4 的 1/2/4/8 GiB sweep，以及
S0/S3/S4 各三次独立 `model-cold` 样本：

~~~bash
PYTHONPATH=src python -m benchmarks.run_stage7_4_matrix \
  --output-dir /tmp/stage7_4 --threads 10 --max-new-tokens 32
~~~

`model-cold` 使用模型局部 `POSIX_FADV_DONTNEED`，并在新进程中运行；
`workload-warm` 在同一进程先执行一次固定 prompt warmup，再保留三次 raw
measurement。结果汇总命令：

~~~bash
PYTHONPATH=src python -m benchmarks.summarize_stage7_4 \
  --input-dir /tmp/stage7_4 \
  --output-json /tmp/stage7_4-summary.json \
  --output-md /tmp/stage7_4-report.md
~~~

C2 generic disk offload 需要一次性生成 Accelerate offload layout。它在
`.cache/` 下占用约 67 GiB，不进入 Git：

~~~bash
PYTHONPATH=src python -m benchmarks.prepare_generic_offload \
  --model model/Qwen3.6-35B-A3B \
  --offload-dir .cache/stage7_4/generic-offload

PYTHONPATH=src python -m benchmarks.run_generic_offload \
  --model model/Qwen3.6-35B-A3B \
  --offload-dir .cache/stage7_4/generic-offload \
  --cache-state model-cold --warmup 0 --runs 1 \
  --max-new-tokens 2 --threads 10 \
  --output /tmp/stage7_4-c2.json
~~~

线程校准和 buffered expert I/O 微基准分别由
`calibrate_stage7_4_threads.py` 与 `io_stage7_4.py` 执行。新增 Stage 7.4
runner、实验与报告由 `[Main Dev]` 记录；本目录原有规范所有者仍为
`[Benchmark]`。

已完成的正式结果入口：

~~~text
benchmarks/results/2026-07-15/stage7_4/report.md
benchmarks/results/2026-07-15/stage7_4/system_summary.json
benchmarks/results/2026-07-15/stage7_4/c3_raw/
~~~

## 结果与身份规范

raw results 放在 `benchmarks/results/`，并进入 Git，保证本机、试验机和
GitHub 之间可以同步实验数据。模型权重、下载缓存和大体积数据集仍不进入
Git；固定 manifest 放在 `benchmarks/manifests/`。

新增实现、实验记录或设计决策的正文末尾必须标注执行身份，例如：

~~~text
[Benchmark]
~~~

本框架设计和本文件由 [Benchmark] 执行。

Stage 7.4 C3/C2 runner、矩阵调度、I/O 校准及汇总说明由 [Main Dev] 补充。

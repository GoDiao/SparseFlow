# Qwen3.6 Stage 7.5.6 formal benchmark

Stage 7.5.6 froze and executed the final BF16/W8A16/W8A8 resident-streaming
matrix on the Xeon Gold 6248R experiment host. The performance matrix used 10
CPU threads, one fixed prompt, 32 greedy output tokens, one warmup plus three
measured warm runs, and three independent model-cold replicates for the final
4 GiB W8A8 streaming path.

## Performance

| Expert path | Cache | Load | TTFT | Decode | P50/P95 | Read/decode token | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 resident | all resident | 324.58 s | 4.67 s | 3.4339 tok/s | 0.289/0.310 s | 0 | 65.50 GiB |
| BF16 S1 | 8 GiB | 4.86 s | 20.86 s | 1.0822 tok/s | 0.884/1.430 s | 701.61 MiB | 13.58 GiB |
| W8A16 reference resident | all INT8 resident | 96.45 s | 7.74 s | 1.7222 tok/s | 0.575/0.672 s | 0 | 36.15 GiB |
| W8A16 reference S1 | 4 GiB | 5.91 s | 19.88 s | 0.9232 tok/s | 1.062/1.566 s | 353.62 MiB | 10.07 GiB |
| W8A8 native resident | all INT8 resident | 33.67 s | 8.25 s | 2.4925 tok/s | 0.400/0.434 s | 0 | 35.78 GiB |
| W8A8 native S1 | 4 GiB | 5.84 s | 20.21 s | 1.0981 tok/s | 0.852/1.347 s | 354.98 MiB | 9.62 GiB |
| W8A8 native S1 | 8 GiB | 5.90 s | 18.01 s | 1.4157 tok/s | 0.669/0.903 s | 192.18 MiB | 13.72 GiB |

Native W8A8 was 1.447x faster than W8A16 reference in resident decode and
1.190x faster in 4 GiB streaming on the formal prompt. The separate paired
Stage 7.5.4 gate measured 1.270x/1.241x; both experiments therefore agree that
native removes a real reference-path bottleneck, while the exact ratio remains
workload dependent.

At the same 8 GiB cache budget, native INT8 was 1.308x faster than BF16
streaming and reduced decode expert reads from 701.61 to 192.18 MiB/token.
At 4 GiB, the final path used 9.62 GiB RSS, about 14.7% of BF16 resident RSS.

The three final model-cold 4 GiB samples were:

```text
decode tok/s   0.8692 / 0.8512 / 0.8632
median         0.8632 tok/s
median TTFT    55.79 s
read/token     355.0 MiB
RSS            <= 9.59 GiB current after generation
```

The frozen Stage 7.4 Generic Offload baseline was not rerun because its runtime
and layout did not change. W8A8 S1 4 GiB was 39.3x faster than workload-warm
Generic Offload and 33.6x faster than its model-cold median.

Every BF16, W8A16, and W8A8 resident/streaming run retained exact generated
IDs and full-logit fingerprints within its precision boundary. All performance
subprocesses used clean commit `2c7e240`.

## Standard quality evaluation

Manifests were frozen from official HellaSwag validation, ARC-Challenge
validation, and MMLU test revisions with `seed=1234`. Formal contains 20
questions per task. Smoke, Pilot, and Development are strict per-task prefixes
of the same Formal file.

All formal backends used one batched choice forward per question. A standard
three-question sequential/batch smoke retained all predictions, but raw
log-likelihoods can differ due to batch GEMM arithmetic; sequential and batch
numbers are therefore not mixed.

### Aggregate Formal result

| Backend | n | Accuracy | Char-normalized | Token-normalized |
|---|---:|---:|---:|---:|
| BF16 resident | 60 | 53.33% | 66.67% | 63.33% |
| W8A8 native resident | 60 | 55.00% | 66.67% | 65.00% |
| W8A8 native streaming 4 GiB | 60 | 55.00% | 66.67% | 65.00% |

W8A8 native raw-accuracy Wilson 95% interval was 42.49–66.91%; the sample is
too small to claim a model-quality improvement over BF16. The correct result is
that no material regression was detected: BF16/native agreed on 59/60 raw
predictions, 60/60 char-normalized predictions, and 59/60 token-normalized
predictions.

### Per-task W8A8 result

| Task | n | Accuracy | Char-normalized | Token-normalized |
|---|---:|---:|---:|---:|
| HellaSwag | 20 | 55% | 75% | 70% |
| ARC-Challenge | 20 | 60% | 65% | 65% |
| MMLU | 20 | 50% | 60% | 60% |

Native resident and streaming matched exactly across all 60 questions, every
prediction, every choice total, and every token log-likelihood. Maximum
resident/streaming log-likelihood delta was `0`. Streaming read 767.36 GiB of
expert payload over the complete quality run while remaining within the 4 GiB
expert cache budget.

The W8A16 reference backend completed the standard Smoke. Its Formal task run
was intentionally stopped after the first question because repeated Python
dequantization dominated the measurement. W8A16 is still represented by the
32-token teacher-forced quality boundary and the complete formal performance
matrix; no missing quality value was imputed.

## Completion gate

```text
performance cells complete             true
all performance Git snapshots clean    true
BF16/W8A16/W8A8 storage exact           true
three native model-cold replicates      true
BF16/native resident/streaming quality  60 rows each
native quality storage exact            true
standard INT8-reference Smoke           true
Generic Offload baseline present        true
all Stage 7.5.6 gates                    PASS
```

Raw evidence:

- `benchmarks/results/2026-07-16/stage7_5/formal/summary.json`
- `benchmarks/results/2026-07-16/stage7_5/formal/performance/`
- `benchmarks/results/2026-07-16/stage7_5/formal/quality/`
- `benchmarks/manifests/quality_manifest_v1.meta.json`

[Main Dev]

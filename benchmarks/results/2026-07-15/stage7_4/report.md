# Qwen3.6 Stage 7.4 formal Benchmark report

> CPU BF16 system baseline, same-kernel C3 attribution matrix, controlled
> cold/warm expert I/O, raw repeated samples, and correctness gates. [Main Dev]

## Acceptance

```text
c3_correctness_gate                    True
single_model_revision                  True
c1_clean                               True
c2_cold_clean                          True
c2_warm_clean                          True
calibration_clean                      True
io_clean                               True
thread_calibration_exact               True
c1_repeated_ids_exact                  True
c2_cold_prefix_matches_c3              True
c2_warm_prefix_matches_c3              True
offload_layout_complete                True
cold_page_eviction_complete            True
```

All C3-R/C3-S runs used one frozen model revision and the same BF16
expert kernel. All 32 next-token logits, routes, token IDs, cache budgets,
and I/O accounting passed exactly.

## Main system results

| Mode | State | Output | n | TTFT s | Decode tok/s | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|
| C1 Transformers resident | warm | 32 | 3 | 4.130 | 2.9595 | 67.074 |
| C2 generic offload | model-cold | 2 | 3 | 308.818 | 0.02567 | 5.960 |
| C2 generic offload | workload-warm | 2 | 3 | 38.597 | 0.02794 | 5.825 |
| C3-R same-kernel resident | warm | 32 | 3 | 5.157 | 2.9618 | 66.355 |
| C3-S0 no cache | model-cold | 32 | 3 | 119.594 | 0.3725 | 6.355 |
| C3-S0 no cache | workload-warm | 32 | 3 | 24.227 | 0.4752 | 6.355 |
| C3-S1 LRU 8 GiB | workload-warm | 32 | 3 | 28.082 | 0.8263 | 13.651 |

C2 uses the same prompt but only two generated tokens because each generic
forward scans the complete offloaded checkpoint. The measured per-token decode
latency is direct; a 32-token cold C2 run was not executed because the observed
35.7–310.7-second decode steps would make the matrix prohibitively long.

## Main findings

- C3-R and C1 have essentially identical throughput: ratio `0.999`.
- C3-S1 8 GiB reaches `27.9%` of C3-R decode speed at `20.6%` of its peak RSS.
- C3-S1 8 GiB is `29.6x` faster than workload-warm generic offload.
- Even zero-cache C3-S0 is `17.0x` faster than workload-warm generic offload.
- Cold S4 is `1.17x` faster than cold S3, showing real synchronous-I/O overlap.
- In the current Python runtime, simple S1 LRU wins the warm throughput sweep;
  S2/S3 reduce logical reads at larger budgets but policy/tensor-management
  overhead prevents that reduction from becoming higher tok/s.

## Cold versus warm storage

C3-S0 TTFT changes from `24.23s` warm to `119.59s` model-cold.
At 4 GiB, cold S3 decodes at `0.3901 tok/s`; cold S4 reaches `0.4578 tok/s` while converting almost all demand misses into prefetch-served reads.
The I/O microbenchmark measured about 0.161 GiB/s model-cold and up to
3.257 GiB/s workload-warm with eight workers. OS page cache state therefore
changes the storage ceiling by roughly twenty times.

## Correctness and comparison boundaries

- C3-R versus C3-S is the algorithmic attribution boundary: same runtime,
  dispatch, expert kernel, attention/DeltaNet, KV cache, and greedy loop.
- C1 uses Transformers grouped-mm and first diverges from C3-R at generated token index `23`; C1 versus C3 is system-level only.
- C2 matches the C3 token prefix for its measured two-token request.
- This is the frozen Python BF16 result. INT8 and native kernels remain Stage 7.5.

## Reproducibility note

C1/C2 benchmark-only fixes were committed after the frozen C3 matrix; the C3 runtime/kernel source did not change.
All result files retain their exact clean commit, model hashes, environment,
raw repetitions, and `[Main Dev]` attribution.

# Qwen3.6 Stage 7.4 formal benchmark

> Frozen BF16 C3-R/C3-S benchmark with controlled workload, thread count,
> cache-state labels, three measured samples, raw JSON evidence, and exact
> same-kernel correctness gates. [Main Dev]

## Correctness gate

```text
all_attributed                         True
single_model_revision                  True
single_git_commit                      True
clean_worktrees                        True
runtime_identity_exact                 True
generated_ids_exact                    True
logit_fingerprints_exact               True
streaming_init_zero_expert_io          True
cache_budgets_respected                True
demand_accounting_exact                True
prefetch_failures_zero                 True
```

## Performance matrix

| Variant | Cache | State | n | TTFT s | Decode tok/s | p50 ms | p95 ms | Read MiB/token | Hit rate | Peak RSS GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3-R | 0 GiB | workload-warm | 3 | 5.157 | 2.9618 | 338.1 | 378.3 | 0.00 | - | 66.355 |
| C3-S0 | 0 GiB | model-cold | 3 | 119.594 | 0.3725 | 2622.7 | 3129.3 | 1920.00 | 0.00% | 6.355 |
| C3-S0 | 0 GiB | workload-warm | 3 | 24.227 | 0.4752 | 2107.4 | 2276.1 | 1920.00 | 0.00% | 6.355 |
| C3-S1 | 1 GiB | workload-warm | 3 | 23.390 | 0.4868 | 2032.9 | 2283.2 | 1920.00 | 0.00% | 6.538 |
| C3-S1 | 2 GiB | workload-warm | 3 | 24.868 | 0.6352 | 1564.5 | 2102.9 | 1251.68 | 25.56% | 7.555 |
| C3-S1 | 4 GiB | workload-warm | 3 | 26.002 | 0.7032 | 1380.7 | 1956.5 | 1022.52 | 34.44% | 9.581 |
| C3-S1 | 8 GiB | workload-warm | 3 | 28.082 | 0.8263 | 1164.1 | 1995.2 | 701.61 | 47.73% | 13.651 |
| C3-S2 | 1 GiB | workload-warm | 3 | 25.300 | 0.3798 | 2609.7 | 2888.6 | 1807.94 | 4.38% | 6.541 |
| C3-S2 | 2 GiB | workload-warm | 3 | 24.879 | 0.3921 | 2551.3 | 2819.3 | 1725.10 | 7.79% | 7.545 |
| C3-S2 | 4 GiB | workload-warm | 3 | 24.053 | 0.5204 | 1878.3 | 2295.3 | 980.71 | 36.72% | 9.581 |
| C3-S2 | 8 GiB | workload-warm | 3 | 25.719 | 0.5579 | 1741.1 | 2162.8 | 647.23 | 51.20% | 13.651 |
| C3-S3 | 1 GiB | workload-warm | 3 | 23.901 | 0.3919 | 2564.6 | 2746.8 | 1806.19 | 4.35% | 6.541 |
| C3-S3 | 2 GiB | workload-warm | 3 | 24.474 | 0.4119 | 2389.5 | 2737.4 | 1710.77 | 8.03% | 7.574 |
| C3-S3 | 4 GiB | model-cold | 3 | 119.775 | 0.3901 | 2556.3 | 3290.3 | 1028.13 | 34.11% | 9.558 |
| C3-S3 | 4 GiB | workload-warm | 3 | 24.865 | 0.5200 | 1885.4 | 2378.1 | 1024.45 | 34.54% | 9.575 |
| C3-S3 | 8 GiB | workload-warm | 3 | 26.388 | 0.5752 | 1711.4 | 2154.1 | 667.16 | 49.25% | 13.621 |
| C3-S4 | 1 GiB | workload-warm | 3 | 28.297 | 0.4324 | 2287.8 | 2608.7 | 1778.84 | 88.39% | 8.495 |
| C3-S4 | 2 GiB | workload-warm | 3 | 27.816 | 0.4387 | 2307.1 | 2511.6 | 1705.03 | 89.81% | 9.963 |
| C3-S4 | 4 GiB | model-cold | 3 | 130.971 | 0.4578 | 2152.0 | 2790.3 | 1028.13 | 80.56% | 10.815 |
| C3-S4 | 4 GiB | workload-warm | 3 | 28.059 | 0.5532 | 1818.8 | 2247.8 | 1024.45 | 90.34% | 11.285 |
| C3-S4 | 8 GiB | workload-warm | 3 | 29.865 | 0.6318 | 1589.4 | 1919.5 | 667.16 | 91.26% | 14.973 |

## Boundary

This report measures the frozen Python BF16 reference runtime on the current
Cascade Lake CPU. It attributes storage policy with C3-R/C3-S same-kernel
comparisons; it is not the Stage 7.5 INT8/native-kernel performance result.
Model-cold means model-local `POSIX_FADV_DONTNEED` was requested and each
sample ran in a fresh process. Workload-warm means one fixed warmup generation
preceded three measured generations in the same process. [Main Dev]

# Qwen3.6 Stage 7.5.5 INT8 cache and prefetch calibration

Stage 7.5.5 connected canonical INT8 experts to the shared SparseFlow cache
and asynchronous prefetch lifecycle, then recalibrated policy, byte budget,
worker count, coalescing gap, cache state, and output length for the W8A8
native runtime.

## Correctness gate

The real four-token current-route prefetch gate submitted 2,819 experts in 145
batches. It served 2,819 demand requests from prefetch, required three
synchronous misses, and recorded zero failure, zero wasted-ready bytes, and
zero transient entries after generation. Resident and streaming routes,
logits, IDs, and text remained exact.

The complete calibration matrix contained 28 independent processes:

- 16 workload-warm policy/budget cells;
- 4 model-cold cells;
- 4 worker/coalesce cells;
- 4 output-length cells.

Machine validation passed every gate:

```text
matrix cells complete                 28 / 28
one implementation commit             true
quality exact by output length         true
cache budgets respected                true
demand accounting exact                true
prefetch failures                      0
```

## Warm cache results

One warmup and one measured 32-token run were used for calibration. These are
selection measurements; Stage 7.5.6 supplies the three-repeat formal result.

| Budget | Best cached policy | Decode | Physical read/token | RSS after generation |
|---:|---|---:|---:|---:|
| 0.5 GiB | S3 heat | 0.6094 tok/s | 1,252.0 MiB | 6.05 GiB |
| 1 GiB | S1 LRU | 0.7696 tok/s | 960.8 MiB | 6.55 GiB |
| 2 GiB | S1 LRU | 0.8888 tok/s | 847.4 MiB | 7.57 GiB |
| 4 GiB | S1 LRU | 1.0816 tok/s | 685.8 MiB | 9.61 GiB |
| 8 GiB | S1 LRU | 1.4312 tok/s | 457.1 MiB | 13.67 GiB |

The no-cache S0 result was 0.6146 tok/s. A 0.5 GiB cache therefore did not
produce a defensible speedup over no-cache in this single calibration sample;
0.5 GiB remains a constrained-memory option rather than a recommended default.

S1 was fastest at every measured 1-8 GiB budget. S3 sometimes read slightly
fewer bytes, but its policy maintenance did not recover that cost in throughput.
S4 had high cache-hit accounting because current-route prefetch inserted data
before demand, but its physical read volume stayed close to S3 and its Future/
coordination overhead made it slower than S1.

## Cold, I/O, and length results

At 4 GiB model-cold:

| Policy | TTFT | Decode |
|---|---:|---:|
| S0 no cache | 55.55 s | 0.5126 tok/s |
| S1 LRU | 55.41 s | 0.8813 tok/s |
| S3 heat | 55.49 s | 0.8391 tok/s |
| S4 prefetch | 71.19 s | 0.8106 tok/s |

Unlike the earlier BF16 experiment, INT8 S4 did not win this model-cold cell.
The smaller 3 MiB expert and native decode changed the I/O/coordination balance.

For S4 at 1 GiB, the best tested I/O setting was two workers with a 64 KiB
coalesce gap at 0.6681 tok/s. It remained slower than S1's 0.7696 tok/s. S1
also won the 8-token and 16-token comparisons, so predictive prefetch is not
enabled dynamically for these short requests.

## Selected defaults

- Default warm policy: S1 LRU.
- Default cold policy on this host: S1 LRU.
- Default prefetch: disabled.
- Recommended development budget: 4 GiB, balancing 1.0816 tok/s and 9.61 GiB
  current RSS.
- Higher-throughput budget: 8 GiB, reaching 1.4312 tok/s at 13.67 GiB RSS.
- S4 remains opt-in and retains two workers/64 KiB as its measured host-specific
  setting; it is not a global default.

Raw evidence:

- `benchmarks/results/2026-07-16/stage7_5/int8_prefetch_4tok.json`
- `benchmarks/results/2026-07-16/stage7_5/cache_matrix/matrix_execution.json`
- `benchmarks/results/2026-07-16/stage7_5/cache_matrix/summary.json`
- all 28 cell JSON files in the same cache-matrix directory

[Main Dev]

# ExpertCache long trace and real-route comparison

**Owner:** `[Main Dev]`  
**Date:** 2026-07-13  
**Status:** initial comparison, not a formal end-to-end benchmark

## Scope

This document covers the Main Dev storage/cache microbenchmark only. It does
not replace the Benchmark workstream's full generation, quality, or backend
comparison framework.

All runs use the real Qwen3.6 safetensors files. The benchmark replays one
fixed trace for every cache capacity in a sweep.

## Generated trace comparison

Common settings:

```text
layers:      0-3
tokens:      16
top-k:       8
requests:    512
capacities:  0, 4, 8, 16, 32 slots/layer
seed:        1234
```

### Locality trace

Trace SHA-256:
`ae2505b54e9558f38a42f174d6e162ba06c3a4e0d46440d89ea3b8e738ddfdc3`

| Slots/layer | Hit rate | Loaded bytes | Cached bytes | Wall time |
|---:|---:|---:|---:|---:|
| 0 | 0.00% | 3.00 GiB | 0 B | 1118.30 ms |
| 4 | 4.30% | 2.87 GiB | 96 MiB | 872.42 ms |
| 8 | 11.72% | 2.65 GiB | 192 MiB | 1165.37 ms |
| 16 | 36.72% | 1.90 GiB | 384 MiB | 824.18 ms |
| 32 | 75.00% | 768 MiB | 768 MiB | 657.35 ms |

Raw result: [`expert_cache_20260713_qwen36_locality_long.json`](./expert_cache_20260713_qwen36_locality_long.json)

### Uniform trace

Trace SHA-256:
`c03d5eb5f46cb9a42bab314f8500698b7a1aba19cb2ce0ce8314c1d24de1b4b0`

| Slots/layer | Hit rate | Loaded bytes | Cached bytes | Wall time |
|---:|---:|---:|---:|---:|
| 0 | 0.00% | 3.00 GiB | 0 B | 7144.25 ms |
| 4 | 0.59% | 2.98 GiB | 96 MiB | 1205.80 ms |
| 8 | 2.15% | 2.94 GiB | 192 MiB | 1292.62 ms |
| 16 | 6.25% | 2.81 GiB | 384 MiB | 1210.25 ms |
| 32 | 11.13% | 2.67 GiB | 768 MiB | 1165.49 ms |

Raw result: [`expert_cache_20260713_qwen36_uniform_long.json`](./expert_cache_20260713_qwen36_uniform_long.json)

### Interpretation

- The cache benefits strongly from route locality. At 32 slots/layer,
  locality reaches 75.00% hit rate while uniform reaches 11.13%.
- The locality trace reduces logical loaded bytes from 3.00 GiB to 768 MiB.
- The uniform trace remains close to the no-cache lower bound because its
  working set is much larger than the tested capacities.
- Wall time is not a clean storage-bandwidth comparison: the first capacity
  can warm the Linux page cache for later capacities. Use loaded bytes and
  application hit rate as the stable result in this run.

## Real Qwen3.6 route trace

The new `route-trace` command uses a forward hook on the actual
`Qwen3_5MoeTopKRouter` modules and records their `selected_experts` output.
This is not a synthetic or logits-reconstructed route.

Capture command:

```bash
PYTHONPATH=src python -m sparseflow route-trace \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --prompt '请用一句话解释什么是稀疏专家模型。' \
  --max-new-tokens 2 \
  --output docs/results/qwen36_route_trace_short_20260713.json
```

Trace facts:

```text
input tokens:       8
generated tokens:   2
forward calls:      2
expert requests:    2880
unique layer/expert pairs: 1577
average unique experts/layer: 39.425
```

Trace file: [`qwen36_route_trace_short_20260713.json`](./qwen36_route_trace_short_20260713.json)

Replayed result:
[`expert_cache_20260713_qwen36_real_route.json`](./expert_cache_20260713_qwen36_real_route.json)

| Slots/layer | Hit rate | Loaded bytes | Cached bytes | Wall time |
|---:|---:|---:|---:|---:|
| 0 | 0.00% | 16.88 GiB | 0 B | 3326.38 ms |
| 8 | 25.94% | 12.50 GiB | 1.88 GiB | 5649.98 ms |
| 32 | 44.93% | 9.29 GiB | 7.50 GiB | 6327.38 ms |
| 64 | 45.24% | 9.24 GiB | 9.24 GiB | 6663.58 ms |

This is the first real-routing result, but it contains a prompt prefill and
only two generated tokens. The current replay treats each selected expert as a
sequential request; batch-union and per-token time-series accounting are not
implemented yet. Therefore this result measures route reuse under the current
cache model, not final decode performance.

## Next Main Dev measurements

1. Capture a longer real route trace, ideally 8–32 generated tokens.
2. Add per-forward/per-token route statistics to the trace result.
3. Add batch-union accounting so prefill does not count duplicate expert loads.
4. Add controlled page-cache handling before claiming cold SSD bandwidth.
5. Feed the route/cache metrics into the separate Benchmark result schema.

All implementation and observations in this document were produced by
`[Main Dev]`.

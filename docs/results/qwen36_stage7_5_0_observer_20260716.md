# Qwen3.6 Stage 7.5.0 observer-effect closure

Stage 7.5.0 separated fixed-cost performance counters from detailed runtime
diagnostics before INT8/native work began. The experiment used one loaded C3-R
BF16 runtime, a fixed prompt, 10 CPU threads, 8 generated tokens, and three
interleaved repetitions of `none`, `summary`, and `layer` telemetry.

## Correctness and acceptance

```text
all_logits_and_ids_exact                 True
summary_decode_delta_within_3_percent    True
summary_records_are_aggregated           True
critical_path_closes_within_5_percent    True
timing_categories_present                True
all_pass                                 True
```

All nine runs produced identical generated IDs and complete next-token logits
fingerprints.

## Observer effect

| Level | Median decode tok/s | Delta vs none | Median wall s | Observer self-time |
|---|---:|---:|---:|---:|
| none | 3.4243 | - | 5.2927 | 0 ms |
| summary | 3.4046 | -0.57% | 5.4473 | 6.97 ms |
| layer | 3.1181 | -8.94% | 5.6893 | 22.65 ms |

`summary` now keeps only O(1) counters and per-forward aggregates. It does not
compute per-layer unique experts or retain layer records. `layer` is explicitly
a diagnostic mode and is excluded from performance headline measurements.

## Timing closure

The median detailed critical-path closure was `98.91%`. The available
categories are router, dispatch, prepare, provider get, expert kernel, routing
accumulation, cache lookup, victim selection, allocation/reuse, policy
maintenance, positional reads, tensor decode/view, and telemetry observer
time. Async prefetch worker time remains separate from additive foreground
critical-path accounting.

## Implementation consequences

- Provider hot-path snapshots no longer sort HeatPolicy diagnostics.
- Exact cache counters remain available to `summary` without full key tables.
- LRU victim selection and cache entry counting have O(1) fast paths.
- Decoded tensor views are removed by exact eviction callbacks rather than a
  full cache scan after each miss.
- Non-prefetch providers bypass Future/reconciliation locks.
- Demand reads can transfer ownership of directly filled writable buffers to
  the cache without a `bytes -> bytearray` copy.

Raw evidence:
`benchmarks/results/2026-07-16/stage7_5/observer_effect.json`.

[Main Dev]

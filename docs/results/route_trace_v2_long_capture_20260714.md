# Qwen3.6 route-trace v2 long capture

**Owner:** `[Main Dev]`
**Date:** 2026-07-14
**Status:** captured and schema-validated

## Workload

Five prompt categories were captured with the real Qwen3.6 Transformers router:

| ID | Category |
|---|---|
| `zh-moe` | Chinese |
| `en-moe` | English |
| `code-moe` | Code |
| `math-moe` | Math |
| `conversation-moe` | Continued tiered-memory conversation |

Manifest: [`route_trace_v2.jsonl`](../../benchmarks/manifests/route_trace_v2.jsonl)

Each capture uses batch size 1, greedy generation, and the actual
`selected_experts` output of `Qwen3_5MoeTopKRouter`.

## Capture sets

| Generated tokens | Trace directory |
|---:|---|
| 8 | [`qwen36_route_v2_20260714`](./qwen36_route_v2_20260714/) |
| 16 | [`qwen36_route_v2_20260714_t16`](./qwen36_route_v2_20260714_t16/) |
| 32 | [`qwen36_route_v2_20260714_t32`](./qwen36_route_v2_20260714_t32/) |

Every trace preserves:

```text
forward -> phase -> row/token -> layer -> expert_ids
```

### Raw request counts

Each request is one selected expert at one layer for one token row. The counts
below include prefill and decode.

| Prompt | Input tokens | 8 generated | 16 generated | 32 generated |
|---|---:|---:|---:|---:|
| `code-moe` | 17 | 7680 | 10240 | 15360 |
| `conversation-moe` | 28 | 11200 | 13760 | 18880 |
| `en-moe` | 25 | 10240 | 12800 | 17920 |
| `math-moe` | 33 | 12800 | 15360 | 20480 |
| `zh-moe` | 18 | 8000 | 10560 | 15680 |

The formula is consistent with batch-1 generation:

```text
(input_tokens + generated_tokens - 1) × 40 layers × top-8
```

## What this enables

The v2 traces can now support separate calculations for:

- prefill unique experts per layer;
- per-decode-forward cache hits and misses;
- token-level loaded bytes;
- batch-union savings during prefill;
- reuse distance and hot-expert promotion policy.

The previous flat v1 replay could not preserve these boundaries.

## Next measurement

Replay each trace with `--batch-union` and capacities `8/16/32/64/128`, then
compare prefill and decode separately. After that, use the same traces for the
single-layer resident/streaming correctness experiment. [Main Dev]

All implementation and observations in this document were produced by
`[Main Dev]`.

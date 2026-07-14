# Route trace v2 and phase-aware batch-union validation

**Owner:** `[Main Dev]`
**Date:** 2026-07-14
**Status:** schema and replay validation

## Why schema v2 exists

The original route trace stored a flat list of `(layer, expert)` requests. That
was enough for a basic LRU replay, but it lost the boundaries needed to reason
about prefill, decode, and batch union.

Schema v2 groups data as:

```text
forward
  └── row/token
        └── layer
              └── expert_ids
```

Each row also keeps `phase`, `token_position`, and `token_id` when available.
The old flat v1 format remains readable.

## Real trace

Prompt:

```text
请用一句话解释什么是稀疏专家模型。
```

Capture command:

```bash
PYTHONPATH=src python -m sparseflow route-trace \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --prompt '请用一句话解释什么是稀疏专家模型。' \
  --max-new-tokens 2 \
  --output docs/results/qwen36_route_trace_v2_short_20260714.json
```

Trace file:
[`qwen36_route_trace_v2_short_20260714.json`](./qwen36_route_trace_v2_short_20260714.json)

Observed structure:

```text
forward 0: prefill, 8 rows
forward 1: decode,   1 row
top-k: 8
layers: 40
raw requests: 2880
```

## Batch-union replay

Command:

```bash
PYTHONPATH=src python -m sparseflow expert-bench \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --capacities 32,64 \
  --trace docs/results/qwen36_route_trace_v2_short_20260714.json \
  --batch-union \
  --output docs/results/expert_cache_20260714_real_route_v2_batch_union.json
```

Raw result:
[`expert_cache_20260714_real_route_v2_batch_union.json`](./expert_cache_20260714_real_route_v2_batch_union.json)

The same replay after the persistent-fd `ShardReader` change is recorded in:
[`expert_cache_20260714_real_route_v2_pread.json`](./expert_cache_20260714_real_route_v2_pread.json)

Request accounting:

| Phase | Raw expert selections | Effective loads after union |
|---|---:|---:|
| Prefill | 2560 | 1468 |
| Decode | 320 | 320 |
| Total | 2880 | 1788 |

The prefill union removes `1092` duplicate expert selections before the cache
replay. Decode has one row, so batch union does not reduce its request count.

## Phase-aware cache result

| Capacity | Decode hit rate | Decode logical bytes | Decode loaded bytes |
|---:|---:|---:|---:|
| 32 slots/layer | 58.75% | 1.875 GiB | 0.773 GiB |
| 64 slots/layer | 65.94% | 1.875 GiB | 0.639 GiB |

The result reproduces the expected conclusion: 64 slots/layer improves decode
hit rate, but the incremental benefit over 32 slots/layer is limited. Prefill
itself has zero application-cache hits in this short run because each layer's
unique working set is larger than the tested cache capacity after union.

The persistent reader reports 3200 positional reads for the 32-slot run and
3154 for the 64-slot run. Each miss currently produces two reads because the
Qwen3.6 `gate_up_proj` and `down_proj` slices are in different shards.

## Limitations

- This is one short prompt and one decode forward.
- It is not a steady-state generation result.
- The current replay measures storage/cache behavior, not MoE computation.
- Page-cache state still affects wall-clock and process I/O measurements.

## Next route experiments

Capture schema v2 traces for 8, 16, and 32 generated tokens across Chinese,
English, code, math, and continued-conversation prompts. Keep each prompt and
trace hash in its own result record. [Main Dev]

All implementation and observations in this document were produced by
`[Main Dev]`.

# ExpertCache byte-budget pilot

**Owner:** `[Main Dev]`
**Date:** 2026-07-14

This pilot verifies the global byte-budget mode in addition to the existing
per-layer slots mode.

## Workload

```text
model:      Qwen3.6-35B-A3B
layers:     0-3
tokens:     16 synthetic locality tokens
top-k:      8
requests:   512
seed:       1234
budgets:    512 MiB, 1 GiB
```

Command:

```bash
PYTHONPATH=src python -m sparseflow expert-bench \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --byte-budgets 512MiB,1GiB \
  --layers 0-3 --tokens 16 --topk 8 \
  --mode locality --seed 1234 \
  --output docs/results/expert_cache_20260714_byte_budget_locality.json
```

Raw result: [`expert_cache_20260714_byte_budget_locality.json`](./expert_cache_20260714_byte_budget_locality.json)

## Result

| Global budget | Hit rate | Loaded bytes | Cached bytes |
|---:|---:|---:|---:|
| 512 MiB | 51.17% | 1.46 GiB | 510 MiB |
| 1 GiB | 75.00% | 768 MiB | 768 MiB |

The result confirms that a global byte budget works independently of the
per-layer slot count. It is the correct abstraction for uneven expert sizes,
future quantized layouts, and models whose layers do not have equal cache
costs.

This run uses a synthetic trace and warm page-cache conditions; it is a cache
policy validation, not a cold SSD result. [Main Dev]

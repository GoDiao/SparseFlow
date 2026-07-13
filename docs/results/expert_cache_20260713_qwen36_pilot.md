# ExpertCache pilot — Qwen3.6

**Owner:** `[Main Dev]`  
**Date:** 2026-07-13  
**Status:** initial storage/cache pilot

## Scope

This is a storage-layer microbenchmark. It does not run Qwen3.6 forward, does
not measure generation quality, and is separate from the Benchmark workstream's
end-to-end evaluation framework.

The benchmark replays the same expert request trace for each application-level
per-layer LRU capacity.

## Model and trace

```text
model:       /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B
layers:      0-3
trace mode:  locality
tokens:      4
top-k:       8
requests:    128
seed:        1234
trace SHA:   aac1845a22190f52b8d0fbc0032bdbf92b8ce177a05048da6737b4ad87fbd448
```

Raw result:

[`expert_cache_20260713_qwen36_pilot.json`](./expert_cache_20260713_qwen36_pilot.json)

Command:

```bash
PYTHONPATH=src python -m sparseflow expert-bench \
  /root/workspace/SparceFlow/model/Qwen3.6-35B-A3B \
  --capacities 0,1,2,4 \
  --layers 0-3 \
  --tokens 4 \
  --topk 8 \
  --mode locality \
  --seed 1234 \
  --output docs/results/expert_cache_20260713_qwen36_pilot.json
```

## Results

This was a warm-page-cache repeat. The logical application-cache results were:

| Slots/layer | Requests | Hit rate | Loaded bytes | Cached bytes | Wall time |
|---:|---:|---:|---:|---:|---:|
| 0 | 128 | 0.00% | 768 MiB | 0 B | 181.48 ms |
| 1 | 128 | 0.78% | 762 MiB | 24 MiB | 191.16 ms |
| 2 | 128 | 1.56% | 756 MiB | 48 MiB | 236.80 ms |
| 4 | 128 | 4.69% | 732 MiB | 96 MiB | 294.35 ms |

The trace has only four tokens and a 32-expert locality working set per layer,
so the small hit rates are expected. Larger traces are required before making
capacity decisions.

## Interpretation

- The cache is functioning: increasing capacity reduces `loaded_bytes`.
- One slot per layer costs 24 MiB for these four layers, matching the 6 MiB
  per-expert measurement.
- The benchmark currently reports logical bytes loaded by the application and
  process timing.
- `/proc/self/io` reported zero additional `read_bytes` in this repeat, which
  means the files were already resident in the Linux page cache.
- These numbers must not be described as cold SSD performance.

## Next measurements

1. Run a longer trace with the same trace hash across capacities.
2. Add explicit page-cache control or a documented cold-read protocol.
3. Compare `uniform` and `locality` traces.
4. Export route traces from a real model backend before calling the hit rate a
   model-real routing result.
5. Use this result contract as an input to the separate Benchmark workstream.

All observations and conclusions in this document were produced by `[Main Dev]`.

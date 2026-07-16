# Qwen3.6 Stage 7.5.1 BF16 hot-path result

Stage 7.5.1 optimized the existing BF16 streaming implementation without
changing model precision, routed-expert arithmetic, cache policy, formal route,
or expert read volume. The clean comparison reused the frozen Stage 7.4
C3-S1 8 GiB workload: 10 CPU threads, 32 generated tokens, one warmup, and
three measured workload-warm runs.

## Result

| Metric | Stage 7.4 S1-8G | Stage 7.5.1 S1-8G | Change |
|---|---:|---:|---:|
| Decode tok/s | 0.8263 | 1.0763 | +30.3% |
| TTFT | 28.08 s | 20.95 s | -25.4% |
| Decode P50 | 1.164 s | 0.890 s | -23.5% |
| Expert read/decode token | 701.61 MiB | 701.61 MiB | exact |
| Peak RSS | 13.651 GiB | 13.593 GiB | -0.4% |

The median measured cache behavior remained exactly `6,449` hits, `7,062`
misses, and `41.3789 GiB` loaded per request. All three measured runs retained
the same 32 generated IDs, route hashes, and complete logits fingerprints as
the Stage 7.4 BF16 result.

The first measured request started from a slightly different warmup cache state
and recorded 29 additional prefill hits; runs two and three exactly matched the
old hit/miss sequence. Median accounting and every decode-forward read volume
therefore remain directly comparable.

## Implemented hot-path changes

- O(1) provider counters and deferred HeatPolicy diagnostics;
- O(1) LRU victim and cache-entry accounting;
- exact decoded-view invalidation through eviction callbacks;
- complete no-prefetch lock/Future/reconciliation bypass;
- direct positional reads into writable cache-owned buffers;
- bounded size-class BufferPool for synchronous expert reads;
- no repeated `bytes -> bytearray` payload copy.

The pool reused buffers a median `14,124` times per measured request. It is
deliberately disabled for asynchronous prefetch until cache entries have an
explicit pin/lease protocol; this prevents reusing a backing buffer while a
concurrent kernel may still reference it.

Raw evidence:
`benchmarks/results/2026-07-16/stage7_5/bf16_s1_8g.json`.

[Main Dev]

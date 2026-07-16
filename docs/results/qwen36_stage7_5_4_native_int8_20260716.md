# Qwen3.6 Stage 7.5.4 native INT8 runtime

Stage 7.5.4 replaced per-request W8A16 reference dequantization with one
canonical-row-major AVX-512 VNNI W8A8 kernel. Resident and streaming providers
use the same native operator and activation quantization; only expert storage
and cache behavior differ.

## Implementation boundary

- Dynamic per-row U8 affine activation quantization.
- Canonical S8 per-output-channel expert weights and FP16 scales.
- AVX-512 VNNI `dpbusd` accumulation with zero-point row-sum correction.
- No weight prepack on a streaming cache miss.
- Native expert views are tied to the exact `CachedExpert` entry and removed
  on eviction.
- Decode and prefill currently share the same correctness-first operator.

## Storage correctness

The real 32-token resident/streaming gate passed every invariant:

```text
same runtime identity             true
same route audit                  true
full logits/IDs/text exact        true
resident expert I/O               0 B
streaming expert I/O              16.975 GiB
streaming current RSS             9.608 GiB
cache budget                      <= 4 GiB
demand accounting                 exact
```

## Paired performance gate

The paired run used 10 CPU threads, one warmup and three measured 32-token
runs per kernel. AB/BA order reduced drift. OS page cache remained explicitly
uncontrolled and workload-warm.

| Path | W8A16 reference | W8A8 native | Speedup |
|---|---:|---:|---:|
| Resident decode | 1.7699 tok/s | 2.2483 tok/s | 1.270x |
| Streaming decode, 4 GiB LRU | 0.9153 tok/s | 1.1361 tok/s | 1.241x |
| Resident prefill | 5.087 s | 8.017 s | 0.635x |
| Streaming prefill | 15.396 s | 13.709 s | 1.123x |

All six measured reports passed storage invariants. Each kernel also retained
identical generated IDs and full-logit fingerprints across all repetitions and
between its resident/streaming paths.

The native kernel improves decode because it removes repeated full-expert
dequantization. Resident prefill remains slower than the W8A16 reference path;
prefill token grouping and a dedicated grouped-GEMM implementation remain a
native-kernel optimization target rather than a completed claim.

## Activation-quantization boundary

W8A8 native was teacher-forced on the W8A16 reference greedy continuation, so
the following difference excludes the already-measured BF16-to-INT8 weight
quantization loss:

```text
greedy first divergence           none in 32 tokens
argmax equal                      32 / 32
maximum absolute logit error      2.000
mean absolute logit error         0.120184
maximum KL(reference || native)   0.016955
mean KL(reference || native)      0.001021
mean top-10 overlap               96.25%
```

Raw evidence:

- `benchmarks/results/2026-07-16/stage7_5/int8_native_32tok.json`
- `benchmarks/results/2026-07-16/stage7_5/int8_native_quality_32tok.json`
- `benchmarks/results/2026-07-16/stage7_5/int8_native_paired.json`

[Main Dev]

# Qwen3.6 Stage 7.5.3 INT8 reference runtime

Stage 7.5.3 connected `canonical-int8-v1` to the complete Qwen3.6 text-only
memory-native runtime. INT8 resident and streaming providers consume identical
quantized bytes and FP16 scales, dequantize each requested expert to BF16, and
then call the same eager routed-expert kernel.

## Storage correctness

The 32-token resident/streaming gate passed every invariant:

```text
same runtime identity                 True
same 1,280 route records              True
32 complete logits fingerprints       True
generated IDs and text exact          True
resident payload exactly 30.078 GiB   True
resident generation expert I/O = 0    True
streaming init expert I/O = 0         True
4 GiB cache budget respected          True
demand accounting exact               True
```

| Path | Load | TTFT | Decode | RSS after generation | Expert I/O |
|---|---:|---:|---:|---:|---:|
| INT8 resident reference | 56.84 s | 5.18 s | 1.649 tok/s | 35.92 GiB | 0 |
| INT8 streaming reference 4 GiB | 2.60 s | 14.19 s | 0.923 tok/s | 9.91 GiB | 16.95 GiB/request |

Streaming read `335.20 MiB/decode token`, versus `701.61 MiB/token` for the
Stage 7.4 BF16 S1-8G baseline. Reference BF16 dequantization itself costs about
`552.5 ms/decode token`; removing that cost is the primary Stage 7.5.4 native
kernel target.

## Quantization boundary

BF16 and INT8 resident paths were compared on a shared BF16 greedy
teacher-forced continuation so every logit pair used the same context:

```text
greedy first divergence          none in 32 tokens
argmax equal                     32 / 32
maximum absolute logit error     1.875
mean absolute logit error        0.101064
maximum KL(BF16 || INT8)         0.015296
mean KL(BF16 || INT8)            0.001185
mean top-10 overlap              97.81%
```

## Choice-scoring smoke

The existing three-question Colibri-style toy manifest was run through the
same continuation log-likelihood scorer:

```text
BF16 accuracy / normalized accuracy   2/3
INT8 accuracy / normalized accuracy   2/3
all three predictions equal           True
mean absolute choice LL delta          0.04898
maximum absolute choice LL delta       0.24948
```

This toy result is a backend regression smoke, not a model-quality claim.
HellaSwag/ARC/MMLU Pilot and Formal manifests remain part of the Stage 7.5.6
Benchmark track.

Raw evidence:

- `benchmarks/results/2026-07-16/stage7_5/int8_reference_32tok.json`
- `benchmarks/results/2026-07-16/stage7_5/int8_quality_32tok.json`
- `benchmarks/results/2026-07-16/stage7_5/choices_bf16_smoke.json`
- `benchmarks/results/2026-07-16/stage7_5/choices_int8_smoke.json`

[Main Dev]

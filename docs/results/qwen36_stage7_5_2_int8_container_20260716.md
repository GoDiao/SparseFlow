# Qwen3.6 Stage 7.5.2 INT8 container result

Stage 7.5.2 produced the first complete `canonical-int8-v1` routed-expert
container for Qwen3.6-35B-A3B. Dense, shared-expert, router, attention, and
DeltaNet weights remain in their original BF16 checkpoint.

## Conversion

```text
layers                         40
experts                        10,240
source routed BF16             60.000 GiB
INT8 logical                   30.059 GiB
INT8 physical                  30.078 GiB
physical / BF16                50.13%
conversion wall                373.03 s
conversion peak RSS            630.84 MiB
```

The converter processed one expert at a time and never materialized a fused
layer or the complete routed checkpoint. Each layer was written to a temporary
file and atomically renamed with a sidecar index.

## Format

- one `.sfi` file per layer;
- expert-major canonical row-major S8 data;
- per-output-channel symmetric quantization;
- FP16 scales and clamp range `[-127, 127]`;
- `gate_up_proj` followed by `down_proj` for every expert;
- 4 KiB-aligned data and scale segments;
- source model hashes, format version, offsets, shapes, per-segment checksums,
  layer checksums, and global index checksum.

## Audit

All 40 layer checksums were re-read through the resume path. It recognized all
40 layers as complete, rewrote none, and finished in 77.69 seconds.

Fixed samples `(0,0)`, `(0,255)`, `(19,127)`, `(39,0)`, and `(39,255)` passed
container checksum, shape, alignment, and finite-value checks. Across the ten
sampled tensor parts:

```text
maximum absolute dequantization error   0.00201416
mean absolute dequantization error      0.00006737
```

The native model artifact remains under `.cache/stage7_5/qwen36-int8` and is
not committed. The portable audit evidence is
`benchmarks/results/2026-07-16/stage7_5/int8_container_audit.json`.

[Main Dev]

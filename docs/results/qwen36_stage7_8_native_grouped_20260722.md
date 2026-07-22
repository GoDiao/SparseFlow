# Qwen3.6 Stage 7.8 Native Grouped MoE Acceleration

**Executor:** `[Main Dev]`
**Date:** 2026-07-22
**Model:** `Qwen/Qwen3.6-35B-A3B`
**Storage:** `canonical-int8-v1` with offline execution metadata

## Executive result

Stage 7.8 produced a real per-expert grouped native operator and integrated
it into a fixed resident Qwen cohort. The operator gate passes. The streaming
sub-cohort policy is implemented and measured, but remains gated because it
does not consistently beat round-robin at 4 GiB and 8 GiB.

The safe runtime decision is therefore:

```text
native grouped operator: GO
resident grouped experiment: GO, opt-in only
streaming grouped scheduler: NO-GO / gated
default dispatch: unchanged
```

## Implemented modules

- `native/grouped_moe.cpp`: GroupPlan construction, grouped gate/up and down
  execution, deterministic routing reduce, and reusable workspace buffers.
- `src/sparseflow/native_moe.py`: `GroupedMoEWorkspace` and
  `run_grouped_native_moe`.
- `src/sparseflow/text_runtime.py`: opt-in `native_dispatch="grouped"`.
- `src/sparseflow/fixed_cohort.py`: equal-length fixed-cohort Qwen runner.
- `src/sparseflow/cohort_policy.py`: bounded working-set partition helper.
- `benchmarks/bench_stage7_8_grouped.py`: real hidden/routes operator matrix.
- `benchmarks/run_stage7_8_cohort.py`: resident full-runtime cohort benchmark.
- `benchmarks/simulate_stage7_8_streaming.py`: raw INT8 cache-aware replay.

## Operator gate

The real Stage 7.7 hidden/routes fixture was run with B=1/2/4/8:

| Batch | Old fused / canonical | Grouped / canonical | Grouped / old fused | Exact |
|---:|---:|---:|---:|:---:|
| 1 | 3.702x | 3.717x | 1.004x | yes |
| 2 | 1.509x | 2.116x | 1.402x | yes |
| 4 | 1.702x | 1.985x | 1.166x | yes |
| 8 | 2.046x | 3.102x | 1.516x | yes |

B=1 stayed within the 3% regression limit. B=4 exceeded the `1.95x`
operator target. All grouped outputs were exact against the old fused output,
including argmax and repeated-call determinism.

The workspace is reused across calls. The captured allocation sizes were
approximately 0.13 MiB, 0.25 MiB, 0.48 MiB, and 0.96 MiB for B=1/2/4/8 in
the fixture benchmark.

## Resident fixed cohort

The full Qwen runner uses equal-length real prompts, independent session rows,
and one grouped decoder forward. Grouped and old fused batch outputs were exact
for both B=4 and B=8: logits fingerprints, token IDs, and text matched.

Against independent batch-one sessions, generated IDs, text, and argmax were
equal. Full-vocabulary logits were not bit-exact because batched ATen dense
operators and independent batch-one operators use different reduction paths;
this is recorded as a separate numeric result rather than attributed to
expert storage.

| Batch | Independent / grouped | Grouped / fused batch | IDs/text/argmax |
|---:|---:|---:|:---:|
| 4 | 2.167x | 1.092x | exact |
| 8 | 2.420x | 0.862x | exact |

The B=4 profile showed routed expert time falling from `5233 ms` to
`4181 ms` across the profiled cohort execution. B=8 also reduced routed
expert time from `8989 ms` to `6425 ms`, but complete decode performance was
more sensitive to linear-attention and host scheduling overhead. Grouped
dispatch is consequently opt-in and requires further paired profiling before
becoming the default.

## Streaming sub-cohort gate

The policy limits each sub-cohort's `(layer, expert)` working set to 25% of
the cache budget and uses real canonical INT8 pread replay. The baseline is
the Stage 7.7 round-robin replay on the same trace.

| Schedule | Cache | Cache-aware loaded | Round-robin loaded | Ratio | Gate |
|---|---:|---:|---:|---:|:---:|
| B=4 | 4 GiB | 32.055 GiB | 32.049 GiB | 1.00018x | fail |
| B=4 | 8 GiB | 16.195 GiB | 16.186 GiB | 1.00054x | fail |
| B=8 | 4 GiB | 101.609 GiB | 101.597 GiB | 1.00012x | fail |
| B=8 | 8 GiB | 43.291 GiB | 43.579 GiB | 0.99340x | pass |

All byte budgets and lease-release invariants passed. The policy is not yet a
general streaming improvement: three core cells have small read amplification.
The previous Stage 7.7 4 GiB union failure therefore remains unresolved.

## Verification and raw evidence

The structured verifier passed all integrity, exactness, budget, accounting,
and lease checks while correctly reporting `all_pass=false` for the explicit
performance/streaming NO-GO reasons.

The later formal resident acceptance expanded the resident evidence to B=1/4/8
with 32-token same-process ABBA repetitions. It passed full-logit, route,
behavior, repeat, and runtime-identity gates; see
`docs/results/qwen36_stage7_8_formal_acceptance_20260722.md`. It does not alter
the separate streaming NO-GO.

Authoritative files:

- `benchmarks/results/2026-07-22/stage7_8/grouped_operator.json`
- `benchmarks/results/2026-07-22/stage7_8/resident_cohort_b4_profile.json`
- `benchmarks/results/2026-07-22/stage7_8/resident_cohort_b8.json`
- `benchmarks/results/2026-07-22/stage7_8/streaming_subcohort_ratio025.json`
- `benchmarks/results/2026-07-22/stage7_8/summary.json`
- `benchmarks/results/2026-07-22/stage7_8/verification.json`
- `benchmarks/results/2026-07-22/stage7_8/formal_resident_abba.json`

<!-- [Main Dev] -->

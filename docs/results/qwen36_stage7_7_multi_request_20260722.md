# Qwen3.6 Stage 7.7 Multi-request Batching Feasibility

**Executor:** `[Main Dev]`
**Date:** 2026-07-22
**Model:** `Qwen/Qwen3.6-35B-A3B`
**Trace:** 20 independent prompts, schema v3, SHA-256
`1d78171f87409ff10d35d0341b68dccb3b8f32041990a95922b50a28edc38245`
**Container:** canonical INT8 routed-expert storage, `canonical-int8-v1`

## Executive Result

Stage 7.7 ends with a **structured NO-GO for the shared streaming scheduler**.
The real grouped routed-MoE kernel gate passes, but the union schedule does not
consistently reduce cache reads against a round-robin baseline. The measured
next scope is therefore a **resident-only fixed-cohort scheduler**. Streaming
batching remains gated until cache-aware ordering/admission can avoid the
working-set thrash measured below.

This is an experimental conclusion, not an implementation failure: the
feasibility question produced the evidence required to stop before building a
misleading scheduler.

## Inputs and Protocol

- 20 unique prompts covered Chinese, English, code, mathematics, knowledge,
  and conversation/locality categories.
- Synchronous decode schedules used real captured decode forwards for B=4 and
  B=8. No route was copied to create a session.
- Raw replay used `Int8ExpertIndex`, `ExpertCache`, persistent `ShardReader`,
  and only INT8 data/scales payload reads. No dequantization or expert kernel
  ran during route-cache replay.
- Core cache budgets were 4 GiB and 8 GiB.
- Grouped kernel used 8 real decode hidden rows and their captured routes and
  routing weights, with 10 CPU threads and two repetitions.

Raw evidence:

- `benchmarks/results/2026-07-22/stage7_7/real_routes_v3.json`
- `benchmarks/results/2026-07-22/stage7_7/route_union_replay_raw.json`
- `benchmarks/results/2026-07-22/stage7_7/grouped_kernel.json`
- `.cache/stage7_7/real_decode_hidden.pt` is intentionally not committed;
  its capture is represented by the grouped result hash.

## Gate 1: Route Union

Across synchronous decode metadata, the union compression was measurable:

| Batch | Union ratio |
|---:|---:|
| B=2 | 1.1902x |
| B=4 | 1.5185x |
| B=8 | 1.7492x |
| B=16 | 2.1690x |

This proves route overlap exists. It does not prove a bounded shared cache can
retain the union efficiently.

## Gate 2: Real Grouped Kernel

The grouped operator was exact against canonical batch-one execution for all
tested batches:

| Batch | Aggregate speedup | Exact | Max abs error |
|---:|---:|:---:|---:|
| 1 | 6.865x | yes | 0 |
| 2 | 1.409x | yes | 0 |
| 4 | 1.707x | yes | 0 |
| 8 | 1.908x | yes | 0 |

The formal B=4 threshold was `1.5x`; it passed at `1.707x`. This is a
routed-MoE operator gate, not a full Qwen generation scheduler benchmark.

## Gate 3: Shared-cache Union Replay

The raw replay results below compare the layer-synchronous union order with a
round-robin order using the same selected experts and the same cache budget.
`Loaded` is logical payload loaded and equals provider raw read bytes in every
cell.

| Schedule | Cache | Mode | Hit rate | Loaded | Reads | Budget | Leases |
|---|---:|---|---:|---:|---:|:---:|:---:|
| B=4 | 4 GiB | union | 57.73% | 32.422 GiB | 32.422 GiB | yes | released |
| B=4 | 4 GiB | round-robin | 72.48% | 32.049 GiB | 32.049 GiB | yes | released |
| B=4 | 8 GiB | union | 78.91% | 16.180 GiB | 16.180 GiB | yes | released |
| B=4 | 8 GiB | round-robin | 86.10% | 16.186 GiB | 16.186 GiB | yes | released |
| B=8 | 4 GiB | union | 8.06% | 122.448 GiB | 122.448 GiB | yes | released |
| B=8 | 4 GiB | round-robin | 56.39% | 101.597 GiB | 101.597 GiB | yes | released |
| B=8 | 8 GiB | union | 67.35% | 43.479 GiB | 43.479 GiB | yes | released |
| B=8 | 8 GiB | round-robin | 81.29% | 43.579 GiB | 43.579 GiB | yes | released |

The normalized union/round-robin loaded-byte ratios are:

| Batch | 4 GiB | 8 GiB |
|---:|---:|---:|
| B=4 | 1.0116x | 0.9996x |
| B=8 | 1.2052x | 0.9977x |

The 4 GiB union cells fail the requirement that normalized loaded bytes not
increase. B=8/4 GiB is the decisive failure: deduplicating the current layer
into a large union creates a working set too large for the cache and causes
thrashing. The 8 GiB cells are close to neutral, not a strong enough basis to
claim a general streaming scheduler win.

## Decision

| Gate | Result |
|---|---|
| Real route overlap | GO |
| B=4 grouped routed-MoE speedup and exactness | GO |
| Shared streaming cache budget/accounting | GO for invariants |
| Shared streaming normalized-read gate | **NO-GO** |
| Full Stage 7.7 scheduler | **Not implemented by design** |

The next implementation scope is resident-only fixed-cohort batching, reusing
the exact grouped operator and independent session state. Before streaming
batching is revisited, the runtime needs an explicitly cache-aware batching
policy, such as bounded union admission, route-aware cohort formation, or
layer ordering that limits working-set peaks. It must be re-evaluated against
the same raw replay contract; simply increasing batch size or duplicating
traces is not acceptable.

## Verification

The structured verifier checks the eight raw replay cells, exact grouped
outputs, cache budgets, lease release, demand accounting, trace identity, and
the explicit NO-GO reason. It is run after the final clean commit and push.

Every implementation and result record in this report is owned by `[Main Dev]`.

<!-- [Main Dev] -->

# Stage 7.7: Multi-request Batching Feasibility

**Owner:** Main Dev
**Status:** Feasibility gates complete; shared streaming scheduler NO-GO
**Scope:** Feasibility evidence first; scheduler implementation only after the
feasibility gates pass.
**Model:** Qwen3.6-35B-A3B text-only, canonical INT8 routed experts.

## 1. Objective and boundary

Stage 7.7 determines whether independently generated requests should share a
decoder step and an expert union. It optimizes aggregate decode throughput,
per-session token latency, normalized expert I/O, cache behavior, and fairness.
It does not use the single-request `2.1513 tok/s` number as its only goal:
batching can improve aggregate throughput while increasing one user's token
latency.

The primary workload is decode-only multi-request batching. All sessions may
have different absolute token positions, but prefill rows and decode rows are
not mixed in the first scheduler gate. Prefill batching is a secondary
measurement because Stage 7.6 already established the grouped-prefill path.

The stage explicitly does not include Vision, MTP, speculative decoding,
production OpenAI serving, INT4, dense Attention/DeltaNet rewrites, Python
generation-loop replacement, or simple Future-based I/O overlap.

## 2. Frozen protocol

The model revision, INT8 container manifest, runtime identity, tokenizer,
prompt manifest, CPU thread count, greedy sampling, and result schema are
recorded in every raw result. The default protocol is 10 CPU threads, 32
generated tokens, `B=1/2/4/8`, S1 LRU at 4 GiB and 8 GiB, one warmup plus three
workload-warm runs. Model-cold cells use three independent processes and
explicit model-page eviction.

The multi-request prompt manifest must contain at least 20 unique prompt IDs
covering Chinese, English, code, mathematics, knowledge, and locality-heavy
conversation. A route trace may not be copied to fill a larger batch. Every
trace and schedule is identified by SHA256.

Metrics have fixed definitions:

| Metric | Definition |
| --- | --- |
| aggregate tok/s | All emitted decode tokens divided by scheduler decode wall time |
| session TPOT | Time between visible tokens for one session, including scheduler wait |
| TTFT | Request arrival to first visible token; queue wait is reported separately |
| logical bytes/token | Provider demand-read bytes divided by emitted decode tokens |
| physical bytes/token | Process physical read bytes divided by emitted decode tokens |
| union ratio | `B * top_k / unique experts`, calculated per layer and step |
| fairness | Jain normalized-service index, max slowdown, max queue wait, starvation steps |

Warm results report logical provider reads and physical process reads separately;
warm page-cache reads may have zero physical process reads. Physical-read claims
are made only for model-cold cells.

## 3. Trace schema and route-union analysis

Do not change the meaning of the existing v2 trace loader. Add a v3
multi-session wrapper in `src/sparseflow/multirequest_trace.py` with these
fields:

```text
session_id, arrival_step, scheduler_step, request_token_index,
absolute_token_position, phase, layer, expert_ids, finished
```

Add:

```text
benchmarks/capture_stage7_7_routes.py
benchmarks/analyze_stage7_7_union.py
tests/test_multirequest_trace.py
benchmarks/manifests/stage7_7_multi_request_v1.jsonl
```

Capture real router output from independent sessions. Construct synchronized,
offset, locality-heavy, and diverse schedules for `B=2/4/8/16`. An offset is a
different decode position or arrival step, not a silent prefill/decode merge.

For every scheduler step and layer, report raw assignments, unique experts,
expert multiplicity histogram, union compression, pairwise Jaccard,
popularity entropy/Gini, reuse distance, cache hits/misses, evictions, and
projected 4/8 GiB loaded bytes. Tests must prove session boundaries cannot leak
into each other, union ordering is deterministic, v2 traces remain loadable,
and duplicate prompt/trace identities are rejected.

This phase is analysis only. It may show low overlap; that is evidence for a
NO-GO, not a reason to synthesize or duplicate traces.

## 4. Real-route grouped-kernel gate

Add:

```text
src/sparseflow/multirequest_moe.py
benchmarks/capture_stage7_7_hidden.py
benchmarks/bench_stage7_7_grouped.py
tests/test_multirequest_moe.py
```

Capture real BF16 hidden rows, selected experts, and routing weights for all
40 layers at representative decode positions. Store large fixtures under
`.cache/stage7_7/`; commit only their manifest, model hash, shapes, dtype, and
SHA256.

Compare, using identical hidden rows and routes:

1. B canonical batch-one calls.
2. The current fused multi-row operator.
3. The fused operator with persistent workspace.
4. A true expert-grouped MxK kernel.
5. Grouped execution with one provider get and one lease per union expert.

The current `native/moe_dispatch.cpp` operator accepts multiple rows, but its
expert loop still performs per-assignment dot products. It must not be called a
true grouped GEMM until that is measured and implemented. Persistent workspace
must reuse projected, contribution, activation-quantization, and output buffers
without changing the deterministic accumulation order.

The gate is: all routed-MoE outputs exact; B=4 routed aggregate throughput at
least `1.5x` the batch-one baseline; logical bytes no higher; no workspace
accounting gap; and every lease released. Failure stops scheduler work.

## 5. Shared-cache union replay

Add `benchmarks/simulate_stage7_7_streaming.py`. Replay the exact scheduler
order as `scheduler step -> layer 0..39 -> union expert admission -> grouped
compute -> release`. Compare a shared-provider round-robin baseline with the
layer-synchronous union schedule. Report assignment-level reuse separately from
provider-level cache hits so deduplication is not mistaken for a cache hit.

Run metadata replay for the full matrix and real pread replay for B=4/8 core
cells at 4/8 GiB warm and model-cold. Include union size, transient payloads,
pinned peak bytes, admission rejections, loaded bytes, physical reads, RSS,
and P50/P95 layer-step time.

The streaming gate requires no budget violation, no lease leak, exact demand
accounting, no failed prefetch, and normalized loaded bytes no higher than the
round-robin baseline. The decision is explicit:

| Result | Next scope |
| --- | --- |
| Kernel NO-GO | End Stage 7.7 with a NO-GO report |
| Kernel GO, streaming NO-GO | Implement resident scheduler only |
| Kernel and streaming GO | Implement resident and 4/8 GiB scheduler |

## 6. Minimal fixed-cohort scheduler

Only after the previous gates pass, add `src/sparseflow/multi_session.py` and
`benchmarks/run_stage7_7.py`. Keep scheduler logic out of the existing
single-session `text_runtime.py` except for narrow reusable hooks.

Required objects are `SessionRequest`, `SessionState`, `BatchCoordinator`, and
`BatchCohort`. A session owns its token IDs, attention mask, KV/DeltaNet cache
row, generated IDs, arrival metadata, latency samples, and finished state.

The first implementation is a fixed-cohort micro-batcher: collect up to
`max_batch_size`, wait at most `max_wait_ms`, run one batched prefill, then run
decode rows together. Each batch row is an independent KV/DeltaNet state.
Start with equal 32-token quotas; then test mixed `[8,16,24,32]` quotas. Finished
rows must be removed through the Transformers cache batch-select/reorder API,
or the implementation must stop and report that the cache API is insufficient.

The baseline uses the same model, provider, and cache but advances sessions
round-robin without cross-session grouped compute. Resident and streaming use
the same coordinator and correctness path; only expert storage/provider differs.
The first scheduler version does not admit new requests into an active cohort
and does not expose an API. Continuous admission and service scheduling remain
Stage 7.8 candidates.

Correctness has three separate gates:

| Gate | Requirement |
| --- | --- |
| Storage | Same schedule, resident vs streaming full logits/routes/IDs exact |
| Scheduler | Against independent sessions, IDs/routes/greedy divergence exact |
| Numeric | Batched logits report max error, argmax equality, top-k overlap, and KL |

## 7. Formal matrix and acceptance

Add:

```text
benchmarks/summarize_stage7_7.py
benchmarks/verify_stage7_7.py
docs/results/qwen36_stage7_7_multi_request_<date>.md
```

Formal cells are `B=1/2/4/8 × resident/S1-4G/S1-8G`; B=16 is stress-only.
The latency profile requires B=2 aggregate `>=1.2x` and P95 TPOT `<=1.5x` solo.
The throughput profile requires B=4 aggregate `>=1.3x`, P95 TPOT `<=3x` solo,
Jain index `>=0.95`, and no starvation. All profiles require normalized
logical bytes not to increase; model-cold cells additionally require physical
bytes and three independent replicates.

The verifier must check model/container/manifest hashes, clean commit, result
schema, output exactness, per-session routes, cache budget, demand accounting,
leases, fairness, and all stated GO gates. It must emit `all_pass=true` or a
structured NO-GO reason.

## 8. Deliverable and execution order

Raw results live under `benchmarks/results/<date>/stage7_7/`, the analysis lives
in the result document above, and `.handoff.md` receives only the status,
commit, decision, and result-document link. Every new source/doc/result carries
`[Main Dev]` ownership.

The execution order is strictly 7.7.0 -> 7.7.1 -> 7.7.2 -> 7.7.3 -> 7.7.4
-> 7.7.5. Each step gets its own tests and commit. No later step may hide a
failed earlier gate by changing the baseline, reducing output length, copying
traces, or weakening correctness requirements.

## 9. Measured completion (2026-07-22)

The feasibility portion completed with 20 independent real sessions. Route
union existed (`B=4: 1.5185x`, `B=8: 1.7492x`), and the real hidden grouped
operator was exact with `B=4: 1.707x` aggregate speedup. The raw INT8 cache
replay then failed the normalized-read gate at 4 GiB: union loaded `32.422`
GiB versus round-robin `32.049` GiB for B=4, and `122.448` GiB versus
`101.597` GiB for B=8. The 8 GiB cells were approximately neutral.

Therefore the measured decision is `resident-only-scheduler`. Do not implement
the streaming fixed-cohort scheduler under this stage result. Full evidence is
in [`docs/results/qwen36_stage7_7_multi_request_20260722.md`](results/qwen36_stage7_7_multi_request_20260722.md).

<!-- [Main Dev] -->

<!-- [Main Dev] -->

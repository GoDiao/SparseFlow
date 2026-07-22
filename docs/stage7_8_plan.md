# Stage 7.8: Native Grouped MoE Acceleration

**Owner:** Main Dev
**Status:** Operator gate complete; resident experimental path complete; streaming remains gated

## Objective

Convert the real cross-request expert overlap measured in Stage 7.7 into a
native per-expert grouped execution path. The stage keeps the existing
canonical and fused operators as baselines, and does not make grouped
dispatch the default until paired full-runtime measurements justify it.

## Execution order

1. Freeze the Stage 7.7 hidden/routes fixture and measure expert-group
   multiplicity `M=1..32`.
2. Build a stable native `GroupPlan` containing expert counts, prefix offsets,
   row indices, assignment indices, and deterministic token reduce order.
3. Implement a grouped W8A8 operator. `M=1` uses the existing VNNI dot path;
   repeated rows are computed under one expert/output-channel task.
4. Reuse plan, quantization, projection, activation, contribution, and output
   buffers through `GroupedMoEWorkspace`.
5. Expose `native_dispatch=grouped` while preserving `legacy`, `fused`, and
   `hybrid` behavior.
6. Run a fixed resident cohort with equal-length prompts and independent
   session state. Compare grouped batch with fused batch and with independent
   sessions.
7. Replay a bounded cache-aware sub-cohort policy against Stage 7.7
   round-robin using raw INT8 reads.
8. Write structured summary/verifier output and record explicit GO/NO-GO gates.

## Correctness contracts

- Grouped output must be exact against the old fused operator for identical
  hidden rows, routes, and weights.
- The final routing-weighted reduce keeps original token top-k order.
- B=1 has a fast path and must not regress by more than 3%.
- Every native cache lease is released on success and exception.
- Resident grouped batch must preserve routes, generated IDs, and text.
- Full batched logits are reported separately from behavior exactness because
  ATen batch GEMM and independent batch-one GEMM can use different reduction
  orders.
- Streaming replay must respect byte budget and release all leases.

## Acceptance

The operator target is B=4 grouped speedup `>=1.95x` against canonical
batch-one execution, with B=1 non-regression and exact output. Resident
cohort throughput is compared with independent sessions and the old fused
batch. Streaming is accepted only when normalized loaded bytes are no higher
than the Stage 7.7 round-robin baseline in every core cell.

Raw results live under
`benchmarks/results/2026-07-22/stage7_8/`; the result document and verifier
are the authoritative completion records.

<!-- [Main Dev] -->

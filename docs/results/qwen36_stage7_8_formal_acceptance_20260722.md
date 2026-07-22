# Qwen3.6 Stage 7.8 Formal Acceptance

**Executor:** `[Main Dev]`
**Date:** 2026-07-22
**Formal baseline commit:** `3f935f24025e2b17af61ab6e92a099947ceb188f`
**Model:** `Qwen/Qwen3.6-35B-A3B`

## Protocol

This is the long-generation resident gate requested after the initial Stage
7.8 pilot. It uses one process with both memory-native INT8 resident runtimes
loaded and executes, for each batch size, three repetitions of:

```text
A = grouped
B = hybrid/fused
B = hybrid/fused
A = grouped
```

The cohort sizes are B=1, B=4, and B=8. Every generation produces 32 tokens.
Each dispatch therefore contributes six observations and 186 decode-token
latency samples. The eight prompts are semantically different and are padded
by the harness to one shared 32-token chat-template length.

This is the equivalent long-generation quality gate, not a 60-question choice
matrix. The earlier independent-session full-logit difference remains a
separate dense ATen reduction-order observation and is not silently relabeled
as exact here.

## Results

| Batch | Grouped tok/s | Hybrid/fused tok/s | Grouped P50/P95 | Hybrid/fused P50/P95 |
|---:|---:|---:|---:|---:|
| 1 | 1.9224 | 1.8982 | 0.5231 / 0.6263 s | 0.5206 / 0.5916 s |
| 4 | 4.2570 | 3.8463 | 0.9116 / 1.1395 s | 1.0342 / 1.1939 s |
| 8 | 5.5650 | 4.8520 | 1.4955 / 2.0356 s | 1.5922 / 2.0309 s |

All six A/B pairs for every batch passed:

- full captured logits exact;
- logit fingerprints exact;
- routes exact;
- generated IDs, text, and argmax exact;
- repeated grouped and fused observations exact;
- runtime identity exact;
- ABBA executed in one process.

The formal harness itself reported `all_gates_pass=true`. It captured the
clean provenance commit before writing the result artifact:
`3f935f24025e2b17af61ab6e92a099947ceb188f`.

## Separate Stage 7.8 decisions

### Native operator

GO. The earlier real hidden/routes operator gate remains valid: B=1 had no
more than 3% regression, B=4 exceeded the 1.95x canonical target, and grouped
output was exact against fused output. This is real per-expert grouped task
execution with weight/task reuse; it is not yet a matrixized grouped GEMM.

### Resident runtime

GO for continued opt-in experimentation. The formal long-generation gate
confirms stable behavior and exact grouped/fused execution at B=1/4/8. It does
not make grouped dispatch the default: the protocol is a fixed equal-length
cohort, not a dynamic serving scheduler, and the earlier independent-session
full-logit reduction-order difference remains documented.

### Streaming policy

Still NO-GO / gated. The existing cache-aware sub-cohort replay passed only
B=8/8 GiB and had small read amplification in the other three core cells.
The resident formal gate does not change that streaming conclusion.

## Raw evidence

- `benchmarks/results/2026-07-22/stage7_8/formal_resident_abba.json`
- `benchmarks/results/2026-07-22/stage7_8/grouped_operator.json`
- `benchmarks/results/2026-07-22/stage7_8/streaming_subcohort_ratio025.json`

<!-- [Main Dev] -->

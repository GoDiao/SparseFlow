# Stage 7.9: Qwen3.6 Public Alpha & Release Hardening

**Owner:** Main Dev  
**Benchmark owner:** Benchmark  
**Status:** complete

## Goal

An unfamiliar user on a supported Linux CPU can inspect the Qwen3.6 checkpoint,
prepare a resumable INT8 container, run a complete text generation, and obtain
the same runtime identity and resource accounting used by the formal report.

Stage 7.9 does not introduce a new expert kernel or a shared streaming
scheduler. It freezes the measured Qwen path and makes its boundaries honest.

## Presets

| Preset | Runtime | Status |
|---|---|---|
| `stable` | INT8 native resident hybrid | stable |
| `low-memory` | INT8 native single-request streaming + S1 LRU, prefetch off | stable low-memory |
| `experimental-batch` | INT8 native resident grouped fixed cohort | opt-in experimental |

Pure fused decode is diagnostics-only. Shared streaming batching/subcohort is
disabled because its Stage 7.8 normalized-read gate did not pass.

## Implemented Release Surface

- `sparseflow preset [name]` exposes the frozen preset contract.
- `sparseflow doctor` performs read-only checks before runtime construction.
- `sparseflow prepare-int8` converts/resumes all routed experts and creates
  offline execution row sums.
- `sparseflow run` maps public presets to the existing memory-native runtime.
- `README.md` contains the supported-source-install Quick Start.

## Verification Ladder

1. Smoke: 32 generated tokens, full logits compared in memory, fingerprints,
   routes, and IDs saved.
2. Standard: 128 tokens, fingerprints/routes/IDs saved, periodic full logits
   compared in memory.
3. Endurance: 512 tokens, IDs/routes/cache accounting saved, periodic
   fingerprints only.
4. Conversation: multiple turns, KV/DeltaNet state and turn boundaries.
5. Quality: the frozen 60-question HellaSwag/ARC/MMLU formal manifest.

Full logits are never serialized for every step of the 128/512-token runs.

## Formal Matrix

The result runner must record resident warm, 4 GiB streaming warm, 8 GiB
streaming warm, 4 GiB model-cold with three independent processes, and grouped
B=1/4/8 ABBA. Every result records TTFT, prefill/decode throughput, P50/P95,
RSS, logical/physical read bytes, cache counters, model/container identity,
CPU, SSD, threads, and clean commit provenance.

The regression gate is no more than 5% degradation from the relevant Stage
7.6/7.8 formal baseline. Resident/streaming with the same arithmetic must be
exact; BF16/INT8 is a separate quality comparison; grouped/independent batch
comparisons report reduction-order error separately.

## Known Limitation

The Public Alpha does not claim a shared streaming batching policy. A future
policy can return only after offline replay passes every 4/8 GiB read-amplification
gate and then a new runtime correctness gate.

## Final Audit

The CLI/runtime and RAM-admission closure fixes were committed in clean code
commit `36aa619901561cf4148dbf8dfaadb8f73b058d53`. The Smoke, Standard, and
Endurance validation artifacts were regenerated from that commit and passed
all exactness, cache-budget, compactness, and identity gates. Experiment-host
Doctor/RSS acceptance is recorded in the Stage 7.9 closure artifacts.

<!-- [Main Dev] -->

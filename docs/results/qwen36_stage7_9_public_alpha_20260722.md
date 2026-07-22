# Qwen3.6 Stage 7.9 Public Alpha & Release Hardening

**Executor:** `[Main Dev]`
**Benchmark executor:** `[Benchmark]`
**Code baseline:** `d3213742c098e4376a2b6123a7e40d770917f04d`
**Model:** `Qwen/Qwen3.6-35B-A3B`

## Decision

Stage 7.9 is **Public Alpha ready** under an explicit runtime boundary:

| Path | Status |
|---|---|
| INT8 native resident hybrid | Stable baseline |
| INT8 native single-request streaming S1 LRU | Stable low-memory |
| INT8 native resident grouped fixed cohort | Experimental opt-in |
| Shared streaming batching/subcohort | Disabled, known limitation |
| Pure fused decode | Diagnostics only |

This is a Qwen3.6 text-only release. It is not a production serving scheduler,
vision runtime, MTP runtime, or multi-model release.

## Release Surface

The following user workflow is available:

```text
install -> doctor -> inspect/plan -> prepare-int8 -> run
```

New entry points:

```text
sparseflow preset [name]
sparseflow doctor <model> --preset <name> --int8-container <dir>
sparseflow prepare-int8 <model> --output <dir>
sparseflow run <model> --preset stable|low-memory|experimental-batch ...
```

`doctor` is header-only for model payloads and checks the model, 40-layer INT8
container, offline row sums, disk reserve, CPU ISA, and optional native build.
The native extension was compiled successfully on the experiment host with
AVX-512 VNNI. INT8 conversion remains resumable and does not overwrite BF16
weights.

## New Validation

The Benchmark ladder compared the same INT8 native hybrid arithmetic in resident
and single-request S1 streaming modes. Full logits were compared only in memory;
the result files contain fingerprints, routes, IDs, cache accounting, and
resource metadata.

| Level | Tokens | Exact | Selected full-logit steps | Result |
|---|---:|---:|---:|---|
| Smoke | 32 | yes | every step | `validation_smoke.json` |
| Standard | 128 | yes | step 0, every 16 | `validation_standard.json` |
| Endurance | 512 | yes | step 0, every 32 | `validation_endurance.json` |

Observed single-prompt decode measurements on this host:

| Level | Resident decode | S1 streaming decode | S1 hit rate | S1 loaded bytes |
|---|---:|---:|---:|---:|
| 32 | 16.699 s | 23.669 s | 51.85% | 17.47 GiB |
| 128 | 65.092 s | 96.128 s | 57.84% | 53.99 GiB |
| 512 | 267.304 s | 391.921 s | 63.98% | 176.16 GiB |

The 512-token run retained a 3.998 GiB cache and released all leases. All
three validation artifacts passed `verify_stage7_9.py`.

## Inherited Formal Evidence

The 60-question HellaSwag/ARC/MMLU quality matrix was completed in Stage 7.5.6
and remains the frozen quality evidence for the same canonical INT8 weights:

- BF16 resident: 60 questions;
- W8A8 native resident: 60 questions;
- W8A8 native streaming: 60 questions;
- native resident and streaming matched every prediction and choice total;
- raw predictions agreed on 59/60 questions and character-normalized results
  agreed on 60/60.

Stage 7.8 additionally verified resident grouped/hybrid behavior with B=1/4/8,
32-token ABBA repetitions, full logits, routes, IDs, and text exact. Grouped
remains opt-in because it is a fixed equal-length cohort path, not a dynamic
serving scheduler. Shared streaming batching remains NO-GO and is not enabled
by this release.

## Reproduction

See [`README.md`](../../README.md) for the complete clean-environment Quick
Start. The authoritative JSON summary is:

```text
benchmarks/results/2026-07-22/stage7_9/summary.json
```

Raw validation and verification files are in the same directory. Model
payloads and INT8 containers remain local/cache artifacts and are not committed.

<!-- [Main Dev] -->

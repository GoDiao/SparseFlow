# SparseFlow Long-Term Architecture Principles

**Decision owner:** `[Board]`  
**Recorded:** 2026-07-22  
**Current priority:** complete and release the Qwen3.6 acceleration path before
generalizing it to additional model families.

## Product identity

SparseFlow is intended to become a model-extensible, tiered MoE acceleration
backend. It should integrate with an existing inference framework and own the
parts where sparse expert execution needs explicit control:

- expert storage formats and tensor views;
- RAM, SSD, and future GPU tier planning;
- expert cache, admission, and I/O accounting;
- route-aware dispatch and grouped execution;
- quantized CPU/GPU expert kernels;
- correctness, performance, and storage telemetry.

SparseFlow is not currently intended to reimplement every model's tokenizer,
attention, KV cache, recurrent state, sampling, and generation loop. Those
parts may remain in Transformers/PyTorch or another host runtime until profiling
shows that ownership is required for correctness or material performance.

## Qwen-first execution rule

The immediate engineering goal is a strong Qwen3.6-35B-A3B implementation.
Generic abstractions must not delay that work or replace measured Qwen results
with untested interface design.

Current Qwen-specific code is acceptable when it represents real architecture
semantics, including fused expert tensor slicing, router behavior, Gated
DeltaNet integration, model loading, and fixed-cohort validation. New core
storage and kernel code should nevertheless avoid unnecessary Qwen assumptions
when the generic boundary is already clear.

Generalization begins only after the Qwen path is release-capable. A second
structurally different MoE model is then required to validate any proposed
adapter contract. Supporting one model plus synthetic tests is not sufficient
evidence that an abstraction is model-independent.

## Intended boundaries

```text
Host inference framework
  tokenizer, dense/attention layers, KV or recurrent state, generation
                         |
Framework integration   | model module replacement and runtime hooks
                         v
Model adapter
  tensor classification, expert layout, router semantics, shared experts,
  activation and shape metadata
                         |
                         v
SparseFlow core
  inspect/plan, container, provider/cache, tier policy, dispatcher, telemetry
                         |
                         v
Native backends
  W8A8/INT4 CPU kernels, grouped expert execution, future GPU expert tier
```

Model-specific tensor names and architecture semantics belong in an adapter or
framework integration. Cache accounting, storage tiers, canonical expert
views, grouped dispatch contracts, and ISA-specific kernel selection belong in
the reusable core.

## Evolution sequence

1. Finish Qwen3.6 correctness, acceleration, resource planning, and a usable
   release path.
2. Stabilize only the interfaces that the measured Qwen implementation has
   proven necessary: expert specification, layout, provider, dispatcher, and
   kernel contracts.
3. Add a second real MoE model and use its differences to validate or revise
   the adapter boundary.
4. Add further framework integrations without duplicating the expert backend.
5. Take ownership of dense layers, session state, or the generation loop only
   when profiling or host-framework limitations provide a concrete gate.

## Design constraints

- Do not build a complete inference framework merely to claim multi-model
  support.
- Do not call Qwen-specific code generic by renaming it.
- Do not introduce an adapter abstraction without a concrete second-model
  validation plan.
- Keep resident and streaming execution on the same arithmetic contracts where
  possible, so storage correctness remains independently testable.
- Preserve benchmark manifests, runtime identities, hashes, cache accounting,
  and explicit GO/NO-GO decisions as the backend evolves.
- Prefer narrow host-runtime hooks over forks of Transformers or other upstream
  frameworks.
- Allow profile-gated native ownership to grow incrementally: expert backend,
  decoder-layer boundary, session runtime, then full generation only if needed.

## Relationship to Colibri

Colibri vertically integrates a model-specific C inference engine and its
storage acceleration. SparseFlow follows a different long-term strategy:
retain the same depth of expert-storage and kernel optimization while keeping
the model boundary extensible and reusing host inference frameworks where they
remain adequate.

Stage 7.8's Qwen-specific grouped operator and fixed-cohort harness are
consistent with this direction. They establish the measured implementation
that later abstractions must preserve; they do not require immediate
multi-model refactoring.

<!-- [Board] -->

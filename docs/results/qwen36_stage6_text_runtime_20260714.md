# Qwen3.6 Stage 6 text-only runtime correctness

> [Main Dev] This report records the full text-only integration of Qwen3.6
> with SparseFlow's ExpertCache path and the exact correctness gate used for
> resident versus streaming execution.

## Runtime boundary

The Python reference composes Transformers' embedding, attention, Gated
DeltaNet, normalization, residual, shared expert, language-model head, and KV
cache with SparseFlow's routed-expert storage path:

```text
chat template
  -> prefill
  -> attention / DeltaNet
  -> router
  -> ExpertCache / ShardReader routed experts
  -> logits + past_key_values
  -> one-token KV-cache decode
```

The public generation and correctness commands are:

```bash
PYTHONPATH=src python -m sparseflow text-generate <model> \
  --mode streaming --dtype bf16 --prompt 'test' --max-new-tokens 4

PYTHONPATH=src python -m sparseflow text-check <model> \
  --dtype bf16 --prompt 'test' --max-new-tokens 4 \
  --cache-slots 16 --json
```

`text-check` loads resident and streaming models sequentially, compares input
IDs, generated IDs, decoded text, and the SHA-256 of the full 248,320-way BF16
next-token logits after prefill and every decode step. It exits non-zero if
any comparison fails.

## Arithmetic-policy finding

Transformers selected `grouped_mm` as its default CPU expert implementation.
SparseFlow's current streaming kernel is eager per-expert matmul plus
`index_add_`. Both generated the same greedy tokens, but their BF16 logits
were not byte-identical; on the same model object, the final prefill logits
had a maximum absolute difference of `0.5703125` after 40 layers.

This is expected from different dispatch and accumulation kernels and is not
a valid storage-policy comparison. The Stage 6 gate therefore fixes both
resident and streaming to Transformers-compatible `eager` arithmetic. Main
Dev also aligned SparseFlow's token dispatch order with the official eager
implementation: `(top-k position, token index)` within each expert.

Benchmark may still measure grouped-mm as a separate performance backend, but
must not call grouped-mm versus eager drift an SSD/cache correctness error.

## Exact four-token result

Hardware: Intel Xeon Gold 6248R, 125 GiB RAM, two Tesla V100S 32GB GPUs; the
model was kept on CPU. Prompt `test` encoded to 11 input tokens.

Resident eager and SparseFlow streaming eager both produced:

```text
token IDs   [8160, 579, 264, 7047]
text        "Here's a thinking"
```

All four full-vocabulary logit fingerprints matched exactly:

```text
prefill   91d94b782fe445f01d2168dec2fc678af5c55f3ff75dc7d49afd453c6bfb6cc4
decode 1  3a9844455d91fb6ea4d7fe9a66581681e78b1264718c443edc7c24246a19c883
decode 2  6f7171a078b52f308324623728f576312e1395c1183522c43dfe1c1a4eefcb8f
decode 3  5ffc284577045869e4d4110a5aa067028037cefa5ba661eae2e47ca77ad7b60c
```

The 16-slot/layer streaming cache recorded:

```text
requests             2591
hits                  300
misses               2291
hit rate           11.58%
evictions            1651
loaded bytes   14,413,725,696
cached bytes    4,026,531,840
cache entries          640
```

This check covers prefill and three consecutive one-token forwards using the
returned and updated Transformers KV cache.

## Prefetch correctness

A second check enabled two prefetch workers with `coalesce_gap=0`. Resident
and streaming again had identical token IDs and full logits for prefill plus
one decode step.

```text
prefetch batches       80
completed              80
failed                  0
submitted experts    1851
coalesced ranges      3186
logical ranges        3702
physical bytes  11,645,485,056
useful bytes    11,645,485,056
wasted bytes              0
```

The small cache caused prefetched entries to be evicted and reloaded, so this
run is a correctness/integration result rather than evidence that the current
prefetch policy is optimal.

## Implemented components

- `Qwen36TextRuntime`: chat encoding, prefill, decode, KV-cache handoff, and
  greedy generation.
- `SparseFlowQwenExperts`: all 40 routed-expert modules backed by
  `ExpertCache`, `ShardReader`, decoded tensor views, and optional prefetch.
- `text-generate`: resident or streaming generation with explicit expert
  implementation reporting.
- `text-check`: sequential resident/streaming gate with full-logit hashes and
  failure exit status.
- Tests cover the generation loop, comparison semantics, CLI behavior, and
  official eager dispatch ordering.

## Remaining boundary

This is the complete Stage 6 Python correctness/reference runtime. It still
asks Transformers to load the full checkpoint before routed modules are
replaced, so it does not yet achieve Colibri-style low peak memory. The next
memory milestone is a meta-device/custom state-dict loader that materializes
dense weights only and leaves routed experts on disk from process start.

Vision, MTP, INT8/INT4 kernels, native C++/Rust hot paths, and production
serving remain outside this text-only milestone.

Raw measurements are in
[`qwen36_stage6_text_runtime_20260714.json`](qwen36_stage6_text_runtime_20260714.json).

[Main Dev]

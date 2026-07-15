# Qwen3.6 Stage 7.1 memory-native acceptance

> Stage 7.1 proves that Qwen3.6 text generation can start and complete without
> ever materializing the complete 60 GiB routed-expert tensors. [Main Dev]

## Implementation

The new path does not call the full-model Transformers `from_pretrained`:

```text
safetensors headers
  -> MemoryLoadPlan
  -> Qwen3_5MoeForCausalLM on meta device
  -> install SparseFlowQwenExperts in all 40 layers
  -> materialize text-resident tensors one at a time
  -> prefill / ExpertCache streaming / KV decode
```

The public commands are:

```bash
sparseflow native-plan <model>
sparseflow native-meta <model>
sparseflow native-load <model> --dtype bf16

sparseflow text-generate <model> \
  --mode streaming --load-mode memory-native \
  --dtype bf16 --prompt test --max-new-tokens 4 --cache-slots 16

sparseflow text-check <model> \
  --streaming-loader memory-native \
  --dtype bf16 --prompt test --max-new-tokens 4 --cache-slots 16
```

## Checkpoint partition

Header-only planning classified all 1045 checkpoint tensors:

```text
text resident       613 tensors    4,896,711,936 bytes  (4.560 GiB)
language experts     80 tensors   64,424,509,440 bytes  (60.000 GiB)
MTP                  19 tensors    1,689,281,536 bytes  (1.573 GiB)
vision              333 tensors      893,142,496 bytes  (0.832 GiB)
planning payload read                                      0 bytes
```

The text-only causal model has exactly 613 state tensors after expert
replacement. Before selective loading all 613 parameters and two derived
rotary buffers were meta tensors; routed-expert Parameter count was zero.
[Main Dev]

## Selective-load audit

The loader called `safe_open.get_tensor` only for resident entries and
materialized one tensor at a time:

```text
resident source bytes read       4,896,711,936
resident bytes materialized      4,896,711,936
expert payload bytes during init             0
expert ShardReader calls after init          0
expert ShardReader bytes after init          0
remaining meta parameters                    0
remaining meta buffers                       0
routed-expert parameters                     0
```

MTP and vision are excluded by using `Qwen3_5MoeForCausalLM`; checkpoint keys
are remapped from `model.language_model.*` to the text model's `model.*`.
[Main Dev]

## Standalone memory result

A fresh process generated four tokens with 16 expert slots per layer:

```text
prompt                         test
generated IDs                  [8160, 579, 264, 7047]
text                           "Here's a thinking"
RSS before materialize         920,580,096 bytes   (0.857 GiB)
RSS after materialize        5,754,114,048 bytes   (5.359 GiB)
process peak after load      6,833,156,096 bytes   (6.364 GiB)
RSS before prefill           5,757,333,504 bytes   (5.362 GiB)
RSS after generation        9,895,874,560 bytes   (9.216 GiB)
whole-process peak RSS      9,895,874,560 bytes   (9.216 GiB)
```

The result is a functional warm-page-cache sample, not a formal cold-I/O
throughput benchmark. It nevertheless proves that the process never needed a
60 GiB expert tensor allocation. Linux filesystem page cache is outside the
process RSS and is not claimed as resident model memory. [Main Dev]

## Correctness

Resident eager and memory-native streaming produced exact full-vocabulary
BF16 logits for prefill and three KV-cache decode forwards. Every step had
shape `[1, 248320]` and matching SHA-256:

```text
prefill   91d94b782fe445f01d2168dec2fc678af5c55f3ff75dc7d49afd453c6bfb6cc4
decode 1  3a9844455d91fb6ea4d7fe9a66581681e78b1264718c443edc7c24246a19c883
decode 2  6f7171a078b52f308324623728f576312e1395c1183522c43dfe1c1a4eefcb8f
decode 3  5ffc284577045869e4d4110a5aa067028037cefa5ba661eae2e47ca77ad7b60c
```

The two-worker prefetch path also matched resident logits exactly: 80 batches
completed, zero failed, and `coalesce_gap=0` produced zero wasted bytes.
[Main Dev]

## Cache observation

The standalone four-token run recorded:

```text
requests           2591
hits                300
misses             2291
hit rate          11.58%
evictions          1651
logical bytes loaded  14,413,725,696
cached bytes           4,026,531,840
entries                         640
```

Stage 7.1 is a capacity and correctness milestone. Cache/prefetch efficiency,
cold-I/O controls, INT8, and native kernels remain Stage 7.2+ work. [Main Dev]

Raw summarized evidence is in
[`qwen36_stage7_1_memory_native_20260715.json`](qwen36_stage7_1_memory_native_20260715.json).

[Main Dev]

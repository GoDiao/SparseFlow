# Qwen3.6 Stage 7.2 C3-R / C3-S same-kernel acceptance

> Stage 7.2 proves that SparseFlow Resident (C3-R) and SparseFlow Streaming
> (C3-S) execute the complete text-only Qwen3.6 path with the same runtime,
> routed dispatch, BF16 expert kernel, attention/DeltaNet, KV cache, and greedy
> loop. Only the routed-expert storage provider differs. [Main Dev]

## Runtime boundary

Both backends now execute this path:

```text
Qwen3_5MoeForCausalLM built on meta
  -> materialize the same 4.560 GiB text-resident tensors
  -> SparseFlowQwenExperts in all 40 layers
  -> run_routed_experts
  -> run_expert_kernel
  -> Transformers attention / DeltaNet / KV cache
  -> SparseFlow greedy loop
```

The recorded identity is identical for C3-R and C3-S:

```text
runtime_id       qwen36-text-memory-native-v1
expert_module    SparseFlowQwenExperts-v1
dispatch_id      qwen36-topk-index-add-v1
kernel_id        bf16-linear-silu-linear-eager-v1
```

C3-R uses `ResidentExpertProvider`. It reads the 80 fused routed tensors once,
keeps 60 GiB in writable backing buffers, and exposes zero-copy expert views.
C3-S uses `StreamingExpertProvider`, the formal `ExpertCache`, and exact
expert slices from `ExpertLocator`. [Main Dev]

## Commands

```bash
sparseflow text-generate <model> \
  --expert-backend sparseflow-resident \
  --dtype bf16 --prompt test --max-new-tokens 4

sparseflow text-generate <model> \
  --expert-backend sparseflow-streaming \
  --dtype bf16 --prompt test --max-new-tokens 4 --cache-slots 16

sparseflow runtime-check <model> \
  --dtype bf16 --prompt test --max-new-tokens 4 --cache-slots 16
```

`--expert-backend` selects the memory-native loader automatically. The older
`--mode` and `--load-mode` options remain available for Stage 6/7.1
compatibility. [Main Dev]

## Real-model acceptance

The real checkpoint was
`/root/workspace/SparceFlow/model/Qwen3.6-35B-A3B`. The chat template expanded
the prompt `test` to 11 input tokens. Both backends produced:

```text
generated IDs   [8160, 579, 264, 7047]
text            "Here's a thinking"
route records   160 (40 layers x prefill/three decode forwards)
```

Every full-vocabulary BF16 logit fingerprint matched exactly:

```text
prefill   91d94b782fe445f01d2168dec2fc678af5c55f3ff75dc7d49afd453c6bfb6cc4
decode 1  3a9844455d91fb6ea4d7fe9a66581681e78b1264718c443edc7c24246a19c883
decode 2  6f7171a078b52f308324623728f576312e1395c1183522c43dfe1c1a4eefcb8f
decode 3  5ffc284577045869e4d4110a5aa067028037cefa5ba661eae2e47ca77ad7b60c
```

All 160 selected-expert route SHA-256 records also matched. Input IDs,
generated IDs, token count, and decoded text were exact. [Main Dev]

Three independent two-token gates expanded the route coverage. Each command
started fresh C3-R and C3-S runtimes so no KV state or ExpertCache entry was
shared across prompts:

| Workload | Input tokens | Route records | C3-S logical reads | Result |
|---|---:|---:|---:|---|
| Chinese explanation | 17 | 80 | 17,018,388,480 bytes | exact |
| Python completion | 21 | 80 | 18,522,046,464 bytes | exact |
| Math question | 20 | 80 | 16,766,730,240 bytes | exact |

For all three, complete logits, routes, IDs, text, runtime identity, and every
storage invariant matched. The two-token limit produced the truncated text
`"Here's"` in both backends; these runs are correctness coverage, not quality
evaluation. [Main Dev]

## Storage invariants

C3-R:

```text
layers                            40
experts per layer                256
logical resident experts      10,240
resident fused buffers            80
resident routed bytes     64,424,509,440 (60.000 GiB)
preload calls                      80
preload bytes              64,424,509,440
generation expert calls             0
generation expert bytes             0
```

C3-S with 16 slots per layer:

```text
initial expert calls                0
initial expert bytes                0
generation expert calls         4,582
generation expert bytes    14,413,725,696 (13.423 GiB)
cache requests                  2,591
cache hits                        300
cache misses                    2,291
hit rate                       11.58%
evictions                       1,651
final cache entries               640
final cached bytes        4,026,531,840 (3.750 GiB)
```

These are SparseFlow logical `pread` counters. They are not a claim about
physical NVMe traffic because the Linux page cache was not flushed. [Main Dev]

## Memory observations

Fresh standalone one-token smoke processes recorded:

```text
C3-R peak RSS   71,257,714,688 bytes (66.364 GiB)
C3-S peak RSS    9,883,435,008 bytes  (9.205 GiB)
```

The combined `runtime-check` runs C3-R and C3-S sequentially in one process.
Linux `ru_maxrss` is a lifetime high-water mark, so the C3-S section of that
combined JSON inherits the earlier C3-R peak. Use the standalone C3-S result
above or the Stage 7.1 fresh-process result for streaming RSS. [Main Dev]

## Timing scope

The warm-page-cache acceptance run observed:

```text
C3-R load       169.610 s (resident expert preload 143.770 s)
C3-R prefill      3.207 s
C3-R decode       1.265 s total for three forwards

C3-S load         2.034 s
C3-S prefill     34.643 s
C3-S decode       5.236 s total for three forwards
```

These timings validate executable boundaries and make the storage-policy cost
visible, but they are not the formal Stage 7.4 benchmark. CPU thread settings,
cold/warm cache state, repeated samples, and workload manifests were not
controlled here. [Main Dev]

## Automated validation

```text
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
Ran 35 tests
OK
```

The suite covers provider protocol conformance, fused resident preload,
post-preload zero I/O, exact resident/streaming tensor views, lifecycle errors,
shared module dispatch, route audit, runtime invariants, and CLI status.
[Main Dev]

Raw evidence is in
[`qwen36_stage7_2_c3_runtime_20260715.json`](qwen36_stage7_2_c3_runtime_20260715.json).
Additional prompt evidence is in
[`qwen36_stage7_2_c3_runtime_zh_20260715.json`](qwen36_stage7_2_c3_runtime_zh_20260715.json),
[`qwen36_stage7_2_c3_runtime_code_20260715.json`](qwen36_stage7_2_c3_runtime_code_20260715.json),
and
[`qwen36_stage7_2_c3_runtime_math_20260715.json`](qwen36_stage7_2_c3_runtime_math_20260715.json).

[Main Dev]

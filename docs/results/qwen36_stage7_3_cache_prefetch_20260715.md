# Qwen3.6 Stage 7.3 telemetry, cache, and prefetch acceptance

> Stage 7.3 turns the Stage 7.2 same-kernel runtime into an observable and
> policy-driven storage runtime. S0-S4 now share the same model/kernel and vary
> only cache admission, eviction, heat, and prefetch behavior. [Main Dev]

## Delivered runtime

The streaming runtime now exposes five internal attribution variants:

| ID | Cache policy | Prefetch policy |
|---|---|---|
| S0 | no admission | none |
| S1 | global/per-layer LRU | none |
| S2 | LRU plus bounded hot tier | none |
| S3 | heat decay, hysteresis, phase-aware admission | none |
| S4 | S3 | current-route async plus bounded stable-route prediction |

`ExpertCache` remains responsible for raw payload ownership and exact byte
budgets. `CachePolicy` now owns admission, hot protection, heat decay,
hysteresis, and victim selection. The default LRU behavior remains backward
compatible with Stage 7.2. [Main Dev]

S3 uses these development parameters:

```text
heat decay                    0.90 / forward
promotion threshold           3.0
demotion threshold            1.0
hot swap margin               1.0
hot tier                      25% of cache entries
prefill admission             second touch or hot
decode admission              immediate
```

A 36-candidate trace search found no fixed heat configuration that won at
every RAM budget. Higher decay improved 4/8 GiB by less than about 1% while
substantially hurting 1/2 GiB, so the more robust small-budget configuration
was retained. [Main Dev]

## Telemetry

`text-generate` can now emit summary or layer-level telemetry:

```bash
sparseflow text-generate <model> \
  --expert-backend sparseflow-streaming \
  --cache-bytes 4GiB \
  --cache-policy heat \
  --prefetch-policy previous-token \
  --prefetch-workers 2 \
  --telemetry-level layer \
  --telemetry-output telemetry.jsonl \
  --prompt test --max-new-tokens 8
```

Every layer record contains:

```text
forward / prefill-or-decode / token position / layer
route requests / unique experts / layer wall interval
cache requests, reuse hits, misses, evictions, admissions
logical/physical reader calls and bytes
demand read calls/bytes/time
prefetch submitted/served/late/useful/wasted
cache entries and bytes after the layer
```

Async reader deltas belong to the layer's wall interval and may include reads
started for later layers; totals are exact, but per-layer physical attribution
is intentionally described as overlap telemetry rather than synchronous
ownership. [Main Dev]

The final standalone S4 run emitted 320 records:

```text
forwards                         8
layers                         320
prefill forwards                 1
decode forwards                  7
raw route selections          5760
unique-expert layer sum        3871
process peak RSS       10,727,739,392 bytes (9.991 GiB)
```

## Prefetch implementation

Current-route prefetch submits the complete unique expert union immediately
after each layer router, then overlaps the read with shared-expert arithmetic.
Entries rejected by admission stay in a transient map only until all selected
experts in that layer consume them; they are never reread or retained in the
long-term cache. [Main Dev]

Predictive prefetch is bounded to 10% of the cache budget in these experiments.
It prioritizes hot entries and only predicts experts selected in three
consecutive decode routes. In-flight keys are deduplicated. At generation end,
unstarted work is cancelled and completed-but-unused payload is charged as
waste, so submitted/completed/useful/wasted accounting closes. [Main Dev]

Telemetry found and drove a real fix: the first current-route implementation
discarded admission-rejected payloads after the first expert consumed a batch,
causing later experts in that batch to be reread. The transient map removed
that duplicate I/O; the 4 GiB S4 total fell from 20.69 GiB to 15.79 GiB.
[Main Dev]

## Real route sweep

The policy sweep reused 15 real Qwen3.6 route traces:

```text
5 workloads: Chinese, English, code, math, continued conversation
3 generation lengths: 8, 16, 32 tokens
4 budgets: 1, 2, 4, 8 GiB
5 variants: S0-S4
300 metadata replays total
```

It uses the production `ExpertCache` and `CachePolicy`, real 6 MiB expert
sizes from `ExpertLocator`, and forward/layer batch union. It does not read
payload bytes. All 300 runs passed request, byte-budget, and prefetch accounting
invariants. [Main Dev]

Average decode demand read per forward:

| Variant | 1 GiB | 2 GiB | 4 GiB | 8 GiB |
|---|---:|---:|---:|---:|
| S0 | 1920.00 MiB | 1920.00 MiB | 1920.00 MiB | 1920.00 MiB |
| S1 | 1916.94 MiB | 1219.54 MiB | 985.72 MiB | 701.30 MiB |
| S2 | 1784.94 MiB | 1596.45 MiB | 944.22 MiB | 648.68 MiB |
| S3 | 1782.25 MiB | 1550.42 MiB | 962.51 MiB | 651.74 MiB |
| S4 | 1742.33 MiB | 1539.01 MiB | 959.50 MiB | 650.29 MiB |

S4 total read, including predictions, was:

| Budget | Demand | Total | Predictive waste across 15 traces |
|---:|---:|---:|---:|
| 1 GiB | 1742.33 MiB/fwd | 1813.18 MiB/fwd | 8.00 GiB |
| 2 GiB | 1539.01 MiB/fwd | 1559.75 MiB/fwd | 2.20 GiB |
| 4 GiB | 959.50 MiB/fwd | 962.51 MiB/fwd | 0 B |
| 8 GiB | 650.29 MiB/fwd | 651.74 MiB/fwd | 0 B |

There is no universal winner. S3 is strongest at 1 GiB among non-prefetch
policies, S1 wins at 2 GiB, and S2 has the lowest total read at 4/8 GiB. S4
reduces synchronous demand but is not suitable as the default at 1 GiB because
prediction waste remains material. Stage 7.4 must retain the S0-S4 matrix
rather than collapsing it to one policy. [Main Dev]

At 4 GiB, S3 decode demand by workload ranged from 853.02 MiB/forward for the
Chinese trace to 1052.38 MiB/forward for continued conversation. Across trace
lengths, longer 32-token traces showed more reuse than 8-token traces, as
expected. [Main Dev]

## Full-runtime correctness

`policy-check` loads C3-R once and then starts a fresh streaming runtime/cache
for every policy:

```bash
sparseflow policy-check <model> \
  --prompt test --max-new-tokens 8 \
  --cache-bytes 4GiB --variants S0,S1,S2,S3,S4
```

All variants produced the same IDs:

```text
[8160, 579, 264, 7047, 1817, 25, 271, 16]
```

For all five variants:

```text
full BF16 logits             exact for prefill + 7 decode forwards
320 route records            exact
generated IDs/text           exact
runtime/kernel identity      exact
initial expert I/O           zero
cache budget                 respected
demand accounting            exact
prefetch failures            zero
```

Observed 4 GiB logical reads and demand classes:

| Variant | Logical reads | Reuse hits | Prefetch served | Sync misses |
|---|---:|---:|---:|---:|
| S0 | 22.68 GiB | 0 | 0 | 3871 |
| S1 | 17.03 GiB | 965 | 0 | 2906 |
| S2 | 16.17 GiB | 1112 | 0 | 2759 |
| S3 | 15.79 GiB | 1177 | 0 | 2694 |
| S4 | 15.79 GiB | 1177 | 2693 | 1 |

The 4 GiB S4 run submitted 2693 current-route expert reads, consumed every
payload exactly once, and recorded zero ready-payload waste. Its single warm
run observed 10.03 seconds for seven decode forwards versus 10.72 seconds for
S3. These timings are development observations, not formal benchmark claims.
[Main Dev]

The separate 1 GiB exact gate exercised predictive routing: 45 previous-token
experts were submitted after the three-route stability filter, all 355 batches
completed, no failures occurred, and 84 MiB was charged as unused prediction.
Correctness remained exact. This path is implemented and bounded, but the
measured waste means it is not promoted as the 1 GiB default. [Main Dev]

## Validation

```text
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
Ran 42 tests
OK
```

Tests cover no-cache admission, global/per-layer budgets, hot protection,
second-touch prefill admission, decay/demotion, policy replay, stable-route
prediction, transient batch payload reuse, failed-future cleanup, telemetry
deltas, CLI contracts, and C3-R/C3-S policy correctness plumbing. [Main Dev]

Raw evidence:

- [`qwen36_stage7_3_policy_sweep_20260715.json`](qwen36_stage7_3_policy_sweep_20260715.json)
- [`qwen36_stage7_3_policy_correctness_20260715.json`](qwen36_stage7_3_policy_correctness_20260715.json)
- [`qwen36_stage7_3_previous_token_correctness_20260715.json`](qwen36_stage7_3_previous_token_correctness_20260715.json)
- [`qwen36_stage7_3_s4_runtime_20260715.json`](qwen36_stage7_3_s4_runtime_20260715.json)
- [`qwen36_stage7_3_s4_telemetry_20260715.jsonl`](qwen36_stage7_3_s4_telemetry_20260715.jsonl)

Stage 7.3 is complete. Stage 7.4 should freeze CPU threads, workload manifests,
cold/warm page-cache protocol, repetitions, and statistics before making
performance claims. INT8 and native kernels remain Stage 7.5. [Main Dev]

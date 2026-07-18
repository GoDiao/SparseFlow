# Qwen3.6 Stage 7.6 Native Decode Critical Path

Stage 7.6 is complete as a profile-gated critical-path stage. It did not turn
the complete Qwen3.6 runtime into native C++, and it did not meet every stretch
throughput target. It established a stable native expert ABI, removed repeated
row-sum work, implemented grouped native prefill, optimized the canonical
batch-one decode dispatcher, measured and rejected regressions, and completed a
clean-commit resident/streaming benchmark matrix. [Main Dev]

## Final runtime decision

The default Stage 7.6 dispatch is hybrid:

```text
prefill, rows > 1
  -> one grouped fused native MoE call per layer

decode, rows = 1
  -> canonical sorted-expert W8A8 path
  -> direct batch-one accumulation without torch.where/nonzero/index_add
```

Pure fused decode remains available for diagnostics, but it is not the default.
The real same-process AB/BA experiment measured fused decode at `0.886x` the
legacy path even though the isolated eight-expert microbenchmark was faster.
The fused kernel reduced routed-MoE time but changed the full-runtime CPU
frequency/thread-pool behavior enough to lose end-to-end decode throughput.
[Main Dev]

## Implemented boundaries

### Operator profiling

`RuntimeTelemetry` now supports `profile` mode and separates model forward,
decoder layers, Attention, DeltaNet, routed/shared MoE, final norm, lm_head,
argmax and token-loop time. Native INT8 counters separately report activation
quantization, GEMV, dynamic-linear and row-sum work. Provider timings retain
cache lookup, victim selection, allocation/reuse, pread, tensor view and policy
maintenance. [Main Dev]

The measured legacy W8A8 resident decode was approximately:

```text
model forward       395 ms/token
decoder layers      371 ms/token
routed experts      208 ms/token
DeltaNet            104 ms/token
Attention            20 ms/token
lm_head              22 ms/token
```

This corrected the earlier hypothesis that lm_head occupied roughly half of
decode time. It is only about 5%, below the native rewrite gate. `perf` was not
available because `perf_event_paranoid=4`; internal profile counters and
PyTorch/ATen boundaries were used instead. [Main Dev]

### Native ABI, leases and execution metadata

`NativeExpertBatch` owns the selected expert IDs, stable native tensor views and
cache leases for one layer call. Cache entries cannot be evicted while leased;
success and exception paths release all pins. A failed admission while every
candidate is leased returns a transient payload instead of invalidating an
active tensor view. [Main Dev]

`canonical-int8-exec-v1` stores offline INT32 row sums alongside the immutable
`canonical-int8-v1` weights. The complete Qwen3.6 sidecar contains 20,480 entries
and occupies 125,992,960 bytes. Runtime row-sum preparation is now zero, and the
sidecar stays resident rather than joining per-token expert reads. [Main Dev]

### Native MoE kernels

The fused operator quantizes each hidden row once, groups assignments by expert,
executes gate/up, SiLU multiplication and down projection, then performs a fixed
BF16 accumulation order. It supports both batch-one decode and grouped prefill.
Resident and streaming pass identical weight tensors and metadata into the same
operator. [Main Dev]

The grouped prefill path was retained. Pure fused decode was rejected as the
default after paired testing. The canonical decode path instead gained a
batch-one fast path that removes repeated route masks and `index_add_` while
preserving sorted expert order and BF16 accumulation. [Main Dev]

### Summary telemetry observer effect

Summary telemetry no longer snapshots a complete provider counter dictionary
before and after every MoE layer. Layer counters are fixed-cost integer updates,
and provider totals reuse the generation-boundary snapshots already required by
the runtime result. [Main Dev]

Nine unpinned AB/BA pairs measured:

```text
none median decode       2.1197 tok/s
summary median decode    2.1911 tok/s
median paired delta      +2.67%
measured slowdown         0.00%
profile path closure     98.94%
```

The host variance was too large to resolve an absolute ±1% difference: paired
deltas ranged from `-8.96%` to `+5.59%`. The required observer gate is therefore
stated narrowly and honestly: no summary-induced slowdown greater than 1% was
observed. A physical-core affinity attempt was stopped because PyTorch used only
about 5.5 cores and generation time increased from about 19 seconds to about 89
seconds; that partial run is retained as development evidence. [Main Dev]

## Correctness result

The real 32-token Qwen3.6 resident/streaming gate passed all required invariants:

```text
32 full-vocabulary logits fingerprints exact       true
1,280 route records exact                          true
generated IDs and text exact                       true
resident weight bytes exact                        true
execution metadata bytes exact                     true
resident generation expert I/O                     0
4 GiB cache budget respected                       true
demand accounting exact                            true
prefetch failures/transients                       0
streaming leases remaining after generation         0
```

The routed resident payload is 30.078 GiB plus the 120.16 MiB execution
metadata sidecar. The corresponding 4 GiB streaming correctness run read 21.29
GiB over prefill plus 31 decode forwards. [Main Dev]

## Formal performance

The final performance files were rerun outside the repository and then copied
back so every JSON records clean commit
`0d4bdf9c847da30e297c80d793f1f59ed3a90fe3`. The protocol remained 10 CPU
threads, the frozen `core-zh-moe` prompt, 32 greedy tokens, one warmup plus three
warm runs, and three independent model-cold processes. [Main Dev]

| Path | TTFT | Decode | P50/P95 | Peak RSS | Expert reads/run |
|---|---:|---:|---:|---:|---:|
| W8A8 hybrid resident warm | 6.09 s | 2.1513 tok/s | 0.460/0.526 s | 36.63 GiB | 0 |
| W8A8 hybrid S1 4 GiB warm | 12.36 s | 1.4128 tok/s | 0.683/0.914 s | 9.87 GiB | 20.76 GiB |
| W8A8 hybrid S1 8 GiB warm | 12.46 s | 1.4944 tok/s | 0.664/0.782 s | 13.94 GiB | 13.84 GiB |
| W8A8 hybrid S1 4 GiB cold median | 57.10 s | 1.0227 tok/s | 0.946/1.243 s | <=9.75 GiB | 21.29 GiB |

Compared with Stage 7.5.6:

| Path | Stage 7.5 | Stage 7.6 | Change | Goal | Result |
|---|---:|---:|---:|---:|---|
| Resident warm | 2.4925 | 2.1513 | -13.7% | >=3.25 | not met |
| S1 4 GiB warm | 1.0981 | 1.4128 | +28.7% | >=1.32 | met |
| S1 8 GiB warm | 1.4157 | 1.4944 | +5.6% | >=1.77 | not met |
| S1 4 GiB model-cold | 0.8632 | 1.0227 | +18.5% | >=1.00 | met |

The resident number was measured after many hours of AVX-512 experiments and
shows substantial host thermal/frequency drift. The same-process hybrid AB/BA
comparison is the correct attribution result: hybrid retained prefill gains and
measured `0.980x` legacy decode, within the observed host variance. No resident
decode speedup is claimed. [Main Dev]

The streaming improvement is stronger evidence because loaded bytes, hit/miss
counts, evictions and routes are exactly unchanged from Stage 7.5. At 4 GiB the
run still performs 7,073 misses and reads exactly 22,293,190,656 bytes; at 8 GiB
it performs 4,714 misses and reads exactly 14,857,924,608 bytes. The speedup is
therefore software critical-path improvement, not reduced work. [Main Dev]

## Stop/go decisions

- **Grouped native prefill: GO.** TTFT fell from 8.25 to 6.09 seconds resident,
  from 20.21 to 12.36 seconds at 4 GiB, and from 18.01 to 12.46 seconds at
  8 GiB. [Main Dev]
- **Pure fused decode: NO-GO as default.** The paired full-runtime ratio was
  `0.886x`; keep it for profiling and future kernel work. [Main Dev]
- **Hybrid dispatch: GO.** It preserves grouped prefill and exact canonical
  decode with no material paired decode regression. [Main Dev]
- **lm_head native rewrite: NO-GO.** Its measured share is about 5%, below the
  10% component gate. [Main Dev]
- **Deterministic current-layer I/O pipeline: NO-GO as default.** It remained
  exact with zero failure/waste but measured 0.8994 tok/s versus 1.2061 tok/s
  for synchronous fused streaming. Shared-expert work hides too little I/O to
  offset Future contention. [Main Dev]
- **DeltaNet projection fusion: NO-GO.** Component-level savings were below 1%
  of total decode and far below the 15% expected-gain gate. The recurrence was
  not rewritten. [Main Dev]

## Completion boundary

Stage 7.6 completes the native expert critical-path investigation. It does not
claim a complete native text runtime, native generation session, multi-request
scheduler, INT4, GPU execution, Vision or MTP support. The next stage should
focus on generation/session ownership and service scheduling only after its own
profile gate; it should not reopen rejected dense rewrites without new evidence.
[Main Dev]

Raw evidence:

- `benchmarks/results/2026-07-18/stage7_6/formal/summary.json`
- `benchmarks/results/2026-07-18/stage7_6/formal/verification.json`
- `benchmarks/results/2026-07-18/stage7_6/formal/`
- `benchmarks/results/2026-07-18/stage7_6/correctness/`
- `benchmarks/results/2026-07-18/stage7_6/development/`
- `benchmarks/results/2026-07-18/stage7_6/profile/`

[Main Dev]

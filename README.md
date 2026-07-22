# SparseFlow

SparseFlow is a Qwen3.6-first research backend for tiered-memory sparse-expert
inference. It keeps dense model state resident and gives routed MoE experts an
explicit resident or SSD/cache storage policy.

The current release target is **Qwen3.6-35B-A3B text-only Public Alpha**.
Vision, MTP, shared streaming batching, and production serving are not part of
this release.

## Supported Environment

The tested environment is Linux x86_64 with Python 3.12, PyTorch 2.9.x,
Transformers 5.x development builds, Safetensors 0.8.x, and Accelerate 1.14.x.
The native W8A8 backend requires AVX-512 VNNI. CPU-only Python inspection and
planning commands do not require the runtime extras.

This is a source checkout release. The native C++ sources are compiled into a
local cache on first native run; no model payload is copied to the Python
package directory.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[runtime]'
export SPARSEFLOW_NATIVE_CACHE="$PWD/.cache/native/int8_vnni"
```

Set paths on a machine with enough disk space:

```bash
MODEL=/data/model/Qwen3.6-35B-A3B
INT8=/data/cache/qwen36-int8
```

Check the model before reading payloads:

```bash
PYTHONPATH=src python -m sparseflow doctor "$MODEL" \
  --preset low-memory --int8-container "$INT8" --check-native
PYTHONPATH=src python -m sparseflow inspect "$MODEL"
PYTHONPATH=src python -m sparseflow plan "$MODEL" --ram 96 --ctx 4096
```

Prepare the versioned INT8 expert container. Conversion is shard/layer
resumable and writes offline row sums required by the native path:

```bash
PYTHONPATH=src python -m sparseflow prepare-int8 "$MODEL" \
  --output "$INT8" \
  --report results/stage7_9_prepare_int8.json \
  --json > results/stage7_9_prepare_int8.stdout.json
```

Run the stable resident path:

```bash
PYTHONPATH=src python -m sparseflow run "$MODEL" \
  --preset stable --int8-container "$INT8" \
  --prompt "Explain sparse expert routing in one paragraph." \
  --max-new-tokens 32 \
  --output results/stage7_9_stable.json
```

Run the stable single-request low-memory path:

```bash
PYTHONPATH=src python -m sparseflow run "$MODEL" \
  --preset low-memory --int8-container "$INT8" \
  --prompt "Explain sparse expert routing in one paragraph." \
  --max-new-tokens 32 \
  --output results/stage7_9_low_memory.json
```

The fixed-cohort grouped path is explicit and experimental:

```bash
PYTHONPATH=src python -m sparseflow run "$MODEL" \
  --preset experimental-batch --int8-container "$INT8" \
  --prompt "Explain cache locality." \
  --prompt "Explain MoE routing." \
  --max-new-tokens 32
```

The grouped preset requires equal encoded prompt lengths, as required by the
fixed-cohort harness. Shared streaming batching is intentionally unavailable.

## Public Alpha Path Status

| Path | Status |
|---|---|
| INT8 native resident hybrid | Stable baseline |
| INT8 native single-request streaming S1 LRU | Stable low-memory |
| INT8 native resident grouped fixed cohort | Experimental opt-in |
| Shared streaming batching/subcohort | Disabled, known limitation |
| Pure fused decode | Diagnostics only |

`doctor` validates model structure, safetensors headers, disk, CPU ISA,
container metadata, and optional native extension loading. The normal model
identity is a metadata plus payload-size digest; use `--full-payload-hash` only
when a complete 67 GiB payload hash is intentionally required.

## Testing

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

The benchmark workstream owns formal quality manifests and raw result schemas.
Large model payloads and caches stay outside Git; result JSON, reports, and
manifests are committed.

## Current Limitations

- CPU native execution requires AVX-512 VNNI.
- The native extension is compiled from source on first use.
- Streaming is supported for one request at a time; prefetch is disabled in
  the Public Alpha preset.
- Grouped execution is a fixed equal-length cohort harness, not a dynamic
  scheduler.
- The host Transformers runtime still owns attention, Gated DeltaNet, KV state,
  tokenizer, sampling, and generation orchestration.

<!-- [Main Dev] -->

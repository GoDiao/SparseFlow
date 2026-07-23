# SparseFlow Installation

This guide installs the Qwen3.6 text-only runtime, builds the AVX-512 VNNI
extension, prepares the INT8 expert container, and starts either the CLI or the
OpenAI-compatible local server.

## Capacity and Runtime Requirements

SparseFlow does not make the source model small. It keeps dense state resident
and streams routed experts from an INT8 container. Plan for all of these files:

| Item | Qwen3.6-35B-A3B size |
|---|---:|
| Source model | about 67 GiB |
| INT8 routed-expert container | about 30.1 GiB |
| Offline row-sum metadata | about 120 MiB |
| Conversion reserve | at least 2 GiB beyond the final container |

The native CPU path requires x86-64 AVX-512 VNNI. `doctor --check-native`
checks the ISA and compiles the extension before a model runtime is created.

Measured Qwen3.6 admission estimates at a 4096-token context are:

| Path | Minimum available RAM | Recommended available RAM |
|---|---:|---:|
| Stable resident | about 38.7 GiB | about 42.6 GiB |
| Low-memory, 4 GiB cache | about 12.6 GiB | about 14.6 GiB |
| Low-memory, 256 MiB cache | about 8.9 GiB | about 10.9 GiB |
| Experimental laptop-16gb, 2048-token context, 256 MiB cache | about 8.7 GiB | about 10.7 GiB |

These values mean **available** RAM, not installed RAM. Close browsers, IDEs,
and other large applications before starting a runtime on a 16 GiB host. Swap
or a Windows page file prevents some allocation failures but is not a
substitute for the Doctor admission gate and can make inference much slower.

## Windows 10/11

### Prerequisites

Install the following before running the setup script:

1. Git.
2. Internet access to PyPI, the PyTorch CPU index, and GitHub releases. The
   setup script bootstraps a checksum-verified `uv.exe` into `.tools\uv` when
   it is not already available. An existing uv can also be passed with
   `-UvExe`.
3. Visual Studio 2022 Build Tools with **Desktop development with C++**, the
   MSVC x64 toolchain, and a Windows SDK.
4. An AVX-512 VNNI capable x86-64 CPU.

Do not create the runtime environment from Anaconda. A Conda base installation
can put an older `MSVCP140.dll` ahead of the Visual Studio runtime and cause
PyTorch to fail while loading `c10.dll` with `WinError 1114`.

### Install on the project drive

From the repository root:

```powershell
.\scripts\setup_windows.ps1
. .\scripts\use_runtime_windows.ps1
```

To download uv separately, or to retry that step through a configured proxy:

```powershell
.\scripts\bootstrap_uv_windows.ps1
```

The scripts keep Python, the virtual environment, downloads, temporary files,
and build output inside the checkout:

```text
.tools/python/                  uv-managed CPython
.venv-runtime/                 runtime environment
.cache/uv/                     package cache
.cache/huggingface/            Hugging Face cache
.cache/torch/                  Torch cache
.cache/tmp/                    temporary files
.cache/native/int8_vnni_windows/ native build output
```

The setup installs CPU-only PyTorch from the official PyTorch CPU wheel index,
then installs SparseFlow and its runtime extra in editable mode. Verify the
environment independently with:

```powershell
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
sparseflow preset low-memory --json
```

### Download the model

Keep model payloads on a drive with at least 105 GiB free for the source model,
the converted container, and conversion reserve. The expected layout is a
normal ModelScope or Hugging Face snapshot containing `config.json`, tokenizer
files, `model.safetensors.index.json`, and all shards.

For networks where Hugging Face is unreliable, ModelScope can download the
model. Install the downloader in the project environment and keep its cache in
the project:

```powershell
$env:MODELSCOPE_CACHE = "$PWD\.cache\modelscope"
uv pip install --python .\.venv-runtime\Scripts\python.exe modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B', local_dir=r'model/Qwen3.6-35B-A3B')"
```

Do not run Hugging Face and ModelScope downloads for the same destination at
the same time. Check for incomplete temporary files before conversion.

### Inspect and prepare the INT8 container

```powershell
$model = "$PWD\model\Qwen3.6-35B-A3B"
$int8 = "$PWD\model\Qwen3.6-35B-A3B-int8"

sparseflow inspect $model
sparseflow plan $model --ram 16 --ctx 4096
sparseflow prepare-int8 $model `
  --output $int8 `
  --report .cache\results\prepare-int8.json `
  --json
```

`prepare-int8` is resumable by layer. Re-run the same command after an
interruption. Do not use a partial layer selection for a release container;
the runtime requires a complete 40-layer container and execution metadata.

### Run Doctor and compile the native extension

Start with the cache size you intend to run:

```powershell
sparseflow doctor $model `
  --preset low-memory `
  --int8-container $int8 `
  --cache-bytes 4GiB `
  --check-native
```

On a 16 GiB development machine, use a smaller smoke-test cache and close other
large applications first:

```powershell
sparseflow doctor $model `
  --preset low-memory `
  --int8-container $int8 `
  --cache-bytes 256MiB `
  --check-native
```

Do not continue to a real runtime when `memory_admission` is `fail`. The model,
container, CPU ISA, and native extension checks can still pass independently.

For the laptop profile, Doctor uses a single-request 2048-token admission
budget and keeps the profile opt-in:

```powershell
sparseflow doctor $model `
  --preset laptop-16gb `
  --int8-container $int8 `
  --check-native
```

The laptop preset is experimental and host-dependent. Do not use
`--available-ram` to manufacture a pass. The server also refuses to load when
the real memory admission check fails.

### Generate text

```powershell
sparseflow run $model `
  --preset low-memory `
  --int8-container $int8 `
  --cache-bytes 256MiB `
  --prompt "Explain sparse expert routing in one sentence." `
  --max-new-tokens 1 `
  --output .cache\results\windows-smoke.json
```

Increase the cache only after Doctor reports enough available RAM. The stable
resident preset is intended for hosts with roughly 43 GiB or more available.

The frontend-facing endpoint and response contract is frozen in
[`docs/api_contract.md`](api_contract.md), with the machine-readable schema in
[`docs/openapi/sparseflow-openapi.json`](openapi/sparseflow-openapi.json).

### Start the local server

```powershell
sparseflow serve $model `
  --preset low-memory `
  --int8-container $int8 `
  --cache-bytes 256MiB `
  --ctx 4096 `
  --host 127.0.0.1 `
  --port 8000
```

For the experimental 16 GiB laptop profile, keep the context and cache
bounded by the preset defaults:

```powershell
sparseflow serve $model `
  --preset laptop-16gb `
  --int8-container $int8 `
  --host 127.0.0.1 `
  --port 8000
```

The server resolves this preset to a 2048-token context, 128-token maximum
completion, and 256 MiB cache unless explicitly overridden. It still runs the
real Doctor admission gate before loading the runtime.

In another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

$body = @{
  model = "qwen3.6-35b-a3b-sparseflow"
  messages = @(@{ role = "user"; content = "Say hello in one sentence." })
  max_tokens = 1
  stream = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body
```

Bind to `127.0.0.1` unless remote access is intentional. Use `--api-key` or the
`SPARSEFLOW_API_KEY` environment variable before exposing the server to a
network.

### Laptop validation matrix

Run the formal laptop CLI matrix only from a clean commit and only after
Doctor passes. This creates 3 prompts x 3 completion lengths x 2 repeats:

```powershell
python -m benchmarks.run_stage7_11_laptop `
  --model $model `
  --int8-container $int8 `
  --repeats 2 `
  --token-count 8 `
  --token-count 16 `
  --token-count 32 `
  --output .cache\results\stage7_11_laptop_cli.json

python -m benchmarks.summarize_stage7_11_laptop `
  --input .cache\results\stage7_11_laptop_cli.json `
  --output .cache\results\stage7_11_laptop_summary.json

python -m benchmarks.verify_stage7_11_laptop `
  --input .cache\results\stage7_11_laptop_cli.json `
  --output .cache\results\stage7_11_laptop_verification.json
```

The verifier compares repeats within the same prompt/token cell. A summary
without an explicitly approved 32-token performance threshold is intentionally
not a performance `PASS`.

Run the real local Server acceptance on the same clean commit and host:

```powershell
python -m benchmarks.run_stage7_11_server_acceptance `
  --model $model `
  --int8-container $int8 `
  --preset laptop-16gb `
  --cache-bytes 256MiB `
  --context-tokens 2048 `
  --max-completion-tokens 32 `
  --request-tokens 8 `
  --output .cache\results\stage7_11_server_acceptance.json
```

The runner owns and cleans the Server process tree. A Doctor memory rejection
is a valid `NO-GO` result; do not bypass it with `--available-ram`.

## Linux

The authoritative performance and quality matrices use Linux x86-64. Keep the
environment and caches on the large data volume:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export TMPDIR="$PWD/.cache/tmp"
export SPARSEFLOW_NATIVE_CACHE="$PWD/.cache/native/int8_vnni"

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$HF_HOME" \
  "$TORCH_HOME" "$TMPDIR" "$SPARSEFLOW_NATIVE_CACHE"
uv python install 3.12
uv venv --python 3.12 .venv-runtime
. .venv-runtime/bin/activate
uv pip install --python .venv-runtime/bin/python \
  'torch==2.9.1+cpu' --index https://download.pytorch.org/whl/cpu
uv pip install --python .venv-runtime/bin/python -e '.[runtime]'
```

Then use the same `inspect`, `prepare-int8`, `doctor`, `run`, and `serve`
workflow shown above, replacing PowerShell path variables with shell variables.
GCC or Clang must support AVX-512 VNNI compilation.

## Troubleshooting

### `WinError 1114` or `c10.dll` fails to load

Check where the Microsoft runtime is coming from:

```powershell
Get-Command python
where.exe MSVCP140.dll
```

Create the environment with uv-managed Python instead of a Conda interpreter.
Do not prepend a Conda installation to `PATH` when starting SparseFlow.

### `cl.exe` is missing

Modify Visual Studio Build Tools and add **Desktop development with C++**.
SparseFlow uses `vswhere.exe` to initialize the latest x64 MSVC environment.

### Compiler-version warning with localized `cl.exe` output

Some Windows locales make PyTorch print a harmless warning while decoding the
compiler version. If `doctor --check-native` reports `native_extension: pass`
and all operators register, the build itself succeeded.

### Doctor reports insufficient RAM

Use `--cache-bytes 256MiB` for a one-token smoke, close other applications, and
run Doctor again. Do not override `--available-ram` merely to force a pass; that
flag is for planning and controlled test fixtures, not for creating memory.

### CPU does not have AVX-512 VNNI

Inspection and planning still work, but the current native Qwen3.6 INT8 runtime
cannot run. Use a compatible host; there is no scalar production fallback in
this release.

## Verified Windows Environment

The Windows path was verified on Windows 10 build 19045 with an Intel
i5-11400H (6 cores, 12 threads, AVX-512 VNNI), uv-managed CPython 3.12.13,
PyTorch 2.9.1+cpu, Transformers 5.14.1, and Visual Studio 2022 Build Tools.
The native `dynamic_linear`, `fused_moe`, and `grouped_moe` operators compiled
and passed their numerical tests. The 40-layer local Qwen3.6 INT8 container and
offline row sums also passed Doctor. Real generation still requires the RAM
admission gate described at the start of this guide.

# Stage 7.9 Closure Fixes

**Owner:** `[Main Dev]`  
**Status:** complete

## Scope

These fixes close two release-boundary gaps found after the Public Alpha
validation:

1. Base installation must not import the optional PyTorch runtime for
   `preset`, `inspect`, `plan`, or header-only `doctor`.
2. `doctor` must admit a preset against usable memory, including cgroup limits
   and the actual resident/streaming resource budget.

This document is deliberately separate from the formal Stage 7.9 result. No
formal release result may be regenerated until the clean-commit experiment-host
acceptance below passes. `[Main Dev]`

## Implementation Order

### 1. CLI/runtime boundary

- Remove the top-level `text_runtime` import from `src/sparseflow/cli.py`.
- Keep runtime imports inside `run`, `text-generate`, `text-check`,
  `runtime-check`, INT8 checks, and policy checks.
- Keep lazy compatibility shims for existing callers that patch the CLI
  comparison boundary; the shims do not import torch at module import time.
- Convert missing runtime extras into exit code 2 with an installation message.
- Verify the complete CLI import chain with `python -S`, which excludes
  site-packages and therefore cannot see torch from the host environment.

### 2. Doctor RAM admission

`doctor` now emits a `memory` object with:

```json
{
  "available_ram_bytes": 0,
  "required_ram_bytes": 0,
  "recommended_ram_bytes": 0,
  "headroom_bytes": 0,
  "source": "proc-meminfo|cgroup|override|unknown",
  "status": "pass|warn|fail",
  "components": {}
}
```

The components include dense resident weights, resident INT8 experts or the
streaming cache, execution row-sum sidecar, KV/DeltaNet state, activation and
kernel workspace, experimental batch state/workspace, and page-cache/runtime
reserve.

Memory source precedence is:

1. `--available-ram` override;
2. remaining cgroup memory (`memory.max - memory.current`) bounded by host
   `MemAvailable`;
3. `/proc/meminfo` `MemAvailable`;
4. `unknown` with a warning when no source exists.

`required` means the process can theoretically start. `recommended` adds a
runtime headroom reserve. Below `required` is `fail`; between `required` and
`recommended` is `warn`; above `recommended` is `pass`.

The planner also accepts `--available-ram` so the same admission fixtures can
be reproduced without changing host memory.

### 3. Local acceptance

The local gate must pass:

```text
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
python -m compileall -q src tests
PYTHONPATH=src python -S -c 'import sparseflow.cli'
git diff --check
```

The subprocess test covers base CLI commands with no site-packages/torch and
requires `run` to return a clean optional-runtime error.

### 4. Clean commit and experiment-host acceptance

After local tests pass, push one clean commit. On the experiment host, create a
fresh base environment with no runtime extras and run:

```bash
uv pip install -e .
sparseflow preset --json
sparseflow inspect "$MODEL" --json
sparseflow plan "$MODEL" --available-ram 16GiB --json
sparseflow doctor "$MODEL" --preset low-memory --int8-container "$INT8" --json
```

The same base environment must show a clear error for `sparseflow run`, not a
traceback. In a runtime environment, run the two native Doctor commands:

```bash
sparseflow doctor "$MODEL" --preset stable \
  --int8-container "$INT8" --check-native --json
sparseflow doctor "$MODEL" --preset low-memory \
  --int8-container "$INT8" --check-native --json
```

Record for each preset:

| preset | doctor required/recommended | actual current RSS | actual peak RSS |
|---|---:|---:|---:|
| stable | JSON fields | about 36-39 GiB expected | measured |
| low-memory | JSON fields | about 8-12 GiB expected | measured |

The clean-commit experiment-host acceptance completed on
`36aa619901561cf4148dbf8dfaadb8f73b058d53`:

| preset | Doctor required | Doctor recommended | actual current RSS | actual peak RSS | result |
|---|---:|---:|---:|---:|---|
| stable | 38.69 GiB | 42.56 GiB | 35.64 GiB | 36.63 GiB | pass |
| low-memory | 12.62 GiB | 14.62 GiB | 5.54 GiB | 9.66 GiB | pass |

The host memory source was `cgroup`, with about 116.92 GiB available. The
stable run loaded 32,422,297,600 bytes of resident INT8/sidecar payload; the
low-memory run retained the configured 4 GiB cache. The compact artifacts are
`closure_doctor_stable.json`, `closure_doctor_low.json`,
`closure_rss.json`, and `closure_base_cli.json` under the Stage 7.9 result
directory. `[Main Dev]`

### 5. Formal evidence last

The Stage 7.9 Smoke 32, Standard 128, and Endurance 512 evidence was then
regenerated from the same clean commit. All three verifiers passed and record
exact resident/streaming IDs, text, routes, logit fingerprints, cache budget,
compact artifacts, and model/container identity. The final result records the
closure-fix commit, Doctor JSON, runtime identity, and the unchanged Public
Alpha boundary. `[Main Dev]`

<!-- [Main Dev] -->

### 6. Post-acceptance micro-fixes

After the first closure evidence was reviewed, three small release-boundary
issues were fixed before the next formal matrix:

- `_load_text_runtime()` now catches both `ImportError` and `OSError`. A
  broken installed PyTorch binary, such as an unloadable `c10.dll`, therefore
  produces the same clean optional-runtime error as a missing dependency.
- The no-site-packages `plan` subprocess test now accepts the existing exit
  code `1` resource-warning contract and rejects only exit code `2`, which is
  the runtime-dependency failure code being tested.
- Experimental-batch RAM accounting keeps one session in
  `kv_deltanet_state_bytes` and adds exactly `(batch_size - 1)` session states
  in `experimental_batch_state_bytes`. The previous implementation multiplied
  the canonical state by the batch size and then added the extra states again.

The regression tests cover the broken-runtime import and the B=4 state
accounting. Local verification after these fixes:

```text
PYTHONPATH=src python -S -m unittest discover -s tests -p 'test_release.py'
Ran 7 tests
OK

PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
Ran 100 tests
OK
```

`compileall` and `git diff --check` also pass. These fixes do not change the
model weights, runtime arithmetic, cache policy, benchmark prompts, or formal
Stage 7.9 result data, so the existing matrix is not rerun in this change.
The next formal matrix must record the new micro-fix commit as its clean
runtime and provenance identity. `[Main Dev]`

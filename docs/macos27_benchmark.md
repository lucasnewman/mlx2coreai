# macOS 27 Stateful Benchmark

Tested September 9, 2026 on Apple M3 Max, 64 GiB RAM, macOS 27.0 build
26A428, and Xcode 27.0 build 27A266a at `/Applications/Xcode.app`.

**Current outcome:** CoreAI b2 Python executes the full model with numerical
parity to Swift. Both runners now complete the 32-step growing-context sweep
through context 2048 using compact position IDs, with no fixed-length model
fallback. See [Compact-Position Decoding](#compact-position-decoding) for the
current configuration and results; earlier experiments are retained below.

## Conversion Environment

The original measurements below used `coreai-core==1.0.0b2`, `mlx==0.31.2`,
`mlx-metal==0.31.2`, and `mlx-lm==0.31.3` with Python 3.12. The benchmark
uses the Swift CoreAI runtime with GPU preferred; this is not a measurement
proving exclusive GPU placement.

MLX is no longer pinned: the converter now supports the `BroadcastAxes`
primitive introduced in MLX 0.32. The following setup uses the current MLX
release in an isolated environment without changing the shared installation:

```bash
python -m venv --system-site-packages .build/coreai-b2-env
.build/coreai-b2-env/bin/python -m pip install --upgrade \
  coreai-core==1.0.0b2 mlx mlx-lm==0.31.3
.build/coreai-b2-env/bin/python -m mlx2coreai convert-mlx-lm-stateful \
  mlx-community/Qwen3-0.6B-bf16 \
  --revision 42096995f6402fde107068cf530136fe64b604f8 \
  --output artifacts/qwen_macos27_26A428_b2 --max-context-length 2048
python scripts/benchmark_aimodel_sampling.py artifacts/qwen_macos27_26A428_b2 \
  --contexts 16,32,64,128,256,512,1024,2048 --steps 32 \
  --position-ids-layout context
```

The recorded conversion used the local Hugging Face snapshot for this revision
because offline mode still attempted a repository metadata lookup. The graph
has dynamic sequence and cache dimensions, one `main` function, and mutable
`keyCache` / `valueCache`. Weights and caches are BF16; logits are cast to FP16.
`AIProgram.optimize()` is enabled, and BF16 constants are preserved.

## Fixed-Position Sweep

Each row prefills the context, warms up once, and times 32 one-token calls plus
host greedy sampling. The decode position is reused for this default sweep.
Prefill and initial compilation are excluded. These are single-run results,
not confidence intervals or a model accuracy comparison.

| Context | Earlier run (tok/s) | Current run (tok/s) | Current elapsed (s) |
| ---: | ---: | ---: | ---: |
| 16 | 17.84 | 19.79 | 1.617 |
| 32 | 16.99 | 19.45 | 1.645 |
| 64 | 16.46 | 19.20 | 1.666 |
| 128 | 15.54 | 16.53 | 1.935 |
| 256 | 14.52 | 16.27 | 1.967 |
| 512 | 12.43 | 13.61 | 2.350 |
| 1024 | 9.97 | 9.16 | 3.493 |
| 2048 | 7.56 | 6.19 | 5.168 |

A second complete fixed-position sweep with the final script also exited
successfully. Its context-16 result was 14.49 tok/s and context-2048 result was
6.31 tok/s (other retained rows: 256 = 11.89, 512 = 10.71, 1024 = 8.97 tok/s).
The short-context variation means the apparent improvement in the first run
should not be treated as a stable speedup. Long-context performance was similar
across these two runs and below the earlier result.

The comparison changes both the OS runtime and authoring package, and uses a
fresh conversion with a 2048-context trace cache instead of 256. Both exports
have dynamic cache shapes. It does not isolate the effect of the OS update.

## Growing-Context Diagnostic

The same sweep with `--grow-context` did not produce a usable throughput result.
After repeated shape-preparation warnings, Metal reported
`MTL4CommandQueueErrorDomain error 1` and warned that encoded operations may not
have completed. The process was then terminated.

A one-second stack sample during the run showed this execution path:

```text
MPSGraphDelegateKernel._coreAI_inferValue
MPSGraphExecutable.runInternalWithDevice
GPURegionRuntime.evaluateOps
GPU::StridedSliceUpdateOpHandler::encodeOp
GPURegionRuntime::waitAndReadIntTensorData
MTL3On4CommandBuffer.waitUntilCompleted
```

This establishes that the runtime entered the GPU execution path, but does not
identify the cause of the later command-queue error or establish exclusive GPU
placement. The full-position-input variant was not re-run at that scale.
The later compact-position Python / Swift sweeps below completed the same
contexts and step counts. Fixed-position measurements must not be presented as
growing-context generation performance.

## Observed Compatibility Issues

- Assets freshly converted with `coreai-core==1.0.0b1` abort at load with
  `expected AICode versioned location`. Re-converting with b2 resolves this.
- The initial MLX 0.32.2 conversion failed on `BroadcastAxes`. This is now
  supported, including ignored axes and runtime-dependent broadcast shapes.
- `expectFrequentReshapes = true` aborts specialization in MPSGraph's
  `MPSMemrefAllocFusion` pass with `operand #1 does not dominate this use`.
  The runner retains `false`; this flag does not make graph dimensions static.
- The working GPU-preferred configuration emits repeated Neural Engine BF16
  compatibility warnings during shape preparation. These did not prevent the
  fixed-position sweep from completing.

The latest authoring release was checked against
[Apple's coreai-core package on PyPI](https://pypi.org/project/coreai-core/1.0.0b2/).

The rebuilt bundle is saved at `artifacts/qwen_macos27_26A428_b2/` in the project workspace.

## Verification

The converter, lowering, and runtime smoke suites passed with the isolated
environment: 33 tests. The Swift runner builds with the current Xcode SDK.
SDK/compiler changes invalidate its cached executable, and benchmark rows are
flushed immediately so a later runtime failure does not hide completed rows.

After adding `BroadcastAxes` support, the isolated environment was upgraded
with an unconstrained `pip install --upgrade mlx` to MLX / mlx-metal 0.32.2.
All 121 tests passed, including 23 BroadcastAxes tests covering callback
capture, dtype preservation, ignored axes, scalar/empty dimensions, invalid
shapes, incomplete upstream inference, and dynamic CPU runtime comparisons.
The op semantics follow [MLX's BroadcastAxes implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/primitives.cpp).

Qwen was re-converted with dynamic sequence/cache dimensions and BF16 weights
using MLX 0.32.2. The new bundle is `artifacts/qwen_mlx0322/`. A short CoreAI run
at contexts 16 and 32 (two timed steps each) completed successfully. This is a
conversion/execution smoke check, not an update to the performance table or a
resolution of the growing-context runtime issue above.

## Python / Swift Parity

Re-tested the full `artifacts/qwen_mlx0322/qwen_mlx0322.aimodel` through both
runners on the same OS and SDK with CoreAI b2 and MLX 0.32.2. Python now
executes this dynamic, stateful model without the previous `Output E had no
value provided` fatal error. The Python `InferenceFunction` API still only
accepts `inputs` and `state`, not Swift's `outputViews`; caller-supplied output
buffers are no longer necessary for this freshly converted asset in this
environment. This does not establish compatibility for every dynamic graph or
old b1 asset, nor isolate which OS, bindings, or authoring change removed the
blocker.

The baseline below used GPU-preferred OS specialization, fill token 0, identical
full-context position tensors and cache capacities, one warmup, and 32 timed calls per
context. Python keeps its NumPy-backed input/state allocation; Swift allocates
from the runtime descriptors and explicitly supplies output views. No static
shape conversion or fixed-length fallback was introduced. Each process reused
one loaded model across all eight contexts, running sequentially to avoid GPU
contention.

| Context | Python (tok/s) | Swift (tok/s) |
| ---: | ---: | ---: |
| 16 | 17.97 | 17.04 |
| 32 | 17.16 | 16.90 |
| 64 | 16.66 | 15.71 |
| 128 | 12.46 | 14.75 |
| 256 | 11.83 | 14.10 |
| 512 | 11.84 | 12.24 |
| 1024 | 9.19 | 9.81 |
| 2048 | 6.33 | 6.96 |

All **256 sampled token IDs matched**. All eight final-step logit vectors
(151,936 values each) were finite and byte-identical after losslessly widening
the FP16 output to FP32: maximum absolute difference **0**. Intermediate-step
logits were not dumped. This establishes numerical parity between these two
runners for the tested workload, not accuracy against the original MLX model.
The throughput results are single sweeps, not confidence intervals; they show
comparable performance, not a uniform Python slowdown or speedup. As above,
these are fixed-position throughput measurements, not advancing-position
generation rates.

The shell's default `python` still has CoreAI b1. Use the isolated b2 interpreter
and a b2-converted asset to reproduce:

```bash
.build/coreai-b2-env/bin/python scripts/benchmark_aimodel_sampling.py \
  artifacts/qwen_mlx0322 --runtime-backend python \
  --contexts 16,32,64,128,256,512,1024,2048 --steps 32 --warmup 1 \
  --position-ids-layout context \
  --json-output .build/python_parity_full.json \
  --logits-dir .build/python_parity_full_logits
```

Repeat with `--runtime-backend swift` and distinct output paths to compare.
Both runners now export JSON with the selected backend, sampled tokens,
positions, and timings. `--logits-dir` writes the last token's final-step logits
as `context_<length>.f32` (raw little-endian float32), outside timing. Use
`np.fromfile(path, dtype="<f4")` to read them. Evidence is retained in
`.build/{python,swift}_parity_full.json`, corresponding `_logits/` directories,
and `.stdout.log` / `.stderr.log` files.

The benchmark fixes a prior input discrepancy: Python now honors
`--fill-token-id` even with an embedded tokenizer, rather than substituting the
tokenizer's fallback token. It also matches the growing-cache capacity across
runners, handles either `None` or negative dynamic state
dimensions, and selects GPU preference when OS specialization is available.
`auto` still prefers Swift when its supported options and SDK are available;
`--json-output` no longer implicitly selects Python. The backend override is
now visible in `--help`.

### Advancing-Position Check

Both runners also completed `--contexts 16 --steps 8 --warmup 1 --grow-context`
with `--position-ids-layout context`,
advancing from position 16 to 24 with the same dynamic asset. The eight sampled
tokens and final logits matched exactly and remained identical on repetition.
The second comparison reversed runner order to check the timing discrepancy:

| Runner | First elapsed (s) | Repeat elapsed (s) | Approximate tok/s |
| --- | ---: | ---: | ---: |
| Python | 22.005 | 22.229 | 0.36 |
| Swift | 11.875 | 11.750 | 0.67-0.68 |

There is therefore **functional and numerical parity for this full-position
growing test, but not performance parity**. Unlike the fixed-position benchmark, these
timings include preparation for newly encountered shapes. The cause of the
additional Python latency has not been isolated. Logs, JSON, and final-logit
files are retained under `.build/{python,swift}_parity_grow*`.

These short successful runs do not establish that the earlier command-queue
failure on the full growing-context sweep is resolved. The full-position 32-step,
eight-context growing sweep was not repeated in this parity check.

## Compact-Position Decoding

The exporter computes its cache offset from `max(position_ids)` and the query
length, not the length of the position vector. For single-token decode,
`[position]` and `[0, ..., position]` therefore have the same meaning. Growing
the latter vector unnecessarily changes an input shape at every step.

Both runners now default to `--position-ids-layout query`: prefill supplies one
position per prompt token, and decode supplies just the current position.
This restores Python's original compact layout and applies it to Swift. The
`context` option preserves the previous full-range behavior for diagnostics or
models that need it. Input token and cache dimensions in the asset remain
dynamic, positions advance normally, and visible KV history grows. The
context-2048 row reaches position 2080; no fixed-length export, context padding,
or constant decode position is used in this test.

An eight-step comparison against the full-range reference produced identical
tokens and byte-identical final logits. Python's timed interval fell from
22.005 s to 0.680 s (0.36 to 11.76 tok/s); Swift's compact run took 0.501 s
(15.96 tok/s). Neither IOSurface-only storage nor descriptor-matched storage
(IOSurface inputs, Metal caches) improved the full-range Python run: they took
22.176 s and 22.350 s respectively, with identical outputs. These storage
experiments were process-local; the runner retains its NumPy-backed arrays.

### Full Growing Sweep

With the final scripts, both processes completed all eight contexts with 32
advancing decode steps and one warmup. All **256 sampled token IDs matched**,
and all eight final logit vectors were finite and byte-identical, with maximum
absolute error **0**. The previous fatal error and command-queue failure did
not reproduce on this path. As with the baseline, these were sequential
single-run measurements and do not prove a stable language-level speedup.

| Initial context | Final position | Python (tok/s) | Swift (tok/s) |
| ---: | ---: | ---: | ---: |
| 16 | 48 | 14.88 | 13.63 |
| 32 | 64 | 19.81 | 13.60 |
| 64 | 96 | 18.00 | 13.47 |
| 128 | 160 | 15.52 | 12.65 |
| 256 | 288 | 13.94 | 11.83 |
| 512 | 544 | 11.54 | 10.34 |
| 1024 | 1056 | 8.50 | 8.27 |
| 2048 | 2080 | 6.19 | 5.75 |

Reproduce the current growing-context test with:

```bash
.build/coreai-b2-env/bin/python scripts/benchmark_aimodel_sampling.py \
  artifacts/qwen_mlx0322 --runtime-backend python \
  --contexts 16,32,64,128,256,512,1024,2048 --steps 32 --warmup 1 \
  --grow-context --position-ids-layout query \
  --json-output .build/python_parity_query_grow_full.json \
  --logits-dir .build/python_parity_query_grow_full_logits
```

Use `--runtime-backend swift` with separate output paths for the other runner.
Omit `--grow-context` to measure fixed-position throughput. The final evidence
is in `.build/{python,swift}_parity_query_grow_full*`, with the short comparison
and storage experiments retained alongside it. No new conversion was needed
for this runner change.

Converted bundles now live in the Git-ignored `artifacts/` directory, including
the older `artifacts/qwen_new/` bundle. Benchmark logs and result files remain
in `.build/`; their recorded asset paths reflect the original locations before
the move. The commands above use the current locations.

The final regression suite passed: **137 tests**, including 16 benchmark tests
covering both position layouts, fixed and advancing positions, persistent state
and per-context reset, BF16 dynamic-state allocation, GPU preference gating,
synthetic token selection, backend selection, and parity artifact export. The
Swift runner compiled and executed with the current Xcode SDK. `git diff
--check` passed.

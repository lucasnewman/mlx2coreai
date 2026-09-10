# LFM2.5-2.6B Conversion

## Model and State

The installed `mlx-lm` supports LFM2.5 through `mlx_lm.models.lfm2`; `mlx-vlm`
is not needed. This port uses LiquidAI's unquantized
[MLX BF16 checkpoint](https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX-bf16),
revision `f2d32094cdd69ed7adb85a4b44accfc8770cd655`, rather than adapting the
Transformers config from the [base model](https://huggingface.co/LiquidAI/LFM2.5-2.6B).

The model contains 22 short-convolution layers and 8 full-attention layers.
The existing convolution lowering handles its kernel-size-3 depthwise Conv1d
through CoreAI Conv2d; no gated-delta or custom kernel is needed.

The stateful exporter now supports MLX-LM's one-element convolution ArraysCache:

| State | Shape | Precision |
| --- | --- | --- |
| `keyCache` | `[8, 1, 8, context, 64]` | Compute precision |
| `valueCache` | `[8, 1, 8, context, 64]` | Compute precision |
| `convState` | `[22, 1, 2, 2048]` | Compute precision |

There is no `recurrentState`. Query length and KV-cache capacity remain dynamic;
the fixed two-token convolution history is intrinsic to the model, not a
fixed-length execution workaround. Only unpadded batch-size-1 execution is
supported. Conversion requires at least one attention layer and uniform,
positive convolution-cache shapes.

## Reproduce

Test environment: M3 Max, macOS 27 build `26A428`, Xcode in
`/Applications/Xcode.app`, `coreai-core==1.0.0b2`, MLX `0.32.2`, and
`mlx-lm==0.31.3`. Commands below use `.build/coreai-b2-env/bin/python`.

Weights are downloaded under `artifacts/weights/LFM2.5-2.6B-MLX-bf16`.
Both the source checkpoint and converted assets stay in Git-ignored `artifacts/`.

```bash
.build/coreai-b2-env/bin/python -m mlx2coreai._convert_mlx_lm_stateful \
  artifacts/weights/LFM2.5-2.6B-MLX-bf16 \
  --output artifacts/lfm25_2_6b_fp32 \
  --revision f2d32094cdd69ed7adb85a4b44accfc8770cd655 \
  --max-context-length 256 --compute-precision fp32

.build/coreai-b2-env/bin/python scripts/validate_aimodel_mlx.py \
  artifacts/lfm25_2_6b_fp32 \
  --model artifacts/weights/LFM2.5-2.6B-MLX-bf16 \
  --compute-precision fp32 --prefill-chunks 3,5 --steps 16 \
  --max-relative-l2 0.01 --max-abs-error 0.01 \
  --json-output artifacts/lfm25_2_6b_fp32/validation.json
```

For an FP16 arithmetic comparison, add `--compute-precision fp16` to conversion
and validation, and use a separate output directory. This casts the same
unquantized checkpoint, not an integer-quantized model. The validation option
casts the MLX reference as well; omitting it compares against the original BF16
reference. Explicit weight casts are materialized before capture so the initial
and dynamic-probe graphs do not disagree because of lazy parameter evaluation.

Use the Python runner for the three-state contract. The current Swift benchmark
only handles the two-state attention-only contract. `--grow-context` is required:
replaying a position cannot rewind convolution history. Warmup uses a copy of
the state so it cannot consume a token in the measured sequence.

```bash
.build/coreai-b2-env/bin/python scripts/benchmark_aimodel_sampling.py \
  artifacts/lfm25_2_6b_bf16 \
  --runtime-backend python --grow-context \
  --contexts 16,32,64,128,256,512,1024,2048 --steps 32
```

## Corrected FP32 Validation

The packed KV-cache update was the correctness issue, not the convolution or
ordinary BF16 rounding. The second attention layer (model layer 5) received
zero K/V tensors internally even though its final packed state output held the
correct updated values. The export wrapper itself agrees with native MLX.

The exporter now updates each layer independently with dynamic gather/select
and stacks the updated layers only at the state output. Reading back a slice
of sequentially updated packed state caused the incorrect result; using a
per-layer `slice_update` instead triggered an MPSGraph `MPSMemrefAllocFusion`
compiler assertion. Gather/select avoids both patterns. Optimization, dynamic
query length, dynamic cache capacity, and the three-state public contract remain
enabled. This workaround processes the allocated cache extent rather than only
the new-token slice; its performance cost has not yet been benchmarked.

The initial cache-fix FP32 asset passed the same 19-call controlled
comparison (3/5/15-token prefill followed by 16 decode calls):

| Tensor | Max relative L2 | Max absolute error |
| --- | --- | --- |
| Logits | 8.47e-6 | 1.58e-4 |
| Keys | 9.05e-7 | 1.45e-5 |
| Values | 2.14e-6 | 1.82e-6 |
| Convolution state | 9.28e-6 | 1.38e-4 |

All 19 last-token choices match. Both maximum absolute and relative L2 error
are below `1e-2` for every tested call and tensor. FP32 is numerically close,
not bitwise identical. The rebuilt asset replaces the old asset at
`artifacts/lfm25_2_6b_fp32`; new metrics are in `cache_fix_validation.json`,
while `precision_comparison.json` retains the pre-fix baseline.

The real-weight seven-layer probe now measures layer 5 SDPA at `8.64e-7`
relative L2 and layer 6 convolution input product at `1.37e-6`. Regression
coverage includes two attention layers, internal cache consumers, nonzero
initial cache contents, untouched regions, and dynamic offsets and shapes.

FP16 remains unsupported on this build: the corrected tiny graph aborts in
`MPSMemrefRegion` with a `placement.region_call` outside-region value error.
Its expected-failure test runs in a subprocess so a native abort cannot take
down the test suite. The older FP16 failure below predates the cache rewrite.

The deeper tiny BF16 model still exceeds the original `0.03` absolute
convolution-state error bound (`0.03125` observed); it is an explicit expected
failure, not a relaxed tolerance. FP32 remains the precision used to isolate
conversion correctness.

A second exporter bug was found with cache capacity 2048 and a 21-token chunk:
the convolution-history slice start was dynamic, but its end retained the
trace-time constant `18`. This produced an `MPSTypeInference` reshape abort,
not a numerical mismatch. Callback capture now records slice input shapes;
dynamic probing uses them to recognize full-extent endpoints of intermediate
tensors such as `concat(history, tokens)`. The endpoint is now the source's
runtime length. Tiny FP32 tests pass 21-token chunks with this fix; the failed
full-model run is retained in `.build/lfm2_fp32_cache_fix_validation_2048.log`.

The final FP32 asset also passes that previously failing full-model case:
cache capacity 2048, prefill chunks 3/5/21, then 32 single-token decode calls.
Across all 35 calls, maximum logit error is `1.47e-4` absolute and `7.87e-6`
relative L2. Maximum absolute error across all states is `1.35e-4`; maximum
state relative L2 is `8.87e-6`. All 35 next-token choices match MLX FP32.
This uses 61 token positions in a 2048-capacity cache, not a 2048-token prompt.
Results are in `artifacts/lfm25_2_6b_fp32/final_validation_2048.json`.

```bash
.build/coreai-b2-env/bin/python scripts/validate_aimodel_mlx.py \
  artifacts/lfm25_2_6b_fp32 \
  --model artifacts/weights/LFM2.5-2.6B-MLX-bf16 \
  --compute-precision fp32 --state-capacity 2048 \
  --prompt 'Explain why the sky appears blue and why sunsets often look red. Use two short sentences.' \
  --prefill-chunks 3,5 --steps 32 \
  --max-relative-l2 0.01 --max-abs-error 0.01
```

The final BF16 asset has both fixes but does **not** meet `1e-2`: the original
19-call control measures maximum logit relative L2 `0.056165` and absolute
error `1.1875`, while all 19 last-token choices match. Maximum state relative
L2 is `0.057731` for convolution state. This is much smaller than the pre-fix
`1.110093` logit relative L2, but is still material drift. Results are in
`artifacts/lfm25_2_6b_bf16/final_validation.json`. Remaining reduced-precision
differences may include delegated-kernel numerical behavior; the FP32 control
does not establish correctness of every BF16 kernel or prove all drift is
ordinary rounding.

Final test suite: **209 passed, 5 expected failures** (two existing Qwen3.5
cases, two FP16 LFM2 cases, and the deeper BF16 LFM2 case). The exact cache test
checks internal consumers, nonzero initial state, untouched regions, and
multiple dynamic capacities. Runtime assets were replaced in place; older
diagnostic JSON/logs were retained. Superseded FP32 compiled caches and the
abandoned seven-layer probe were removed, without deleting source weights or
the current assets.

## Pre-Fix Validation

The following observations describe the original assets before the cache fix.

Tiny FP32 and BF16 real-architecture models pass dynamic prefill/decode and convolution-cache
checks across multiple query lengths and cache capacities. FP32 checks use tight
elementwise tolerances; BF16 checks bound both relative L2 and maximum absolute
error because runtime arithmetic is not bitwise identical to MLX.

The full BF16 model converts, optimizes, loads, and executes, but **does not pass
logit parity**. Its first three-token prefill has relative L2 error `0.832916`
against MLX. A diagnostic continuation with the failure threshold relaxed to
collect evidence matched 18 of 19 next-token choices, but retained large errors;
this is not a successful validation. Results are in
`artifacts/lfm25_2_6b_bf16/diagnostic_generation.json`.

Layer-state probes show close early convolution and attention states, then
amplified differences later in the network. A standalone real-weight attention
block is close to MLX, while a seven-layer prefix shows hidden-state drift up to
about 1.3%. Explicit FP32 normalization accumulation did not clearly resolve
that drift. CPU-only specialization failed to load the BF16 asset, and an
unoptimized reduced stateful asset failed at `keyCache` resolution. These probes
do not yet establish a single root cause.

FP16 is not a validated workaround. The tiny model reproducibly fails its first
convolution-state comparison after attention (`1.217285` maximum absolute
error), even after disk cleanup. This is retained as a strict expected-failure
test. Full-model FP16 conversion succeeded, but runtime compilation reported an
ANE compiler-service connection error and ultimately exhausted disk space.
That attempt provides no full-model FP16 numerical result. Its 5.5 GiB asset and
9.5 GiB process-specific compiler temporary directory were removed; the BF16
asset, original weights, and diagnostic logs remain. About 21 GiB was available
after cleanup. No unrelated caches or user artifacts were removed.

The focused diagnostic scripts and logs are under `.build/probe_lfm2_*.py` and
`.build/lfm2_*.log`. Performance benchmarking is deferred until numerical parity
is established; a successful conversion or plausible text alone is insufficient.

## Pre-Fix FP32 Control Experiment

Full FP32 conversion does **not** resolve the discrepancy. The same BF16
checkpoint values were widened to FP32, with FP32 weights, logits, KV caches,
and convolution state in the exported graph. Both dynamic dimensions and
authoring optimization remained enabled. No fixed-length workaround or different
checkpoint was introduced.

The comparison replays identical tokens in every case: prefill chunks of 3, 5,
and 15 tokens, then 16 single-token decode calls. Decode inputs are chosen by the
BF16 MLX reference and then held fixed, so divergent sampling cannot contaminate
the comparisons. The maximum errors below are across those 19 calls.

| Comparison | Max relative L2 (logits) | Max absolute logit error | Matching last-token choices |
| --- | --- | --- | --- |
| MLX FP32 vs MLX BF16 | 0.031816 | 1.105369 | 19/19 |
| CoreAI BF16 vs MLX BF16 | 1.110093 | 23.589844 | 18/19 |
| CoreAI FP32 vs MLX FP32 | 1.118896 | 23.493227 | 18/19 |

The first prefill's relative L2 is `0.823470` in FP32 versus `0.832916` in
BF16. FP32 is not bitwise-close: none of the tested output logit elements have
identical FP32 bit patterns to the matching MLX reference. Centering logits
over the vocabulary does not remove the discrepancy (maximum centered relative
L2 `0.635601`), so it is not merely an irrelevant constant offset.

The FP32 cache comparisons also fail: maximum relative L2 across calls is
`0.267057` for keys, `0.497399` for values, and `0.492882` for convolution state.
A separate first-prefill probe with cache capacity 64 localizes the boundary:

- Running the export wrapper directly in MLX gives zero maximum absolute logit
  error versus MLX's ordinary cache implementation on this input.
- In CoreAI, convolution states for layers 0, 1, 3, and 4 and KV states for
  layers 2 and 5 agree within about `2.1e-6` relative L2.
- Layer 6 convolution state jumps to `0.141566` relative L2, with errors
  propagating afterward. Layer numbers are zero-based. This brackets the issue
  between layer 5's KV projections and layer 6's convolution-state update; it
  does not yet identify the individual faulty operation.

These controls rule out ordinary BF16 rounding as the sole explanation and
point to a correctness issue after the MLX export wrapper, in graph capture,
lowering, optimization, or runtime execution. An FP32 graph alone does not prove
every delegated kernel internally uses FP32, so this is not a complete audit of
the runtime's arithmetic choices.

The approximately 11 GiB FP32 bundle is retained at
`artifacts/lfm25_2_6b_fp32`. Per-call logits/state metrics and the exact input
batches are in `artifacts/lfm25_2_6b_fp32/precision_comparison.json`. The larger
maximum BF16 error than the earlier diagnostic comes from using the MLX-chosen
token stream rather than the CoreAI-chosen stream after their first disagreement.

```bash
.build/coreai-b2-env/bin/python -m mlx2coreai._convert_mlx_lm_stateful \
  artifacts/weights/LFM2.5-2.6B-MLX-bf16 \
  --output artifacts/lfm25_2_6b_fp32 \
  --revision f2d32094cdd69ed7adb85a4b44accfc8770cd655 \
  --max-context-length 256 --compute-precision fp32

.build/coreai-b2-env/bin/python .build/probe_lfm2_precision.py
.build/coreai-b2-env/bin/python .build/probe_lfm2_states.py fp32
```

Logs: `.build/lfm2_conversion_fp32.log`,
`.build/lfm2_precision_comparison.log`, and `.build/lfm2_fp32_states.log`.

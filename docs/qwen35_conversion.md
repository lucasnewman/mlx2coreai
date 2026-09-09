# Qwen3.5-0.8B Conversion Investigation

## Status

Conversion and CoreAI asset verification succeed. **Full-model numerical
validation fails on the current runtime. These are experimental artifacts,
not a working model release.** No meaningful CoreAI throughput is reported
for incorrect output, and sequence lengths have not been fixed to work around
the failures.

Tested on an M3 Max (64 GiB), macOS 27 build 26A428, Xcode build 27A266a,
CoreAI Python 1.0.0b2, MLX/MLX Metal 0.32.2, and MLX-LM 0.31.3.
Use `.build/coreai-b2-env/bin/python`; the system Python environment has a
different CoreAI version.

## Weights And Export

Downloaded the official [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
weights at revision `2fc06364715b967f1860aea9cf38778875588b17` into
`artifacts/weights/Qwen3.5-0.8B`. MLX-LM loads these directly and its BF16
forward pass produces finite logits. This exports the text decoder, not the
vision encoder. No weight quantization is applied.

```bash
.build/coreai-b2-env/bin/python -m mlx2coreai._convert_mlx_lm_stateful \
  artifacts/weights/Qwen3.5-0.8B \
  --output artifacts/qwen35_0_8b_bf16 \
  --max-context-length 256
```

The bundle contains `qwen35_0_8b_bf16.aimodel`, the tokenizer,
`metadata.json`, and `conversion.json` (precision, state specs, optimization
status, and weight manifest). The model has one dynamic `main` entrypoint:

| State | Shape | Dtype |
| --- | --- | --- |
| keyCache | `[6, 1, 2, context, 256]` | BF16 |
| valueCache | `[6, 1, 2, context, 256]` | BF16 |
| convState | `[18, 1, 3, 6144]` | BF16 |
| recurrentState | `[18, 1, 16, 128, 128]` | FP32 |

Query length and KV capacity remain dynamic. Convolution/recurrent state sizes
are architectural constants, not fixed token lengths. Model parameter dtypes
are preserved under `--compute-precision auto`, including FP32 parameters;
only the final BF16 logits are cast to FP16 for the existing runner contract.
The manifest reports no weight downcasts.

## Gated Delta

[Apple's converter](https://github.com/apple/coreai-torch/blob/main/coreai_torch/composite_ops/_gated_delta_update.py)
exposes `gated_delta_update` as a v1 composite, rather than a generated primitive.
The MLX scalar, unmasked `gated_delta_step` custom kernel is recognized and
lowered into that composite. Equal Q/K and value head counts are supported;
masked, vector-gated, and grouped-value variants are not claimed as supported.

The adapter transposes MLX's `[B,T,H,D]` inputs to `[B,H,T,D]`, converts
positive decay to log-decay, and transposes the recurrent state from
`[B,H,Dv,Dk]` to `[B,H,Dk,Dv]`. Recurrence uses FP32. MLX has already normalized
Q/K and scaled Q. In this runtime, `use_qk_l2_norm=False` skips both operations;
this differs from the current Torch reference's unconditional query scaling.
The fallback body follows the verified runtime convention.

The [model-zoo recipes](https://github.com/john-rocky/coreai-model-zoo/tree/main/models/qwen3.5)
and [published CoreAI artifacts](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI)
were references only. Their quantization and fixed-query/decode recipes were
not used as a substitute for the MLX conversion.

## Runtime Failures

1. The standalone native composite passes FP32 CPU/GPU numerical tests for
   sequence lengths 1 and 3, including 128-by-128 state. But when integrated
   into an actual MLX GatedDeltaNet layer, some per-token output elements are
   zeroed/incorrect. Q/K/V and final recurrent state match in the reduced
   single-layer repro. Native substitution was confirmed by a diagnostic-only
   replacement of the fallback body; the native result was unchanged.
2. Swift and Python reproduce the same integrated native result. In the
   reduced FP32 layer, maximum error was about `1.03e-4` before gated RMSNorm
   and `0.01824` after output projection. The error is large relative to these
   small intermediate activations, not merely BF16 rounding.
3. Full BF16 Qwen3.5 validation failed on its first three-token prefill:
   relative L2 error approximately `0.957`, maximum logit error `19.55`, and
   a different greedy token. A single-token prefill also failed. It is not
   a usable generation model on this path yet.
4. `--gated-delta-implementation decomposed` retains the dynamic recurrence
   but prevents native substitution. It matches MLX for the standalone
   GatedDeltaNet layer. In the tiny full hybrid model, specialization aborts
   in `ANERegionFormationPass` with `operand #0 does not dominate this use`.
   Explicit loop-carried invariants did not avoid that failure.

Skipping authoring optimization, enabling debug specialization, independently
initializing the loop output, changing slice-update to scatter, and
materializing native inputs/outputs did not repair the integrated native
failure. These unsuccessful materialization experiments are not retained.

The decomposed experimental bundle is at `artifacts/qwen35_0_8b_bf16_decomposed`.
It can be regenerated with the command above, that output directory, and
`--gated-delta-implementation decomposed`. Do not treat it as a validated
full-model fallback.

## Loop-Free Recurrence Probe (2026-09-09)

A bounded follow-up replaced the recurrence with an inline FP32 chunked
matrix formulation. It emits neither the native gated-delta composite nor
a runtime `while` loop. Query and KV dimensions remain dynamic. The diagnostic
uses 12 fixed matrix-doubling rounds for the triangular inverse (an algebraic
query limit of 4096, not a fixed query shape); only small queries were tested.
It is not a production conversion mode or a validated large-chunk algorithm.

Results on the same macOS 27 build 26A428 / CoreAI b2 runtime:

- All 12 standalone CPU/GPU numerical cases passed: query lengths 1 and 3,
  with key/value dimensions 32/16, 32/32, and 128/128.
- The complete unquantized FP32 MLX GatedDeltaNet layer passed numerical
  comparison, including its captured intermediate outputs.
- The four-layer, four-state FP32 hybrid model still aborted during GPU
  specialization, before its first prefill could finish. The failure is now
  `MPSMemrefRegion`: `'placement.region_call' op using value defined outside
  the region`. Preceding diagnostics mention `ConvertBinaryCompareToZero`
  and an unsupported ANE I/O cast. An uncaptured rerun reproduced the abort.

The saved hybrid source asset passes CoreAI IR verification. Inspection
confirms zero `coreai.while` operations and zero gated-delta composite
references, so the failing graph is genuinely the loop-free replacement.

This does not establish a usable full-model path. Investigation stopped at
the small hybrid failure; no full Qwen3.5 weight re-export or benchmark was
attempted, and converter defaults were not changed. The failure does not
require BF16 weights or native gated-delta substitution.

The isolated probe is retained locally at `.build/probe_chunked_gated_delta.py`.
It replaces the lowering function only in its own process. Test parameter
names and scratch conversion metadata still say `native`/`decomposed`; those
labels do not describe the injected probe implementation. Reproduce with:

```bash
.build/coreai-b2-env/bin/python .build/probe_chunked_gated_delta.py \
  tests/test_gated_delta.py -k 'dynamic_runtime and decomposed' -q
.build/coreai-b2-env/bin/python .build/probe_chunked_gated_delta.py \
  tests/test_hybrid_stateful.py -k 'live_gated_delta_net and decomposed' -q
# Runs in a separate process because the beta compiler aborts:
.build/coreai-b2-env/bin/python .build/probe_chunked_gated_delta.py \
  tests/test_hybrid_stateful.py -k tiny_qwen35 --runxfail -q -s
```

Logs: `.build/chunked_delta_unit.log`, `.build/chunked_delta_layer.log`, and
`.build/chunked_delta_hybrid_uncaptured.log`. The probe, logs, and tiny model
assets remain ignored by Git.

## Reproduce

This compares CoreAI with MLX on identical tokens, using chunked prefill and
subsequent decode. It fails loudly on nonfinite logits, shape differences,
or excessive relative L2 error:

```bash
.build/coreai-b2-env/bin/python scripts/validate_aimodel_mlx.py \
  artifacts/qwen35_0_8b_bf16 \
  --model artifacts/weights/Qwen3.5-0.8B \
  --prefill-chunks 3,5 --steps 8
```

Reduced tests require no downloaded weights:

```bash
.build/coreai-b2-env/bin/python -m pytest \
  tests/test_gated_delta.py tests/test_hybrid_stateful.py -q

# Expose the native runtime failures rather than treating them as expected:
.build/coreai-b2-env/bin/python -m pytest tests/test_hybrid_stateful.py --runxfail -q
```

The two native integration regressions are strict expected failures, limited
to numerical assertions, so a fixed runtime will surface as XPASS rather than
silently remaining marked broken. The aborting decomposed hybrid runtime test
is not included in the default suite. Its standalone layer is covered.

For future benchmarks use `--runtime-backend python --grow-context` with the
four-state bundles. Python isolates recurrent warmup state so warmup does not
consume history. The existing Swift benchmark still requires two states;
the Swift result above used a separate reduced, stateless layer probe.

Diagnostics from this investigation are in `.build/qwen35_validation.log`,
`.build/gdn_swift.log`, and `.build/hybrid_model_test.log`. Artifacts, downloaded
weights, and scratch diagnostics remain ignored by Git.

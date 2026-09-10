# Qwen3.5

Convert the text-only hybrid decoder from `Qwen/Qwen3.5-0.8B` using MLX-LM.
The recipe includes attention caches, convolution state, and gated-delta
recurrence state. It does not export the vision encoder.

**Experimental:** export works, but reliable decoding/logit parity remains
unresolved. Neither BF16 nor FP32 conversion is a known correctness fix.
The run commands below are for diagnostics, not a validated deployment path.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. From the
repository root:

```bash
pip install -e .
```

## Build

```bash
python -m recipes.qwen35 convert --output artifacts/recipes/qwen35
```

Missing checkpoint files are downloaded as needed. The default compute policy
is `auto`, retaining source BF16 precision with FP32 recurrent state. No
additional quantization is applied. Supply a local directory or Hub ID after
`convert` to override the source, and `--revision` to pin a Hub version.

To isolate precision or recurrence-lowering behavior, export to a separate path:

```bash
python -m recipes.qwen35 convert --compute-precision fp32 \
  --gated-delta-implementation decomposed \
  --output artifacts/recipes/qwen35_fp32_decomposed
```

The default gated-delta implementation is `native`; `decomposed` is an
alternative diagnostic lowering, not a guarantee of correct execution.

## Run and Validate

```bash
python -m recipes.qwen35 run artifacts/recipes/qwen35 --allow-experimental \
  --chat --prompt "What is the capital of France?" --max-new-tokens 8
```

The CLI requires `--allow-experimental` before it opens the executable. Runtime
loading or execution can still fail. Add `--validate-mlx --prefill-chunks 3,5`
to compare logits and state against the source model; parity failures are
currently expected, and validation timings are not inference benchmarks.

Query length and cache capacity remain dynamic; this recipe does not require
fixed-length execution. See the [shared language-model guide](../../docs/lm_recipes.md)
for capacity, sampling, reports, and Python execution. Python callers must also
explicitly opt into experimental requests.

[All recipes](../README.md)

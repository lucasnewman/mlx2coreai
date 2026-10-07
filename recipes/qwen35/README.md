# Qwen3.5

Convert the text-only hybrid decoder from `Qwen/Qwen3.5-0.8B` using MLX-LM.
The recipe includes attention caches, convolution state, and gated-delta
recurrence state. It does not export the vision encoder.

This recipe is experimental and requires `--allow-experimental` to run.
Use FP32 with decomposed recurrence and an explicit state capacity, as shown
below. Validate new checkpoint, precision, or capacity settings against MLX.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. From the
repository root:

```bash
pip install -e .
```

## Build

```bash
python -m recipes.qwen35 convert mlx-community/qwen3.5-0.8B-bf16 \
  --compute-precision fp32 \
  --gated-delta-implementation decomposed \
  --output artifacts/recipes/qwen35_fp32_decomposed
```

Missing checkpoint files are downloaded as needed. Supply a local directory
or Hub ID after `convert` to choose a source, and `--revision` to pin a Hub
version. No additional quantization is applied.

The recipe defaults to `auto` precision and `native` recurrence. The example
above selects FP32 and decomposed recurrence explicitly.

## Run and Validate

```bash
python -m recipes.qwen35 run artifacts/recipes/qwen35_fp32_decomposed --allow-experimental \
  --chat --prompt "What is the capital of France?" --max-new-tokens 8 \
  --state-capacity 128
```

Add `--validate-mlx --prefill-chunks 3,5` to compare logits and state against
the source model with chunked prefill. Capacity must cover the prompt and
generation budget. Validation timings include reference execution.

Query length and cache capacity remain dynamic; this recipe does not require
fixed-length execution. See the [shared language-model guide](../../docs/lm_recipes.md)
for capacity, sampling, reports, and Python execution. Python callers must also
explicitly opt into experimental requests.

[All recipes](../README.md)

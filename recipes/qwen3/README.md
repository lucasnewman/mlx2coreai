# Qwen3

Convert an MLX-LM Qwen3 checkpoint into a stateful CoreAI text-generation bundle.
The default `mlx-community/Qwen3-0.6B-bf16` checkpoint has validated FP32
generation. Query length and KV-cache capacity remain dynamic.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. From the
repository root, install the project and its MLX-LM dependency:

```bash
pip install -e .
```

## Build

```bash
python -m recipes.qwen3 convert --output artifacts/recipes/qwen3_fp32
```

Missing weights are downloaded as needed. Conversion defaults to FP32, even
though the source weights are BF16; no additional quantization is applied.
To use another checkpoint, supply a local directory or Hub ID after `convert`,
optionally with `--revision` to pin its version. Validation of the default
checkpoint does not guarantee correctness for every Qwen3 variant.

## Run

```bash
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France?" --max-new-tokens 32
```

The command uses the packaged tokenizer and prints a JSON report containing
generated text, token IDs, and timings. It does not reload the source weights.
Generation is greedy by default and stops at EOS or the token budget.

## Validate

```bash
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France?" --prefill-chunks 3,5 \
  --max-new-tokens 32 --state-capacity 2048 --validate-mlx \
  --json-output artifacts/recipes/qwen3_fp32/validation.json
```

This compares logits and cache state against MLX. Validation loads both models;
omit `--validate-mlx` when measuring inference performance.

See the [shared language-model guide](../../docs/lm_recipes.md) for precision,
sampling, dynamic state capacity, and Python execution, or the
[recipe API guide](../../docs/recipe_api.md) to build and load bundles in Python.

[All recipes](../README.md)

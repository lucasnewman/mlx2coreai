# LFM2 / LFM2.5

Convert MLX-LM LFM models with attention and short-convolution state. The
default checkpoint is `LiquidAI/LFM2.5-2.6B-MLX-bf16`; the same recipe also
accepts the `LiquidAI/LFM2-8B-A1B` mixture-of-experts model.

Use FP32 with byte-backed inputs and state for LFM2.5. These are the CLI
defaults; computation runs on the GPU. The run example below uses an explicit
state capacity of 257.

LFM2 MoE execution is experimental and requires `--allow-experimental`.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. From the
repository root:

```bash
pip install -e .
```

## Build LFM2.5

```bash
python -m recipes.lfm2 convert --output artifacts/recipes/lfm25_fp32
```

Missing weights are downloaded as needed. Conversion defaults to FP32 without
additional quantization. Supply a local checkpoint directory or Hub ID after
`convert` and optionally `--revision` to select another source. FP32 is the
correctness baseline, not a promise of parity; reduced-precision execution
can fail validation or compilation.

## Run LFM2.5

```bash
python -m recipes.lfm2 run artifacts/recipes/lfm25_fp32 --chat \
  --prompt "What is the capital of France?" --max-new-tokens 32 \
  --state-capacity 257
```

The runner uses the packaged tokenizer and maintains both attention and
convolution state. Query length and cache capacity remain dynamic. The JSON
report includes generated text, token IDs, and timings.

Add `--validate-mlx --prefill-chunks 3,5` to compare logits and state against
MLX. Python callers should use `bundle.session(storage_kind="bytes", ...)`.
Validation loads both models and should not be used to measure inference
performance.

## Build and Run LFM2 MoE

The FP32 MoE asset is approximately **32 GiB**, with additional conversion
memory and disk requirements. It may not fit alongside its MLX reference.

```bash
python -m recipes.lfm2 convert LiquidAI/LFM2-8B-A1B \
  --revision c1c44ff9fc00db3ebf4516970563f5f383d23670 \
  --output artifacts/recipes/lfm2_moe_fp32

python -m recipes.lfm2 run artifacts/recipes/lfm2_moe_fp32 \
  --allow-experimental --chat --prompt "Hello!" --max-new-tokens 8
```

Add `--validate-mlx` to compare logits and state against the source model if
there is enough memory for both implementations.

See the [shared language-model guide](../../docs/lm_recipes.md) for precision,
capacity, sampling, and Python execution, and the
[recipe API guide](../../docs/recipe_api.md) for bundle/session usage.

[All recipes](../README.md)

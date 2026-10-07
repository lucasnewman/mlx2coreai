# mlx2coreai

Experimental MLX to [CoreAI](https://developer.apple.com/documentation/coreai/) conversion.

Capture MLX graphs, lower supported ops to CoreAI, and build executable
`.aimodel` assets. Model recipes package conversion, tokenizers, state, and
execution into a reusable bundle.

This project uses a beta SDK. Start with a validated recipe; successful export
alone does not guarantee correct execution.

## Install

```bash
pip install mlx2coreai
```

Requires Python 3.11+. GPU-preferred execution requires macOS 27 and compatible
CoreAI developer tools. To use the scripts and recipes from this checkout,
run `pip install -e .` from the repository root. Audio recipes list their
additional dependencies in their own guides.

## Quick Start

Build the default Qwen3-0.6B checkpoint and generate text:

```bash
python -m recipes.qwen3 convert --output artifacts/recipes/qwen3_fp32
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France?" --max-new-tokens 32
```

Missing weights are downloaded during conversion. Keep generated bundles in
the git-ignored `artifacts/` directory. See the [Qwen3 guide](recipes/qwen3/README.md)
for validation and configuration.

## Model Recipes

Each guide includes setup, build, run, and validation instructions.

| Recipe | What it does | Compatibility |
| --- | --- | --- |
| [Qwen3](recipes/qwen3/README.md) | Stateful text generation | Default 0.6B checkpoint validated in FP32 |
| [Qwen3.5](recipes/qwen35/README.md) | Hybrid text decoder | Experimental; use FP32 with decomposed recurrence |
| [LFM2 / LFM2.5](recipes/lfm2/README.md) | Dense and MoE text generation | LFM2.5 FP32 with byte-backed state; MoE experimental |
| [Mimi](recipes/mimi/README.md) | Offline audio encode/decode | FP32; not streaming |
| [Pocket TTS](recipes/pocket_tts/README.md) | Stateful streaming speech generation | FP32; precomputed voices |
| [SmartTurn v3](recipes/smart_turn/README.md) | Speech endpoint detection from mel features | FP32; dynamic batches; preprocessing outside asset |

## Further Usage

- [Recipe overview](recipes/README.md): the shared build/run interface and where to start.
- [Language-model options](docs/lm_recipes.md): sampling, dynamic state, Python generation, and validation.
- [Recipe API](docs/recipe_api.md): build/load bundles, manage sessions, and create a recipe.
- [Generic conversion and execution](docs/conversion.md): convert an MLX function or run an asset directly.

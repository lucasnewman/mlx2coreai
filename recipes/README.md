# Model Recipes

A recipe owns the model-specific conversion and execution behavior:
`build(...)` produces a conversion plan, and `run(session, Request(...))`
executes a request. Start with the guide for your model.

| Recipe | Default source | Runtime output |
| --- | --- | --- |
| [Qwen3](qwen3/README.md) | `mlx-community/Qwen3-0.6B-bf16` | Token IDs |
| [Qwen3.5](qwen35/README.md) | `Qwen/Qwen3.5-0.8B` | Token IDs |
| [LFM2 / LFM2.5](lfm2/README.md) | `LiquidAI/LFM2.5-2.6B-MLX-bf16` | Token IDs |
| [Mimi](mimi/README.md) | Local Mimi codec checkpoint | Offline codes or audio arrays |
| [Pocket TTS](pocket_tts/README.md) | `mlx-community/pocket-tts` | Streaming audio chunks |
| [SmartTurn v3](smart_turn/README.md) | `mlx-community/smart-turn-v3` | Logits, probabilities, and endpoint decisions |
| [Parakeet Redux](parakeet_redux/README.md) | `moondream/parakeet-redux` | Transcript and token timestamps |

Each README covers dependencies, build/run examples, validation, and known
limitations. LFM2 MoE instructions are in the LFM guide.

## Shared Conventions

Run shell examples from the repository root. Keep generated bundles in
`artifacts/`, which is ignored by Git. Recipe bundles use `manifest.json`
and are executed through their recipe.

Build settings describe the checkpoint and conversion policy. Requests carry
runtime inputs, sampling settings, and generation budgets. Opening a bundle
does not reload the source checkpoint.

Use sessions as async context managers. Stateful recipes reset buffers for
each request and preserve them across generation steps; requests in one
session must be serial.

## More Detail

- [Language-model options](../docs/lm_recipes.md): shared CLI flags, dynamic capacity, Python generation, and parity validation.
- [Recipe API](../docs/recipe_api.md): build/export/load examples, session lifecycle, and authoring a new recipe.
- [Project README](../README.md): installation and generic conversion entry points.

# Audio Models

Choose a model's README for dependencies, conversion, execution, and validation:

- [Mimi codec](../recipes/mimi/README.md): offline FP32 encoding and decoding with dynamic sequence lengths; not streaming.
- [Pocket TTS](../recipes/pocket_tts/README.md): FP32 streaming speech generation with persistent KV and convolution state.

Both use the [recipe API](recipe_api.md). SmartTurn is script-based instead.

## SmartTurn

SmartTurn conversion is available through the validation script rather than a
recipe. Use Python 3.11+, macOS 27, and compatible CoreAI developer tools.
From the repository root, install the project and audio dependency:

```bash
pip install -e .
pip install mlx-audio
```

Convert and validate:

```bash
python scripts/validate_smart_turn.py --model mlx-community/smart-turn-v3 \
  --output artifacts/smart_turn_v3
```

Add `--audio speech.wav` to test an audio file. The FP32 asset accepts mel
features shaped `[batch, 80, 800]` and returns logits and probabilities. Batch
size is dynamic; audio preprocessing remains outside the asset in mlx-audio.
Conversion parity does not establish turn-detection accuracy on real speech.

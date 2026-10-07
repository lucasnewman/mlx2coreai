# Audio Models

Choose a model's README for dependencies, conversion, execution, and validation:

- [Mimi codec](../recipes/mimi/README.md): offline FP32 encoding and decoding with dynamic sequence lengths; not streaming.
- [Pocket TTS](../recipes/pocket_tts/README.md): FP32 streaming speech generation with persistent KV and convolution state.
- [SmartTurn v3](../recipes/smart_turn/README.md): stateless endpoint detection from mel features with dynamic batch size.

All three use the [recipe API](recipe_api.md). Each guide describes its audio
input or preprocessing requirements; these are not interchangeable across models.

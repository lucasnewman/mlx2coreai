# Audio Models

Audio conversion requires `mlx-audio` in addition to `mlx2coreai`. Pocket TTS
also requires SentencePiece. Use a compatible CoreAI SDK and macOS 27 for
GPU-preferred execution. The examples below run from a repository checkout.

## Mimi Codec

Mimi supports offline FP32 encoding and decoding with dynamic sequence lengths.
Streaming encode/decode is not exported: independent chunks do not retain
attention history and are not equivalent to streaming.

Obtain mlx-audio's Mimi checkpoint, such as
`tokenizer-e351c8d8-checkpoint125.safetensors` from `kyutai/moshiko-mlx-q4`, then
pass its local path to the recipe. The codec checkpoint is floating-point;
the repository's `q4` suffix describes its language model.

```python
from mlx2coreai.recipe import export
from recipes import mimi

bundle = export(mimi.build("/path/to/tokenizer-e351c8d8-checkpoint125.safetensors"),
                "artifacts/recipes/mimi_fp32")
```

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import mimi

bundle = Bundle.open("artifacts/recipes/mimi_fp32")
options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def roundtrip(audio):
    async with bundle.session(specialization_options=options) as session:
        codes = await mimi.run(session, mimi.Request("encode", audio))
        return await mimi.run(session, mimi.Request("decode", codes))
```

Input audio must be mono, 24 kHz, shaped `[1, 1, samples]`, with a positive
multiple of 1920 samples. Resample and right-pad before encoding, then trim
decoded audio to the original length. Codes are integers shaped `[1, 32, frames]`
in `[0, 2048)`. Runtime requests return NumPy arrays. These assets are stateless;
no state reset is needed.

To convert and validate each component from the command line:

```bash
python scripts/convert_mimi.py --weights /path/to/mimi.safetensors \
  --component encode --output artifacts/recipes/mimi_fp32
python scripts/convert_mimi.py --weights /path/to/mimi.safetensors \
  --component decode --output artifacts/recipes/mimi_fp32
```

Add `--audio speech.wav` to include an audio fixture, or use `--frames 2,3,1,5,13`
to choose tested sequence lengths.

## Pocket TTS

Pocket TTS generates 24 kHz mono speech with persistent KV and convolution state.
Its FP32 pipeline uses precomputed voice conditioning; arbitrary voice cloning
is not exported.

```bash
python scripts/convert_pocket_tts.py --model mlx-community/pocket-tts \
  --output artifacts/recipes/pocket_tts_fp32
python scripts/run_pocket_tts.py artifacts/recipes/pocket_tts_fp32 \
  --text "Hello! This is Pocket TTS running with Core AI." \
  --output artifacts/recipes/pocket_tts_fp32/speech.wav
```

The converter downloads missing files and defaults to the Alba voice and one
flow integration step. Set `--voice`, `--steps`, and optionally `--revision`
during conversion. Rebuild the entire bundle when changing the checkpoint or
flow-step configuration. Older bundles without the recipe manifest need
reconversion.

The CLI writes a WAV after generation; it is not a live audio player. Supply a
punctuated utterance and use `--max-frames` to bound generation. Text is processed
as one prompt, without automatic sentence splitting. Use `--help` for sampling,
EOS, and validation options. `--validate-mlx` is a correctness check and adds
reference-model execution to the timings.

For streaming application consumption:

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import pocket_tts

bundle = Bundle.open("artifacts/recipes/pocket_tts_fp32")
options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def speak(consume):
    async with bundle.session(specialization_options=options, storage_kind="metal") as session:
        async for audio in pocket_tts.run(session, pocket_tts.Request(text="Hello world.")):
            consume(audio)
```

Each request resets state and allocates capacity for its generation budget.
Requests within a session must be serial; use separate sessions for concurrent
utterances. Runtime uses the packaged tokenizer and voice resources, without
loading MLX or the original model weights.

## SmartTurn

SmartTurn conversion is available through the validation script rather than a
recipe:

```bash
python scripts/validate_smart_turn.py --model mlx-community/smart-turn-v3 \
  --output artifacts/smart_turn_v3
```

Add `--audio speech.wav` to test an audio file. The FP32 asset accepts mel
features shaped `[batch, 80, 800]` and returns logits and probabilities. Batch
size is dynamic; audio preprocessing remains outside the asset in mlx-audio.
Conversion parity does not establish turn-detection accuracy on real speech.

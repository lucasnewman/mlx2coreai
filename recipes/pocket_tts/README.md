# Pocket TTS

Generate 24 kHz mono speech from mlx-audio's Pocket TTS using an FP32 CoreAI
pipeline. KV caches and convolution state persist across generation steps.
The bundle uses precomputed voice conditioning; arbitrary voice cloning is
not exported.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. Run these
commands from the repository root:

```bash
pip install -e .
pip install mlx-audio sentencepiece
```

## Build

```bash
python -m recipes.pocket_tts convert --model mlx-community/pocket-tts \
  --output artifacts/recipes/pocket_tts_fp32
```

Missing files are downloaded automatically. The defaults are the Alba voice
and one flow integration step. Set `--voice`, `--steps`, and optionally
`--revision` during conversion. Rebuild the entire bundle when changing the
checkpoint, voice, or flow-step configuration; use a separate output directory
to keep a previous configuration.

## Run

```bash
python -m recipes.pocket_tts run artifacts/recipes/pocket_tts_fp32 \
  --text "Hello! This is Pocket TTS running with Core AI." \
  --max-frames 150 --output artifacts/recipes/pocket_tts_fp32/speech.wav
```

The CLI writes a WAV and a sibling JSON report after generation; it is not a
live audio player. Supply a punctuated utterance and use `--max-frames` to bound
generation. Text is processed as one prompt without automatic sentence
splitting. Use `--help` for temperature, seed, prefill, and EOS controls.

## Stream in Python

The runtime yields one-dimensional FP32 NumPy audio chunks. This example
prints each chunk's shape; replace that operation with your audio consumer:

```python
import asyncio
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import pocket_tts

async def main():
    bundle = Bundle.open("artifacts/recipes/pocket_tts_fp32")
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    report = {}
    async with bundle.session(specialization_options=options, storage_kind="metal") as session:
        request = pocket_tts.Request(text="Hello world.", max_frames=150)
        async for audio in pocket_tts.run(session, request, report=report):
            print(audio.shape)
    print(report)

asyncio.run(main())
```

Each request resets state and allocates capacity for its generation budget.
Requests in one session must be serial; concurrent utterances need separate
sessions. Runtime uses the packaged tokenizer and voice resources without
loading MLX or the original weights. Voice and flow steps are build settings,
not request options.

## Validate

Add `--validate-mlx` to the run command to compare component outputs and state
against MLX. Use `--source /path/to/checkpoint` if the source checkpoint moved.
Validation requires the original weights, adds reference execution to timings,
and is not an inference benchmark.

Validation replays the MLX components on the same inputs and checks their
outputs and state. The CLI uses full-precision MLX. For Python validation,
set `MLX_ENABLE_TF32=0` before any MLX kernels run.

See the [recipe API guide](../../docs/recipe_api.md) for Python build/export
and session lifecycle details.

[All recipes](../README.md)

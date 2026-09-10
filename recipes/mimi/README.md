# Mimi Codec

Offline FP32 audio encoding and decoding from mlx-audio, with dynamic sequence
lengths. These assets are **not streaming**: independent chunks do not retain
attention history and are not equivalent to streaming encode/decode.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. Run these
commands from the repository root:

```bash
pip install -e .
pip install mlx-audio
```

Obtain mlx-audio's Mimi checkpoint, such as
`tokenizer-e351c8d8-checkpoint125.safetensors` from `kyutai/moshiko-mlx-q4`, and
use its local path below. The codec checkpoint is floating-point; the
repository's `q4` suffix describes its language model. The recipe does not
download weights automatically.

## Build

```python
from mlx2coreai.recipe import export
from recipes import mimi

bundle = export(
    mimi.build("/path/to/tokenizer-e351c8d8-checkpoint125.safetensors"),
    "artifacts/recipes/mimi_fp32",
)
```

This exports both `encode` and `decode` into one recipe bundle.

## Run

This runnable smoke test encodes and decodes two frames of silence:

```python
import asyncio
import numpy as np
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import mimi

async def main():
    bundle = Bundle.open("artifacts/recipes/mimi_fp32")
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    audio = np.zeros((1, 1, 2 * 1920), dtype=np.float32)
    async with bundle.session(specialization_options=options) as session:
        codes = await mimi.run(session, mimi.Request("encode", audio))
        decoded = await mimi.run(session, mimi.Request("decode", codes))
    print("Codes:", codes.shape, "Audio:", decoded.shape)

asyncio.run(main())
```

Input audio must be mono, 24 kHz, shaped `[1, 1, samples]`, with a positive
multiple of 1920 samples. Resample and right-pad real audio before encoding,
then trim the decoded audio to the original length. Codes are integers shaped
`[1, 32, frames]` in `[0, 2048)`. Both requests return NumPy arrays and require
no state reset. The codec is lossy; conversion parity compares CoreAI with
MLX, not decoded audio with the original waveform.

## Validate

These commands each convert and validate one component. Run both to populate
a complete bundle:

```bash
python scripts/convert_mimi.py --weights /path/to/mimi.safetensors \
  --component encode --output artifacts/recipes/mimi_fp32
python scripts/convert_mimi.py --weights /path/to/mimi.safetensors \
  --component decode --output artifacts/recipes/mimi_fp32
```

Add `--audio speech.wav` to include a real audio fixture, or
`--frames 2,3,1,5,13` to test other sequence lengths. Results are saved as
`encode_validation.json` and `decode_validation.json` in the bundle directory.

See the [recipe API guide](../../docs/recipe_api.md) for bundle/session usage.

[All recipes](../README.md)

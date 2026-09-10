# SmartTurn v3

Detect whether a speaker has finished their turn using mlx-audio's SmartTurn
model. The recipe exports an FP32 feature-to-endpoint network with dynamic
batch size. Audio preprocessing remains outside the CoreAI asset.

## Setup

Use Python 3.11+, macOS 27, and compatible CoreAI developer tools. From the
repository root:

```bash
pip install -e .
pip install mlx-audio
```

MLX and mlx-audio are needed for conversion and audio-based validation, not
for execution from prepared features.

## Build

```bash
python -m recipes.smart_turn convert --output artifacts/recipes/smart_turn_fp32
```

The default checkpoint is `mlx-community/smart-turn-v3`; missing weights are
downloaded as needed. Supply a local directory or Hub ID after `convert` to
override it, and `--revision` to pin a Hub version. Conversion uses FP32 without
additional quantization. Authoring optimization is disabled because the beta
optimizer can break bias reshapes for batches larger than one.

For Python conversion:

```python
from mlx2coreai.recipe import export
from recipes import smart_turn

bundle = export(smart_turn.build(), "artifacts/recipes/smart_turn_fp32")
```

## Run

Inputs are finite floating-point mel features shaped `[batch, 80, 800]`, with
a positive batch size. Use mlx-audio's `prepare_input_features` output (adding
a batch dimension), not raw waveform samples or an arbitrary spectrogram.
The default processor resamples to 16 kHz, keeps the last eight seconds,
left-pads shorter audio, and normalizes it before computing mel features.

Given a NumPy `.npy` feature batch:

```bash
python -m recipes.smart_turn run artifacts/recipes/smart_turn_fp32 \
  --features features.npy --json-output artifacts/recipes/smart_turn_fp32/result.json
```

The result contains `logits`, `probability`, and `prediction`, each shaped
`[batch, 1]`. A prediction of `1` means an endpoint was detected. The rule is
strictly `probability > threshold`, using the checkpoint's default threshold
(normally 0.5). Override it per request with `--threshold`.

This Python smoke test uses zero-valued features to demonstrate the runtime
API. Zero features are not equivalent to preprocessing silence:

```python
import asyncio
import numpy as np
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle
from recipes import smart_turn

async def main():
    bundle = Bundle.open("artifacts/recipes/smart_turn_fp32")
    options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())
    features = np.zeros((2, 80, 800), dtype=np.float32)
    async with bundle.session(specialization_options=options) as session:
        result = await smart_turn.run(session, smart_turn.Request(features, threshold=0.5))
    print(result)

asyncio.run(main())
```

Results are NumPy arrays. Requests are stateless and need no state reset.
Opening a bundle does not load source weights, MLX, or mlx-audio. Feature
dimensions and the default threshold are recorded in bundle metadata.

## Validate

```bash
python -m recipes.smart_turn validate --model mlx-community/smart-turn-v3 \
  --output artifacts/recipes/smart_turn_fp32
```

This builds the bundle and compares logits, probabilities, and endpoint
decisions against MLX. Cases cover silence, noise, a tone, resampling, cropping,
and batch sizes 1, 2, and 3. Add `--audio speech.wav` to include real audio.
The default `--atol` and `--rtol` are both `1e-4`; results are written to
`validation.json` in the bundle directory.

Validation also checks the source model's audio-to-endpoint API. Matching MLX
does not establish turn-detection accuracy on real speech. This recipe is not
a streaming VAD or a stateful audio-window manager; callers choose when to
submit each window.

See the [recipe API guide](../../docs/recipe_api.md) for bundle/session usage.

[All recipes](../README.md)

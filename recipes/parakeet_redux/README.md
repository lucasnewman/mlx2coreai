# Parakeet Redux

Offline speech recognition from [`moondream/parakeet-redux`](https://huggingface.co/moondream/parakeet-redux)
using the [mlx-audio implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/redux.py).
The bundle contains a Conformer encoder with dynamic mel sequence length, a
single-step TDT decoder, and its vocabulary. The recipe returns text and
token timestamps.

## Setup

```bash
pip install -e .
pip install 'mlx-audio>=0.5.6'
```

Use macOS 27 with compatible CoreAI developer tools and GPU-preferred execution.
Conversion and optional audio preprocessing use MLX. Running from prepared
mel features needs only CoreAI and NumPy; opening a bundle never reloads the
checkpoint.

## Build and Transcribe

```bash
python -m recipes.parakeet_redux convert --output artifacts/recipes/parakeet_redux
python -m recipes.parakeet_redux run artifacts/recipes/parakeet_redux \
  --audio speech.wav --json-output artifacts/recipes/transcript.json
```

Conversion accepts a local checkpoint or Hugging Face source and `--revision`.
The source's ternary weights are losslessly decompressed to FP32 before
capture: the default bundle is approximately 2.4 GiB. Conversion requires
more memory than normal ternary MLX inference. Authoring optimization is
disabled for the beta runtime.

To run without MLX or mlx-audio imports, pass a `.npy` mel array:

```bash
python -m recipes.parakeet_redux run artifacts/recipes/parakeet_redux \
  --mel mel.npy
```

Mel features must use the source's `log_mel_spectrogram` preprocessing, in
FP32, with shape `[time, 128]` or `[1, time, 128]`. Redux uses 16 kHz audio,
10 ms feature hops, preemphasis, and normalization over valid frames only.
Its preprocessing appends one zero frame, so valid length defaults to
`time - 1`. `--length` can specify a shorter valid prefix; masking applies
after each strided convolution and inside every Conformer layer. Audio
preprocessing remains outside the CoreAI assets.

## Python

```python
import asyncio
import numpy as np
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle, export
from recipes import parakeet_redux

# Build once, or open the existing bundle below.
# export(parakeet_redux.build(), "artifacts/recipes/parakeet_redux")
bundle = Bundle.open("artifacts/recipes/parakeet_redux")
options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def transcribe():
    async with bundle.session(specialization_options=options) as session:
        report = {}
        result = await parakeet_redux.run(
            session, parakeet_redux.Request(np.load("mel.npy")), report=report,
        )
        print(result["text"])
        print(result["tokens"])  # id, text, start, duration, end (seconds)
        print(report)

asyncio.run(transcribe())
```

Each request creates fresh decoder hidden/cell arrays. Blank emissions retain
the previous decoder state and advance at least one encoder frame; nonblank
emissions update it. Nonblank zero-duration emissions retry the same frame.
Like mlx-audio Redux, total steps are bounded by
`max_symbols_per_step * encoded_valid_frames`. `result["completed"]` is false
and `report["budget_exhausted"]` is true if the bound is reached early.
Special tokens update decoder state but are omitted from the transcript.

One request accepts one utterance. This recipe uses greedy offline decoding;
it does not implement streaming, overlapping chunk merging, the unused VAD
head, or beam search. For long audio, memory grows with full-sequence
attention. Requests in one session must be serial.

## Validation

```bash
MLX_ENABLE_TF32=0 python -m recipes.parakeet_redux validate \
  --output artifacts/parakeet_redux_validation --audio speech.wav
python -m pytest tests/test_parakeet_redux_recipe.py -q
```

Validation builds a bundle, compares encoder features to the original packed
source, checks every decoder step's logits and hidden/cell outputs, and
requires identical text, token IDs, and timestamps. It includes silence and
noise plus supplied speech files, and writes `validation.json`. The CLI
sets `MLX_ENABLE_TF32=0` before loading MLX; Python callers must set it before
the first MLX kernels run. Tests use tiny models without downloading weights.

The cached default checkpoint was validated on GPU-preferred CoreAI execution
with silence, noise, and a speech clip: identical transcripts and timestamps,
with maximum speech encoder error below `5e-7`. Tiny models and the full
checkpoint's encoder also pass CPU execution. GPU-preferred specialization
can emit ANE compilation diagnostics before falling back successfully.

Capture/probe frame counts default to `(32, 49)` and do not restrict runtime
audio length. If overriding `frames=` or `--frames`, use one count divisible
by the subsampling factor and one with remainder one, with different encoded
lengths. This prevents shape probing from confusing intermediate subsampler
dimensions with relative-position dimensions.

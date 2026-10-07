# Parakeet Redux

Offline speech recognition from [`moondream/parakeet-redux`](https://huggingface.co/moondream/parakeet-redux)
using the [mlx-audio implementation](https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/parakeet/redux.py).
The bundle contains a Conformer encoder with dynamic mel sequence length, a
single-step TDT decoder, and its vocabulary. The recipe returns text and
token timestamps.

New exports contain three runtime files:

```text
config.json
model.aimodel/       # encode and decode entry points
vocabulary.json
```

`config.json` contains only the runtime contract: model filename, entry points
and output mappings, preprocessing settings, dimensions, and TDT decoding
parameters. It contains no source paths, revision, conversion counts, or
workaround notes. Both functions share one loaded CoreAI executable. Existing
bundles with `manifest.json` and separate assets remain readable.

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
`--weight-format fp32` (the default) stores expanded weights, producing a
bundle of approximately 2.4 GiB. `--weight-format uint2` preserves the source's
264 quantized encoder modules as packed 2-bit constants with their original
FP32 group scales and biases, producing approximately 269 MiB of assets:

```bash
python -m recipes.parakeet_redux convert --weight-format uint2 \
  --output artifacts/recipes/parakeet_redux_uint2
```

The uint2 path checks exact reconstruction of every weight and fails if a
source constant cannot be matched in capture. It introduces no new weight
rounding, activation quantization, or calibration. The decoder and other
unquantized weights remain FP32. CoreAI reconstructs FP32 tensors with
`blockwise_shift_scale`; packed storage does not guarantee packed matrix
computation or proportional runtime memory savings on every backend.

The recipe uses the shared `prepare_quantized_linears` preservation helper;
it no longer has a Redux-specific packing implementation. `uint2` continues
to preserve the checkpoint's original weights.

`--weight-format uint4` and `--weight-format uint8` instead apply the shared
affine linear-weight quantizer, with group size 128, to eligible FP32 projection
weights in both encoder and decoder. These are new lossy exports and need their
own quality evaluation. They are separate from the validated uint2 preservation
path. For custom groups, exclusions, or coverage reports, use the generic
`export(..., quantization=WeightQuantization(...), quantization_report=report)`
API documented in the [recipe guide](../../docs/recipe_api.md#weight-quantization).

Both paths materialize FP32 weights temporarily for MLX capture, so conversion
requires more memory than normal ternary MLX inference. Authoring optimization
is disabled for the beta runtime. Initial specialization of the packed graph
can be slow in the tested beta SDK.

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
# export(parakeet_redux.build(weight_format="uint2"), "artifacts/recipes/parakeet_redux")
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

Pass `--weight-format uint2` to validate the packed export against MLX. To
evaluate two saved CoreAI bundles directly, use:

```bash
python -m recipes.parakeet_redux compare \
  artifacts/recipes/parakeet_redux artifacts/recipes/parakeet_redux_uint2 \
  --audio speech.wav --output artifacts/parakeet_redux_comparison.json
```

The reference must have FP32 weights. Comparison checks encoder features,
every decoder step's logits and state, token/duration decisions, transcripts,
timestamps, completion, and asset sizes. It saves the report even when a
quality check fails. Passing `--mel file.npy` evaluates prepared features
without importing MLX or mlx-audio. Audio comparisons add silence and noise
cases. This measures agreement with FP32; a labeled speech corpus is needed
to evaluate general ASR accuracy.

The packed export was compared with the existing FP32 CoreAI bundle on six
synthetic speech fixtures (three distinct phrases), silence, and noise:

| Check | Result |
| --- | --- |
| Transcripts, token IDs, timestamps | Identical in all eight cases |
| Token and duration decisions | Identical across 137 decoder steps |
| Speech encoder features, decoder logits and state | Numerically identical |
| Maximum encoder difference across all cases | `4.10e-8` |
| Maximum token-logit difference across all cases | `1.91e-5` |
| CoreAI asset size | Approximately 2.4 GiB → 269 MiB (88.8% smaller) |

No quality loss relative to FP32 was observed on this small sample. These
fixtures do not establish corpus-level word error rate, multilingual quality,
or performance improvements. The packed bundle also passes parity against
the original MLX source on these eight cases.

The combined `model.aimodel` package with runtime-only `config.json` also passes
these eight source-parity and FP32-comparison cases, with identical transcripts,
token IDs, timestamps, and decisions across 137 decoder steps.

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

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

## Experimental Neural Engine specialization

The recipe can export a fixed-shape, dense FP16 encoder for Neural Engine
specialization. The decoder remains FP32. CoreAI's preference permits fallback,
so this is mixed ANE/GPU execution, rather than an ANE-only model.

```bash
python -m recipes.parakeet_redux convert --neural-engine-frames 512 \
  --output artifacts/parakeet_redux_neural_engine
python -m recipes.parakeet_redux run artifacts/parakeet_redux_neural_engine \
  --audio speech.wav --compute-unit neural-engine \
  --specialization-report artifacts/parakeet_redux_neural_engine/placement.json
```

The runtime pads mel arrays to 512 frames and preserves the supplied valid
length. This example accepts roughly 5.1 seconds of audio. Longer inputs raise
an error; choose a larger frame count when converting. The count must be a
multiple of the subsampling factor (8 for Redux), at most 4096. Only even input
positions are used after the first stride-2 convolution, so those positions
and the host-computed subsampling lengths remain exact in FP16. `--frames`
capture/probe settings apply only to dynamic exports. Dense `--weight-format
fp32` is required: the beta ANE compiler rejects the packed model's blockwise
group quantization. Despite that weight-format name, this option stores encoder
weights and computes encoder activations in FP16. The full package is about
1.22 GiB.

Three adaptations enable this path:

- Boolean inversion captures as logical NOT rather than XOR. The encoder mask
  uses a direct greater-or-equal comparison.
- Fixed sequence shapes and FP16 tensors enable actual ANE placement.
- The host computes the valid lengths after each strided convolution. This
  avoids integer division and an observed beta ANE ceiling bug returning
  negative lengths.

For app integrations, the fixed encoder inputs are FP16: `mel` is
`[1, encoder_frames, 128]`, `positions` is `[1, encoder_frames]`, and `pos_emb`
is `[1, 2 * (encoder_frames / 8) - 1, 1024]`. `lengths` is **`[1, 3]`**, holding
`ceil(valid_mel_length / 2)`, `ceil(valid_mel_length / 4)`, and
`ceil(valid_mel_length / 8)`. Frame positions are `0 .. encoder_frames - 1`.
The output features and final length are FP16; cast each selected feature to
FP32 before calling the unchanged decoder. The config includes
`encoder_frames` and `encoder_dtype`; dynamic bundles retain their existing
FP32 mel/PE and int32 length/index contracts.

The full 512-frame model was tested with Neural Engine preference on six speech
fixtures, silence, and noise. All eight transcripts and token ID sequences
matched FP32. Seven cases also matched every duration decision and timestamp;
the longer speech clip had timestamp differences. Valid encoder feature
relative L2 differences ranged from 1.18% to 4.39%. These results reproduced
across two runs. Optimized placement diagnostics reported 2,826 ANE operations
and 75 GPU operations. These counts do not measure runtime share or establish
a speedup.

To evaluate a saved candidate and inspect its placement:

```bash
python -m recipes.parakeet_redux compare artifacts/parakeet_redux_fp32 \
  artifacts/parakeet_redux_neural_engine --audio speech.wav \
  --compute-unit neural-engine --decisions-only \
  --specialization-report artifacts/parakeet_redux_neural_engine/placement.json \
  --output artifacts/parakeet_redux_neural_engine/comparison.json
```

`--decisions-only` still requires identical token IDs, transcripts, timestamps,
and token/duration decisions. It reports numerical differences without requiring
FP32-level `allclose`. The eight-case evaluation fails this strict decision
criterion on the longer clip, so this path remains experimental. Broader ASR
quality and performance evaluation is needed before app deployment.

## Encoder performance at 30 seconds

375 **encoded** frames represent 30 seconds: `375 * 0.08 s`. The conversion
option counts **mel** frames, so use `--neural-engine-frames 3000`. The runtime
can discard the preprocessing's redundant final zero frame at this exact
bucket boundary; valid audio frames are retained.

```bash
python -m recipes.parakeet_redux convert --neural-engine-frames 3000 \
  --output artifacts/parakeet_redux_neural_engine_30s
python -m recipes.parakeet_redux benchmark-encoder \
  artifacts/parakeet_redux_neural_engine_30s --audio speech_30s.wav \
  --warmup 5 --iterations 50 --placement \
  --output artifacts/parakeet_redux_neural_engine_30s/encoding_benchmark.json
```

`benchmark-encoder` compares GPU and Neural Engine preference by default.
Use `--compute-unit gpu` for a GPU baseline. Prepared `.npy` features can be
supplied with `--mel`; `--length` specifies valid mel frames. The benchmark
prepares CoreAI input buffers once, times serial completed encoder calls with
output readback, and checks output shape, finite values, and the valid length.
It excludes audio preprocessing, positional-embedding preparation, session
opening/specialization, and decoding. First-call latency is recorded separately.
`--placement` uses a separate debug session outside the measurements.

Measured on an Apple M5 Max, macOS 27.0.1, CoreAI 1.0.0b3, with identical
3,000-frame features from a repeated local speech fixture, five warmup calls
after the initial call, and 50 timed calls per backend:

| Encoder | Median | P90 |
| --- | ---: | ---: |
| FP32, GPU preferred (dynamic baseline) | 51.17 ms | 54.11 ms |
| FP16, GPU preferred (fixed 3,000 mel frames) | 15.88 ms | 17.13 ms |
| FP16, Neural Engine preferred (same fixed model) | 89.71 ms | 90.11 ms |

GPU-preferred FP16 encoding was 3.22 times faster than the FP32 baseline.
Neural Engine preference was 5.65 times slower than GPU FP16 and 1.75 times
slower than GPU FP32, despite confirmed ANE placement. Its median real-time
factor was 0.00299 (30 seconds encoded in about 90 ms). These are encoder-only
wall times with readback, rather than end-to-end ASR latency or power measurements.
The 30-second benchmark validates finite outputs and lengths, rather than
transcript quality at that duration.

Power sampling on the same M5 Max showed a different tradeoff. With
`powermetrics` CPU/GPU/ANE samplers running at a requested 100 ms interval, two
20-second sustained windows per backend were run in GPU/ANE/ANE/GPU order.
Both sessions were warmed, inputs were reused, and outputs were read back.
Nearby settled idle windows supplied a baseline, and interval-average power
was integrated against timestamped workload boundaries.

| FP16 encoder preference | Mean latency | Additional CPU+GPU+ANE power above idle | Estimated incremental energy per encoding |
| --- | ---: | ---: | ---: |
| GPU | 15.89 ms | 68.34 W | 1.086 J |
| Neural Engine | 89.82 ms | 4.80 W | 0.431 J |

The ANE-preferred path used about **60% less incremental energy per encoding**,
despite taking 5.65 times longer. Individual trial estimates were 1.070/1.102 J
for GPU and 0.430/0.432 J for ANE. Including three-second post-workload settling
periods gave 1.087 J and 0.434 J respectively, preserving the result.
The estimate includes CPU work and GPU fallback, rather than only the active
accelerator rail. Active combined power averaged 69.90 W on GPU and 6.08 W
on ANE; the nearby idle baselines averaged 1.56 W and 1.28 W.

These are system-wide, estimated CPU+GPU+ANE energy counters with idle power
subtracted, rather than whole-device battery or wall-plug measurements.
Background activity and sustained-versus-isolated operation affect the result.
For app integration, repeat the comparison at the app's actual request cadence.

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

The reference must have FP32 weights. Comparison checks valid encoder features,
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

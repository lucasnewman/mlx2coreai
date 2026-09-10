# Mimi Codec Conversion

## Status

Mimi's **offline FP32 encoder and decoder are validated** against mlx-audio.
The assets retain dynamic audio/code sequence lengths; batch is one and all 32
codebooks are included. Streaming `encode_step` / `decode_step` are **not yet
exported**. Calling the offline assets on consecutive chunks resets attention
history and is not equivalent to streaming.

Tested with macOS 27 build 26A428, CoreAI 1.0.0b2, MLX 0.32.2, and
mlx-audio 0.5.3 on the M3 Max. Runtime specialization is GPU-preferred; that does
not prove exclusive GPU placement. ANE compiler diagnostics occur, but the
tested calls complete with finite, numerically validated outputs.

## Checkpoint and Contracts

The original `tokenizer-e351c8d8-checkpoint125.safetensors` checkpoint was already
cached under `kyutai/moshiko-mlx-q4`, revision
`18e4df760a34d5977a34517d7d1580e07acbb2f1`. The repository name describes its
language model: the Mimi codec checkpoint loaded here uses FP32 floating-point
weights, not four-bit weights. It is loaded strictly through
`Mimi.load_pytorch_weights`; no missing weights or random parameter fallback is
allowed. Integer graph constants are preserved separately.

This is mlx-audio's [Mimi implementation](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/codec/models/mimi),
not a reimplementation based on the differently named Transformers checkpoint.
Mimi's residual vector quantization is part of the codec itself, not additional
weight quantization introduced by conversion.

| Asset | Input | Output |
| --- | --- | --- |
| `artifacts/mimi_fp32/encode.aimodel` | `audio`: FP32 `[1, 1, samples]` | INT32 `[1, 32, frames]` |
| `artifacts/mimi_fp32/decode.aimodel` | `codes`: INT32 `[1, 32, frames]` | FP32 `[1, 1, samples]` |

Audio is mono, 24 kHz, with 1920 samples per codec frame (12.5 frames/second).
Encoding requires a positive multiple of 1920 samples. Right-pad shorter input
to that boundary before encoding and trim the decoded waveform to the original
length. Resampling and padding are host responsibilities. Decoder codes must
be in `[0, 2048)`. Arbitrary non-aligned input lengths, variable batch sizes, and
variable codebook counts are not part of these exported contracts.

## Validation

Both graphs are captured at two frames and probed at three. Each asset then
executes six cases: noise at 2, 3, 1, 5, and 13 frames, plus the first second of
locally synthesized speech, resampled to 24 kHz and padded to 13 frames.
The one-, five-, and thirteen-frame lengths are unseen during capture.

- Encoder: **1184/1184 indices match exactly**, across all 32 codebooks.
- Decoder: **71,040 samples compared**, maximum absolute error
  **0.00000737607**, maximum relative L2 **0.00000729608**.
- The speech decoder case has maximum absolute error **0.00000125170**.
- The offline capture adapter is separately checked against unmodified
  mlx-audio before conversion. Decoder inputs come from the native encoder;
  the corresponding CoreAI encoder checks establish identical codes for these
  same deterministic inputs.

References compare conversion parity, not the lossy codec's reconstruction
error against original audio, perceptual audio quality, long-duration behavior,
or streaming state correctness. Authoring optimization is disabled for this
first codec export; optimized Mimi graphs have not been validated.

Reports are `artifacts/mimi_fp32/{encode,decode}_validation.json`, with captured
IR alongside them. Logs are `.build/mimi_{encode,decode}.log`.

## Reproduce

Use the b2 environment rather than the shell's older CoreAI installation.
mlx-audio remains an optional dependency of these conversion scripts.

```bash
WEIGHTS="$HOME/.cache/huggingface/hub/models--kyutai--moshiko-mlx-q4/snapshots/18e4df760a34d5977a34517d7d1580e07acbb2f1/tokenizer-e351c8d8-checkpoint125.safetensors"
.build/coreai-b2-env/bin/python scripts/convert_mimi.py \
  --weights "$WEIGHTS" --component encode --frames 2,3,1,5,13 \
  --audio .build/smart_turn_complete.wav
.build/coreai-b2-env/bin/python scripts/convert_mimi.py \
  --weights "$WEIGHTS" --component decode --frames 2,3,1,5,13 \
  --audio .build/smart_turn_complete.wav
```

Omit `--audio` if the local fixture is unavailable, or supply another speech
file. The script limits each audio fixture to one second after resampling.
It converts and validates in the same invocation, failing on any nonfinite
output, shape mismatch, token mismatch, or decoder error above 0.001 absolute
or relative L2.

For execution, use `CoreAISession` with the input names in the table. Outputs
have compiler-generated names; retrieve the single value from the returned
mapping. Do not call `reset_state`: these two assets have no mutable state.

## Implementation

- Added captured constant padding and `ArgReduce` support, including min/max
  selection and rank-preserving primitive semantics.
- Decode MLX convolution's input dilation and kernel-flip fields correctly.
  One-dimensional input dilation is lowered to zero insertion, trimming, and
  ordinary grouped convolution. This handles Mimi's grouped/depthwise
  transposed convolutions without a placeholder or custom exotic operation.
  Higher-dimensional nonunit input dilation is explicitly rejected.
- Retain static channel/head dimensions when CoreAI loses them across dynamic
  slices. Traditional/interleaved RoPE now reshapes sliced pairs rather than
  trying to squeeze a dimension that the compiler considers unknown.
- Derived intermediate extents are available to range/slice attributes; tail
  trims retain relative negative bounds instead of trace-length constants.
- Decode the leading overwrite-mode field in current MLX `SliceUpdate` events;
  contiguous updates use CoreAI's native slice-update operation.
- The small offline Mimi adapter removes redundant temporary KV allocations
  and spells causal edge padding as concatenation. It retains the original
  learned modules and is checked against the source implementation. It does
  not trace or pretend to implement streaming history.

Tiny end-to-end tests require no checkpoint download and cover dynamic encode
and decode with nonzero codebooks. Operator tests separately exercise grouped
transposed convolution, padding, ArgReduce, relative trims, edge-padding
updates, and dynamic traditional RoPE.

Final regression suite: **267 passed, 5 existing expected failures**.
`git diff --check` passes. No changes were made to installed mlx-audio sources.

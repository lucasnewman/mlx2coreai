# Pocket TTS Conversion

The implementation now lives in `recipes/pocket_tts/`. The two existing scripts
are compatibility commands for its build/runtime API; parity checks are a
separate session observer. See [recipe authoring](../recipes/README.md).
New recipe bundles use the generic manifest format and require reconversion
of older Pocket bundles. Migration validation assets are under
`artifacts/recipes/pocket_tts_fp32`; the original assets are retained.

## Status

Pocket TTS from mlx-audio runs through CoreAI with **persistent KV caches and
streaming convolution state**. Text/voice prefill and audio decoding retain
dynamic sequence lengths; no fixed-length padding or cache recomputation is
used. The Python runner performs all neural-network inference through CoreAI.
MLX is only required for conversion and optional reference validation.

Tested on an M3 Max with macOS 27 build 26A428, CoreAI 1.0.0b2, MLX 0.32.2,
and mlx-audio 0.5.3. Specialization is GPU-preferred, not proof of exclusive
GPU placement. Nonfatal ANE compiler warnings occur with these FP32 assets.

The source is `mlx-community/pocket-tts`, revision
`caa18f93d68118bbddc60a8146d8bf5625159c6f`. Its BF16 checkpoint weights are
widened to FP32 before both conversion and MLX reference execution. This
introduces no weight quantization, but cannot recover precision absent from
the original BF16 checkpoint. The loader reads only `model.safetensors` and
loads strictly: voice-prompt safetensors must not be included as model weights.

## Reproduce

Use the b2 environment and keep generated assets in the ignored `artifacts/`
directory. The converter uses cached weights/tokenizer when available and
downloads missing files. mlx-audio and SentencePiece are optional dependencies
for this model, not new mandatory dependencies of the core converter.

```bash
.build/coreai-b2-env/bin/python scripts/convert_pocket_tts.py \
  --model mlx-community/pocket-tts \
  --revision caa18f93d68118bbddc60a8146d8bf5625159c6f \
  --output artifacts/pocket_tts_fp32

.build/coreai-b2-env/bin/python scripts/run_pocket_tts.py \
  artifacts/pocket_tts_fp32 \
  --text "Hello! This is Pocket TTS running with Core AI." \
  --output artifacts/pocket_tts_fp32/speech.wav
```

Conversion defaults to the Alba voice and one flow integration step. Use
`--voice` or `--steps` during conversion to change them. More flow steps are
unrolled into the sampler graph; one and four steps have tiny-model tests.
The full checkpoint benchmark below uses one step. `--component` can rebuild
one component group, including all six backbone layers, without reconverting
the others. Checkpoint/flow-step changes require a complete rebuild.

The output is mono PCM WAV at 24 kHz. The runner processes 1920-sample audio
chunks internally but writes the WAV after generation; it is not a live audio
player. Noise seed, temperature, EOS threshold, tail frames, and maximum frames
are runtime options. It uses precomputed voice conditioning; arbitrary voice
cloning/voice encoding is not exported. Text is passed directly to SentencePiece
as one prompt; the higher-level mlx-audio sentence splitting and automatic text
cleanup are not reproduced. Supply a punctuated utterance and an appropriate
frame budget rather than an unbounded document.

## Assets and State

The approximately 382 MiB bundle contains a manifest, tokenizer, voice/BOS
conditioning arrays, and nine assets:

| Asset | Ordinary Inputs | Mutable State | Outputs |
| --- | --- | --- | --- |
| `conditioner` | INT32 tokens `[1,T]` | None | Text embeddings `[1,T,1024]` |
| `backbone0` through `backbone5` | FP32 embeddings `[1,T,1024]`, INT32 position `[1]` | K and V, each `[1,1,16,C,64]` | Hidden `[1,T,1024]`; final layer also EOS `[1,T,1]` |
| `sampler` | Hidden `[1,1024]`, temperature-scaled noise `[1,32]` | None | Latent `[1,32]`, next embedding `[1,1,1024]` |
| `decoder` | Normalized latent `[1,F,32]`, INT32 position `[1]` | K and V, each `[2,1,8,D,64]`, plus nine convolution buffers | Audio `[1,1,1920*F]` |

`T`, `F`, `C`, and `D` are dynamic. Batch size is one. Backbone position
advances by the number of input tokens; decoder position advances by 16 per
latent frame, reflecting its temporal upsampler. The host supplies positions
as ordinary inputs, while CoreAI mutates persistent state buffers in place.
Hidden values between backbone assets remain CoreAI NDArrays without NumPy
readback. Normal generation does not copy KV buffers back to the CPU.

The decoder preserves causal convolution input histories and bias-free
transposed-convolution overlap tails. Its transformer uses the source's
250-step sliding attention mask. It is Pocket TTS's continuous-latent Mimi
variant, not the separate discrete-codebook Mimi asset.

Capacity is allocated once per utterance: voice + text + maximum audio frames
for the backbone, and 16 * maximum frames for the decoder. This bounds storage,
not query length, and calls advance through the cache. Generation never writes
beyond these allocations. Cache growth/reallocation and an indefinitely rolling
decoder buffer are not implemented. Start a new session/reset all state for a
new utterance.

### Beta Runtime Workaround

A single six-layer backbone graph produced incorrect unwritten KV-cache tails
at some capacities (including 292), and larger full-pipeline runs subsequently
diverged. The failure persisted with functional state outputs, optimization
disabled, and debug specialization. A reduced cache-update-only graph passed,
as did a full-width single-layer graph. The exact runtime/compiler root cause
is not established.

The converter therefore exports **one stateful asset per backbone layer**.
This preserves the original weights, equations, dynamic shapes, and caching,
at the cost of six dispatches instead of one. The final layer owns the output
norm/EOS head. The decoder remains one stateful asset. `backbone_forward` can
still capture the monolithic graph for diagnosis; the shipped converter does
not select that unvalidated execution path.

## Validation and Performance

Add `--validate-mlx` to replay identical component inputs through FP32 MLX.
Checks include text embeddings, hidden states/EOS, every occupied KV entry,
unchanged unwritten cache slots, flow latents/embeddings, each waveform chunk,
and all nine convolution histories. Values must be finite and satisfy both
0.001 maximum absolute and relative L2 error limits. KV tails must match zero
exactly. This isolates conversion error; it is not bitwise BF16 parity or an
independently sampled, free-running MLX trajectory comparison.

The default sentence completed at EOS frame 44 with three tail frames:
47 frames / 3.76 seconds of audio, **953 tensor comparisons passed**. That run
used `--prefill-chunk-size 256`, including the complete 125-token voice prompt
in one call. A second test with 64-token prefill chunks and
`--max-frames 150 --ignore-eos --validate-mlx` exercised **150 frames / 12 seconds**,
filled backbone capacity 292 and decoder capacity 2400, and passed **3,013
tensor comparisons**. Ignoring EOS is a state stress test, not a recommendation
for normal synthesis.

The rebuilt bundle was also checked with the default runner settings:
47 frames through EOS and **957 tensor comparisons passed**.

Maximum absolute errors across those two runs:

| Tensor | Maximum Error |
| --- | ---: |
| Text embeddings | 0 |
| Backbone hidden | 5.61e-6 |
| EOS logits | 3.75e-5 |
| Backbone K / V | 2.29e-5 / 6.05e-6 |
| Flow latent | 1.20e-6 |
| Decoder K / V | 8.59e-6 / 4.06e-6 |
| Decoder waveform | 1.74e-6 |
| Convolution histories | 5.15e-5 |

Without MLX validation, the 47-frame run took **3.11 seconds** including
prefill and first-use specialization: 0.91 seconds prefill, 2.13 seconds to
first audio, and **46.7 codec frames/second after the first frame**, about
3.7x real-time throughput. End-to-end real-time factor was 0.828. Asset opening
is outside the timed region; first-call work is included. These are one-run
measurements, not a broad hardware/latency guarantee. Validation reports are
written beside each WAV as JSON; validation timings include reference work.

The port also fixes three general capture/lowering gaps exposed by the flow
network: `Square`, selected-axis/inverted `NumberOfElements` (variance), and
MLX's inverted `Sqrt` flag (reciprocal square root). Repeated compiled SiLU
modules are temporarily expressed as `x * sigmoid(x)` during sampler capture;
the original modules are restored afterward. No installed mlx-audio source
files are modified.

Checkpoint-free tests cover partitioned and monolithic tiny backbones, reset
and poisoned unused slots, different dynamic capacities/chunk lengths,
one/four-step flow sampling, and streaming decoder KV/convolution state.

Regression suite: **277 passed, 5 existing expected failures**. `git diff --check`
passes.

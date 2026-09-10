# mlx2coreai

Experimental MLX to [CoreAI](https://developer.apple.com/documentation/coreai/) conversion.

`mlx2coreai` captures MLX graphs, lowers supported ops to CoreAI MLIR, and writes
`.aimodel` assets or coreai-models-style LLM bundles.

See the [architecture and refactoring notes](docs/refactoring.md) for the
conversion pipeline, extension points, and full-model correctness gates.
Post-refactor [Qwen and SmartTurn validation](docs/qwen_smart_turn_validation.md)
records numerical results and the SmartTurn optimizer limitation.

Mimi's offline FP32 encoder and decoder from mlx-audio also convert with dynamic
sequence lengths. See [Mimi conversion](docs/mimi_conversion.md) for validated
artifacts, reproduction commands, and the separate streaming work still needed.

## Install

```bash
pip install mlx2coreai
```

## Convert an mlx-lm Model

For autoregressive language models, use the stateful converter. It writes a
bundle containing `metadata.json`, `tokenizer/`, and a nested `.aimodel`.
Keep converted models in `artifacts/`, which is ignored by Git.

```bash
mlx2coreai convert-mlx-lm-stateful mlx-community/Qwen3-0.6B-bf16 \
  --output artifacts/qwen \
  --max-context-length 256
```

The exported model has one `main` entrypoint with `input_ids`, `position_ids`,
and mutable `keyCache` / `valueCache` state.

Qwen3.5 hybrid text-decoder export is **experimental, not runtime-validated**.
It preserves unquantized BF16 weights and adds convolution and FP32 recurrent
states. The native gated-delta op currently corrupts integrated outputs on
macOS 27 build 26A428; a dynamic decomposed alternative hits a compiler error
in the hybrid graph. See [Qwen3.5 conversion notes](docs/qwen35_conversion.md)
for artifacts, validation commands, and reduced repros. No fixed-length
execution workaround is used.

LFM2.5-2.6B export has **validated FP32 logit and state parity** after correcting
a packed KV-cache read failure in the beta runtime. It uses
the existing `mlx-lm` implementation and adds short-convolution cache support,
without gated-delta ops or fixed-length execution. Reduced-precision execution
still needs separate validation; FP16 can abort during compilation. Use the Python
runner with `--grow-context` for its three-state contract. See
[LFM2.5 conversion notes](docs/lfm25_conversion.md) for artifacts and repros.

## Benchmark Sampling

```bash
python scripts/benchmark_aimodel_sampling.py artifacts/qwen \
  --contexts 16,32,64,128,256 \
  --steps 16
```

The benchmark accepts either the bundle directory (`artifacts/qwen`) or the nested asset
path (`artifacts/qwen/qwen.aimodel`). On macOS 27, synthetic greedy benchmarks use the Swift
CoreAI runner, which supplies explicit buffers for dynamic outputs and mutable
KV-cache state. Install Xcode with a macOS 27 SDK; SDK discovery uses `SDKROOT`,
the selected `xcrun` SDK, or `/Applications/Xcode.app` / `Xcode-beta.app`.

The default sweep keeps each decode position fixed to measure steady-state
throughput. Add `--grow-context` to advance the position after every token.
Both runners default to one position ID per input token
(`--position-ids-layout query`). The exported model reads the maximum position ID for its cache offset;
there is no need to grow the position tensor during single-token decoding.
This avoids repeated input-shape preparation without fixing model or cache
dimensions. Use `--position-ids-layout context` for models that require the full
cached position range, or to reproduce the previous Swift behavior.

Use `--runtime-backend python` to run directly through the Python bindings.
With `coreai-core==1.0.0b2` and macOS 27 build 26A428, freshly converted Qwen
assets now execute with dynamic outputs and mutable state in Python without
caller-provided output buffers. Python prefers GPU specialization on the OS
runtime, matching Swift; the local runtime remains available without delegates.
Tokenizer options such as `--prompt` and `--decode` select Python automatically.

Both runners accept `--json-output results.json` for timings and sampled token
IDs, and `--logits-dir logits` for final-step last-token logits per context
(`context_<length>.f32`, raw little-endian float32). Logit files are written
outside the timed interval. JSON output no longer implicitly selects Python;
specify `--runtime-backend` when comparing runners. Synthetic benchmarks honor
`--fill-token-id` even when a bundle tokenizer is present.

See [macOS 27 benchmark notes](docs/macos27_benchmark.md) for tested package
versions, runtime limitations, and results.

## Convert a Generic MLX Function

```python
import mlx.core as mx
import numpy as np

from mlx2coreai import ConversionConfig, convert_mlx_to_coreai


def model(x, w):
    return mx.tanh(mx.matmul(x, w))


converted = convert_mlx_to_coreai(
    model,
    {
        "x": np.ones((2, 3), dtype=np.float32),
        "w": np.ones((3, 4), dtype=np.float32),
    },
    config=ConversionConfig(optimize=True),
    output_path="artifacts/model.aimodel",
)

print(converted.asset_path)
```

## Run an Asset

For repeated calls, `CoreAISession` keeps the executable and state alive. The
benchmark and validator use this same API; existing one-shot helpers remain
available. State is reset between independent sequences, not between decode
steps. Results remain CoreAI NDArrays until explicitly read back.

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai import CoreAISession

options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def decode(token_ids, position_ids):
    async with CoreAISession("artifacts/qwen", specialization_options=options) as session:
        session.reset_state(state_capacity=2048)
        outputs = await session.run_tokens(token_ids, position_ids)
        # Continue calling run_tokens in this session to preserve cache history.
        return outputs
```

When the local CoreAI runtime is available:

```python
import asyncio
import numpy as np

from mlx2coreai import run_aimodel


async def main():
    result = await run_aimodel(
        "artifacts/model.aimodel",
        {"x": np.ones((2, 3), dtype=np.float32)},
    )
    print(result.outputs)


asyncio.run(main())
```

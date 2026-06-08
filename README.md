# mlx2coreai

Experimental MLX to CoreAI conversion pipeline.

The package captures MLX execution into a small graph IR, lowers supported MLX
ops to CoreAI MLIR with `coreai.GraphOp`, and saves `.aimodel` assets.

## Usage

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
    output_path="model.aimodel",
)

print(converted.metadata)
```

The saved asset directory contains `main.mlirb`, `main.hash`, and
`metadata.json`.

## mlx-lm Models

Install `mlx-lm`, then use the helper to load a Hugging Face model with
`mlx_lm.load()`, synthesize an `input_ids` capture batch, and save a CoreAI
asset:

```bash
mlx2coreai convert-mlx-lm mlx-community/Qwen3-0.6B-Instruct-bf16 \
  --output qwen.aimodel
```

The same path is available from Python:

```python
from mlx2coreai import ConversionConfig, convert_mlx_lm

converted = convert_mlx_lm(
    "mlx-community/Qwen3-0.6B-Instruct-bf16",
    "qwen.aimodel",
    config=ConversionConfig(optimize=True),
)
```

By default the helper captures `model(input_ids)` and selects the first output
when the model returns a tuple, list, or mapping. Pass `capture_function=` for
models that need masks, cache/state arguments, or a custom output selection.

`convert-mlx-lm` emits the token axis as a ranked dynamic CoreAI dimension by
default, so both `--prompt` and `--sequence-length` are optional. When neither is
provided, the helper synthesizes a one-token capture input from the tokenizer's
fallback special token. Pass `--prompt` to use real text as the capture example,
`--sequence-length` to synthesize or truncate/pad to a specific example length,
or `--no-dynamic-sequence` / `convert_mlx_lm(..., dynamic_sequence=False)` when a
fixed-shape asset is desired. Generic conversions can opt into the same
mechanism with `ConversionConfig(dynamic_axes={"input": [axis]},
dynamic_probe_inputs={...})`.

## Coverage

The current backend covers the generic op families needed by the vendored
`mlx2coreml`-derived static model zoo:

- arithmetic, comparisons, casts, `where`, `isclose`, `allclose`, and
  finite/NaN helpers;
- reductions including `argmax`, `argmin`, `var`, `std`, and `logsumexp`;
- shape/index ops including reshape, flatten/unflatten, transpose, move/swap
  axes, squeeze/expand, slice/update, split, take, take-along-axis, concat,
  broadcast, meshgrid, diagonal/trace, triangular bands, eye, and kron;
- tensor creation helpers including zeros/ones/full, like variants, arange, and
  linspace;
- matmul/addmm, outer/inner/tensordot, softmax, layernorm, RMSNorm, RoPE, and
  scaled dot-product attention;
- `conv2d`/`conv3d`, plus a CoreAI-asset-safe 1x1 stride-1 transposed-conv
  lowering used by the reference conv block.

MLX `rmsnorm`, `rope`, and `scaled_dot_product_attention` are emitted as
private CoreAI composite declarations. Mutable buffer metadata is emitted for
state update nodes using CoreAI's `MutableBuffers.buffer_mutation` attribute.

## Validation

Run:

```bash
pytest -q
```

The suite saves CoreAI assets for hand-authored graphs, live MLX captures, and
the model-zoo graphs in `tests/model_zoo.py`.

Generate the op coverage report with:

```bash
mlx2coreai ops --validate-assets
```

This writes `docs/op_coverage.md` and `docs/op_coverage.json`.

## Runtime Execution

When the local CoreAI runtime is available, saved assets can be executed through
the thin `coreai-core` wrapper:

```python
import asyncio
import numpy as np

from mlx2coreai import run_aimodel


async def main():
    result = await run_aimodel(
        "model.aimodel",
        {"x": np.ones((2, 3), dtype=np.float32)},
    )
    print(result.outputs)


asyncio.run(main())
```

For captured conversions, `validate_converted_model(converted)` runs the saved
asset or transiently saves the `AIProgram`, then compares CoreAI outputs against
the MLX capture outputs. `run_aimodel_sync(...)` and
`validate_converted_model_sync(...)` are available for scripts that are not
already inside an event loop.

To sample a converted language model and benchmark repeated forwards at fixed
context lengths:

```bash
python scripts/benchmark_aimodel_sampling.py qwen3.aimodel \
  --contexts 16,32,64,128,256 \
  --steps 8
```

Pass `--model-id mlx-community/Qwen3-0.6B-bf16 --prompt "hello"` to seed the
benchmark with real tokenized text, and `--decode` to print sampled text. The
script keeps the CoreAI executable loaded for the whole run and reports timed
tokens/sec after per-context warmup.

## Caveats

Dynamic causal `scaled_dot_product_attention` graphs currently skip
`AIProgram.optimize()` because the beta optimizer rewrites the causal mask into
an invalid runtime reshape for dynamic sequence shapes. MLX BF16 constants are
currently widened to FP32 during capture so executable assets can pass CoreAI
verification; expect small full-model logit drift against the original BF16 MLX
forward pass.

The beta asset writer currently rejects the native `coreai.conv_transpose2d`
op. The backend lowers the common 1x1 stride-1 case to reshape/matmul/transpose
and emits a named composite fallback for other transposed-conv shapes so assets
can still be generated with an explicit marker.

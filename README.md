# mlx2coreai

Experimental MLX to [CoreAI](https://developer.apple.com/documentation/coreai/) conversion.

`mlx2coreai` captures MLX graphs, lowers supported ops to CoreAI MLIR, and writes
`.aimodel` assets or coreai-models-style LLM bundles.

## Install

```bash
pip install mlx2coreai
```

## Convert an mlx-lm Model

For autoregressive language models, use the stateful converter. It writes a
bundle containing `metadata.json`, `tokenizer/`, and a nested `.aimodel`.

```bash
mlx2coreai convert-mlx-lm-stateful mlx-community/Qwen3-0.6B-bf16 \
  --output qwen \
  --max-context-length 256
```

The exported model has one `main` entrypoint with `input_ids`, `position_ids`,
and mutable `keyCache` / `valueCache` state.

## Benchmark Sampling

```bash
python scripts/benchmark_aimodel_sampling.py qwen \
  --contexts 16,32,64,128,256 \
  --steps 16 \
  --decode
```

The benchmark accepts either the bundle directory (`qwen`) or the nested asset
path (`qwen/qwen.aimodel`). It uses the embedded tokenizer when present.

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
    output_path="model.aimodel",
)

print(converted.asset_path)
```

## Run an Asset

When the local CoreAI runtime is available:

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

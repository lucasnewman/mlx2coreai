# Generic Conversion and Execution

For a supported model, prefer its [recipe](../recipes/README.md), which handles
model-specific inputs and state. Use these APIs for an arbitrary MLX function
or direct asset access. See [installation](../README.md#install) for prerequisites.

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

Use `run_aimodel` with an explicit `.aimodel` path for one-shot execution.
For repeated calls, `CoreAISession` keeps the executable and state alive;
call `session.run(inputs)` with named input arrays. Results stay as CoreAI
NDArrays until read back with `.numpy()`.

For model recipes, use `Bundle.open(...)` and the
[recipe session API](recipe_api.md), which manages component inputs and state.

When the local CoreAI runtime is available:

```python
import asyncio
import numpy as np

from mlx2coreai import run_aimodel


async def main():
    result = await run_aimodel(
        "artifacts/model.aimodel",
        {"x": np.ones((2, 3), dtype=np.float32),
         "w": np.ones((3, 4), dtype=np.float32)},
    )
    print(result.outputs)


asyncio.run(main())
```

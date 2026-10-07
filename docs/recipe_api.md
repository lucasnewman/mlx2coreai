# Recipe API

All [model recipes](../recipes/README.md) expose `build(...)`, `Request`, and
`run(session, request)`. Use the model's README for its setup, build options,
and compatibility warnings. This guide covers the shared bundle/session API
and how to author your own recipe.

## Build and Load

```python
from mlx2coreai.recipe import Bundle, export
from recipes import qwen3

bundle = export(qwen3.build(), "artifacts/recipes/qwen3_fp32")

# Load an existing bundle without loading the source checkpoint.
bundle = Bundle.open("artifacts/recipes/qwen3_fp32")
```

Keep generated bundles in the git-ignored `artifacts/` directory. Bundle
metadata records the source and conversion options. Runtime options, such as
text, sampling settings, and generation budget, belong to a request.

## Manage a Session

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from recipes import qwen3

options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def generate(bundle):
    async with bundle.session(specialization_options=options, storage_kind="metal") as session:
        request = qwen3.Request(prompt="Hello!", chat=True, max_new_tokens=32)
        async for token_id in qwen3.run(session, request):
            print(token_id)
```

Use the session as an async context manager so resources are released after
completion, cancellation, or exceptions. Stateful recipes reset buffers at
the start of each request. Requests in one session must be serial; independent
concurrent requests need independent sessions.

Mimi returns a NumPy array per request, SmartTurn returns a dictionary of NumPy
logits, probabilities, and predictions, language-model recipes yield token IDs,
Pocket TTS yields NumPy audio chunks, and Parakeet Redux returns a dictionary
with text, token timestamps, and a completion flag. Language models, Parakeet
Redux, and Pocket TTS accept an optional `report={}` argument to collect
generation statistics when the
request completes.

Inspect `bundle.metadata.get("experimental")` before opening a session. Only
opt into experimental execution for diagnostics; opening the executable itself
can fail on an incompatible runtime.

## Create a Recipe

Use the public `Build`, `Component`, and `ConversionConfig` APIs to wrap a model:

```python
import mlx.core as mx
import numpy as np
from mlx2coreai import ConversionConfig
from mlx2coreai.recipe import Build, Component, export

def build():
    return Build("tanh", {"main": Component(
        forward=lambda x: mx.tanh(x),
        inputs={"x": np.zeros((1, 8), np.float32)},
        outputs=("result",),
        config=ConversionConfig(),
    )})

async def run(session, value):
    result = await session.run("main", {"x": value}, readback=True)
    return result["result"]

bundle = export(build(), "artifacts/recipes/tanh")
```

Declare one unique output name per public output. Use `ConversionConfig` for
dynamic axes and probe inputs; mutable state is declared with `CaptureSignature`
and `StateBinding`. Keep optimization enabled for mutable-state components.

`Build.resources` maps relative bundle paths to file paths or bytes, for example
a tokenizer. Resources must not collide with the manifest or generated assets.

`export(plan, path, only=["component"])` rebuilds selected components only when
the recipe metadata and resources are unchanged. Otherwise perform a full
rebuild. Use a new output directory to preserve a previous bundle.

For direct component access, `session.run(name, inputs)` returns named CoreAI
NDArrays. Set `readback=True` for NumPy copies. `reset_state({name: capacity})`
allocates fresh state buffers, and `snapshot_state(name)` copies state for
diagnostics. Normal generation should use the recipe's `run` function so
positions, capacities, and model-specific request handling are applied.

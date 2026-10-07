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
manifests record the source and conversion options; runtime configs contain only
execution parameters. Runtime options, such as
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

`ConversionConfig.graph_transform` optionally rewrites the captured graph
after dynamic shape probing and signature binding, before normalization and
type inference. Recipes can use it to preserve compressed checkpoint weights
as packed constants with explicit decompression operations. Preserve the
public input/output contract and verify numerical parity with the source.

## Weight Quantization

Existing recipes can opt into the shared FP32 linear-weight baseline at export:

```python
from mlx2coreai import WeightQuantization
from mlx2coreai.recipe import export
from recipes import qwen3

report = {}
bundle = export(qwen3.build(), "artifacts/recipes/qwen3_uint4",
                quantization=WeightQuantization(bits=4, group_size=128),
                quantization_report=report)
print(report)  # Per-component coverage, skipped weights, sizes, and weight errors.
```

Use `bits=4` or `bits=8`. This compresses constant rank-two RHS matrices of
`matmul`/`addmm` projections, tracing through rank-two transposes and batch broadcasts and grouping
along the input/contraction dimension independently for each output channel.
It uses affine min/max groups with round-to-nearest-even codes. Bias vectors,
normalization weights, standalone embedding lookups, runtime inputs, and
activations are unaffected. Shared constants remain shared: quantizing a tied
projection weight also affects its other consumers, including embedding lookups.

Weights whose contraction dimension does not divide into complete groups, and
weights outside FP32, are skipped and recorded in the report. Smaller positive
group sizes are allowed. `exclude=("pattern", ...)` matches captured constant
names or source labels; these are graph identifiers, not model module paths.
Dynamic shapes and mutable-state signatures keep their existing contracts.
The export option overrides component policies without mutating the build plan.
Alternatively, set `ConversionConfig.weight_quantization` on individual components.

Generic conversion callers can use the same configuration with
`convert_mlx_to_coreai`; coverage is available in `prepared.quantization_report`
and `converted.metadata["quantization"]`. Graph transforms run before this pass,
so already preserved packed weights are not requantized. Evaluate lazily
initialized MLX parameters before capture so weights appear as constants.

For an already quantized affine MLX checkpoint, use the shared preservation helper:

```python
import mlx.core as mx
from mlx2coreai import ConversionConfig, prepare_quantized_linears

model.set_dtype(mx.float32)
packed = prepare_quantized_linears(model)
config = ConversionConfig(graph_transform=packed)
```

This mutates `nn.QuantizedLinear` modules into dense Linears for capture and
restores their original 2/4/8-bit codes, scales, and biases in the graph. It
supports affine checkpoints with compatible group shapes. Uncaptured modules
are allowed for component/subgraph exports; use `require_all=True` to enforce
complete capture. MLX/CoreAI fused-versus-separate FP32 arithmetic can differ
slightly even when packed parameters are identical. Other quantized module
types and non-affine formats are outside this baseline.

The runtime API remains the same. Computation remains FP32; this primarily
reduces stored weights. Weight-error diagnostics do not establish model quality
or guarantee memory/latency gains. Evaluate model outputs against an unquantized
export. Reports are returned to the caller and are not added to runtime configs.

## Packaging

`Build.resources` maps relative bundle paths to file paths or bytes, for example
a tokenizer. Resources must not collide with the config, manifest, or generated assets.

Set `Build.runtime_metadata` to the parameters needed by the runtime to export
`config.json` instead of a build manifest. Source paths and conversion
diagnostics can stay in `Build.metadata`; they are not written to the config.
`Bundle.open` reads both formats and exposes the runtime parameters through
`bundle.metadata`.

Set `Build.asset_name="model.aimodel"` to package multiple components as named
functions in one asset. Give each component a unique
`ConversionConfig.entrypoint_name`. Components must share their CoreAI lowering
policy, capture their constants, and pass state explicitly as ordinary inputs
and outputs. Combined packages require a full rebuild. At runtime, components
in the same package share one executable with separate function sessions.
Parakeet Redux uses both options for its `encode` and `decode` functions.

`export(plan, path, only=["component"])` rebuilds selected components only when
the recipe metadata and resources are unchanged. Otherwise perform a full
rebuild. Use a new output directory to preserve a previous bundle.

For direct component access, `session.run(name, inputs)` returns named CoreAI
NDArrays. Set `readback=True` for NumPy copies. `reset_state({name: capacity})`
allocates fresh state buffers, and `snapshot_state(name)` copies state for
diagnostics. Normal generation should use the recipe's `run` function so
positions, capacities, and model-specific request handling are applied.

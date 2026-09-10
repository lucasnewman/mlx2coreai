# Model Recipes

Recipes own source-model loading, capture adaptations, input examples,
partitioning, and execution policy. The core owns conversion and runtime
mechanics. Adding a model should normally add a recipe, not a model-name branch
in `mlx2coreai`.

The examples exercise different contracts:

- **Mimi:** two stateless offline components, encode and decode; each request
  returns one NumPy array.
- **Pocket TTS:** nine components, mutable KV/convolution buffers, prefill and
  autoregressive sampling; a request yields NumPy audio chunks.
- **Qwen3, Qwen3.5, and LFM2/2.5:** a single dynamic stateful decoder, shared
  token generation, and family-local KV/convolution/recurrent layouts. A request
  yields integer token IDs. See [language-model recipes](../docs/lm_recipes.md)
  for commands and precision/runtime limitations.

The existing scripts remain compatibility commands. Recipe bundles now use a
shared manifest format; older Pocket bundles need reconversion. Original
validation assets are retained, and the recipe migration was checked in
`artifacts/recipes/{mimi,pocket_tts}_fp32`.

## Using a Recipe

Conversion is separate from execution. Build-time options (source revision,
precision, flow steps, voice resources) are recorded in the bundle. Runtime
options (text, sampling seed, frame budget) belong to a request.

```python
from mlx2coreai.recipe import Bundle, export
from recipes import mimi, pocket_tts

codec = export(mimi.build("/path/to/mimi.safetensors"), "artifacts/my_mimi")
tts = export(pocket_tts.build(steps=1, voice="alba"), "artifacts/my_pocket")

# Later, without loading either source model:
tts = Bundle.open("artifacts/my_pocket")
```

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions

options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def roundtrip(codec, audio):
    async with codec.session(specialization_options=options) as session:
        codes = await mimi.run(session, mimi.Request("encode", audio))
        return await mimi.run(session, mimi.Request("decode", codes))

async def speak(tts, consume):
    async with tts.session(specialization_options=options, storage_kind="metal") as session:
        async for chunk in pocket_tts.run(session, pocket_tts.Request(text="Hello world.")):
            consume(chunk)
```

The caller owns the session context. Exiting it releases all component sessions,
including after exceptions or interrupted streaming. Requests within one session
are serial, not concurrent. Pocket resets every mutable buffer and its host
positions at the beginning of each request; independent concurrent requests
need independent sessions. The optional `report={}` argument collects Pocket's
generation statistics on successful exhaustion. It excludes caller time between
yields, and is not a completion report when streaming is cancelled.

Normal runtime imports neither mlx-audio nor source-model weights. Conversion
and optional parity validation require the model's dependencies. The `recipes`
package ships alongside the core, but its root never eagerly imports recipes.
Bundle metadata is data: `Bundle.open` does not import or execute the recipe
named in a manifest.

Language-model generation also avoids importing MLX-LM. Tokenizers are packaged
as bundle resources and loaded locally through Transformers. The original
`convert-mlx-lm-stateful` command and public Python API retain their old bundle
format; their implementation now lives in `recipes/_mlx_lm`, with a compatibility
shim in the core. Old benchmark scripts still use those legacy bundles.

## Authoring Surface

Two ordinary functions are the convention, not an inheritance hierarchy:

```python
def build(source, **options) -> Build:
    # Load source, adapt forward calls, construct examples and resources.
    ...

async def run(session, request):
    # Preprocess, reset state, call components, return or yield results.
    ...
```

`Build` contains a recipe identifier, a name-to-`Component` mapping, optional
resources, and JSON-serializable metadata. Resources are relative bundle paths
mapped to source `Path`s or raw `bytes`; learned parameters remain captured
constants inside the converted assets.

`Component` has only four fields:

```python
Component(
    forward=adapted_forward,
    inputs=example(sequence=3, capacity=32),
    outputs=("hidden", "eos"),
    config=ConversionConfig(...),
)
```

Use the existing `ConversionConfig` for explicit dynamic axes, probe inputs,
optimization, and `CaptureSignature` state bindings. No second tensor schema
or automatic shape guessing is introduced. The exporter maps the declared
public output names to the compiler's output names; state outputs are excluded
using their bindings, not an assumption that state is always last. Functional
state outputs become mutable CoreAI state through the existing conversion path.

Examples are real arrays. Small recipe-local functions can construct matching
capture/probe inputs, including zero state with different capacities. Those
functions should not hardcode trace lengths into the adapted forward pass.
Adapters still need independent native-model parity tests.

Mutable-state components currently require optimization: the beta optimizer
also promotes annotated buffers to runtime state. Export rejects disabled or
skipped optimization for these components rather than publishing an invalid
state contract. This does not affect stateless recipes such as Mimi.

`export(build, path, only=[...])` supports partial component builds when the
recipe metadata/resource names are unchanged. Changes to the source/precision/
partitioning contract require a full rebuild. Selected conversions are staged
before publication; conversion failure leaves an existing bundle untouched.
A filesystem failure during publication invalidates its manifest instead of
advertising a mixed bundle. Resources are republished together; recipes must
record changes in resource identity/configuration in their metadata. Use a new
output directory when preserving previous assets matters. `save_graphs=True`
also writes diagnostic IR files.

## Runtime Boundary

`Bundle.session()` wraps existing `CoreAISession`s, not a new inference engine:

- `reset_state({component: capacity, ...})` allocates fresh buffers for every
  stateful component in the session. Fixed-size state ignores capacity axes but
  still requires a positive allocation value.
- `await run(component, inputs)` returns named CoreAI NDArrays, allowing
  intermediate values to remain on device between components.
- `readback=True` explicitly returns copied NumPy outputs instead.
- `snapshot_state(component)` explicitly copies that component's state for
  diagnostics. There is no routine KV readback during normal generation.

Position advancement, cache capacity calculation, attention-window semantics,
EOS policy, noise generation, and output formatting are recipe responsibilities.
The runtime does not infer that all states advance once per token. Pocket's
decoder advances 16 positions per latent frame, for example.

## Workarounds and Tests

Keep source-specific adaptations in `adapter.py`; choose partitions and
conversion settings in `build.py`. Record the chosen workarounds in metadata
and explain/test them beside the recipe. There is no automatic runtime-version
matrix or plugin registry. Generic operator correctness fixes belong in the
core, not duplicated across recipes.

Validation stays outside production execution. A session may receive an
optional synchronous `observer(session, component, inputs, outputs)` called
after each inference, with raw NDArrays. Pocket's `validation.Reference`
uses this to replay native MLX and check all caches/history without putting
MLX branches in the generation loop. Observers can raise on mismatches;
context cleanup still runs. No observer runs by default.

The migration retains the original adapter tests and adds tests for named
outputs, non-trailing state bindings, partial builds, failed builds, session
cleanup, deferred optional imports, dynamic Mimi calls, and repeated/interrupted
Pocket requests. Checkpoint validation uses the existing script commands with
`--output artifacts/recipes/...`; see the detailed [Mimi](../docs/mimi_conversion.md)
and [Pocket TTS](../docs/pocket_tts_conversion.md) notes.

## Layout

```text
mlx2coreai/recipe.py        Component / Build / export / Bundle / session
mlx2coreai/kv_cache.py      Shared functional KV capture helper
recipes/
  mimi/
    build.py               Source loading and component contracts
    adapter.py             Offline cache/padding adaptations
    runtime.py             Request and run
    validation.py          Native parity checks and conversion command
  pocket_tts/
    build.py               Source loading, fixtures, resources, partitioning
    adapter.py             Stateful backbone and streaming decoder adapters
    runtime.py             Request and streaming run
    validation.py          Independent native reference observer
    cli.py                 Existing command-line options and WAV/report output
  qwen3/                   Full-attention layout and FP32 default
  qwen35/                  Hybrid layout and experimental gated-delta policy
  lfm2/                    Short-convolution layout and precision guard
  _mlx_lm/
    build.py               Shared decoder component and tokenizer packaging
    stateful.py            Shared capture adapters and legacy conversion API
    policy.py              Legacy automatic adapter selection only
    runtime.py             Request, prefill, sampling, and per-request reset
    validation.py          Native logit and all-state observer
    cli.py                 Shared convert/run command implementation
```

Not every recipe needs this many files. A small recipe can keep `build` and
`run` in one module; split files only to make the model-specific code readable.

## Migration Verification

- Suite after the language-model migration: **290 passed, 5 existing expected
  failures**; `git diff --check` passes.
- Pocket: the sentence finishes at the same EOS step, with 957 comparisons;
  150-frame stress execution passes 3,013 comparisons including all state,
  with maximum waveform error 1.736e-6 against FP32 MLX.
- Mimi: dynamic 2/3/1/5/13-frame encode/decode cases retain exact encoder codes
  and maximum decoder waveform error 7.377e-6. The speech fixture is also
  exercised through the retained `--audio` command.
- Language-model recipes add native logit/all-state checks, repeated and
  interrupted requests with different capacities, tokenization, sampling,
  and pre-specialization experimental guards. See
  [language-model verification](../docs/lm_recipes.md#checkpoint-verification)
  for full-checkpoint results.

The recipe execution benchmark generated 3.76 seconds of audio in 3.44-3.48
seconds including first-call work, with roughly 35 codec frames/second after
the first frame. This is a fresh measurement, not a claim of reproducing the
earlier machine-state-dependent throughput exactly. Reports and converted
assets are in `artifacts/recipes/`; original bundles are not overwritten.

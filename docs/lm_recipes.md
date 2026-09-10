# Language-Model Recipes

`recipes.qwen3`, `recipes.qwen35`, and `recipes.lfm2` use the same `build` /
`Request` / `run` convention as Mimi and Pocket TTS. Each recipe owns its family
adaptations and precision restrictions. Shared MLX-LM scaffolding lives in
`recipes/_mlx_lm`, not in the conversion core.

## Build and Run

```bash
python -m recipes.qwen3 convert --output artifacts/recipes/qwen3_fp32
python -m recipes.lfm2 convert --output artifacts/recipes/lfm25_fp32
python -m recipes.qwen35 convert --output artifacts/recipes/qwen35_bf16

python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prompt "What is the capital of France? Answer in one short sentence." \
  --max-new-tokens 32
python -m recipes.lfm2 run artifacts/recipes/lfm25_fp32 --chat \
  --prompt "What is the capital of France? Answer in one short sentence." \
  --max-new-tokens 32
```

Default sources are `mlx-community/Qwen3-0.6B-bf16`, `Qwen/Qwen3.5-0.8B`, and
`LiquidAI/LFM2.5-2.6B-MLX-bf16`. The tested Qwen3 checkpoint is **0.6B**, not
0.8B. Supply an optional positional source path or Hub ID to `convert`, and
`--revision` to pin a Hub revision. No new quantization is applied.

Qwen3 and LFM default to FP32; Qwen3.5 defaults to source precision (`auto`,
BF16 for this checkpoint), retaining FP32 recurrent state. Override with
`--compute-precision` and `--cache-dtype` only when intentionally testing other
precision policies. Metadata records these choices and the known workarounds.

All three export one `main.aimodel`, `tokenizer/`, and the generic recipe
`manifest.json`. This is different from the old `metadata.json` LLM bundle:
the original converter and benchmark commands remain unchanged, but do not
consume recipe manifests. Use the recipe's `run` command for new bundles.

```python
from coreai.runtime import ComputeUnitKind, SpecializationOptions
from mlx2coreai.recipe import Bundle, export
from recipes import qwen3

bundle = export(qwen3.build(), "artifacts/recipes/my_qwen")
# In a later process, without the source model:
bundle = Bundle.open("artifacts/recipes/my_qwen")
options = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu())

async def generate():
    async with bundle.session(specialization_options=options, storage_kind="metal") as session:
        request = qwen3.Request(prompt="Hello!", chat=True, max_new_tokens=32)
        async for token_id in qwen3.run(session, request):
            print(token_id)
```

Use `token_ids=(...)` instead of `prompt` to bypass tokenization. Runtime loads
only the packaged tokenizer, never the checkpoint or MLX-LM. `run` yields token
IDs; the CLI decodes them and emits a JSON report. `temperature=0` is greedy;
positive temperature supports seeded sampling with optional `top_k`.

## State and Execution

Query length and KV capacity remain dynamic. `--max-context-length` at conversion
sets the capture example, not a fixed execution length. At runtime, capacity
defaults to prompt length plus generation budget; `--state-capacity` selects
a larger allocation. Too-small allocations are rejected before inference.

Prefill uses variable-length chunks (`--prefill-chunk-size`, or initial explicit
sizes with `--prefill-chunks 3,5`). Decode advances position on every call.
Each request resets all buffers and host positions, including after an
interrupted previous request. Requests within a session must be serial.

- Qwen3 has `keyCache` and `valueCache`.
- LFM adds `convState`, whose history length is architectural, not query length.
- Qwen3.5 adds both convolution history and FP32 `recurrentState`.

The shared cache adapter retains the per-layer gather/select workaround and
final packing. Model-local adapters select which layers consume each state.
LFM additionally materializes each convolution-history update with a gather
before packing as an experiment targeting possible slice-view / fixed-reshape
aliasing. Repeat validation still fails; this is not a confirmed fix.
The adapter lives in `recipes/lfm2/adapter.py`;
the legacy converter selects the same adapter through recipe-local policy.
Normal generation does not read KV/conv/recurrent buffers back to the CPU.

## Validation and Limits

```bash
python -m recipes.qwen3 run artifacts/recipes/qwen3_fp32 --chat \
  --prefill-chunks 3,5 --max-new-tokens 33 --state-capacity 2048 \
  --ignore-eos --validate-mlx \
  --json-output artifacts/recipes/qwen3_fp32/validation.json
```

Use the same arguments with `recipes.lfm2` and its bundle. The optional native
observer checks logits and every occupied state against an independent MLX
forward pass after each call, and checks that unused KV tails remain zero.
Validation fails if either the maximum absolute error or relative L2 error
exceeds its tolerance (both default to 0.01). It also records greedy-token
agreement. `--source` can override a moved validation checkpoint.

Validation timings include reference execution and state readbacks; they are
not inference benchmarks. Omit `--validate-mlx` for normal generation.

Do not disable authoring optimization on mutable-state components in this SDK:
that pass also promotes buffer arguments to runtime state. Recipe export rejects
such configurations instead of publishing a bundle with missing state.

Qwen3.5 is still **experimental**, regardless of precision. Its known native
gated-delta corruption and decomposed-loop compiler failure are not solved by
moving to recipes. LFM reduced precision and mixed compute/cache precision
also require explicit `--allow-experimental` for diagnostic execution. The CLI
checks this before loading an executable, because beta failures can abort
during specialization. Library callers must check experimental metadata before
opening a session; `run` additionally guards before inference.

Qwen3 BF16 has measured logit drift and is not a close-parity baseline, though
it can execute. See the original [Qwen validation](qwen_smart_turn_validation.md),
[Qwen3.5](qwen35_conversion.md), and [LFM2.5](lfm25_conversion.md) notes for the
runtime investigations. The migration retains their existing expected failures
rather than masking them with fixed-length graphs or MLX fallbacks.

## Checkpoint Verification

Re-exported from cached unquantized checkpoints on macOS 27 build 26A428 with
CoreAI 1.0.0b2, MLX 0.32.2, and MLX-LM 0.31.3. Original assets were retained;
new bundles and reports are under `artifacts/recipes/`.

Qwen3-0.6B FP32 passes 35 calls (3/5/remainder prefill, then 32 advancing decode
steps) at capacity 2048, beyond the capture capacity of 256. All 35 greedy-token
comparisons match MLX. Maximum absolute errors are 1.241e-4 for logits,
8.240e-4 for keys, and 7.877e-4 for values; unused KV tails remain zero.

LFM2.5-2.6B FP32 with recipe-local gather materialization passed one 35-call
comparison at capacity 2048 with matching greedy tokens, but the process later
reported a disk-space error. A repeat failed at position 8 with maximum absolute
logit error 5.082. Full-model recipe parity is therefore unresolved; the saved
successful validation report must not be treated as reproducible verification.
The initial slice-view packing export failed during decode, and switching buffer
backing did not fix it. The old validated asset passed the same request and
observer once. Graph comparison identified a packing difference worth testing,
but has not established a root cause.

The full Qwen3.5-0.8B BF16 checkpoint exports and verifies as an optimized
2,916-node component with two dynamic KV states, BF16 convolution state, and
FP32 recurrent state. The CLI correctly refuses execution without opt-in.
This is an export/contract check, not a new full-model numerical parity claim.

The full suite passes **290 tests, with five existing expected failures**.

# LFM2 MoE Conversion

**Status: exports and executes, but full-model decoding is not validated.**
MoE bundles are marked experimental and require explicit opt-in to run. The
full checkpoint has cache corruption and subsequent Metal command-buffer errors;
do not treat it as a working deployment or a meaningful performance benchmark.

`recipes.lfm2` also accepts the `lfm2_moe` implementation in MLX-LM. The initial
checkpoint is [LiquidAI/LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B),
revision `c1c44ff9fc00db3ebf4516970563f5f383d23670`. No quantization is applied.
MLX-VLM is not needed: MLX-LM 0.31.3 contains the native implementation.

The checkpoint has 24 layers, 32 experts with top-4 routing, two initial dense
feed-forward layers, six attention layers, and 18 short-convolution layers.
The four source weight shards total 16,679,862,528 bytes (8,339,929,856 parameters).
FP32 conversion is the correctness baseline; it is not a memory-efficient
deployment configuration on a 64 GiB machine.

## Commands

```bash
python -m recipes.lfm2 convert LiquidAI/LFM2-8B-A1B \
  --revision c1c44ff9fc00db3ebf4516970563f5f383d23670 \
  --compute-precision fp32 --max-context-length 256 \
  --output artifacts/recipes/lfm2_8b_a1b_fp32

python -m recipes.lfm2 run artifacts/recipes/lfm2_8b_a1b_fp32 \
  --allow-experimental \
  --chat --prompt "What is the capital of France? Answer in one short sentence." \
  --max-new-tokens 32 --state-capacity 256
```

A local checkpoint directory can replace the Hub ID. Our downloaded source is
`artifacts/sources/LFM2-8B-A1B`. The state contract remains the LFM recipe's
dynamic query length, dynamic KV capacity, and fixed architectural convolution
history. It does not use fixed-length execution or fall back to MLX at runtime.

For the actual 64 GiB-machine export, MLX's allocator cache was disabled before
loading, and a memory guard stopped the process if RSS exceeded 48 GiB or
available system memory dropped below 3 GiB. The guard was not triggered. The
equivalent conversion entry point without the external monitoring is:

```python
import mlx.core as mx
from mlx2coreai.recipe import export
from recipes import lfm2

mx.set_cache_limit(0)
plan = lfm2.build("artifacts/sources/LFM2-8B-A1B", compute_precision="fp32")
mx.clear_cache()
export(plan, "artifacts/recipes/lfm2_8b_a1b_fp32")
```

## Adaptation

The recipe replaces only the switch-MLP execution wrapper, retaining native
router computation, selected-expert probabilities, expert weights, and SiLU
gating. MLX's token-count-dependent expert sorting is a locality optimization;
the export consistently uses unsorted GatherMM dispatch. The down projection
explicitly flattens token/expert batches before generating batch indices, so
the derived extent is visible to dynamic shape capture.

Two generic issues surfaced during integration:

- GatherMM batch flattening must retain statically known matrix dimensions.
  Otherwise a subsequent squeeze fails CoreAI verification for dynamic batches.
- A varying Arange bound that probing cannot resolve must fail conversion,
  rather than silently retaining its trace value. The original down-projection
  range stopped at 16 * 4 entries and produced incorrect results beyond 16
  tokens. The explicit flatten resolves it without guessing arithmetic.

MLX-LM's `cast_predicate` consumes parameter paths, not dtypes. Precision casting
now preserves that contract, including the MoE router's excluded expert bias
and non-floating parameters.

## Validation

`tests/test_lfm2_moe.py` independently compares native and adapted MLX expert
dispatch and 24-layer recurrence, checks a converted dynamic MoE block, and compares a three-layer
stateful CoreAI model against native MLX. Inputs cross the native sorting
threshold and include lengths 1, 3, 16, and 32. Stateful tests reset between
capacities 64 and 80, compare logits and all occupied KV/convolution state,
and require untouched KV tails to remain zero.

Full-checkpoint validation must not hold FP32 MLX and CoreAI copies concurrently
on this machine. Record native logits and cache snapshots first, then replay
the same inputs in a separate CoreAI process. Greedy-token agreement is checked
as well as numerical errors; successful text generation alone is insufficient.

The full FP32 checkpoint exports as one optimized component with 2,604 nodes
and three mutable states. The asset occupies approximately 32 GiB. On this
64 GiB machine the SDK's `save_asset` reads the whole serialized bytecode back
into RAM to hash it, adding substantial memory pressure at the end of export.

Regression suite: **581 passed, five expected failures (four existing and one
new deep-MoE reproducer)** on macOS 27
build 26A428, CoreAI 1.0.0b2, MLX 0.32.2, and MLX-LM 0.31.3.

## Full-Checkpoint Findings

The independent native FP32 reference completed two requests and 66 forward
calls. The diagnostic generator intentionally continues beyond EOS to exercise
32 generation steps per request. Normal generation should still stop at EOS.

The first three-token CoreAI prefill matched native logits within `1.7643e-5`,
occupied keys within `3.34e-6`, values within `7.46e-8`, and convolution state
within `3.92e-5`. However, attention layer 10's unused value-cache tail contained
nonzero values up to `92.2871`. Later calls reported
`MTL4CommandQueueErrorDomain error 1`, returned incorrect outputs, and took tens
of seconds. That continuation overlapped a small GPU test, so its Metal errors
are not an isolated performance result. The failing replay was stopped rather
than benchmarked.

A fresh, isolated normal CLI run (no concurrent GPU tests) independently
reproduced repeated `MTL4CommandQueueErrorDomain error 1` failures with
`--prefill-chunks 3,5 --state-capacity 512 --max-new-tokens 8`. It was stopped
without producing a successful generation report. Its log is
`artifacts/lfm2_moe_cli.log`. Reliable decoding performance remains unmeasured.

A 24-layer random-weight reproducer also fails decoding, without needing the
8B checkpoint. It is retained as a strict expected-failure test. Its adapted
MLX forward matches native MLX, so the discrepancy arises after conversion;
we have not established whether lowering or the beta backend is responsible.
The first observed deep-model divergence involves the second attention layer.
Keeping capacity unchanged, materializing KV reads, separating every layer's
state buffers, removing mutable-state promotion, and making fresh contiguous
state copies did not resolve it. These unsuccessful changes are not part of
the production recipe. No fixed-length or quantized workaround was substituted.

To expose the small failure rather than counting it as expected:

```bash
python -m pytest tests/test_lfm2_moe.py --runxfail \
  -k 'dynamic_stateful and 24' -xq
```

Local diagnostic evidence is under `artifacts/lfm2_moe_validation/`, including
the native reference, initial CoreAI outputs, and the partial `coreai.json`
comparison. Logs are `artifacts/lfm2_moe_{convert,reference,verify,verify_diagnostic}.log`.
One-off reproducer scripts are `artifacts/lfm2_moe_validate.py`,
`artifacts/lfm2_split_state.py`, and `artifacts/lfm2_functional_state.py`.

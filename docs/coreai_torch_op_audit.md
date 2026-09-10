# coreai-torch Operator Audit

Inspected 2026-09-10 against upstream commit
`799d990deb85af350aa6a4aa23c85f211f2207ad` (checked with `git ls-remote`, then
fetched). Its operator files are identical to reference checkout `a89a50c`.
Local probes used the installed MLX 0.32.2 / CoreAI 1.0.0b2 environment.
The initial audit below is retained as a baseline. Follow-up implementation
and numerical verification are summarized here; neither registry counts nor
small-graph tests certify full-model parity.

## Follow-Up Implementation

The ordinary operator gaps from the initial audit now have lowerings and small
runtime comparisons in `tests/test_extended_ops.py`:

- Along-axis and multi-axis/window gather; scatter updates and reductions,
  including duplicate and negative indices.
- Boolean logic, rounding/sign, additional transcendental functions, and
  quadrant/signed-zero/infinity-aware `atan2`.
- Inclusive/exclusive, forward/reverse sum/product/min/max scans.
- Sort/argsort and partition/argpartition. Partition currently uses a full sort:
  it satisfies the partition contract, but does not preserve MLX's unspecified
  partition order or its performance characteristics.
- Static strided views, overlapping 1D/2D/3D pooling, reflection/edge padding,
  fractional nearest and linear resizing. Correct fractional arange dtypes and
  negative-stride slice extents were required in addition to new registrations.
- Named `gather_mm` composites, including multi-axis and dynamic batch shapes.
- Complex64 capture, constants, casts, real/imaginary components, conjugation,
  storage views, absolute value, arithmetic, and explicit `complex` / `polar`
  construction with broadcasting. This is not blanket coverage
  of every complex-valued operator, FFT, or complex state buffer.
- A scoped capture compatibility patch for `mx.bitwise_invert` and `~array`,
  restoring MLX functions even on capture failure. Pre-bound aliases to MLX's
  original non-serializable function are not intercepted.

`tests/test_data_dependent_ops.py` exercises explicit IR for nonzero, masked
scatter, and truncation. Nonzero is tested with varying result cardinality,
including zero results. Installed MLX has no public nonzero capture API; these
tests do not imply a new MLX frontend API.

`tests/test_control_flow.py` exercises explicit `Graph` subgraphs through `cond`
and `while_loop`, including dynamic branch shapes, multiple results, invariant
loop inputs, zero/multiple iterations, nested conditionals, and composites in
branches. Branch result types must match, and loops preserve carried tensor
types. Subgraphs bind inputs positionally and have independent tensor-name
scopes. This does not capture ordinary Python control flow from MLX.

`tests/test_dynamic_pooling.py` validates dynamic 1D/2D/3D max and average pooling
using MLX references, including padded, non-overlapping, asymmetric, and unit
windows. A scoped capture patch expresses MLX sliding windows with relative
slices and stacks, rather than fitting shape/stride formulas from two probes.
Explicit `adaptive_avg_pool` IR also supports 1-3 channels-last spatial dimensions
with runtime-computed boundaries and fixed output sizes (`None` preserves an
axis). Its small-graph tests use independent NumPy references.

Backend and remaining feature boundaries:

- GatherMM passes on the default backend; CPU-only execution reports an
  inference failure in this SDK. Structured control flow passes CPU-only tests;
  default-backend conditional probes hung or crashed. Do not infer portable
  backend support from asset verification alone.
- Dynamic batch and spatial pooling are covered. Arbitrary unresolved dynamic
  `as_strided` expressions still fail clearly instead of freezing capture
  geometry; they are not needed for the upstream pooling capability.
- Explicit compression IR now covers affine quantize/dequantize, blockwise
  expansion, LUT expansion, and sparse bitmasks, with byte and packed integer
  storage. See [compression contracts and tests](compression.md). These are not
  aliases for MLX's packed quantization primitives; automatic MLX quantized-layer
  capture and float4/float8 storage remain separate features.

The regenerated asset-coverage report has 163 lowering keys / 206 source aliases
across 33 fixture graphs. All **49 original audit probes now capture and verify**
(up from 20). These probe results are in
`.build/audit_upstream_ops_followup.jsonl`; the original results are unchanged.
Asset verification and numerical runtime tests remain separate checks.
The final follow-up suite passed with **574 passed / 4 existing expected
failures** on MLX 0.32.2 / CoreAI 1.0.0b2. `git diff --check` also passed.
Preserving rank-zero constants also makes the tiny native
gated-delta layer parity test pass: repeated runs passed, and restoring the old
rank-widening behavior reproduced its mismatch. Its expected-failure marker was
removed. The larger hybrid-model and LFM limitations remain separate.
Scalar gather lowering keeps a rank-one intermediate to avoid an MPS compiler
assertion, then restores the declared result shape. The Pocket TTS integration
suite and standalone scalar-gather GPU tests pass with this workaround.

## Follow-Up Evidence

| Initial gap | Implementation / runtime evidence |
|---|---|
| GatherAxis, boolean logic, rounding/sign, transcendental functions | `tests/test_extended_ops.py`, native MLX references |
| Scans, scatter/indexed updates, sort/partition/top-k | `tests/test_extended_ops.py`, including negative/duplicate indices and partition invariants |
| Pooling, resizing, reflection padding | `tests/test_extended_ops.py` and `tests/test_dynamic_pooling.py`, including dynamic spatial sizes |
| GatherMM | `tests/test_extended_ops.py`, static/dynamic batch tests on default backend |
| Bitwise invert capture and complex tensors | `tests/test_extended_ops.py`; broadcast complex/polar construction in `tests/test_data_dependent_ops.py` |
| Nonzero, masked scatter, truncation | `tests/test_data_dependent_ops.py`, synthetic IR with empty and data-dependent outputs |
| Structured conditionals and loops | `tests/test_control_flow.py`, synthetic IR executed CPU-only |
| Compression custom ops and integer storage | `tests/test_compression.py`, affine/blockwise/LUT/sparse cases and checked sub-byte packing |
| Adaptive average pooling | `tests/test_dynamic_pooling.py`, static/dynamic synthetic IR with independent references |

This closes the operator-capability gaps identified in the audit using MLX
capture where available and explicit IR otherwise. It does not claim every
ATen overload/dtype/backend combination, automatic quantized-model conversion,
or resolution of existing full-model runtime failures. The initial audit below
is historical, not the current supported-op list.

## Initial Audit

## Explicit Upstream Coverage

Apple publishes a [Supported ATen ops page](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/docs/api/supported-aten-ops.md).
It lists qualified operator/overload names and selected lowering notes, not a
complete one-to-one ATen-to-CoreAI mapping table. Many lowerings emit several
CoreAI operations or a named composite with a decomposition.

The executable inventory is
[`_aten_to_core_resolver`](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/coreai_torch/_aten_to_core.py#L3561):
190 target keys, including overloads and Python operators, dispatching to 114
distinct handlers. A separate higher-order registry handles `cond`,
`while_loop`, and internal `_yield`. These private registries are better audit
inputs than a library dependency for our converter.

The documentation currently omits four registered keys: `atan2.default`,
`masked_scatter.default`, and the Python-operator aliases `pow` and `round`.
All ATen names on the page are present in the registry. Its decomposition note
also understates the current preserve list; consult
[`_decomp.py`](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/coreai_torch/_decomp.py)
for the actual composite/direct-lowering boundaries.

Two additional inventories matter beyond ATen:

- [Composite modules](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/docs/api/composite-ops.md): `GatherMM`, `GatedDeltaUpdate`, `RMSNormImpl`, `RoPE`, and `SDPA`, plus automatically recognized ATen composites.
- [Compression custom ops](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/coreai_torch/_custom_to_core.py): quantize/dequantize, blockwise shift-scale, LUT expansion, and sparse bitmask expansion.

At the initial audit our registry had 126 lowering keys / 161 source aliases. The
then-stale `op_coverage.md` reported 123 / 158. Neither count is directly comparable
with ATen overload counts or establishes coverage of actual MLX capture names.

## Initial Confirmed Local Gaps

The following MLX cases were captured locally, not inferred only from names.
CoreAI targets below come from the corresponding upstream handlers. A target
being available does not establish numerical correctness for every MLX variant.

| Area | Upstream route | What blocks us today |
|---|---|---|
| Along-axis gather | `aten.gather` -> `gather_along_axis` | `mx.take_along_axis` emits `GatherAxis`; our existing lowering is registered only under `take_along_axis`. Needs normalization and axis decoding, not a new backend operation. |
| Boolean logic | `logical_and/or/not` -> boolean operations | MLX emits `LogicalAnd`, `LogicalOr`, `LogicalNot`; these are unregistered. Our `BitwiseBinary` coverage does not cover these primitives. |
| Rounding and sign | Floor division, negation, `round_`, comparisons | `Floor`, `Ceil`, `Round`, `Sign` are unregistered. |
| Additional transcendental functions | `acosh`, `asinh`, `cosh`, `sinh`, `tan`; decomposed `atan2` | `ArcCosh`, `ArcSinh`, `Cosh`, `Sinh`, `Tan`, `ArcTan2` are unregistered. |
| Prefix sums | `aten.cumsum` -> `scan(..., combiner="sum")` | `mx.cumsum` emits unsupported `Scan`. MLX's other scan modes need their own attribute/semantic tests. |
| Selection and sorting | `aten.topk` -> `sort` + `argsort` + slices | MLX `Sort`, `ArgSort`, and `Partition` are unsupported. `mx.topk` emits `Partition` + slice; it is not simply an alias for ATen's sorted values/indices contract. |
| General indexed updates | `index_put` -> `scatter_nd`; scatter variants -> `scatter_along_axis` | MLX indexed assignment/add emits `Scatter`; `put_along_axis` emits `ScatterAxis`. Both are unsupported. Our slice/state updates already use scatter internally, but do not expose general scatter semantics. |
| Overlapping pooling | `maxpool2d`, sum-pooling composites | Overlapping MLX max/average pooling emits unsupported `AsStrided`; non-overlapping cases can already decompose successfully. |
| Resizing | `interpolate` for nearest and bilinear | Fractional nearest MLX upsampling hits missing `Round`; linear hits missing `Floor`/`Ceil`. Integer nearest already lowers. Native interpolation is an optional optimization beyond closing these blockers. |
| Reflection padding | `pad(..., mode="reflect")` | MLX decomposes into registered slice/update ops, but the tested case fails CoreAI verification: an update expects shape `[2,1,4,4]` and receives `[2,0,4,4]`. This is a lowering correctness gap, not a missing op name. |
| MoE matmul | Externalized `GatherMM` -> named `gather_mm` composite | `mx.gather_mm` emits unsupported `GatherMM`. Ordinary gather/matmul support does not automatically decompose it. |

Two probes failed before ordinary operator lowering:

- `mx.bitwise_invert`: MLX export reports that `BitwiseInvert` cannot serialize its state. A registry entry alone will not fix that capture failure; upstream has a bitwise-NOT lowering.
- Complex output: our capture dtype parser rejects `mlx.core.complex64`. Upstream supports complex construction, polar form, and real/complex views. This needs dtype/constant/runtime handling as well as op lowerings; it does not imply upstream supports FFT.

Additional source-level gaps, not exercised by the small MLX probes:

- Upstream lowers `nonzero` to `non_zero` and has a scan/gather/select lowering for `masked_scatter`. Our frontend has no corresponding general data-dependent-output path. This is not equivalent to fixed-shape `where`.
- Upstream exports condition and loop subgraphs through `if_`/`while_`. Our flat captured graph has no general control-flow conversion; the handwritten loop inside gated-delta lowering is a special case, not general support.
- Upstream's quantization/compression custom-op pipeline has no counterpart here. Porting it involves storage dtypes, packed weights, scale/offset semantics, and capture, not just adding a matmul alias.
- ATen `trunc` and division with truncating rounding have explicit decompositions upstream; we have floor division but no general truncation lowering. Their adaptive average pooling implementation is also useful for future image-model adapters, especially runtime-computed pooling boundaries.

## Apparent Gaps That Already Lower

Static FP32 probes successfully captured and verified CoreAI IR for:

- ReLU, leaky ReLU, hard swish, hard tanh, and log-softmax.
- GroupNorm, InstanceNorm, inference BatchNorm, and an L2 vector norm.
- Integer bitwise AND/OR/XOR through `BitwiseBinary`.
- Repeat, tile, reverse slicing, and edge padding.
- Non-overlapping max/average pooling and integer nearest upsampling.
- The equivalent of `exp2` written as `mx.power(2, x)`.

These are decomposition coverage, not necessarily matching named composites or
identical performance. Pooling windows, norm variants, padding extents, dynamic
dimensions, and exceptional numerical inputs need additional tests before
generalizing these results. Our existing matmul/convolution, LayerNorm/RMSNorm,
RoPE, SDPA, and gated-delta lowerings also overlap upstream; registry presence
does not resolve the known full-model gated-delta runtime failures.

## How Useful Is the Reference?

The most useful pieces are precise CoreAI builder signatures, axis/type
normalization, dynamic shape construction, composite declarations, and the
[numerical operator tests](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/tests/ops/test_ops.py).
Their [test helper](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/tests/utils.py)
actually saves assets, executes them, and compares against Torch. It defaults to
authoring optimization disabled and uses configurable tolerances, so its
coverage is not automatically evidence for our optimized runtime path.

Do not copy contracts blindly. In particular,
[`replace_maxpool2d_with_indices`](https://github.com/apple/coreai-torch/blob/799d990deb85af350aa6a4aa23c85f211f2207ad/coreai_torch/_aten_to_core.py#L2208)
returns dummy zero indices because the CoreAI pooling op only returns values.
That registry entry does not guarantee correct programs that consume indices.
ATen/MLX layouts, top-k ordering, scatter reduction modes, signed-zero/NaN
behavior, and dtype promotion also require independent MLX references.

Suggested order, keeping generic capabilities in the core and model adaptation
in recipes:

1. Close capture-name/elementwise gaps: `GatherAxis`, logical ops, rounding/sign, then the missing transcendental functions. Recheck upsampling afterward.
2. Add scan, general scatter, sort/partition, and `GatherMM` with native MLX capture tests. These expand sequence-model and MoE coverage without adding model names to the core.
3. Add pooling/interpolation composite preservation when models justify it; do not require arbitrary strided-memory semantics merely to support pooling.
4. Treat compression, complex tensors, and general control flow as separate features rather than small op-list cleanup.

## Initial Audit Verification Scope

49 small static capture probes: 20 reached explicitly verified, unoptimized
CoreAI IR; 26 stopped on unregistered primitives; two stopped on
capture/serialization or dtype handling; one (reflection padding) emitted IR
that failed verification. Cases are deliberately gap-focused, so these counts
are not a coverage percentage. No new runtime numerical comparisons or upstream
test suite were run during this audit.

Local diagnostic script: `.build/audit_upstream_ops.py`.
Local results: `.build/audit_upstream_ops.jsonl`.
Both are ignored scratch files, not new public conversion APIs.

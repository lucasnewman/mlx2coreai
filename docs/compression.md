# Compression IR

These are generic explicit-IR operations for recipes. They are not aliases for
MLX's `quantize`/`dequantize`, which use a different packed-uint32, grouped-weight
contract. Automatic capture of MLX quantized layers is not implemented yet.

## Operations

| IR op | Inputs | Attributes / contract |
|---|---|---|
| `affine_quantize` | data, scale, offset1, offset2 | `axis=None` for per-tensor or an integer channel axis; output dtype comes from offset1 |
| `affine_dequantize` | data, scale, offset1, offset2 | Same axis convention; output dtype comes from offset2 |
| `blockwise_shift_scale` | data, scale, offset1, offset2 | Parameters have data's rank and a shape dividing data's shape along each axis |
| `lut_to_dense` | indices, table | `axis` selects the dimension expanded by the palette vector size |
| `sparse_to_dense` | nonzero values, mask | Flat values in row-major mask order; mask is a packed `uint1` constant |

Affine quantization computes
`clip(round_even((data - offset2) / scale) + offset1, qmin, qmax)`.
Dequantization and blockwise expansion compute
`(data - offset1) * scale + offset2`.
Scale must be finite and positive; offsets must be finite. Runtime-valued
parameters are the caller's responsibility. Non-finite inputs are not a
portable numerical contract across these beta backends.

Per-tensor parameters are scalars. Per-axis parameters are all scalars or
matching channel-length vectors. Scale and offset2 share a floating dtype;
offset1 has the quantized dtype. All three parameter shapes must match.

For rank-K indices, the LUT shape is `[block_0, ..., block_K-1, palettes, vector]`.
Each block count divides its corresponding indices dimension. The number of
palettes equals two raised to the index bit width. The output has the indices
shape, with the selected axis multiplied by the vector size.

## Storage

Signed/unsigned 8-bit inputs, outputs, constants, and native MLX capture preserve
their storage dtype. Affine quantization supports int8/uint8 and int4/uint4.

Constants also support `int2`, `int4`, `uint1`, `uint2`, `uint3`, `uint4`, and
`uint6`. Set a constant node's `dtype` attribute and supply ordinary unpacked
integer values in `value`. The writer validates their range and packs them in
row-major, least-significant-bit-first order, padding only the final byte.
The weight manifest records the logical dtype/shape and actual packed byte size.
Packed constants always use resource storage, regardless of the inline threshold.

Sub-byte tensors are constants or internal values, not public Python runtime
inputs/outputs. Cast or dequantize them before returning them. The packed formats
here are CoreAI storage formats, not MLX's quantized weight buffers.

## Example

```python
import numpy as np
from mlx2coreai.conversion import lower_graph_to_coreai
from mlx2coreai.ir import Graph, Node, TensorSpec

graph = Graph(
    [TensorSpec("x", (2, 3), "fp32")],
    [
        Node("constant", (), "scale", {"value": np.array(0.25, np.float32)}),
        Node("constant", (), "zero", {"value": np.array(0, np.int8)}),
        Node("constant", (), "bias", {"value": np.array(0, np.float32)}),
        Node("affine_quantize", ("x", "scale", "zero", "bias"), "q"),
        Node("affine_dequantize", ("q", "scale", "zero", "bias"), "out"),
    ],
    ["q", "out"],
)
program = lower_graph_to_coreai(graph).program
program.save_asset("artifacts/affine.aimodel")
```

## Verification And Beta Workarounds

`tests/test_compression.py` compares small graphs against independent NumPy
references, with authoring optimization both enabled and disabled. Coverage
includes saturation, half-way rounding, odd/negative zero points, per-axis and
dynamic-batch FP16/FP32 graphs, signed low-bit weights, palette block boundaries,
vector palettes, and empty sparse tensors.

- Static and dynamic 8-bit affine tests pass CPU-only execution.
- Four-bit activation quantization passes the default backend. CPU-only probes
  hung; do not force that backend for this path.
- Packed LUT and blockwise weight expansion and sparse expansion pass CPU tests.
- The native quantizer rounds after adding an odd zero point, contrary to its
  documented formula. Its FP16 per-axis offset broadcasting also produced wrong
  results. The lowering explicitly normalizes and rounds first, adds the offset,
  then uses native per-tensor quantization for saturation and packing.
- Per-axis dequantization uses explicit broadcasts and arithmetic rather than
  relying on native per-axis handling. Per-tensor and blockwise dequantization
  retain their native operations.

These tests do not establish quantized full-model accuracy, performance, or
automatic model-recipe conversion support. Float4/float8 formats are not covered.

# mlx2coreai Op Coverage

Coverage type: CoreAI asset generation. This does not imply runtime numerical parity.

## Summary

- Supported source op names in registry: 156
- Distinct lowering keys in registry: 121
- Coverage modules: `tests.model_zoo, tests.coverage_zoo`
- Coverage graphs: 26
- Coverage graph nodes: 252
- Unique source ops exercised: 156
- Unique lowering keys exercised: 121
- Asset validation: passed

## Exercised Ops

| Op | Lowering | Nodes | Models |
| --- | --- | ---: | --- |
| `abs` | `abs` | 1 | `supplemental_unary_canonical` |
| `add` | `add` | 38 | `arithmetic_chain`, `broadcast_tensordot`, `conv_block`, `diagonal_trace`, `linear_relu`, `logical_checks`, `meshgrid_kron`, `numeric_sanity`, `p0_math_pack`, `stats_divmod`, `tri_band` |
| `addmm` | `addmm` | 2 | `mlp_2layer` |
| `all` | `all` | 1 | `logical_checks` |
| `allclose` | `allclose` | 1 | `numeric_sanity` |
| `any` | `any` | 1 | `logical_checks` |
| `arange` | `arange` | 1 | `creation_helpers` |
| `arccos` | `acos` | 1 | `p0_math_pack` |
| `arcsin` | `asin` | 1 | `p0_math_pack` |
| `arctan` | `atan` | 1 | `p0_math_pack` |
| `arctanh` | `atanh` | 1 | `p0_math_pack` |
| `argmax` | `reduce_argmax` | 1 | `reduction_suite` |
| `argmin` | `reduce_argmin` | 1 | `reduction_suite` |
| `array_equal` | `array_equal` | 1 | `logical_checks` |
| `astype` | `cast` | 2 | `creation_helpers` |
| `atleast_1d` | `atleast_1d` | 1 | `shape_helpers` |
| `atleast_2d` | `atleast_2d` | 1 | `shape_helpers` |
| `atleast_3d` | `atleast_3d` | 1 | `shape_helpers` |
| `bitwisebinary` | `bitwisebinary` | 1 | `supplemental_aliases_and_bitwise` |
| `broadcast` | `broadcast_to` | 1 | `supplemental_constants_identity` |
| `broadcast_arrays` | `broadcast_arrays` | 2 | `broadcast_tensordot` |
| `broadcast_to` | `broadcast_to` | 1 | `supplemental_shape_index` |
| `cast` | `cast` | 13 | `logical_checks`, `numeric_sanity`, `p0_math_pack`, `stats_divmod` |
| `concatenate` | `concat` | 1 | `p0_math_pack` |
| `const` | `const` | 1 | `supplemental_constants_identity` |
| `constant` | `const` | 1 | `supplemental_constants_identity` |
| `contiguous` | `identity` | 1 | `supplemental_constants_identity` |
| `conv1d` | `conv` | 1 | `supplemental_convolutions` |
| `conv2d` | `conv` | 1 | `conv_block` |
| `conv3d` | `conv` | 1 | `supplemental_convolutions` |
| `conv_general` | `conv` | 1 | `conv_block` |
| `conv_transpose1d` | `conv_transpose` | 1 | `supplemental_convolutions` |
| `conv_transpose2d` | `conv_transpose` | 1 | `conv_block` |
| `conv_transpose3d` | `conv_transpose` | 1 | `supplemental_convolutions` |
| `convolution` | `conv` | 1 | `supplemental_convolutions` |
| `copy` | `identity` | 1 | `supplemental_constants_identity` |
| `cos` | `cos` | 1 | `supplemental_unary_canonical` |
| `degrees` | `degrees` | 1 | `p0_math_pack` |
| `diag` | `diag` | 2 | `diagonal_trace` |
| `diagonal` | `diagonal` | 1 | `diagonal_trace` |
| `divide` | `real_div` | 1 | `arithmetic_chain` |
| `divmod` | `divmod` | 2 | `stats_divmod` |
| `dynamic_slice_update` | `dynamic_slice_update` | 1 | `supplemental_shape_index` |
| `dynamicsliceupdate` | `dynamic_slice_update` | 1 | `supplemental_shape_index` |
| `equal` | `equal` | 1 | `supplemental_binary_canonical` |
| `erf` | `erf` | 1 | `supplemental_unary_canonical` |
| `exp` | `exp` | 1 | `supplemental_unary_canonical` |
| `expand_dims` | `expand_dims` | 1 | `supplemental_shape_index` |
| `expanddims` | `expand_dims` | 1 | `supplemental_aliases_and_bitwise` |
| `expm1` | `expm1` | 1 | `p0_math_pack` |
| `eye` | `eye` | 1 | `meshgrid_kron` |
| `flatten` | `flatten` | 1 | `shape_helpers` |
| `floor_div` | `floor_div` | 1 | `supplemental_aliases_and_bitwise` |
| `floor_divide` | `floor_div` | 1 | `p0_math_pack` |
| `full` | `full` | 1 | `creation_helpers` |
| `full_like` | `full_like` | 1 | `creation_helpers` |
| `gather` | `gather` | 1 | `supplemental_shape_index` |
| `gelu` | `gelu` | 1 | `supplemental_unary_canonical` |
| `greater` | `greater` | 1 | `supplemental_binary_canonical` |
| `greater_equal` | `greater_equal` | 1 | `supplemental_binary_canonical` |
| `greaterequal` | `greater_equal` | 1 | `supplemental_aliases_and_bitwise` |
| `inner` | `inner` | 1 | `supplemental_linear_misc` |
| `inverse` | `inverse` | 1 | `supplemental_aliases_and_bitwise` |
| `isclose` | `isclose` | 1 | `numeric_sanity` |
| `isfinite` | `isfinite` | 1 | `logical_checks` |
| `isinf` | `isinf` | 1 | `logical_checks` |
| `isnan` | `isnan` | 1 | `logical_checks` |
| `isneginf` | `isneginf` | 1 | `logical_checks` |
| `isposinf` | `isposinf` | 1 | `logical_checks` |
| `kron` | `kron` | 1 | `meshgrid_kron` |
| `layernorm` | `layernorm` | 1 | `supplemental_nn_composites` |
| `less` | `less` | 1 | `supplemental_binary_canonical` |
| `less_equal` | `less_equal` | 1 | `supplemental_binary_canonical` |
| `lessequal` | `less_equal` | 1 | `supplemental_aliases_and_bitwise` |
| `linspace` | `linspace` | 1 | `creation_helpers` |
| `log` | `log` | 1 | `supplemental_unary_canonical` |
| `log10` | `log10` | 1 | `p0_math_pack` |
| `log1p` | `log1p` | 1 | `p0_math_pack` |
| `log2` | `log2` | 1 | `p0_math_pack` |
| `logaddexp` | `logaddexp` | 1 | `meshgrid_kron` |
| `logsumexp` | `reduce_log_sum_exp` | 1 | `p0_math_pack` |
| `matmul` | `matmul` | 1 | `linear_relu` |
| `max` | `reduce_max` | 1 | `reduction_suite` |
| `maximum` | `maximum` | 3 | `conv_block`, `linear_relu`, `mlp_2layer` |
| `mean` | `reduce_mean` | 1 | `reduction_suite` |
| `meshgrid` | `meshgrid` | 2 | `meshgrid_kron` |
| `min` | `reduce_min` | 1 | `reduction_suite` |
| `minimum` | `minimum` | 1 | `supplemental_binary_canonical` |
| `mod` | `mod` | 1 | `supplemental_binary_canonical` |
| `moveaxis` | `moveaxis` | 1 | `indexing_transforms` |
| `mul` | `mul` | 1 | `supplemental_binary_canonical` |
| `multiply` | `mul` | 1 | `arithmetic_chain` |
| `nan_to_num` | `nan_to_num` | 1 | `numeric_sanity` |
| `negative` | `negative` | 1 | `p0_math_pack` |
| `not_equal` | `not_equal` | 1 | `supplemental_binary_canonical` |
| `notequal` | `not_equal` | 1 | `supplemental_aliases_and_bitwise` |
| `number_of_elements` | `number_of_elements` | 1 | `creation_helpers` |
| `ones` | `ones` | 1 | `creation_helpers` |
| `ones_like` | `ones_like` | 1 | `creation_helpers` |
| `outer` | `outer` | 1 | `supplemental_linear_misc` |
| `pow` | `pow` | 1 | `supplemental_binary_canonical` |
| `power` | `pow` | 1 | `arithmetic_chain` |
| `prod` | `reduce_prod` | 1 | `reduction_suite` |
| `radians` | `radians` | 1 | `p0_math_pack` |
| `read_state` | `read_state` | 1 | `supplemental_state_ops` |
| `real_div` | `real_div` | 1 | `supplemental_binary_canonical` |
| `reciprocal` | `inverse` | 1 | `arithmetic_chain` |
| `reduce` | `reduce` | 1 | `supplemental_reductions_canonical` |
| `reduce_argmax` | `reduce_argmax` | 1 | `supplemental_reductions_canonical` |
| `reduce_argmin` | `reduce_argmin` | 1 | `supplemental_reductions_canonical` |
| `reduce_max` | `reduce_max` | 1 | `supplemental_reductions_canonical` |
| `reduce_mean` | `reduce_mean` | 1 | `supplemental_reductions_canonical` |
| `reduce_min` | `reduce_min` | 1 | `supplemental_reductions_canonical` |
| `reduce_prod` | `reduce_prod` | 1 | `supplemental_reductions_canonical` |
| `reduce_sum` | `reduce_sum` | 1 | `supplemental_reductions_canonical` |
| `remainder` | `mod` | 1 | `arithmetic_chain` |
| `reshape` | `reshape` | 1 | `supplemental_shape_index` |
| `rmsnorm` | `rmsnorm` | 1 | `supplemental_nn_composites` |
| `rope` | `rope` | 1 | `supplemental_nn_composites` |
| `rsqrt` | `rsqrt` | 1 | `supplemental_unary_canonical` |
| `scaled_dot_product_attention` | `scaled_dot_product_attention` | 1 | `supplemental_nn_composites` |
| `scaleddotproductattention` | `scaled_dot_product_attention` | 1 | `supplemental_aliases_and_bitwise` |
| `select` | `select` | 1 | `supplemental_linear_misc` |
| `sigmoid` | `sigmoid` | 1 | `supplemental_unary_canonical` |
| `silu` | `silu` | 1 | `supplemental_unary_canonical` |
| `sin` | `sin` | 1 | `supplemental_unary_canonical` |
| `slice` | `slice_by_index` | 1 | `indexing_transforms` |
| `slice_by_index` | `slice_by_index` | 1 | `supplemental_shape_index` |
| `slice_update` | `slice_update` | 1 | `supplemental_shape_index` |
| `sliceupdate` | `slice_update` | 1 | `supplemental_aliases_and_bitwise` |
| `softmax` | `softmax` | 1 | `supplemental_nn_composites` |
| `split` | `split` | 1 | `supplemental_shape_index` |
| `sqrt` | `sqrt` | 1 | `supplemental_unary_canonical` |
| `squeeze` | `squeeze` | 1 | `supplemental_shape_index` |
| `state_update_masked` | `state_update_masked` | 1 | `supplemental_state_ops` |
| `std` | `std` | 1 | `stats_divmod` |
| `stop_gradient` | `identity` | 1 | `creation_helpers` |
| `sub` | `sub` | 1 | `supplemental_binary_canonical` |
| `subtract` | `sub` | 1 | `arithmetic_chain` |
| `sum` | `reduce_sum` | 38 | `broadcast_tensordot`, `diagonal_trace`, `logical_checks`, `meshgrid_kron`, `numeric_sanity`, `p0_math_pack`, `reduction_suite`, `stats_divmod`, `tri_band` |
| `swapaxes` | `swapaxes` | 1 | `indexing_transforms` |
| `take` | `gather` | 1 | `indexing_transforms` |
| `take_along_axis` | `gather_along_axis` | 1 | `indexing_transforms` |
| `tanh` | `tanh` | 1 | `supplemental_unary_canonical` |
| `tensordot` | `tensordot` | 1 | `broadcast_tensordot` |
| `trace` | `trace` | 1 | `diagonal_trace` |
| `transpose` | `transpose` | 1 | `supplemental_shape_index` |
| `tri` | `tri` | 1 | `tri_band` |
| `tril` | `tril` | 2 | `tri_band` |
| `triu` | `triu` | 2 | `tri_band` |
| `unflatten` | `unflatten` | 1 | `shape_helpers` |
| `var` | `var` | 1 | `stats_divmod` |
| `where` | `select` | 1 | `creation_helpers` |
| `write_state` | `write_state` | 1 | `supplemental_state_ops` |
| `zeros` | `zeros` | 1 | `creation_helpers` |
| `zeros_like` | `zeros_like` | 1 | `creation_helpers` |

## Coverage Graph Assets

| Module | Graph | Nodes | Unique Ops | Asset |
| --- | --- | ---: | ---: | --- |
| `tests.model_zoo` | `arithmetic_chain` | 7 | 7 | passed |
| `tests.model_zoo` | `broadcast_tensordot` | 6 | 4 | passed |
| `tests.model_zoo` | `conv_block` | 5 | 5 | passed |
| `tests.model_zoo` | `creation_helpers` | 13 | 12 | passed |
| `tests.model_zoo` | `diagonal_trace` | 11 | 5 | passed |
| `tests.model_zoo` | `indexing_transforms` | 5 | 5 | passed |
| `tests.model_zoo` | `linear_relu` | 3 | 3 | passed |
| `tests.model_zoo` | `logical_checks` | 28 | 11 | passed |
| `tests.model_zoo` | `meshgrid_kron` | 14 | 6 | passed |
| `tests.model_zoo` | `mlp_2layer` | 3 | 2 | passed |
| `tests.model_zoo` | `numeric_sanity` | 7 | 6 | passed |
| `tests.model_zoo` | `p0_math_pack` | 38 | 17 | passed |
| `tests.model_zoo` | `reduction_suite` | 7 | 7 | passed |
| `tests.model_zoo` | `shape_helpers` | 5 | 5 | passed |
| `tests.model_zoo` | `stats_divmod` | 13 | 6 | passed |
| `tests.model_zoo` | `tri_band` | 14 | 5 | passed |
| `tests.coverage_zoo` | `supplemental_aliases_and_bitwise` | 9 | 9 | passed |
| `tests.coverage_zoo` | `supplemental_binary_canonical` | 12 | 12 | passed |
| `tests.coverage_zoo` | `supplemental_constants_identity` | 5 | 5 | passed |
| `tests.coverage_zoo` | `supplemental_convolutions` | 5 | 5 | passed |
| `tests.coverage_zoo` | `supplemental_linear_misc` | 3 | 3 | passed |
| `tests.coverage_zoo` | `supplemental_nn_composites` | 5 | 5 | passed |
| `tests.coverage_zoo` | `supplemental_reductions_canonical` | 8 | 8 | passed |
| `tests.coverage_zoo` | `supplemental_shape_index` | 11 | 11 | passed |
| `tests.coverage_zoo` | `supplemental_state_ops` | 3 | 3 | passed |
| `tests.coverage_zoo` | `supplemental_unary_canonical` | 12 | 12 | passed |

## Unexercised Registry Ops

None

## Notes

- Coverage is asset-generation coverage, not runtime numerical parity.
- Runtime parity requires the macOS / iOS 27+ CoreAI execution stack.
- General transposed convolution uses a named composite fallback when the beta CoreAI asset writer rejects native conv_transpose IR; the vendored 1x1 stride-1 case lowers without that fallback.

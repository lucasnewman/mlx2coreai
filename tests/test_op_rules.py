import numpy as np
import pytest

from mlx2coreai.op_registry import (
    SUPPORTED_MLX_TO_COREAI_OPS, coreai_op_for_mlx,
    decode_primitive_arguments, rule_for_mlx,
)
from mlx2coreai.op_rules import OPERATIONS
from mlx2coreai.lower_to_coreai import CoreAILowerer


def test_supported_operations_are_derived_from_executable_rules():
    aliases = [alias for rule in OPERATIONS.values() for alias in rule.aliases]
    assert len(aliases) == len(set(aliases))
    assert set(aliases) == set(SUPPORTED_MLX_TO_COREAI_OPS)

    class MethodProbe:
        def __getattr__(self, name):
            assert callable(getattr(CoreAILowerer, name)), name
            return lambda *args, **kwargs: name

    for key, rule in OPERATIONS.items():
        assert rule.key == key
        assert callable(rule.lower)
        assert rule.decode is None or callable(rule.decode)
        assert rule.infer is None or callable(rule.infer)
        rule.lower(MethodProbe(), None)
        for alias in rule.aliases:
            assert coreai_op_for_mlx(alias) == key
            assert rule_for_mlx(alias) is rule


@pytest.mark.parametrize("op,args,shape,dtype,expected", [
    ("reshape", [], (2, 3), "fp32", {"shape": [2, 3]}),
    ("broadcast_axes", [[-2, -1]], (3, 4), "fp32", {"ignore_axes": [-2, -1]}),
    ("take", [2], (2, 3), "fp32", {"axis": 2}),
    ("gather", [[1], [2, 1]], (2, 3), "fp32", {"axis": 1, "axes": [1], "slice_shape": [2, 1], "shape": [2, 3]}),
    ("reduce", [2, [1]], (2, 1), "fp32", {"mode": 2, "axes": [1], "keep_dims": True}),
    ("slice", [[0, 2], [2, 5], [1, 1]], (2, 3), "fp32", {"begin": [0, 2], "end": [2, 5], "stride": [1, 1]}),
    ("cast", [], (2,), "bf16", {"dtype": "bf16"}),
    ("scaled_dot_product_attention", [None, 0.125, True, False, False], (1, 2, 3, 4), "fp32",
     {"scale": 0.125, "do_causal": True, "has_sinks": False, "output_logsumexp": False}),
    ("convolution", [[1], [2], [3], [1], [], 4, np.bool_(False)], (1, 8, 4), "fp32",
     {"channels_last": True, "strides": [1], "padding": [2, 3], "pad_type": "custom", "dilations": [1], "input_dilations": [], "groups": 4, "flip": False}),
])
def test_argument_codecs_preserve_primitive_contracts(op, args, shape, dtype, expected):
    assert decode_primitive_arguments(op, args, shape, dtype) == expected

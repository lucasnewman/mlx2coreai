from __future__ import annotations

from pathlib import Path

import numpy as np

from mlx2coreai.conversion import ConversionConfig, lower_graph_to_coreai
from mlx2coreai.dynamic_shapes import dynamicize_graph_from_probe
from mlx2coreai.ir import Graph, Node, TensorSpec, dynamic_dim_ref, is_dynamic_dim_ref


def test_smoke_asset_generation(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("y", (2, 3), "fp32")],
        nodes=[
            Node("add", ("x", "y"), "z"),
            Node("tanh", ("z",), "out"),
        ],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=True))
    asset_path = tmp_path / "smoke.aimodel"
    lowered.program.save_asset(asset_path)
    assert sorted(path.name for path in asset_path.iterdir()) == [
        "main.hash",
        "main.mlirb",
        "metadata.json",
    ]


def test_externalized_weight_is_not_public_input(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("w", (3, 4), "fp32")],
        nodes=[Node("matmul", ("x", "w"), "out")],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(
        graph,
        config=ConversionConfig(
            optimize=True,
            constant_inputs={"w": np.ones((3, 4), dtype=np.float32)},
        ),
        public_input_names={"x"},
    )
    assert [spec.name for spec in lowered.public_inputs] == ["x"]
    assert [entry.name for entry in lowered.weight_manifest] == ["w"]
    asset_path = tmp_path / "weighted.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_large_constant_inputs_use_dense_resources(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("w", (3, 4), "fp32")],
        nodes=[Node("matmul", ("x", "w"), "out")],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(
        graph,
        config=ConversionConfig(
            optimize=False,
            constant_inputs={"w": np.ones((3, 4), dtype=np.float32)},
            external_weight_threshold=10,
        ),
        public_input_names={"x"},
    )
    assert lowered.weight_manifest[0].storage == "resource"
    assert lowered.weight_manifest[0].resource_name == "mlx2coreai_w"
    assert lowered.weight_manifest[0].nbytes == 48
    assert "dense_resource" in str(lowered.program)
    asset_path = tmp_path / "resource_weight.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_small_constant_inputs_stay_inline() -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 3), "fp32"), TensorSpec("bias", (3,), "fp32")],
        nodes=[Node("add", ("x", "bias"), "out")],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(
        graph,
        config=ConversionConfig(
            optimize=False,
            constant_inputs={"bias": np.ones((3,), dtype=np.float32)},
            external_weight_threshold=10,
        ),
        public_input_names={"x"},
    )
    assert lowered.weight_manifest[0].storage == "inline"
    assert "dense_resource" not in str(lowered.program)


def test_coreai_composite_declarations_are_emitted() -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (2, 4), "fp32"), TensorSpec("scale", (4,), "fp32")],
        nodes=[Node("rmsnorm", ("x", "scale"), "out", attrs={"eps": 1e-5})],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    text = str(lowered.program)
    assert 'composite_declaration<"rms_norm"' in text


def test_sdpa_and_rope_composites_save(tmp_path: Path) -> None:
    sdpa = Graph(
        inputs=[
            TensorSpec("q", (1, 2, 3, 4), "fp32"),
            TensorSpec("k", (1, 2, 3, 4), "fp32"),
            TensorSpec("v", (1, 2, 3, 4), "fp32"),
        ],
        nodes=[
            Node(
                "scaled_dot_product_attention",
                ("q", "k", "v"),
                "attn",
                attrs={"scale": 0.5, "do_causal": True},
            )
        ],
        outputs=["attn"],
    )
    rope = Graph(
        inputs=[TensorSpec("x", (1, 2, 3, 4), "fp32")],
        nodes=[Node("rope", ("x",), "out", attrs={"dims": 4, "base": 10000.0})],
        outputs=["out"],
    )
    for name, graph in {"sdpa": sdpa, "rope": rope}.items():
        lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
        asset_path = tmp_path / f"{name}.aimodel"
        lowered.program.save_asset(asset_path)
        assert (asset_path / "main.mlirb").exists()


def test_sdpa_composite_supports_grouped_query_attention(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[
            TensorSpec("q", (1, 16, 4, 8), "fp32"),
            TensorSpec("k", (1, 8, 4, 8), "fp32"),
            TensorSpec("v", (1, 8, 4, 8), "fp32"),
        ],
        nodes=[
            Node(
                "scaled_dot_product_attention",
                ("q", "k", "v"),
                "out",
                attrs={"scale": 0.3535533905932738, "do_causal": True},
            )
        ],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    text = str(lowered.program)
    assert "coreai.greater" in text
    assert "coreai.broadcast_to" in text
    asset_path = tmp_path / "gqa.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_rope_composite_explicitly_broadcasts_trig_tables(tmp_path: Path) -> None:
    graph = Graph(
        inputs=[TensorSpec("x", (1, 1, 4, 128), "fp32")],
        nodes=[Node("rope", ("x",), "out", attrs={"dims": 128, "base": 10000.0})],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    text = str(lowered.program)
    assert "coreai.broadcast_to" in text
    assert "tensor<1x1x4x64xf32>" in text
    asset_path = tmp_path / "rope_single_head.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_dynamic_dim_refs_lower_to_coreai_runtime_shapes(tmp_path: Path) -> None:
    seq = dynamic_dim_ref("input_ids", 1)
    graph = Graph(
        inputs=[
            TensorSpec("embedding", (32, 8), "fp32"),
            TensorSpec("input_ids", (1, -1), "int32"),
        ],
        nodes=[
            Node(
                "gather",
                ("embedding", "input_ids"),
                "gathered",
                attrs={"axis": 0, "slice_shape": [1, 8], "shape": [1, seq, 1, 8]},
            ),
            Node("squeeze", ("gathered",), "tokens", attrs={"axes": [2]}),
            Node("reshape", ("tokens",), "out", attrs={"shape": [1, seq, 8]}),
        ],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    text = str(lowered.program)
    assert "tensor<1x?xsi32>" in text
    assert "coreai.get_shape" in text
    asset_path = tmp_path / "dynamic_embedding.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_dynamic_broadcast_preserves_static_result_dims(tmp_path: Path) -> None:
    seq = dynamic_dim_ref("x", 1)
    graph = Graph(
        inputs=[TensorSpec("x", (1, -1, 1024), "fp32")],
        nodes=[
            Node("broadcast_to", ("x",), "out", attrs={"shape": [1, seq, 1024]}),
        ],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=False))
    text = str(lowered.program)
    assert "coreai.broadcast_to" in text
    assert "tensor<1x?x1024xf32>" in text
    assert "tensor<?x?x?xf32>" not in text
    asset_path = tmp_path / "dynamic_broadcast.aimodel"
    lowered.program.save_asset(asset_path)
    assert (asset_path / "main.mlirb").exists()


def test_dynamic_causal_sdpa_runs_coreai_optimizer() -> None:
    graph = Graph(
        inputs=[
            TensorSpec("q", (1, 16, -1, 8), "fp32"),
            TensorSpec("k", (1, 8, -1, 8), "fp32"),
            TensorSpec("v", (1, 8, -1, 8), "fp32"),
        ],
        nodes=[
            Node(
                "scaled_dot_product_attention",
                ("q", "k", "v"),
                "out",
                attrs={"scale": 0.3535533905932738, "do_causal": True},
            )
        ],
        outputs=["out"],
    )
    lowered = lower_graph_to_coreai(graph, config=ConversionConfig(optimize=True))
    assert lowered.optimized is True
    assert lowered.optimization_skip_reason is None


def test_probe_dynamicizes_sequence_attrs() -> None:
    base = Graph(
        inputs=[TensorSpec("input_ids", (1, 4), "int32")],
        nodes=[
            Node("arange", tuple(), "pos", attrs={"start": 0, "end": 4, "step": 1}),
            Node("reshape", ("input_ids",), "out", attrs={"shape": [1, 4]}),
        ],
        outputs=["out"],
    )
    probe = Graph(
        inputs=[TensorSpec("input_ids", (1, 5), "int32")],
        nodes=[
            Node("arange", tuple(), "pos", attrs={"start": 0, "end": 5, "step": 1}),
            Node("reshape", ("input_ids",), "out", attrs={"shape": [1, 5]}),
        ],
        outputs=["out"],
    )
    dynamic = dynamicize_graph_from_probe(
        base,
        probe,
        dynamic_axes={"input_ids": [1]},
        base_inputs={"input_ids": np.zeros((1, 4), dtype=np.int32)},
        probe_inputs={"input_ids": np.zeros((1, 5), dtype=np.int32)},
    )
    assert dynamic.inputs[0].shape == (1, -1)
    assert is_dynamic_dim_ref(dynamic.nodes[0].attrs["end"])
    assert is_dynamic_dim_ref(dynamic.nodes[1].attrs["shape"][1])

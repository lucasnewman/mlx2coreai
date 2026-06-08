from pathlib import Path

from mlx2coreai.conversion import ConversionConfig, lower_graph_to_coreai

from . import model_zoo


def test_static_mlx2coreml_model_zoo_assets_save(tmp_path: Path) -> None:
    builders = [
        name
        for name in dir(model_zoo)
        if name.startswith("_build_") and name != "_build_smoke_numpy_inputs"
    ]
    assert builders

    for builder_name in builders:
        spec = getattr(model_zoo, builder_name)(0)
        lowered = lower_graph_to_coreai(spec.graph, config=ConversionConfig(optimize=False))
        asset_path = tmp_path / f"{spec.name}.aimodel"
        lowered.program.save_asset(asset_path)
        assert (asset_path / "main.mlirb").exists(), spec.name

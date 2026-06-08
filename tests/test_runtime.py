from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ml_dtypes

from mlx2coreai import (
    compare_coreai_outputs,
    run_aimodel,
    run_aimodel_sync,
    validate_aimodel_outputs,
    validate_aimodel_outputs_sync,
)
import mlx2coreai.runtime as runtime


class FakeStorageKind(Enum):
    BYTES = "bytes"
    METAL = "metal"


class FakeNDArray:
    def __init__(self, data, backing=None):
        self.data = data
        self.backing = backing

    def numpy(self):
        return np.asarray(self.data)


class FakeFunction:
    def __init__(self, calls: dict[str, object]):
        self.calls = calls

    async def __call__(self, *, inputs):
        self.calls["inputs"] = inputs
        self.calls["input_backing"] = inputs["x"].backing
        return {"out": FakeNDArray(np.asarray(inputs["x"].data) + 1.0)}


class FakeModel:
    def __init__(self, calls: dict[str, object]):
        self.calls = calls

    def load_function(self, function_name: str):
        self.calls["function_name"] = function_name
        return FakeFunction(self.calls)


class FakeExecutable:
    def __init__(self, calls: dict[str, object], specialization_options):
        self.calls = calls
        self.specialization_options = specialization_options

    async def __aenter__(self):
        self.calls["specialization_options"] = self.specialization_options
        return FakeModel(self.calls)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAsset:
    def __init__(self, path: Path, calls: dict[str, object]):
        self.path = path
        self.calls = calls

    def executable(self, specialization_options=None):
        self.calls["asset_path"] = self.path
        return FakeExecutable(self.calls, specialization_options)


class FakeAIModelAsset:
    calls: dict[str, object] = {}

    @classmethod
    def load(cls, path):
        cls.calls["loaded_path"] = Path(path)
        return FakeAsset(Path(path), cls.calls)


def _install_fake_runtime(monkeypatch):
    FakeAIModelAsset.calls = {}
    monkeypatch.setattr(
        runtime,
        "_load_coreai_runtime",
        lambda: SimpleNamespace(
            AIModelAsset=FakeAIModelAsset,
            NDArray=FakeNDArray,
            StorageKind=FakeStorageKind,
        ),
    )
    return FakeAIModelAsset.calls


def test_run_aimodel_wraps_inputs_and_outputs(monkeypatch, tmp_path: Path) -> None:
    calls = _install_fake_runtime(monkeypatch)
    asset_path = tmp_path / "model.aimodel"

    result = asyncio.run(
        run_aimodel(
            asset_path,
            {"x": np.asarray([1.0, 2.0], dtype=np.float32)},
            function_name="main",
            specialization_options="options",
            storage_kind="metal",
        )
    )

    assert calls["loaded_path"] == asset_path
    assert calls["asset_path"] == asset_path
    assert calls["specialization_options"] == "options"
    assert calls["function_name"] == "main"
    assert calls["input_backing"] is FakeStorageKind.METAL
    assert result.asset_path == asset_path
    assert result.outputs["out"].tolist() == [2.0, 3.0]


def test_run_aimodel_sync(monkeypatch, tmp_path: Path) -> None:
    _install_fake_runtime(monkeypatch)

    result = run_aimodel_sync(
        tmp_path / "model.aimodel",
        {"x": [3.0]},
    )

    assert result.outputs["out"].tolist() == [4.0]


def test_compare_coreai_outputs_matches_by_order() -> None:
    comparisons = compare_coreai_outputs(
        {"runtime_out": np.asarray([1.0, 2.0], dtype=np.float32)},
        {"captured_out": np.asarray([1.0, 2.0], dtype=np.float32)},
    )

    assert len(comparisons) == 1
    assert comparisons[0].passed
    assert comparisons[0].actual_name == "runtime_out"
    assert comparisons[0].expected_name == "captured_out"


def test_compare_coreai_outputs_reports_failures() -> None:
    comparisons = compare_coreai_outputs(
        {"out": np.asarray([1.0, 3.0], dtype=np.float32)},
        {"out": np.asarray([1.0, 2.0], dtype=np.float32)},
        atol=0.0,
        rtol=0.0,
    )

    assert len(comparisons) == 1
    assert not comparisons[0].passed
    assert comparisons[0].message == "values differ"
    assert comparisons[0].max_abs_error == 1.0


def test_compare_coreai_outputs_handles_bfloat16() -> None:
    comparisons = compare_coreai_outputs(
        {"out": np.asarray([1.0, 2.0], dtype=ml_dtypes.bfloat16)},
        {"out": np.asarray([1.0, 2.003], dtype=np.float32)},
        atol=0.01,
        rtol=0.0,
    )

    assert comparisons[0].passed
    assert comparisons[0].max_abs_error is not None


def test_validate_aimodel_outputs(monkeypatch, tmp_path: Path) -> None:
    _install_fake_runtime(monkeypatch)

    result = asyncio.run(
        validate_aimodel_outputs(
            tmp_path / "model.aimodel",
            {"x": np.asarray([1.0], dtype=np.float32)},
            {"out": np.asarray([2.0], dtype=np.float32)},
        )
    )

    assert result.passed
    assert result.runtime.outputs["out"].tolist() == [2.0]


def test_validate_aimodel_outputs_sync(monkeypatch, tmp_path: Path) -> None:
    _install_fake_runtime(monkeypatch)

    result = validate_aimodel_outputs_sync(
        tmp_path / "model.aimodel",
        {"x": np.asarray([4.0], dtype=np.float32)},
        {"out": np.asarray([5.0], dtype=np.float32)},
    )

    assert result.passed

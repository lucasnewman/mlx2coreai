from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import ml_dtypes
import pytest

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


def test_shared_function_session_lifetime(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)
    opened, closed = [], []
    original_enter = FakeExecutable.__aenter__

    async def enter(self):
        opened.append(self)
        return await original_enter(self)

    async def close(self, *exc):
        closed.append(self)

    monkeypatch.setattr(FakeExecutable, "__aenter__", enter)
    monkeypatch.setattr(FakeExecutable, "__aexit__", close)

    async def check():
        async with runtime.CoreAISession(tmp_path / "model.aimodel", function_name="encode") as owner:
            child = owner.function_session("decode")
            async with child:
                result = await child.run({"x": np.array([2], np.float32)})
                assert result["out"].numpy().tolist() == [3]
                assert child.asset_path == owner.asset_path
                assert len(opened) == 1 and not closed
            assert not closed
            assert (await owner.run({"x": [4]}))["out"].numpy().tolist() == [5]
            with pytest.raises(RuntimeError, match="async with"):
                child.function
        assert len(closed) == 1
        with pytest.raises(RuntimeError, match="async with"):
            owner.function_session("decode")

    asyncio.run(check())


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


def test_session_reuses_function_and_isolates_state(monkeypatch, tmp_path):
    calls = _install_fake_runtime(monkeypatch)
    loaded, closed = [], []

    class StatefulFunction:
        desc = SimpleNamespace(
            state_names=["cache"],
            state_descriptor=lambda name: SimpleNamespace(shape=(1, -1), dtype="float32"),
        )

        async def __call__(self, *, inputs, state):
            calls.setdefault("states", []).append(state)
            state["cache"].data += 1
            return {"logits": inputs["input_ids"]}

    def load_function(self, name):
        loaded.append(name)
        return StatefulFunction()

    async def close(self, *exc):
        closed.append(exc)

    monkeypatch.setattr(FakeModel, "load_function", load_function)
    monkeypatch.setattr(FakeExecutable, "__aexit__", close)

    async def check():
        session = runtime.CoreAISession(tmp_path / "model.aimodel", storage_kind="metal")
        with pytest.raises(RuntimeError, match="async with"):
            await session.run({})
        async with session:
            with pytest.raises(RuntimeError, match="already open"):
                await session.__aenter__()
            state = session.reset_state(state_capacity=4)
            assert state["cache"].backing is FakeStorageKind.METAL
            snapshot = session.snapshot_state()
            clone = session.clone_state()
            for buffers in (None, clone, None):
                await session.run({"input_ids": np.array([[7]], np.int32),
                                   "position_ids": np.array([[0]], np.int32)}, state=buffers)
            np.testing.assert_array_equal(snapshot["cache"], [[0] * 4])
            np.testing.assert_array_equal(clone["cache"].numpy(), [[1] * 4])
            np.testing.assert_array_equal(state["cache"].numpy(), [[2] * 4])
            assert calls["states"][0] is calls["states"][2] is state
            assert session.reset_state(state_capacity=2)["cache"].numpy().shape == (1, 2)
        assert session.state == {}
        with pytest.raises(RuntimeError, match="async with"):
            session.snapshot_state()
        assert len(loaded) == len(closed) == 1

    asyncio.run(check())


def test_session_closes_executable_if_function_load_fails(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)
    closed = []

    def load_function(self, name):
        raise ValueError("missing function")

    async def close(self, exc_type, exc, tb):
        closed.append(exc)

    monkeypatch.setattr(FakeModel, "load_function", load_function)
    monkeypatch.setattr(FakeExecutable, "__aexit__", close)

    async def check():
        async with runtime.CoreAISession(tmp_path / "model.aimodel"):
            pytest.fail("must not enter")

    with pytest.raises(ValueError, match="missing function"):
        asyncio.run(check())
    assert len(closed) == 1
    assert isinstance(closed[0], ValueError)


@pytest.mark.parametrize("nested_asset", [False, True])
def test_runtime_requires_explicit_asset_path(monkeypatch, tmp_path, nested_asset):
    calls = _install_fake_runtime(monkeypatch)
    if nested_asset:
        (tmp_path / "main.aimodel").mkdir()
    with pytest.raises(ValueError, match="explicit .aimodel asset path"):
        run_aimodel_sync(tmp_path, {"x": [1.0]})
    assert "loaded_path" not in calls

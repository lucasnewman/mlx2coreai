from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai import ConversionConfig, build_mlx_lm_inputs, convert_mlx_lm
from mlx2coreai import convert_mlx_lm as exported_convert_mlx_lm
from mlx2coreai import convert_mlx_to_coreai as exported_convert_mlx_to_coreai
from mlx2coreai._convert_mlx_lm import load_mlx_lm_model, parse_args


class FakeTokenizer:
    eos_token_id = 9

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "hello"
        return [1, 2, 3]


def test_build_mlx_lm_inputs_tokenizes_pads_and_batches() -> None:
    built = build_mlx_lm_inputs(
        tokenizer=FakeTokenizer(),
        prompt="hello",
        sequence_length=5,
        batch_size=2,
    )
    assert built.input_ids.dtype == np.int32
    assert built.input_ids.shape == (2, 5)
    assert built.input_ids.tolist() == [[1, 2, 3, 9, 9], [1, 2, 3, 9, 9]]
    assert built.token_count == 3
    assert built.padded_token_count == 2
    assert built.as_dict()["input_ids"] is built.input_ids


def test_build_mlx_lm_inputs_uses_prompt_length_when_sequence_length_omitted() -> None:
    built = build_mlx_lm_inputs(
        tokenizer=FakeTokenizer(),
        prompt="hello",
    )
    assert built.input_ids.shape == (1, 3)
    assert built.input_ids.tolist() == [[1, 2, 3]]
    assert built.token_count == 3
    assert built.padded_token_count == 0
    assert built.synthetic is False


def test_build_mlx_lm_inputs_synthesizes_without_prompt() -> None:
    built = build_mlx_lm_inputs(
        tokenizer=FakeTokenizer(),
    )
    assert built.input_ids.shape == (1, 1)
    assert built.input_ids.tolist() == [[9]]
    assert built.prompt is None
    assert built.token_count == 1
    assert built.padded_token_count == 0
    assert built.synthetic is True


def test_build_mlx_lm_inputs_synthesizes_requested_length() -> None:
    built = build_mlx_lm_inputs(
        tokenizer=FakeTokenizer(),
        sequence_length=4,
        batch_size=2,
    )
    assert built.input_ids.shape == (2, 4)
    assert built.input_ids.tolist() == [[9, 9, 9, 9], [9, 9, 9, 9]]
    assert built.token_count == 4
    assert built.padded_token_count == 0
    assert built.synthetic is True


def test_build_mlx_lm_inputs_accepts_explicit_ids() -> None:
    built = build_mlx_lm_inputs(
        tokenizer=None,
        input_ids=[5, 6, 7],
        sequence_length=16,
        batch_size=4,
    )
    assert built.input_ids.shape == (1, 3)
    assert built.input_ids.tolist() == [[5, 6, 7]]
    assert built.token_count == 3
    assert built.padded_token_count == 0


def test_load_mlx_lm_model_forwards_lazy_and_revision() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_load(model_id: str, **kwargs: object):
        calls.append((model_id, kwargs))
        return object(), FakeTokenizer()

    load_mlx_lm_model("mlx-community/foo", lazy_load=True, revision="abc", load_fn=fake_load)
    assert calls == [("mlx-community/foo", {"lazy": True, "revision": "abc"})]


def test_convert_mlx_lm_uses_loader_tokenizer_and_converter(monkeypatch, tmp_path: Path) -> None:
    module = importlib.import_module("mlx2coreai._convert_mlx_lm")

    class FakeModel:
        eval_called = False

        def eval(self) -> None:
            self.eval_called = True

    model = FakeModel()
    calls: dict[str, object] = {}

    def fake_load(model_id: str, **kwargs: object):
        calls["load"] = (model_id, kwargs)
        return model, FakeTokenizer()

    def fake_converter(target, inputs, *, config, output_path, dot_output_path, capture_function):
        calls["convert"] = {
            "target": target,
            "inputs": inputs,
            "config": config,
            "output_path": output_path,
            "dot_output_path": dot_output_path,
            "capture_function": capture_function,
        }
        return SimpleNamespace(metadata={}, asset_path=output_path)

    monkeypatch.setattr(module, "convert_mlx_to_coreai", fake_converter)

    result = convert_mlx_lm(
        "mlx-community/foo",
        tmp_path / "foo.aimodel",
        prompt="hello",
        sequence_length=4,
        lazy_load=True,
        revision="abc",
        config=ConversionConfig(optimize=False),
        dot_output_path=tmp_path / "capture.dot",
        load_fn=fake_load,
    )

    assert model.eval_called
    assert calls["load"] == ("mlx-community/foo", {"lazy": True, "revision": "abc"})
    convert_call = calls["convert"]
    assert convert_call["target"] is model
    assert convert_call["output_path"] == tmp_path / "foo.aimodel"
    assert convert_call["dot_output_path"] == tmp_path / "capture.dot"
    assert convert_call["inputs"]["input_ids"].tolist() == [[1, 2, 3, 9]]
    assert convert_call["config"].dynamic_axes == {"input_ids": [1]}
    assert convert_call["config"].dynamic_probe_inputs["input_ids"].tolist() == [[1, 2, 3, 9, 9]]
    assert result.metadata["mlx_lm"]["model_id"] == "mlx-community/foo"
    assert result.metadata["mlx_lm"]["sequence_length"] == 4
    assert result.metadata["mlx_lm"]["capture_sequence_length"] == 4
    assert result.metadata["mlx_lm"]["dynamic_sequence"] is True
    assert result.metadata["mlx_lm"]["synthetic_input_ids"] is False


def test_convert_mlx_lm_live_mlx_smoke_saves_asset(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    class TinyModel:
        eval_called = False

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, input_ids):
            return input_ids.astype(mx.float32) + np.float32(1.0)

    model = TinyModel()

    def fake_load(model_id: str, **kwargs: object):
        assert model_id == "tiny"
        assert kwargs == {"lazy": False}
        return model, None

    converted = convert_mlx_lm(
        "tiny",
        tmp_path / "tiny.aimodel",
        input_ids=np.ones((1, 4), dtype=np.int32),
        config=ConversionConfig(optimize=False),
        load_fn=fake_load,
    )
    assert model.eval_called
    assert (tmp_path / "tiny.aimodel" / "main.mlirb").exists()
    assert converted.metadata["mlx_lm"]["token_count"] == 4


def test_parse_args_accepts_model_and_output() -> None:
    args = parse_args(
        [
            "mlx-community/foo",
            "--output",
            "foo.aimodel",
            "--seq-len",
            "8",
            "--external-weight-threshold",
            "32",
            "--no-externalize-weights",
        ]
    )
    assert args.model_id == "mlx-community/foo"
    assert args.output == Path("foo.aimodel")
    assert args.sequence_length == 8
    assert args.external_weight_threshold == 32
    assert not args.externalize_weights
    assert args.dynamic_sequence is True


def test_parse_args_sequence_length_is_optional() -> None:
    args = parse_args(["mlx-community/foo", "--output", "foo.aimodel"])
    assert args.sequence_length is None
    assert args.prompt is None


def test_public_exports_are_available() -> None:
    assert exported_convert_mlx_lm is convert_mlx_lm
    assert callable(exported_convert_mlx_to_coreai)

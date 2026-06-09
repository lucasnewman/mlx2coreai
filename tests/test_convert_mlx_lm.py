from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai import ConversionConfig, build_mlx_lm_inputs, convert_mlx_lm, convert_mlx_lm_stateful
from mlx2coreai import convert_mlx_lm as exported_convert_mlx_lm
from mlx2coreai import convert_mlx_lm_stateful as exported_convert_mlx_lm_stateful
from mlx2coreai import convert_mlx_to_coreai as exported_convert_mlx_to_coreai
from mlx2coreai._convert_mlx_lm import load_mlx_lm_model, parse_args


class FakeTokenizer:
    eos_token_id = 9
    vocab_size = 10

    def encode(self, prompt: str) -> list[int]:
        assert prompt == "hello"
        return [1, 2, 3]

    def save_pretrained(self, path: str) -> None:
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "tokenizer.json").write_text("{}", encoding="utf-8")


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


def test_convert_mlx_lm_stateful_live_mlx_smoke_saves_unified_asset(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")

    class TinyModel:
        eval_called = False
        args = SimpleNamespace(num_key_value_heads=1, head_dim=1)
        layers = [SimpleNamespace(self_attn=SimpleNamespace(n_kv_heads=1))]

        def eval(self) -> None:
            self.eval_called = True

        def __call__(self, input_ids, *, cache):
            values = mx.reshape(input_ids.astype(mx.float32), (1, 1, input_ids.shape[1], 1))
            cache[0].update_and_fetch(values, values)
            return input_ids.astype(mx.float32)

    model = TinyModel()

    def fake_load(model_id: str, **kwargs: object):
        assert model_id == "tiny-stateful"
        assert kwargs == {"lazy": False}
        return model, FakeTokenizer()

    converted = convert_mlx_lm_stateful(
        "tiny-stateful",
        tmp_path / "tiny-stateful",
        input_name="input_ids",
        max_context_length=8,
        dynamic_sequence=False,
        dynamic_state=False,
        config=ConversionConfig(optimize=False),
        load_fn=fake_load,
    )

    assert model.eval_called
    bundle_path = tmp_path / "tiny-stateful"
    asset_path = bundle_path / "tiny-stateful.aimodel"
    assert (asset_path / "main.mlirb").exists()
    assert (bundle_path / "tokenizer" / "tokenizer.json").exists()
    bundle_metadata = json.loads((bundle_path / "metadata.json").read_text(encoding="utf-8"))
    assert bundle_metadata["metadata_version"] == "0.2"
    assert bundle_metadata["kind"] == "llm"
    assert bundle_metadata["assets"] == {"main": "tiny-stateful.aimodel"}
    assert bundle_metadata["language"]["tokenizer"] == "tiny-stateful"
    assert bundle_metadata["language"]["vocab_size"] == 10
    assert bundle_metadata["language"]["max_context_length"] == 8
    assert bundle_metadata["language"]["embedded_tokenizer"] is True
    assert bundle_metadata["language"]["function_map"] == {"main": ["main"]}
    assert bundle_metadata["source"] == {
        "model_definition": "mlx",
        "hf_model_id": "tiny-stateful",
    }
    assert bundle_metadata["compression"] is None
    assert bundle_metadata["compilation"]["targets"] == []
    assert converted.bundle_path == bundle_path
    assert converted.asset_path == asset_path
    assert converted.bundle_metadata == bundle_metadata
    assert converted.lowered.entrypoint_names == ["main"]
    assert converted.metadata["mlx_lm_stateful"]["state_count"] == 2
    assert converted.metadata["mlx_lm_stateful"]["key_cache_name"] == "keyCache"
    assert converted.metadata["mlx_lm_stateful"]["value_cache_name"] == "valueCache"
    assert "MutableBuffers.buffer_mutation" in str(converted.program)


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
    assert exported_convert_mlx_lm_stateful is convert_mlx_lm_stateful
    assert callable(exported_convert_mlx_to_coreai)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2coreai.recipe import Bundle, export
from recipes import qwen3
from recipes._mlx_lm.source import build_mlx_lm_inputs, load_mlx_lm_model


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


def test_recipe_exports_stateful_asset_and_tokenizer(tmp_path):
    mx = pytest.importorskip("mlx.core")

    class TinyModel:
        eval_called = False
        args = SimpleNamespace(model_type="qwen3", num_key_value_heads=1, head_dim=1)
        layers = [SimpleNamespace(self_attn=SimpleNamespace(n_kv_heads=1))]

        def eval(self):
            self.eval_called = True

        def __call__(self, input_ids, *, cache):
            values = mx.reshape(input_ids.astype(mx.float32), (1, 1, input_ids.shape[1], 1))
            cache[0].update_and_fetch(values, values)
            return input_ids.astype(mx.float32)

    model = TinyModel()
    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return model, FakeTokenizer()

    plan = qwen3.build("tiny-stateful", max_context_length=8, revision="abc", load_fn=load)
    bundle = export(plan, tmp_path / "tiny-stateful")
    assert model.eval_called
    assert calls == [("tiny-stateful", {"lazy": False, "revision": "abc"})]
    assert (bundle.path / "main.aimodel" / "main.mlirb").is_file()
    assert bundle.resource("tokenizer/tokenizer.json").is_file()
    assert Bundle.open(bundle.path).manifest == bundle.manifest
    assert bundle.manifest["recipe"] == "qwen3"
    assert bundle.metadata["source"] == "tiny-stateful"
    assert bundle.metadata["revision"] == "abc"
    assert bundle.metadata["dynamic_sequence"] and bundle.metadata["dynamic_state"]
    main = bundle.manifest["components"]["main"]
    assert main["asset"] == "main.aimodel"
    assert main["optimized"]
    assert list(main["outputs"]) == ["logits"]
    assert [state["name"] for state in main["states"]] == ["keyCache", "valueCache"]

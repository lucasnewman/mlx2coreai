from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from scripts import benchmark_aimodel_sampling as benchmark


@pytest.mark.parametrize("tokenizer,prompt", [(None, None), (object(), None), (None, "hello")])
def test_synthetic_context_respects_fill_token(tokenizer, prompt):
    tokens = benchmark.context_token_ids(4, tokenizer=tokenizer, prompt=prompt, fill_token_id=42)
    np.testing.assert_array_equal(tokens, [42] * 4)
    assert tokens.dtype == np.int32


def test_prompt_context_uses_tokenizer(monkeypatch):
    calls = []

    def build_inputs(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(input_ids=np.array([[5, 6]], dtype=np.int32))

    monkeypatch.setattr(benchmark, "build_mlx_lm_inputs", build_inputs)
    tokenizer = object()
    tokens = benchmark.context_token_ids(2, tokenizer=tokenizer, prompt="hello", fill_token_id=42)
    assert tokens.tolist() == [5, 6]
    assert calls == [dict(tokenizer=tokenizer, prompt="hello", sequence_length=2, batch_size=1)]


class FakeNDArray:
    def __init__(self, data):
        self.data = np.asarray(data)

    def numpy(self):
        return self.data


@pytest.mark.parametrize("dynamic_dim", [None, -1])
def test_state_allocation_resolves_dynamic_dims_and_preserves_bf16(dynamic_dim):
    function = SimpleNamespace(desc=SimpleNamespace(
        state_names=["cache"],
        state_descriptor=lambda name: SimpleNamespace(shape=(2, 1, dynamic_dim, 4), dtype="bfloat16"),
    ))
    state = benchmark.allocate_state(function, FakeNDArray, state_capacity=7)
    assert state["cache"].data.shape == (2, 1, 7, 4)
    assert state["cache"].data.dtype == ml_dtypes.bfloat16
    assert not np.any(state["cache"].data)


@pytest.mark.parametrize("grow", [False, True])
@pytest.mark.parametrize("os_runtime", [False, True])
@pytest.mark.parametrize("layout", ["query", "context"])
@pytest.mark.parametrize("history_state", [None, "recurrentState", "convState"])
def test_python_benchmark_state_positions_and_artifacts(monkeypatch, tmp_path, grow, os_runtime, layout, history_state):
    import coreai.authoring
    import coreai.runtime

    calls = []
    options = []
    recurrent = history_state is not None

    class FakeFunction:
        desc = SimpleNamespace(
            output_names=["logits"], state_names=[history_state or "cache"],
            state_descriptor=lambda name: SimpleNamespace(shape=(1, -1, 1), dtype="float32"),
        )

        async def __call__(self, *, inputs, state):
            calls.append((inputs, state))
            for value in state.values():
                value.data += 1
            token_ids = inputs["input_ids"].data
            logits = np.zeros((*token_ids.shape, 16), dtype=np.float16)
            logits[0, -1, int(token_ids[0, -1]) + 1] = 10
            return {"logits": FakeNDArray(logits)}

    class FakeAsset:
        def executable(self, specialization_options):
            options.append(specialization_options)
            return self

        async def __aenter__(self):
            return SimpleNamespace(load_function=lambda name: FakeFunction())

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(coreai.authoring, "AIModelAsset", SimpleNamespace(load=lambda path: FakeAsset()))
    monkeypatch.setattr(coreai.runtime, "NDArray", FakeNDArray)
    monkeypatch.setattr(coreai.runtime, "ComputeUnitKind", SimpleNamespace(gpu=lambda: "GPU"), raising=False)
    monkeypatch.setattr(coreai.runtime, "SpecializationOptions", SimpleNamespace(
        is_supported=lambda: os_runtime,
        from_preferred_compute_unit_kind=lambda kind: kind,
    ))
    monkeypatch.setattr(benchmark, "load_tokenizer", lambda *args, **kwargs: object())
    args = benchmark.parse_args([
        str(tmp_path / "model.aimodel"), "--runtime-backend", "python", "--contexts", "2,4",
        "--steps", "3", "--warmup", "1", "--fill-token-id", "1",
        "--position-ids-layout", layout,
        "--logits-dir", str(tmp_path / "logits"),
        *(["--grow-context"] if grow else []),
    ])
    if recurrent and not grow:
        with pytest.raises(ValueError, match="require --grow-context"):
            asyncio.run(benchmark.benchmark(args))
        return
    rows = asyncio.run(benchmark.benchmark(args))
    assert options == ["GPU" if os_runtime else None]
    assert len(calls) == 10
    assert calls[0][1] is not calls[5][1]
    for index, context in enumerate([2, 4]):
        interval = calls[index * 5:(index + 1) * 5]
        state = interval[0][1]
        assert all(interval[i][1] is state for i in (0, 2, 3, 4))
        assert (interval[1][1] is state) == (not recurrent)
        assert all(np.all(value.data == (4 if recurrent else 5)) for value in state.values())
        assert state[history_state or "cache"].data.shape == (1, context + (4 if grow else 1), 1)
        assert interval[0][0]["input_ids"].data.tolist() == [[1] * context]
        lengths = [context, context + 1, context + 1, context + (2 if grow else 1), context + (3 if grow else 1)]
        for step, ((inputs, _), length) in enumerate(zip(interval, lengths, strict=True)):
            expected = [length - 1] if layout == "query" and step > 0 else list(range(length))
            assert inputs["position_ids"].data.tolist() == [expected]
        row = rows[index]
        expected_tokens = [3, 4, 5] if recurrent else [4, 5, 6]
        assert row.sampled_tokens == expected_tokens
        assert row.position_start == context
        assert row.position_end == context + (3 if grow else 0)
        logits = np.fromfile(args.logits_dir / f"context_{context}.f32", dtype="<f4")
        assert logits.shape == (16,)
        assert np.argmax(logits) == expected_tokens[-1]
    json_path = tmp_path / "results.json"
    benchmark.write_json(json_path, rows, args)
    payload = json.loads(json_path.read_text())
    assert payload["runtime_backend"] == "python"
    assert payload["fill_token_id"] == 1
    assert payload["position_ids_layout"] == layout
    assert payload["results"][0]["sampled_tokens"] == expected_tokens


def test_swift_backend_forwards_parity_artifacts(monkeypatch, tmp_path):
    args = benchmark.parse_args([
        "model.aimodel", "--runtime-backend", "swift", "--json-output", str(tmp_path / "out.json"),
        "--logits-dir", str(tmp_path / "logits"), "--grow-context",
        "--position-ids-layout", "context",
    ])
    commands = []
    monkeypatch.setattr(benchmark, "ensure_swift_backend", lambda: (Path("runner"), {}))

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    assert benchmark.unsupported_swift_backend_options(args) == []
    assert benchmark.run_swift_backend(args) == 0
    command = commands[0]
    assert command[command.index("--json-output") + 1] == str(args.json_output)
    assert command[command.index("--logits-dir") + 1] == str(args.logits_dir)
    assert "--grow-context" in command
    assert command[command.index("--position-ids-layout") + 1] == "context"


def test_backend_selection_with_artifacts_and_tokenizer_options(monkeypatch):
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(benchmark, "swift_backend_sdk_path", lambda: Path("SDK"))
    args = benchmark.parse_args(["model.aimodel", "--json-output", "out.json"])
    assert benchmark.should_use_swift_backend(args)
    args.runtime_backend = "python"
    assert not benchmark.should_use_swift_backend(args)
    args.runtime_backend = "auto"
    args.prompt = "hello"
    assert not benchmark.should_use_swift_backend(args)

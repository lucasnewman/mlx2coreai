from __future__ import annotations

import sys

import pytest

from scripts import validate_aimodel_mlx as validator


def test_validator_forwards_precision_capacity_and_error_limits(monkeypatch):
    seen = []

    async def validate(args):
        seen.append(args)

    monkeypatch.setattr(validator, "validate", validate)
    monkeypatch.setattr(sys, "argv", [
        "validate", "asset", "--model", "weights", "--compute-precision", "fp32",
        "--state-capacity", "2048", "--max-abs-error", "0.01", "--max-relative-l2", "0.01",
    ])
    validator.main()
    assert len(seen) == 1
    assert seen[0].compute_precision == "fp32"
    assert seen[0].state_capacity == 2048
    assert seen[0].max_abs_error == seen[0].max_relative_l2 == 0.01


@pytest.mark.parametrize("option", ["--state-capacity", "--max-abs-error"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_validator_rejects_nonpositive_limits(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["validate", "asset", "--model", "weights", option, value])
    with pytest.raises(SystemExit) as error:
        validator.main()
    assert error.value.code == 2

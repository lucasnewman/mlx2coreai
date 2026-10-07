from __future__ import annotations

import json
import importlib
from pathlib import Path

import pytest

from mlx2coreai.op_coverage import write_coverage_report


@pytest.mark.parametrize('module_name,args', [
    ('mlx2coreai.cli', ['ops']),
    ('mlx2coreai.op_coverage', []),
])
@pytest.mark.parametrize('custom_paths', [False, True])
def test_coverage_cli_output_paths(monkeypatch, tmp_path, module_name, args, custom_paths):
    module = importlib.import_module(module_name)
    captured = {}

    def write_report(**kwargs):
        captured.update(kwargs)
        return {'model_zoo': None}

    monkeypatch.setattr(module, 'write_coverage_report', write_report)
    markdown = tmp_path / 'report.md' if custom_paths else Path('dev/op_coverage.md')
    structured = tmp_path / 'report.json' if custom_paths else Path('dev/op_coverage.json')
    if custom_paths:
        args = [*args, '--output', str(markdown), '--json-output', str(structured)]
    assert module.main(args) == 0
    assert captured['output_path'] == markdown
    assert captured['json_output_path'] == structured


def test_op_coverage_report_writes_markdown_and_json(tmp_path: Path) -> None:
    markdown_path = tmp_path / "op_coverage.md"
    json_path = tmp_path / "op_coverage.json"
    payload = write_coverage_report(
        output_path=markdown_path,
        json_output_path=json_path,
        validate_assets=True,
    )
    assert markdown_path.exists()
    assert json_path.exists()
    assert payload["model_zoo"]["unique_source_ops"] > 0
    assert payload["model_zoo"]["unique_source_ops"] == payload["registry"]["supported_source_op_names"]
    assert payload["model_zoo"]["asset_validation_passed"] is True
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "mlx2coreai.op_coverage.v1"
    assert "## Exercised Ops" in markdown_path.read_text(encoding="utf-8")

from __future__ import annotations

import json
from pathlib import Path

from mlx2coreai.op_coverage import write_coverage_report


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

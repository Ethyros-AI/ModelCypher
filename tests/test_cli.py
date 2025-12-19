from __future__ import annotations

from typer.testing import CliRunner

from modelcypher.cli.app import app


runner = CliRunner()


def test_inventory_command():
    result = runner.invoke(app, ["inventory", "--output", "json"])
    assert result.exit_code == 0
    assert "models" in result.stdout


def test_dataset_validate_command(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{\"text\": \"hello world\"}\n", encoding="utf-8")
    result = runner.invoke(app, ["dataset", "validate", str(dataset), "--output", "json"])
    assert result.exit_code == 0
    assert "totalExamples" in result.stdout

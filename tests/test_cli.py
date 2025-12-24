# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ModelCypher is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ModelCypher.  If not, see <https://www.gnu.org/licenses/>.

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


def test_explain_command():
    result = runner.invoke(app, ["explain", "inventory", "--output", "json"])
    assert result.exit_code == 0
    assert "command" in result.stdout
    assert "affectedResources" in result.stdout


def test_geometry_validate_command():
    result = runner.invoke(app, ["geometry", "validate", "--output", "json"])
    assert result.exit_code == 0
    # JSON output uses camelCase
    assert "gromovWasserstein" in result.stdout


def test_estimate_train_command(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{\"text\": \"train data\"}\n", encoding="utf-8")
    result = runner.invoke(app, ["estimate", "train", "--model", "test-model", "--dataset", str(dataset), "--output", "json"])
    assert result.exit_code == 0
    assert "willFit" in result.stdout

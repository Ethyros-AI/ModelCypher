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

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


def test_inventory_command():
    result = runner.invoke(app, ["inventory", "--output", "json"])
    assert result.exit_code == 0
    assert "models" in result.stdout


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
    dataset.write_text('{"text": "train data"}\n', encoding="utf-8")
    out_path = tmp_path / "output"

    # Mock the training service since it requires calibrated resource profiles
    mock_service = MagicMock()
    mock_service.preflight.return_value = {
        "canProceed": True,
        "predictedBatchSize": 4,
        "estimatedVRAMUsageBytes": 4 * 1024**3,  # 4 GB
        "availableVRAMBytes": 16 * 1024**3,  # 16 GB
    }

    with patch("modelcypher.cli.app.get_training_service", return_value=mock_service):
        result = runner.invoke(
            app,
            [
                "estimate",
                "train",
                "--model",
                "test-model",
                "--dataset",
                str(dataset),
                "--out",
                str(out_path),
                "--batch-size",
                "4",
                "--sequence-length",
                "512",
                "--learning-rate",
                "0.0001",
                "--epochs",
                "1",
                "--grad-accum",
                "1",
                "--warmup-steps",
                "10",
                "--weight-decay",
                "0.01",
                "--no-gradient-checkpointing",
                "--no-mixed-precision",
                "--compute-precision",
                "float32",
                "--optimizer-type",
                "adamw",
                "--seed",
                "42",
                "--deterministic",
                "--output",
                "json",
            ],
        )
    assert result.exit_code == 0
    assert "willFit" in result.stdout

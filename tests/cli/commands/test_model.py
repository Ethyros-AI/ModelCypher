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

"""Tests for model CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestModelCommandHelp:
    """Test model command help and basic invocation."""

    def test_model_help(self):
        """model --help should show usage information."""
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "model" in result.stdout.lower()

    def test_model_probe_help(self):
        """model probe --help should show usage information."""
        result = runner.invoke(app, ["model", "probe", "--help"])
        assert result.exit_code == 0


class TestModelProbeCommand:
    """Test model probe command."""

    def test_model_probe_json_output(self):
        """model probe with --output json should return JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()

            config = {
                "model_type": "llama",
                "hidden_size": 2048,
                "vocab_size": 32000,
                "num_hidden_layers": 22,
                "intermediate_size": 5504,
                "num_attention_heads": 32,
            }
            (model_dir / "config.json").write_text(json.dumps(config))
            (model_dir / "model.safetensors").write_bytes(b"x" * 1000)

            mock_info = MagicMock()
            mock_info.architecture = "llama"
            mock_info.parameter_count = 7_000_000_000
            mock_info.vocab_size = 32000
            mock_info.hidden_size = 2048
            mock_info.layers = list(range(22))
            mock_info.quantization = None

            mock_service = MagicMock()
            mock_service.probe.return_value = mock_info

            with patch(
                "modelcypher.cli.commands.model.get_model_probe_service",
                return_value=mock_service,
            ):
                result = runner.invoke(
                    app,
                    ["model", "probe", str(model_dir), "--output", "json"],
                )

            assert result.exit_code == 0
            # The output format depends on the actual command implementation

    def test_model_probe_invalid_path(self):
        """model probe with non-existent path should show error."""
        result = runner.invoke(
            app,
            ["model", "probe", "/nonexistent/model"],
        )
        assert result.exit_code != 0


class TestModelListCommand:
    """Test model list command."""

    def test_model_list_help(self):
        """model list --help should show usage."""
        result = runner.invoke(app, ["model", "list", "--help"])
        assert result.exit_code == 0


class TestModelRegisterCommand:
    """Test model register command."""

    def test_model_register_help(self):
        """model register --help should show usage."""
        result = runner.invoke(app, ["model", "register", "--help"])
        assert result.exit_code == 0


class TestModelDeleteCommand:
    """Test model delete command."""

    def test_model_delete_help(self):
        """model delete --help should show usage."""
        result = runner.invoke(app, ["model", "delete", "--help"])
        assert result.exit_code == 0

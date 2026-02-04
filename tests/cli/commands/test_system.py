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

"""Tests for system CLI commands."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestSystemStatusCommand:
    """Test system status command."""

    def test_system_status_help(self):
        """system status --help should show usage."""
        result = runner.invoke(app, ["system", "status", "--help"])
        assert result.exit_code == 0
        assert "--require-backend mlx" in result.stdout

    def test_system_status_json_output(self):
        """system status --output json should return JSON."""
        mock_status = {
            "platform": "darwin",
            "architecture": "arm64",
            "metalAvailable": True,
            "mlxVersion": "0.1.0",
        }
        mock_service = MagicMock()
        mock_service.status.return_value = mock_status

        with patch(
            "modelcypher.cli.commands.system.get_system_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["system", "status", "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert "platform" in payload
        assert "metalAvailable" in payload

    def test_system_status_require_metal_fails(self):
        """system status --require-backend mlx should fail if metal unavailable."""
        mock_status = {
            "platform": "linux",
            "architecture": "x86_64",
            "metalAvailable": False,
        }
        mock_service = MagicMock()
        mock_service.status.return_value = mock_status

        with patch(
            "modelcypher.cli.commands.system.get_system_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["system", "status", "--require-backend mlx"])

        assert result.exit_code == 3  # Special exit code for missing metal

    def test_system_status_require_metal_passes(self):
        """system status --require-backend mlx should pass if metal available."""
        mock_status = {
            "platform": "darwin",
            "architecture": "arm64",
            "metalAvailable": True,
        }
        mock_service = MagicMock()
        mock_service.status.return_value = mock_status

        with patch(
            "modelcypher.cli.commands.system.get_system_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["system", "status", "--require-backend mlx"])

        assert result.exit_code == 0


class TestSystemProbeCommand:
    """Test system probe command."""

    def test_system_probe_help(self):
        """system probe --help should show usage."""
        result = runner.invoke(app, ["system", "probe", "--help"])
        assert result.exit_code == 0

    def test_system_probe_gpu(self):
        """system probe gpu should return GPU info."""
        mock_probe_result = {
            "target": "gpu",
            "available": True,
            "name": "Apple M1",
            "memory": 16_000_000_000,
        }
        mock_service = MagicMock()
        mock_service.probe.return_value = mock_probe_result

        with patch(
            "modelcypher.cli.commands.system.get_system_service",
            return_value=mock_service,
        ):
            result = runner.invoke(app, ["system", "probe", "gpu", "--output", "json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["target"] == "gpu"

    def test_system_probe_missing_target(self):
        """system probe without target should show error."""
        result = runner.invoke(app, ["system", "probe"])
        assert result.exit_code != 0


class TestSystemBenchmarkCommand:
    """Test system benchmark commands."""

    def test_system_benchmark_help(self):
        """system benchmark --help should show usage."""
        result = runner.invoke(app, ["system", "benchmark", "--help"])
        assert result.exit_code == 0

    def test_system_benchmark_cache_help(self):
        """system benchmark cache --help should show usage."""
        result = runner.invoke(app, ["system", "benchmark", "cache", "--help"])
        assert result.exit_code == 0

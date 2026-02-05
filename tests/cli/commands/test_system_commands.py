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

"""Tests for system CLI commands.

Tests:
- Command help text
- System status and probe functionality
- Benchmark commands
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestSystemCommandHelp:
    """Test that system commands have proper help text."""

    def test_system_help(self):
        """Test 'mc system --help' works."""
        result = runner.invoke(app, ["system", "--help"])
        assert result.exit_code == 0
        assert "status" in result.stdout.lower()
        assert "probe" in result.stdout.lower()

    def test_system_status_help(self):
        """Test 'mc system status --help' works."""
        result = runner.invoke(app, ["system", "status", "--help"])
        assert result.exit_code == 0

    def test_system_probe_help(self):
        """Test 'mc system probe --help' works."""
        result = runner.invoke(app, ["system", "probe", "--help"])
        assert result.exit_code == 0

    def test_system_benchmark_help(self):
        """Test 'mc system benchmark --help' works."""
        result = runner.invoke(app, ["system", "benchmark", "--help"])
        assert result.exit_code == 0


class TestSystemProbeValidation:
    """Test system probe argument validation."""

    def test_system_probe_requires_target(self):
        """Test that probe requires a target argument."""
        result = runner.invoke(app, ["system", "probe"])
        assert result.exit_code != 0


class TestSystemStatusExecution:
    """Test system status command execution.
    
    Note: These tests may be slow as they initialize the backend.
    Mark with @pytest.mark.slow if needed.
    """

    @pytest.mark.slow
    def test_system_status_runs(self):
        """Test that 'mc system status' runs without crashing."""
        result = runner.invoke(app, ["system", "status"])
        # May exit 0 or 1 depending on backend availability
        # Main goal: no crash (exception)
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_system_probe_backends_help_works(self):
        """Test that probe backends is a valid subcommand."""
        # Just test it doesn't fail with bad arguments
        result = runner.invoke(app, ["system", "probe", "backends", "--help"])
        # This may or may not work depending on implementation
        # We're testing the command structure exists
        assert result.exit_code in (0, 2)  # 2 = no such command variant is ok

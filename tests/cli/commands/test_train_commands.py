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

"""Tests for train CLI commands.

Tests:
- Command help text
- Required argument validation
- Training workflow commands
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestTrainCommandHelp:
    """Test that train commands have proper help text."""

    def test_train_help(self):
        """Test 'mc train --help' works."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0

    def test_train_command_help(self):
        """Test 'mc train [default command] --help' shows options."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.stdout or "--model" in result.stdout or "status" in result.stdout.lower()

    def test_train_status_help(self):
        """Test 'mc train status --help' works."""
        result = runner.invoke(app, ["train", "status", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.stdout
        assert "--model" in result.stdout

    def test_train_run_help(self):
        """Test 'mc train run --help' works."""
        result = runner.invoke(app, ["train", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--data" in result.stdout

    def test_train_merge_help(self):
        """Test 'mc train merge --help' works."""
        result = runner.invoke(app, ["train", "merge", "--help"])
        assert result.exit_code == 0
        assert "--agent" in result.stdout
        assert "--model" in result.stdout

    def test_train_export_help(self):
        """Test 'mc train export --help' works."""
        result = runner.invoke(app, ["train", "export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout


class TestTrainCommandValidation:
    """Test train command argument validation."""

    def test_train_status_requires_agent(self):
        """Test that --agent is required for status."""
        result = runner.invoke(app, ["train", "status", "--model", "/some/path"])
        assert result.exit_code != 0

    def test_train_status_requires_model(self):
        """Test that --model is required for status."""
        result = runner.invoke(app, ["train", "status", "--agent", "test-agent"])
        assert result.exit_code != 0

    def test_train_merge_requires_agent(self):
        """Test that --agent is required for merge."""
        result = runner.invoke(app, ["train", "merge", "--model", "/some/path"])
        assert result.exit_code != 0

    def test_train_merge_requires_model(self):
        """Test that --model is required for merge."""
        result = runner.invoke(app, ["train", "merge", "--agent", "test-agent"])
        assert result.exit_code != 0

    def test_train_export_requires_output(self):
        """Test that --output is required for export."""
        result = runner.invoke(
            app, ["train", "export", "--agent", "test", "--model", "/path"]
        )
        assert result.exit_code != 0

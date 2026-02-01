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


class TestModelListCommand:
    """Test model list command."""

    def test_model_list_help(self):
        """model list --help should show usage."""
        result = runner.invoke(app, ["model", "list", "--help"])
        assert result.exit_code == 0


class TestModelDeleteCommand:
    """Test model delete command."""

    def test_model_delete_help(self):
        """model delete --help should show usage."""
        result = runner.invoke(app, ["model", "delete", "--help"])
        assert result.exit_code == 0

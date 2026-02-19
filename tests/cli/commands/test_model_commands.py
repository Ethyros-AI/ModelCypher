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

"""Tests for model CLI commands.

Tests:
- Command help text
- Model registry operations
- Input validation
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestModelCommandHelp:
    """Test that model commands have proper help text."""

    def test_model_help(self):
        """Test 'mc model --help' works."""
        result = runner.invoke(app, ["model", "--help"])
        assert result.exit_code == 0
        assert "list" in result.stdout.lower()
        assert "add" in result.stdout.lower()
        assert "info" in result.stdout.lower()
        assert "capacity" in result.stdout.lower()

    def test_model_list_help(self):
        """Test 'mc model list --help' works."""
        result = runner.invoke(app, ["model", "list", "--help"])
        assert result.exit_code == 0

    def test_model_add_help(self):
        """Test 'mc model add --help' works."""
        result = runner.invoke(app, ["model", "add", "--help"])
        assert result.exit_code == 0
        assert "--alias" in result.stdout or "-a" in result.stdout

    def test_model_info_help(self):
        """Test 'mc model info --help' works."""
        result = runner.invoke(app, ["model", "info", "--help"])
        assert result.exit_code == 0

    def test_model_search_help(self):
        """Test 'mc model search --help' works."""
        result = runner.invoke(app, ["model", "search", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.stdout or "-n" in result.stdout

    def test_model_quantize_help(self):
        """Test 'mc model quantize --help' works."""
        result = runner.invoke(app, ["model", "quantize", "--help"])
        assert result.exit_code == 0
        assert "--bits" in result.stdout

    def test_model_capacity_help(self):
        """Test 'mc model capacity --help' works."""
        result = runner.invoke(app, ["model", "capacity", "--help"])
        assert result.exit_code == 0
        assert "--top" in result.stdout


class TestModelAddValidation:
    """Test model add argument validation."""

    def test_model_add_requires_path(self):
        """Test that path argument is required."""
        result = runner.invoke(app, ["model", "add"])
        assert result.exit_code != 0

    def test_model_add_validates_path(self):
        """Test that invalid paths are rejected."""
        result = runner.invoke(app, ["model", "add", "/nonexistent/model/path"])
        assert result.exit_code != 0


class TestModelInfoValidation:
    """Test model info argument validation."""

    def test_model_info_requires_path(self):
        """Test that model path is required."""
        result = runner.invoke(app, ["model", "info"])
        assert result.exit_code != 0

    def test_model_info_validates_path(self):
        """Test that invalid model paths are rejected."""
        result = runner.invoke(app, ["model", "info", "/nonexistent/path"])
        assert result.exit_code != 0


class TestModelSearchValidation:
    """Test model search argument validation."""

    def test_model_search_requires_query(self):
        """Test that search query is required."""
        result = runner.invoke(app, ["model", "search"])
        assert result.exit_code != 0


class TestModelQuantizeValidation:
    """Test model quantize argument validation."""

    def test_model_quantize_requires_input(self):
        """Test that input path is required."""
        result = runner.invoke(app, ["model", "quantize"])
        assert result.exit_code != 0

    def test_model_quantize_requires_output(self):
        """Test that output path is required."""
        result = runner.invoke(app, ["model", "quantize", "/some/model"])
        assert result.exit_code != 0

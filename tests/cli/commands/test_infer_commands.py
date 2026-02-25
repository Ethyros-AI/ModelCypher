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

"""Tests for infer CLI commands.

Tests:
- Command help text
- Argument validation
- Error handling for invalid inputs
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestInferCommandHelp:
    """Test that infer commands have proper help text."""

    def test_infer_help(self):
        """Test 'mc infer --help' works."""
        result = runner.invoke(app, ["infer", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout.lower()
        assert "suite" in result.stdout.lower()

    def test_infer_run_help(self):
        """Test 'mc infer run --help' works."""
        result = runner.invoke(app, ["infer", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--prompt" in result.stdout
        assert "--max-tokens" in result.stdout

    def test_infer_suite_help(self):
        """Test 'mc infer suite --help' works."""
        result = runner.invoke(app, ["infer", "suite", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--suite" in result.stdout


class TestInferRunValidation:
    """Test infer run argument validation."""

    def test_infer_run_requires_model(self):
        """Test that --model is required."""
        result = runner.invoke(app, ["infer", "run", "--prompt", "hello"])
        assert result.exit_code != 0
        # Should fail due to missing required --model

    def test_infer_run_validates_model_path(self):
        """Test that invalid model paths are rejected."""
        result = runner.invoke(
            app,
            ["infer", "run", "--model", "/nonexistent/path", "--prompt", "hello"],
        )
        assert result.exit_code != 0
        # Should fail due to invalid model path

    def test_infer_run_requires_prompt_source(self):
        """Test that at least one prompt source is required."""
        result = runner.invoke(
            app,
            ["infer", "run", "--model", "/some/path"],
        )
        # Should fail - no prompt provided
        assert result.exit_code != 0

    def test_infer_run_passes_max_tokens_to_engine(self, monkeypatch, tmp_path):
        """Test that --max-tokens is forwarded to inference engine."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        captured: dict[str, object] = {}

        class _StubEngine:
            def run(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    model=str(model_dir),
                    prompt=kwargs["prompt"],
                    response="ok",
                    token_count=1,
                    tokens_per_second=10.0,
                    time_to_first_token=None,
                    total_duration=0.1,
                    stop_reason="length",
                    adapter=kwargs.get("adapter"),
                    security=None,
                )

        monkeypatch.setattr(
            "modelcypher.cli.commands.infer.get_inference_engine",
            lambda: _StubEngine(),
        )

        result = runner.invoke(
            app,
            [
                "--output", "json",
                "infer",
                "run",
                "--model", str(model_dir),
                "--prompt", "hello",
                "--max-tokens", "17",
            ],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["stopReason"] == "length"
        assert captured["max_tokens"] == 17


class TestInferSuiteValidation:
    """Test infer suite argument validation."""

    def test_infer_suite_requires_model(self):
        """Test that --model is required."""
        result = runner.invoke(app, ["infer", "suite", "--suite", "test.txt"])
        assert result.exit_code != 0

    def test_infer_suite_requires_suite_file(self):
        """Test that --suite is required."""
        result = runner.invoke(app, ["infer", "suite", "--model", "/some/path"])
        assert result.exit_code != 0

    def test_infer_suite_validates_suite_file(self):
        """Test that invalid suite file paths are rejected."""
        result = runner.invoke(
            app,
            ["infer", "suite", "--model", "/some/path", "--suite", "/nonexistent.txt"],
        )
        assert result.exit_code != 0

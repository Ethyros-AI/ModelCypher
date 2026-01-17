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

"""Tests for infer CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestInferRunCommand:
    """Test infer run command."""

    def test_infer_run_help(self):
        """infer run --help should show usage information."""
        result = runner.invoke(app, ["infer", "run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--prompt" in result.stdout
        assert "--adapter" in result.stdout
        assert "--security-scan" in result.stdout

    def test_infer_run_missing_model_error(self):
        """infer run without --model should show error."""
        result = runner.invoke(
            app,
            ["infer", "run", "--prompt", "Hello"],
        )
        assert result.exit_code != 0

    def test_infer_run_missing_prompt_error(self):
        """infer run without --prompt should show error."""
        result = runner.invoke(
            app,
            ["infer", "run", "--model", "/tmp/model"],
        )
        assert result.exit_code != 0

    def test_infer_run_invalid_model_path(self):
        """infer run with non-existent model should show error."""
        result = runner.invoke(
            app,
            [
                "infer",
                "run",
                "--model", "/nonexistent/model",
                "--prompt", "Hello",
            ],
        )
        assert result.exit_code != 0

    def test_infer_run_json_output(self):
        """infer run with --output json should return JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "llama"}')
            (model_dir / "model.safetensors").write_bytes(b"x" * 100)

            mock_result = MagicMock()
            mock_result.model = str(model_dir)
            mock_result.prompt = "Hello"
            mock_result.response = "Hello! How can I help you?"
            mock_result.token_count = 10
            mock_result.tokens_per_second = 50.0
            mock_result.time_to_first_token = 0.1
            mock_result.total_duration = 0.2
            mock_result.stop_reason = "eos"
            mock_result.adapter = None
            mock_result.security = None

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result

            with patch(
                "modelcypher.cli.commands.infer.get_inference_engine",
                return_value=mock_engine,
            ):
                result = runner.invoke(
                    app,
                    [
                        "infer",
                        "run",
                        "--model", str(model_dir),
                        "--prompt", "Hello",
                        "--output", "json",
                    ],
                )

            assert result.exit_code == 0
            payload = json.loads(result.stdout)
            assert "model" in payload
            assert "response" in payload
            assert "tokenCount" in payload

    def test_infer_run_with_security_scan(self):
        """infer run with --security-scan should include security metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "llama"}')
            (model_dir / "model.safetensors").write_bytes(b"x" * 100)

            mock_security = MagicMock()
            mock_security.anomaly_count = 0
            mock_security.max_anomaly_score = 0.1
            mock_security.avg_delta = 0.05
            mock_security.disagreement_rate = 0.02

            mock_result = MagicMock()
            mock_result.model = str(model_dir)
            mock_result.prompt = "Hello"
            mock_result.response = "Hello!"
            mock_result.token_count = 5
            mock_result.tokens_per_second = 50.0
            mock_result.time_to_first_token = 0.1
            mock_result.total_duration = 0.1
            mock_result.stop_reason = "eos"
            mock_result.adapter = None
            mock_result.security = mock_security

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result

            with patch(
                "modelcypher.cli.commands.infer.get_inference_engine",
                return_value=mock_engine,
            ):
                result = runner.invoke(
                    app,
                    [
                        "infer",
                        "run",
                        "--model", str(model_dir),
                        "--prompt", "Hello",
                        "--security-scan",
                        "--output", "json",
                    ],
                )

            assert result.exit_code == 0
            payload = json.loads(result.stdout)
            assert "security" in payload
            assert "anomalyCount" in payload["security"]

    def test_infer_run_with_adapter(self):
        """infer run with --adapter should pass adapter path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            adapter_dir = Path(tmpdir) / "adapter"
            model_dir.mkdir()
            adapter_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "llama"}')
            (model_dir / "model.safetensors").write_bytes(b"x" * 100)
            (adapter_dir / "adapters.safetensors").write_bytes(b"x" * 100)

            mock_result = MagicMock()
            mock_result.model = str(model_dir)
            mock_result.prompt = "Hello"
            mock_result.response = "Hello!"
            mock_result.token_count = 5
            mock_result.tokens_per_second = 50.0
            mock_result.time_to_first_token = 0.1
            mock_result.total_duration = 0.1
            mock_result.stop_reason = "eos"
            mock_result.adapter = str(adapter_dir)
            mock_result.security = None

            mock_engine = MagicMock()
            mock_engine.run.return_value = mock_result

            with patch(
                "modelcypher.cli.commands.infer.get_inference_engine",
                return_value=mock_engine,
            ):
                result = runner.invoke(
                    app,
                    [
                        "infer",
                        "run",
                        "--model", str(model_dir),
                        "--prompt", "Hello",
                        "--adapter", str(adapter_dir),
                        "--output", "json",
                    ],
                )

            assert result.exit_code == 0
            payload = json.loads(result.stdout)
            assert payload["adapter"] == str(adapter_dir)


class TestInferSuiteCommand:
    """Test infer suite command."""

    def test_infer_suite_help(self):
        """infer suite --help should show usage information."""
        result = runner.invoke(app, ["infer", "suite", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout
        assert "--suite" in result.stdout

    def test_infer_suite_missing_model_error(self):
        """infer suite without --model should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_file = Path(tmpdir) / "suite.txt"
            suite_file.write_text("prompt 1\nprompt 2\n")

            result = runner.invoke(
                app,
                ["infer", "suite", "--suite", str(suite_file)],
            )
            assert result.exit_code != 0

    def test_infer_suite_missing_suite_error(self):
        """infer suite without --suite should show error."""
        result = runner.invoke(
            app,
            ["infer", "suite", "--model", "/tmp/model"],
        )
        assert result.exit_code != 0

    def test_infer_suite_nonexistent_suite_file(self):
        """infer suite with non-existent suite file should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "infer",
                    "suite",
                    "--model", str(model_dir),
                    "--suite", "/nonexistent/suite.txt",
                ],
            )
            assert result.exit_code != 0

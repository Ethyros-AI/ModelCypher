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

"""Tests for multimodal CLI commands.

All geometric parameters are auto-derived - CLI exposes only essential inputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app


runner = CliRunner()


class TestMultimodalCLI:
    """Tests for multimodal CLI subcommands."""

    def test_multimodal_help(self) -> None:
        """Should show help for multimodal command."""
        result = runner.invoke(app, ["multimodal", "--help"])
        assert result.exit_code == 0
        assert "inject-image" in result.output
        # probe-bridge command removed - was diagnostic noise

    def test_inject_image_help(self) -> None:
        """Should show help for inject-image command."""
        result = runner.invoke(app, ["multimodal", "inject-image", "--help"])
        assert result.exit_code == 0
        # Essential parameters only
        assert "--model" in result.output
        assert "--image" in result.output
        assert "--prompt" in result.output
        # Geometry knobs should NOT exist
        assert "--scale" not in result.output
        assert "--temperature" not in result.output
        assert "--layer" not in result.output
        assert "--null-space" not in result.output

    def test_inject_image_requires_model(self) -> None:
        """Should require --model option."""
        result = runner.invoke(app, ["multimodal", "inject-image", "--image", "test.jpg"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_inject_image_requires_image(self) -> None:
        """Should require --image option."""
        result = runner.invoke(app, ["multimodal", "inject-image", "--model", "/path/to/model"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    @patch("modelcypher.cli.commands.multimodal._run_visual_injection")
    @patch("modelcypher.cli.commands.multimodal.validate_model_path")
    @patch("modelcypher.cli.commands.multimodal.validate_file_exists")
    def test_inject_image_calls_pipeline(
        self,
        mock_validate_file: MagicMock,
        mock_validate_model: MagicMock,
        mock_injection: MagicMock,
    ) -> None:
        """Should call visual injection pipeline with essential args only."""
        mock_injection.return_value = {
            "response": "A beautiful forest scene",
            "nearest_tokens": [" forest", " trees", " nature"],
            "derived_scale": 12.5,  # Auto-derived
            "derived_temperature": 0.087,  # Auto-derived
            "injection_layer": 8,  # Auto-determined
            "token_count": 10,
        }

        result = runner.invoke(app, [
            "multimodal", "inject-image",
            "--model", "/path/to/model",
            "--image", "/path/to/image.jpg",
            "--prompt", "Describe the image",
            "--ai",
        ])

        # Check mock was called with essential args only
        mock_injection.assert_called_once()
        call_kwargs = mock_injection.call_args[1]
        assert call_kwargs["model_path"] == "/path/to/model"
        assert call_kwargs["image_path"] == "/path/to/image.jpg"
        assert call_kwargs["prompt"] == "Describe the image"
        # No scale, temperature, layer_idx args - all auto-derived


class TestMultimodalCLIOutput:
    """Tests for CLI output formatting."""

    @patch("modelcypher.cli.commands.multimodal._run_visual_injection")
    @patch("modelcypher.cli.commands.multimodal.validate_model_path")
    @patch("modelcypher.cli.commands.multimodal.validate_file_exists")
    def test_inject_image_json_output(
        self,
        mock_validate_file: MagicMock,
        mock_validate_model: MagicMock,
        mock_injection: MagicMock,
    ) -> None:
        """Should output JSON with auto-derived parameters."""
        mock_injection.return_value = {
            "response": "A forest with tall trees",
            "nearest_tokens": [" forest", " trees"],
            "derived_scale": 12.5,
            "derived_temperature": 0.087,
            "injection_layer": 8,
            "token_count": 8,
        }

        result = runner.invoke(app, [
            "multimodal", "inject-image",
            "--model", "/path/to/model",
            "--image", "/path/to/image.jpg",
            "--ai",
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.output)
        assert "response" in output
        assert "visualMemory" in output
        # Auto-derived values exposed in output
        assert "derivedScale" in output["visualMemory"]
        assert "derivedTemperature" in output["visualMemory"]
        assert "injectionLayer" in output["visualMemory"]

    @patch("modelcypher.cli.commands.multimodal._run_visual_injection")
    @patch("modelcypher.cli.commands.multimodal.validate_model_path")
    @patch("modelcypher.cli.commands.multimodal.validate_file_exists")
    def test_inject_image_text_output(
        self,
        mock_validate_file: MagicMock,
        mock_validate_model: MagicMock,
        mock_injection: MagicMock,
    ) -> None:
        """Should output formatted text with auto-derived info."""
        mock_injection.return_value = {
            "response": "A forest with tall trees",
            "nearest_tokens": [" forest", " trees"],
            "derived_scale": 12.5,
            "derived_temperature": 0.087,
            "injection_layer": 8,
            "token_count": 8,
        }

        result = runner.invoke(app, [
            "multimodal", "inject-image",
            "--model", "/path/to/model",
            "--image", "/path/to/image.jpg",
            "--output", "text",
        ])

        assert result.exit_code == 0
        assert "VISUAL INJECTION RESULT" in result.output
        assert "auto-derived" in result.output.lower()
        assert "forest" in result.output

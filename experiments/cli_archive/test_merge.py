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

"""Tests for merge CLI commands."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestMergeCommandHelp:
    """Test merge command help and basic invocation."""

    def test_merge_help(self):
        """merge --help should show usage information."""
        result = runner.invoke(app, ["merge", "--help"])
        assert result.exit_code == 0
        assert "merge" in result.stdout.lower()
        assert "--source" in result.stdout or "-s" in result.stdout

    def test_merge_run_help(self):
        """merge run --help should show usage information."""
        result = runner.invoke(app, ["merge", "run", "--help"])
        assert result.exit_code == 0
        assert "--source" in result.stdout
        assert "--target" in result.stdout
        assert "--output-dir" in result.stdout

    def test_merge_batch_help(self):
        """merge batch --help should show usage information."""
        result = runner.invoke(app, ["merge", "batch", "--help"])
        assert result.exit_code == 0
        assert "batch" in result.stdout.lower()
        assert "--source" in result.stdout

    def test_merge_deviation_help(self):
        """merge deviation --help should show usage information."""
        result = runner.invoke(app, ["merge", "deviation", "--help"])
        assert result.exit_code == 0
        assert "--baseline" in result.stdout
        assert "--current" in result.stdout

    def test_merge_validate_help(self):
        """merge validate --help should show usage information."""
        result = runner.invoke(app, ["merge", "validate", "--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout.lower()


class TestMergeCommandValidation:
    """Test merge command argument validation."""

    def test_merge_missing_source_error(self):
        """merge without --source should show error."""
        result = runner.invoke(app, ["merge", "-t", "/tmp/target", "-o", "/tmp/out"])
        # Should fail with missing required option
        assert result.exit_code != 0

    def test_merge_missing_target_error(self):
        """merge without --target should show error."""
        result = runner.invoke(app, ["merge", "-s", "/tmp/source", "-o", "/tmp/out"])
        assert result.exit_code != 0

    def test_merge_missing_output_error(self):
        """merge without --output-dir should show error."""
        result = runner.invoke(app, ["merge", "-s", "/tmp/source", "-t", "/tmp/target"])
        assert result.exit_code != 0

    def test_merge_partial_options_error(self):
        """merge with only some options should show clear error."""
        result = runner.invoke(app, ["merge", "-s", "/tmp/source"])
        assert result.exit_code != 0
        assert "missing" in result.stdout.lower() or result.exit_code == 1

    def test_merge_run_invalid_source_path(self):
        """merge run with non-existent source should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target"
            target.mkdir()
            (target / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "merge",
                    "run",
                    "-s", "/nonexistent/source",
                    "-t", str(target),
                    "-o", str(Path(tmpdir) / "output"),
                ],
            )
            assert result.exit_code != 0

    def test_merge_run_invalid_target_path(self):
        """merge run with non-existent target should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "merge",
                    "run",
                    "-s", str(source),
                    "-t", "/nonexistent/target",
                    "-o", str(Path(tmpdir) / "output"),
                ],
            )
            assert result.exit_code != 0


class TestMergeDryRun:
    """Test merge --dry-run functionality."""

    def test_merge_dry_run_shows_models(self):
        """merge --dry-run should show model info without merging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model directories
            source = Path(tmpdir) / "source"
            target = Path(tmpdir) / "target"
            source.mkdir()
            target.mkdir()

            source_config = {
                "model_type": "llama",
                "hidden_size": 2048,
                "vocab_size": 32000,
                "num_hidden_layers": 22,
            }
            target_config = {
                "model_type": "qwen2",
                "hidden_size": 1536,
                "vocab_size": 151936,
                "num_hidden_layers": 28,
            }
            (source / "config.json").write_text(json.dumps(source_config))
            (target / "config.json").write_text(json.dumps(target_config))

            # Create dummy safetensors files for size calculation
            (source / "model.safetensors").write_bytes(b"x" * 100)
            (target / "model.safetensors").write_bytes(b"x" * 100)

            # Mock the model probe service
            mock_source_info = MagicMock()
            mock_source_info.architecture = "llama"
            mock_source_info.parameter_count = 7_000_000_000
            mock_source_info.vocab_size = 32000
            mock_source_info.hidden_size = 2048
            mock_source_info.layers = list(range(22))
            mock_source_info.quantization = None

            mock_target_info = MagicMock()
            mock_target_info.architecture = "qwen2"
            mock_target_info.parameter_count = 3_000_000_000
            mock_target_info.vocab_size = 151936
            mock_target_info.hidden_size = 1536
            mock_target_info.layers = list(range(28))
            mock_target_info.quantization = None

            mock_service = MagicMock()
            mock_service.probe.side_effect = lambda p: (
                mock_source_info if "source" in p else mock_target_info
            )

            with patch(
                "modelcypher.cli.commands.merge.get_model_probe_service",
                return_value=mock_service,
            ):
                result = runner.invoke(
                    app,
                    [
                        "merge",
                        "-s", str(source),
                        "-t", str(target),
                        "-o", str(Path(tmpdir) / "output"),
                        "--dry-run",
                        "--output", "json",
                    ],
                )

            assert result.exit_code == 0
            payload = json.loads(result.stdout)
            assert "_schema" in payload
            assert "source" in payload
            assert "target" in payload


class TestMergeBatchCommand:
    """Test merge batch command."""

    def test_merge_batch_missing_sources_error(self):
        """merge batch without --source should show error."""
        result = runner.invoke(
            app,
            ["merge", "batch", "-t", "/tmp/target", "-o", "/tmp/out"],
        )
        assert result.exit_code != 0

    def test_merge_batch_missing_target_error(self):
        """merge batch without --target should show error."""
        result = runner.invoke(
            app,
            ["merge", "batch", "-s", "/tmp/source1", "-o", "/tmp/out"],
        )
        assert result.exit_code != 0

    def test_merge_batch_invalid_source_path(self):
        """merge batch with non-existent source should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target"
            target.mkdir()
            (target / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "merge",
                    "batch",
                    "-s", "/nonexistent/source1",
                    "-s", "/nonexistent/source2",
                    "-t", str(target),
                    "-o", str(Path(tmpdir) / "output"),
                ],
            )
            assert result.exit_code != 0


class TestMergeDeviationCommand:
    """Test merge deviation command."""

    def test_merge_deviation_missing_baseline_error(self):
        """merge deviation without --baseline should show error."""
        result = runner.invoke(
            app,
            ["merge", "deviation", "-c", "/tmp/current"],
        )
        assert result.exit_code != 0

    def test_merge_deviation_missing_current_error(self):
        """merge deviation without --current should show error."""
        result = runner.invoke(
            app,
            ["merge", "deviation", "-b", "/tmp/baseline"],
        )
        assert result.exit_code != 0

    def test_merge_deviation_invalid_baseline_path(self):
        """merge deviation with non-existent baseline should show error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current = Path(tmpdir) / "current"
            current.mkdir()
            (current / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "merge",
                    "deviation",
                    "-b", "/nonexistent/baseline",
                    "-c", str(current),
                ],
            )
            assert result.exit_code != 0


class TestMergeMultiChannelCommand:
    """Test merge multi-channel command."""

    def test_merge_multi_channel_help(self):
        """merge multi-channel --help should show usage."""
        result = runner.invoke(app, ["merge", "multi-channel", "--help"])
        assert result.exit_code == 0
        assert "--channel" in result.stdout
        assert "--routing" in result.stdout

    def test_merge_multi_channel_invalid_channel_format(self):
        """merge multi-channel with invalid channel format should error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target"
            target.mkdir()
            (target / "config.json").write_text('{"model_type": "llama"}')

            result = runner.invoke(
                app,
                [
                    "merge",
                    "multi-channel",
                    "-c", "invalid_no_colon",  # Missing colon
                    "-t", str(target),
                    "-o", str(Path(tmpdir) / "output"),
                ],
            )
            assert result.exit_code != 0
            assert "invalid" in result.stdout.lower() or "format" in result.stdout.lower()


class TestMergeBridgeCommand:
    """Test merge bridge command."""

    def test_merge_bridge_help(self):
        """merge bridge --help should show usage."""
        result = runner.invoke(app, ["merge", "bridge", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout
        assert "--samples" in result.stdout

    def test_merge_apply_bridge_help(self):
        """merge apply-bridge --help should show usage."""
        result = runner.invoke(app, ["merge", "apply-bridge", "--help"])
        assert result.exit_code == 0
        assert "--inverse" in result.stdout
        assert "--normalize" in result.stdout


class TestMergeValidateCommand:
    """Test merge validate command."""

    def test_merge_validate_help(self):
        """merge validate --help should show usage."""
        result = runner.invoke(app, ["merge", "validate", "--help"])
        assert result.exit_code == 0
        assert "--baseline" in result.stdout
        assert "--output" in result.stdout
        assert "--num-prompts" in result.stdout

    def test_merge_validate_invalid_model_path(self):
        """merge validate with non-existent model should show error."""
        result = runner.invoke(
            app,
            ["merge", "validate", "/nonexistent/model"],
        )
        assert result.exit_code != 0

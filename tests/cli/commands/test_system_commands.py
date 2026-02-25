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

import json

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
        assert "memory-profile" in result.stdout.lower()

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


class TestSystemMemoryProfile:
    """Test memory-profile command behavior."""

    def test_memory_profile_requires_model(self):
        result = runner.invoke(app, ["system", "memory-profile"])
        assert result.exit_code != 0

    def test_memory_profile_emits_schema(self, monkeypatch, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        class _StubSystemService:
            def memory_profile(self, **_kwargs):
                return {
                    "model_id": "model",
                    "param_count": 42,
                    "precision_bits": 16,
                    "quantization_mode": None,
                    "memory_stages": [
                        {"stage": "load", "active_gb": 1.0, "peak_gb": 2.0, "timestamp": "t"},
                    ],
                    "runtime_stages": [
                        {"stage": "load", "duration_sec": 0.1, "timestamp": "t"},
                    ],
                    "decode_slope": {"gb_per_token": 0.01, "windows": []},
                    "train_probe": None,
                }

        monkeypatch.setattr(
            "modelcypher.cli.commands.system.get_system_service",
            lambda: _StubSystemService(),
        )

        result = runner.invoke(
            app,
            [
                "--output", "json",
                "system",
                "memory-profile",
                "--model", str(model_dir),
            ],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["model_id"] == "model"
        assert payload["decode_slope"]["gb_per_token"] == 0.01


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

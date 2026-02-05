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

"""Tests for analyze (safety) CLI commands.

The 'analyze' command is the largest CLI module, containing ~25 subcommands
for geometric analysis, safety probing, entropy monitoring, and benchmarks.

Tests:
- Command help text for each subcommand
- Input validation for key commands
- Error handling for missing paths
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app

runner = CliRunner()


class TestAnalyzeCommandHelp:
    """Test that analyze commands have proper help text."""

    def test_analyze_help(self):
        """Test 'mc analyze --help' lists subcommands."""
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        # Should list various analyze subcommands
        stdout_lower = result.stdout.lower()
        # Check for some key commands (not all 25+)
        assert any(
            cmd in stdout_lower
            for cmd in ["adapter-probe", "dimension-profile", "entropy", "benchmark"]
        )


class TestSafetyAdapterProbe:
    """Tests for 'mc analyze adapter-probe' command."""

    def test_adapter_probe_help(self):
        """Test help text for adapter-probe."""
        result = runner.invoke(app, ["analyze", "adapter-probe", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.stdout

    def test_adapter_probe_requires_adapter(self):
        """Test that --adapter is required."""
        result = runner.invoke(app, ["analyze", "adapter-probe"])
        assert result.exit_code != 0


class TestSafetyBehavioralSignature:
    """Tests for 'mc analyze behavioral-signature' command."""

    def test_behavioral_signature_help(self):
        """Test help text for behavioral-signature."""
        result = runner.invoke(app, ["analyze", "behavioral-signature", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_behavioral_signature_requires_model(self):
        """Test that --model is required."""
        result = runner.invoke(app, ["analyze", "behavioral-signature"])
        assert result.exit_code != 0


class TestSafetyDimensionProfile:
    """Tests for 'mc analyze dimension-profile' command."""

    def test_dimension_profile_help(self):
        """Test help text for dimension-profile."""
        result = runner.invoke(app, ["analyze", "dimension-profile", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_dimension_profile_requires_model(self):
        """Test that model is required."""
        result = runner.invoke(app, ["analyze", "dimension-profile"])
        assert result.exit_code != 0


class TestSafetyEntropyCommands:
    """Tests for entropy-related analyze commands."""

    def test_entropy_trajectory_help(self):
        """Test help text for entropy-trajectory."""
        result = runner.invoke(app, ["analyze", "entropy-trajectory", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

    def test_spectral_trajectory_help(self):
        """Test help text for spectral-trajectory."""
        result = runner.invoke(app, ["analyze", "spectral-trajectory", "--help"])
        assert result.exit_code == 0


class TestSafetyReasoningFlow:
    """Tests for 'mc analyze reasoning-flow' command."""

    def test_reasoning_flow_help(self):
        """Test help text for reasoning-flow."""
        result = runner.invoke(app, ["analyze", "reasoning-flow", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout


class TestBenchmarkCommands:
    """Tests for benchmark commands under analyze."""

    def test_benchmark_help(self):
        """Test help text for benchmark command."""
        result = runner.invoke(app, ["analyze", "benchmark", "--help"])
        assert result.exit_code == 0


class TestLoRADiagnosticCommands:
    """Tests for LoRA diagnostic commands."""

    def test_lora_svd_help(self):
        """Test help text for lora-svd."""
        result = runner.invoke(app, ["analyze", "lora-svd", "--help"])
        assert result.exit_code == 0
        assert "--base" in result.stdout or "-b" in result.stdout

    def test_lora_svd_requires_adapter(self):
        """Test that adapter path is required."""
        result = runner.invoke(app, ["analyze", "lora-svd"])
        assert result.exit_code != 0


class TestCRMCommands:
    """Tests for Concept Response Matrix commands."""

    def test_crm_build_help(self):
        """Test help text for crm-build."""
        result = runner.invoke(app, ["analyze", "crm-build", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.stdout

    def test_crm_compare_help(self):
        """Test help text for crm-compare."""
        result = runner.invoke(app, ["analyze", "crm-compare", "--help"])
        assert result.exit_code == 0


class TestCurriculumProfile:
    """Tests for curriculum-profile command."""

    def test_curriculum_profile_help(self):
        """Test help text for curriculum-profile."""
        result = runner.invoke(app, ["analyze", "curriculum-profile", "--help"])
        assert result.exit_code == 0
        # Takes MODEL as positional arg, not --model
        assert "model" in result.stdout.lower() or "problems" in result.stdout.lower()


class TestJailbreakTest:
    """Tests for jailbreak-test command."""

    def test_jailbreak_test_help(self):
        """Test help text for jailbreak-test."""
        result = runner.invoke(app, ["analyze", "jailbreak-test", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.stdout

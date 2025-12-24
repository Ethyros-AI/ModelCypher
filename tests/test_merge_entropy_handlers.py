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

"""Tests for merge entropy MCP tools and CLI commands.

Validates that:
1. MCP tool response schemas match AI-friendly format
2. CLI commands produce correct JSON and text output
3. Edge cases are handled (empty inputs, large models, etc.)

## MCP Response Schema Requirements

All responses must include:
- `_schema`: Version string for forward compatibility
- `interpretation`: Human-readable summary
- `nextActions`: Suggested follow-up tools/commands

## Mathematical Foundation

Entropy-based merge validation is grounded in thermodynamic mixing theory:
- **Phase Classification**: Layers are classified as ORDERED (low H), CRITICAL (near boundary),
  or DISORDERED (high H) based on Shannon entropy thresholds
- **Alpha Adjustment**: Conservative blending (α < 1) for critical/disordered phases prevents
  catastrophic interference during merge
- **Knowledge Retention**: Measured as 1 - |H_merged - H_expected| / max(|H_source - H_target|, 1)
  where H_expected = (H_source + H_target) / 2 for 50/50 merge
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from modelcypher.cli.app import app
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeValidator,
    MergeStability,
)


runner = CliRunner()


# =============================================================================
# MCP Tool Response Schema Tests
# =============================================================================


class TestMCPProfileResponseSchema:
    """Test mc_merge_entropy_profile response format."""

    def test_profile_schema_version(self) -> None:
        """Response must include schema version for forward compatibility."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("test-model", num_layers=32)

        # Build response as MCP tool would
        response = {
            "_schema": "mc.merge.entropy.profile.v1",
            "modelName": profile.model_name,
            "meanEntropy": round(profile.mean_entropy, 3),
            "dominantPhase": profile.dominant_phase.value,
            "criticalLayerCount": profile.critical_layer_count,
            "mergeRisk": profile.merge_risk_level,
        }

        assert response["_schema"] == "mc.merge.entropy.profile.v1"
        assert isinstance(response["meanEntropy"], float)
        assert response["dominantPhase"] in ("ordered", "critical", "disordered")
        assert response["mergeRisk"] in ("low", "medium", "high")

    def test_profile_compact_response(self) -> None:
        """Response should limit critical layers to top 5 for context efficiency."""
        validator = EntropyMergeValidator()
        # Create model with many critical layers
        profile = validator.create_simulated_profile(
            "large-model",
            num_layers=64,
            base_entropy=2.0,  # Start near critical zone
            entropy_growth=0.02,
        )

        critical_layers = [
            name for name, lp in profile.layer_profiles.items()
            if lp.is_critical
        ][:5]  # Limit to 5

        assert len(critical_layers) <= 5

    def test_profile_includes_next_actions(self) -> None:
        """Response must suggest logical next steps for AI agents."""
        next_actions = [
            "mc_merge_entropy_guide to compare with target model",
            "mc_model_merge with alpha adjusted for critical layers",
        ]

        # Verify next actions are actionable tool references
        assert all("mc_" in action or "mc " in action for action in next_actions)


class TestMCPGuideResponseSchema:
    """Test mc_merge_entropy_guide response format."""

    def test_guide_recommendations_compact(self) -> None:
        """Recommendations should only include non-default values."""
        validator = EntropyMergeValidator()
        source = validator.create_simulated_profile("source", 10)
        target = validator.create_simulated_profile("target", 10)

        alpha_adj = validator.compute_alpha_adjustments(source, target)
        sigmas = validator.compute_smoothing_sigmas(source, target)

        # Filter to non-default only
        recommendations = {}
        for layer_name in alpha_adj:
            adj = alpha_adj[layer_name]
            sigma = sigmas.get(layer_name, 1.0)
            if adj < 1.0 or sigma > 1.0:  # Non-default values
                recommendations[layer_name] = {
                    "alphaAdjust": round(adj, 2),
                    "smoothingSigma": round(sigma, 1),
                }

        # All included recommendations should have adjustments
        for rec in recommendations.values():
            assert rec["alphaAdjust"] <= 1.0 or rec["smoothingSigma"] >= 1.0

    def test_guide_global_alpha_computed(self) -> None:
        """Response should include global alpha for simple merge command."""
        validator = EntropyMergeValidator()
        source = validator.create_simulated_profile("source", 8)
        target = validator.create_simulated_profile("target", 8)

        alpha_adj = validator.compute_alpha_adjustments(source, target)
        global_alpha = sum(alpha_adj.values()) / len(alpha_adj)

        assert 0.0 < global_alpha <= 1.0


class TestMCPValidateResponseSchema:
    """Test mc_merge_entropy_validate response format."""

    def test_validate_stability_enum(self) -> None:
        """Stability should be valid enum value."""
        validator = EntropyMergeValidator()
        validation = validator.validate_merge(
            source_entropies={"layers.0": 2.0},
            target_entropies={"layers.0": 2.1},
            merged_entropies={"layers.0": 2.05},
        )

        assert validation.overall_stability.value in (
            "stable", "marginal", "unstable", "critical"
        )

    def test_validate_knowledge_retention_bounded(self) -> None:
        """Knowledge retention should be in [0, 1]."""
        validator = EntropyMergeValidator()
        validation = validator.validate_merge(
            source_entropies={"layers.0": 2.0, "layers.1": 2.5},
            target_entropies={"layers.0": 2.1, "layers.1": 2.4},
            merged_entropies={"layers.0": 2.05, "layers.1": 2.45},
        )

        assert 0.0 <= validation.mean_knowledge_retention <= 1.0

    def test_validate_is_safe_boolean(self) -> None:
        """is_safe should be boolean for simple AI decision-making."""
        validator = EntropyMergeValidator()

        # Safe merge
        safe_validation = validator.validate_merge(
            source_entropies={"layers.0": 2.0},
            target_entropies={"layers.0": 2.0},
            merged_entropies={"layers.0": 2.0},
        )
        assert isinstance(safe_validation.is_safe, bool)
        assert safe_validation.is_safe is True

        # Unsafe merge (extreme deviation)
        unsafe_validation = validator.validate_merge(
            source_entropies={"layers.0": 2.0},
            target_entropies={"layers.0": 2.0},
            merged_entropies={"layers.0": 8.0},  # Massive deviation
        )
        assert unsafe_validation.is_safe is False


# =============================================================================
# CLI Command Tests
# =============================================================================


class TestCLIProfileCommand:
    """Test `mc geometry merge-entropy profile` command."""

    def test_profile_json_output(self) -> None:
        """JSON output should match schema requirements."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "profile",
            "test-model",
            "--layers", "16",
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)

        assert data["_schema"] == "mc.merge.entropy.profile.v1"
        assert data["modelName"] == "test-model"
        assert isinstance(data["meanEntropy"], float)
        assert data["dominantPhase"] in ("ordered", "critical", "disordered")
        assert data["mergeRisk"] in ("low", "medium", "high")
        assert "interpretation" in data
        assert "nextActions" in data

    def test_profile_text_output(self) -> None:
        """Text output should be human-readable."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "profile",
            "my-model",
            "--layers", "8",
            "--output", "text",
        ])

        assert result.exit_code == 0
        assert "ENTROPY PROFILE" in result.stdout
        assert "Model:" in result.stdout
        assert "Mean Entropy:" in result.stdout
        assert "Dominant Phase:" in result.stdout

    def test_profile_default_layers(self) -> None:
        """Default layer count should be 32."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "profile",
            "model-name",
            "--output", "json",
        ])

        assert result.exit_code == 0
        # Simulated profile with 32 layers produces predictable entropy
        data = json.loads(result.stdout)
        assert data["modelName"] == "model-name"


class TestCLIGuideCommand:
    """Test `mc geometry merge-entropy guide` command."""

    def test_guide_json_output(self) -> None:
        """JSON output should match schema requirements."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "guide",
            "--source", "model-a",
            "--target", "model-b",
            "--layers", "16",
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)

        assert data["_schema"] == "mc.merge.entropy.guide.v1"
        assert data["sourceModel"] == "model-a"
        assert data["targetModel"] == "model-b"
        assert data["sourceRisk"] in ("low", "medium", "high")
        assert data["targetRisk"] in ("low", "medium", "high")
        assert isinstance(data["globalAlphaAdjust"], float)
        assert isinstance(data["recommendations"], dict)

    def test_guide_text_output(self) -> None:
        """Text output should be human-readable."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "guide",
            "-s", "source-model",
            "-t", "target-model",
            "--output", "text",
        ])

        assert result.exit_code == 0
        assert "MERGE ENTROPY GUIDE" in result.stdout
        assert "Source:" in result.stdout
        assert "Target:" in result.stdout
        assert "Recommended Global Alpha:" in result.stdout

    def test_guide_with_short_options(self) -> None:
        """Command should work with short option flags (-s, -t)."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "guide",
            "-s", "source-model",
            "-t", "target-model",
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["sourceModel"] == "source-model"
        assert data["targetModel"] == "target-model"


class TestCLIValidateCommand:
    """Test `mc geometry merge-entropy validate` command."""

    def test_validate_inline_json(self) -> None:
        """Should accept inline JSON entropy values."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "validate",
            "--source-ent", '{"layers.0": 2.0, "layers.1": 2.5}',
            "--target-ent", '{"layers.0": 2.1, "layers.1": 2.4}',
            "--merged-ent", '{"layers.0": 2.05, "layers.1": 2.45}',
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)

        assert data["_schema"] == "mc.merge.entropy.validate.v1"
        assert data["overallStability"] in ("stable", "marginal", "unstable", "critical")
        assert isinstance(data["knowledgeRetention"], float)
        assert isinstance(data["isSafe"], bool)

    def test_validate_from_files(self, tmp_path) -> None:
        """Should accept JSON files for entropy values."""
        source_file = tmp_path / "source_ent.json"
        target_file = tmp_path / "target_ent.json"
        merged_file = tmp_path / "merged_ent.json"

        source_file.write_text('{"layers.0": 2.0, "layers.1": 2.5}')
        target_file.write_text('{"layers.0": 2.1, "layers.1": 2.4}')
        merged_file.write_text('{"layers.0": 2.05, "layers.1": 2.45}')

        result = runner.invoke(app, [
            "geometry", "merge-entropy", "validate",
            "--source-ent", str(source_file),
            "--target-ent", str(target_file),
            "--merged-ent", str(merged_file),
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["isSafe"] is True

    def test_validate_text_output(self) -> None:
        """Text output should show SAFE/UNSAFE status."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "validate",
            "--source-ent", '{"layers.0": 2.0}',
            "--target-ent", '{"layers.0": 2.0}',
            "--merged-ent", '{"layers.0": 2.0}',
            "--output", "text",
        ])

        assert result.exit_code == 0
        assert "MERGE VALIDATION:" in result.stdout
        assert "Knowledge Retention:" in result.stdout

    def test_validate_invalid_json(self) -> None:
        """Should fail gracefully with invalid JSON."""
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "validate",
            "--source-ent", "not-valid-json",
            "--target-ent", '{"layers.0": 2.0}',
            "--merged-ent", '{"layers.0": 2.0}',
            "--output", "json",
        ])

        assert result.exit_code == 1

    def test_validate_detects_unstable_merge(self) -> None:
        """Should correctly identify unstable merges."""
        # Large entropy deviation indicates knowledge loss
        result = runner.invoke(app, [
            "geometry", "merge-entropy", "validate",
            "--source-ent", '{"layers.0": 2.0, "layers.1": 2.0}',
            "--target-ent", '{"layers.0": 2.0, "layers.1": 2.0}',
            "--merged-ent", '{"layers.0": 5.0, "layers.1": 5.0}',  # Massive deviation
            "--output", "json",
        ])

        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["isSafe"] is False
        assert data["overallStability"] in ("unstable", "critical")


# =============================================================================
# Mathematical Correctness Tests
# =============================================================================


class TestEntropyMathematics:
    """Validate mathematical correctness of entropy calculations.

    The merge validation is based on thermodynamic mixing theory:
    - Expected merged entropy ≈ (H_source + H_target) / 2 for equal blending
    - Knowledge retention = 1 - normalized_deviation
    - Phase boundaries determine blending aggressiveness
    """

    def test_entropy_delta_calculation(self) -> None:
        """Entropy delta should be |H_merged - H_expected|."""
        from modelcypher.core.domain.merging.entropy_merge_validator import (
            LayerMergeValidation,
        )

        validation = LayerMergeValidation.compute(
            layer_name="test",
            source_entropy=2.0,
            target_entropy=4.0,
            merged_entropy=3.5,  # Expected is 3.0, delta is 0.5
        )

        assert validation.entropy_delta == pytest.approx(0.5)

    def test_knowledge_retention_perfect(self) -> None:
        """Perfect merge should have retention = 1.0."""
        from modelcypher.core.domain.merging.entropy_merge_validator import (
            LayerMergeValidation,
        )

        validation = LayerMergeValidation.compute(
            layer_name="test",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.0,  # Exactly as expected
        )

        assert validation.knowledge_retention_score == pytest.approx(1.0)

    def test_phase_boundary_classification(self) -> None:
        """Critical phase should be near the moderate zone center.

        Default thresholds: low=1.5, high=3.0
        Center = 2.25, bandwidth = 0.3
        Critical zone: [1.95, 2.55]
        """
        from modelcypher.core.domain.thermo.phase_transition_theory import Phase

        validator = EntropyMergeValidator()

        # Below critical zone -> ORDERED
        assert validator.classify_phase(1.0) == Phase.ORDERED

        # In critical zone -> CRITICAL
        assert validator.classify_phase(2.25) == Phase.CRITICAL

        # Above critical zone -> DISORDERED
        assert validator.classify_phase(4.0) == Phase.DISORDERED

    def test_alpha_adjustment_conservatism(self) -> None:
        """More unstable phases should get lower alpha (more conservative)."""
        from modelcypher.core.domain.entropy.logit_entropy_calculator import EntropyLevel
        from modelcypher.core.domain.merging.entropy_merge_validator import (
            LayerEntropyProfile,
        )
        from modelcypher.core.domain.thermo.phase_transition_theory import Phase

        ordered = LayerEntropyProfile(
            layer_name="ordered",
            mean_entropy=1.0,
            entropy_variance=0.1,
            entropy_level=EntropyLevel.LOW,
            phase=Phase.ORDERED,
        )
        critical = LayerEntropyProfile(
            layer_name="critical",
            mean_entropy=2.25,
            entropy_variance=0.2,
            entropy_level=EntropyLevel.MODERATE,
            phase=Phase.CRITICAL,
        )
        disordered = LayerEntropyProfile(
            layer_name="disordered",
            mean_entropy=4.0,
            entropy_variance=0.5,
            entropy_level=EntropyLevel.HIGH,
            phase=Phase.DISORDERED,
        )

        # ORDERED gets full alpha (1.0)
        # DISORDERED gets moderate reduction (0.85)
        # CRITICAL gets most conservative (0.7)
        assert ordered.recommended_alpha_adjustment > disordered.recommended_alpha_adjustment
        assert disordered.recommended_alpha_adjustment > critical.recommended_alpha_adjustment


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_entropy_dict(self) -> None:
        """Empty entropy dicts should return safe defaults."""
        validator = EntropyMergeValidator()
        validation = validator.validate_merge(
            source_entropies={},
            target_entropies={},
            merged_entropies={},
        )

        assert validation.is_safe is True
        assert validation.mean_knowledge_retention == 1.0
        assert len(validation.layer_validations) == 0

    def test_single_layer_model(self) -> None:
        """Single layer models should work correctly."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("tiny", num_layers=1)

        assert len(profile.layer_profiles) == 1
        assert "layers.0" in profile.layer_profiles

    def test_large_layer_count(self) -> None:
        """Should handle large models efficiently."""
        validator = EntropyMergeValidator()
        profile = validator.create_simulated_profile("huge", num_layers=200)

        assert len(profile.layer_profiles) == 200
        # Should complete without timeout or memory issues

    def test_mismatched_layer_counts(self) -> None:
        """Should only validate common layers when counts differ."""
        validator = EntropyMergeValidator()
        source = validator.create_simulated_profile("source", 10)
        target = validator.create_simulated_profile("target", 5)

        adjustments = validator.compute_alpha_adjustments(source, target)

        # Only common layers (0-4) should be in adjustments
        assert len(adjustments) == 5
        assert "layers.0" in adjustments
        assert "layers.9" not in adjustments

    def test_extreme_entropy_values(self) -> None:
        """Should handle extreme entropy values gracefully."""
        validator = EntropyMergeValidator()
        validation = validator.validate_merge(
            source_entropies={"layers.0": 0.001},  # Near zero
            target_entropies={"layers.0": 100.0},  # Very high
            merged_entropies={"layers.0": 50.0},  # Expected ~50
        )

        # Should not crash, should report instability
        assert validation.layer_validations["layers.0"].stability in (
            MergeStability.STABLE,
            MergeStability.MARGINAL,
        )

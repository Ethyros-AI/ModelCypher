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

"""Tests for EntropyMergeValidator."""

from __future__ import annotations

import pytest

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    EntropyThresholds,
)
# EntropyLevel enum removed - use raw entropy values with Phase
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeConfig,
    EntropyMergeValidator,
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    ModelEntropyProfile,
    PhaseAdjustments,
)

# MergeStability enum removed - tests now verify raw entropy_ratio values
from modelcypher.core.domain.thermo.phase_transition_theory import Phase


def _create_test_profile(name: str, num_layers: int) -> ModelEntropyProfile:
    """Create a test ModelEntropyProfile with deterministic entropy values.

    This replaces the deleted create_simulated_profile method.
    Creates layers with entropy increasing by depth (common pattern).
    """
    layer_profiles = {}
    for i in range(num_layers):
        # Entropy increases with depth
        entropy = 1.5 + i * 0.15
        variance = 0.1 + (i / num_layers) * 0.2

        # Classify based on raw entropy (no EntropyLevel enum)
        if entropy < 2.0:
            phase = Phase.ORDERED
        elif entropy < 2.5:
            phase = Phase.CRITICAL
        else:
            phase = Phase.DISORDERED

        layer_profiles[f"layers.{i}"] = LayerEntropyProfile(
            layer_name=f"layers.{i}",
            mean_entropy=entropy,
            entropy_variance=variance,
            phase=phase,
        )

    return ModelEntropyProfile.from_layer_profiles(name, layer_profiles)


class TestLayerEntropyProfile:
    """Tests for LayerEntropyProfile dataclass."""

    def test_ordered_phase_properties(self) -> None:
        """Ordered phase should have stable properties."""
        profile = LayerEntropyProfile(
            layer_name="layers.0",
            mean_entropy=1.0,
            entropy_variance=0.1,
            phase=Phase.ORDERED,
        )

        assert profile.is_stable
        assert not profile.is_critical
        assert profile.recommended_alpha_adjustment == 1.0
        assert profile.recommended_smoothing_sigma == 1.0

    def test_critical_phase_properties(self) -> None:
        """Critical phase should have conservative properties."""
        profile = LayerEntropyProfile(
            layer_name="layers.5",
            mean_entropy=2.25,
            entropy_variance=0.3,
            phase=Phase.CRITICAL,
        )

        assert not profile.is_stable
        assert profile.is_critical
        assert profile.recommended_alpha_adjustment == 0.7
        assert profile.recommended_smoothing_sigma == 2.0

    def test_disordered_phase_properties(self) -> None:
        """Disordered phase should have moderate properties."""
        profile = LayerEntropyProfile(
            layer_name="layers.10",
            mean_entropy=4.0,
            entropy_variance=0.5,
            phase=Phase.DISORDERED,
        )

        assert not profile.is_stable
        assert not profile.is_critical
        assert profile.recommended_alpha_adjustment == 0.85
        assert profile.recommended_smoothing_sigma == 1.5


class TestModelEntropyProfile:
    """Tests for ModelEntropyProfile dataclass."""

    def test_from_layer_profiles_computes_statistics(self) -> None:
        """Should compute aggregate statistics from layers."""
        layers = {
            "layers.0": LayerEntropyProfile(
                layer_name="layers.0",
                mean_entropy=1.0,
                entropy_variance=0.1,
                phase=Phase.ORDERED,
            ),
            "layers.1": LayerEntropyProfile(
                layer_name="layers.1",
                mean_entropy=2.0,
                entropy_variance=0.2,
                phase=Phase.ORDERED,
            ),
            "layers.2": LayerEntropyProfile(
                layer_name="layers.2",
                mean_entropy=3.0,
                entropy_variance=0.3,
                phase=Phase.CRITICAL,
            ),
        }

        profile = ModelEntropyProfile.from_layer_profiles("test_model", layers)

        assert profile.model_name == "test_model"
        assert profile.mean_entropy == 2.0  # (1 + 2 + 3) / 3
        assert profile.critical_layer_count == 1
        assert profile.dominant_phase == Phase.ORDERED  # 2 ordered vs 1 critical

    def test_empty_layers_returns_defaults(self) -> None:
        """Empty layers should return safe defaults."""
        profile = ModelEntropyProfile.from_layer_profiles("empty", {})

        assert profile.mean_entropy == 0.0
        assert profile.dominant_phase == Phase.ORDERED
        assert profile.critical_layer_count == 0

    def test_merge_risk_level_low(self) -> None:
        """Low risk when all ordered and no critical layers."""
        layers = {
            "layers.0": LayerEntropyProfile(
                layer_name="layers.0",
                mean_entropy=1.0,
                entropy_variance=0.1,
                phase=Phase.ORDERED,
            ),
        }
        profile = ModelEntropyProfile.from_layer_profiles("test", layers)

        assert profile.merge_risk_level == "low"

    def test_merge_risk_level_high(self) -> None:
        """High risk when many critical layers."""
        layers = {
            f"layers.{i}": LayerEntropyProfile(
                layer_name=f"layers.{i}",
                mean_entropy=2.25,
                entropy_variance=0.2,
                phase=Phase.CRITICAL,
            )
            for i in range(5)
        }
        profile = ModelEntropyProfile.from_layer_profiles("test", layers)

        assert profile.merge_risk_level == "high"


class TestLayerMergeValidation:
    """Tests for LayerMergeValidation.

    Tests verify raw measurements, not classifications.
    entropy_ratio IS the stability signal. Lower = more stable.
    """

    def test_perfect_merge_has_zero_ratio(self) -> None:
        """Merge with matching entropies has zero entropy ratio."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.0,
        )

        assert validation.entropy_delta == 0.0
        assert validation.entropy_ratio == 0.0
        assert validation.knowledge_retention_score == 1.0

    def test_small_deviation_has_small_ratio(self) -> None:
        """Merge with small delta has proportionally small ratio."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.5,  # Delta of 0.5
        )

        assert validation.entropy_delta == pytest.approx(0.5)
        # Ratio = 0.5 / 2.0 (expected) = 0.25
        assert validation.entropy_ratio == pytest.approx(0.25)

    def test_large_deviation_has_large_ratio(self) -> None:
        """Merge with large delta has correspondingly large ratio."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=3.0,  # Delta of 1.0
        )

        # Ratio = 1.0 / 2.0 = 0.5
        assert validation.entropy_ratio == pytest.approx(0.5)

    def test_severe_deviation_has_high_ratio(self) -> None:
        """Merge with severe delta has high entropy ratio and low retention."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=5.0,  # Delta of 3.0
        )

        # Ratio = 3.0 / 2.0 = 1.5
        assert validation.entropy_ratio == pytest.approx(1.5)
        assert validation.knowledge_retention_score < 0.5

    def test_ratio_ordering_reflects_stability(self) -> None:
        """Lower entropy_ratio indicates more stable merge."""
        stable = LayerMergeValidation.compute("l0", 2.0, 2.0, 2.0)
        moderate = LayerMergeValidation.compute("l1", 2.0, 2.0, 2.5)
        severe = LayerMergeValidation.compute("l2", 2.0, 2.0, 5.0)

        assert stable.entropy_ratio < moderate.entropy_ratio < severe.entropy_ratio


class TestMergeEntropyValidation:
    """Tests for MergeEntropyValidation.

    Tests verify raw aggregate measurements, not classifications.
    Lower mean_entropy_ratio and max_entropy_ratio = more stable.
    """

    def test_from_layer_validations_stable(self) -> None:
        """Stable merge should have low entropy ratios."""
        layers = {
            "layers.0": LayerMergeValidation.compute("layers.0", 2.0, 2.0, 2.0),
            "layers.1": LayerMergeValidation.compute("layers.1", 2.0, 2.0, 2.05),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        # Low entropy ratios indicate stable merge - this is the primary signal
        assert validation.mean_entropy_ratio < 0.05
        assert validation.max_entropy_ratio < 0.05
        # Knowledge retention is secondary; derived from entropy ratio
        assert validation.mean_knowledge_retention > 0.7

    def test_from_layer_validations_problematic(self) -> None:
        """Problematic merge should have high entropy ratios."""
        layers = {
            "layers.0": LayerMergeValidation.compute(
                "layers.0",
                2.0,
                2.0,
                5.0,  # Large deviation
            ),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        # High entropy ratio indicates problems
        assert validation.mean_entropy_ratio > 1.0
        assert validation.max_entropy_ratio > 1.0
        assert validation.mean_knowledge_retention < 0.5

    def test_layers_by_entropy_ratio(self) -> None:
        """Should sort layers by entropy ratio."""
        layers = {
            "best": LayerMergeValidation.compute("best", 2.0, 2.0, 2.0),
            "mid": LayerMergeValidation.compute("mid", 2.0, 2.0, 2.5),
            "worst": LayerMergeValidation.compute("worst", 2.0, 2.0, 4.0),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        # Descending order (worst first)
        worst_first = validation.layers_by_entropy_ratio(descending=True)
        assert worst_first[0] == "worst"
        assert worst_first[-1] == "best"

        # Ascending order (best first)
        best_first = validation.layers_by_entropy_ratio(descending=False)
        assert best_first[0] == "best"
        assert best_first[-1] == "worst"

    def test_summary_formatting(self) -> None:
        """Summary should show raw measurements."""
        layers = {
            "layers.0": LayerMergeValidation.compute("layers.0", 2.0, 2.0, 2.0),
        }

        validation = MergeEntropyValidation.from_layer_validations("source", "target", layers)

        assert "entropy ratio" in validation.summary.lower()
        assert "Knowledge retention:" in validation.summary


class TestEntropyMergeValidator:
    """Tests for EntropyMergeValidator."""

    @pytest.fixture
    def validator(self) -> EntropyMergeValidator:
        """Create default validator."""
        return EntropyMergeValidator()

    # classify_entropy method removed - using raw entropy with classify_phase
    # Tests now verify phase classification directly from raw entropy values

    def test_classify_phase_ordered(self, validator: EntropyMergeValidator) -> None:
        """Low entropy should be ORDERED phase."""
        phase = validator.classify_phase(1.0)
        assert phase == Phase.ORDERED

    def test_classify_phase_critical(self, validator: EntropyMergeValidator) -> None:
        """Near-boundary entropy should be CRITICAL phase."""
        # Default thresholds: low=1.5, high=3.0, center=2.25
        # With bandwidth 0.3, critical range is [1.95, 2.55]
        phase = validator.classify_phase(2.25)
        assert phase == Phase.CRITICAL

    def test_classify_phase_disordered(self, validator: EntropyMergeValidator) -> None:
        """High entropy should be DISORDERED phase."""
        phase = validator.classify_phase(4.0)
        assert phase == Phase.DISORDERED

    def test_create_layer_profile(self, validator: EntropyMergeValidator) -> None:
        """Should create profile from entropy values."""
        entropies = [1.0, 1.1, 0.9, 1.0, 1.0]
        profile = validator.create_layer_profile("layers.0", entropies)

        assert profile.layer_name == "layers.0"
        assert profile.mean_entropy == pytest.approx(1.0)
        # entropy_level field removed - just check raw value and phase
        assert profile.mean_entropy < 2.0  # Low entropy
        assert profile.phase == Phase.ORDERED

    def test_create_layer_profile_empty(self, validator: EntropyMergeValidator) -> None:
        """Empty entropy list should return defaults."""
        profile = validator.create_layer_profile("layers.0", [])

        assert profile.mean_entropy == 0.0
        assert profile.phase == Phase.ORDERED

    def test_create_test_profile_structure(self, validator: EntropyMergeValidator) -> None:
        """Test profile creation has expected structure."""
        profile = _create_test_profile("test_model", num_layers=10)

        assert profile.model_name == "test_model"
        assert len(profile.layer_profiles) == 10
        assert "layers.0" in profile.layer_profiles
        assert "layers.9" in profile.layer_profiles

    def test_compute_alpha_adjustments(self, validator: EntropyMergeValidator) -> None:
        """Should compute per-layer alpha adjustments."""
        source = _create_test_profile("source", 5)
        target = _create_test_profile("target", 5)

        adjustments = validator.compute_alpha_adjustments(source, target)

        assert len(adjustments) == 5
        assert all(0.0 < adj <= 1.0 for adj in adjustments.values())

    def test_compute_smoothing_sigmas(self, validator: EntropyMergeValidator) -> None:
        """Should compute per-layer smoothing sigmas."""
        source = _create_test_profile("source", 5)
        target = _create_test_profile("target", 5)

        sigmas = validator.compute_smoothing_sigmas(source, target)

        assert len(sigmas) == 5
        assert all(sigma >= 1.0 for sigma in sigmas.values())

    def test_validate_merge(self, validator: EntropyMergeValidator) -> None:
        """Should validate merge from entropy measurements."""
        source_entropies = {"layers.0": 2.0, "layers.1": 2.5}
        target_entropies = {"layers.0": 2.1, "layers.1": 2.4}
        merged_entropies = {"layers.0": 2.05, "layers.1": 2.45}

        validation = validator.validate_merge(
            source_entropies=source_entropies,
            target_entropies=target_entropies,
            merged_entropies=merged_entropies,
        )

        # Low entropy ratios indicate good merge
        assert validation.mean_entropy_ratio < 0.1
        assert len(validation.layer_validations) == 2

    def test_validate_merge_missing_layers(self, validator: EntropyMergeValidator) -> None:
        """Should only validate common layers."""
        source_entropies = {"layers.0": 2.0, "layers.1": 2.5}
        target_entropies = {"layers.0": 2.1}  # Missing layer.1
        merged_entropies = {"layers.0": 2.05, "layers.1": 2.45}

        validation = validator.validate_merge(
            source_entropies=source_entropies,
            target_entropies=target_entropies,
            merged_entropies=merged_entropies,
        )

        # Only layers.0 is in all three
        assert len(validation.layer_validations) == 1
        assert "layers.0" in validation.layer_validations

    def test_generate_merge_guidance(self, validator: EntropyMergeValidator) -> None:
        """Should generate markdown guidance."""
        source = _create_test_profile("source", 3)
        target = _create_test_profile("target", 3)

        guidance = validator.generate_merge_guidance(source, target)

        assert "# Entropy-Guided Merge Recommendations" in guidance
        assert "## Model Analysis" in guidance
        assert "## Per-Layer Recommendations" in guidance
        assert "source" in guidance
        assert "target" in guidance


class TestIntegrationWithThresholds:
    """Integration tests with custom thresholds."""

    def test_custom_thresholds(self) -> None:
        """Validator should respect custom thresholds."""
        thresholds = EntropyThresholds(low=2.0, high=4.0, circuit_breaker=5.0)
        validator = EntropyMergeValidator(thresholds=thresholds)

        # With higher thresholds, 1.5 is below low threshold (ORDERED)
        phase = validator.classify_phase(1.5)
        assert phase == Phase.ORDERED

        # And 2.5 is between low and high (could be ORDERED, CRITICAL, or DISORDERED)
        # depending on position relative to center (3.0)
        phase = validator.classify_phase(2.5)
        assert phase == Phase.ORDERED  # Below center, so ORDERED

    def test_custom_critical_bandwidth(self) -> None:
        """Validator should respect custom critical bandwidth."""
        validator = EntropyMergeValidator(critical_bandwidth=0.5)

        # With wider bandwidth, more values classify as CRITICAL
        # Center is 2.25, bandwidth 0.5 means [1.75, 2.75] is critical
        phase = validator.classify_phase(2.0)
        assert phase == Phase.CRITICAL

        phase = validator.classify_phase(2.5)
        assert phase == Phase.CRITICAL


class TestPhaseAdjustments:
    """Tests for PhaseAdjustments dataclass."""

    def test_alpha_for_phase(self) -> None:
        """Should return correct alpha for each phase."""
        adj = PhaseAdjustments(
            ordered_alpha=1.0,
            critical_alpha=0.6,
            disordered_alpha=0.8,
            ordered_sigma=1.0,
            critical_sigma=2.5,
            disordered_sigma=1.5,
        )

        assert adj.alpha_for_phase(Phase.ORDERED) == 1.0
        assert adj.alpha_for_phase(Phase.CRITICAL) == 0.6
        assert adj.alpha_for_phase(Phase.DISORDERED) == 0.8

    def test_sigma_for_phase(self) -> None:
        """Should return correct sigma for each phase."""
        adj = PhaseAdjustments(
            ordered_alpha=1.0,
            critical_alpha=0.6,
            disordered_alpha=0.8,
            ordered_sigma=1.0,
            critical_sigma=2.5,
            disordered_sigma=1.5,
        )

        assert adj.sigma_for_phase(Phase.ORDERED) == 1.0
        assert adj.sigma_for_phase(Phase.CRITICAL) == 2.5
        assert adj.sigma_for_phase(Phase.DISORDERED) == 1.5


class TestEntropyMergeConfig:
    """Tests for EntropyMergeConfig dataclass."""

    def test_from_entropy_statistics(self) -> None:
        """Should derive config from entropy statistics."""
        config = EntropyMergeConfig.from_entropy_statistics(
            entropy_mean=2.5,
            entropy_std=0.75,
        )

        # low = mean - std = 1.75
        assert config.entropy_thresholds.low == pytest.approx(1.75)
        # high = mean + std = 3.25
        assert config.entropy_thresholds.high == pytest.approx(3.25)
        # circuit_breaker = mean + 2*std = 4.0
        assert config.entropy_thresholds.circuit_breaker == pytest.approx(4.0)
        # critical_bandwidth = 0.5 * std = 0.375
        assert config.critical_bandwidth == pytest.approx(0.375)

    def test_from_calibration_data(self) -> None:
        """Should derive config from calibration data."""
        entropy_samples = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        merge_deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        config = EntropyMergeConfig.from_calibration_data(
            entropy_samples=entropy_samples,
            merge_deltas=merge_deltas,
            percentile_low=25.0,
            percentile_high=75.0,
        )

        # Thresholds derived from percentiles
        assert config.entropy_thresholds.low > 0
        assert config.entropy_thresholds.high > config.entropy_thresholds.low
        assert config.stability_thresholds[0] > 0  # stable_mult
        assert config.stability_thresholds[1] > config.stability_thresholds[0]  # marginal > stable

    def test_from_calibration_data_empty_raises(self) -> None:
        """Should raise error for empty calibration data."""
        with pytest.raises(ValueError, match="cannot be empty"):
            EntropyMergeConfig.from_calibration_data(
                entropy_samples=[],
                merge_deltas=[0.1],
            )

        with pytest.raises(ValueError, match="cannot be empty"):
            EntropyMergeConfig.from_calibration_data(
                entropy_samples=[1.0],
                merge_deltas=[],
            )


class TestConfigBasedValidator:
    """Tests for validator with explicit config."""

    def test_validator_with_config(self) -> None:
        """Validator should use config values."""
        config = EntropyMergeConfig.from_entropy_statistics(
            entropy_mean=2.0,
            entropy_std=0.5,
        )
        validator = EntropyMergeValidator(config)

        assert validator.thresholds.low == pytest.approx(1.5)
        assert validator.thresholds.high == pytest.approx(2.5)

    def test_validator_config_adjustments_used(self) -> None:
        """Alpha adjustments should use config phase adjustments."""
        custom_adj = PhaseAdjustments(
            ordered_alpha=0.9,  # Different from default 1.0
            critical_alpha=0.5,  # Different from default 0.7
            disordered_alpha=0.7,  # Different from default 0.85
            ordered_sigma=1.5,
            critical_sigma=3.0,
            disordered_sigma=2.0,
        )
        config = EntropyMergeConfig(
            entropy_thresholds=EntropyThresholds(low=1.5, high=3.0, circuit_breaker=4.0),
            critical_bandwidth=0.3,
            phase_adjustments=custom_adj,
            high_risk_fraction=0.3,
            unstable_fraction=0.2,
            stability_thresholds=(0.2, 0.5, 0.5),
        )
        validator = EntropyMergeValidator(config)

        # Create profiles with known phases
        ordered_profile = LayerEntropyProfile(
            layer_name="layers.0",
            mean_entropy=1.0,
            entropy_variance=0.1,
            entropy_level=EntropyLevel.LOW,
            phase=Phase.ORDERED,
        )

        # Verify custom alpha adjustment is used
        adj = ordered_profile.alpha_adjustment(config.phase_adjustments)
        assert adj == 0.9  # Custom value, not default 1.0

    def test_backward_compatible_api(self) -> None:
        """Old API should still work."""
        # Old-style initialization with keyword args
        validator = EntropyMergeValidator(
            thresholds=EntropyThresholds(low=2.0, high=4.0, circuit_breaker=5.0),
            critical_bandwidth=0.5,
        )

        # Should have created config internally
        assert validator.config is not None
        assert validator.config.entropy_thresholds.low == 2.0
        assert validator.config.critical_bandwidth == 0.5

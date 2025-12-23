"""Tests for EntropyMergeValidator."""
from __future__ import annotations

import pytest

from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    EntropyLevel,
    EntropyThresholds,
)
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeValidator,
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    MergeStability,
    ModelEntropyProfile,
)
from modelcypher.core.domain.thermo.phase_transition_theory import Phase


class TestLayerEntropyProfile:
    """Tests for LayerEntropyProfile dataclass."""

    def test_ordered_phase_properties(self) -> None:
        """Ordered phase should have stable properties."""
        profile = LayerEntropyProfile(
            layer_name="layers.0",
            mean_entropy=1.0,
            entropy_variance=0.1,
            entropy_level=EntropyLevel.LOW,
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
            entropy_level=EntropyLevel.MODERATE,
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
            entropy_level=EntropyLevel.HIGH,
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
                entropy_level=EntropyLevel.LOW,
                phase=Phase.ORDERED,
            ),
            "layers.1": LayerEntropyProfile(
                layer_name="layers.1",
                mean_entropy=2.0,
                entropy_variance=0.2,
                entropy_level=EntropyLevel.MODERATE,
                phase=Phase.ORDERED,
            ),
            "layers.2": LayerEntropyProfile(
                layer_name="layers.2",
                mean_entropy=3.0,
                entropy_variance=0.3,
                entropy_level=EntropyLevel.MODERATE,
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
                entropy_level=EntropyLevel.LOW,
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
                entropy_level=EntropyLevel.MODERATE,
                phase=Phase.CRITICAL,
            )
            for i in range(5)
        }
        profile = ModelEntropyProfile.from_layer_profiles("test", layers)

        assert profile.merge_risk_level == "high"


class TestLayerMergeValidation:
    """Tests for LayerMergeValidation."""

    def test_stable_merge(self) -> None:
        """Merge with matching entropies should be stable."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.0,
        )

        assert validation.stability == MergeStability.STABLE
        assert validation.entropy_delta == 0.0
        assert validation.knowledge_retention_score == 1.0

    def test_marginal_merge(self) -> None:
        """Merge with small delta should be marginal."""
        # Thresholds: low=1.5, stable if delta < 0.3, marginal if delta < 0.75
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=2.5,  # Delta of 0.5 is in marginal range
        )

        assert validation.stability == MergeStability.MARGINAL
        assert validation.entropy_delta == pytest.approx(0.5)

    def test_unstable_merge(self) -> None:
        """Merge with large delta should be unstable."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=3.0,  # Large deviation
        )

        assert validation.stability == MergeStability.UNSTABLE

    def test_critical_merge(self) -> None:
        """Merge with very large delta should be critical."""
        validation = LayerMergeValidation.compute(
            layer_name="layers.0",
            source_entropy=2.0,
            target_entropy=2.0,
            merged_entropy=5.0,  # Very large deviation
        )

        assert validation.stability == MergeStability.CRITICAL
        assert validation.knowledge_retention_score < 0.5


class TestMergeEntropyValidation:
    """Tests for MergeEntropyValidation."""

    def test_from_layer_validations_safe(self) -> None:
        """Safe merge should have stable overall status."""
        layers = {
            "layers.0": LayerMergeValidation.compute(
                "layers.0", 2.0, 2.0, 2.0
            ),
            "layers.1": LayerMergeValidation.compute(
                "layers.1", 2.0, 2.0, 2.1
            ),
        }

        validation = MergeEntropyValidation.from_layer_validations(
            "source", "target", layers
        )

        assert validation.is_safe
        assert validation.overall_stability in (MergeStability.STABLE, MergeStability.MARGINAL)
        assert len(validation.critical_layer_names) == 0

    def test_from_layer_validations_unsafe(self) -> None:
        """Unsafe merge should have critical status."""
        layers = {
            "layers.0": LayerMergeValidation.compute(
                "layers.0", 2.0, 2.0, 5.0  # Critical
            ),
        }

        validation = MergeEntropyValidation.from_layer_validations(
            "source", "target", layers
        )

        assert not validation.is_safe
        assert validation.overall_stability == MergeStability.CRITICAL
        assert "layers.0" in validation.critical_layer_names

    def test_summary_formatting(self) -> None:
        """Summary should be human-readable."""
        layers = {
            "layers.0": LayerMergeValidation.compute(
                "layers.0", 2.0, 2.0, 2.0
            ),
        }

        validation = MergeEntropyValidation.from_layer_validations(
            "source", "target", layers
        )

        assert "Merge validation:" in validation.summary
        assert "Knowledge retention:" in validation.summary


class TestEntropyMergeValidator:
    """Tests for EntropyMergeValidator."""

    @pytest.fixture
    def validator(self) -> EntropyMergeValidator:
        """Create default validator."""
        return EntropyMergeValidator()

    def test_classify_entropy_low(self, validator: EntropyMergeValidator) -> None:
        """Low entropy should classify as LOW."""
        level = validator.classify_entropy(1.0)
        assert level == EntropyLevel.LOW

    def test_classify_entropy_moderate(self, validator: EntropyMergeValidator) -> None:
        """Moderate entropy should classify as MODERATE."""
        level = validator.classify_entropy(2.0)
        assert level == EntropyLevel.MODERATE

    def test_classify_entropy_high(self, validator: EntropyMergeValidator) -> None:
        """High entropy should classify as HIGH."""
        level = validator.classify_entropy(4.0)
        assert level == EntropyLevel.HIGH

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
        assert profile.entropy_level == EntropyLevel.LOW
        assert profile.phase == Phase.ORDERED

    def test_create_layer_profile_empty(self, validator: EntropyMergeValidator) -> None:
        """Empty entropy list should return defaults."""
        profile = validator.create_layer_profile("layers.0", [])

        assert profile.mean_entropy == 0.0
        assert profile.phase == Phase.ORDERED

    def test_create_simulated_profile(self, validator: EntropyMergeValidator) -> None:
        """Should create simulated profile with expected structure."""
        profile = validator.create_simulated_profile("test_model", num_layers=10)

        assert profile.model_name == "test_model"
        assert len(profile.layer_profiles) == 10
        assert "layers.0" in profile.layer_profiles
        assert "layers.9" in profile.layer_profiles

    def test_compute_alpha_adjustments(self, validator: EntropyMergeValidator) -> None:
        """Should compute per-layer alpha adjustments."""
        source = validator.create_simulated_profile("source", 5)
        target = validator.create_simulated_profile("target", 5)

        adjustments = validator.compute_alpha_adjustments(source, target)

        assert len(adjustments) == 5
        assert all(0.0 < adj <= 1.0 for adj in adjustments.values())

    def test_compute_smoothing_sigmas(self, validator: EntropyMergeValidator) -> None:
        """Should compute per-layer smoothing sigmas."""
        source = validator.create_simulated_profile("source", 5)
        target = validator.create_simulated_profile("target", 5)

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

        assert validation.is_safe
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
        source = validator.create_simulated_profile("source", 3)
        target = validator.create_simulated_profile("target", 3)

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

        # With higher thresholds, 1.5 is now LOW
        level = validator.classify_entropy(1.5)
        assert level == EntropyLevel.LOW

        # And 2.5 is now MODERATE
        level = validator.classify_entropy(2.5)
        assert level == EntropyLevel.MODERATE

    def test_custom_critical_bandwidth(self) -> None:
        """Validator should respect custom critical bandwidth."""
        validator = EntropyMergeValidator(critical_bandwidth=0.5)

        # With wider bandwidth, more values classify as CRITICAL
        # Center is 2.25, bandwidth 0.5 means [1.75, 2.75] is critical
        phase = validator.classify_phase(2.0)
        assert phase == Phase.CRITICAL

        phase = validator.classify_phase(2.5)
        assert phase == Phase.CRITICAL

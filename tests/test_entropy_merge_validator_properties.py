# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Property-based tests for the entropy merge validator.

Uses Hypothesis to verify mathematical properties and invariants.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, assume, settings

from modelcypher.core.domain.entropy.logit_entropy_calculator import EntropyThresholds
from modelcypher.core.domain.merging.entropy_merge_validator import (
    EntropyMergeConfig,
    EntropyMergeValidator,
    LayerEntropyProfile,
    LayerMergeValidation,
    MergeEntropyValidation,
    ModelEntropyProfile,
    PhaseAdjustments,
)
from modelcypher.core.domain.thermo.phase_transition_theory import Phase


# =============================================================================
# Strategy Definitions
# =============================================================================


# Entropy values are typically positive and bounded
entropy_strategy = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

# Non-negative entropy for edge cases
non_negative_entropy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Positive floats for config values
positive_float = st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)


DEFAULT_ADJUSTMENTS = PhaseAdjustments(
    ordered_alpha=1.0,
    critical_alpha=0.7,
    disordered_alpha=0.85,
    ordered_sigma=1.0,
    critical_sigma=2.0,
    disordered_sigma=1.5,
)


def _make_config(thresholds: EntropyThresholds, bandwidth: float) -> EntropyMergeConfig:
    return EntropyMergeConfig(
        entropy_thresholds=thresholds,
        critical_bandwidth=bandwidth,
        phase_adjustments=DEFAULT_ADJUSTMENTS,
        high_risk_fraction=0.3,
        unstable_fraction=0.2,
        stability_thresholds=(0.2, 0.5, 0.5),
    )

# Thresholds where low < high < circuit_breaker
@st.composite
def entropy_thresholds_strategy(draw):
    low = draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False))
    high = draw(st.floats(min_value=low + 0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    circuit = draw(st.floats(min_value=high + 0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    return EntropyThresholds(low=low, high=high, circuit_breaker=circuit)


@st.composite
def phase_adjustments_strategy(draw):
    return PhaseAdjustments(
        ordered_alpha=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        critical_alpha=draw(st.floats(min_value=0.1, max_value=0.8, allow_nan=False)),
        disordered_alpha=draw(st.floats(min_value=0.3, max_value=0.9, allow_nan=False)),
        ordered_sigma=draw(st.floats(min_value=0.5, max_value=2.0, allow_nan=False)),
        critical_sigma=draw(st.floats(min_value=1.0, max_value=3.0, allow_nan=False)),
        disordered_sigma=draw(st.floats(min_value=0.8, max_value=2.5, allow_nan=False)),
    )


# =============================================================================
# Phase Classification Properties
# =============================================================================


class TestPhaseClassificationProperties:
    """Property tests for phase classification."""

    @given(
        thresholds=entropy_thresholds_strategy(),
        entropy=entropy_strategy,
        bandwidth=st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_phase_classification_exhaustive(
        self, thresholds: EntropyThresholds, entropy: float, bandwidth: float
    ) -> None:
        """Every entropy value should be classified into exactly one phase."""
        validator = EntropyMergeValidator(_make_config(thresholds, bandwidth))

        phase = validator.classify_phase(entropy)

        # Must be one of the three phases
        assert phase in (Phase.ORDERED, Phase.CRITICAL, Phase.DISORDERED)

    @given(
        thresholds=entropy_thresholds_strategy(),
        bandwidth=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_low_entropy_is_ordered(
        self, thresholds: EntropyThresholds, bandwidth: float
    ) -> None:
        """Entropy below low threshold should always be ORDERED."""
        validator = EntropyMergeValidator(_make_config(thresholds, bandwidth))

        # Test at half the low threshold
        low_entropy = thresholds.low / 2
        assume(low_entropy > 0)

        phase = validator.classify_phase(low_entropy)
        assert phase == Phase.ORDERED

    @given(
        thresholds=entropy_thresholds_strategy(),
        bandwidth=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_high_entropy_is_disordered(
        self, thresholds: EntropyThresholds, bandwidth: float
    ) -> None:
        """Entropy at or above high threshold should be DISORDERED."""
        validator = EntropyMergeValidator(_make_config(thresholds, bandwidth))

        # Test at exactly high threshold
        phase = validator.classify_phase(thresholds.high)
        assert phase == Phase.DISORDERED

        # Test above high threshold
        phase_high = validator.classify_phase(thresholds.high + 0.5)
        assert phase_high == Phase.DISORDERED

    @given(
        thresholds=entropy_thresholds_strategy(),
    )
    @settings(max_examples=50)
    def test_center_of_moderate_zone_is_critical(
        self, thresholds: EntropyThresholds
    ) -> None:
        """Entropy at center of moderate zone should be CRITICAL."""
        moderate_center = (thresholds.low + thresholds.high) / 2
        bandwidth = (thresholds.high - thresholds.low) / 4  # Covers center

        validator = EntropyMergeValidator(_make_config(thresholds, bandwidth))

        phase = validator.classify_phase(moderate_center)
        assert phase == Phase.CRITICAL


# =============================================================================
# Phase Adjustments Properties
# =============================================================================


class TestPhaseAdjustmentsProperties:
    """Property tests for phase adjustment values."""

    @given(adjustments=phase_adjustments_strategy())
    @settings(max_examples=50)
    def test_alpha_for_phase_returns_correct_value(
        self, adjustments: PhaseAdjustments
    ) -> None:
        """alpha_for_phase should return correct value for each phase."""
        assert adjustments.alpha_for_phase(Phase.ORDERED) == adjustments.ordered_alpha
        assert adjustments.alpha_for_phase(Phase.CRITICAL) == adjustments.critical_alpha
        assert adjustments.alpha_for_phase(Phase.DISORDERED) == adjustments.disordered_alpha

    @given(adjustments=phase_adjustments_strategy())
    @settings(max_examples=50)
    def test_sigma_for_phase_returns_correct_value(
        self, adjustments: PhaseAdjustments
    ) -> None:
        """sigma_for_phase should return correct value for each phase."""
        assert adjustments.sigma_for_phase(Phase.ORDERED) == adjustments.ordered_sigma
        assert adjustments.sigma_for_phase(Phase.CRITICAL) == adjustments.critical_sigma
        assert adjustments.sigma_for_phase(Phase.DISORDERED) == adjustments.disordered_sigma


# =============================================================================
# Layer Merge Validation Properties
# =============================================================================


class TestLayerMergeValidationProperties:
    """Property tests for layer merge validation."""

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_entropy_ratio_is_non_negative(
        self, source: float, target: float, merged: float
    ) -> None:
        """Entropy ratio should always be non-negative."""
        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert validation.entropy_ratio >= 0.0

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_knowledge_retention_in_bounds(
        self, source: float, target: float, merged: float
    ) -> None:
        """Knowledge retention score should be in [0, 1]."""
        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert 0.0 <= validation.knowledge_retention_score <= 1.0

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_perfect_merge_has_high_retention(
        self, source: float, target: float
    ) -> None:
        """Merged entropy equal to expected should have high retention."""
        expected = (source + target) / 2

        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=expected,
        )

        # Perfect merge should have retention close to 1.0
        assert validation.knowledge_retention_score >= 0.99

    @given(
        source=entropy_strategy,
        target=entropy_strategy,
        merged=entropy_strategy,
    )
    @settings(max_examples=50)
    def test_entropy_delta_is_absolute(
        self, source: float, target: float, merged: float
    ) -> None:
        """Entropy delta should be absolute value of difference from expected."""
        expected = (source + target) / 2
        expected_delta = abs(merged - expected)

        validation = LayerMergeValidation.compute(
            layer_name="test_layer",
            source_entropy=source,
            target_entropy=target,
            merged_entropy=merged,
        )

        assert validation.entropy_delta == pytest.approx(expected_delta, rel=1e-6)


# =============================================================================
# Merge Entropy Validation Properties
# =============================================================================


class TestMergeEntropyValidationProperties:
    """Property tests for overall merge validation."""

    @given(
        layer_data=st.lists(
            st.tuples(entropy_strategy, entropy_strategy, entropy_strategy),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_max_ratio_gte_mean_ratio(
        self, layer_data: list[tuple[float, float, float]]
    ) -> None:
        """Max entropy ratio should be >= mean entropy ratio."""
        layer_validations = {}
        for i, (source, target, merged) in enumerate(layer_data):
            layer_validations[f"layer_{i}"] = LayerMergeValidation.compute(
                layer_name=f"layer_{i}",
                source_entropy=source,
                target_entropy=target,
                merged_entropy=merged,
            )

        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations=layer_validations,
        )

        assert validation.max_entropy_ratio >= validation.mean_entropy_ratio

    @given(
        layer_data=st.lists(
            st.tuples(entropy_strategy, entropy_strategy, entropy_strategy),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_mean_retention_in_bounds(
        self, layer_data: list[tuple[float, float, float]]
    ) -> None:
        """Mean knowledge retention should be in [0, 1]."""
        layer_validations = {}
        for i, (source, target, merged) in enumerate(layer_data):
            layer_validations[f"layer_{i}"] = LayerMergeValidation.compute(
                layer_name=f"layer_{i}",
                source_entropy=source,
                target_entropy=target,
                merged_entropy=merged,
            )

        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations=layer_validations,
        )

        assert 0.0 <= validation.mean_knowledge_retention <= 1.0

    def test_empty_validations_handles_gracefully(self) -> None:
        """Empty layer validations should produce sensible defaults."""
        validation = MergeEntropyValidation.from_layer_validations(
            source_model="source",
            target_model="target",
            layer_validations={},
        )

        assert validation.mean_entropy_ratio == 0.0
        assert validation.max_entropy_ratio == 0.0
        assert validation.mean_knowledge_retention == 1.0
        assert validation.entropy_ratio_std == 0.0


# =============================================================================
# Model Entropy Profile Properties
# =============================================================================


class TestModelEntropyProfileProperties:
    """Property tests for model entropy profiles."""

    @given(
        layer_count=st.integers(min_value=1, max_value=10),
        entropies=st.lists(
            entropy_strategy,
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_from_layer_profiles_computes_mean_correctly(
        self, layer_count: int, entropies: list[float]
    ) -> None:
        """Mean entropy should be average of layer entropies."""
        assume(len(entropies) >= layer_count)
        entropies = entropies[:layer_count]

        layer_profiles = {}
        for i, entropy in enumerate(entropies):
            layer_profiles[f"layer_{i}"] = LayerEntropyProfile(
                layer_name=f"layer_{i}",
                mean_entropy=entropy,
                entropy_variance=0.1,
                phase=Phase.ORDERED,
            )

        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles=layer_profiles,
        )

        expected_mean = sum(entropies) / len(entropies)
        assert profile.mean_entropy == pytest.approx(expected_mean, rel=1e-6)

    @given(
        layer_count=st.integers(min_value=1, max_value=10),
        entropies=st.lists(
            entropy_strategy,
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_variance_is_non_negative(
        self, layer_count: int, entropies: list[float]
    ) -> None:
        """Entropy variance should always be non-negative."""
        assume(len(entropies) >= layer_count)
        entropies = entropies[:layer_count]

        layer_profiles = {}
        for i, entropy in enumerate(entropies):
            layer_profiles[f"layer_{i}"] = LayerEntropyProfile(
                layer_name=f"layer_{i}",
                mean_entropy=entropy,
                entropy_variance=0.1,
                phase=Phase.ORDERED,
            )

        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles=layer_profiles,
        )

        assert profile.entropy_variance >= 0.0

    def test_empty_profiles_handles_gracefully(self) -> None:
        """Empty layer profiles should produce sensible defaults."""
        profile = ModelEntropyProfile.from_layer_profiles(
            model_name="test",
            layer_profiles={},
        )

        assert profile.mean_entropy == 0.0
        assert profile.entropy_variance == 0.0
        assert profile.dominant_phase == Phase.ORDERED
        assert profile.critical_layer_count == 0


# =============================================================================
# Config Factory Properties
# =============================================================================


class TestConfigFactoryProperties:
    """Property tests for configuration factories."""

    @given(
        entropy_samples=st.lists(
            entropy_strategy,
            min_size=10,
            max_size=100,
        ),
        merge_deltas=st.lists(
            st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=100,
        ),
    )
    @settings(max_examples=20)
    def test_from_calibration_produces_valid_thresholds(
        self, entropy_samples: list[float], merge_deltas: list[float]
    ) -> None:
        """Calibration should produce valid ordered thresholds."""
        # Filter out degenerate cases where all samples are identical
        # (which leads to equal percentiles)
        unique_entropies = len(set(entropy_samples))
        assume(unique_entropies >= 3)  # Need variance for percentiles to differ

        config = EntropyMergeConfig.from_calibration_data(
            entropy_samples=entropy_samples,
            merge_deltas=merge_deltas,
        )

        # Thresholds should be ordered: low < high < circuit_breaker
        assert config.entropy_thresholds.low <= config.entropy_thresholds.high
        assert config.entropy_thresholds.high <= config.entropy_thresholds.circuit_breaker

    @given(
        mean=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        std=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_from_entropy_statistics_produces_valid_config(
        self, mean: float, std: float
    ) -> None:
        """Statistics-based factory should produce valid config."""
        # Avoid degenerate case where std >= mean (cv >= 1.0 leads to zero alpha)
        assume(std < mean)

        config = EntropyMergeConfig.from_entropy_statistics(
            entropy_mean=mean,
            entropy_std=std,
        )

        # Thresholds should be ordered
        assert config.entropy_thresholds.low < config.entropy_thresholds.high
        assert config.entropy_thresholds.high < config.entropy_thresholds.circuit_breaker

        # Critical bandwidth should be positive
        assert config.critical_bandwidth > 0

        # All alpha adjustments should be in (0, 1]
        assert 0 < config.phase_adjustments.ordered_alpha <= 1.0
        assert 0 < config.phase_adjustments.critical_alpha <= 1.0
        assert 0 < config.phase_adjustments.disordered_alpha <= 1.0

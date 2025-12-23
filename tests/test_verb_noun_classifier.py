"""Tests for VerbNounDimensionClassifier.

Tests the core stability metrics used to classify embedding dimensions
as Verb (skill/trajectory) or Noun (knowledge/position) for smarter
model merging.

Mathematical invariants tested:
- NounStability is bounded [0, 1]
- VerbVariance is non-negative
- Ratio = VerbVariance / NounStability
- High ratio -> Verb, Low ratio -> Noun
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from modelcypher.core.domain.geometry.verb_noun_classifier import (
    DimensionClass,
    DimensionResult,
    VerbNounClassification,
    VerbNounConfig,
    VerbNounDimensionClassifier,
    modulate_with_confidence,
    summarize_verb_noun_classification,
)


class TestNounStability:
    """Tests for compute_noun_stability."""

    def test_uniform_activations_high_stability(self) -> None:
        """Uniform activations should have high stability (low variance)."""
        # All primes activate dimension identically
        prime_activations = np.ones((10, 64)) * 0.5
        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)

        assert stability.shape == (64,)
        # Uniform -> zero variance -> high stability
        assert np.all(stability > 0.99), f"Expected high stability, got {stability.min()}"

    def test_high_variance_low_stability(self) -> None:
        """High variance activations should have low stability."""
        # Each prime activates differently
        np.random.seed(42)
        prime_activations = np.random.randn(10, 64) * 5  # Large variance
        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)

        assert stability.shape == (64,)
        # High variance -> low stability
        assert np.mean(stability) < 0.8

    def test_stability_bounded_zero_one(self) -> None:
        """Stability should always be in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            activations = np.random.randn(20, 128) * np.random.uniform(0.1, 10)
            stability = VerbNounDimensionClassifier.compute_noun_stability(activations)

            assert np.all(stability >= 0.0), "Stability should be >= 0"
            assert np.all(stability <= 1.0), "Stability should be <= 1"

    def test_single_prime_returns_stability(self) -> None:
        """Single prime should still compute (variance=0)."""
        activations = np.array([[1.0, 2.0, 3.0]])
        stability = VerbNounDimensionClassifier.compute_noun_stability(activations)

        assert stability.shape == (3,)
        # Single sample -> zero variance -> high stability
        assert np.all(stability > 0.9)


class TestVerbVariance:
    """Tests for compute_verb_variance."""

    def test_uniform_gates_zero_variance(self) -> None:
        """Identical gate activations should have zero variance."""
        gate_activations = np.ones((10, 64)) * 2.0
        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)

        assert variance.shape == (64,)
        assert np.allclose(variance, 0.0)

    def test_varying_gates_positive_variance(self) -> None:
        """Varying gate activations should have positive variance."""
        np.random.seed(42)
        gate_activations = np.random.randn(10, 64)
        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)

        assert variance.shape == (64,)
        assert np.all(variance >= 0.0)
        assert np.mean(variance) > 0.5  # Should have substantial variance

    def test_variance_non_negative(self) -> None:
        """Variance should always be non-negative."""
        np.random.seed(42)
        for _ in range(10):
            activations = np.random.randn(15, 128)
            variance = VerbNounDimensionClassifier.compute_verb_variance(activations)
            assert np.all(variance >= 0.0)


class TestClassify:
    """Tests for the main classify method."""

    def test_high_ratio_classifies_as_verb(self) -> None:
        """High VerbVariance / NounStability ratio -> VERB classification."""
        # Create primes with high stability (uniform)
        prime_activations = np.ones((10, 32)) * 1.0

        # Create gates with high variance
        np.random.seed(42)
        gate_activations = np.random.randn(10, 32) * 5

        config = VerbNounConfig(verb_threshold=1.0, noun_threshold=0.3)
        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations, config
        )

        assert result.verb_count > 0
        assert result.verb_fraction > 0.3

    def test_low_ratio_classifies_as_noun(self) -> None:
        """Low VerbVariance / NounStability ratio -> NOUN classification."""
        # Create primes with varying activations (low stability)
        np.random.seed(42)
        prime_activations = np.random.randn(10, 32) * 3

        # Create gates with uniform activations (low variance)
        gate_activations = np.ones((10, 32)) * 0.5

        config = VerbNounConfig(verb_threshold=2.0, noun_threshold=0.5)
        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations, config
        )

        assert result.noun_count > 0

    def test_classification_covers_all_dimensions(self) -> None:
        """All dimensions should be classified."""
        np.random.seed(42)
        hidden_dim = 64
        prime_activations = np.random.randn(10, hidden_dim)
        gate_activations = np.random.randn(10, hidden_dim)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations)

        total = result.verb_count + result.noun_count + result.mixed_count
        assert total == hidden_dim
        assert len(result.dimensions) == hidden_dim

    def test_alpha_vector_matches_classification(self) -> None:
        """Alpha vector should reflect classification."""
        np.random.seed(42)
        prime_activations = np.random.randn(10, 32)
        gate_activations = np.random.randn(10, 32)

        config = VerbNounConfig(
            verb_alpha=0.9,
            noun_alpha=0.1,
            mixed_alpha=0.5,
        )
        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations, config
        )

        for dim_result in result.dimensions:
            actual = result.alpha_vector[dim_result.dimension]
            if dim_result.classification == DimensionClass.VERB:
                assert np.isclose(actual, config.verb_alpha)
            elif dim_result.classification == DimensionClass.NOUN:
                assert np.isclose(actual, config.noun_alpha)
            else:
                assert np.isclose(actual, config.mixed_alpha)


class TestVerbNounConfig:
    """Tests for configuration presets."""

    def test_default_config(self) -> None:
        """Default config should have reasonable values."""
        config = VerbNounConfig.default()
        assert config.verb_threshold > config.noun_threshold
        assert 0 <= config.verb_alpha <= 1
        assert 0 <= config.noun_alpha <= 1
        assert config.epsilon > 0

    def test_conservative_config_narrower_thresholds(self) -> None:
        """Conservative should be less aggressive."""
        default = VerbNounConfig.default()
        conservative = VerbNounConfig.conservative()

        assert conservative.verb_threshold >= default.verb_threshold
        assert conservative.modulation_strength <= default.modulation_strength

    def test_aggressive_config_wider_separation(self) -> None:
        """Aggressive should have more extreme alphas."""
        default = VerbNounConfig.default()
        aggressive = VerbNounConfig.aggressive()

        # More extreme verb/noun alphas
        assert aggressive.verb_alpha >= default.verb_alpha
        assert aggressive.noun_alpha <= default.noun_alpha


class TestModulateWeights:
    """Tests for weight modulation."""

    def test_zero_strength_returns_original(self) -> None:
        """Zero modulation strength should return original weights."""
        np.random.seed(42)
        correlation_weights = np.random.rand(32).astype(np.float32)

        prime_activations = np.random.randn(10, 32)
        gate_activations = np.random.randn(10, 32)
        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=0.0
        )

        assert np.allclose(result, correlation_weights)

    def test_full_strength_returns_classification(self) -> None:
        """Full modulation strength should return classification alphas."""
        np.random.seed(42)
        correlation_weights = np.random.rand(32).astype(np.float32)

        prime_activations = np.random.randn(10, 32)
        gate_activations = np.random.randn(10, 32)
        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=1.0
        )

        assert np.allclose(result, classification.alpha_vector)

    def test_partial_strength_interpolates(self) -> None:
        """Partial strength should interpolate between weights."""
        np.random.seed(42)
        correlation_weights = np.zeros(32, dtype=np.float32)

        prime_activations = np.random.randn(10, 32)
        gate_activations = np.random.randn(10, 32)
        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=0.5
        )

        # Result should be between original (0) and classification alpha
        expected = 0.5 * classification.alpha_vector
        assert np.allclose(result, expected)


class TestModulateWithConfidence:
    """Tests for confidence-weighted modulation."""

    def test_low_confidence_preserves_base(self) -> None:
        """Low confidence dimensions should preserve base alpha."""
        np.random.seed(42)
        base_alpha = np.full(32, 0.5, dtype=np.float32)

        # Create classification with mostly MIXED (low confidence)
        prime_activations = np.random.randn(10, 32) * 0.5
        gate_activations = np.random.randn(10, 32) * 0.5

        config = VerbNounConfig(verb_threshold=10.0, noun_threshold=0.01)
        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations, config
        )

        result = modulate_with_confidence(base_alpha, classification)

        # Most should stay near 0.5 (base)
        assert np.mean(np.abs(result - 0.5)) < 0.2


class TestDimensionResult:
    """Tests for DimensionResult dataclass."""

    def test_dimension_result_creation(self) -> None:
        """Should create valid DimensionResult."""
        result = DimensionResult(
            dimension=5,
            classification=DimensionClass.VERB,
            noun_stability=0.3,
            verb_variance=1.2,
            ratio=4.0,
            alpha=0.8,
        )

        assert result.dimension == 5
        assert result.classification == DimensionClass.VERB
        assert result.alpha == 0.8


class TestVerbNounClassification:
    """Tests for VerbNounClassification dataclass."""

    def test_fractions_sum_to_one(self) -> None:
        """Verb + Noun fractions should not exceed 1."""
        np.random.seed(42)
        prime_activations = np.random.randn(10, 64)
        gate_activations = np.random.randn(10, 64)

        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        # Sum of all fractions should be 1
        total_fraction = (
            result.verb_fraction +
            result.noun_fraction +
            (result.mixed_count / result.total_dimensions)
        )
        assert abs(total_fraction - 1.0) < 1e-6


class TestSummarizeClassification:
    """Tests for summarize_verb_noun_classification."""

    def test_summary_has_expected_keys(self) -> None:
        """Summary should contain all expected statistics."""
        np.random.seed(42)
        prime_activations = np.random.randn(10, 32)
        gate_activations = np.random.randn(10, 32)

        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )
        summary = summarize_verb_noun_classification(classification)

        expected_keys = [
            "total_dimensions", "verb_count", "noun_count", "mixed_count",
            "verb_fraction", "noun_fraction", "mean_noun_stability",
            "mean_verb_variance", "overall_ratio", "mean_alpha", "alpha_std",
        ]

        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    @given(
        n_primes=st.integers(min_value=2, max_value=50),
        n_gates=st.integers(min_value=2, max_value=50),
        hidden_dim=st.integers(min_value=4, max_value=128),
    )
    @settings(max_examples=20)
    def test_stability_always_bounded(
        self, n_primes: int, n_gates: int, hidden_dim: int
    ) -> None:
        """NounStability should always be in [0, 1]."""
        np.random.seed(42)
        prime_activations = np.random.randn(n_primes, hidden_dim)

        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)

        assert np.all(stability >= 0.0)
        assert np.all(stability <= 1.0)

    @given(
        n_gates=st.integers(min_value=2, max_value=50),
        hidden_dim=st.integers(min_value=4, max_value=128),
    )
    @settings(max_examples=20)
    def test_variance_always_non_negative(
        self, n_gates: int, hidden_dim: int
    ) -> None:
        """VerbVariance should always be >= 0."""
        np.random.seed(42)
        gate_activations = np.random.randn(n_gates, hidden_dim)

        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)

        assert np.all(variance >= 0.0)

    @given(
        n_primes=st.integers(min_value=5, max_value=30),
        n_gates=st.integers(min_value=5, max_value=30),
        hidden_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(max_examples=15)
    def test_classification_is_exhaustive(
        self, n_primes: int, n_gates: int, hidden_dim: int
    ) -> None:
        """Every dimension should be classified exactly once."""
        np.random.seed(42)
        prime_activations = np.random.randn(n_primes, hidden_dim)
        gate_activations = np.random.randn(n_gates, hidden_dim)

        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        assert result.verb_count + result.noun_count + result.mixed_count == hidden_dim
        assert len(result.dimensions) == hidden_dim
        assert len(result.alpha_vector) == hidden_dim


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_dimension(self) -> None:
        """Should handle single dimension."""
        prime_activations = np.random.randn(10, 1)
        gate_activations = np.random.randn(10, 1)

        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        assert result.total_dimensions == 1

    def test_single_sample_each(self) -> None:
        """Should handle single sample."""
        prime_activations = np.array([[1.0, 2.0, 3.0]])
        gate_activations = np.array([[4.0, 5.0, 6.0]])

        result = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations
        )

        assert result.total_dimensions == 3

    def test_identical_primes_and_gates(self) -> None:
        """Should handle identical activations."""
        activations = np.random.randn(10, 32)

        result = VerbNounDimensionClassifier.classify(activations, activations)

        assert result.total_dimensions == 32

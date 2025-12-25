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

"""Tests for VerbNounDimensionClassifier.

Tests the core stability metrics used to analyze embedding dimensions
for verb/noun character. The ratio IS the verb/noun-ness:
- High ratio → verb-like (high variance, skill dimension)
- Low ratio → noun-like (high stability, knowledge dimension)

Mathematical invariants tested:
- NounStability is bounded [0, 1]
- VerbVariance is non-negative
- Ratio = VerbVariance / NounStability
- Alpha is geometry-derived from ratio
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.verb_noun_classifier import (
    DimensionResult,
    VerbNounConfig,
    VerbNounDimensionClassifier,
    modulate_with_confidence,
    summarize_verb_noun_classification,
)


def _test_config() -> VerbNounConfig:
    """Create test VerbNounConfig with explicit parameters."""
    return VerbNounConfig.with_parameters(
        epsilon=1e-6,
        alpha_scale=1.5,
        min_activation_variance=1e-8,
    )


class TestNounStability:
    """Tests for compute_noun_stability."""

    def test_uniform_activations_high_stability(self) -> None:
        """Uniform activations should have high stability (low variance)."""
        backend = get_default_backend()
        # All primes activate dimension identically
        prime_activations = backend.ones((10, 64)) * 0.5
        backend.eval(prime_activations)
        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)
        backend.eval(stability)

        stability_np = backend.to_numpy(stability)
        assert stability_np.shape == (64,)
        # Uniform -> zero variance -> high stability
        assert (stability_np > 0.99).all(), f"Expected high stability, got {stability_np.min()}"

    def test_high_variance_low_stability(self) -> None:
        """High variance activations should have low stability."""
        backend = get_default_backend()
        backend.random_seed(42)
        # Each prime activates differently
        prime_activations = backend.random_normal((10, 64)) * 5  # Large variance
        backend.eval(prime_activations)
        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)
        backend.eval(stability)

        stability_np = backend.to_numpy(stability)
        assert stability_np.shape == (64,)
        # High variance -> low stability
        assert stability_np.mean() < 0.8

    def test_stability_bounded_zero_one(self) -> None:
        """Stability should always be in [0, 1]."""
        backend = get_default_backend()
        backend.random_seed(42)
        for _ in range(10):
            scale = float(backend.to_numpy(backend.random_uniform(low=0.1, high=10.0, shape=(1,))))
            activations = backend.random_normal((20, 128)) * scale
            backend.eval(activations)
            stability = VerbNounDimensionClassifier.compute_noun_stability(activations)
            backend.eval(stability)

            stability_np = backend.to_numpy(stability)
            assert (stability_np >= 0.0).all(), "Stability should be >= 0"
            assert (stability_np <= 1.0).all(), "Stability should be <= 1"

    def test_single_prime_returns_stability(self) -> None:
        """Single prime should still compute (variance=0)."""
        backend = get_default_backend()
        activations = backend.array([[1.0, 2.0, 3.0]])
        backend.eval(activations)
        stability = VerbNounDimensionClassifier.compute_noun_stability(activations)
        backend.eval(stability)

        stability_np = backend.to_numpy(stability)
        assert stability_np.shape == (3,)
        # Single sample -> zero variance -> high stability
        assert (stability_np > 0.9).all()


class TestVerbVariance:
    """Tests for compute_verb_variance."""

    def test_uniform_gates_zero_variance(self) -> None:
        """Identical gate activations should have zero variance."""
        backend = get_default_backend()
        gate_activations = backend.ones((10, 64)) * 2.0
        backend.eval(gate_activations)
        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)
        backend.eval(variance)

        variance_np = backend.to_numpy(variance)
        assert variance_np.shape == (64,)
        assert abs(variance_np).max() < 1e-6

    def test_varying_gates_positive_variance(self) -> None:
        """Varying gate activations should have positive variance."""
        backend = get_default_backend()
        backend.random_seed(42)
        gate_activations = backend.random_normal((10, 64))
        backend.eval(gate_activations)
        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)
        backend.eval(variance)

        variance_np = backend.to_numpy(variance)
        assert variance_np.shape == (64,)
        assert (variance_np >= 0.0).all()
        assert variance_np.mean() > 0.5  # Should have substantial variance

    def test_variance_non_negative(self) -> None:
        """Variance should always be non-negative."""
        backend = get_default_backend()
        backend.random_seed(42)
        for _ in range(10):
            activations = backend.random_normal((15, 128))
            backend.eval(activations)
            variance = VerbNounDimensionClassifier.compute_verb_variance(activations)
            backend.eval(variance)
            variance_np = backend.to_numpy(variance)
            assert (variance_np >= 0.0).all()


class TestClassify:
    """Tests for the main classify method."""

    def test_high_ratio_gives_high_alpha(self) -> None:
        """High VerbVariance / NounStability ratio -> high alpha values."""
        backend = get_default_backend()
        backend.random_seed(42)
        # Create primes with high stability (uniform)
        prime_activations = backend.ones((10, 32)) * 1.0
        # Create gates with high variance
        gate_activations = backend.random_normal((10, 32)) * 5
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        # High ratio should produce high alphas (verb-like)
        mean_alpha = float(backend.to_numpy(backend.mean(result.alpha_vector)))
        assert mean_alpha > 0.5, f"Expected high alpha for verb-like dims, got {mean_alpha}"

    def test_low_ratio_gives_low_alpha(self) -> None:
        """Low VerbVariance / NounStability ratio -> low alpha values."""
        backend = get_default_backend()
        backend.random_seed(42)
        # Create primes with varying activations (low stability)
        prime_activations = backend.random_normal((10, 32)) * 3
        # Create gates with uniform activations (low variance)
        gate_activations = backend.ones((10, 32)) * 0.5
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        # Low ratio should produce low alphas (noun-like)
        mean_alpha = float(backend.to_numpy(backend.mean(result.alpha_vector)))
        assert mean_alpha < 0.5, f"Expected low alpha for noun-like dims, got {mean_alpha}"

    def test_analysis_covers_all_dimensions(self) -> None:
        """All dimensions should be analyzed."""
        backend = get_default_backend()
        backend.random_seed(42)
        hidden_dim = 64
        prime_activations = backend.random_normal((10, hidden_dim))
        gate_activations = backend.random_normal((10, hidden_dim))
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        assert result.total_dimensions == hidden_dim
        assert len(result.dimensions) == hidden_dim
        assert len(result.alpha_vector) == hidden_dim

    def test_alpha_vector_matches_dimension_results(self) -> None:
        """Alpha vector should match per-dimension results.

        Alphas are derived from variance ratios using ratio_to_alpha:
        - High ratio → high alpha (verb-like)
        - Low ratio → low alpha (noun-like)
        - ratio ≈ 1 → alpha ≈ 0.5 (mixed)
        """
        backend = get_default_backend()
        backend.random_seed(42)
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        for dim_result in result.dimensions:
            actual = float(backend.to_numpy(result.alpha_vector[dim_result.dimension]))
            # Alpha should be derived from ratio, so verify it matches the DimensionResult
            assert abs(actual - dim_result.alpha) < 1e-6
            # Verify geometric relationship: ratio determines alpha direction
            if dim_result.ratio > 1.0:
                # High ratio → verb-like → should tend toward higher alpha
                assert actual >= 0.5 - 0.05, f"High ratio dim should have alpha >= 0.45, got {actual}"
            elif dim_result.ratio < 1.0:
                # Low ratio → noun-like → should tend toward lower alpha
                assert actual <= 0.5 + 0.05, f"Low ratio dim should have alpha <= 0.55, got {actual}"


class TestVerbNounConfig:
    """Tests for configuration factory."""

    def test_with_parameters_default_values(self) -> None:
        """with_parameters() with defaults should have reasonable values."""
        config = VerbNounConfig.with_parameters()
        assert config.alpha_scale > 0  # Controls steepness of ratio→alpha mapping
        assert config.epsilon > 0
        assert config.min_activation_variance >= 0

    def test_with_parameters_custom_values(self) -> None:
        """with_parameters() should accept custom values."""
        config = VerbNounConfig.with_parameters(
            alpha_scale=2.0,
            epsilon=1e-5,
            min_activation_variance=1e-7,
        )
        assert config.alpha_scale == 2.0
        assert config.epsilon == 1e-5
        assert config.min_activation_variance == 1e-7

    def test_with_parameters_validates_epsilon(self) -> None:
        """with_parameters() should reject invalid epsilon."""
        import pytest

        with pytest.raises(ValueError, match="epsilon must be > 0"):
            VerbNounConfig.with_parameters(epsilon=0)

    def test_with_parameters_validates_alpha_scale(self) -> None:
        """with_parameters() should reject invalid alpha_scale."""
        import pytest

        with pytest.raises(ValueError, match="alpha_scale must be > 0"):
            VerbNounConfig.with_parameters(alpha_scale=-1.0)


class TestModulateWeights:
    """Tests for weight modulation."""

    def test_zero_strength_returns_original(self) -> None:
        """Zero modulation strength should return original weights."""
        backend = get_default_backend()
        backend.random_seed(42)
        correlation_weights = backend.random_uniform(shape=(32,))
        correlation_weights = backend.astype(correlation_weights, "float32")
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(correlation_weights, prime_activations, gate_activations)

        classification = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())
        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=0.0
        )
        backend.eval(result)

        result_np = backend.to_numpy(result)
        corr_np = backend.to_numpy(correlation_weights)
        assert abs(result_np - corr_np).max() < 1e-6

    def test_full_strength_returns_classification(self) -> None:
        """Full modulation strength should return classification alphas."""
        backend = get_default_backend()
        backend.random_seed(42)
        correlation_weights = backend.random_uniform(shape=(32,))
        correlation_weights = backend.astype(correlation_weights, "float32")
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(correlation_weights, prime_activations, gate_activations)

        classification = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())
        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=1.0
        )
        backend.eval(result)

        result_np = backend.to_numpy(result)
        alpha_np = backend.to_numpy(classification.alpha_vector)
        assert abs(result_np - alpha_np).max() < 1e-6

    def test_partial_strength_interpolates(self) -> None:
        """Partial strength should interpolate between weights."""
        backend = get_default_backend()
        backend.random_seed(42)
        correlation_weights = backend.zeros((32,), dtype="float32")
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(correlation_weights, prime_activations, gate_activations)

        classification = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())
        result = VerbNounDimensionClassifier.modulate_weights(
            correlation_weights, classification, strength=0.5
        )
        backend.eval(result)

        # Result should be between original (0) and classification alpha
        result_np = backend.to_numpy(result)
        alpha_np = backend.to_numpy(classification.alpha_vector)
        expected = 0.5 * alpha_np
        assert abs(result_np - expected).max() < 1e-6


class TestModulateWithConfidence:
    """Tests for confidence-weighted modulation."""

    def test_low_extremity_preserves_base(self) -> None:
        """Low ratio-extremity dimensions should preserve base alpha."""
        backend = get_default_backend()
        backend.random_seed(42)
        base_alpha = backend.full((32,), 0.5, dtype="float32")
        # Create balanced classification (ratio ≈ 1.0 = low extremity)
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(base_alpha, prime_activations, gate_activations)

        classification = VerbNounDimensionClassifier.classify(
            prime_activations, gate_activations, _test_config()
        )

        result = modulate_with_confidence(base_alpha, classification)
        backend.eval(result)

        # With balanced ratios (≈1.0), low extremity means less modulation
        # so results should stay relatively close to base
        result_np = backend.to_numpy(result)
        assert abs(result_np - 0.5).mean() < 0.3


class TestDimensionResult:
    """Tests for DimensionResult dataclass."""

    def test_dimension_result_creation(self) -> None:
        """Should create valid DimensionResult with raw measurements."""
        result = DimensionResult(
            dimension=5,
            noun_stability=0.3,
            verb_variance=1.2,
            ratio=4.0,
            alpha=0.8,
        )

        assert result.dimension == 5
        assert result.noun_stability == 0.3
        assert result.verb_variance == 1.2
        assert result.ratio == 4.0
        assert result.alpha == 0.8


class TestVerbNounClassification:
    """Tests for VerbNounClassification dataclass."""

    def test_dimension_count_matches(self) -> None:
        """Dimension count should match input."""
        backend = get_default_backend()
        backend.random_seed(42)
        hidden_dim = 64
        prime_activations = backend.random_normal((10, hidden_dim))
        gate_activations = backend.random_normal((10, hidden_dim))
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        assert result.total_dimensions == hidden_dim
        assert len(result.dimensions) == hidden_dim
        assert len(result.alpha_vector) == hidden_dim


class TestSummarizeClassification:
    """Tests for summarize_verb_noun_classification."""

    def test_summary_has_expected_keys(self) -> None:
        """Summary should contain raw measurement statistics."""
        backend = get_default_backend()
        backend.random_seed(42)
        prime_activations = backend.random_normal((10, 32))
        gate_activations = backend.random_normal((10, 32))
        backend.eval(prime_activations, gate_activations)

        classification = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())
        summary = summarize_verb_noun_classification(classification)

        # Summary returns raw measurements - no categorical counts
        expected_keys = [
            "total_dimensions",
            "mean_noun_stability",
            "mean_verb_variance",
            "overall_ratio",
            "mean_alpha",
            "alpha_std",
        ]

        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

        # Verify no categorical keys
        assert "verb_count" not in summary
        assert "noun_count" not in summary
        assert "mixed_count" not in summary


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    @given(
        n_primes=st.integers(min_value=2, max_value=50),
        n_gates=st.integers(min_value=2, max_value=50),
        hidden_dim=st.integers(min_value=4, max_value=128),
    )
    @settings(max_examples=20)
    def test_stability_always_bounded(self, n_primes: int, n_gates: int, hidden_dim: int) -> None:
        """NounStability should always be in [0, 1]."""
        backend = get_default_backend()
        backend.random_seed(42)
        prime_activations = backend.random_normal((n_primes, hidden_dim))
        backend.eval(prime_activations)

        stability = VerbNounDimensionClassifier.compute_noun_stability(prime_activations)
        backend.eval(stability)

        stability_np = backend.to_numpy(stability)
        assert (stability_np >= 0.0).all()
        assert (stability_np <= 1.0).all()

    @given(
        n_gates=st.integers(min_value=2, max_value=50),
        hidden_dim=st.integers(min_value=4, max_value=128),
    )
    @settings(max_examples=20)
    def test_variance_always_non_negative(self, n_gates: int, hidden_dim: int) -> None:
        """VerbVariance should always be >= 0."""
        backend = get_default_backend()
        backend.random_seed(42)
        gate_activations = backend.random_normal((n_gates, hidden_dim))
        backend.eval(gate_activations)

        variance = VerbNounDimensionClassifier.compute_verb_variance(gate_activations)
        backend.eval(variance)

        variance_np = backend.to_numpy(variance)
        assert (variance_np >= 0.0).all()

    @given(
        n_primes=st.integers(min_value=5, max_value=30),
        n_gates=st.integers(min_value=5, max_value=30),
        hidden_dim=st.integers(min_value=8, max_value=64),
    )
    @settings(max_examples=15)
    def test_analysis_is_exhaustive(
        self, n_primes: int, n_gates: int, hidden_dim: int
    ) -> None:
        """Every dimension should be analyzed with results."""
        backend = get_default_backend()
        backend.random_seed(42)
        prime_activations = backend.random_normal((n_primes, hidden_dim))
        gate_activations = backend.random_normal((n_gates, hidden_dim))
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        assert result.total_dimensions == hidden_dim
        assert len(result.dimensions) == hidden_dim
        assert len(result.alpha_vector) == hidden_dim


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_dimension(self) -> None:
        """Should handle single dimension."""
        backend = get_default_backend()
        backend.random_seed(42)
        prime_activations = backend.random_normal((10, 1))
        gate_activations = backend.random_normal((10, 1))
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        assert result.total_dimensions == 1

    def test_single_sample_each(self) -> None:
        """Should handle single sample."""
        backend = get_default_backend()
        prime_activations = backend.array([[1.0, 2.0, 3.0]])
        gate_activations = backend.array([[4.0, 5.0, 6.0]])
        backend.eval(prime_activations, gate_activations)

        result = VerbNounDimensionClassifier.classify(prime_activations, gate_activations, _test_config())

        assert result.total_dimensions == 3

    def test_identical_primes_and_gates(self) -> None:
        """Should handle identical activations."""
        backend = get_default_backend()
        backend.random_seed(42)
        activations = backend.random_normal((10, 32))
        backend.eval(activations)

        result = VerbNounDimensionClassifier.classify(activations, activations, _test_config())

        assert result.total_dimensions == 32

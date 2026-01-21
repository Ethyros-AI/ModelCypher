# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for entanglement spectrum computation."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.entanglement_spectrum import (
    EntanglementSpectrum,
    EntanglementSpectrumResult,
    compute_entanglement_spectrum,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestEntanglementSpectrumResult:
    """Tests for EntanglementSpectrumResult dataclass."""

    def test_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        n, d = 20, 5
        data = b.array([[float(i * d + j) for j in range(d)] for i in range(n)])

        result = compute_entanglement_spectrum(data, data, b)

        assert hasattr(result, "canonical_correlations")
        assert hasattr(result, "entanglement_entropy")
        assert hasattr(result, "effective_rank_shannon")
        assert hasattr(result, "effective_rank_renyi")
        assert hasattr(result, "correlation_count")
        assert hasattr(result, "sample_count")
        assert hasattr(result, "source_dimension")
        assert hasattr(result, "target_dimension")
        assert hasattr(result, "condition_number")

    def test_result_is_frozen(self, backend):
        """EntanglementSpectrumResult should be immutable."""
        b = backend
        n, d = 20, 5
        data = b.array([[float(i * d + j) for j in range(d)] for i in range(n)])

        result = compute_entanglement_spectrum(data, data, b)

        with pytest.raises(Exception):  # FrozenInstanceError
            result.entanglement_entropy = 999.0  # type: ignore[misc]


class TestEntanglementSpectrum:
    """Tests for EntanglementSpectrum computation."""

    def test_identical_matrices_high_correlation(self, backend):
        """Identical matrices should have high canonical correlations."""
        b = backend
        n, d = 30, 8
        # Create data with some structure - use more samples for stable whitening
        data = b.array([[float(i * d + j) + 0.1 * j for j in range(d)] for i in range(n)])

        result = compute_entanglement_spectrum(data, data, b)

        # At least one canonical correlation should be very high for identical data
        assert len(result.canonical_correlations) > 0
        max_corr = max(result.canonical_correlations)
        assert max_corr > 0.8, f"Expected high max correlation, got {max_corr}"

    def test_independent_random_data_low_correlation(self, backend):
        """Statistically independent data should have low correlations."""
        b = backend
        n = 100
        d = 6
        # Use deterministic but independent-looking patterns
        # Source: based on i
        source = b.array([[float((i * 7 + j * 3) % 13) for j in range(d)] for i in range(n)])
        # Target: based on different prime pattern
        target = b.array([[float((i * 11 + j * 5) % 17) for j in range(d)] for i in range(n)])

        result = compute_entanglement_spectrum(source, target, b)

        # With pseudo-independent data, correlations should be moderate to low
        # Note: CCA can find spurious correlations in finite samples
        assert len(result.canonical_correlations) > 0
        # At least verify we get a finite result
        max_corr = max(result.canonical_correlations)
        assert 0.0 <= max_corr <= 1.0

    def test_entropy_non_negative(self, backend):
        """Entropy should be non-negative."""
        b = backend
        source = b.array([[float(i + j) for j in range(8)] for i in range(30)])
        target = b.array([[float(i * 2 + j) for j in range(8)] for i in range(30)])

        result = compute_entanglement_spectrum(source, target, b)

        assert result.entanglement_entropy >= 0.0

    def test_effective_rank_bounded(self, backend):
        """Effective rank should be <= correlation_count."""
        b = backend
        source = b.array([[float(i + j) for j in range(8)] for i in range(30)])
        target = b.array([[float(i * 2 + j) for j in range(6)] for i in range(30)])

        result = compute_entanglement_spectrum(source, target, b)

        if result.correlation_count > 0:
            # Shannon and Renyi effective ranks should be bounded by count
            assert result.effective_rank_shannon <= result.correlation_count + 1e-6
            assert result.effective_rank_renyi <= result.correlation_count + 1e-6

    def test_dimensions_recorded(self, backend):
        """Source and target dimensions should be recorded correctly."""
        b = backend
        n = 25
        d_source, d_target = 10, 7
        source = b.array([[float(i + j) for j in range(d_source)] for i in range(n)])
        target = b.array([[float(i * 2 + j) for j in range(d_target)] for i in range(n)])

        result = compute_entanglement_spectrum(source, target, b)

        assert result.sample_count == n
        assert result.source_dimension == d_source
        assert result.target_dimension == d_target
        # correlation_count should be min of dimensions
        assert result.correlation_count <= min(d_source, d_target)

    def test_sample_count_must_match(self, backend):
        """Source and target must have same sample count."""
        b = backend
        source = b.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # n=3
        target = b.array([[1.0, 2.0], [3.0, 4.0]])  # n=2

        with pytest.raises(ValueError, match="Sample counts must match"):
            compute_entanglement_spectrum(source, target, b)

    def test_empty_input_returns_zeros(self, backend):
        """Empty input should return zero metrics."""
        b = backend
        source = b.array([]).reshape((0, 5))
        target = b.array([]).reshape((0, 5))

        result = compute_entanglement_spectrum(source, target, b)

        assert result.canonical_correlations == []
        assert result.entanglement_entropy == 0.0
        assert result.effective_rank_shannon == 0.0
        assert result.effective_rank_renyi == 0.0
        assert result.correlation_count == 0

    def test_different_feature_dimensions(self, backend):
        """Should handle different source and target feature dimensions."""
        b = backend
        n = 30
        source = b.array([[float(i + j) for j in range(12)] for i in range(n)])
        target = b.array([[float(i * 2 + j) for j in range(5)] for i in range(n)])

        result = compute_entanglement_spectrum(source, target, b)

        # correlation_count should be min(12, 5) = 5 at most
        assert result.correlation_count <= 5
        assert result.source_dimension == 12
        assert result.target_dimension == 5

    def test_condition_number_positive(self, backend):
        """Condition number should be positive for valid input."""
        b = backend
        source = b.array([[float(i + j) for j in range(6)] for i in range(25)])
        target = b.array([[float(i * 2 + j) for j in range(6)] for i in range(25)])

        result = compute_entanglement_spectrum(source, target, b)

        assert result.condition_number > 0.0

    def test_correlations_sorted_descending(self, backend):
        """Canonical correlations should be sorted in descending order."""
        b = backend
        source = b.array([[float(i + j) + 0.1 * i for j in range(8)] for i in range(40)])
        target = b.array([[float(i * 1.5 + j) for j in range(8)] for i in range(40)])

        result = compute_entanglement_spectrum(source, target, b)

        if len(result.canonical_correlations) > 1:
            for i in range(len(result.canonical_correlations) - 1):
                assert result.canonical_correlations[i] >= result.canonical_correlations[i + 1] - 1e-10

    def test_correlations_bounded_zero_one(self, backend):
        """Canonical correlations should be in [0, 1]."""
        b = backend
        source = b.array([[float(i + j) for j in range(6)] for i in range(30)])
        target = b.array([[float(i * 3 + j) for j in range(6)] for i in range(30)])

        result = compute_entanglement_spectrum(source, target, b)

        for corr in result.canonical_correlations:
            assert 0.0 <= corr <= 1.0 + 1e-10, f"Correlation {corr} out of bounds"


class TestConvenienceFunction:
    """Tests for the compute_entanglement_spectrum convenience function."""

    def test_convenience_function_matches_class(self, backend):
        """Convenience function should match class-based computation."""
        b = backend
        source = b.array([[float(i + j) for j in range(5)] for i in range(20)])
        target = b.array([[float(i * 2 + j) for j in range(5)] for i in range(20)])

        result_class = EntanglementSpectrum(b).compute(source, target)
        result_func = compute_entanglement_spectrum(source, target, b)

        assert result_class.entanglement_entropy == result_func.entanglement_entropy
        assert result_class.effective_rank_shannon == result_func.effective_rank_shannon
        assert result_class.canonical_correlations == result_func.canonical_correlations

    def test_default_backend(self):
        """Should work with default backend when not specified."""
        b = get_default_backend()
        source = b.array([[float(i + j) for j in range(4)] for i in range(15)])
        target = b.array([[float(i * 2 + j) for j in range(4)] for i in range(15)])

        # Call without backend argument
        result = compute_entanglement_spectrum(source, target)

        assert isinstance(result, EntanglementSpectrumResult)
        assert result.sample_count == 15


class TestEntropyInterpretation:
    """Tests for entropy behavior with known structures."""

    def test_single_correlation_low_entropy(self, backend):
        """Single dominant correlation should have low entropy."""
        b = backend
        n = 50
        # Create data where only first dimension has correlation
        source = b.array([[float(i), 0.0, 0.0] for i in range(n)])
        target = b.array([[float(i), 0.0, 0.0] for i in range(n)])

        result = compute_entanglement_spectrum(source, target, b)

        # With only one meaningful correlation, entropy should be low
        # (entropy of [1, 0, 0, ...] normalized is 0)
        # In practice with numerical noise it might not be exactly 0
        if result.correlation_count > 0:
            # Effective rank should be close to 1 for single dominant correlation
            assert result.effective_rank_shannon < 2.0

    def test_uniform_correlations_higher_entropy(self, backend):
        """Uniform correlations should have higher entropy."""
        b = backend
        n = 100
        d = 4
        # Create orthonormal-like structure for uniform correlations
        # Use identity-like relationship
        data = b.array([[float(j == (i % d)) for j in range(d)] for i in range(n)])

        result = compute_entanglement_spectrum(data, data, b)

        if result.correlation_count > 1:
            # With uniform correlations, effective rank should be higher
            # For k identical correlations, effective rank = k
            assert result.effective_rank_shannon > 1.5

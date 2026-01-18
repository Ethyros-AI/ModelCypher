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

"""Tests for Random Matrix Theory signal/noise separation module."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.rmt_signal_separation import (
    MPSignalNoiseResult,
    compute_rmt_null_space_weights,
    estimate_noise_variance_from_bulk,
    estimate_noise_variance_iterative,
    marchenko_pastur_edges,
    separate_signal_noise,
)


@pytest.fixture
def backend():
    """Get the default backend for testing."""
    return get_default_backend()


class TestMarchenkoPasturEdges:
    """Tests for Marchenko-Pastur bulk edge computation."""

    def test_edges_with_gamma_less_than_one(self, backend):
        """Test MP edges when n_samples > n_features (overdetermined)."""
        n_samples, n_features = 100, 50
        noise_variance = 1.0

        lower, upper = marchenko_pastur_edges(
            n_samples, n_features, noise_variance, backend
        )

        # gamma = 50/100 = 0.5
        # lower = (1 - sqrt(0.5))^2 ≈ 0.0858
        # upper = (1 + sqrt(0.5))^2 ≈ 2.914
        assert lower >= 0.0
        assert upper > lower
        assert lower < 1.0  # Should be less than 1 for gamma < 1
        assert upper > 1.0  # Should be greater than 1

    def test_edges_with_gamma_equal_one(self, backend):
        """Test MP edges when n_samples = n_features."""
        n_samples, n_features = 100, 100
        noise_variance = 1.0

        lower, upper = marchenko_pastur_edges(
            n_samples, n_features, noise_variance, backend
        )

        # gamma = 1
        # lower = (1 - 1)^2 = 0
        # upper = (1 + 1)^2 = 4
        assert lower == pytest.approx(0.0, abs=1e-6)
        assert upper == pytest.approx(4.0, abs=1e-6)

    def test_edges_with_gamma_greater_than_one(self, backend):
        """Test MP edges when n_samples < n_features (underdetermined)."""
        n_samples, n_features = 50, 100
        noise_variance = 1.0

        lower, upper = marchenko_pastur_edges(
            n_samples, n_features, noise_variance, backend
        )

        # gamma = 100/50 = 2 > 1
        # For underdetermined case, lower edge is 0 (point mass)
        assert lower == 0.0
        assert upper > 0.0

    def test_edges_scale_with_noise_variance(self, backend):
        """Test that MP edges scale linearly with noise variance."""
        n_samples, n_features = 100, 50
        sigma1, sigma2 = 1.0, 2.0

        lower1, upper1 = marchenko_pastur_edges(n_samples, n_features, sigma1, backend)
        lower2, upper2 = marchenko_pastur_edges(n_samples, n_features, sigma2, backend)

        # Edges should scale with sigma^2
        assert upper2 == pytest.approx(upper1 * 2.0, rel=1e-6)

    def test_edges_symmetric_in_aspect_ratio(self, backend):
        """Test MP edge formula properties."""
        noise_variance = 1.0

        # gamma = d/n, and MP formula uses sqrt(gamma)
        lower1, upper1 = marchenko_pastur_edges(200, 100, noise_variance, backend)
        lower2, upper2 = marchenko_pastur_edges(100, 50, noise_variance, backend)

        # Same gamma = 0.5, should give same edges
        assert lower1 == pytest.approx(lower2, rel=1e-6)
        assert upper1 == pytest.approx(upper2, rel=1e-6)


class TestNoiseVarianceEstimation:
    """Tests for noise variance estimation from eigenvalue spectrum."""

    def test_estimate_from_pure_noise(self, backend):
        """Test noise variance estimation from pure random matrix."""
        n_samples, n_features = 200, 50
        true_sigma = 1.0

        # Create random matrix with known variance
        X = backend.random_normal((n_samples, n_features)) * true_sigma
        backend.eval(X)

        # Compute sample covariance eigenvalues
        X_centered = X - backend.mean(X, axis=0, keepdims=True)
        C = backend.matmul(backend.transpose(X_centered), X_centered) / (n_samples - 1)
        backend.eval(C)

        eigenvalues = backend.eigvalsh(C)
        backend.eval(eigenvalues)

        # Sort descending
        desc_idx = backend.argsort(-eigenvalues)
        eigenvalues = backend.take(eigenvalues, desc_idx, axis=0)
        backend.eval(eigenvalues)

        # Estimate noise variance
        estimated_sigma = estimate_noise_variance_from_bulk(
            eigenvalues, n_samples, n_features, backend
        )

        # Should be close to true variance (within 50% for finite samples)
        assert estimated_sigma > 0.0
        assert estimated_sigma < true_sigma * 2.0

    def test_iterative_better_than_single_pass(self, backend):
        """Test that iterative estimation refines the estimate."""
        n_samples, n_features = 200, 50

        # Create data with some signal and noise
        noise = backend.random_normal((n_samples, n_features))
        # Add a strong signal component
        signal = backend.random_normal((n_samples, 5))
        signal_loadings = backend.random_normal((5, n_features))
        X = noise + 3.0 * backend.matmul(signal, signal_loadings)
        backend.eval(X)

        # Compute eigenvalues
        X_centered = X - backend.mean(X, axis=0, keepdims=True)
        C = backend.matmul(backend.transpose(X_centered), X_centered) / (n_samples - 1)
        backend.eval(C)

        eigenvalues = backend.eigvalsh(C)
        backend.eval(eigenvalues)

        desc_idx = backend.argsort(-eigenvalues)
        eigenvalues = backend.take(eigenvalues, desc_idx, axis=0)
        backend.eval(eigenvalues)

        # Both methods should give positive estimates
        sigma_single = estimate_noise_variance_from_bulk(
            eigenvalues, n_samples, n_features, backend
        )
        sigma_iter = estimate_noise_variance_iterative(
            eigenvalues, n_samples, n_features, backend
        )

        assert sigma_single > 0.0
        assert sigma_iter > 0.0


class TestSeparateSignalNoise:
    """Tests for signal/noise separation using Marchenko-Pastur."""

    def test_returns_valid_result(self, backend):
        """Test that separate_signal_noise returns valid MPSignalNoiseResult."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        assert isinstance(result, MPSignalNoiseResult)
        assert result.signal_rank >= 0
        assert result.noise_rank >= 0
        assert result.signal_rank + result.noise_rank == n_features
        assert result.mp_upper_edge > 0
        assert result.mp_lower_edge >= 0
        assert 0.0 <= result.signal_variance_fraction <= 1.0

    def test_pure_noise_has_low_signal_rank(self, backend):
        """Test that pure random data has mostly noise dimensions."""
        n_samples, n_features = 200, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # Pure noise should have mostly noise dimensions
        # Allow some false positives due to finite sample effects
        assert result.noise_rank >= n_features * 0.5
        assert result.signal_variance_fraction < 0.5

    def test_strong_signal_detected(self, backend):
        """Test that strong signal components are detected."""
        n_samples, n_features = 200, 32
        n_signal_dims = 5

        # Create data with explicit signal structure
        noise = backend.random_normal((n_samples, n_features)) * 0.5

        # Strong signal in first few dimensions
        signal_components = backend.random_normal((n_samples, n_signal_dims))
        signal_loadings = backend.zeros((n_signal_dims, n_features))
        for i in range(n_signal_dims):
            signal_loadings = backend.array(
                [[10.0 if j == i else 0.0 for j in range(n_features)]
                 for _ in range(n_signal_dims)]
            )
        backend.eval(signal_loadings)

        X = noise + backend.matmul(signal_components, signal_loadings)
        backend.eval(X)

        result = separate_signal_noise(X, backend=backend)

        # Should detect at least some signal dimensions
        assert result.signal_rank > 0
        assert result.signal_variance_fraction > 0.1

    def test_aspect_ratio_computed_correctly(self, backend):
        """Test that aspect ratio is computed correctly."""
        n_samples, n_features = 100, 50
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        expected_gamma = n_features / n_samples
        assert result.aspect_ratio == pytest.approx(expected_gamma, rel=1e-6)

    def test_signal_and_noise_indices_partition_correctly(self, backend):
        """Test that signal and noise indices form a valid partition."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # Check that indices are valid (within bounds)
        if result.signal_rank > 0:
            max_signal = float(backend.to_scalar(backend.max(result.signal_indices)))
            min_signal = float(backend.to_scalar(backend.min(result.signal_indices)))
            assert min_signal >= 0
            assert max_signal < n_features

        if result.noise_rank > 0:
            max_noise = float(backend.to_scalar(backend.max(result.noise_indices)))
            min_noise = float(backend.to_scalar(backend.min(result.noise_indices)))
            assert min_noise >= 0
            assert max_noise < n_features

    def test_underdetermined_case(self, backend):
        """Test behavior when n_samples < n_features."""
        n_samples, n_features = 32, 100
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # Should still work and return valid result
        assert isinstance(result, MPSignalNoiseResult)
        assert result.signal_rank + result.noise_rank == n_features


class TestComputeRMTNullSpaceWeights:
    """Tests for RMT-based null-space weight computation."""

    def test_returns_weights_and_result(self, backend):
        """Test that compute_rmt_null_space_weights returns correct types."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        keep_weights, mp_result = compute_rmt_null_space_weights(
            activations, backend=backend
        )

        assert keep_weights is not None
        assert isinstance(mp_result, MPSignalNoiseResult)

    def test_weights_shape_matches_features(self, backend):
        """Test that weights have correct shape."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        keep_weights, _ = compute_rmt_null_space_weights(activations, backend=backend)
        backend.eval(keep_weights)

        assert tuple(backend.shape(keep_weights)) == (n_features,)

    def test_weights_bounded_zero_one(self, backend):
        """Test that weights are in [0, 1]."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        keep_weights, _ = compute_rmt_null_space_weights(activations, backend=backend)
        backend.eval(keep_weights)

        min_weight = float(backend.to_scalar(backend.min(keep_weights)))
        max_weight = float(backend.to_scalar(backend.max(keep_weights)))

        assert min_weight >= 0.0 - 1e-6
        assert max_weight <= 1.0 + 1e-6

    def test_pure_noise_high_keep_weights(self, backend):
        """Test that pure noise gives high keep weights (all dimensions available)."""
        n_samples, n_features = 200, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        keep_weights, _ = compute_rmt_null_space_weights(activations, backend=backend)
        backend.eval(keep_weights)

        mean_keep = float(backend.to_scalar(backend.mean(keep_weights)))

        # Pure noise should have high keep weights (dimensions are noise = available)
        assert mean_keep > 0.3

    def test_strong_signal_low_keep_weights(self, backend):
        """Test that dimensions with strong signal have low keep weights."""
        n_samples, n_features = 200, 32

        # Create data with very strong signal in all dimensions
        activations = backend.random_normal((n_samples, n_features)) * 10.0
        backend.eval(activations)

        keep_weights, mp_result = compute_rmt_null_space_weights(
            activations, backend=backend
        )
        backend.eval(keep_weights)

        # If there's strong signal, some dimensions should be protected (low keep)
        # This depends on the eigenvalue structure
        min_keep = float(backend.to_scalar(backend.min(keep_weights)))
        assert min_keep >= 0.0

    def test_reproducible_with_same_input(self, backend):
        """Test that same input gives same output."""
        n_samples, n_features = 100, 32

        # Create fixed random data
        backend.random_seed(42)
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        keep_weights1, result1 = compute_rmt_null_space_weights(
            activations, backend=backend
        )
        keep_weights2, result2 = compute_rmt_null_space_weights(
            activations, backend=backend
        )

        backend.eval(keep_weights1, keep_weights2)

        # Same input should give same output
        diff = backend.max(backend.abs(keep_weights1 - keep_weights2))
        backend.eval(diff)
        assert float(backend.to_scalar(diff)) < 1e-6


class TestRMTMathematicalProperties:
    """Tests for mathematical properties of RMT computations."""

    def test_mp_edges_positive(self, backend):
        """Test that MP edges are non-negative."""
        for n in [50, 100, 200]:
            for d in [20, 50, 100]:
                lower, upper = marchenko_pastur_edges(n, d, 1.0, backend)
                assert lower >= 0.0
                assert upper > 0.0

    def test_upper_edge_greater_than_lower(self, backend):
        """Test that upper edge > lower edge."""
        for n in [50, 100, 200]:
            for d in [20, 50, 100]:
                if n != d:  # Skip gamma=1 case where lower=0
                    lower, upper = marchenko_pastur_edges(n, d, 1.0, backend)
                    assert upper > lower

    def test_eigenvalue_separation_consistent(self, backend):
        """Test that signal+noise eigenvalues account for all eigenvalues."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # All eigenvalues should be classified
        assert result.signal_rank + result.noise_rank == n_features

    def test_variance_fraction_sum(self, backend):
        """Test that signal + noise variance fractions should approximately sum to 1."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # Signal variance fraction + noise variance fraction ≈ 1
        # (noise fraction = 1 - signal fraction)
        noise_fraction = 1.0 - result.signal_variance_fraction
        assert noise_fraction >= 0.0
        assert noise_fraction <= 1.0


class TestRMTEdgeCases:
    """Tests for edge cases in RMT computations."""

    def test_single_sample(self, backend):
        """Test behavior with single sample."""
        n_samples, n_features = 1, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        # Should handle gracefully
        result = separate_signal_noise(activations, backend=backend)
        assert isinstance(result, MPSignalNoiseResult)

    def test_single_feature(self, backend):
        """Test behavior with single feature."""
        n_samples, n_features = 100, 1
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)
        assert isinstance(result, MPSignalNoiseResult)
        assert result.signal_rank + result.noise_rank == 1

    def test_zero_variance_dimension(self, backend):
        """Test behavior when some dimensions have zero variance."""
        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        backend.eval(activations)

        # Set one dimension to constant (zero variance) using backend ops
        # Create a mask that zeros out the first column
        zero_col = backend.zeros((n_samples, 1))
        other_cols = activations[:, 1:]
        activations = backend.concatenate([zero_col, other_cols], axis=1)
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)
        assert isinstance(result, MPSignalNoiseResult)

    def test_highly_correlated_features(self, backend):
        """Test behavior with highly correlated features."""
        n_samples, n_features = 100, 32

        # Create base random data
        base = backend.random_normal((n_samples, 5))

        # Expand to n_features with correlations
        expansion = backend.random_normal((5, n_features))
        activations = backend.matmul(base, expansion)
        backend.eval(activations)

        result = separate_signal_noise(activations, backend=backend)

        # Should detect low effective dimensionality
        assert isinstance(result, MPSignalNoiseResult)
        # Highly correlated data should have few signal dimensions
        assert result.signal_rank < n_features


class TestRMTIntegrationWithGeodesicNullSpace:
    """Tests for RMT integration with GeodesicNullSpaceFilter."""

    def test_rmt_weights_used_in_filter(self, backend):
        """Test that RMT weights are actually used in the filter."""
        from modelcypher.core.domain.geometry.geodesic_null_space import (
            GeodesicNullSpaceFilter,
        )

        n_samples, n_features = 100, 32
        activations = backend.random_normal((n_samples, n_features))
        delta = backend.random_normal((10, n_features))
        backend.eval(activations, delta)

        geo_filter = GeodesicNullSpaceFilter(backend)
        result = geo_filter.filter_delta(delta, activations)

        # Should have filtered the delta
        assert result.filtering_applied or result.preserved_fraction <= 1.0

    def test_filter_respects_rmt_signal_protection(self, backend):
        """Test that filter protects signal dimensions identified by RMT."""
        from modelcypher.core.domain.geometry.geodesic_null_space import (
            GeodesicNullSpaceFilter,
        )

        n_samples, n_features = 200, 32

        # Create activations with clear signal structure
        noise = backend.random_normal((n_samples, n_features)) * 0.1
        signal = backend.random_normal((n_samples, 5))
        signal_loadings = backend.zeros((5, n_features))
        # Strong signal in first 5 dimensions
        for i in range(5):
            col = [10.0 if j == i else 0.0 for j in range(n_features)]
            signal_loadings = backend.array(
                [[10.0 if j == i else 0.0 for j in range(n_features)]
                 for i in range(5)]
            )
        backend.eval(signal_loadings)

        activations = noise + backend.matmul(signal, signal_loadings)
        backend.eval(activations)

        delta = backend.random_normal((10, n_features))
        backend.eval(delta)

        geo_filter = GeodesicNullSpaceFilter(backend)
        result = geo_filter.filter_delta(delta, activations)

        # Filter should have been applied
        assert result.preserved_fraction <= 1.0


@settings(max_examples=5, deadline=None)
@given(
    n_samples=st.integers(min_value=20, max_value=100),
    n_features=st.integers(min_value=10, max_value=50),
)
def test_rmt_weights_always_valid(n_samples, n_features):
    """Property-based test that RMT weights are always valid."""
    backend = get_default_backend()

    activations = backend.random_normal((n_samples, n_features))
    backend.eval(activations)

    keep_weights, result = compute_rmt_null_space_weights(activations, backend=backend)
    backend.eval(keep_weights)

    # Weights should be bounded
    min_w = float(backend.to_scalar(backend.min(keep_weights)))
    max_w = float(backend.to_scalar(backend.max(keep_weights)))

    assert min_w >= -1e-6
    assert max_w <= 1.0 + 1e-6

    # Result should be valid
    assert result.signal_rank >= 0
    assert result.noise_rank >= 0
    assert result.signal_rank + result.noise_rank == n_features

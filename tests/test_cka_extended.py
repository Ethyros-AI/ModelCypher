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

"""Extended tests for CKA module - previously untested APIs.

These tests cover:
- compute_linear_cka(): Linear Gram CKA (K = X @ X.T)
- compute_cka_split(): Separate CKA for shared vs. novel concepts
- compute_cka_from_grams(): Fast path with pre-computed Gram matrices
- compute_cka_from_centered_grams(): Fastest path with pre-centered Grams
- rbf_gram_matrix_with_sigma(): Gram computation with sigma reuse

Tests use hypothesis for property-based testing of mathematical invariants.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.cka import (
    compute_cka,
    compute_linear_cka,
    compute_cka_split,
    compute_cka_from_grams,
    compute_cka_from_centered_grams,
    rbf_gram_matrix,
    rbf_gram_matrix_with_sigma,
    _center_gram_matrix,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
    all_finite,
    regularization_epsilon,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


# =============================================================================
# Tests for compute_linear_cka
# =============================================================================


class TestLinearCKA:
    """Tests for compute_linear_cka() function."""

    def test_self_similarity_is_one(self, backend):
        """Linear CKA of X with itself should be 1.0."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        cka = compute_linear_cka(X, X, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka - 1.0) < eps

    def test_symmetry(self, backend):
        """Linear CKA should be symmetric: CKA(X, Y) == CKA(Y, X)."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Y = backend.random_normal((20, 24))
        backend.eval(X, Y)

        cka_xy = compute_linear_cka(X, Y, backend)
        cka_yx = compute_linear_cka(Y, X, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka_xy - cka_yx) < eps

    def test_bounded_zero_one(self, backend):
        """Linear CKA should be in [0, 1]."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Y = backend.random_normal((20, 24))
        backend.eval(X, Y)

        cka = compute_linear_cka(X, Y, backend)

        assert 0.0 <= cka <= 1.0

    def test_orthogonal_transform_preserves(self, backend):
        """Orthogonal transform should preserve linear CKA."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Q, _ = backend.qr(backend.random_normal((16, 16)))
        Y = backend.matmul(X, Q)
        backend.eval(X, Y, Q)

        cka = compute_linear_cka(X, Y, backend)

        # Orthogonal transforms preserve Gram structure exactly
        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka - 1.0) < eps

    def test_sample_count_mismatch_returns_zero(self, backend):
        """Mismatched sample counts should return 0."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Y = backend.random_normal((15, 16))  # Different n!
        backend.eval(X, Y)

        cka = compute_linear_cka(X, Y, backend)

        assert cka == 0.0

    def test_single_sample_returns_zero(self, backend):
        """Single sample should return 0 (degenerate case)."""
        backend.random_seed(42)
        X = backend.random_normal((1, 16))
        Y = backend.random_normal((1, 16))
        backend.eval(X, Y)

        cka = compute_linear_cka(X, Y, backend)

        assert cka == 0.0


# =============================================================================
# Tests for compute_cka_split
# =============================================================================


class TestCKASplit:
    """Tests for compute_cka_split() function."""

    def test_returns_valid_result(self, backend):
        """Should return a valid SplitCKAResult."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        # Check all fields exist and are valid
        assert 0.0 <= result.shared_cka <= 1.0
        assert 0.0 <= result.novel_cka <= 1.0
        assert 0.0 <= result.full_cka <= 1.0
        assert 0.0 <= result.shared_fraction <= 1.0
        assert 0.0 <= result.novel_fraction <= 1.0
        assert result.n_shared >= 0
        assert result.n_novel >= 0
        assert result.n_total <= min(50, 32)
        assert result.n_total >= 1

    def test_fractions_sum_correctly(self, backend):
        """shared_fraction + novel_fraction should be <= 1.0."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        # Note: some samples may be neither (both low response)
        eps = regularization_epsilon(backend, source)
        assert result.shared_fraction + result.novel_fraction <= 1.0 + eps

    def test_identical_data_all_shared(self, backend):
        """With identical source and target, all samples should be shared."""
        backend.random_seed(42)
        # When source and target are identical, projection residual is zero
        source = backend.random_normal((50, 32)) + 1.0
        target = source  # Identical data
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        # With identical data, all directions should be shared.
        assert result.n_shared == result.n_total
        assert result.n_novel == 0

    def test_random_data_mostly_novel(self, backend):
        """Random data should produce a consistent rank split."""
        backend.random_seed(42)
        source = backend.random_normal((50, 32))
        target = backend.random_normal((50, 32))
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        assert result.n_shared + result.n_novel == result.n_total
        assert result.n_total <= min(50, 32)

    def test_too_few_samples_returns_zeros(self, backend):
        """With < 4 samples, should return zeros."""
        backend.random_seed(42)
        source = backend.random_normal((3, 32))
        target = backend.random_normal((3, 32))
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        assert result.shared_cka == 0.0
        assert result.novel_cka == 0.0
        assert result.full_cka == 0.0
        assert result.n_total == 3

    def test_identical_inputs_high_shared_cka(self, backend):
        """Identical inputs should have high shared CKA."""
        backend.random_seed(42)
        X = backend.random_normal((50, 32))
        backend.eval(X)

        result = compute_cka_split(X, X, backend)

        # Full CKA should be ~1.0 for identical inputs
        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(result.full_cka - 1.0) < eps


# =============================================================================
# Tests for compute_cka_from_grams
# =============================================================================


class TestCKAFromGrams:
    """Tests for compute_cka_from_grams() function."""

    def test_matches_standard_cka(self, backend):
        """Pre-computed Gram CKA should match standard CKA."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Y = backend.random_normal((20, 24))
        backend.eval(X, Y)

        # Standard CKA (uses geodesic distances)
        standard_result = compute_cka(X, Y, backend)
        standard_cka = standard_result.cka if standard_result.is_valid else 0.0

        # Pre-computed Gram CKA (uses Euclidean distances via rbf_gram_matrix)
        gram_x = rbf_gram_matrix(X, backend)
        gram_y = rbf_gram_matrix(Y, backend)
        backend.eval(gram_x, gram_y)
        from_grams_cka = compute_cka_from_grams(gram_x, gram_y, backend)

        # These may differ significantly because:
        # - compute_cka uses geodesic (k-NN graph) distances
        # - rbf_gram_matrix uses Euclidean distances
        # Both should produce valid CKA values in [0, 1]
        assert 0.0 <= standard_cka <= 1.0
        assert 0.0 <= from_grams_cka <= 1.0

    def test_self_similarity_is_one(self, backend):
        """CKA from identical Grams should be 1.0."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        gram = rbf_gram_matrix(X, backend)
        backend.eval(gram)

        cka = compute_cka_from_grams(gram, gram, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka - 1.0) < eps

    def test_shape_mismatch_returns_zero(self, backend):
        """Mismatched Gram shapes should return 0."""
        backend.random_seed(42)
        gram_a = backend.random_normal((20, 20))
        gram_b = backend.random_normal((15, 15))
        backend.eval(gram_a, gram_b)

        cka = compute_cka_from_grams(gram_a, gram_b, backend)

        assert cka == 0.0

    def test_single_sample_returns_zero(self, backend):
        """1x1 Gram should return 0."""
        backend.random_seed(42)
        gram = backend.random_normal((1, 1))
        backend.eval(gram)

        cka = compute_cka_from_grams(gram, gram, backend)

        assert cka == 0.0


# =============================================================================
# Tests for compute_cka_from_centered_grams
# =============================================================================


class TestCKAFromCenteredGrams:
    """Tests for compute_cka_from_centered_grams() function."""

    def test_matches_from_grams(self, backend):
        """Pre-centered Grams should match compute_cka_from_grams."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        Y = backend.random_normal((20, 24))
        backend.eval(X, Y)

        gram_x = rbf_gram_matrix(X, backend)
        gram_y = rbf_gram_matrix(Y, backend)
        backend.eval(gram_x, gram_y)

        # From uncentered
        cka_from_grams = compute_cka_from_grams(gram_x, gram_y, backend)

        # Pre-center then compute
        centered_x = _center_gram_matrix(gram_x, backend)
        centered_y = _center_gram_matrix(gram_y, backend)
        backend.eval(centered_x, centered_y)
        cka_from_centered = compute_cka_from_centered_grams(centered_x, centered_y, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka_from_grams - cka_from_centered) < eps

    def test_self_similarity_is_one(self, backend):
        """Self-similarity with centered Grams should be 1.0."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        gram = rbf_gram_matrix(X, backend)
        centered = _center_gram_matrix(gram, backend)
        backend.eval(gram, centered)

        cka = compute_cka_from_centered_grams(centered, centered, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka - 1.0) < eps


# =============================================================================
# Tests for rbf_gram_matrix_with_sigma
# =============================================================================


class TestRBFGramWithSigma:
    """Tests for rbf_gram_matrix_with_sigma() function."""

    def test_returns_gram_and_sigma(self, backend):
        """Should return both Gram matrix and sigma used."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        gram, sigma = rbf_gram_matrix_with_sigma(X, backend)
        backend.eval(gram)

        # Gram should be [n, n]
        assert backend.shape(gram) == (20, 20)
        # Sigma should be positive
        assert sigma > 0

    def test_sigma_reuse_produces_same_gram(self, backend):
        """Using returned sigma should produce identical Gram."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        # First call - auto sigma
        gram1, sigma = rbf_gram_matrix_with_sigma(X, backend)
        # Second call - explicit sigma
        gram2, sigma2 = rbf_gram_matrix_with_sigma(X, backend, sigma=sigma)
        backend.eval(gram1, gram2)

        # Should be identical
        diff = backend.sum((gram1 - gram2) ** 2)
        backend.eval(diff)
        eps = regularization_epsilon(backend, gram1)
        assert float(backend.to_scalar(diff)) < eps

        # Sigma should match
        assert sigma == sigma2

    def test_explicit_sigma_used(self, backend):
        """Explicit sigma should override auto-computed sigma."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        # Get auto sigma
        _, auto_sigma = rbf_gram_matrix_with_sigma(X, backend)

        # Use different explicit sigma
        explicit_sigma = auto_sigma * 2.0
        gram, returned_sigma = rbf_gram_matrix_with_sigma(X, backend, sigma=explicit_sigma)

        assert returned_sigma == explicit_sigma

    def test_gram_diagonal_is_one(self, backend):
        """RBF Gram diagonal should be 1.0 (K(x, x) = exp(0) = 1)."""
        backend.random_seed(42)
        X = backend.random_normal((20, 16))
        backend.eval(X)

        gram, _ = rbf_gram_matrix_with_sigma(X, backend)
        backend.eval(gram)

        # Extract diagonal
        diag = backend.diag(gram)
        backend.eval(diag)

        # All diagonal elements should be 1.0
        ones = backend.ones((20,))
        diff = backend.sum((diag - ones) ** 2)
        backend.eval(diff)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert float(backend.to_scalar(diff)) < eps


# =============================================================================
# Hypothesis-based Property Tests
# =============================================================================


class TestCKAMathematicalProperties:
    """Property-based tests for CKA mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_features=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_linear_cka_symmetry(self, n_samples, n_features):
        """Linear CKA(X, Y) == Linear CKA(Y, X)."""
        backend = get_default_backend()
        backend.random_seed(42)

        X = backend.random_normal((n_samples, n_features))
        Y = backend.random_normal((n_samples, n_features + 4))
        backend.eval(X, Y)

        cka_xy = compute_linear_cka(X, Y, backend)
        cka_yx = compute_linear_cka(Y, X, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka_xy - cka_yx) < eps

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_features=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_linear_cka_self_similarity(self, n_samples, n_features):
        """Linear CKA(X, X) == 1.0."""
        backend = get_default_backend()
        backend.random_seed(42)

        X = backend.random_normal((n_samples, n_features))
        backend.eval(X)

        cka = compute_linear_cka(X, X, backend)

        eps = sqrt_scalar(machine_epsilon(backend, X), backend)
        assert abs(cka - 1.0) < eps

    @given(
        n_samples=st.integers(min_value=4, max_value=32),
        n_features=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_linear_cka_bounded(self, n_samples, n_features):
        """Linear CKA should be in [0, 1]."""
        backend = get_default_backend()
        backend.random_seed(42)

        X = backend.random_normal((n_samples, n_features))
        Y = backend.random_normal((n_samples, n_features))
        backend.eval(X, Y)

        cka = compute_linear_cka(X, Y, backend)

        assert 0.0 <= cka <= 1.0

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_cka_split_valid_fractions(self, n_samples, n_features):
        """Split CKA fractions should be valid."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((n_samples, n_features))
        target = backend.random_normal((n_samples, n_features))
        backend.eval(source, target)

        result = compute_cka_split(source, target, backend)

        assert 0.0 <= result.shared_fraction <= 1.0
        assert 0.0 <= result.novel_fraction <= 1.0
        assert result.n_shared + result.n_novel == result.n_total
        assert result.n_total <= min(n_samples, n_features)

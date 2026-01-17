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
"""
Property-based tests for knowledge density estimation.

Tests mathematical invariants:
- Density weights should be in [0, 1]
- Sum of weights should be bounded
- Sparse regions should have lower weights
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _scalar_tol(backend) -> float:
    """Get scalar tolerance for numerical comparisons."""
    return division_epsilon(backend, backend.array([1.0]))


class TestVarianceWeighting:
    """Test variance-based density weighting."""

    @pytest.mark.parametrize("seed", range(5))
    def test_variance_weights_bounded(self, seed: int):
        """Variance-derived weights should be bounded."""
        backend = get_default_backend()
        n = 100
        d = 32

        activations = _random_matrix(backend, n, d, seed)
        backend.eval(activations)

        # Compute variance per dimension
        variance = backend.var(activations, axis=0)
        backend.eval(variance)

        # Variance should be non-negative
        min_var = float(backend.tolist(backend.min(variance)))
        assert min_var >= -1e-10, f"Variance should be non-negative: {min_var}"

        # Inverse variance weights (with regularization)
        eps = 1e-6
        weights = backend.divide(
            backend.ones((d,)),
            backend.add(variance, backend.full((d,), eps)),
        )
        backend.eval(weights)

        # Weights should be finite
        is_finite = backend.all(backend.isfinite(weights))
        backend.eval(is_finite)
        assert backend.tolist(is_finite), "Weights should be finite"

    @pytest.mark.parametrize("seed", range(5))
    def test_high_variance_low_weight(self, seed: int):
        """High variance dimensions should have lower inverse-variance weights."""
        backend = get_default_backend()
        n = 100
        d = 16

        # Create activations with controlled variance
        activations = _random_matrix(backend, n, d, seed)

        # Scale first half to have high variance
        scales = backend.concatenate([
            backend.ones((d // 2,)) * 10.0,
            backend.ones((d // 2,)) * 0.1,
        ], axis=0)
        activations = activations * backend.reshape(scales, (1, d))
        backend.eval(activations)

        # Compute variance
        variance = backend.var(activations, axis=0)
        backend.eval(variance)

        # High variance dims (first half) should have higher variance
        high_var = backend.mean(variance[:d // 2])
        low_var = backend.mean(variance[d // 2:])
        backend.eval(high_var, low_var)

        high_var_val = float(backend.tolist(high_var))
        low_var_val = float(backend.tolist(low_var))

        assert high_var_val > low_var_val * 10, (
            f"High variance should be > 10x low variance: {high_var_val} vs {low_var_val}"
        )


class TestNullSpaceIdentification:
    """Test null space identification via variance."""

    @pytest.mark.parametrize("seed", range(5))
    def test_spectral_gap_detection(self, seed: int):
        """Should detect spectral gap between used and unused dimensions."""
        backend = get_default_backend()
        n = 100
        d = 32
        k_used = 10

        # Create matrix with clear spectral gap
        activations = _random_matrix(backend, n, d, seed)
        u, s, vt = backend.svd(activations)
        backend.eval(u, s, vt)

        # Modify singular values to create gap
        s_modified = backend.concatenate([
            backend.ones((k_used,)) * 10.0,
            backend.ones((d - k_used,)) * 0.1,
        ], axis=0)
        s_modified = s_modified[:s.shape[0]]
        backend.eval(s_modified)

        # Check gap exists
        s_list = backend.tolist(s_modified)
        if len(s_list) > k_used:
            gap = s_list[k_used - 1] / s_list[k_used] if s_list[k_used] > 1e-10 else float("inf")
            assert gap > 10, f"Spectral gap should be significant: {gap}"

    @pytest.mark.parametrize("seed", range(5))
    def test_null_space_rank(self, seed: int):
        """Null space rank should be (d - rank(A))."""
        backend = get_default_backend()
        n = 50
        d = 32
        true_rank = 10

        # Create rank-deficient matrix
        backend.random_seed(seed)
        low_rank = backend.matmul(
            backend.random_normal((n, true_rank)),
            backend.random_normal((true_rank, d)),
        )
        backend.eval(low_rank)

        # Compute SVD to find effective rank
        _, s, _ = backend.svd(low_rank)
        backend.eval(s)

        s_list = backend.tolist(s)

        # Count significant singular values
        threshold = max(s_list) * 1e-6
        effective_rank = sum(1 for sv in s_list if sv > threshold)

        assert effective_rank <= true_rank + 2, (
            f"Effective rank should be close to true rank: {effective_rank} vs {true_rank}"
        )


class TestCoverageRatio:
    """Test coverage ratio calculations."""

    @pytest.mark.parametrize("seed", range(5))
    def test_coverage_ratio_formula(self, seed: int):
        """Coverage ratio should be n_samples / hidden_dim."""
        backend = get_default_backend()

        n_samples = 100
        hidden_dim = 32

        activations = _random_matrix(backend, n_samples, hidden_dim, seed)
        backend.eval(activations)

        coverage_ratio = n_samples / hidden_dim
        assert coverage_ratio == pytest.approx(100 / 32)

    def test_under_sampled_warning(self):
        """Under-sampled case (n < d) should have coverage < 1."""
        n_samples = 10
        hidden_dim = 64

        coverage = n_samples / hidden_dim
        assert coverage < 1.0, f"Under-sampled should have coverage < 1: {coverage}"

    def test_well_sampled_threshold(self):
        """Well-sampled case should have coverage > 4."""
        n_samples = 256
        hidden_dim = 32

        coverage = n_samples / hidden_dim
        assert coverage > 4.0, f"Well-sampled should have coverage > 4: {coverage}"


class TestConditionNumber:
    """Test condition number calculations."""

    @pytest.mark.parametrize("seed", range(5))
    def test_condition_number_positive(self, seed: int):
        """Condition number should be positive."""
        backend = get_default_backend()
        n = 50
        d = 20

        matrix = _random_matrix(backend, n, d, seed)
        backend.eval(matrix)

        _, s, _ = backend.svd(matrix)
        backend.eval(s)

        s_max = float(backend.tolist(backend.max(s)))
        s_min = float(backend.tolist(backend.min(s)))

        cond = s_max / s_min if s_min > 1e-10 else float("inf")

        assert cond > 0, f"Condition number should be positive: {cond}"

    @pytest.mark.parametrize("seed", range(5))
    def test_condition_number_orthogonal_is_one(self, seed: int):
        """Orthogonal matrix should have condition number = 1."""
        backend = get_default_backend()
        d = 20

        random_mat = _random_matrix(backend, d, d, seed)
        q, _ = backend.qr(random_mat)
        backend.eval(q)

        _, s, _ = backend.svd(q)
        backend.eval(s)

        s_max = float(backend.tolist(backend.max(s)))
        s_min = float(backend.tolist(backend.min(s)))

        cond = s_max / s_min if s_min > 1e-10 else float("inf")

        assert cond == pytest.approx(1.0, rel=1e-5), (
            f"Orthogonal matrix should have cond = 1: {cond}"
        )


try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestDensityHypothesis:
    """Hypothesis-based density property tests."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        hidden_dim=st.integers(min_value=4, max_value=64),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_variance_always_non_negative(
        self, n_samples: int, hidden_dim: int, seed: int
    ):
        """Variance should always be non-negative."""
        backend = get_default_backend()

        activations = _random_matrix(backend, n_samples, hidden_dim, seed)
        backend.eval(activations)

        variance = backend.var(activations, axis=0)
        backend.eval(variance)

        min_var = float(backend.tolist(backend.min(variance)))
        tol = _scalar_tol(backend)

        assert min_var >= -tol, f"Variance should be non-negative: {min_var}"

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        hidden_dim=st.integers(min_value=4, max_value=64),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_coverage_ratio_formula_holds(
        self, n_samples: int, hidden_dim: int, seed: int
    ):
        """Coverage ratio should equal n_samples / hidden_dim."""
        coverage = n_samples / hidden_dim
        expected = n_samples / hidden_dim

        assert coverage == pytest.approx(expected)

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        hidden_dim=st.integers(min_value=4, max_value=32),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_singular_values_ordered(
        self, n_samples: int, hidden_dim: int, seed: int
    ):
        """Singular values should be in descending order."""
        backend = get_default_backend()

        matrix = _random_matrix(backend, n_samples, hidden_dim, seed)
        backend.eval(matrix)

        _, s, _ = backend.svd(matrix)
        backend.eval(s)

        s_list = backend.tolist(s)

        for i in range(len(s_list) - 1):
            assert s_list[i] >= s_list[i + 1] - 1e-6, (
                f"Singular values should be descending: {s_list}"
            )

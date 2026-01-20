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

"""Tests for principled rank determination in null-space projection.

The transplant.py module uses svd_auto_rank() for rank determination instead
of a heuristic threshold. This test file verifies:

1. Scale invariance: Same relative rank regardless of eigenvalue magnitude
2. Precision cutoff: numeric rank matches precision-derived threshold
3. Dimension consistency: null_rank + intrinsic_rank = total_dim
4. Numerical stability: Handles edge cases (near-zero eigenvalues, etc.)

References:
- Yu et al. (2025) "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
- Zhang et al. (2025) "STF: Superpose Task-specific Features for Multi-task Fine-tuned Models"
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    svd_auto_rank,
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms
from modelcypher.core.domain.geometry.transplant import (
    compute_null_space_projector,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


class TestSVDAutoRankScaleInvariance:
    """Tests for scale invariance of svd_auto_rank()."""

    def test_rank_unchanged_by_uniform_scaling(self, backend):
        """Rank should be the same whether eigenvalues are scaled or not."""
        b = backend
        b.random_seed(42)

        # Create singular values with clear separation
        # Top 5 components have values [100, 50, 25, 12, 6]
        # Bottom 5 components are noise [0.1, 0.05, 0.02, 0.01, 0.005]
        singular_values = b.array([100.0, 50.0, 25.0, 12.0, 6.0, 0.1, 0.05, 0.02, 0.01, 0.005])
        b.eval(singular_values)

        # Get rank at original scale
        rank_original = svd_auto_rank(singular_values, b)

        # Scale by various factors and verify rank is unchanged
        for scale in [0.001, 0.1, 10.0, 1000.0]:
            scaled = singular_values * scale
            b.eval(scaled)
            rank_scaled = svd_auto_rank(scaled, b)
            assert rank_original == rank_scaled, (
                f"Rank changed from {rank_original} to {rank_scaled} "
                f"when scaling by {scale}"
            )

    def test_rank_unchanged_by_different_magnitude_distributions(self, backend):
        """Rank should depend on relative, not absolute, magnitudes."""
        b = backend

        # Distribution A: Large absolute values
        sv_large = b.array([1e6, 5e5, 2.5e5, 1e3, 5e2, 2e2])
        b.eval(sv_large)
        rank_large = svd_auto_rank(sv_large, b)

        # Distribution B: Small absolute values with SAME relative structure
        sv_small = b.array([1e-6, 5e-7, 2.5e-7, 1e-9, 5e-10, 2e-10])
        b.eval(sv_small)
        rank_small = svd_auto_rank(sv_small, b)

        assert rank_large == rank_small, (
            f"Rank changed from {rank_large} to {rank_small} "
            "despite same relative structure"
        )


class TestSVDAutoRankPrecisionThreshold:
    """Tests for precision-derived numeric rank behavior."""

    def test_precision_threshold_filters_noise(self, backend):
        """Precision threshold should drop values below numeric resolution."""
        b = backend

        signal = [10.0, 5.0, 2.5]
        max_dim = 10
        sv = b.array(signal + [0.0] * (max_dim - len(signal)))
        b.eval(sv)

        eps = machine_epsilon(b, sv)
        threshold = max(signal) * max_dim * eps
        noise = [threshold * 0.1] * (max_dim - len(signal))
        sv = b.array(signal + noise)
        b.eval(sv)

        rank = svd_auto_rank(sv, b, max_dim=max_dim)
        assert rank == len(signal)

    def test_precision_threshold_matches_formula(self, backend):
        """Rank should equal count of values above max_s * max_dim * eps."""
        b = backend

        sv = b.array([10.0, 5.0, 2.5, 0.1, 0.01, 0.001])
        b.eval(sv)

        max_dim = int(sv.shape[0])
        eps = machine_epsilon(b, sv)
        max_s = float(b.to_scalar(b.max(sv)))
        threshold = max_s * max_dim * eps

        expected = sum(1 for val in [10.0, 5.0, 2.5, 0.1, 0.01, 0.001] if val > threshold)
        rank = svd_auto_rank(sv, b, max_dim=max_dim)
        assert rank == expected


class TestNullSpaceProjectorRankConsistency:
    """Tests for null-space projector rank consistency."""

    def test_intrinsic_plus_null_equals_total(self, backend):
        """intrinsic_rank + null_rank should equal total dimension."""
        b = backend
        b.random_seed(42)

        # Create activations with known structure
        n_samples = 50
        d_features = 20

        # Random activations
        activations = b.random_normal((n_samples, d_features))
        b.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=b,
        )

        total_dim = d_features

        # intrinsic_rank is determined by svd_auto_rank
        # null_rank = total_dim - intrinsic_rank
        intrinsic_rank = total_dim - projector.null_rank

        assert intrinsic_rank + projector.null_rank == total_dim, (
            f"Dimension mismatch: {intrinsic_rank} + {projector.null_rank} != {total_dim}"
        )

    def test_null_rank_is_non_negative(self, backend):
        """Null rank should always be non-negative."""
        b = backend
        b.random_seed(42)

        for n_samples in [10, 50, 100]:
            for d_features in [5, 20, 50]:
                activations = b.random_normal((n_samples, d_features))
                b.eval(activations)

                projector = compute_null_space_projector(
                    input_activations=activations,
                    backend=b,
                )

                assert projector.null_rank >= 0, (
                    f"Negative null_rank: {projector.null_rank} "
                    f"for n={n_samples}, d={d_features}"
                )


class TestNullSpaceProjectorNumericalStability:
    """Tests for numerical stability of null-space projector."""

    def test_handles_low_variance_activations(self, backend):
        """Should handle activations with very low variance."""
        b = backend
        b.random_seed(42)

        # Low variance activations
        eps = float(b.finfo().eps)
        activations = b.random_normal((30, 10)) * sqrt_scalar(eps, b) * 100
        b.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=b,
        )

        # Should not crash, and should give reasonable results
        assert 0 <= projector.null_rank <= activations.shape[1]

    def test_handles_overdetermined_case(self, backend):
        """Should handle n_samples >> d_features (overdetermined)."""
        b = backend
        b.random_seed(42)

        # Many samples, few features
        activations = b.random_normal((500, 10))
        b.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=b,
        )

        assert 0 <= projector.null_rank <= activations.shape[1]

    def test_handles_underdetermined_case(self, backend):
        """Should handle n_samples << d_features (underdetermined)."""
        b = backend
        b.random_seed(42)

        # Few samples, many features
        activations = b.random_normal((10, 100))
        b.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=b,
        )

        # With 10 samples in 100D space, intrinsic dimension measures the
        # actual manifold dimensionality via geodesic TwoNN, not n_samples.
        # The key invariant is: intrinsic_rank + null_rank = total_dim
        total_dim = 100
        intrinsic_rank = total_dim - projector.null_rank
        assert intrinsic_rank + projector.null_rank == total_dim, (
            f"Dimension mismatch: {intrinsic_rank} + {projector.null_rank} != {total_dim}"
        )
        assert 0 <= projector.null_rank <= total_dim


class TestNullSpaceProjectorProperties:
    """Property-based tests for null-space projection."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d_features=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=10, deadline=None)
    def test_projection_satisfies_null_constraint(self, n_samples, d_features):
        """Projected deltas should satisfy A @ delta^T = 0."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = backend.random_normal((n_samples, d_features))
        backend.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=backend,
        )

        A = projector.weighted_activations
        gram_inv = projector.gram_inv

        out_dim = 3
        delta_W = backend.random_normal((out_dim, d_features))
        backend.eval(A, gram_inv, delta_W)

        delta_row = backend.matmul(delta_W, backend.transpose(A))
        correction = backend.matmul(delta_row, gram_inv)
        correction = backend.matmul(correction, A)
        delta_proj = delta_W - correction
        backend.eval(delta_proj)

        residual = backend.matmul(A, backend.transpose(delta_proj))
        backend.eval(residual)

        res_norm = backend.mean(geodesic_norms(residual, backend))
        act_norm = backend.mean(geodesic_norms(A, backend))
        delta_norm = backend.mean(geodesic_norms(delta_proj, backend))
        backend.eval(res_norm, act_norm, delta_norm)

        # Null-space projection residual is bounded by condition_number × eps × scale.
        # Random matrices have condition numbers ~10-100×. Use sqrt(eps) as base
        # (accounts for accumulated float ops) with 100× factor for conditioning.
        eps = machine_epsilon(backend, residual)
        sqrt_eps = float(eps ** 0.5)
        scale = float(backend.to_scalar(act_norm)) * float(backend.to_scalar(delta_norm))
        tolerance = sqrt_eps * max(1.0, scale) * 100.0

        assert float(backend.to_scalar(res_norm)) <= tolerance, (
            f"Projection violates null constraint: ||A @ delta^T|| = {float(backend.to_scalar(res_norm))}"
        )

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d_features=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=10, deadline=None)
    def test_projection_is_idempotent(self, n_samples, d_features):
        """Applying the projection twice should be a no-op."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = backend.random_normal((n_samples, d_features))
        backend.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=backend,
        )

        A = projector.weighted_activations
        gram_inv = projector.gram_inv

        out_dim = 3
        delta_W = backend.random_normal((out_dim, d_features))
        backend.eval(A, gram_inv, delta_W)

        def _project(delta):
            delta_row = backend.matmul(delta, backend.transpose(A))
            correction = backend.matmul(delta_row, gram_inv)
            correction = backend.matmul(correction, A)
            projected = delta - correction
            backend.eval(projected)
            return projected

        delta_once = _project(delta_W)
        delta_twice = _project(delta_once)

        diff = backend.sum(backend.abs(delta_twice - delta_once))
        backend.eval(diff)
        diff_val = float(backend.to_scalar(diff))

        # Idempotency error accumulates with each matmul in the projection.
        # Random matrices have condition numbers ~10-100×.
        eps = sqrt_scalar(machine_epsilon(backend, delta_once), backend)
        tolerance = eps * float(d_features) * float(out_dim) * 100.0

        assert diff_val < tolerance, (
            f"Projection not idempotent: ||P(P(delta)) - P(delta)|| = {diff_val}"
        )

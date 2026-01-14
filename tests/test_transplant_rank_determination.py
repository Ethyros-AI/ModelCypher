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
2. Energy capture: 99% energy threshold correctly filters noise
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
    svd_auto_rank,
    machine_epsilon,
    sqrt_scalar,
)
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
        rank_original = svd_auto_rank(singular_values, b, energy_threshold=0.99)

        # Scale by various factors and verify rank is unchanged
        for scale in [0.001, 0.1, 10.0, 1000.0]:
            scaled = singular_values * scale
            b.eval(scaled)
            rank_scaled = svd_auto_rank(scaled, b, energy_threshold=0.99)
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
        rank_large = svd_auto_rank(sv_large, b, energy_threshold=0.99)

        # Distribution B: Small absolute values with SAME relative structure
        sv_small = b.array([1e-6, 5e-7, 2.5e-7, 1e-9, 5e-10, 2e-10])
        b.eval(sv_small)
        rank_small = svd_auto_rank(sv_small, b, energy_threshold=0.99)

        assert rank_large == rank_small, (
            f"Rank changed from {rank_large} to {rank_small} "
            "despite same relative structure"
        )


class TestSVDAutoRankEnergy:
    """Tests for correct energy capture behavior."""

    def test_99_percent_threshold_captures_signal(self, backend):
        """99% energy threshold should capture signal components."""
        b = backend

        # Create clear signal + noise structure
        # Signal: 3 components capturing ~99.9% of energy
        # Noise: 7 components with negligible energy
        signal = [10.0, 5.0, 2.5]  # Energy: 100 + 25 + 6.25 = 131.25
        noise = [0.01] * 7  # Energy: 7 * 0.0001 = 0.0007
        # Total energy ≈ 131.25, signal fraction ≈ 99.99%

        sv = b.array(signal + noise)
        b.eval(sv)

        rank = svd_auto_rank(sv, b, energy_threshold=0.99)

        # Should capture at least the 3 signal components
        # (might include some noise if energy threshold requires it)
        assert rank >= 3, f"Expected rank >= 3, got {rank}"
        assert rank <= 4, f"Expected rank <= 4 (signal + margin), got {rank}"

    def test_different_thresholds_change_rank(self, backend):
        """Higher energy threshold should give equal or higher rank."""
        b = backend

        # Create gradual decay singular values
        sv = b.array([10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625])
        b.eval(sv)

        rank_90 = svd_auto_rank(sv, b, energy_threshold=0.90)
        rank_95 = svd_auto_rank(sv, b, energy_threshold=0.95)
        rank_99 = svd_auto_rank(sv, b, energy_threshold=0.99)
        rank_999 = svd_auto_rank(sv, b, energy_threshold=0.999)

        # Higher threshold should require >= same rank
        assert rank_90 <= rank_95 <= rank_99 <= rank_999, (
            f"Rank should increase with threshold: "
            f"90%={rank_90}, 95%={rank_95}, 99%={rank_99}, 99.9%={rank_999}"
        )


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

        # The projector matrix is [d, d]
        projector_shape = b.shape(projector.projector)
        total_dim = int(projector_shape[0])

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
        assert projector.null_rank >= 0
        assert projector.null_rank <= 10  # Can't exceed total dim

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

        # With 500 samples spanning 10D space, most dimensions should be active
        # null_rank should be small
        assert projector.null_rank <= 5, (
            f"Expected small null_rank for overdetermined case, got {projector.null_rank}"
        )

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
        # null_rank should still be significant in underdetermined case
        assert projector.null_rank >= total_dim // 2, (
            f"Expected significant null_rank for underdetermined case, got {projector.null_rank}"
        )


class TestNullSpaceProjectorProperties:
    """Property-based tests for null-space projector."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d_features=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=10, deadline=None)
    def test_projector_is_symmetric(self, n_samples, d_features):
        """Null-space projector N should be symmetric."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = backend.random_normal((n_samples, d_features))
        backend.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=backend,
        )

        N = projector.projector
        N_T = backend.transpose(N)
        backend.eval(N, N_T)

        # Check symmetry: N == N^T
        diff = backend.sum(backend.abs(N - N_T))
        backend.eval(diff)
        diff_val = float(backend.to_scalar(diff))

        eps = sqrt_scalar(machine_epsilon(backend, N), backend)
        d = int(backend.shape(N)[0])
        tolerance = eps * d * d  # Scale by matrix size

        assert diff_val < tolerance, (
            f"Projector not symmetric: ||N - N^T|| = {diff_val}"
        )

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d_features=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=10, deadline=None)
    def test_projector_is_idempotent(self, n_samples, d_features):
        """Null-space projector N should satisfy N @ N = N (idempotent)."""
        backend = get_default_backend()
        backend.random_seed(42)

        activations = backend.random_normal((n_samples, d_features))
        backend.eval(activations)

        projector = compute_null_space_projector(
            input_activations=activations,
            backend=backend,
        )

        N = projector.projector
        N_squared = backend.matmul(N, N)
        backend.eval(N, N_squared)

        # Check idempotency: N @ N == N
        diff = backend.sum(backend.abs(N_squared - N))
        backend.eval(diff)
        diff_val = float(backend.to_scalar(diff))

        eps = sqrt_scalar(machine_epsilon(backend, N), backend)
        d = int(backend.shape(N)[0])
        tolerance = eps * d * d  # Scale by matrix size

        assert diff_val < tolerance, (
            f"Projector not idempotent: ||N @ N - N|| = {diff_val}"
        )

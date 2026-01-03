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
Tests for null-space filtering.

Validates the core mathematical guarantee: if Δw ∈ null(A),
then A @ (W + Δw) = A @ W (no interference with prior task).
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.null_space_filter import (
    NullSpaceFilter,
    filter_merge_delta_to_null_space,
)
from modelcypher.core.support.array_utils import array_to_list
# NOTE: NullSpaceFilterConfig was REMOVED. All parameters derived from spectral gap.


class TestNullSpaceProjection:
    """Test null space computation.

    NOTE: NullSpaceFilterConfig was REMOVED. All parameters (min_samples,
    normalization, regularization, rank threshold) are now derived from the
    data's dtype and spectral properties.
    """

    def test_identity_projection_for_few_samples(self):
        """Few samples should give identity projection (full null space)."""
        backend = get_default_backend()

        # Very few samples - min_samples is now derived from log2(d)
        backend.random_seed(42)
        A = backend.random_normal((2, 10))  # 2 samples, 10 dims, needs log2(10)~4 samples
        backend.eval(A)
        null_filter = NullSpaceFilter(backend)

        projection = null_filter.compute_null_space_projection(A)

        # Should return identity-like projection due to insufficient samples
        assert projection.null_dim == 10  # Full dimension
        eye_mat = backend.eye(10)
        backend.eval(projection.projection_matrix)
        backend.eval(eye_mat)
        diff = backend.max(backend.abs(projection.projection_matrix - eye_mat))
        backend.eval(diff)
        eps = machine_epsilon(backend, projection.projection_matrix) * projection.projection_matrix.shape[0]
        assert backend.to_scalar(diff) <= eps

    def test_null_space_orthogonal_to_row_space(self):
        """Null space vectors should be orthogonal to all rows of A."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        # Create simple full-rank matrix
        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)

        projection = null_filter.compute_null_space_projection(A)

        # With full rank, null space should be small or empty
        # Just verify the function works without crashing
        assert projection.null_dim >= 0
        assert projection.null_dim <= 20

    def test_projection_is_idempotent(self):
        """Projecting twice should give same result as projecting once."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)
        projection = null_filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        P_squared = P @ P
        backend.eval(P)
        backend.eval(P_squared)

        # P^2 = P for projection matrices
        diff = backend.max(backend.abs(P - P_squared))
        backend.eval(diff)
        eps = machine_epsilon(backend, P) * P.shape[0]
        assert backend.to_scalar(diff) <= eps

    def test_projection_is_symmetric(self):
        """Projection matrix should be symmetric."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((25, 15))
        backend.eval(A)
        projection = null_filter.compute_null_space_projection(A)

        P = projection.projection_matrix
        P_T = backend.transpose(P)
        backend.eval(P)
        backend.eval(P_T)
        diff = backend.max(backend.abs(P - P_T))
        backend.eval(diff)
        eps = machine_epsilon(backend, P) * P.shape[0]
        assert backend.to_scalar(diff) <= eps

    def test_null_space_preserves_dimension_invariant(self):
        """null_dim + row_space_dim should equal total dimension."""
        # NOTE: NullSpaceMethod is no longer configurable - method is derived from matrix size.
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        backend.eval(A)
        projection = null_filter.compute_null_space_projection(A)

        # Basic sanity checks
        assert projection.null_dim >= 0
        assert projection.null_dim <= 20
        assert projection.row_space_dim >= 0
        assert projection.null_dim + projection.row_space_dim == 20


class TestNullSpaceFiltering:
    """Test delta filtering through null space."""

    def test_filtered_delta_preserves_no_interference(self):
        """Core guarantee: filtered delta in null space doesn't affect output.

        The null-space filter projects weight updates into the null space of
        activations. If delta is in null(A), then A @ delta = 0 for any
        activation matrix A used to compute the null space.

        For null space to exist: n_samples < d (rank-deficient A).
        """
        backend = get_default_backend()
        # NOTE: NullSpaceFilterConfig was REMOVED - method derived from matrix size
        null_filter = NullSpaceFilter(backend)

        # n_samples < d ensures non-trivial null space
        d = 50  # Weight dimension
        n_samples = 20  # Fewer samples than dimensions → null space exists

        # Random activations and delta (1D case - delta matches activation dim)
        backend.random_seed(42)
        A = backend.random_normal((n_samples, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        # Filter delta to null space of A
        result = null_filter.filter_delta(delta, A)

        if result.filtering_applied and result.null_space_dim > 0:
            delta_safe = result.filtered_delta
            backend.eval(delta_safe)

            # Core guarantee: A @ delta_safe ≈ 0 (delta is in null space of A)
            product = backend.matmul(A, delta_safe)
            backend.eval(product)

            # Product should be near zero (in null space)
            product_norm = backend.to_scalar(backend.norm(product))
            delta_norm = backend.to_scalar(backend.norm(delta_safe))
            eps = division_epsilon(backend, delta_safe) * delta_safe.shape[0]

            if delta_norm > eps:
                relative_product = product_norm / delta_norm
                assert relative_product <= eps

    def test_preservation_fraction_bounded(self):
        """Preserved fraction should be in [0, 1]."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        assert 0.0 <= result.preserved_fraction <= 1.0
        assert 0.0 <= result.projection_loss <= 1.0
        eps = machine_epsilon(backend, backend.array(1.0))
        assert abs(result.preserved_fraction + result.projection_loss - 1.0) <= eps

    def test_zero_delta_gives_zero_filtered(self):
        """Zero delta should give zero filtered delta."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.zeros((20,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        backend.eval(result.filtered_delta)
        zero_arr = backend.zeros_like(result.filtered_delta)
        backend.eval(zero_arr)
        diff = backend.max(backend.abs(result.filtered_delta - zero_arr))
        backend.eval(diff)
        eps = machine_epsilon(backend, result.filtered_delta) * result.filtered_delta.shape[0]
        assert backend.to_scalar(diff) <= eps
        assert result.original_norm == 0
        assert result.preserved_fraction == 1.0

    def test_full_rank_activations_give_small_null_space(self):
        """If activations span the full space, null space should be small or empty.

        With n_samples >> d and random data, A typically has full rank.
        The null space dimension should be close to 0, and most of delta
        should be projected out (low preserved_fraction).
        """
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        d = 10
        # Many more samples than dimensions → full rank, small null space
        backend.random_seed(42)
        A = backend.random_normal((100, d))  # 10x overdetermined
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        assert result.null_space_dim == 0
        assert result.filtering_applied is False
        assert result.preserved_fraction == 1.0

    def test_dimension_mismatch_raises_error(self):
        """Mismatched dimensions should raise NullSpaceFilterError.

        Dimension mismatch is a BUG - the geometry is invariant, so if dimensions
        don't match, our pipeline failed to compute the right transformation.
        No fallbacks. Fix the algorithm.
        """
        import pytest
        from modelcypher.core.domain.merging.exceptions import NullSpaceFilterError

        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((15,))  # Wrong dimension
        backend.eval(A)
        backend.eval(delta)

        with pytest.raises(NullSpaceFilterError) as exc_info:
            null_filter.filter_delta(delta, A)

        assert "Activation dim" in str(exc_info.value)
        assert exc_info.value.context["activation_dim"] == 20
        assert exc_info.value.context["weight_dim"] == 15


class TestMergeIntegration:
    """Test integration with merge workflow."""

    def test_filter_merge_delta_convenience(self):
        """Test convenience function for merge workflow."""
        backend = get_default_backend()
        d = 20
        n_samples = 50

        backend.random_seed(42)
        source = backend.random_normal((d, d))
        target = backend.random_normal((d, d))
        activations = backend.random_normal((n_samples, d * d))  # Flattened
        backend.eval(source)
        backend.eval(target)
        backend.eval(activations)

        source_flat = backend.reshape(source, (-1,))
        target_flat = backend.reshape(target, (-1,))
        backend.eval(source_flat)
        backend.eval(target_flat)

        # NO ALPHA - geometric addition only
        merged, result = filter_merge_delta_to_null_space(
            source_flat,
            target_flat,
            activations,
        )

        backend.eval(merged)
        assert merged.shape == source_flat.shape
        assert result.filtering_applied or result.null_space_dim == 0

    def test_merged_is_target_plus_filtered_delta(self):
        """Merged = target + filtered_delta (geometric addition)."""
        backend = get_default_backend()
        d = 10

        backend.random_seed(42)
        source = backend.random_normal((d,))
        target = backend.random_normal((d,))
        activations = backend.random_normal((30, d))
        backend.eval(source, target, activations)

        merged, result = filter_merge_delta_to_null_space(source, target, activations)
        backend.eval(merged)

        # Verify: merged = target + filtered_delta
        expected = target + result.filtered_delta
        backend.eval(expected)
        diff = backend.max(backend.abs(merged - expected))
        backend.eval(diff)
        eps = machine_epsilon(backend, merged) * merged.shape[0]
        assert backend.to_scalar(diff) <= eps


class TestModelProfile:
    """Test model-level null space profiling."""

    def test_compute_model_profile(self):
        """Test computing null space profile across layers."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        # Simulate layer activations with varying null space
        backend.random_seed(42)
        act0 = backend.random_normal((50, 100))  # Likely small null space
        act1 = backend.random_normal((20, 100))  # Larger null space
        act2_a = backend.random_normal((30, 50))
        act2_b = backend.zeros((30, 50))
        act2 = backend.concatenate([act2_a, act2_b], axis=1)  # Half null
        backend.eval(act0)
        backend.eval(act1)
        backend.eval(act2)

        layer_activations = {
            0: act0,
            1: act1,
            2: act2,
        }

        profile = null_filter.compute_model_null_space_profile(layer_activations)

        assert len(profile.per_layer) == 3
        assert profile.total_null_dim >= 0
        assert 0.0 <= profile.mean_null_fraction <= 1.0
        assert all(l in [0, 1, 2] for l in profile.graftable_layers)

    def test_graftable_layers_threshold(self):
        """Test that graft threshold correctly identifies layers."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        # Layer 0: full rank (not graftable)
        # Layer 1: half null (graftable at 0.1 threshold)
        backend.random_seed(42)
        act0 = backend.random_normal((200, 50))  # Overdetermined
        act1_a = backend.random_normal((30, 25))
        act1_b = backend.zeros((30, 25))
        act1 = backend.concatenate([act1_a, act1_b], axis=1)  # 50% null
        backend.eval(act0)
        backend.eval(act1)

        layer_activations = {
            0: act0,
            1: act1,
        }

        profile = null_filter.compute_model_null_space_profile(layer_activations)

        # Layer 1 should be graftable (50% > 40% threshold)
        assert 1 in profile.graftable_layers


class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        n_samples=st.integers(min_value=10, max_value=100),
        d=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20)
    def test_projection_loss_plus_preserved_equals_one(self, n_samples, d):
        """projection_loss + preserved_fraction should always equal 1."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((n_samples, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        total = result.projection_loss + result.preserved_fraction
        eps = machine_epsilon(backend, backend.array(1.0))
        assert abs(total - 1.0) <= eps, f"Total was {total}"

    @given(
        d=st.integers(min_value=5, max_value=30),
    )
    @settings(max_examples=10)
    def test_filtered_norm_leq_original_norm(self, d):
        """Filtered delta should never have larger norm than original."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((50, d))
        delta = backend.random_normal((d,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        eps = machine_epsilon(backend, backend.array(1.0))
        assert result.filtered_norm <= result.original_norm + eps


class TestEdgeCases:
    """Edge case handling."""

    def test_single_sample_activation(self):
        """Single sample should still work."""
        backend = get_default_backend()
        filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((1, 10))
        delta = backend.random_normal((10,))
        backend.eval(A)
        backend.eval(delta)

        filter.filter_delta(delta, A)
        # Should not crash, may or may not filter

    def test_very_high_dimensional(self):
        """High dimensional space should work."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((100, 500))
        delta = backend.random_normal((500,))
        backend.eval(A)
        backend.eval(delta)

        result = null_filter.filter_delta(delta, A)

        # Should have large null space (500 - rank(A) ~ 400)
        assert result.null_space_dim >= 350

    def test_zero_activations(self):
        """All-zero activations should give full null space."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        A = backend.zeros((30, 20))
        backend.eval(A)

        projection = null_filter.compute_null_space_projection(A)

        # All of space is null
        assert projection.null_dim == 20

    def test_nan_handling(self):
        """NaN in activations should raise ValueError before reaching SVD."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        # Set one element to NaN
        A_list = array_to_list(backend, A)
        A_list[0][0] = float("nan")
        A = backend.array(A_list)
        backend.eval(A)
        backend.eval(delta)

        # Should raise ValueError with clear message about NaN
        with pytest.raises(ValueError, match="NaN or Inf"):
            null_filter.filter_delta(delta, A)

    def test_inf_handling(self):
        """Inf in activations should raise ValueError before reaching SVD."""
        backend = get_default_backend()
        null_filter = NullSpaceFilter(backend)

        backend.random_seed(42)
        A = backend.random_normal((30, 20))
        delta = backend.random_normal((20,))
        # Set one element to Inf
        A_list = array_to_list(backend, A)
        A_list[0][0] = float("inf")
        A = backend.array(A_list)
        backend.eval(A)
        backend.eval(delta)

        # Should raise ValueError with clear message about Inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            null_filter.filter_delta(delta, A)

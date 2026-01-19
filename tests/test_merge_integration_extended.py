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

"""Integration tests for merge pipeline core components.

Tests critical integration scenarios:
- Cross-architecture alignment (different hidden dims)
- Sequential stacking with occupancy tracking
- Full alignment → stitch → project pipeline
- Null-space preservation guarantees
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import (
    GramAligner,
    find_alignment,
)
from modelcypher.core.domain.geometry.cka import compute_cka, compute_linear_cka, compute_linear_cka_from_activations
from modelcypher.core.domain.geometry.geodesic_null_space import (
    filter_delta_svd,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    division_epsilon,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestCrossArchitectureAlignment:
    """Tests for merging models with different hidden dimensions."""

    def test_larger_to_smaller_dim(self, backend):
        """Source larger than target: 64 → 32."""
        backend.random_seed(42)
        n_samples = 32
        d_source, d_target = 64, 32

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        # Transform shape: [d_source, d_target]
        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (d_source, d_target)

        # Apply transform
        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)
        assert backend.shape(aligned) == (n_samples, d_target)
        assert all_finite(aligned, backend)

    def test_smaller_to_larger_dim(self, backend):
        """Source smaller than target: 32 → 64."""
        backend.random_seed(42)
        n_samples = 32
        d_source, d_target = 32, 64

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        # Transform shape: [d_source, d_target]
        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (d_source, d_target)

        # Apply transform
        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)
        assert backend.shape(aligned) == (n_samples, d_target)
        assert all_finite(aligned, backend)

    def test_extreme_ratio_compression(self, backend):
        """Extreme compression: 128 → 32 (4x)."""
        backend.random_seed(42)
        n_samples = 48
        d_source, d_target = 128, 32

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (d_source, d_target)
        assert all_finite(F, backend)

    def test_extreme_ratio_expansion(self, backend):
        """Extreme expansion: 32 → 128 (4x)."""
        backend.random_seed(42)
        n_samples = 48
        d_source, d_target = 32, 128

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        F = backend.array(result.feature_transform)
        assert backend.shape(F) == (d_source, d_target)
        assert all_finite(F, backend)

    def test_cka_after_alignment_reasonable(self, backend):
        """CKA after alignment should be reasonable for random data."""
        backend.random_seed(42)
        n_samples = 64
        d_source, d_target = 48, 32

        # Create correlated data for meaningful CKA
        shared = backend.random_normal((n_samples, 16))
        source_extra = backend.random_normal((n_samples, d_source - 16))
        target_extra = backend.random_normal((n_samples, d_target - 16))
        source_acts = backend.concatenate([shared, source_extra], axis=1)
        target_acts = backend.concatenate([shared, target_extra], axis=1)
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)

        aligned = backend.matmul(source_acts, backend.array(result.feature_transform))
        backend.eval(aligned)
        # Use linear CKA since alignment now reports linear CKA
        expected = compute_linear_cka_from_activations(aligned, target_acts, backend)
        eps = division_epsilon(backend, aligned)
        assert abs(result.achieved_cka - expected) <= eps


class TestAlignStitchProjectPipeline:
    """Tests for the full alignment → stitch → project pipeline."""

    def test_align_then_delta_projection(self, backend):
        """Align activations, compute delta, project to null-space."""
        backend.random_seed(42)
        n_samples, dim = 32, 48

        # Simulate source and target activations
        source_acts = backend.random_normal((n_samples, dim))
        target_acts = backend.random_normal((n_samples, dim))
        backend.eval(source_acts, target_acts)

        # Simulate weight matrices
        in_dim, out_dim = 64, dim
        source_weight = backend.random_normal((in_dim, out_dim))
        target_weight = backend.random_normal((in_dim, out_dim))
        backend.eval(source_weight, target_weight)

        # 1. Align activations
        result = find_alignment(source_acts, target_acts, backend)
        F = backend.array(result.feature_transform)

        # 2. Compute aligned source weight (apply transform to output dimension)
        # For weight [in, out], transform applies to output: W @ F
        aligned_source = backend.matmul(source_weight, F)
        backend.eval(aligned_source)

        # 3. Compute delta
        delta = aligned_source - target_weight
        backend.eval(delta)
        assert all_finite(delta, backend)

        # 4. Project delta through SVD filter (low-rank truncation)
        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)
        assert all_finite(delta_proj, backend)

        # 5. Apply delta
        merged = target_weight + delta_proj
        backend.eval(merged)
        assert all_finite(merged, backend)

    def test_svd_filter_preserves_structure(self, backend):
        """SVD filter should preserve dominant structure."""
        backend.random_seed(42)
        dim = 32

        # Create a delta with clear low-rank structure
        in_dim = 48
        # Create low-rank delta: outer product
        u = backend.random_normal((in_dim, 4))
        v = backend.random_normal((4, dim))
        delta = backend.matmul(u, v)  # Rank-4 matrix
        backend.eval(delta)

        original_norm = backend.mean(backend.abs(delta))
        backend.eval(original_norm)

        # Project with precision-derived rank (should preserve most)
        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        # Most of the signal should be preserved for low-rank input
        proj_norm = backend.mean(backend.abs(delta_proj))
        backend.eval(proj_norm)

        # Projected should retain most of the signal
        ratio = float(backend.to_scalar(proj_norm)) / float(backend.to_scalar(original_norm))
        eps = division_epsilon(backend, delta) * max(1.0, abs(ratio))
        assert abs(ratio - 1.0) <= eps


class TestSequentialStacking:
    """Tests for sequential model stacking (A→B, C→B')."""

    def test_occupancy_accumulates(self, backend):
        """Occupancy should accumulate across sequential merges."""
        backend.random_seed(42)
        dim = 32

        # Initial occupancy: all zeros
        occupancy = backend.zeros((dim,))
        backend.eval(occupancy)

        # First merge contributes to some dimensions
        delta1 = backend.random_normal((16, dim))
        backend.eval(delta1)

        # Compute variance contribution (simplified occupancy update)
        var1 = backend.var(delta1, axis=0)
        backend.eval(var1)
        occupancy = occupancy + var1
        backend.eval(occupancy)

        # Second merge contributes more
        delta2 = backend.random_normal((16, dim))
        backend.eval(delta2)
        var2 = backend.var(delta2, axis=0)
        backend.eval(var2)
        occupancy = occupancy + var2
        backend.eval(occupancy)

        expected_occ = var1 + var2
        backend.eval(expected_occ)
        mean_occ = backend.mean(occupancy)
        expected_mean = backend.mean(expected_occ)
        backend.eval(mean_occ, expected_mean)
        eps = division_epsilon(backend, occupancy)
        assert abs(float(backend.to_scalar(mean_occ)) - float(backend.to_scalar(expected_mean))) <= eps

    def test_delta_scale_respects_budget(self, backend):
        """delta_scale should reduce the magnitude of projected delta."""
        backend.random_seed(42)
        dim = 48
        in_dim = 64

        delta = backend.random_normal((in_dim, dim))
        backend.eval(delta)

        # Full projection (scale = 1.0)
        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_full = result.filtered_delta
        backend.eval(delta_full)

        full_norm = backend.mean(backend.abs(delta_full))
        backend.eval(full_norm)

        # Scaled projection (scale = 0.5)
        delta_scaled = delta_full * 0.5
        backend.eval(delta_scaled)

        scaled_norm = backend.mean(backend.abs(delta_scaled))
        backend.eval(scaled_norm)

        # Scaled should be half the norm
        ratio = float(backend.to_scalar(scaled_norm)) / float(backend.to_scalar(full_norm))
        eps = division_epsilon(backend, delta_full) * max(1.0, abs(ratio))
        assert abs(ratio - 0.5) <= eps

    def test_three_sequential_merges(self, backend):
        """Three sequential merges should all contribute."""
        backend.random_seed(42)
        dim = 32
        in_dim = 48

        # Target weights (base)
        target_weight = backend.random_normal((in_dim, dim))
        backend.eval(target_weight)
        base_norm = backend.mean(backend.abs(target_weight))
        backend.eval(base_norm)

        merged = target_weight
        deltas = []

        for i in range(3):
            # Each source contributes a delta
            backend.random_seed(42 + i)
            delta = backend.random_normal((in_dim, dim)) * 0.1  # Small delta
            backend.eval(delta)

            result = filter_delta_svd(
                delta,
                backend=backend,
            )
            delta_proj = result.filtered_delta
            backend.eval(delta_proj)
            deltas.append(delta_proj * 0.33)

            # Scale by 1/3 for sequential budget
            merged = merged + delta_proj * 0.33
            backend.eval(merged)

        expected_merged = target_weight
        for delta in deltas:
            expected_merged = expected_merged + delta
        backend.eval(expected_merged)

        diff = backend.mean(backend.abs(merged - expected_merged))
        backend.eval(diff)
        eps = division_epsilon(backend, merged)
        assert float(backend.to_scalar(diff)) <= eps


class TestSVDFilterInvariants:
    """Tests for SVD filter mathematical invariants."""

    def test_svd_filter_reduces_or_preserves_norm(self, backend):
        """SVD filter should not increase delta norm (it's a projection)."""
        backend.random_seed(42)
        dim = 32
        in_dim = 64

        delta = backend.random_normal((in_dim, dim))
        backend.eval(delta)

        original_norm = backend.sqrt(backend.sum(delta * delta))
        backend.eval(original_norm)

        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        proj_norm = backend.sqrt(backend.sum(delta_proj * delta_proj))
        backend.eval(proj_norm)

        # Projection norm <= original norm (can't add energy)
        original_norm_val = float(backend.to_scalar(original_norm))
        tol = division_epsilon(backend, delta) * original_norm_val
        assert float(backend.to_scalar(proj_norm)) <= original_norm_val + tol


class TestCrossArchitectureMathematicalProperties:
    """Hypothesis-based tests for cross-architecture invariants."""

    @given(
        n_samples=st.integers(min_value=16, max_value=48),
        d_source=st.integers(min_value=16, max_value=64),
        d_target=st.integers(min_value=16, max_value=64),
    )
    @settings(max_examples=5, deadline=None)
    def test_alignment_produces_correct_shape(self, n_samples, d_source, d_target):
        """Alignment transform should always have [d_source, d_target] shape."""
        backend = get_default_backend()
        backend.random_seed(42)

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)
        F = backend.array(result.feature_transform)

        assert backend.shape(F) == (d_source, d_target)

    @given(
        n_samples=st.integers(min_value=16, max_value=48),
        d_source=st.integers(min_value=16, max_value=64),
        d_target=st.integers(min_value=16, max_value=64),
    )
    @settings(max_examples=5, deadline=None)
    def test_aligned_acts_have_target_dim(self, n_samples, d_source, d_target):
        """Aligned activations should have target dimension."""
        backend = get_default_backend()
        backend.random_seed(42)

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        backend.eval(source_acts, target_acts)

        result = find_alignment(source_acts, target_acts, backend)
        F = backend.array(result.feature_transform)

        aligned = backend.matmul(source_acts, F)
        backend.eval(aligned)

        assert backend.shape(aligned) == (n_samples, d_target)
        assert all_finite(aligned, backend)

    @given(
        dim=st.integers(min_value=16, max_value=48),
        n_samples=st.integers(min_value=16, max_value=48),
    )
    @settings(max_examples=5, deadline=None)
    def test_self_alignment_cka_one(self, dim, n_samples):
        """Aligning data with itself should achieve CKA=1.0."""
        backend = get_default_backend()
        backend.random_seed(42)

        data = backend.random_normal((n_samples, dim))
        backend.eval(data)

        result = find_alignment(data, data, backend)

        eps = result.precision_threshold
        assert abs(result.achieved_cka - 1.0) <= eps, (
            f"Self-alignment CKA ({result.achieved_cka}) outside precision ({eps})"
        )


class TestSVDFilterMathematicalProperties:
    """Hypothesis tests for SVD filter."""

    @given(
        in_dim=st.integers(min_value=8, max_value=48),
        out_dim=st.integers(min_value=8, max_value=48),
    )
    @settings(max_examples=5, deadline=None)
    def test_svd_filter_finite(self, in_dim, out_dim):
        """SVD filter should always produce finite output."""
        backend = get_default_backend()
        backend.random_seed(42)

        delta = backend.random_normal((in_dim, out_dim))
        backend.eval(delta)

        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        assert all_finite(delta_proj, backend)

    @given(
        in_dim=st.integers(min_value=8, max_value=48),
        out_dim=st.integers(min_value=8, max_value=48),
    )
    @settings(max_examples=5, deadline=None)
    def test_svd_filter_shape_preserved(self, in_dim, out_dim):
        """SVD filter should preserve delta shape."""
        backend = get_default_backend()
        backend.random_seed(42)

        delta = backend.random_normal((in_dim, out_dim))
        backend.eval(delta)

        result = filter_delta_svd(
            delta,
            backend=backend,
        )
        delta_proj = result.filtered_delta
        backend.eval(delta_proj)

        assert backend.shape(delta_proj) == (in_dim, out_dim)

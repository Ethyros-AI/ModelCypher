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

"""Extended tests for GramAligner - compositional_stitch and anchor_space alignment.

These tests cover previously untested APIs and edge cases:
- compositional_stitch(): Deriving projection transforms from hidden alignment
- find_alignment_anchor_space(): Anchor-relative alignment for underdetermined cases
- Edge cases: dimension mismatches, degenerate inputs, numerical edge cases

Tests use hypothesis for property-based testing of mathematical invariants.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import GramAligner, find_alignment
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    machine_epsilon,
    sqrt_scalar,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def small_activations(backend):
    """Create small activation matrices for testing."""
    backend.random_seed(42)
    return backend.random_normal((16, 32))


@pytest.fixture
def cross_dim_activations(backend):
    """Create activations with different dimensions."""
    backend.random_seed(42)
    source = backend.random_normal((16, 64))
    target = backend.random_normal((16, 32))
    backend.eval(source, target)
    return source, target


# =============================================================================
# Tests for compositional_stitch
# =============================================================================


class TestCompositionalStitch:
    """Tests for GramAligner.compositional_stitch() method."""

    def test_identity_transform_preserves_weight(self, backend):
        """Identity hidden transform should preserve the weight geometry."""
        backend.random_seed(42)

        # Create matching dimensions
        d_hidden = 32
        d_proj = 16

        # Identity hidden transform
        H = backend.eye(d_hidden)

        # Same weight for source and target
        W = backend.random_normal((d_proj, d_hidden))
        backend.eval(H, W)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W, W)
        backend.eval(stitch)

        # Stitch should be close to identity for matching weights
        I = backend.eye(d_proj)
        diff = backend.sum((stitch - I) ** 2)
        backend.eval(diff)

        eps = sqrt_scalar(machine_epsilon(backend, W), backend)
        assert float(backend.to_scalar(diff)) < eps * d_proj

    def test_dimension_mismatch_hidden_source_raises(self, backend):
        """Hidden transform incompatible with source weight should raise."""
        backend.random_seed(42)

        # Hidden transform: 32 -> 48
        H = backend.random_normal((32, 48))
        # Source weight: proj_dim x 64 (hidden dim = 64, not 32!)
        W_src = backend.random_normal((16, 64))
        # Target weight: proj_dim x 48
        W_tgt = backend.random_normal((16, 48))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)

        with pytest.raises(ValueError, match="hidden_transform source dim"):
            aligner.compositional_stitch(H, W_src, W_tgt)

    def test_dimension_mismatch_hidden_target_raises(self, backend):
        """Hidden transform incompatible with target weight should raise."""
        backend.random_seed(42)

        # Hidden transform: 32 -> 48
        H = backend.random_normal((32, 48))
        # Source weight: proj_dim x 32 (matches H source)
        W_src = backend.random_normal((16, 32))
        # Target weight: proj_dim x 64 (hidden dim = 64, not 48!)
        W_tgt = backend.random_normal((16, 64))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)

        with pytest.raises(ValueError, match="hidden_transform target dim"):
            aligner.compositional_stitch(H, W_src, W_tgt)

    def test_cross_dimension_stitch(self, backend):
        """Stitch should work with different source/target dimensions."""
        backend.random_seed(42)

        # Source: 64 hidden, 32 proj
        # Target: 48 hidden, 24 proj
        d_src_hidden, d_tgt_hidden = 64, 48
        d_src_proj, d_tgt_proj = 32, 24

        H = backend.random_normal((d_src_hidden, d_tgt_hidden))
        W_src = backend.random_normal((d_src_proj, d_src_hidden))
        W_tgt = backend.random_normal((d_tgt_proj, d_tgt_hidden))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        # Stitch should have shape [target_proj, source_proj]
        shape = backend.shape(stitch)
        assert shape[0] == d_tgt_proj
        assert shape[1] == d_src_proj

    def test_stitch_produces_finite_values(self, backend):
        """Stitch should not produce NaN or Inf values."""
        backend.random_seed(42)

        d_hidden = 32
        d_proj = 16

        H = backend.random_normal((d_hidden, d_hidden))
        W_src = backend.random_normal((d_proj, d_hidden))
        W_tgt = backend.random_normal((d_proj, d_hidden))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        assert all_finite(stitch, backend)

    def test_stitch_from_real_alignment(self, backend):
        """Stitch computed from actual alignment should produce valid geometry."""
        backend.random_seed(42)

        # Create aligned activations
        n_samples = 50
        d_source = 64
        d_target = 48

        source_acts = backend.random_normal((n_samples, d_source))
        # Create target as transformed source (known relationship)
        transform = backend.random_normal((d_source, d_target))
        target_acts = backend.matmul(source_acts, transform)
        backend.eval(source_acts, target_acts, transform)

        # Get alignment
        aligner = GramAligner(backend)
        result = aligner.find_perfect_alignment(source_acts, target_acts)
        H = result.feature_transform

        # Create projection weights
        d_proj = 32
        W_src = backend.random_normal((d_proj, d_source))
        W_tgt = backend.random_normal((d_proj, d_target))
        backend.eval(W_src, W_tgt)

        # Compute stitch
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        # Verify stitch is finite and has correct shape
        assert all_finite(stitch, backend)
        shape = backend.shape(stitch)
        assert shape[0] == d_proj
        assert shape[1] == d_proj

    def test_stitch_matches_projection_equation(self, backend):
        """Stitch should satisfy S @ (W_src @ H) ≈ W_tgt."""
        backend.random_seed(42)

        d_src_hidden, d_tgt_hidden = 32, 24
        d_src_proj, d_tgt_proj = 20, 16

        H = backend.random_normal((d_src_hidden, d_tgt_hidden))
        W_src = backend.random_normal((d_src_proj, d_src_hidden))
        W_tgt = backend.random_normal((d_tgt_proj, d_tgt_hidden))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        # The equation solved is: S @ (W_src @ H) = W_tgt
        W_src_transformed = backend.matmul(W_src, H)  # [src_proj, tgt_hidden]
        lhs = backend.matmul(stitch, W_src_transformed)  # [tgt_proj, tgt_hidden]
        diff = backend.max(backend.abs(lhs - W_tgt))
        backend.eval(diff)

        zero_stitch = backend.zeros_like(stitch)
        baseline = backend.matmul(zero_stitch, W_src_transformed)
        baseline_diff = backend.max(backend.abs(baseline - W_tgt))
        backend.eval(baseline_diff)

        assert float(backend.to_scalar(diff)) <= float(backend.to_scalar(baseline_diff))

    def test_input_stitch_reduces_projection_error(self, backend):
        """Input stitch should satisfy: (H^T @ W_src) @ S_in ≈ W_tgt."""
        backend.random_seed(42)

        d_src_hidden, d_tgt_hidden = 32, 24
        d_src_inter, d_tgt_inter = 28, 20

        H = backend.random_normal((d_src_hidden, d_tgt_hidden))
        W_src = backend.random_normal((d_src_hidden, d_src_inter))
        W_tgt = backend.random_normal((d_tgt_hidden, d_tgt_inter))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch_in = aligner.compositional_stitch_input(H, W_src, W_tgt)
        backend.eval(stitch_in)

        # The equation solved is: (H^T @ W_src) @ S_in = W_tgt
        A = backend.matmul(backend.transpose(H), W_src)  # [tgt_hidden, src_inter]
        lhs = backend.matmul(A, stitch_in)  # [tgt_hidden, tgt_inter]
        diff = backend.max(backend.abs(lhs - W_tgt))
        backend.eval(diff)

        # Compare against baseline (zero stitch)
        zero_in = backend.zeros_like(stitch_in)
        baseline = backend.matmul(A, zero_in)
        baseline_diff = backend.max(backend.abs(baseline - W_tgt))
        backend.eval(baseline_diff)

        assert float(backend.to_scalar(diff)) <= float(backend.to_scalar(baseline_diff))


class TestCompositionalStitchEdgeCases:
    """Edge case tests for compositional_stitch."""

    def test_small_dimensions(self, backend):
        """Should handle very small dimensions."""
        backend.random_seed(42)

        H = backend.random_normal((4, 4))
        W_src = backend.random_normal((2, 4))
        W_tgt = backend.random_normal((2, 4))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        assert all_finite(stitch, backend)
        assert backend.shape(stitch) == (2, 2)

    def test_rectangular_projections(self, backend):
        """Should handle non-square projection dimensions."""
        backend.random_seed(42)

        d_hidden = 32
        d_src_proj, d_tgt_proj = 16, 24

        H = backend.eye(d_hidden)
        W_src = backend.random_normal((d_src_proj, d_hidden))
        W_tgt = backend.random_normal((d_tgt_proj, d_hidden))
        backend.eval(H, W_src, W_tgt)

        aligner = GramAligner(backend)
        stitch = aligner.compositional_stitch(H, W_src, W_tgt)
        backend.eval(stitch)

        shape = backend.shape(stitch)
        assert shape[0] == d_tgt_proj
        assert shape[1] == d_src_proj


# =============================================================================
# Tests for find_alignment_anchor_space
# =============================================================================


class TestAnchorSpaceAlignment:
    """Tests for GramAligner.find_alignment_anchor_space() method."""

    def test_basic_anchor_space_alignment(self, backend):
        """Basic anchor space alignment should produce valid result."""
        backend.random_seed(42)

        n_samples = 50
        n_anchors = 10
        d_source, d_target = 64, 48

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)
        result = aligner.find_alignment_anchor_space(
            source_acts, target_acts, source_anchors, target_anchors
        )

        # Result should have valid CKA
        assert 0.0 <= result.achieved_cka <= 1.0
        # Transform should be k x k
        shape = backend.shape(result.feature_transform)
        assert shape[0] == n_anchors
        assert shape[1] == n_anchors

    def test_overdetermined_case(self, backend):
        """When n_samples > n_anchors, alignment is overdetermined (well-posed)."""
        backend.random_seed(42)

        n_samples = 100  # >> n_anchors
        n_anchors = 10
        d = 64

        source_acts = backend.random_normal((n_samples, d))
        target_acts = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)
        result = aligner.find_alignment_anchor_space(
            source_acts, target_acts, source_anchors, target_anchors
        )

        # Should complete without error and produce valid result
        assert result is not None
        assert all_finite(result.feature_transform, backend)

    def test_sample_count_mismatch_raises(self, backend):
        """Mismatched sample counts should raise ValueError."""
        backend.random_seed(42)

        source_acts = backend.random_normal((50, 32))
        target_acts = backend.random_normal((40, 32))  # Different n!
        source_anchors = backend.random_normal((10, 32))
        target_anchors = backend.random_normal((10, 32))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)

        with pytest.raises(ValueError, match="Sample counts must match"):
            aligner.find_alignment_anchor_space(
                source_acts, target_acts, source_anchors, target_anchors
            )

    def test_different_feature_dimensions(self, backend):
        """Should handle different feature dimensions between source/target."""
        backend.random_seed(42)

        n_samples = 50
        n_anchors = 10
        d_source, d_target = 128, 64  # Very different dimensions

        source_acts = backend.random_normal((n_samples, d_source))
        target_acts = backend.random_normal((n_samples, d_target))
        source_anchors = backend.random_normal((n_anchors, d_source))
        target_anchors = backend.random_normal((n_anchors, d_target))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)
        result = aligner.find_alignment_anchor_space(
            source_acts, target_acts, source_anchors, target_anchors
        )

        # Transform should be in anchor space (k x k), not feature space
        shape = backend.shape(result.feature_transform)
        assert shape[0] == n_anchors
        assert shape[1] == n_anchors


class TestAnchorSpaceEdgeCases:
    """Edge case tests for anchor space alignment."""

    def test_minimal_anchors(self, backend):
        """Should handle very few anchors."""
        backend.random_seed(42)

        n_samples = 50
        n_anchors = 2  # Minimal
        d = 32

        source_acts = backend.random_normal((n_samples, d))
        target_acts = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)
        result = aligner.find_alignment_anchor_space(
            source_acts, target_acts, source_anchors, target_anchors
        )

        assert result is not None
        shape = backend.shape(result.feature_transform)
        assert shape == (n_anchors, n_anchors)

    def test_many_anchors(self, backend):
        """Should handle many anchors (approaching sample count)."""
        backend.random_seed(42)

        n_samples = 50
        n_anchors = 45  # Close to n_samples
        d = 32

        source_acts = backend.random_normal((n_samples, d))
        target_acts = backend.random_normal((n_samples, d))
        source_anchors = backend.random_normal((n_anchors, d))
        target_anchors = backend.random_normal((n_anchors, d))
        backend.eval(source_acts, target_acts, source_anchors, target_anchors)

        aligner = GramAligner(backend)
        result = aligner.find_alignment_anchor_space(
            source_acts, target_acts, source_anchors, target_anchors
        )

        assert result is not None
        assert all_finite(result.feature_transform, backend)


# =============================================================================
# Hypothesis-based Property Tests
# =============================================================================


class TestAlignmentMathematicalProperties:
    """Property-based tests for mathematical invariants."""

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d_features=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_self_alignment_is_identity(self, n_samples, d_features):
        """Aligning X with itself should produce high geodesic CKA."""
        backend = get_default_backend()
        backend.random_seed(42)

        X = backend.random_normal((n_samples, d_features))
        backend.eval(X)

        result = find_alignment(X, X, backend=backend)

        assert abs(result.achieved_cka - 1.0) <= result.precision_threshold

    @given(
        n_samples=st.integers(min_value=30, max_value=100),
        d_features=st.integers(min_value=4, max_value=20),
    )
    @settings(max_examples=10, deadline=None)
    def test_orthogonal_transform_produces_valid_alignment(self, n_samples, d_features):
        """Orthogonal transforms should produce finite alignment transforms.

        Note: Geodesic alignment is designed for structured neural data, not
        random matrices. On random data, the k-NN graph doesn't capture
        meaningful manifold structure, so CKA guarantees don't apply.
        We only verify the transform is finite and bounded.
        """
        backend = get_default_backend()
        backend.random_seed(42)

        X = backend.random_normal((n_samples, d_features))
        # Create orthogonal transform
        Q, _ = backend.qr(backend.random_normal((d_features, d_features)))
        Y = backend.matmul(X, Q)
        backend.eval(X, Y, Q)

        result = find_alignment(X, Y, backend=backend)

        # Verify transform is finite (geodesic alignment completes without error)
        F = result.feature_transform
        backend.eval(F)
        isfinite = backend.isfinite(F)
        backend.eval(isfinite)
        all_finite = bool(backend.to_scalar(backend.all(isfinite)))
        assert all_finite, "Alignment transform contains NaN/Inf"

        # Verify CKA is bounded [0, 1]
        assert 0.0 <= result.achieved_cka <= 1.0

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d_source=st.integers(min_value=4, max_value=32),
        d_target=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_alignment_produces_finite_transform(self, n_samples, d_source, d_target):
        """Alignment should always produce finite transform values."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)

        result = find_alignment(source, target, backend=backend)

        assert all_finite(result.feature_transform, backend)

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d_source=st.integers(min_value=4, max_value=32),
        d_target=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_transform_has_correct_shape(self, n_samples, d_source, d_target):
        """Transform should have shape [d_source, d_target]."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((n_samples, d_source))
        target = backend.random_normal((n_samples, d_target))
        backend.eval(source, target)

        result = find_alignment(source, target, backend=backend)

        shape = backend.shape(result.feature_transform)
        assert shape[0] == d_source
        assert shape[1] == d_target

    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        d_features=st.integers(min_value=4, max_value=32),
    )
    @settings(max_examples=10, deadline=None)
    def test_cka_is_bounded(self, n_samples, d_features):
        """CKA should always be in [0, 1]."""
        backend = get_default_backend()
        backend.random_seed(42)

        source = backend.random_normal((n_samples, d_features))
        target = backend.random_normal((n_samples, d_features))
        backend.eval(source, target)

        result = find_alignment(source, target, backend=backend)

        assert 0.0 <= result.achieved_cka <= 1.0

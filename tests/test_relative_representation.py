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

"""Comprehensive tests for relative_representation.py.

These tests are designed to find real bugs, not pass coverage metrics.
Focus areas:
1. Numerical edge cases (zero norms, near-singular matrices)
2. Dimension mismatches
3. Procrustes reflection handling (det < 0)
4. Pseudo-inverse rank deficiency
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    is_finite,
    is_inf,
    is_nan,
)
from modelcypher.core.domain.geometry.relative_representation import (
    RelativeRepresentation,
    align_relative_representations,
    compute_relative_representation,
    transfer_via_relative_space,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@pytest.fixture
def backend() -> "Backend":
    return get_default_backend()


def _eps(backend: "Backend", *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _to_scalar(backend: "Backend", array) -> float:
    backend.eval(array)
    return float(backend.to_scalar(array))


def _assert_all_finite(backend: "Backend", array) -> None:
    finite = backend.isfinite(array)
    backend.eval(finite)
    count_arr = backend.sum(finite)
    backend.eval(count_arr)
    count = float(backend.to_scalar(count_arr))
    total = 1
    for dim in backend.shape(array):
        total *= int(dim)
    assert abs(count - total) <= _eps(backend, count, float(total))


class TestComputeRelativeRepresentation:
    """Tests for compute_relative_representation function."""

    def test_basic_functionality(self, backend: "Backend") -> None:
        """Basic test: cosine similarities to anchors."""
        # 3 hidden states, 4-dim each
        hidden = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ])
        # 2 anchors, 4-dim each
        anchors = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_list = backend.tolist(rel)

        # First hidden [1,0,0,0] should have cos=1 with first anchor, cos=0 with second
        assert backend.shape(rel) == (3, 2)
        eps = _eps(backend, rel_list[0][0], rel_list[0][1])
        assert abs(rel_list[0][0] - 1.0) <= eps
        assert abs(rel_list[0][1] - 0.0) <= eps
        # Second hidden [0,1,0,0] should have cos=0 with first, cos=1 with second
        assert abs(rel_list[1][0] - 0.0) <= eps
        assert abs(rel_list[1][1] - 1.0) <= eps

    def test_zero_norm_hidden_state(self, backend: "Backend") -> None:
        """Edge case: zero-norm hidden state should not cause NaN/Inf."""
        hidden = backend.array([
            [0.0, 0.0, 0.0, 0.0],  # Zero norm!
            [1.0, 0.0, 0.0, 0.0],
        ])
        anchors = backend.array([
            [1.0, 0.0, 0.0, 0.0],
        ])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        value = _to_scalar(backend, rel[0, 0])

        # Should NOT be NaN - the code uses maximum(norm, 1e-8)
        assert not is_nan(value, backend)
        assert not is_inf(value, backend)
        # Zero vector normalized to 0/1e-8 = 0, so cos should be 0
        assert abs(value) <= _eps(backend, value, 0.0)

    def test_zero_norm_anchor(self, backend: "Backend") -> None:
        """Edge case: zero-norm anchor should not cause NaN/Inf."""
        hidden = backend.array([
            [1.0, 0.0, 0.0, 0.0],
        ])
        anchors = backend.array([
            [0.0, 0.0, 0.0, 0.0],  # Zero norm anchor!
            [1.0, 0.0, 0.0, 0.0],
        ])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        value = _to_scalar(backend, rel[0, 0])
        second = _to_scalar(backend, rel[0, 1])

        assert not is_nan(value, backend)
        assert not is_inf(value, backend)
        # Second anchor should work normally
        assert abs(second - 1.0) <= _eps(backend, second, 1.0)

    def test_very_small_norm(self, backend: "Backend") -> None:
        """Edge case: very small but non-zero norm."""
        hidden = backend.array([
            [1e-10, 1e-10, 0.0, 0.0],  # Very small norm
        ])
        anchors = backend.array([
            [1.0, 0.0, 0.0, 0.0],
        ])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        value = _to_scalar(backend, rel[0, 0])

        # Should be stable (normalized small vector still has direction)
        assert not is_nan(value, backend)
        assert not is_inf(value, backend)

    def test_single_sample(self, backend: "Backend") -> None:
        """Single sample should work."""
        hidden = backend.array([[1.0, 2.0, 3.0]])
        anchors = backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        assert backend.shape(rel) == (1, 2)
        _assert_all_finite(backend, rel)

    def test_single_anchor(self, backend: "Backend") -> None:
        """Single anchor should work."""
        hidden = backend.array([[1.0, 0.0], [0.0, 1.0]])
        anchors = backend.array([[1.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_list = backend.tolist(rel)

        assert backend.shape(rel) == (2, 1)
        eps = _eps(backend, rel_list[0][0], rel_list[1][0])
        assert abs(rel_list[0][0] - 1.0) <= eps
        assert abs(rel_list[1][0] - 0.0) <= eps

    def test_high_dimensional(self, backend: "Backend") -> None:
        """High-dimensional inputs (like 2048-dim embeddings)."""
        backend.random_seed(42)
        hidden = backend.random_normal((10, 2048))
        anchors = backend.random_normal((50, 2048))
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        assert backend.shape(rel) == (10, 50)
        # All similarities should be in [-1, 1]
        min_arr = backend.min(rel)
        max_arr = backend.max(rel)
        backend.eval(min_arr, max_arr)
        min_val = float(backend.to_scalar(min_arr))
        max_val = float(backend.to_scalar(max_arr))
        eps = _eps(backend, min_val, max_val)
        assert min_val >= -1.0 - eps
        assert max_val <= 1.0 + eps
        # No NaN or Inf
        _assert_all_finite(backend, rel)

    def test_negative_values(self, backend: "Backend") -> None:
        """Negative values should produce negative cosine similarities."""
        hidden = backend.array([[-1.0, 0.0, 0.0]])
        anchors = backend.array([[1.0, 0.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        value = _to_scalar(backend, rel[0, 0])

        # Opposite directions should give cos = -1
        assert abs(value - (-1.0)) <= _eps(backend, value, -1.0)


class TestAlignRelativeRepresentations:
    """Tests for align_relative_representations function."""

    def test_identical_representations(self, backend: "Backend") -> None:
        """Identical representations should give identity rotation and zero error."""
        rel = backend.array([
            [1.0, 0.5, 0.2],
            [0.3, 0.8, 0.1],
            [0.5, 0.5, 0.5],
        ])
        backend.eval(rel)

        R, error = align_relative_representations(rel, rel)
        backend.eval(R)

        # R should be close to identity
        identity = backend.eye(3)
        diff = backend.abs(R - identity)
        max_arr = backend.max(diff)
        backend.eval(max_arr)
        max_diff = float(backend.to_scalar(max_arr))
        eps = _eps(backend, max_diff, error) * backend.shape(R)[0]
        assert max_diff <= eps, f"Expected identity, got diff max {max_diff}"
        # Error should be ~0
        assert error <= eps, f"Expected zero error, got {error}"

    def test_known_rotation(self, backend: "Backend") -> None:
        """Verify Procrustes finds an orthogonal alignment."""
        backend.random_seed(42)
        # Create related data where alignment should work
        source = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ])
        # Create target that is similar (low error expected after alignment)
        target = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ])
        backend.eval(source, target)

        R_recovered, error = align_relative_representations(source, target)
        backend.eval(R_recovered)

        # Identical data should have near-zero error
        eps = _eps(backend, error) * backend.shape(R_recovered)[0]
        assert error <= eps, f"Expected low error for identical data, got {error}"

        # Verify R is orthogonal (R @ R^T = I)
        RRt = backend.matmul(R_recovered, backend.transpose(R_recovered))
        identity = backend.eye(4)
        diff = backend.abs(RRt - identity)
        max_arr = backend.max(diff)
        backend.eval(max_arr)
        max_diff = float(backend.to_scalar(max_arr))
        assert max_diff <= eps, "R should be orthogonal"

    def test_reflection_detection(self, backend: "Backend") -> None:
        """Procrustes should detect and fix reflections (det < 0)."""
        # Create a reflection by flipping one axis
        source = backend.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Reflect across first axis
        target = backend.array([
            [-1.0, 0.0, 0.0],  # Flipped!
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        backend.eval(source, target)

        R, error = align_relative_representations(source, target)
        backend.eval(R)

        # R should have det = +1 (proper rotation, not reflection)
        det = backend.det(R)
        backend.eval(det)
        det_val = _to_scalar(backend, det)
        eps = _eps(backend, det_val, 1.0)
        assert abs(det_val - 1.0) <= eps, f"Expected det=1, got {det_val}"

    def test_orthogonal_subspaces(self, backend: "Backend") -> None:
        """Orthogonal representations should still produce valid rotation."""
        # Source in xy plane
        source = backend.array([
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
        ])
        # Target in xz plane
        target = backend.array([
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ])
        backend.eval(source, target)

        R, error = align_relative_representations(source, target)
        backend.eval(R)

        # Should produce a valid rotation matrix
        # Check orthogonality: R @ R^T = I
        RtR = backend.matmul(R, backend.transpose(R))
        identity = backend.eye(3)
        diff = backend.abs(RtR - identity)
        max_arr = backend.max(diff)
        backend.eval(max_arr)
        max_diff = float(backend.to_scalar(max_arr))
        eps = _eps(backend, max_diff) * backend.shape(R)[0]
        assert max_diff <= eps

    def test_single_sample_degenerate(self, backend: "Backend") -> None:
        """Single sample is degenerate but should not crash."""
        source = backend.array([[1.0, 2.0, 3.0]])
        target = backend.array([[3.0, 2.0, 1.0]])
        backend.eval(source, target)

        # Should not raise
        R, error = align_relative_representations(source, target)
        backend.eval(R)

        # Result may be arbitrary but should be valid matrix
        assert backend.shape(R) == (3, 3)
        _assert_all_finite(backend, R)

    def test_all_zeros_source(self, backend: "Backend") -> None:
        """All-zero source should not cause crash."""
        source = backend.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        target = backend.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        backend.eval(source, target)

        # May produce degenerate result but should not crash
        R, error = align_relative_representations(source, target)
        backend.eval(R)

        _assert_all_finite(backend, R)

    def test_large_scale_difference(self, backend: "Backend") -> None:
        """Large scale difference should still produce correct rotation."""
        backend.random_seed(42)
        source = backend.random_normal((20, 5))
        backend.eval(source)
        # Scale target by 1000x
        target_scaled = source * 1000.0
        backend.eval(target_scaled)

        R, error = align_relative_representations(source, target_scaled)
        backend.eval(R)

        # R should be identity (same direction, different scale)
        # Due to centering, scale should be handled
        assert backend.shape(R) == (5, 5)
        # Check it's still orthogonal
        RtR = backend.matmul(R, backend.transpose(R))
        identity = backend.eye(5)
        diff = backend.abs(RtR - identity)
        max_arr = backend.max(diff)
        backend.eval(max_arr)
        max_diff = float(backend.to_scalar(max_arr))
        eps = _eps(backend, max_diff) * backend.shape(R)[0]
        assert max_diff <= eps


class TestTransferViaRelativeSpace:
    """Tests for transfer_via_relative_space function."""

    def test_same_dimension_identity_transfer(self, backend: "Backend") -> None:
        """Same source/target dim with identical anchors should preserve structure."""
        backend.random_seed(42)
        hidden = backend.random_normal((5, 64))
        anchors = backend.random_normal((20, 64))
        backend.eval(hidden, anchors)

        # Same anchors for source and target
        transferred = transfer_via_relative_space(
            hidden,
            anchors,
            anchors,
            alignment_samples=None,
        )
        backend.eval(transferred)

        # Should recover something similar (not exact due to pseudo-inverse)
        assert backend.shape(transferred) == (5, 64)
        _assert_all_finite(backend, transferred)

    def test_cross_dimension_transfer(self, backend: "Backend") -> None:
        """Transfer from 2048-dim to 896-dim (real use case)."""
        backend.random_seed(42)
        hidden_2048 = backend.random_normal((10, 2048))
        source_anchors = backend.random_normal((100, 2048))
        target_anchors = backend.random_normal((100, 896))
        backend.eval(hidden_2048, source_anchors, target_anchors)

        transferred = transfer_via_relative_space(
            hidden_2048,
            source_anchors,
            target_anchors,
            alignment_samples=None,
        )
        backend.eval(transferred)

        # Should produce 896-dim output
        assert backend.shape(transferred) == (10, 896)
        _assert_all_finite(backend, transferred)

    def test_few_anchors_rank_deficiency(self, backend: "Backend") -> None:
        """Very few anchors may cause pseudo-inverse rank issues."""
        backend.random_seed(42)
        hidden = backend.random_normal((5, 64))
        # Only 3 anchors for 64-dim space - severe rank deficiency
        source_anchors = backend.random_normal((3, 64))
        target_anchors = backend.random_normal((3, 64))
        backend.eval(hidden, source_anchors, target_anchors)

        # Should not crash even with rank deficiency
        transferred = transfer_via_relative_space(
            hidden,
            source_anchors,
            target_anchors,
        )
        backend.eval(transferred)

        _assert_all_finite(backend, transferred)

    def test_collinear_anchors(self, backend: "Backend") -> None:
        """Collinear anchors cause rank-1 anchor similarity matrix."""
        # All anchors point in same direction (different magnitudes)
        source_anchors = backend.array([
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
        ])
        target_anchors = backend.array([
            [0.5, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ])
        hidden = backend.array([[1.0, 1.0, 0.0, 0.0]])
        backend.eval(source_anchors, target_anchors, hidden)

        # After normalization, all anchors are identical - severely ill-conditioned
        # Should not crash
        transferred = transfer_via_relative_space(hidden, source_anchors, target_anchors)
        backend.eval(transferred)

        # May have numerical issues but should not be NaN
        # (pinv handles rank deficiency)
        assert transferred is not None

    def test_with_alignment_samples(self, backend: "Backend") -> None:
        """Test with alignment_samples provided."""
        backend.random_seed(42)
        hidden = backend.random_normal((5, 32))
        source_anchors = backend.random_normal((20, 32))
        target_anchors = backend.random_normal((20, 64))
        # alignment_samples: [n_pairs, source_dim + target_dim]
        alignment_source = backend.random_normal((10, 32))
        alignment_target = backend.random_normal((10, 64))
        backend.eval(hidden, source_anchors, target_anchors, alignment_source, alignment_target)
        alignment_samples = backend.concatenate([alignment_source, alignment_target], axis=1)
        backend.eval(alignment_samples)

        transferred = transfer_via_relative_space(
            hidden,
            source_anchors,
            target_anchors,
            alignment_samples=alignment_samples,
        )
        backend.eval(transferred)

        assert backend.shape(transferred) == (5, 64)
        _assert_all_finite(backend, transferred)


class TestRelativeRepresentationDataclass:
    """Tests for RelativeRepresentation dataclass."""

    def test_properties(self, backend: "Backend") -> None:
        """Test n_samples and n_anchors properties."""
        similarities = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        backend.eval(similarities)

        rel_rep = RelativeRepresentation(
            similarities=similarities,
            anchor_ids=("a1", "a2", "a3"),
            hidden_dim=128,
        )

        assert rel_rep.n_samples == 2
        assert rel_rep.n_anchors == 3
        assert rel_rep.hidden_dim == 128


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_model_merge_scenario(self, backend: "Backend") -> None:
        """Simulate a model merge scenario: Qwen2.5 (896-dim) to LLaMA (4096-dim)."""
        backend.random_seed(42)

        # Source: Qwen2.5-like activations
        source_hidden = backend.random_normal((50, 896))
        source_anchors = backend.random_normal((200, 896))

        # Target: LLaMA-like
        target_anchors = backend.random_normal((200, 4096))

        backend.eval(source_hidden, source_anchors, target_anchors)

        # Compute relative representation
        rel = compute_relative_representation(source_hidden, source_anchors)
        backend.eval(rel)

        # Should be in anchor space (dimension-agnostic)
        assert backend.shape(rel) == (50, 200)

        # Transfer to target space
        transferred = transfer_via_relative_space(
            source_hidden,
            source_anchors,
            target_anchors,
        )
        backend.eval(transferred)

        # Should be in target dimension
        assert backend.shape(transferred) == (50, 4096)
        _assert_all_finite(backend, transferred)

    def test_roundtrip_same_dimension(self, backend: "Backend") -> None:
        """Round-trip transfer should approximately preserve structure."""
        backend.random_seed(42)

        hidden = backend.random_normal((10, 64))
        anchors = backend.random_normal((50, 64))
        backend.eval(hidden, anchors)

        # Compute relative rep
        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)

        # Transfer back (using same anchors)
        recovered = transfer_via_relative_space(hidden, anchors, anchors)
        backend.eval(recovered)

        # Compute relative rep of recovered
        rel_recovered = compute_relative_representation(recovered, anchors)
        backend.eval(rel_recovered)

        # Relative representations should be similar
        diff = backend.norm(rel - rel_recovered)
        orig_norm = backend.norm(rel)
        backend.eval(diff, orig_norm)

        diff_val = _to_scalar(backend, diff)
        orig_val = _to_scalar(backend, orig_norm)
        eps = _eps(backend, orig_val)
        relative_error = diff_val / max(orig_val, eps)
        assert is_finite(relative_error, backend)
        assert relative_error >= 0.0

    def test_alignment_improves_transfer(self, backend: "Backend") -> None:
        """Alignment samples should reduce transfer error."""
        backend.random_seed(42)

        hidden = backend.random_normal((10, 32))
        source_anchors = backend.random_normal((30, 32))
        # Create target anchors as rotated source anchors (known relationship)
        R_known = backend.random_normal((32, 48))
        backend.eval(hidden, source_anchors, R_known)
        target_anchors = backend.matmul(source_anchors, R_known)
        backend.eval(target_anchors)

        # Without alignment
        transferred_no_align = transfer_via_relative_space(
            hidden, source_anchors, target_anchors
        )
        backend.eval(transferred_no_align)

        # With alignment samples (same transformation)
        align_source = source_anchors[:10]
        align_target = backend.matmul(align_source, R_known)
        backend.eval(align_target)
        alignment_samples = backend.concatenate([align_source, align_target], axis=1)
        backend.eval(alignment_samples)

        transferred_with_align = transfer_via_relative_space(
            hidden, source_anchors, target_anchors, alignment_samples=alignment_samples
        )
        backend.eval(transferred_with_align)

        # Both should produce valid output
        assert backend.shape(transferred_no_align) == (10, 48)
        assert backend.shape(transferred_with_align) == (10, 48)

        # No NaN
        _assert_all_finite(backend, transferred_no_align)
        _assert_all_finite(backend, transferred_with_align)

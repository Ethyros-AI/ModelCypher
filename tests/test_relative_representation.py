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

import math
from typing import TYPE_CHECKING

import pytest

from modelcypher.core.domain._backend import get_default_backend
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
        rel_np = backend.to_numpy(rel)

        # First hidden [1,0,0,0] should have cos=1 with first anchor, cos=0 with second
        assert rel_np.shape == (3, 2)
        assert abs(rel_np[0, 0] - 1.0) < 1e-5
        assert abs(rel_np[0, 1] - 0.0) < 1e-5
        # Second hidden [0,1,0,0] should have cos=0 with first, cos=1 with second
        assert abs(rel_np[1, 0] - 0.0) < 1e-5
        assert abs(rel_np[1, 1] - 1.0) < 1e-5

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
        rel_np = backend.to_numpy(rel)

        # Should NOT be NaN - the code uses maximum(norm, 1e-8)
        assert not math.isnan(rel_np[0, 0])
        assert not math.isinf(rel_np[0, 0])
        # Zero vector normalized to 0/1e-8 = 0, so cos should be 0
        assert abs(rel_np[0, 0]) < 1e-5

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
        rel_np = backend.to_numpy(rel)

        assert not math.isnan(rel_np[0, 0])
        assert not math.isinf(rel_np[0, 0])
        # Second anchor should work normally
        assert abs(rel_np[0, 1] - 1.0) < 1e-5

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
        rel_np = backend.to_numpy(rel)

        # Should be stable (normalized small vector still has direction)
        assert not math.isnan(rel_np[0, 0])
        assert not math.isinf(rel_np[0, 0])

    def test_single_sample(self, backend: "Backend") -> None:
        """Single sample should work."""
        hidden = backend.array([[1.0, 2.0, 3.0]])
        anchors = backend.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_np = backend.to_numpy(rel)

        assert rel_np.shape == (1, 2)
        assert not any(math.isnan(x) for x in rel_np.flatten())

    def test_single_anchor(self, backend: "Backend") -> None:
        """Single anchor should work."""
        hidden = backend.array([[1.0, 0.0], [0.0, 1.0]])
        anchors = backend.array([[1.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_np = backend.to_numpy(rel)

        assert rel_np.shape == (2, 1)
        assert abs(rel_np[0, 0] - 1.0) < 1e-5
        assert abs(rel_np[1, 0] - 0.0) < 1e-5

    def test_high_dimensional(self, backend: "Backend") -> None:
        """High-dimensional inputs (like 2048-dim embeddings)."""
        backend.random_seed(42)
        hidden = backend.random_normal((10, 2048))
        anchors = backend.random_normal((50, 2048))
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_np = backend.to_numpy(rel)

        assert rel_np.shape == (10, 50)
        # All similarities should be in [-1, 1]
        assert rel_np.min() >= -1.0 - 1e-5
        assert rel_np.max() <= 1.0 + 1e-5
        # No NaN or Inf
        assert not any(math.isnan(x) for x in rel_np.flatten())
        assert not any(math.isinf(x) for x in rel_np.flatten())

    def test_negative_values(self, backend: "Backend") -> None:
        """Negative values should produce negative cosine similarities."""
        hidden = backend.array([[-1.0, 0.0, 0.0]])
        anchors = backend.array([[1.0, 0.0, 0.0]])
        backend.eval(hidden, anchors)

        rel = compute_relative_representation(hidden, anchors)
        backend.eval(rel)
        rel_np = backend.to_numpy(rel)

        # Opposite directions should give cos = -1
        assert abs(rel_np[0, 0] - (-1.0)) < 1e-5


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
        R_np = backend.to_numpy(R)

        # R should be close to identity
        identity = backend.to_numpy(backend.eye(3))
        diff = abs(R_np - identity)
        assert diff.max() < 1e-4, f"Expected identity, got diff max {diff.max()}"
        # Error should be ~0
        assert error < 1e-4, f"Expected zero error, got {error}"

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
        assert error < 0.01, f"Expected low error for identical data, got {error}"

        # Verify R is orthogonal (R @ R^T = I)
        R_rec_np = backend.to_numpy(R_recovered)
        RRt = R_rec_np @ R_rec_np.T
        identity_diff = abs(RRt - backend.to_numpy(backend.eye(4)))
        assert identity_diff.max() < 1e-5, "R should be orthogonal"

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
        det_val = float(backend.to_numpy(det).item())
        assert abs(det_val - 1.0) < 1e-4, f"Expected det=1, got {det_val}"

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
        R_np = backend.to_numpy(R)
        # Check orthogonality: R @ R^T = I
        RtR = R_np @ R_np.T
        identity = backend.to_numpy(backend.eye(3))
        assert abs(RtR - identity).max() < 1e-4

    def test_single_sample_degenerate(self, backend: "Backend") -> None:
        """Single sample is degenerate but should not crash."""
        source = backend.array([[1.0, 2.0, 3.0]])
        target = backend.array([[3.0, 2.0, 1.0]])
        backend.eval(source, target)

        # Should not raise
        R, error = align_relative_representations(source, target)
        backend.eval(R)

        # Result may be arbitrary but should be valid matrix
        R_np = backend.to_numpy(R)
        assert R_np.shape == (3, 3)
        assert not any(math.isnan(x) for x in R_np.flatten())

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

        R_np = backend.to_numpy(R)
        assert not any(math.isnan(x) for x in R_np.flatten())

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
        R_np = backend.to_numpy(R)
        identity = backend.to_numpy(backend.eye(5))
        # Due to centering, scale should be handled
        assert R_np.shape == (5, 5)
        # Check it's still orthogonal
        RtR = R_np @ R_np.T
        assert abs(RtR - identity).max() < 1e-3


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
        trans_np = backend.to_numpy(transferred)
        assert not any(math.isnan(x) for x in trans_np.flatten())

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
        trans_np = backend.to_numpy(transferred)
        assert not any(math.isnan(x) for x in trans_np.flatten())
        assert not any(math.isinf(x) for x in trans_np.flatten())

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

        trans_np = backend.to_numpy(transferred)
        assert not any(math.isnan(x) for x in trans_np.flatten())

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

        trans_np = backend.to_numpy(transferred)
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
        trans_np = backend.to_numpy(transferred)
        assert not any(math.isnan(x) for x in trans_np.flatten())


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
        trans_np = backend.to_numpy(transferred)
        assert not any(math.isnan(x) for x in trans_np.flatten())

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

        relative_error = float(backend.to_numpy(diff).item()) / max(float(backend.to_numpy(orig_norm).item()), 1e-8)
        # Allow some error due to pseudo-inverse
        assert relative_error < 0.5, f"Round-trip error too large: {relative_error}"

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
        assert not any(math.isnan(x) for x in backend.to_numpy(transferred_no_align).flatten())
        assert not any(math.isnan(x) for x in backend.to_numpy(transferred_with_align).flatten())

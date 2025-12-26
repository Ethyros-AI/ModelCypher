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
Tests for GramAligner - verifies CKA = 1.0 is always achievable.

The core principle: CKA = 1.0 represents perfect alignment of relational
structure between representations. This is always achievable because
concept relationships (like in relativity) are invariant - we just need
to find the right coordinate transformation.
"""

import math
import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import (
    GramAligner,
    AlignmentResult,
    find_alignment,
)
from modelcypher.core.domain.geometry.cka import compute_cka


class TestGramAlignerBasic:
    """Basic tests for GramAligner."""

    def test_identical_activations_achieve_cka_1(self):
        """Identical activations should trivially achieve CKA = 1.0."""
        b = get_default_backend()

        # Create some random activations
        b.random_seed(42)
        activations = b.random_normal((50, 64))
        b.eval(activations)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(activations, activations)

        assert result.achieved_cka >= 0.9999, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"
        assert result.is_perfect

    def test_rotated_activations_achieve_cka_1(self):
        """Rotated activations should achieve CKA = 1.0.

        Since rotation doesn't change Gram matrices, this is a sanity check.
        """
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((50, 64))
        b.eval(source)

        # Create a random orthogonal rotation matrix
        random_matrix = b.random_normal((64, 64))
        U, _, Vt = b.svd(random_matrix)
        rotation = b.matmul(U, Vt)  # Orthogonal matrix
        b.eval(rotation)

        # Rotate the activations
        target = b.matmul(source, rotation)
        b.eval(target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        assert result.achieved_cka >= 0.9999, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"


class TestGramAlignerDifferentDistributions:
    """Tests with activations from different distributions.

    The key claim: CKA = 1.0 is ALWAYS achievable, regardless of how
    different the activations look. The relational structure can always
    be aligned.
    """

    def test_scaled_activations_achieve_cka_1(self):
        """Scaled activations should achieve CKA = 1.0."""
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((50, 64))
        target = source * 2.5  # Scaled version
        b.eval(source, target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        assert result.achieved_cka >= 0.9999, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"

    def test_different_random_seeds_achieve_cka_1(self):
        """Different random activations should still achieve CKA = 1.0.

        This is the core test. Even with completely different activations,
        CKA = 1.0 should be achievable because the transformation exists.
        """
        b = get_default_backend()

        # Generate two completely different activation sets
        b.random_seed(42)
        source = b.random_normal((50, 64))
        b.eval(source)

        b.random_seed(999)
        target = b.random_normal((50, 64))
        b.eval(target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        # This is the key assertion: CKA = 1.0 is ALWAYS achievable
        assert result.achieved_cka >= 0.99, (
            f"Expected CKA ≈ 1.0, got {result.achieved_cka}. "
            "CKA = 1.0 should always be achievable."
        )

    def test_different_dimensions_same_structure_achieve_cka_1(self):
        """Activations with different dimensions but SAME structure should achieve CKA = 1.0.

        The key insight: CKA = 1.0 is achievable when the underlying relational
        structure is the same (which is true for model activations on the same inputs).

        For random activations with completely independent structures, CKA = 1.0
        is NOT achievable due to rank constraints.
        """
        b = get_default_backend()

        # Create base structure (this represents the underlying concept relationships)
        b.random_seed(42)
        base = b.random_normal((50, 20))  # 20-dimensional latent structure
        b.eval(base)

        # Source: project to 32 dimensions (with some noise)
        proj_s = b.random_normal((20, 32))
        b.eval(proj_s)
        source = b.matmul(base, proj_s)
        b.eval(source)

        # Target: project to 128 dimensions (same base, different projection)
        proj_t = b.random_normal((20, 128))
        b.eval(proj_t)
        target = b.matmul(base, proj_t)
        b.eval(target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        # Same underlying structure → CKA = 1.0 should be achievable
        assert result.achieved_cka >= 0.99, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"


class TestGramAlignerTransformation:
    """Tests that the returned transformation actually works."""

    def test_transformed_activations_have_cka_1(self):
        """Verify that applying the transform gives CKA = 1.0."""
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((50, 64))
        b.eval(source)

        b.random_seed(999)
        target = b.random_normal((50, 64))
        b.eval(target)

        result = find_alignment(source, target, b)

        # Apply the transformation
        transform = b.array(result.feature_transform)
        aligned_source = b.matmul(source, transform)
        b.eval(aligned_source)

        # Compute CKA between aligned source and target
        cka_result = compute_cka(aligned_source, target, b)

        assert cka_result.cka >= 0.99, (
            f"After transformation, expected CKA ≈ 1.0, got {cka_result.cka}"
        )


class TestGramAlignerConvergence:
    """Tests for convergence behavior."""

    def test_converges_in_reasonable_iterations(self):
        """Should converge well before max iterations."""
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((30, 32))
        target = b.random_normal((30, 32))
        b.eval(source, target)

        aligner = GramAligner(b, max_iterations=1000)
        result = aligner.find_perfect_alignment(source, target)

        # Should not need many iterations if the algorithm is correct
        assert result.iterations < 500, (
            f"Expected convergence in <500 iterations, took {result.iterations}"
        )


class TestGramAlignerEdgeCases:
    """Edge case tests."""

    def test_small_sample_count(self):
        """Should work with small sample counts."""
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((10, 32))
        target = b.random_normal((10, 32))
        b.eval(source, target)

        result = find_alignment(source, target, b)

        assert result.achieved_cka >= 0.95, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"

    def test_high_dimensional_features(self):
        """Should work with high-dimensional features."""
        b = get_default_backend()

        b.random_seed(42)
        source = b.random_normal((20, 512))
        target = b.random_normal((20, 512))
        b.eval(source, target)

        result = find_alignment(source, target, b)

        assert result.achieved_cka >= 0.95, f"Expected CKA ≈ 1.0, got {result.achieved_cka}"

    def test_sample_mismatch_raises(self):
        """Should raise if sample counts don't match."""
        b = get_default_backend()

        source = b.random_normal((50, 64))
        target = b.random_normal((30, 64))  # Different sample count
        b.eval(source, target)

        aligner = GramAligner(b)

        with pytest.raises(ValueError, match="Sample counts must match"):
            aligner.find_perfect_alignment(source, target)


class TestRelationalInvariance:
    """Tests demonstrating the relativity principle.

    Concept relationships are INVARIANT - only relative positions matter.
    """

    def test_gram_matrices_capture_relationships(self):
        """Verify that Gram matrices capture the relational structure."""
        b = get_default_backend()

        b.random_seed(42)
        activations = b.random_normal((20, 64))
        b.eval(activations)

        # Compute Gram matrix (captures all pairwise relationships)
        gram = b.matmul(activations, b.transpose(activations))
        b.eval(gram)

        # Rotate the activations using a proper orthogonal matrix
        # Create orthogonal matrix from QR decomposition style: U @ U^T = I
        random_matrix = b.random_normal((64, 64))
        U, _, Vt = b.svd(random_matrix)
        # For SVD of square matrix, U and Vt are both orthogonal
        # Use just U for rotation
        b.eval(U)

        rotated = b.matmul(activations, U)
        b.eval(rotated)
        gram_rotated = b.matmul(rotated, b.transpose(rotated))
        b.eval(gram_rotated)

        # Gram matrices should be nearly identical (rotation doesn't change relationships)
        # Note: small numerical differences may occur due to float32 precision
        diff = gram - gram_rotated
        norm = float(b.to_numpy(b.sqrt(b.sum(diff * diff))))
        gram_norm = float(b.to_numpy(b.sqrt(b.sum(gram * gram))))

        relative_error = norm / (gram_norm + 1e-10)
        assert relative_error < 1e-4, f"Rotation should preserve Gram matrix (rel error: {relative_error})"

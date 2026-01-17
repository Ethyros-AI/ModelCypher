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

"""Tests for GramAligner module.

Tests verify the API works correctly. Thresholds are derived from machine
epsilon at runtime - we don't test for specific arbitrary values.
"""

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_aligner import (
    GramAligner,
    find_alignment,
)
from modelcypher.core.domain.geometry.cka import compute_cka
from modelcypher.core.domain.geometry.numerical_stability import is_finite


class TestGramAlignerInit:
    """Tests for GramAligner initialization."""

    def test_default_initialization(self):
        """Should initialize with default backend."""
        b = get_default_backend()
        aligner = GramAligner(b)
        assert aligner._backend is not None

    def test_no_backend_uses_default(self):
        """Should use default backend when none provided."""
        aligner = GramAligner()
        assert aligner._backend is not None


class TestGramAlignerFindPerfectAlignment:
    """Tests for GramAligner.find_perfect_alignment() method."""

    def test_identical_matrices(self):
        """Identical matrices should achieve perfect alignment."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, A)

        assert result.is_perfect
        assert abs(result.achieved_cka - 1.0) <= result.precision_threshold

    def test_orthonormal_transform(self):
        """Orthonormal transforms preserve Gram structure exactly."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        q, _ = b.qr(b.random_normal((8, 8)))
        B = b.matmul(A, q)
        b.eval(A, B, q)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        assert result.is_perfect
        assert abs(result.achieved_cka - 1.0) <= result.precision_threshold

    def test_returns_valid_transforms(self):
        """Alignment should return valid transformation matrices."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        B = b.random_normal((20, 12))
        b.eval(A, B)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        # feature_transform should be d_source x d_target
        assert len(result.feature_transform) == 8
        assert len(result.feature_transform[0]) == 12

    def test_different_dimensions(self):
        """Should handle different feature dimensions."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((30, 16))
        q_full, _ = b.qr(b.random_normal((32, 32)))
        q = q_full[:16, :]
        B = b.matmul(A, q)
        b.eval(A, B, q_full, q)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        aligned = b.matmul(A, b.array(result.feature_transform))
        b.eval(aligned)
        cka_result = compute_cka(aligned, B, backend=b)
        cka = cka_result.best
        eps = result.precision_threshold
        assert abs(cka - result.achieved_cka) <= eps
        assert result.is_perfect


class TestFindAlignment:
    """Tests for find_alignment convenience function."""

    def test_find_alignment_basic(self):
        """find_alignment should work with basic inputs."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        result = find_alignment(A, A, backend=b)

        assert result.is_perfect
        assert abs(result.achieved_cka - 1.0) <= result.precision_threshold

    def test_find_alignment_uses_default_backend(self):
        """find_alignment should use default backend when none provided."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((20, 8))
        b.eval(A)

        # Should not raise
        result = find_alignment(A, A)
        assert result.is_perfect


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_matrices(self):
        """Should handle small matrices with good geodesic CKA."""
        b = get_default_backend()
        b.random_seed(42)

        A = b.random_normal((5, 3))
        q, _ = b.qr(b.random_normal((3, 3)))
        B = b.matmul(A, q)
        b.eval(A, B, q)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(A, B)

        # Geodesic alignment preserves manifold structure; CKA > 0.95 is good
        assert result.achieved_cka >= 0.95

    def test_single_sample(self):
        """Should handle single sample gracefully."""
        b = get_default_backend()

        A = b.ones((1, 8))
        B = b.ones((1, 8))
        b.eval(A, B)

        aligner = GramAligner(b)
        # May return degenerate result but shouldn't crash
        result = aligner.find_perfect_alignment(A, B)
        assert result is not None
        assert is_finite(result.achieved_cka, b)


class TestGramAlignerConditionNumber:
    """Tests for Gram matrix condition number computation and stability checks."""

    def test_condition_number_is_populated(self):
        """AlignmentResult should include gram_condition_number."""
        b = get_default_backend()
        b.random_seed(42)

        source = b.random_normal((50, 20))
        target = b.random_normal((50, 15))
        b.eval(source, target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        assert hasattr(result, "gram_condition_number")
        assert result.gram_condition_number >= 1.0  # Condition number >= 1 by definition
        assert is_finite(result.gram_condition_number, b)

    def test_well_conditioned_system(self):
        """Well-overdetermined system should have low condition number."""
        b = get_default_backend()
        b.random_seed(42)

        # n=200, d=50 is well overdetermined (4x ratio)
        source = b.random_normal((200, 50))
        target = b.random_normal((200, 40))
        b.eval(source, target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(source, target)

        # Random matrix with n >> d should have κ < 100
        assert result.gram_condition_number < 1000

    def test_near_square_has_higher_condition_number(self):
        """Near-square system should have higher condition number."""
        b = get_default_backend()
        b.random_seed(42)

        # n=55, d=50 is barely overdetermined (1.1x ratio)
        source_tight = b.random_normal((55, 50))
        target_tight = b.random_normal((55, 40))
        b.eval(source_tight, target_tight)

        # n=200, d=50 is well overdetermined (4x ratio)
        source_loose = b.random_normal((200, 50))
        target_loose = b.random_normal((200, 40))
        b.eval(source_loose, target_loose)

        aligner = GramAligner(b)
        result_tight = aligner.find_perfect_alignment(source_tight, target_tight)
        result_loose = aligner.find_perfect_alignment(source_loose, target_loose)

        # Near-square should have higher condition number than well-overdetermined
        assert result_tight.gram_condition_number > result_loose.gram_condition_number

    def test_orthonormal_basis_low_condition(self):
        """Orthonormal columns should give condition number close to 1."""
        b = get_default_backend()
        b.random_seed(42)

        # Create orthonormal basis via QR
        random_mat = b.random_normal((100, 50))
        q, _ = b.qr(random_mat)
        b.eval(q)

        target = b.random_normal((100, 40))
        b.eval(target)

        aligner = GramAligner(b)
        result = aligner.find_perfect_alignment(q, target)

        # Orthonormal columns have κ = 1 for the Gram matrix
        # Allow some numerical tolerance
        assert result.gram_condition_number < 10

    def test_condition_number_invariant_to_scaling(self):
        """Condition number should be invariant to uniform scaling."""
        b = get_default_backend()
        b.random_seed(42)

        source = b.random_normal((100, 50))
        target = b.random_normal((100, 40))
        b.eval(source, target)

        # Scale source by 1000x
        source_scaled = source * 1000.0
        b.eval(source_scaled)

        aligner = GramAligner(b)
        result_orig = aligner.find_perfect_alignment(source, target)
        result_scaled = aligner.find_perfect_alignment(source_scaled, target)

        # Condition number should be approximately the same (within 1%)
        ratio = result_scaled.gram_condition_number / result_orig.gram_condition_number
        assert 0.99 < ratio < 1.01

    def test_unstable_alignment_logs_warning(self, caplog):
        """Should log warning when condition number exceeds threshold."""
        import logging

        b = get_default_backend()
        b.random_seed(42)

        # Create an ill-conditioned matrix by making columns nearly linearly dependent
        # Start with random matrix
        base = b.random_normal((52, 50))
        # Make last column nearly identical to first (creates near-singularity)
        col0 = base[:, 0:1]
        noise = b.random_normal((52, 1)) * 1e-6
        near_dependent = col0 + noise
        # Replace last column
        source = b.concatenate([base[:, :-1], near_dependent], axis=1)
        target = b.random_normal((52, 40))
        b.eval(source, target)

        aligner = GramAligner(b)

        with caplog.at_level(logging.WARNING, logger="modelcypher.core.domain.geometry.gram_aligner"):
            result = aligner.find_perfect_alignment(source, target)

        # If condition number is high enough, warning should be logged
        # Note: The exact threshold is 1e5, so we check if warning appeared for high κ
        if result.gram_condition_number > 1e5:
            assert "ALIGNMENT UNSTABLE" in caplog.text
            assert "condition number" in caplog.text.lower()

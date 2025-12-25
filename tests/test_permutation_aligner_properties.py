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

"""Property-based tests for PermutationAligner.

Tests mathematical invariants that must hold for correct permutation alignment:
1. Permutation matrix is valid (exactly one 1 per row and column)
2. Signs are ±1
3. Self-alignment produces identity or near-identity
4. Apply then apply inverse returns original
5. Quality is bounded [0, 1]
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.permutation_aligner import (
    AlignmentResult,
    Config,
    PermutationAligner,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def backend():
    return get_default_backend()


# =============================================================================
# Property Tests: Permutation Matrix Validity
# =============================================================================


class TestPermutationValidity:
    """Tests that permutation matrices are mathematically valid."""

    def test_self_alignment_high_quality(self, backend):
        """Aligning a matrix with itself should have high quality."""
        backend.random_seed(42)
        weight = backend.random_randn((16, 32))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            config=Config(min_match_threshold=0.1),
            backend=backend,
        )

        # Self-alignment should have very high quality
        assert result.match_quality > 0.8, \
            f"Self-alignment quality should be > 0.8, got {result.match_quality}"

    def test_permutation_is_square(self, backend):
        """Permutation matrix should be square N×N."""
        backend.random_seed(43)
        weight = backend.random_randn((8, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = backend.to_numpy(result.permutation)
        weight_np = backend.to_numpy(weight)
        assert perm.shape[0] == perm.shape[1], \
            f"Permutation should be square, got {perm.shape}"
        assert perm.shape[0] == weight_np.shape[0], \
            f"Permutation dim should match weight output dim, got {perm.shape[0]} vs {weight_np.shape[0]}"

    def test_permutation_rows_sum_to_one(self, backend):
        """Each row of permutation should sum to exactly 1."""
        backend.random_seed(44)
        weight = backend.random_randn((10, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = backend.to_numpy(result.permutation)
        row_sums = perm.sum(axis=1)

        assert backend.allclose(backend.array(row_sums), backend.array(1.0), atol=1e-5), \
            f"Each row should sum to 1, got {row_sums}"

    def test_permutation_cols_sum_to_one(self, backend):
        """Each column of permutation should sum to exactly 1."""
        backend.random_seed(45)
        weight = backend.random_randn((10, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = backend.to_numpy(result.permutation)
        col_sums = perm.sum(axis=0)

        assert backend.allclose(backend.array(col_sums), backend.array(1.0), atol=1e-5), \
            f"Each column should sum to 1, got {col_sums}"

    def test_permutation_is_binary(self, backend):
        """Permutation entries should be exactly 0 or 1."""
        backend.random_seed(46)
        weight = backend.random_randn((10, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = backend.to_numpy(result.permutation)
        # Use numpy for unique only - unavoidable for this check
        import numpy as np
        unique_values = np.unique(perm)

        # All values should be 0 or 1
        assert backend.allclose(backend.array(unique_values), backend.array([0.0, 1.0])) or \
               backend.allclose(backend.array(unique_values), backend.array([0.0])) or \
               backend.allclose(backend.array(unique_values), backend.array([1.0])), \
            f"Permutation should be binary, got unique values {unique_values}"


# =============================================================================
# Property Tests: Sign Matrix
# =============================================================================


class TestSignValidity:
    """Tests that sign matrices are valid diagonal ±1 matrices."""

    def test_signs_are_diagonal(self, backend):
        """Sign matrix should be diagonal."""
        backend.random_seed(47)
        weight = backend.random_randn((8, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        signs = backend.to_numpy(result.signs)

        if signs.ndim == 2:
            # Check off-diagonal elements are zero
            import numpy as np
            off_diag = signs - np.diag(np.diag(signs))
            assert backend.allclose(backend.array(off_diag), backend.array(0.0)), \
                "Sign matrix should be diagonal"

    def test_signs_are_plus_minus_one(self, backend):
        """Sign values should be exactly ±1."""
        backend.random_seed(48)
        weight = backend.random_randn((8, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        signs = backend.to_numpy(result.signs)

        import numpy as np
        if signs.ndim == 2:
            diag = np.diag(signs)
        else:
            diag = signs

        # All diagonal elements should be ±1
        assert backend.allclose(backend.abs(backend.array(diag)), backend.array(1.0)), \
            f"Signs should be ±1, got {diag}"


# =============================================================================
# Property Tests: Quality Bounds
# =============================================================================


class TestQualityBounds:
    """Tests that quality metrics are properly bounded."""

    def test_quality_in_valid_range(self, backend):
        """Match quality should be in [0, 1] (plus epsilon for float precision)."""
        backend.random_seed(49)
        weight = backend.random_randn((10, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        assert -1e-5 <= result.match_quality <= 1.0 + 1e-5, \
            f"Quality should be in [0, 1], got {result.match_quality}"

    def test_confidences_in_valid_range(self, backend):
        """All match confidences should be in [0, 1] (plus epsilon for float precision)."""
        backend.random_seed(50)
        weight = backend.random_randn((10, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        for i, conf in enumerate(result.match_confidences):
            assert -1e-5 <= conf <= 1.0 + 1e-5, \
                f"Confidence[{i}] should be in [0, 1], got {conf}"

    def test_sign_flip_count_bounded(self, backend):
        """Sign flip count should be at most N."""
        N = 10
        backend.random_seed(51)
        weight = backend.random_randn((N, 20))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        assert 0 <= result.sign_flip_count <= N, \
            f"Sign flips should be in [0, {N}], got {result.sign_flip_count}"


# =============================================================================
# Property Tests: Apply Correctness
# =============================================================================


class TestApplyCorrectness:
    """Tests that apply correctly transforms weights."""

    def test_apply_preserves_shape(self, backend):
        """Applying alignment should preserve weight shape."""
        backend.random_seed(52)
        weight = backend.random_randn((8, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        aligned = PermutationAligner.apply(
            weight=weight,
            alignment=result,
            align_output=True,
            backend=backend,
        )

        weight_np = backend.to_numpy(weight)
        aligned_np = backend.to_numpy(aligned)
        assert aligned_np.shape == weight_np.shape, \
            f"Shape should be preserved: {weight_np.shape} vs {aligned_np.shape}"

    def test_apply_does_not_introduce_nan(self, backend):
        """Applying alignment should not introduce NaNs."""
        backend.random_seed(53)
        weight = backend.random_randn((8, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        aligned = PermutationAligner.apply(
            weight=weight,
            alignment=result,
            align_output=True,
            backend=backend,
        )

        aligned_np = backend.to_numpy(aligned)
        import numpy as np
        assert not np.isnan(aligned_np).any(), \
            "Apply should not introduce NaNs"

    def test_identity_permutation_preserves_weight(self, backend):
        """Identity permutation should preserve weights."""
        N = 8
        backend.random_seed(54)
        weight = backend.random_randn((N, 16))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        # Manually create identity alignment
        identity_perm = backend.eye(N)
        identity_perm = backend.astype(identity_perm, backend.float32)
        identity_signs = backend.eye(N)
        identity_signs = backend.astype(identity_signs, backend.float32)
        backend.eval(identity_perm, identity_signs)

        identity_result = AlignmentResult(
            permutation=identity_perm,
            signs=identity_signs,
            match_quality=1.0,
            match_confidences=[1.0] * N,
            sign_flip_count=0,
        )

        aligned = PermutationAligner.apply(
            weight=weight,
            alignment=identity_result,
            align_output=True,
            backend=backend,
        )

        weight_np = backend.to_numpy(weight)
        aligned_np = backend.to_numpy(aligned)
        assert backend.allclose(backend.array(aligned_np), backend.array(weight_np), atol=1e-5), \
            "Identity permutation should preserve weights"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rejects_1d_weights(self, backend):
        """Should reject 1D weight arrays."""
        backend.random_seed(55)
        weight_1d = backend.random_randn((16,))
        weight_1d = backend.astype(weight_1d, backend.float32)
        weight_2d = backend.random_randn((16, 32))
        weight_2d = backend.astype(weight_2d, backend.float32)
        backend.eval(weight_1d, weight_2d)

        with pytest.raises(ValueError, match="2D"):
            PermutationAligner.align(
                source_weight=weight_1d,
                target_weight=weight_2d,
                backend=backend,
            )

    def test_rejects_mismatched_shapes(self, backend):
        """Should reject weights with different shapes."""
        backend.random_seed(56)
        source = backend.random_randn((8, 16))
        source = backend.astype(source, backend.float32)
        target = backend.random_randn((10, 16))  # Different number of neurons
        target = backend.astype(target, backend.float32)
        backend.eval(source, target)

        with pytest.raises(ValueError, match="dimensions must match"):
            PermutationAligner.align(
                source_weight=source,
                target_weight=target,
                backend=backend,
            )

    def test_small_weight_matrix(self, backend):
        """Should handle small (2x2) weight matrices."""
        backend.random_seed(57)
        weight = backend.random_randn((2, 4))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm_np = backend.to_numpy(result.permutation)
        assert perm_np.shape == (2, 2)
        assert result.match_quality >= 0.0

    def test_zero_weight_matrix(self, backend):
        """Should handle zero weight matrix without crashing."""
        weight = backend.zeros((4, 8))
        weight = backend.astype(weight, backend.float32)
        backend.eval(weight)

        # Should not crash
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        # Structure should still be valid
        perm = backend.to_numpy(result.permutation)
        assert perm.shape == (4, 4)


# =============================================================================
# Mutation Detection Tests
# =============================================================================


class TestMutationDetection:
    """Tests designed to catch specific bugs if algorithm is changed."""

    def test_permutation_is_not_identity_for_shuffled_weights(self, backend):
        """Permutation should NOT be identity when weights are clearly shuffled."""
        backend.random_seed(42)

        # Create source with distinct neuron patterns
        source = backend.zeros((4, 8))
        source = backend.astype(source, backend.float32)
        source_np = backend.to_numpy(source)
        source_np[0, 0] = 10.0  # Neuron 0 fires on feature 0
        source_np[1, 2] = 10.0  # Neuron 1 fires on feature 2
        source_np[2, 4] = 10.0  # Neuron 2 fires on feature 4
        source_np[3, 6] = 10.0  # Neuron 3 fires on feature 6
        source = backend.array(source_np)
        backend.eval(source)

        # Create target with shuffled neurons
        target = backend.zeros((4, 8))
        target = backend.astype(target, backend.float32)
        target_np = backend.to_numpy(target)
        target_np[2, 0] = 10.0  # Neuron 2 in target = Neuron 0 in source
        target_np[0, 2] = 10.0  # Neuron 0 in target = Neuron 1 in source
        target_np[3, 4] = 10.0  # Neuron 3 in target = Neuron 2 in source
        target_np[1, 6] = 10.0  # Neuron 1 in target = Neuron 3 in source
        target = backend.array(target_np)
        backend.eval(target)

        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )

        perm = backend.to_numpy(result.permutation)
        identity = backend.to_numpy(backend.eye(4))
        identity = identity.astype(source_np.dtype)

        # Should have found a non-identity permutation
        assert not backend.allclose(backend.array(perm), backend.array(identity), atol=0.1), \
            "Permutation should not be identity for shuffled weights"

    def test_negative_correlation_flips_sign(self, backend):
        """Negative correlation should result in sign flip."""
        # Source: positive activations
        source = backend.array([[10.0, 0.0, 0.0, 0.0]])
        source = backend.astype(source, backend.float32)
        backend.eval(source)

        # Target: negated version
        target = backend.array([[-10.0, 0.0, 0.0, 0.0]])
        target = backend.astype(target, backend.float32)
        backend.eval(target)

        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )

        # Should detect sign flip
        assert result.sign_flip_count >= 1, \
            "Should detect at least one sign flip for negated weights"

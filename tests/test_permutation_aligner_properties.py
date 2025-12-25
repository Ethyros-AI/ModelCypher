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

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain.geometry.permutation_aligner import (
    AlignmentResult,
    Config,
    PermutationAligner,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockBackend:
    """Minimal numpy backend for testing."""

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float32)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, arr, axes=None):
        return np.transpose(arr, axes)

    def astype(self, arr, dtype):
        if dtype == "float32":
            return arr.astype(np.float32)
        return arr

    def reshape(self, arr, shape):
        return np.reshape(arr, shape)

    def eval(self, *arrays):
        pass

    def to_numpy(self, arr):
        return np.asarray(arr)

    def sum(self, arr, axis=None, keepdims=False):
        return np.sum(arr, axis=axis, keepdims=keepdims)

    def sqrt(self, arr):
        return np.sqrt(arr)

    def abs(self, arr):
        return np.abs(arr)

    def argmax(self, arr, axis=None):
        return np.argmax(arr, axis=axis)

    def max(self, arr, axis=None):
        return np.max(arr, axis=axis)

    def arange(self, n):
        return np.arange(n)

    def diag(self, arr):
        return np.diag(arr)

    def take(self, arr, indices, axis=0):
        return np.take(arr, indices, axis=axis)


@pytest.fixture
def backend():
    return MockBackend()


# =============================================================================
# Property Tests: Permutation Matrix Validity
# =============================================================================


class TestPermutationValidity:
    """Tests that permutation matrices are mathematically valid."""

    def test_self_alignment_high_quality(self, backend):
        """Aligning a matrix with itself should have high quality."""
        weight = np.random.randn(16, 32).astype(np.float32)
        
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
        weight = np.random.randn(8, 16).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        perm = backend.to_numpy(result.permutation)
        assert perm.shape[0] == perm.shape[1], \
            f"Permutation should be square, got {perm.shape}"
        assert perm.shape[0] == weight.shape[0], \
            f"Permutation dim should match weight output dim, got {perm.shape[0]} vs {weight.shape[0]}"

    def test_permutation_rows_sum_to_one(self, backend):
        """Each row of permutation should sum to exactly 1."""
        weight = np.random.randn(10, 20).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        perm = backend.to_numpy(result.permutation)
        row_sums = perm.sum(axis=1)
        
        assert np.allclose(row_sums, 1.0, atol=1e-5), \
            f"Each row should sum to 1, got {row_sums}"

    def test_permutation_cols_sum_to_one(self, backend):
        """Each column of permutation should sum to exactly 1."""
        weight = np.random.randn(10, 20).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        perm = backend.to_numpy(result.permutation)
        col_sums = perm.sum(axis=0)
        
        assert np.allclose(col_sums, 1.0, atol=1e-5), \
            f"Each column should sum to 1, got {col_sums}"

    def test_permutation_is_binary(self, backend):
        """Permutation entries should be exactly 0 or 1."""
        weight = np.random.randn(10, 20).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        perm = backend.to_numpy(result.permutation)
        unique_values = np.unique(perm)
        
        # All values should be 0 or 1
        assert np.allclose(unique_values, np.array([0.0, 1.0])) or \
               np.allclose(unique_values, np.array([0.0])) or \
               np.allclose(unique_values, np.array([1.0])), \
            f"Permutation should be binary, got unique values {unique_values}"


# =============================================================================
# Property Tests: Sign Matrix
# =============================================================================


class TestSignValidity:
    """Tests that sign matrices are valid diagonal ±1 matrices."""

    def test_signs_are_diagonal(self, backend):
        """Sign matrix should be diagonal."""
        weight = np.random.randn(8, 16).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        signs = backend.to_numpy(result.signs)
        
        if signs.ndim == 2:
            # Check off-diagonal elements are zero
            off_diag = signs - np.diag(np.diag(signs))
            assert np.allclose(off_diag, 0.0), \
                "Sign matrix should be diagonal"

    def test_signs_are_plus_minus_one(self, backend):
        """Sign values should be exactly ±1."""
        weight = np.random.randn(8, 16).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        signs = backend.to_numpy(result.signs)
        
        if signs.ndim == 2:
            diag = np.diag(signs)
        else:
            diag = signs
        
        # All diagonal elements should be ±1
        assert np.allclose(np.abs(diag), 1.0), \
            f"Signs should be ±1, got {diag}"


# =============================================================================
# Property Tests: Quality Bounds
# =============================================================================


class TestQualityBounds:
    """Tests that quality metrics are properly bounded."""

    def test_quality_in_valid_range(self, backend):
        """Match quality should be in [0, 1] (plus epsilon for float precision)."""
        weight = np.random.randn(10, 20).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        assert -1e-5 <= result.match_quality <= 1.0 + 1e-5, \
            f"Quality should be in [0, 1], got {result.match_quality}"

    def test_confidences_in_valid_range(self, backend):
        """All match confidences should be in [0, 1] (plus epsilon for float precision)."""
        weight = np.random.randn(10, 20).astype(np.float32)
        
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
        weight = np.random.randn(N, 20).astype(np.float32)
        
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
        weight = np.random.randn(8, 16).astype(np.float32)
        
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
        
        assert aligned.shape == weight.shape, \
            f"Shape should be preserved: {weight.shape} vs {aligned.shape}"

    def test_apply_does_not_introduce_nan(self, backend):
        """Applying alignment should not introduce NaNs."""
        weight = np.random.randn(8, 16).astype(np.float32)
        
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
        
        assert not np.isnan(aligned).any(), \
            "Apply should not introduce NaNs"

    def test_identity_permutation_preserves_weight(self, backend):
        """Identity permutation should preserve weights."""
        N = 8
        weight = np.random.randn(N, 16).astype(np.float32)
        
        # Manually create identity alignment
        identity_perm = np.eye(N, dtype=np.float32)
        identity_signs = np.eye(N, dtype=np.float32)
        
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
        
        assert np.allclose(aligned, weight, atol=1e-5), \
            "Identity permutation should preserve weights"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rejects_1d_weights(self, backend):
        """Should reject 1D weight arrays."""
        weight_1d = np.random.randn(16).astype(np.float32)
        weight_2d = np.random.randn(16, 32).astype(np.float32)
        
        with pytest.raises(ValueError, match="2D"):
            PermutationAligner.align(
                source_weight=weight_1d,
                target_weight=weight_2d,
                backend=backend,
            )

    def test_rejects_mismatched_shapes(self, backend):
        """Should reject weights with different shapes."""
        source = np.random.randn(8, 16).astype(np.float32)
        target = np.random.randn(10, 16).astype(np.float32)  # Different number of neurons
        
        with pytest.raises(ValueError, match="dimensions must match"):
            PermutationAligner.align(
                source_weight=source,
                target_weight=target,
                backend=backend,
            )

    def test_small_weight_matrix(self, backend):
        """Should handle small (2x2) weight matrices."""
        weight = np.random.randn(2, 4).astype(np.float32)
        
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )
        
        assert result.permutation.shape == (2, 2)
        assert result.match_quality >= 0.0

    def test_zero_weight_matrix(self, backend):
        """Should handle zero weight matrix without crashing."""
        weight = np.zeros((4, 8), dtype=np.float32)
        
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
        np.random.seed(42)
        
        # Create source with distinct neuron patterns
        source = np.zeros((4, 8), dtype=np.float32)
        source[0, 0] = 10.0  # Neuron 0 fires on feature 0
        source[1, 2] = 10.0  # Neuron 1 fires on feature 2
        source[2, 4] = 10.0  # Neuron 2 fires on feature 4
        source[3, 6] = 10.0  # Neuron 3 fires on feature 6
        
        # Create target with shuffled neurons
        target = np.zeros((4, 8), dtype=np.float32)
        target[2, 0] = 10.0  # Neuron 2 in target = Neuron 0 in source
        target[0, 2] = 10.0  # Neuron 0 in target = Neuron 1 in source
        target[3, 4] = 10.0  # Neuron 3 in target = Neuron 2 in source
        target[1, 6] = 10.0  # Neuron 1 in target = Neuron 3 in source
        
        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )
        
        perm = backend.to_numpy(result.permutation)
        identity = np.eye(4, dtype=np.float32)
        
        # Should have found a non-identity permutation
        assert not np.allclose(perm, identity, atol=0.1), \
            "Permutation should not be identity for shuffled weights"

    def test_negative_correlation_flips_sign(self, backend):
        """Negative correlation should result in sign flip."""
        # Source: positive activations
        source = np.array([
            [10.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        # Target: negated version
        target = np.array([
            [-10.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        
        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )
        
        # Should detect sign flip
        assert result.sign_flip_count >= 1, \
            "Should detect at least one sign flip for negated weights"

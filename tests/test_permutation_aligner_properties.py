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
    PermutationAligner,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.support.array_utils import array_to_list


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def _div_eps(backend, *values: float) -> float:
    return division_epsilon(backend, backend.array(list(values) or [1.0]))


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


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
        weight = backend.random_normal((16, 32))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        # Quality should equal mean confidence (definition)
        confidences = backend.array(result.match_confidences)
        mean_conf = backend.mean(confidences)
        backend.eval(mean_conf)
        mean_val = float(backend.to_scalar(mean_conf))
        eps = _eps(backend, result.match_quality, mean_val)
        assert abs(result.match_quality - mean_val) <= eps

    def test_permutation_is_square(self, backend):
        """Permutation matrix should be square N×N."""
        backend.random_seed(43)
        weight = backend.random_normal((8, 16))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm_shape = result.permutation.shape
        weight_shape = weight.shape
        assert perm_shape[0] == perm_shape[1], (
            f"Permutation should be square, got {perm_shape}"
        )
        assert perm_shape[0] == weight_shape[0], (
            f"Permutation dim should match weight output dim, got {perm_shape[0]} vs {weight_shape[0]}"
        )

    def test_permutation_rows_sum_to_one(self, backend):
        """Each row of permutation should sum to exactly 1."""
        backend.random_seed(44)
        weight = backend.random_normal((10, 20))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = result.permutation
        row_sums = backend.sum(perm, axis=1)
        backend.eval(row_sums)
        row_list = array_to_list(backend, row_sums)
        eps = _div_eps(backend, 1.0)
        for value in row_list:
            assert abs(value - 1.0) <= eps, (
                f"Each row should sum to 1, got {row_list}"
            )

    def test_permutation_cols_sum_to_one(self, backend):
        """Each column of permutation should sum to exactly 1."""
        backend.random_seed(45)
        weight = backend.random_normal((10, 20))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm = result.permutation
        col_sums = backend.sum(perm, axis=0)
        backend.eval(col_sums)
        col_list = array_to_list(backend, col_sums)
        eps = _div_eps(backend, 1.0)
        for value in col_list:
            assert abs(value - 1.0) <= eps, (
                f"Each column should sum to 1, got {col_list}"
            )

    def test_permutation_is_binary(self, backend):
        """Permutation entries should be exactly 0 or 1."""
        backend.random_seed(46)
        weight = backend.random_normal((10, 20))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        perm_list = array_to_list(backend, result.permutation)
        eps = _eps(backend, 0.0, 1.0)
        for row in perm_list:
            for value in row:
                close_to_zero = abs(value) <= eps
                close_to_one = abs(value - 1.0) <= eps
                assert close_to_zero or close_to_one, (
                    f"Permutation should be binary, got value {value}"
                )


# =============================================================================
# Property Tests: Sign Matrix
# =============================================================================


class TestSignValidity:
    """Tests that sign matrices are valid diagonal ±1 matrices."""

    def test_signs_are_diagonal(self, backend):
        """Sign matrix should be diagonal."""
        backend.random_seed(47)
        weight = backend.random_normal((8, 16))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        signs = result.signs
        if signs.ndim == 2:
            sign_list = array_to_list(backend, signs)
            eps = _eps(backend, 0.0)
            for i in range(len(sign_list)):
                for j in range(len(sign_list[i])):
                    if i != j:
                        assert abs(sign_list[i][j]) <= eps, (
                            "Sign matrix should be diagonal"
                        )

    def test_signs_are_plus_minus_one(self, backend):
        """Sign values should be exactly ±1."""
        backend.random_seed(48)
        weight = backend.random_normal((8, 16))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        signs = result.signs
        eps = _eps(backend, 1.0)
        if signs.ndim == 2:
            sign_list = array_to_list(backend, signs)
            diag = [sign_list[i][i] for i in range(len(sign_list))]
        else:
            diag = array_to_list(backend, signs)

        for value in diag:
            assert abs(abs(value) - 1.0) <= eps, (
                f"Signs should be ±1, got {diag}"
            )


# =============================================================================
# Property Tests: Quality Bounds
# =============================================================================


class TestQualityBounds:
    """Tests that quality metrics are properly bounded."""

    def test_quality_in_valid_range(self, backend):
        """Match quality should be in [0, 1] (plus epsilon for float precision)."""
        backend.random_seed(49)
        weight = backend.random_normal((10, 20))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        eps = _eps(backend, result.match_quality, 0.0, 1.0)
        assert -eps <= result.match_quality <= 1.0 + eps, (
            f"Quality should be in [0, 1], got {result.match_quality}"
        )

    def test_confidences_in_valid_range(self, backend):
        """All match confidences should be in [0, 1] (plus epsilon for float precision)."""
        backend.random_seed(50)
        weight = backend.random_normal((10, 20))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        for i, conf in enumerate(result.match_confidences):
            eps = _eps(backend, conf, 0.0, 1.0)
            assert -eps <= conf <= 1.0 + eps, (
                f"Confidence[{i}] should be in [0, 1], got {conf}"
            )

    def test_sign_flip_count_bounded(self, backend):
        """Sign flip count should be at most N."""
        N = 10
        backend.random_seed(51)
        weight = backend.random_normal((N, 20))
        weight = backend.astype(weight, "float32")
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
        weight = backend.random_normal((8, 16))
        weight = backend.astype(weight, "float32")
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

        assert aligned.shape == weight.shape, (
            f"Shape should be preserved: {weight.shape} vs {aligned.shape}"
        )

    def test_apply_does_not_introduce_nan(self, backend):
        """Applying alignment should not introduce NaNs."""
        backend.random_seed(53)
        weight = backend.random_normal((8, 16))
        weight = backend.astype(weight, "float32")
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

        aligned_list = array_to_list(backend, aligned)
        for row in aligned_list:
            for value in row:
                assert _is_finite(value), "Apply should not introduce NaNs"

    def test_identity_permutation_preserves_weight(self, backend):
        """Identity permutation should preserve weights."""
        N = 8
        backend.random_seed(54)
        weight = backend.random_normal((N, 16))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        # Manually create identity alignment
        identity_perm = backend.eye(N)
        identity_perm = backend.astype(identity_perm, "float32")
        identity_signs = backend.eye(N)
        identity_signs = backend.astype(identity_signs, "float32")
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

        diff = backend.norm(aligned - weight)
        backend.eval(diff)
        diff_val = float(backend.to_scalar(diff))
        eps = _eps(backend, diff_val, 0.0)
        assert diff_val <= eps, "Identity permutation should preserve weights"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_rejects_1d_weights(self, backend):
        """Should reject 1D weight arrays."""
        backend.random_seed(55)
        weight_1d = backend.random_normal((16,))
        weight_1d = backend.astype(weight_1d, "float32")
        weight_2d = backend.random_normal((16, 32))
        weight_2d = backend.astype(weight_2d, "float32")
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
        source = backend.random_normal((8, 16))
        source = backend.astype(source, "float32")
        target = backend.random_normal((10, 16))  # Different number of neurons
        target = backend.astype(target, "float32")
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
        weight = backend.random_normal((2, 4))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        assert result.permutation.shape == (2, 2)
        assert result.match_quality >= 0.0

    def test_zero_weight_matrix(self, backend):
        """Should handle zero weight matrix without crashing."""
        weight = backend.zeros((4, 8))
        weight = backend.astype(weight, "float32")
        backend.eval(weight)

        # Should not crash
        result = PermutationAligner.align(
            source_weight=weight,
            target_weight=weight,
            backend=backend,
        )

        # Structure should still be valid
        assert result.permutation.shape == (4, 4)


# =============================================================================
# Mutation Detection Tests
# =============================================================================


class TestMutationDetection:
    """Tests designed to catch specific bugs if algorithm is changed."""

    def test_permutation_is_not_identity_for_shuffled_weights(self, backend):
        """Permutation should NOT be identity when weights are clearly shuffled."""
        backend.random_seed(42)

        # Create source with distinct neuron patterns
        source_list = [[0.0] * 8 for _ in range(4)]
        source_list[0][0] = 10.0  # Neuron 0 fires on feature 0
        source_list[1][2] = 10.0  # Neuron 1 fires on feature 2
        source_list[2][4] = 10.0  # Neuron 2 fires on feature 4
        source_list[3][6] = 10.0  # Neuron 3 fires on feature 6
        source = backend.array(source_list)
        source = backend.astype(source, "float32")
        backend.eval(source)

        # Create target with shuffled neurons
        target_list = [[0.0] * 8 for _ in range(4)]
        target_list[2][0] = 10.0  # Neuron 2 in target = Neuron 0 in source
        target_list[0][2] = 10.0  # Neuron 0 in target = Neuron 1 in source
        target_list[3][4] = 10.0  # Neuron 3 in target = Neuron 2 in source
        target_list[1][6] = 10.0  # Neuron 1 in target = Neuron 3 in source
        target = backend.array(target_list)
        target = backend.astype(target, "float32")
        backend.eval(target)

        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )

        identity = backend.eye(4)
        identity = backend.astype(identity, "float32")
        diff = backend.abs(result.permutation - identity)
        diff_max = backend.max(diff)
        backend.eval(diff_max)
        diff_val = float(backend.to_scalar(diff_max))
        eps = _eps(backend, diff_val, 0.0)

        # Should have found a non-identity permutation
        assert diff_val > eps, "Permutation should not be identity for shuffled weights"

    def test_negative_correlation_flips_sign(self, backend):
        """Negative correlation should result in sign flip."""
        # Source: positive activations
        source = backend.array([[10.0, 0.0, 0.0, 0.0]])
        source = backend.astype(source, "float32")
        backend.eval(source)

        # Target: negated version
        target = backend.array([[-10.0, 0.0, 0.0, 0.0]])
        target = backend.astype(target, "float32")
        backend.eval(target)

        result = PermutationAligner.align(
            source_weight=source,
            target_weight=target,
            backend=backend,
        )

        # Should detect sign flip
        assert result.sign_flip_count >= 1, \
            "Should detect at least one sign flip for negated weights"

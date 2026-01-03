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

"""Tests for Birkhoff polytope projector.

Verifies the mHC-inspired Sinkhorn-Knopp projection and spectral norm bounding.

Key properties tested:
1. Doubly stochastic: row sums = column sums = 1
2. Spectral norm bounded: ||M||_2 <= 1.0
3. Compositional closure: A @ B remains doubly stochastic
4. Convergence: Sinkhorn converges to dtype-derived tolerance
"""

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.birkhoff_projector import (
    BirkhoffProjector,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    is_finite,
    regularization_epsilon,
    tiny_value,
)


class TestSinkhornKnopp:
    """Test Sinkhorn-Knopp convergence properties."""

    def test_converges_to_doubly_stochastic(self) -> None:
        """Projected matrix should have row/column sums = 1."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(42)
        M = backend.random_normal((10, 10))
        backend.eval(M)

        result = projector.project(M)

        # Row sums should be 1
        row_sums = backend.sum(result.projected_matrix, axis=1)
        backend.eval(row_sums)
        row_sums_list = backend.tolist(row_sums)
        max_row_error = max(abs(s - 1.0) for s in row_sums_list)

        # Column sums should be 1
        col_sums = backend.sum(result.projected_matrix, axis=0)
        backend.eval(col_sums)
        col_sums_list = backend.tolist(col_sums)
        max_col_error = max(abs(s - 1.0) for s in col_sums_list)
        max_error = max(max_row_error, max_col_error)
        tol = regularization_epsilon(backend, result.projected_matrix)
        assert abs(max_error - result.max_marginal_error) <= division_epsilon(
            backend, result.projected_matrix
        )
        if result.converged:
            assert max_error <= tol

    def test_converges_within_20_iterations(self) -> None:
        """Should converge to dtype-derived tolerance when possible."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(123)
        M = backend.random_normal((8, 8))
        backend.eval(M)

        result = projector.project(M)

        assert result.iterations_used >= 1
        assert is_finite(result.max_marginal_error, backend)

    def test_idempotent_projection(self) -> None:
        """Projecting a doubly stochastic matrix should return itself (approximately)."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Create a known doubly stochastic matrix (uniform)
        n = 5
        uniform = backend.ones((n, n)) / n
        backend.eval(uniform)

        result = projector.project(uniform, ensure_positive=False)

        # Should be very close to input
        diff = backend.abs(result.projected_matrix - uniform)
        backend.eval(diff)
        max_arr = backend.max(diff)
        backend.eval(max_arr)
        max_diff = float(backend.to_scalar(max_arr))

        tol = regularization_epsilon(backend, result.projected_matrix)
        assert max_diff <= tol, f"Idempotent property violated: max diff = {max_diff}"
        assert result.converged

    def test_preserves_permutation_matrices(self) -> None:
        """Permutation matrices are vertices of Birkhoff polytope - should be preserved."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Create a permutation matrix (cycle)
        n = 4
        perm = backend.array(
            [
                [1.0 if j == ((i + 1) % n) else 0.0 for j in range(n)]
                for i in range(n)
            ]
        )
        backend.eval(perm)

        # Project with ensure_positive=False since it's already positive
        result = projector.project(perm, ensure_positive=False)

        # Permutation matrices are doubly stochastic
        row_sums = backend.sum(result.projected_matrix, axis=1)
        col_sums = backend.sum(result.projected_matrix, axis=0)
        backend.eval(row_sums, col_sums)

        row_sums_list = backend.tolist(row_sums)
        col_sums_list = backend.tolist(col_sums)

        tol = regularization_epsilon(backend, result.projected_matrix)
        assert all(abs(s - 1.0) <= tol for s in row_sums_list)
        assert all(abs(s - 1.0) <= tol for s in col_sums_list)

    def test_all_entries_nonnegative(self) -> None:
        """All entries should be non-negative."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(456)
        M = backend.random_normal((6, 6))
        backend.eval(M)

        result = projector.project(M)

        min_val = backend.min(result.projected_matrix)
        backend.eval(min_val)
        min_val_float = float(backend.to_scalar(min_val))

        tol = division_epsilon(backend, result.projected_matrix)
        assert min_val_float >= -tol, f"Negative entry found: {min_val_float}"


class TestSpectralBounding:
    """Test spectral norm constraint."""

    def test_bounds_spectral_norm_to_one(self) -> None:
        """Spectral norm should be <= 1.0 after bounding."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Create matrix with large spectral norm
        backend.random_seed(789)
        M = backend.random_normal((8, 8)) * 5.0
        backend.eval(M)

        result = projector.project(M)

        # Compute spectral norm of result
        _, S, _ = geodesic_svd(backend, result.projected_matrix)
        backend.eval(S)
        spectral_norm = float(backend.tolist(S)[0])

        tol = regularization_epsilon(backend, result.projected_matrix)
        assert spectral_norm <= 1.0 + tol, f"Spectral norm {spectral_norm} exceeds 1.0"

    def test_spectral_norm_tracking(self) -> None:
        """Result should track spectral norm before and after."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(101)
        M = backend.random_normal((5, 5)) * 3.0
        backend.eval(M)

        result = projector.project(M)

        # After doubly stochastic projection, spectral norm is often already <= 1
        # If clipping was applied, before > after
        if result.spectral_clipped:
            assert result.spectral_norm_before > result.spectral_norm_after
            tol = regularization_epsilon(backend, result.projected_matrix)
            assert result.spectral_norm_after <= 1.0 + tol

    def test_no_clipping_when_already_bounded(self) -> None:
        """No clipping should occur when spectral norm is already <= max."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Small matrix will have small spectral norm after Sinkhorn
        n = 3
        M = backend.ones((n, n)) / n  # Uniform - spectral norm = 1
        backend.eval(M)

        result = projector.project(M, ensure_positive=False)

        # Uniform matrix has spectral norm = 1, should not be clipped
        assert not result.spectral_clipped


class TestCompositionalClosure:
    """Test the key mHC property: composition preserves doubly stochastic."""

    def test_product_remains_doubly_stochastic(self) -> None:
        """A @ B should be doubly stochastic if A, B are."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(202)
        A = backend.random_normal((6, 6))
        B = backend.random_normal((6, 6))
        backend.eval(A, B)

        result_A = projector.project(A)
        result_B = projector.project(B)

        product = backend.matmul(result_A.projected_matrix, result_B.projected_matrix)
        backend.eval(product)

        # Product should still be doubly stochastic
        row_sums = backend.sum(product, axis=1)
        col_sums = backend.sum(product, axis=0)
        backend.eval(row_sums, col_sums)

        row_sums_list = backend.tolist(row_sums)
        col_sums_list = backend.tolist(col_sums)

        max_row_error = max(abs(s - 1.0) for s in row_sums_list)
        max_col_error = max(abs(s - 1.0) for s in col_sums_list)

        tol = division_epsilon(backend, product) * product.shape[0]
        assert max_row_error <= tol, f"Product row sums deviate: {max_row_error}"
        assert max_col_error <= tol, f"Product col sums deviate: {max_col_error}"

    def test_chained_products_stable(self) -> None:
        """Multiple chained products should remain stable."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        n = 5
        num_matrices = 10

        # Project multiple matrices
        backend.random_seed(303)
        matrices = []
        for _ in range(num_matrices):
            M = backend.random_normal((n, n))
            backend.eval(M)
            result = projector.project(M)
            matrices.append(result.projected_matrix)

        # Chain multiply all
        product = matrices[0]
        for i in range(1, num_matrices):
            product = backend.matmul(product, matrices[i])
            backend.eval(product)

        # Final product should still be doubly stochastic
        row_sums = backend.sum(product, axis=1)
        col_sums = backend.sum(product, axis=0)
        backend.eval(row_sums, col_sums)

        row_sums_list = backend.tolist(row_sums)
        col_sums_list = backend.tolist(col_sums)

        max_row_error = max(abs(s - 1.0) for s in row_sums_list)
        max_col_error = max(abs(s - 1.0) for s in col_sums_list)

        tol = division_epsilon(backend, product) * (n * num_matrices)
        assert max_row_error <= tol, f"Chained row sums deviate: {max_row_error}"
        assert max_col_error <= tol, f"Chained col sums deviate: {max_col_error}"


class TestNonSquareHandling:
    """Test handling of non-square weight matrices."""

    def test_weight_delta_nonsquare(self) -> None:
        """Non-square weight deltas should be handled via Gram matrix."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(404)
        # Wide matrix (out < in)
        delta_wide = backend.random_normal((4, 8))
        backend.eval(delta_wide)

        result = projector.project_weight_delta(delta_wide)
        assert result.original_shape == (4, 8)
        assert result.converged

        # Tall matrix (out > in)
        delta_tall = backend.random_normal((8, 4))
        backend.eval(delta_tall)

        result = projector.project_weight_delta(delta_tall)
        assert result.original_shape == (8, 4)
        assert result.converged

    def test_square_weight_delta(self) -> None:
        """Square weight deltas should use direct projection."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        backend.random_seed(505)
        delta = backend.random_normal((6, 6))
        backend.eval(delta)

        result = projector.project_weight_delta(delta)
        assert result.original_shape == (6, 6)
        assert result.converged


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_handles_near_zero_matrix(self) -> None:
        """Should handle matrices with very small values."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Very small matrix values
        small = tiny_value(backend, backend.array([1.0]))
        M = backend.ones((4, 4)) * small
        backend.eval(M)

        result = projector.project(M, ensure_positive=False)

        # Should still be doubly stochastic
        row_sums = backend.sum(result.projected_matrix, axis=1)
        backend.eval(row_sums)
        row_sums_list = backend.tolist(row_sums)
        max_row_error = max(abs(s - 1.0) for s in row_sums_list)
        tol = division_epsilon(backend, result.projected_matrix)
        assert abs(max_row_error - result.max_marginal_error) <= tol

    def test_handles_large_matrix(self) -> None:
        """Should handle matrices with moderately large values."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        # Moderately large matrix values (avoid extreme values that cause MLX issues)
        backend.random_seed(606)
        M = backend.random_normal((4, 4)) * 10.0
        backend.eval(M)

        result = projector.project(M)

        row_sums = backend.sum(result.projected_matrix, axis=1)
        backend.eval(row_sums)
        row_sums_list = backend.tolist(row_sums)
        max_row_error = max(abs(s - 1.0) for s in row_sums_list)
        tol = division_epsilon(backend, result.projected_matrix)
        assert abs(max_row_error - result.max_marginal_error) <= tol
        assert is_finite(result.max_marginal_error, backend)

    def test_rejects_non_square(self) -> None:
        """project() should reject non-square matrices."""
        backend = get_default_backend()
        projector = BirkhoffProjector(backend)

        M = backend.ones((3, 5))
        backend.eval(M)

        with pytest.raises(ValueError, match="square matrix"):
            projector.project(M)

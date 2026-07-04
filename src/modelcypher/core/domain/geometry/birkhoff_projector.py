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

"""Birkhoff polytope projector for matrix normalization.

Projects matrices onto the set of doubly stochastic matrices using
Sinkhorn-Knopp normalization and optional spectral norm bounding.

Reference:
    - DeepSeek mHC (Manifold-Constrained Hyper-Connections, arXiv:2512.24880)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    regularization_epsilon,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

# Spectral norm bound from DeepSeek mHC paper (arXiv:2512.24880).
_MAX_SPECTRAL_NORM = 1.0  # Spectral norm bound used in the paper


@dataclass
class BirkhoffProjectionResult:
    """Result of projecting a matrix onto the Birkhoff polytope."""

    # The projected doubly stochastic matrix
    projected_matrix: Any

    # Whether Sinkhorn-Knopp converged within max_iterations
    converged: bool

    # Number of Sinkhorn iterations used
    iterations_used: int

    # Final max deviation of row/column sums from 1.0
    max_marginal_error: float

    # Spectral norm before and after bounding
    spectral_norm_before: float
    spectral_norm_after: float

    # Whether spectral clipping was applied
    spectral_clipped: bool

    # Original matrix shape (for non-square handling)
    original_shape: tuple[int, ...]


class BirkhoffProjector:
    """
    Projects matrices onto the Birkhoff polytope (doubly stochastic matrices).

    Implements Sinkhorn-Knopp normalization and optional spectral norm bounding.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def project(
        self,
        matrix: "Array",
        *,
        ensure_positive: bool = True,
    ) -> BirkhoffProjectionResult:
        """Project a square matrix onto the Birkhoff polytope.

        Args:
            matrix: Square matrix [n, n] to project.
            ensure_positive: If True, apply exp() to ensure positivity
                            before Sinkhorn (required for convergence).

        Returns:
            BirkhoffProjectionResult with the doubly stochastic matrix.

        Raises:
            ValueError: If matrix is not square.
        """
        backend = self._backend
        matrix = backend.array(matrix)
        backend.eval(matrix)

        original_shape = tuple(int(d) for d in matrix.shape)

        if len(original_shape) != 2 or original_shape[0] != original_shape[1]:
            raise ValueError(
                f"Birkhoff projection requires square matrix, got shape {original_shape}"
            )

        n = original_shape[0]

        # Ensure positivity for Sinkhorn convergence
        if ensure_positive:
            # Use exp() like DeepSeek mHC (Eq. 9 in the paper)
            # This maps any real matrix to positive values
            M = backend.exp(matrix)
        else:
            # Assume already positive, just add floor for numerical stability
            floor = tiny_value(backend, matrix)
            M = backend.maximum(matrix, backend.full(matrix.shape, floor))
        backend.eval(M)

        # Convergence threshold derived from dtype (sqrt of machine epsilon)
        threshold = regularization_epsilon(backend, matrix)

        # Run Sinkhorn-Knopp iteration
        M, iterations, max_error = self._sinkhorn_knopp(M, n, threshold)

        # Compute spectral norm before bounding
        spectral_norm_before = self._compute_spectral_norm(M)

        # Always enforce spectral norm bound (mHC paper requirement)
        M, spectral_clipped = self.bound_spectral_norm(M, _MAX_SPECTRAL_NORM)
        backend.eval(M)

        spectral_norm_after = self._compute_spectral_norm(M)

        # Spectral bounding changes the returned matrix. Report the marginals
        # of that matrix, not the pre-bound Sinkhorn iterate.
        ones = backend.ones((n,))
        row_error = backend.max(backend.abs(backend.sum(M, axis=1) - ones))
        col_error = backend.max(backend.abs(backend.sum(M, axis=0) - ones))
        backend.eval(row_error, col_error)
        max_error = max(
            float(backend.to_scalar(row_error)),
            float(backend.to_scalar(col_error)),
        )
        converged = max_error < threshold

        return BirkhoffProjectionResult(
            projected_matrix=M,
            converged=converged,
            iterations_used=iterations,
            max_marginal_error=max_error,
            spectral_norm_before=spectral_norm_before,
            spectral_norm_after=spectral_norm_after,
            spectral_clipped=spectral_clipped,
            original_shape=original_shape,
        )

    def _sinkhorn_knopp(
        self,
        matrix: "Array",
        n: int,
        threshold: float,
    ) -> tuple["Array", int, float]:
        """Run Sinkhorn-Knopp iteration to make matrix doubly stochastic.

        Algorithm:
            1. Divide each row by its sum (row normalization)
            2. Divide each column by its sum (column normalization)
            3. Repeat until convergence or max iterations

        This converges to the unique doubly stochastic matrix in the
        equivalence class of matrices with the same support pattern.

        Returns:
            (projected_matrix, iterations_used, max_marginal_error)
        """
        backend = self._backend

        # Numerical stability floor - derived from dtype
        eps = division_epsilon(backend, matrix)
        M = matrix

        ones = backend.ones((n,))
        max_error = float("inf")

        # Derive max_iterations from problem size
        # Sinkhorn-Knopp has linear convergence rate, iterations scale with n
        # Upper bound: 10 * n (Sinkhorn-Knopp linear convergence; matches ConvergenceMonitor)
        max_iterations = max(10, 10 * n)

        for iteration in range(max_iterations):
            # Row normalization: M_ij / sum_j M_ij
            row_sums = backend.sum(M, axis=1, keepdims=True)
            row_sums = backend.maximum(row_sums, backend.full(row_sums.shape, eps))
            M = M / row_sums
            backend.eval(M)

            # Column normalization: M_ij / sum_i M_ij
            col_sums = backend.sum(M, axis=0, keepdims=True)
            col_sums = backend.maximum(col_sums, backend.full(col_sums.shape, eps))
            M = M / col_sums
            backend.eval(M)

            # Check convergence: max deviation from 1.0 in marginals
            row_sums_final = backend.sum(M, axis=1)
            col_sums_final = backend.sum(M, axis=0)
            backend.eval(row_sums_final, col_sums_final)

            row_error = backend.max(backend.abs(row_sums_final - ones))
            col_error = backend.max(backend.abs(col_sums_final - ones))
            backend.eval(row_error, col_error)

            max_error = max(
                float(backend.to_scalar(row_error)),
                float(backend.to_scalar(col_error)),
            )

            if max_error < threshold:
                logger.debug(
                    f"Sinkhorn-Knopp converged in {iteration + 1} iterations "
                    f"(error={max_error:.2e})"
                )
                return M, iteration + 1, max_error

        logger.debug(
            f"Sinkhorn-Knopp reached max iterations ({max_iterations}), "
            f"final error={max_error:.2e}"
        )
        return M, max_iterations, max_error

    def _compute_spectral_norm(self, matrix: "Array") -> float:
        """Compute spectral norm (largest singular value) of a matrix."""
        backend = self._backend

        # Geodesic SVD - GPU-only, iterates until convergence
        _, S, _ = geodesic_svd(backend, matrix)
        backend.eval(S)
        count = int(S.shape[0])
        if count > 0:
            max_sv_arr = backend.take(S, backend.array([0]), axis=0)
            max_sv_arr = backend.squeeze(max_sv_arr)
            backend.eval(max_sv_arr)
            return float(backend.to_scalar(max_sv_arr))
        return 0.0

    def bound_spectral_norm(
        self,
        matrix: "Array",
        max_norm: float = _MAX_SPECTRAL_NORM,
    ) -> tuple["Array", bool]:
        """Bound spectral norm by scaling singular values.

        If ||M||_2 > max_norm, computes M' = U @ S' @ Vh where
        S' = S * (max_norm / ||M||_2).

        Args:
            matrix: Matrix to bound.
            max_norm: Maximum spectral norm allowed (default 1.0 per mHC).

        Returns:
            (bounded_matrix, was_clipped)
        """
        backend = self._backend

        spectral_norm = self._compute_spectral_norm(matrix)

        if spectral_norm <= max_norm:
            return matrix, False

        # Need to clip: scale the matrix
        scale = max_norm / spectral_norm

        # Geodesic SVD - GPU-only, iterates until convergence
        U, S, Vh = geodesic_svd(backend, matrix)
        backend.eval(U, S, Vh)

        # Scale singular values
        S_clipped = S * scale
        backend.eval(S_clipped)

        # Reconstruct: U @ diag(S) @ Vh
        S_diag = backend.reshape(S_clipped, (-1, 1))
        M_clipped = backend.matmul(U, S_diag * Vh)
        backend.eval(M_clipped)

        logger.debug(
            f"Spectral norm clipped: {spectral_norm:.4f} -> {max_norm:.4f}"
        )
        return M_clipped, True

    def project_weight_delta(
        self,
        delta: "Array",
    ) -> BirkhoffProjectionResult:
        """Project a weight delta for use in merge pipeline.

        For non-square weight matrices, this method embeds the delta
        in a larger square matrix, projects, then extracts the result.

        Args:
            delta: Weight delta [out_dim, in_dim].

        Returns:
            BirkhoffProjectionResult with transformed delta.
        """
        backend = self._backend
        delta = backend.array(delta)
        backend.eval(delta)

        original_shape = tuple(int(d) for d in delta.shape)

        if len(original_shape) != 2:
            raise ValueError(
                f"Weight delta must be 2D, got shape {original_shape}"
            )

        out_dim, in_dim = original_shape

        if out_dim == in_dim:
            # Square matrix - direct projection
            return self.project(delta, ensure_positive=True)

        # Non-square: use Gram matrix approach
        # Project the relational structure via G = delta @ delta^T
        # This preserves the geometric relationships while fitting Birkhoff

        if out_dim < in_dim:
            # Wide matrix: use delta @ delta^T (out_dim x out_dim)
            gram = backend.matmul(delta, backend.transpose(delta))
        else:
            # Tall matrix: use delta^T @ delta (in_dim x in_dim)
            gram = backend.matmul(backend.transpose(delta), delta)

        backend.eval(gram)

        # Normalize gram matrix for faster Sinkhorn convergence
        gram_norm = geodesic_norms(backend.reshape(gram, (1, -1)), backend)
        backend.eval(gram_norm)
        gram_norm_val = float(backend.to_scalar(gram_norm[0]))

        if gram_norm_val > 0:
            gram = gram / gram_norm_val
            backend.eval(gram)

        # Project the gram matrix
        result = self.project(gram, ensure_positive=True)

        # For non-square matrices, we return the projected gram matrix
        # The caller decides how to apply it (e.g., sqrt factorization)
        return BirkhoffProjectionResult(
            projected_matrix=result.projected_matrix,
            converged=result.converged,
            iterations_used=result.iterations_used,
            max_marginal_error=result.max_marginal_error,
            spectral_norm_before=result.spectral_norm_before,
            spectral_norm_after=result.spectral_norm_after,
            spectral_clipped=result.spectral_clipped,
            original_shape=original_shape,
        )


__all__ = [
    "BirkhoffProjectionResult",
    "BirkhoffProjector",
]

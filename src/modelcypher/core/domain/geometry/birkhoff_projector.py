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
Birkhoff Polytope Projector for manifold-constrained weight transformations.

Based on DeepSeek mHC (Manifold-Constrained Hyper-Connections, arXiv:2512.24880):
Projects matrices onto the Birkhoff polytope (doubly stochastic matrices) using
Sinkhorn-Knopp normalization, then bounds spectral norm to ensure compositional
stability across layer chains.

Key insight: Doubly stochastic matrices have spectral norm <= 1 and are closed
under multiplication. This prevents gradient explosion when chaining weight
transformations across many layers.

Mathematical background:
    The Birkhoff polytope B_n is the set of n x n doubly stochastic matrices:
    {M : M_ij >= 0, sum_j M_ij = 1, sum_i M_ij = 1}

    Properties:
    - Convex hull of permutation matrices (vertices)
    - Spectral norm ||M||_2 <= 1 for all M in B_n
    - Closed under multiplication: A, B in B_n => A @ B in B_n

Usage:
    projector = BirkhoffProjector(config)
    result = projector.project(matrix)
    doubly_stochastic = result.projected_matrix  # Row/col sums = 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
    svd_via_eigh,
    tiny_value,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BirkhoffProjectorConfig:
    """Configuration for Birkhoff polytope projection.

    All thresholds are derived from the data's dtype and spectral properties
    unless explicitly overridden. This ensures deterministic, geometry-driven
    behavior without arbitrary "vibes" values.
    """

    # Maximum Sinkhorn-Knopp iterations (DeepSeek mHC uses 20)
    max_iterations: int = 20

    # Convergence threshold for marginal deviation from 1.0.
    # If None, derived from regularization_epsilon (sqrt of machine epsilon).
    convergence_threshold: float | None = None

    # Whether to enforce spectral norm <= 1.0 after Sinkhorn projection.
    # This is the key stability guarantee from mHC.
    enforce_spectral_bound: bool = True

    # Maximum spectral norm allowed (1.0 for strict mHC, slightly higher for flexibility)
    max_spectral_norm: float = 1.0

    # Method for spectral norm computation and bounding
    spectral_method: Literal["svd", "power_iteration"] = "svd"

    # Power iteration count (if using power_iteration method)
    power_iterations: int = 10


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

    This class implements the core algorithm from DeepSeek's mHC paper:
    1. Sinkhorn-Knopp iteration for doubly stochastic projection
    2. Spectral norm bounding to ensure ||M||_2 <= 1.0

    The combination guarantees that chained transformations remain stable:
    if A, B are doubly stochastic with spectral norm <= 1, then A @ B
    has the same properties.
    """

    def __init__(
        self,
        config: BirkhoffProjectorConfig | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or BirkhoffProjectorConfig()
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

        # Derive convergence threshold from dtype if not specified
        threshold = self.config.convergence_threshold
        if threshold is None:
            threshold = regularization_epsilon(backend, matrix)

        # Run Sinkhorn-Knopp iteration
        M, iterations, max_error = self._sinkhorn_knopp(M, n, threshold)

        converged = max_error < threshold

        # Compute spectral norm before bounding
        spectral_norm_before = self._compute_spectral_norm(M)

        # Apply spectral norm bound if configured
        spectral_clipped = False
        if self.config.enforce_spectral_bound:
            M, spectral_clipped = self.bound_spectral_norm(
                M, self.config.max_spectral_norm
            )
            backend.eval(M)

        spectral_norm_after = self._compute_spectral_norm(M)

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
        max_iter = self.config.max_iterations

        # Numerical stability floor - derived from dtype
        eps = division_epsilon(backend, matrix)
        M = matrix

        ones = backend.ones((n,))
        max_error = float("inf")

        for iteration in range(max_iter):
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
                float(backend.to_numpy(row_error)),
                float(backend.to_numpy(col_error)),
            )

            if max_error < threshold:
                logger.debug(
                    f"Sinkhorn-Knopp converged in {iteration + 1} iterations "
                    f"(error={max_error:.2e})"
                )
                return M, iteration + 1, max_error

        logger.debug(
            f"Sinkhorn-Knopp reached max iterations ({max_iter}), "
            f"final error={max_error:.2e}"
        )
        return M, max_iter, max_error

    def _compute_spectral_norm(self, matrix: "Array") -> float:
        """Compute spectral norm (largest singular value) of a matrix."""
        backend = self._backend

        if self.config.spectral_method == "power_iteration":
            return self._spectral_norm_power_iteration(matrix)

        # SVD method (more accurate, slightly slower)
        try:
            _, S, _ = svd_via_eigh(backend, matrix, full_matrices=False)
            backend.eval(S)
            S_np = backend.to_numpy(S)
            return float(S_np[0]) if len(S_np) > 0 else 0.0
        except Exception as e:
            logger.warning(f"SVD failed for spectral norm: {e}")
            return float("inf")

    def _spectral_norm_power_iteration(self, matrix: "Array") -> float:
        """Compute spectral norm via power iteration."""
        backend = self._backend
        n = int(matrix.shape[0])

        # Initialize random vector
        backend.random_seed(42)  # Deterministic for reproducibility
        v = backend.random_normal((n,))
        v = v / backend.norm(v)
        backend.eval(v)

        for _ in range(self.config.power_iterations):
            u = backend.matmul(matrix, v)
            u = u / backend.maximum(backend.norm(u), backend.array(1e-10))
            v = backend.matmul(backend.transpose(matrix), u)
            v = v / backend.maximum(backend.norm(v), backend.array(1e-10))
            backend.eval(u, v)

        # Spectral norm = ||M @ v||
        Mv = backend.matmul(matrix, v)
        backend.eval(Mv)
        norm = backend.norm(Mv)
        backend.eval(norm)
        return float(backend.to_numpy(norm))

    def bound_spectral_norm(
        self,
        matrix: "Array",
        max_norm: float = 1.0,
    ) -> tuple["Array", bool]:
        """Bound spectral norm by scaling singular values.

        If ||M||_2 > max_norm, computes M' = U @ S' @ Vh where
        S' = S * (max_norm / ||M||_2).

        Args:
            matrix: Matrix to bound.
            max_norm: Maximum spectral norm allowed (default 1.0).

        Returns:
            (bounded_matrix, was_clipped)
        """
        backend = self._backend

        spectral_norm = self._compute_spectral_norm(matrix)

        if spectral_norm <= max_norm:
            return matrix, False

        # Need to clip: scale the matrix
        # For doubly stochastic matrices, simple scaling works because
        # we can re-normalize after
        scale = max_norm / spectral_norm

        if self.config.spectral_method == "svd":
            # SVD-based clipping (more precise)
            try:
                U, S, Vh = svd_via_eigh(backend, matrix, full_matrices=False)
                backend.eval(U, S, Vh)

                # Scale singular values
                S_clipped = S * scale
                backend.eval(S_clipped)

                # Reconstruct: U @ diag(S) @ Vh
                # For square matrices: M = U @ (S * Vh)
                S_diag = backend.reshape(S_clipped, (-1, 1))
                M_clipped = backend.matmul(U, S_diag * Vh)
                backend.eval(M_clipped)

                return M_clipped, True

            except Exception as e:
                logger.warning(f"SVD clipping failed: {e}, using simple scaling")

        # Simple scaling fallback
        M_scaled = matrix * scale
        backend.eval(M_scaled)

        logger.debug(
            f"Spectral norm clipped: {spectral_norm:.4f} -> {max_norm:.4f}"
        )
        return M_scaled, True

    def project_weight_delta(
        self,
        delta: "Array",
        reference_scale: float | None = None,
    ) -> BirkhoffProjectionResult:
        """Project a weight delta for use in merge pipeline.

        For non-square weight matrices, this method embeds the delta
        in a larger square matrix, projects, then extracts the result.

        Args:
            delta: Weight delta [out_dim, in_dim].
            reference_scale: Optional scale for normalization.

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

        # Normalize gram matrix for better Sinkhorn convergence
        gram_norm = backend.norm(gram)
        backend.eval(gram_norm)
        gram_norm_val = float(backend.to_numpy(gram_norm))

        if gram_norm_val > 0:
            gram = gram / gram_norm_val
            backend.eval(gram)

        # Project the gram matrix
        result = self.project(gram, ensure_positive=True)

        # For non-square matrices, we return the projected gram matrix
        # The caller decides how to apply it (e.g., sqrt factorization)
        # Override the shape to reflect original for tracking
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
    "BirkhoffProjectorConfig",
    "BirkhoffProjectionResult",
    "BirkhoffProjector",
]

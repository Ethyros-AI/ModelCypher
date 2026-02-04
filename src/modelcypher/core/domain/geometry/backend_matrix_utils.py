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

"""Backend-aware matrix utilities for high-dimensional geometry.

This module provides hardware-accelerated matrix operations using the
Backend protocol. Operations run on the configured accelerator,
or CPU fallback depending on the backend passed.

Relationship with GeometryEngine:
    BackendMatrixUtils (this module) provides canonical matrix operations
    for geometry pipelines. GeometryEngine composes these utilities for
    use-case level workflows.

Example:
    from modelcypher.core.domain._backend import get_default_backend
    backend = get_default_backend()
    utils = BackendMatrixUtils(backend)
    gram = utils.compute_gram_matrix(activations)
    result = utils.procrustes_rotation(source, target)

Canonical operations:
- Gram matrix computation (with caching)
- Pairwise geodesic distances
- Procrustes rotation (SVD-based orthogonal alignment)
- Effective rank estimation
- Cosine similarity matrix

Note: Matrix centering is in cka.py (_center_gram_matrix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    power_iteration_eigh,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix
from modelcypher.core.domain.geometry.types import PairwiseProcrustesResult

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

# TypeVar for array types from any backend
Array = TypeVar("Array")

# Session-scoped cache for Gram matrices
_cache = ComputationCache.shared()

class BackendMatrixUtils:
    """Backend-aware matrix utilities for geometry operations.

    This class uses the Backend protocol for all tensor operations,
    enabling hardware acceleration instead of CPU-only NumPy.

    Example:
        from modelcypher.core.domain._backend import get_default_backend
        backend = get_default_backend()
        utils = BackendMatrixUtils(backend)
        gram = utils.compute_gram_matrix(activations)
        result = utils.procrustes_rotation(source, target)
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a specific backend.

        Args:
            backend: Backend instance implementing the Backend protocol.
        """
        self.backend = backend

    def compute_gram_matrix(self, X: Array, kernel: str = "linear") -> Array:
        """Compute the Gram matrix (kernel matrix) of X.

        Uses session-scoped caching to avoid redundant computation.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            kernel: Kernel type ('linear' or 'rbf')

        Returns:
            Gram matrix of shape (n_samples, n_samples)
        """
        if kernel == "linear":
            # Use cached Gram matrix
            return _cache.get_or_compute_gram(X, self.backend, kernel_type="linear")
        elif kernel == "rbf":
            # Gaussian RBF kernel with median bandwidth (data-derived)
            sq_dists = self.pairwise_squared_distances(X)

            # Compute median of non-zero distances for bandwidth
            # Flatten, select median without full sort
            flat = self.backend.reshape(sq_dists, (-1,))

            # Find median of positive values (exclude near-zeros by dtype epsilon)
            n = int(flat.shape[0]) if hasattr(flat, "shape") else len(flat)
            div_eps = division_epsilon(self.backend, sq_dists)
            zero_mask = flat <= div_eps
            zero_count_arr = self.backend.sum(zero_mask)
            self.backend.eval(zero_count_arr)
            zero_count = int(self.backend.to_scalar(zero_count_arr))
            non_zero_count = max(n - zero_count, 1)
            median_idx = min(zero_count + (non_zero_count // 2), n - 1)
            partitioned = self.backend.argpartition(flat, median_idx)
            prefix = self.backend.take(
                partitioned, self.backend.arange(median_idx + 1), axis=0
            )
            median_val = self.backend.max(self.backend.take(flat, prefix, axis=0))
            median_val = self.backend.squeeze(median_val)
            self.backend.eval(median_val)
            median_dist = float(self.backend.to_scalar(median_val))
            gamma = 1.0 / (2.0 * (median_dist + div_eps))

            # exp(-gamma * sq_dists)
            scaled = self._scalar_multiply(sq_dists, -gamma)
            return self.backend.exp(scaled)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def _scalar_multiply(self, arr: Array, scalar: float) -> Array:
        """Multiply array by scalar using backend operations."""
        # Element-wise multiply via where trick or direct
        # Most backends support arr * scalar directly, but we use backend ops
        # Create ones and scale
        return (
            self.backend.full(arr.shape, scalar) * arr
            if hasattr(arr, "__mul__")
            else self.backend.matmul(
                self.backend.diag(self.backend.full((arr.shape[0],), scalar)), arr
            )
        )

    def pairwise_squared_distances(self, X: Array) -> Array:
        """Compute pairwise squared geodesic distances.

        Uses k-NN graph shortest paths to estimate manifold distances.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        geo_dist = geodesic_distance_matrix(X, k_neighbors=None, backend=self.backend)
        self.backend.eval(geo_dist)
        return geo_dist * geo_dist  # Squared

    def pairwise_distances(self, X: Array) -> Array:
        """Compute pairwise geodesic distances.

        Uses k-NN graph shortest paths for manifold distances.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        return geodesic_distance_matrix(X, k_neighbors=None, backend=self.backend)

    def procrustes_rotation(
        self,
        source: Array,
        target: Array,
        allow_scaling: bool = False,
    ) -> PairwiseProcrustesResult[Array]:
        """Compute optimal orthogonal rotation to align source to target.

        Finds the orthogonal matrix R that minimizes ||target - source @ R||_F.
        This uses the SVD-based Procrustes solution.

        The algorithm:
        1. Compute M = source.T @ target
        2. SVD: M = U @ S @ V^T
        3. R = U @ V^T (optimal rotation)

        Args:
            source: Source matrix of shape (n, d)
            target: Target matrix of shape (n, d)
            allow_scaling: If True, also compute optimal scale factor

        Returns:
            PairwiseProcrustesResult with rotation, scale, and residual
        """
        # Compute cross-covariance matrix: M = source.T @ target
        source_T = self.backend.transpose(source)
        M = self.backend.matmul(source_T, target)

        # Geodesic SVD: M = U @ S @ Vt (GPU-only, iterates until convergence)
        U, S, Vt = geodesic_svd(self.backend, M)

        # Optimal orthogonal rotation: R = U @ Vt
        R = self.backend.matmul(U, Vt)
        # Re-project to the closest orthogonal matrix to suppress numeric drift.
        U_r, _, Vt_r = geodesic_svd(self.backend, R)
        R = self.backend.matmul(U_r, Vt_r)

        det_arr = self.backend.det(R)
        self.backend.eval(det_arr)
        det_val = float(self.backend.to_scalar(det_arr))

        if det_val < 0:
            U_fixed = self.backend.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            R = self.backend.matmul(U_fixed, Vt)

        # Compute scale if requested
        if allow_scaling:
            # Optimal scale: sum(S) / trace(source.T @ source)
            S_sum_arr = self.backend.sum(S)
            source_cov = self.backend.matmul(source_T, source)
            # trace = sum of diagonal
            diag_vals = self.backend.diag(source_cov)
            source_variance_arr = self.backend.sum(diag_vals)
            self.backend.eval(S_sum_arr, source_variance_arr)
            S_sum = float(self.backend.to_scalar(S_sum_arr))
            source_variance = float(self.backend.to_scalar(source_variance_arr))

            if source_variance > 0:
                scale = S_sum / source_variance
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Compute residual: ||target - scale * (source @ R)||^2
        aligned = self.backend.matmul(source, R)
        if scale != 1.0:
            scale_arr = self.backend.full(aligned.shape, scale)
            aligned = aligned * scale_arr

        diff = target - aligned
        diff_sq = diff * diff
        residual_arr = self.backend.sum(diff_sq)
        self.backend.eval(residual_arr)
        residual = float(self.backend.to_scalar(residual_arr))

        # Translation (zeros for rotation-only)
        d = source.shape[1]
        translation = self.backend.zeros((d,))

        return PairwiseProcrustesResult(
            rotation=R,
            scale=scale,
            translation=translation,
            residual=residual,
        )

    def procrustes_align(
        self,
        source: Array,
        target: Array,
        center: bool = True,
        allow_scaling: bool = False,
    ) -> tuple[Array, PairwiseProcrustesResult[Array]]:
        """Align source to target using Procrustes analysis.

        Full Procrustes alignment with optional centering and scaling.

        Args:
            source: Source matrix of shape (n, d)
            target: Target matrix of shape (n, d)
            center: If True, center both matrices before alignment
            allow_scaling: If True, compute optimal scale factor

        Returns:
            Tuple of (aligned_source, PairwiseProcrustesResult)
        """
        if center:
            source_mean = self.backend.mean(source, axis=0, keepdims=True)
            target_mean = self.backend.mean(target, axis=0, keepdims=True)
            source_centered = source - source_mean
            target_centered = target - target_mean
        else:
            source_centered = source
            target_centered = target
            source_mean = self.backend.zeros((1, source.shape[1]))
            target_mean = self.backend.zeros((1, target.shape[1]))

        result = self.procrustes_rotation(source_centered, target_centered, allow_scaling)

        if center:
            # Update translation
            # translation = target_mean - scale * (source_mean @ R)
            source_mean_rot = self.backend.matmul(source_mean, result.rotation)
            if result.scale != 1.0:
                scale_arr = self.backend.full(source_mean_rot.shape, result.scale)
                source_mean_rot = source_mean_rot * scale_arr
            result.translation = self.backend.squeeze(target_mean - source_mean_rot)

        # Aligned = scale * (source @ R) + translation
        aligned = self.backend.matmul(source, result.rotation)
        if result.scale != 1.0:
            scale_arr = self.backend.full(aligned.shape, result.scale)
            aligned = aligned * scale_arr
        aligned = aligned + result.translation

        return aligned, result

    def spectral_gap_rank(self, eigenvalues: Array) -> int:
        """Compute effective rank from spectral gap detection.

        Finds the natural boundary between signal and noise by locating
        the maximum relative drop in eigenvalues. This is geometry-derived,
        not an arbitrary threshold.

        Returns:
            The number of eigenvalues before the spectral gap.
        """
        b = self.backend
        eig_flat = b.reshape(eigenvalues, (-1,))
        n = int(eig_flat.shape[0])

        if n < 2:
            # Count positive values
            mask = eig_flat > 0
            count_dtype = precision_dtype(b, reference=eig_flat)
            count_arr = b.sum(b.astype(mask, count_dtype))
            b.eval(count_arr)
            count = int(b.to_scalar(count_arr))
            return count

        # Machine epsilon for numerical stability
        eps = division_epsilon(b, eig_flat)

        # Replace non-positive with -inf so they sort to the end
        neg_inf = b.full(eig_flat.shape, float("-inf"))
        eig_masked = b.where(eig_flat > 0, eig_flat, neg_inf)

        # Sort ascending, then reverse to get descending
        sorted_asc = b.sort(eig_masked)
        reverse_idx = b.arange(n - 1, -1, -1)
        b.eval(reverse_idx)
        sorted_desc = b.take(sorted_asc, reverse_idx, axis=0)
        b.eval(sorted_desc)

        # Count positive eigenvalues (those not -inf after sort)
        is_positive = sorted_desc > float("-inf")
        count_dtype = precision_dtype(b, reference=sorted_desc)
        positive_count_arr = b.sum(b.astype(is_positive, count_dtype))
        b.eval(positive_count_arr)
        positive_count = int(b.to_scalar(positive_count_arr))

        if positive_count < 2:
            return positive_count

        # Compute relative drops: (λ_i - λ_{i+1}) / λ_i for i in [0, positive_count-1)
        # Use slicing via take
        idx_i = b.arange(0, positive_count - 1)
        idx_ip1 = b.arange(1, positive_count)
        b.eval(idx_i, idx_ip1)

        lambda_i = b.take(sorted_desc, idx_i, axis=0)
        lambda_ip1 = b.take(sorted_desc, idx_ip1, axis=0)
        b.eval(lambda_i, lambda_ip1)

        # Floor lambda_i to avoid division by zero
        eps_arr = b.full(lambda_i.shape, eps)
        lambda_i_safe = b.maximum(lambda_i, eps_arr)

        # Relative drop = (λ_i - λ_{i+1}) / λ_i
        relative_drops = (lambda_i - lambda_ip1) / lambda_i_safe
        b.eval(relative_drops)

        # Find max gap using argmax
        max_gap_idx_arr = b.argmax(relative_drops)
        max_gap_val_arr = b.max(relative_drops)
        b.eval(max_gap_idx_arr, max_gap_val_arr)
        max_gap_idx = int(b.to_scalar(max_gap_idx_arr))
        max_gap_val = float(b.to_scalar(max_gap_val_arr))

        if max_gap_val <= 0.0:
            return positive_count

        # gap_index is the number of components before the gap (1-indexed)
        return max_gap_idx + 1

    def effective_rank(
        self,
        eigenvalues: Array,
        variance_threshold: float | None = None,
    ) -> int:
        """Compute effective rank from eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues (squared singular values).
            variance_threshold: If provided, use variance-based rank (the number
                of eigenvalues needed to capture this fraction of total variance).
                If None (default), use spectral gap detection which finds the natural
                signal/noise boundary from the data.

        Returns:
            Effective rank (number of significant dimensions).
        """
        # Default: use spectral gap detection (geometry-derived)
        if variance_threshold is None:
            return self.spectral_gap_rank(eigenvalues)

        # If variance threshold provided, use cumulative variance method
        b = self.backend
        eig_flat = b.reshape(eigenvalues, (-1,))
        n = int(eig_flat.shape[0])

        if n == 0:
            return 0

        # Replace non-positive with 0 for sorting
        zeros = b.zeros_like(eig_flat)
        eig_positive = b.where(eig_flat > 0, eig_flat, zeros)

        # Sort descending
        sorted_asc = b.sort(eig_positive)
        reverse_idx = b.arange(n - 1, -1, -1)
        b.eval(reverse_idx)
        eig_sorted = b.take(sorted_asc, reverse_idx, axis=0)
        b.eval(eig_sorted)

        # Compute total
        total_arr = b.sum(eig_sorted)
        b.eval(total_arr)
        total = float(b.to_scalar(total_arr))

        if total <= 0:
            return 0

        # Compute cumulative sum normalized by total
        cumsum_arr = b.cumsum(eig_sorted) / total
        b.eval(cumsum_arr)

        # Find first index where cumsum >= threshold using backend
        threshold_arr = b.full(cumsum_arr.shape, variance_threshold)
        exceeds_threshold = cumsum_arr >= threshold_arr
        b.eval(exceeds_threshold)

        # Convert to float mask and find first True (argmax on boolean gives first 1)
        exceeds_float = b.astype(exceeds_threshold, precision_dtype(b, reference=cumsum_arr))
        # If no element exceeds threshold, argmax returns 0, so we check max
        max_exceeds_arr = b.max(exceeds_float)
        b.eval(max_exceeds_arr)
        max_exceeds = float(b.to_scalar(max_exceeds_arr))

        if max_exceeds < 1:  # All values are 0.0 (False), none exceed threshold
            # No element exceeds threshold, return full count of positive eigenvalues
            positive_mask = eig_sorted > 0
            positive_count_arr = b.sum(
                b.astype(positive_mask, precision_dtype(b, reference=eig_sorted))
            )
            b.eval(positive_count_arr)
            return int(b.to_scalar(positive_count_arr))

        # argmax returns first index where value is max (which is 1.0 for True)
        first_exceed_idx_arr = b.argmax(exceeds_float)
        b.eval(first_exceed_idx_arr)
        first_exceed_idx = int(b.to_scalar(first_exceed_idx_arr))
        return first_exceed_idx + 1

    def cosine_similarity_matrix(self, X: Array) -> Array:
        """Compute pairwise geodesic cosine similarity matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        return geodesic_cosine_matrix(X, self.backend)

    def eigendecomposition(self, K: Array) -> tuple[Array, Array]:
        """Compute eigendecomposition of symmetric matrix.

        Uses GPU-only power iteration, iterates until convergence.

        Args:
            K: Symmetric matrix of shape (n, n)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        n = int(K.shape[0])
        return power_iteration_eigh(self.backend, K, k=n)

    def trace(self, A: Array) -> float:
        """Compute trace of a matrix.

        Args:
            A: Square matrix

        Returns:
            Trace (sum of diagonal elements)
        """
        diag_vals = self.backend.diag(A)
        trace_arr = self.backend.sum(diag_vals)
        self.backend.eval(trace_arr)
        return float(self.backend.to_scalar(trace_arr))


# =============================================================================
# Pure Python list-based matrix utilities (no Backend required)
# =============================================================================
# These functions operate on Python lists for cases where Backend acceleration
# is not needed or not available. Useful for small matrices and pure computation.


def reshape_flat_to_matrix(flat: list[float], rows: int, cols: int) -> list[list[float]]:
    """Reshape a flat list into a 2D matrix (list of lists).

    Args:
        flat: 1D list of floats with length = rows * cols
        rows: Number of rows in output matrix
        cols: Number of columns in output matrix

    Returns:
        2D list with shape [rows, cols]

    Example:
        >>> flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> reshape_flat_to_matrix(flat, 2, 3)
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    """
    result: list[list[float]] = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(flat[i * cols + j])
        result.append(row)
    return result


def transpose_flat_matrix(matrix: list[float], m: int, n: int) -> list[float]:
    """Transpose a flat matrix from m×n to n×m.

    Args:
        matrix: Flat list representing m×n matrix (row-major order)
        m: Number of rows in input matrix
        n: Number of columns in input matrix

    Returns:
        Flat list representing n×m transposed matrix (row-major order)

    Example:
        >>> # 2×3 matrix: [[1,2,3], [4,5,6]]
        >>> flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> transpose_flat_matrix(flat, 2, 3)
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]  # 3×2: [[1,4], [2,5], [3,6]]
    """
    result = [0.0 for _ in range(m * n)]
    for i in range(m):
        for j in range(n):
            result[j * m + i] = matrix[i * n + j]
    return result


def compute_frobenius_norm_squared(matrix: list[float]) -> float:
    """Compute the squared Frobenius norm of a flat matrix.

    Args:
        matrix: Flat list representing a matrix

    Returns:
        Sum of squares of all elements
    """
    return sum(value * value for value in matrix)

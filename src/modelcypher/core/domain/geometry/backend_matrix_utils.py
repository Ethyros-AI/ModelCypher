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
Backend protocol. Operations run on MLX (Apple Silicon), JAX (TPU/GPU),
or NumPy (CPU fallback) depending on the backend passed.

Use this instead of matrix_utils.py when you need accelerated computation
on the actual backend rather than NumPy CPU fallback.

Canonical operations:
- Gram matrix computation
- Matrix centering
- Pairwise squared distances
- Procrustes rotation (SVD-based orthogonal alignment)
- Effective rank estimation
- Cosine similarity matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from modelcypher.core.domain.cache import ComputationCache

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

# TypeVar for array types from any backend
Array = TypeVar("Array")

# Session-scoped cache for Gram matrices
_cache = ComputationCache.shared()


@dataclass
class ProcrustesResult(Generic[Array]):
    """Result of Procrustes alignment.

    Generic over Array type to support MLX, JAX, or NumPy arrays.
    """

    rotation: Array  # Orthogonal rotation matrix
    scale: float  # Optimal scale factor
    translation: Array  # Translation vector (if computed)
    residual: float  # Procrustes distance (sum of squared errors)


class BackendMatrixUtils:
    """Backend-aware matrix utilities for geometry operations.

    This class uses the Backend protocol for all tensor operations,
    enabling hardware acceleration on MLX/JAX instead of CPU-only NumPy.

    Example:
        backend = MLXBackend()
        utils = BackendMatrixUtils(backend)
        gram = utils.compute_gram_matrix(activations)
        result = utils.procrustes_rotation(source, target)
    """

    def __init__(self, backend: "Backend"):
        """Initialize with a specific backend.

        Args:
            backend: Backend instance (MLXBackend, JAXBackend)
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
            # Gaussian RBF kernel with median heuristic
            sq_dists = self.pairwise_squared_distances(X)

            # Compute median of non-zero distances for bandwidth
            # Flatten, sort, find median
            flat = self.backend.reshape(sq_dists, (-1,))
            sorted_dists = self.backend.sort(flat)

            # Find median of positive values (approximate)
            n = flat.shape[0] if hasattr(flat, "shape") else len(flat)
            mid_idx = n // 2
            median_dist = float(self.backend.to_numpy(sorted_dists)[mid_idx])

            gamma = 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0

            # exp(-gamma * sq_dists)
            neg_gamma = self.backend.full(sq_dists.shape, -gamma)
            scaled = (
                self.backend.matmul(sq_dists, neg_gamma)
                if hasattr(sq_dists, "shape")
                else sq_dists * (-gamma)
            )
            # Actually just multiply element-wise
            scaled = self._scalar_multiply(sq_dists, -gamma)
            return self.backend.exp(scaled)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def _scalar_multiply(self, arr: Array, scalar: float) -> Array:
        """Multiply array by scalar using backend operations."""
        self.backend.full(arr.shape, scalar)
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

    def center_matrix(self, K: Array, weights: Array | None = None) -> Array:
        """Center a kernel matrix (double centering).

        For unweighted centering, computes: H @ K @ H
        where H = I - (1/n) * 11^T is the centering matrix.

        Args:
            K: Kernel/Gram matrix of shape (n, n)
            weights: Optional sample weights of shape (n,)

        Returns:
            Centered matrix of shape (n, n)
        """
        n = K.shape[0]

        if weights is None:
            # Standard unweighted centering
            # row_mean = K.mean(axis=1, keepdims=True)
            row_mean = self.backend.mean(K, axis=1, keepdims=True)
            col_mean = self.backend.mean(K, axis=0, keepdims=True)
            grand_mean = self.backend.mean(K)

            # K - row_mean - col_mean + grand_mean
            result = K
            # Subtract row_mean (broadcast)
            result = result - row_mean
            # Subtract col_mean (broadcast)
            result = result - col_mean
            # Add grand_mean
            grand_mean_arr = self.backend.full(K.shape, float(self.backend.to_numpy(grand_mean)))
            result = result + grand_mean_arr
            return result
        else:
            # Weighted centering
            w_sum = self.backend.sum(weights)
            weights_norm = weights / w_sum

            # Weighted row mean
            weighted_K = K * self.backend.reshape(weights_norm, (1, n))
            row_mean = self.backend.sum(weighted_K, axis=1, keepdims=True)

            # Weighted col mean
            weighted_K_col = K * self.backend.reshape(weights_norm, (n, 1))
            col_mean = self.backend.sum(weighted_K_col, axis=0, keepdims=True)

            # Weighted grand mean
            outer_weights = self.backend.matmul(
                self.backend.reshape(weights_norm, (n, 1)),
                self.backend.reshape(weights_norm, (1, n)),
            )
            grand_mean = self.backend.sum(K * outer_weights)

            result = K - row_mean - col_mean
            grand_mean_arr = self.backend.full(K.shape, float(self.backend.to_numpy(grand_mean)))
            result = result + grand_mean_arr
            return result

    def pairwise_squared_distances(self, X: Array) -> Array:
        """Compute pairwise squared geodesic distances.

        Uses k-NN graph shortest paths to estimate true manifold distances.
        This is the correct metric for curved high-dimensional spaces.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        n = X.shape[0] if hasattr(X, "shape") else len(X)
        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(X, k_neighbors=k_neighbors, backend=self.backend)
        self.backend.eval(geo_dist)
        return geo_dist * geo_dist  # Squared

    def pairwise_distances(self, X: Array) -> Array:
        """Compute pairwise geodesic distances.

        Uses k-NN graph shortest paths for true manifold distances.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        from modelcypher.core.domain.geometry.riemannian_utils import (
            geodesic_distance_matrix,
        )

        n = X.shape[0] if hasattr(X, "shape") else len(X)
        k_neighbors = min(max(3, n // 3), n - 1)
        return geodesic_distance_matrix(X, k_neighbors=k_neighbors, backend=self.backend)

    def procrustes_rotation(
        self,
        source: Array,
        target: Array,
        allow_scaling: bool = False,
    ) -> ProcrustesResult[Array]:
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
            ProcrustesResult with rotation, scale, and residual
        """
        # Compute cross-covariance matrix: M = source.T @ target
        source_T = self.backend.transpose(source)
        M = self.backend.matmul(source_T, target)

        # SVD: M = U @ S @ Vt
        U, S, Vt = self.backend.svd(M, compute_uv=True)

        # Optimal orthogonal rotation: R = U @ Vt
        R = self.backend.matmul(U, Vt)

        det_arr = self.backend.det(R)
        self.backend.eval(det_arr)
        det_val = float(self.backend.to_numpy(det_arr).item())

        if det_val < 0:
            U_fixed = self.backend.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            R = self.backend.matmul(U_fixed, Vt)

        # Compute scale if requested
        if allow_scaling:
            # Optimal scale: sum(S) / trace(source.T @ source)
            S_sum = float(self.backend.to_numpy(self.backend.sum(S)))
            source_cov = self.backend.matmul(source_T, source)
            # trace = sum of diagonal
            diag_vals = self.backend.diag(source_cov)
            source_variance = float(self.backend.to_numpy(self.backend.sum(diag_vals)))

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
        residual = float(self.backend.to_numpy(self.backend.sum(diff_sq)))

        # Translation (zeros for rotation-only)
        d = source.shape[1]
        translation = self.backend.zeros((d,))

        return ProcrustesResult(
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
    ) -> tuple[Array, ProcrustesResult[Array]]:
        """Align source to target using Procrustes analysis.

        Full Procrustes alignment with optional centering and scaling.

        Args:
            source: Source matrix of shape (n, d)
            target: Target matrix of shape (n, d)
            center: If True, center both matrices before alignment
            allow_scaling: If True, compute optimal scale factor

        Returns:
            Tuple of (aligned_source, ProcrustesResult)
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

    def effective_rank(
        self,
        eigenvalues: Array,
        variance_threshold: float = 0.95,
    ) -> int:
        """Compute effective rank from eigenvalues.

        Finds the number of eigenvalues needed to capture the specified
        fraction of total variance.
        """
        b = self.backend
        eig_flat = b.reshape(eigenvalues, (-1,))
        mask = eig_flat > 0
        b.eval(mask)

        eig_np = b.to_numpy(eig_flat)
        mask_np = b.to_numpy(mask)
        eig_positive = eig_np[mask_np]

        if len(eig_positive) == 0:
            return 0

        eig_sorted = b.array([float(x) for x in sorted(eig_positive, reverse=True)])
        total_arr = b.sum(eig_sorted)
        b.eval(total_arr)
        total = float(b.to_numpy(total_arr).item())

        if total <= 0:
            return 0

        cumsum_arr = b.cumsum(eig_sorted)
        b.eval(cumsum_arr)
        cumsum = b.to_numpy(cumsum_arr) / total

        for i, val in enumerate(cumsum):
            if val >= variance_threshold:
                return i + 1
        return len(cumsum)

    def entropy_effective_rank(self, eigenvalues: Array) -> float:
        """Compute entropy-based effective rank.

        Uses the exponential of Shannon entropy of normalized eigenvalues:
        erank = exp(-sum(p * log(p)))
        """
        import math

        b = self.backend
        eig_flat = b.reshape(eigenvalues, (-1,))
        mask = eig_flat > 0
        b.eval(mask)

        eig_np = b.to_numpy(eig_flat)
        mask_np = b.to_numpy(mask)
        eig_positive = eig_np[mask_np]

        if len(eig_positive) == 0:
            return 0.0

        eig_arr = b.array([float(x) for x in eig_positive])
        total_arr = b.sum(eig_arr)
        b.eval(total_arr)
        total = float(b.to_numpy(total_arr).item())

        if total <= 0:
            return 0.0

        p = eig_arr / total
        eps = b.full(p.shape, 1e-12)
        log_p = b.log(p + eps)
        entropy_arr = -b.sum(p * log_p)
        b.eval(entropy_arr)
        entropy = float(b.to_numpy(entropy_arr).item())

        return math.exp(entropy)

    def cosine_similarity_matrix(self, X: Array) -> Array:
        """Compute pairwise cosine similarity matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        # Normalize rows: X / ||X||
        norms = self.backend.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        eps = self.backend.full(norms.shape, 1e-12)
        norms = self.backend.maximum(norms, eps)

        X_normalized = X / norms

        # Cosine similarity = dot product of normalized vectors
        X_normalized_T = self.backend.transpose(X_normalized)
        return self.backend.matmul(X_normalized, X_normalized_T)

    def eigendecomposition(self, K: Array) -> tuple[Array, Array]:
        """Compute eigendecomposition of symmetric matrix.

        Args:
            K: Symmetric matrix of shape (n, n)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        return self.backend.eigh(K)

    def trace(self, A: Array) -> float:
        """Compute trace of a matrix.

        Args:
            A: Square matrix

        Returns:
            Trace (sum of diagonal elements)
        """
        diag_vals = self.backend.diag(A)
        return float(self.backend.to_numpy(self.backend.sum(diag_vals)))

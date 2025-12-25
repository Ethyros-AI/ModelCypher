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

"""Matrix utilities for high-dimensional geometry operations.

This module provides canonical implementations of common matrix operations
used across the geometry domain. All implementations here are the single
source of truth - do not duplicate these operations elsewhere.

Canonical operations:
- Gram matrix computation
- Matrix centering
- Pairwise squared distances
- Procrustes rotation (SVD-based orthogonal alignment)
- Effective rank estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

# Session-scoped cache for Gram matrices
_cache = ComputationCache.shared()


@dataclass
class ProcrustesResult:
    """Result of Procrustes alignment."""

    rotation: "Array"  # Orthogonal rotation matrix
    scale: float  # Optimal scale factor
    translation: "Array"  # Translation vector (if computed)
    residual: float  # Procrustes distance (sum of squared errors)


class MatrixUtils:
    """Matrix utilities for geometry operations.

    This is the canonical implementation for matrix operations.
    DO NOT reimplement these operations elsewhere.
    """

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute_gram_matrix(self, X: "Array", kernel: str = "linear") -> "Array":
        """Compute the Gram matrix (kernel matrix) of X.

        Uses session-scoped caching to avoid redundant computation.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            kernel: Kernel type ('linear' or 'rbf')

        Returns:
            Gram matrix of shape (n_samples, n_samples)
        """
        b = self._backend
        if kernel == "linear":
            # Use cached Gram matrix
            return _cache.get_or_compute_gram(X, b, kernel_type="linear")
        elif kernel == "rbf":
            # RBF kernel - check cache first
            cache_key = _cache.make_gram_key(X, b, kernel_type="rbf")
            cached = _cache.get_gram(cache_key)
            if cached is not None:
                return cached

            # Gaussian RBF kernel with default bandwidth
            sq_dists = self.pairwise_squared_distances(X)
            # Use median heuristic for bandwidth
            b.eval(sq_dists)
            sq_dists_np = b.to_numpy(sq_dists).flatten()
            positive_dists = sq_dists_np[sq_dists_np > 0]
            if len(positive_dists) > 0:
                sorted_dists = sorted(positive_dists)
                median_dist = sorted_dists[len(sorted_dists) // 2]
            else:
                median_dist = 1.0
            gamma = 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0
            gram = b.exp(-gamma * sq_dists)

            _cache.set_gram(cache_key, gram)
            return gram
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    def center_matrix(self, K: "Array", weights: "Array | None" = None) -> "Array":
        """Center a kernel matrix (double centering).

        For unweighted centering, this computes: H @ K @ H
        where H = I - (1/n) * 11^T is the centering matrix.

        Args:
            K: Kernel/Gram matrix of shape (n, n)
            weights: Optional sample weights of shape (n,)

        Returns:
            Centered matrix of shape (n, n)
        """
        b = self._backend

        if weights is None:
            # Standard unweighted centering
            row_mean = b.mean(K, axis=1, keepdims=True)
            col_mean = b.mean(K, axis=0, keepdims=True)
            grand_mean = b.mean(K)
            return K - row_mean - col_mean + grand_mean
        else:
            # Weighted centering
            weights = weights / b.sum(weights)
            row_mean = b.sum(K * weights, axis=1, keepdims=True)
            col_mean = b.sum(K * b.reshape(weights, (-1, 1)), axis=0, keepdims=True)
            grand_mean = b.sum(K * b.outer(weights, weights))
            return K - row_mean - col_mean + grand_mean

    def pairwise_squared_distances(self, X: "Array") -> "Array":
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

        b = self._backend
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        k_neighbors = min(max(3, n // 3), n - 1)
        geo_dist = geodesic_distance_matrix(X, k_neighbors=k_neighbors, backend=b)
        b.eval(geo_dist)
        return geo_dist * geo_dist  # Squared

    def pairwise_distances(self, X: "Array") -> "Array":
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

        b = self._backend
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        k_neighbors = min(max(3, n // 3), n - 1)
        return geodesic_distance_matrix(X, k_neighbors=k_neighbors, backend=b)

    def procrustes_rotation(
        self,
        source: "Array",
        target: "Array",
        allow_scaling: bool = False,
    ) -> ProcrustesResult:
        """Compute optimal orthogonal rotation to align source to target.

        Finds the orthogonal matrix R that minimizes ||target - source @ R||_F.
        This is the canonical SVD-based Procrustes solution.

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
        b = self._backend

        # Compute cross-covariance matrix
        M = b.matmul(b.transpose(source), target)

        # SVD
        U, S, Vt = b.svd(M)

        # Optimal orthogonal rotation
        R = b.matmul(U, Vt)

        # Handle reflection if determinant is -1
        det_val = b.det(R)
        b.eval(det_val)
        if float(b.to_numpy(det_val).item()) < 0:
            # Flip sign of last column of U
            U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            R = b.matmul(U_fixed, Vt)

        # Compute scale if requested
        if allow_scaling:
            # Optimal scale: trace(S) / trace(source.T @ source)
            source_variance = b.sum(b.diag(b.matmul(b.transpose(source), source)))
            b.eval(source_variance, S)
            source_var_val = float(b.to_numpy(source_variance))
            if source_var_val > 0:
                scale = float(b.to_numpy(b.sum(S))) / source_var_val
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Compute residual
        aligned = scale * b.matmul(source, R)
        residual_arr = b.sum((target - aligned) ** 2)
        b.eval(residual_arr)
        residual = float(b.to_numpy(residual_arr))

        return ProcrustesResult(
            rotation=R,
            scale=scale,
            translation=b.zeros((source.shape[1],)),
            residual=residual,
        )

    def procrustes_align(
        self,
        source: "Array",
        target: "Array",
        center: bool = True,
        allow_scaling: bool = False,
    ) -> tuple["Array", ProcrustesResult]:
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
        b = self._backend

        if center:
            source_mean = b.mean(source, axis=0)
            target_mean = b.mean(target, axis=0)
            source_centered = source - source_mean
            target_centered = target - target_mean
        else:
            source_mean = b.zeros((source.shape[1],))
            target_mean = b.zeros((target.shape[1],))
            source_centered = source
            target_centered = target

        result = self.procrustes_rotation(source_centered, target_centered, allow_scaling)

        if center:
            result = ProcrustesResult(
                rotation=result.rotation,
                scale=result.scale,
                translation=target_mean - result.scale * b.matmul(source_mean, result.rotation),
                residual=result.residual,
            )

        aligned = result.scale * b.matmul(source, result.rotation) + result.translation

        return aligned, result

    def effective_rank(
        self,
        eigenvalues: "Array",
        variance_threshold: float = 0.95,
    ) -> int:
        """Compute effective rank from eigenvalues.

        Finds the number of eigenvalues needed to capture the specified
        fraction of total variance.

        Args:
            eigenvalues: Eigenvalues in descending order
            variance_threshold: Fraction of variance to capture (0-1)

        Returns:
            Number of components needed
        """
        b = self._backend
        b.eval(eigenvalues)
        eig_raw = b.to_numpy(eigenvalues)
        # Filter to positive eigenvalues using pure Python
        eig_positive = [float(e) for e in eig_raw if e > 0]

        if len(eig_positive) == 0:
            return 0

        total = sum(eig_positive)
        if total <= 0:
            return 0

        cumulative = 0.0
        for i, val in enumerate(eig_positive):
            cumulative += val
            if cumulative / total >= variance_threshold:
                return i + 1

        return len(eig_positive)

    def entropy_effective_rank(self, eigenvalues: "Array") -> float:
        """Compute entropy-based effective rank.

        Uses the exponential of Shannon entropy of normalized eigenvalues:
        erank = exp(-sum(p * log(p)))

        This gives a continuous measure of dimensionality.

        Args:
            eigenvalues: Eigenvalues (any order)

        Returns:
            Entropy-based effective rank
        """
        b = self._backend
        b.eval(eigenvalues)
        eig_np = b.to_numpy(eigenvalues)
        # Filter positive eigenvalues using pure Python
        eig_positive = [float(e) for e in eig_np if e > 0]

        if len(eig_positive) == 0:
            return 0.0

        # Normalize to probability distribution
        total = sum(eig_positive)
        p = [e / total for e in eig_positive]

        # Shannon entropy using pure Python
        entropy = -sum(pi * math.log(pi + 1e-12) for pi in p)

        return math.exp(entropy)

    def cosine_similarity_matrix(self, X: "Array") -> "Array":
        """Compute pairwise cosine similarity matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        b = self._backend
        # Normalize rows
        norms = b.norm(X, axis=1, keepdims=True)
        norms = b.maximum(norms, b.full(norms.shape, 1e-12))
        X_normalized = X / norms

        # Compute cosine similarity as dot product of normalized vectors
        return b.matmul(X_normalized, b.transpose(X_normalized))

    def weighted_cosine_similarity(
        self,
        a: "Array",
        b_vec: "Array",
        weights: "Array",
    ) -> float:
        """Compute weighted cosine similarity between vectors.

        Args:
            a: First vector
            b_vec: Second vector
            weights: Feature weights

        Returns:
            Weighted cosine similarity
        """
        backend = self._backend

        # Apply weights
        wa = a * weights
        wb = b_vec * weights

        # Compute weighted norms
        norm_a = backend.sqrt(backend.sum(wa * a))
        norm_b = backend.sqrt(backend.sum(wb * b_vec))

        backend.eval(norm_a, norm_b)
        norm_a_val = float(backend.to_numpy(norm_a))
        norm_b_val = float(backend.to_numpy(norm_b))

        if norm_a_val < 1e-12 or norm_b_val < 1e-12:
            return 0.0

        # Weighted dot product divided by weighted norms
        dot = backend.sum(wa * b_vec)
        backend.eval(dot)
        return float(backend.to_numpy(dot)) / (norm_a_val * norm_b_val)

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

import numpy as np
from dataclasses import dataclass


@dataclass
class ProcrustesResult:
    """Result of Procrustes alignment."""
    rotation: np.ndarray  # Orthogonal rotation matrix
    scale: float  # Optimal scale factor
    translation: np.ndarray  # Translation vector (if computed)
    residual: float  # Procrustes distance (sum of squared errors)


class MatrixUtils:
    """Matrix utilities for geometry operations.

    This is the canonical implementation for matrix operations.
    DO NOT reimplement these operations elsewhere.
    """

    @staticmethod
    def compute_gram_matrix(X: np.ndarray, kernel: str = 'linear') -> np.ndarray:
        """Compute the Gram matrix (kernel matrix) of X.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            kernel: Kernel type ('linear' or 'rbf')

        Returns:
            Gram matrix of shape (n_samples, n_samples)
        """
        if kernel == 'linear':
            return X @ X.T
        elif kernel == 'rbf':
            # Gaussian RBF kernel with default bandwidth
            sq_dists = MatrixUtils.pairwise_squared_distances(X)
            # Use median heuristic for bandwidth
            median_dist = np.median(sq_dists[sq_dists > 0])
            gamma = 1.0 / (2.0 * median_dist) if median_dist > 0 else 1.0
            return np.exp(-gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

    @staticmethod
    def center_matrix(K: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Center a kernel matrix (double centering).

        For unweighted centering, this computes: H @ K @ H
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
            row_mean = K.mean(axis=1, keepdims=True)
            col_mean = K.mean(axis=0, keepdims=True)
            grand_mean = K.mean()
            return K - row_mean - col_mean + grand_mean
        else:
            # Weighted centering
            weights = weights / weights.sum()
            row_mean = (K * weights).sum(axis=1, keepdims=True)
            col_mean = (K * weights[:, np.newaxis]).sum(axis=0, keepdims=True)
            grand_mean = (K * np.outer(weights, weights)).sum()
            return K - row_mean - col_mean + grand_mean

    @staticmethod
    def pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances.

        Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y

        This is the canonical implementation used by CKA, intrinsic dimension,
        and other geometry operations.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        # Compute squared norms
        sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
        sq_dists = sq_norms + sq_norms.T - 2 * (X @ X.T)
        # Ensure non-negative (numerical precision)
        np.maximum(sq_dists, 0, out=sq_dists)
        return sq_dists

    @staticmethod
    def pairwise_distances(X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        return np.sqrt(MatrixUtils.pairwise_squared_distances(X))

    @staticmethod
    def procrustes_rotation(
        source: np.ndarray,
        target: np.ndarray,
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
        # Compute cross-covariance matrix
        M = source.T @ target

        # SVD
        U, S, Vt = np.linalg.svd(M)

        # Optimal orthogonal rotation
        R = U @ Vt

        # Handle reflection if determinant is -1
        if np.linalg.det(R) < 0:
            # Flip sign of last column of U
            U[:, -1] *= -1
            R = U @ Vt

        # Compute scale if requested
        if allow_scaling:
            # Optimal scale: trace(S) / trace(source.T @ source)
            source_variance = np.trace(source.T @ source)
            if source_variance > 0:
                scale = np.sum(S) / source_variance
            else:
                scale = 1.0
        else:
            scale = 1.0

        # Compute residual
        aligned = scale * (source @ R)
        residual = np.sum((target - aligned) ** 2)

        return ProcrustesResult(
            rotation=R,
            scale=scale,
            translation=np.zeros(source.shape[1]),
            residual=residual,
        )

    @staticmethod
    def procrustes_align(
        source: np.ndarray,
        target: np.ndarray,
        center: bool = True,
        allow_scaling: bool = False,
    ) -> tuple[np.ndarray, ProcrustesResult]:
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
        source_centered = source.copy()
        target_centered = target.copy()

        if center:
            source_mean = source.mean(axis=0)
            target_mean = target.mean(axis=0)
            source_centered = source - source_mean
            target_centered = target - target_mean
        else:
            source_mean = np.zeros(source.shape[1])
            target_mean = np.zeros(target.shape[1])

        result = MatrixUtils.procrustes_rotation(
            source_centered, target_centered, allow_scaling
        )

        if center:
            result.translation = target_mean - result.scale * (source_mean @ result.rotation)

        aligned = result.scale * (source @ result.rotation) + result.translation

        return aligned, result

    @staticmethod
    def effective_rank(
        eigenvalues: np.ndarray,
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
        eigenvalues = np.asarray(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) == 0:
            return 0

        total = np.sum(eigenvalues)
        if total <= 0:
            return 0

        cumulative = np.cumsum(eigenvalues) / total
        return int(np.searchsorted(cumulative, variance_threshold) + 1)

    @staticmethod
    def entropy_effective_rank(eigenvalues: np.ndarray) -> float:
        """Compute entropy-based effective rank.

        Uses the exponential of Shannon entropy of normalized eigenvalues:
        erank = exp(-sum(p * log(p)))

        This gives a continuous measure of dimensionality.

        Args:
            eigenvalues: Eigenvalues (any order)

        Returns:
            Entropy-based effective rank
        """
        eigenvalues = np.asarray(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 0]

        if len(eigenvalues) == 0:
            return 0.0

        # Normalize to probability distribution
        p = eigenvalues / np.sum(eigenvalues)

        # Shannon entropy
        entropy = -np.sum(p * np.log(p + 1e-12))

        return float(np.exp(entropy))

    @staticmethod
    def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        # Normalize rows
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        X_normalized = X / norms

        # Compute cosine similarity as dot product of normalized vectors
        return X_normalized @ X_normalized.T

    @staticmethod
    def weighted_cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Compute weighted cosine similarity between vectors.

        Args:
            a: First vector
            b: Second vector
            weights: Feature weights

        Returns:
            Weighted cosine similarity
        """
        # Apply weights
        wa = a * weights
        wb = b * weights

        # Compute weighted norms
        norm_a = np.sqrt(np.sum(wa * a))
        norm_b = np.sqrt(np.sum(wb * b))

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        # Weighted dot product divided by weighted norms
        dot = np.sum(wa * b)
        return float(dot / (norm_a * norm_b))

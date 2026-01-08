# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Geodesic RBF CKA - Centered Kernel Alignment on Riemannian Manifolds.

This module implements Centered Kernel Alignment (CKA) using RBF kernels with
geodesic distances. This is the mathematically correct construction for
high-dimensional neural representation spaces.

Mathematical Foundation:
    1. Geodesic Distance: d_geo(x, y) = shortest path on manifold
    2. RBF Gram: K(x, y) = exp(-d_geo²(x, y) / 2σ²)
    3. Centering: K_c = H @ K @ H where H = I - (1/n)11ᵀ
    4. CKA: HSIC(K_a, K_b) / √(HSIC(K_a, K_a) × HSIC(K_b, K_b))

The computation chain:
    activations → geodesic distances → RBF Gram → centered Gram → CKA
                     ↑ cache            ↑ cache      ↑ cache

Each step caches its result. Everything derives from cached intermediates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median_nonzero,
    division_epsilon,
    is_finite,
    machine_epsilon,
    power_iteration_eigh,
    regularization_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


# =============================================================================
# CACHE
# =============================================================================

def _cache() -> ComputationCache:
    """Get the shared computation cache."""
    return ComputationCache.shared()


# =============================================================================
# TYPES
# =============================================================================

class HSICEstimator(Enum):
    """HSIC estimator type. BIASED is default and fastest."""
    BIASED = "biased"
    UNBIASED = "unbiased"
    AUTO = "auto"


@dataclass
class CKAResult:
    """Result of CKA computation."""
    cka: float
    hsic_xy: float
    hsic_xx: float
    hsic_yy: float
    sample_count: int
    cka_corrected: float | None = None
    correction_factor: float | None = None

    @property
    def is_valid(self) -> bool:
        """Check if result is valid (not NaN/Inf)."""
        import math
        return math.isfinite(self.cka) and self.cka >= 0.0

    @property
    def best(self) -> float:
        """Return corrected CKA if available, else raw."""
        if self.cka_corrected is not None:
            return self.cka_corrected
        return self.cka


# =============================================================================
# CORE COMPUTATION CHAIN
# =============================================================================

def geodesic_squared_distances(
    X: "Array",
    backend: "Backend",
) -> "Array":
    """
    Compute pairwise squared geodesic distances.

    This is the foundation of all geometry. Cached.

    Returns:
        [n, n] matrix of squared geodesic distances.
    """
    cache = _cache()
    key = cache.make_array_key(X, backend)
    geodesic_key = f"geo:{key}"

    cached = cache.get_geodesic(geodesic_key)
    if cached is not None:
        return cached

    # Compute geodesic distances via Riemannian geometry
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(X)
    dist_matrix = geo_result.distances

    # Square the distances for RBF kernel
    sq_dist = dist_matrix * dist_matrix
    backend.eval(sq_dist)

    cache.set_geodesic(geodesic_key, sq_dist, 0.0)
    return sq_dist


def rbf_gram_matrix(
    X: "Array",
    backend: "Backend",
    sigma: float | None = None,
) -> "Array":
    """
    Compute RBF Gram matrix from activations.

    K(x, y) = exp(-d_geo²(x, y) / 2σ²)

    Caches the result when sigma is auto-computed (None).
    Custom sigma values bypass the cache.
    """
    cache = _cache()
    gram_key = cache.make_gram_key(X, backend, kernel_type="rbf")

    # Only use cache when sigma is auto-computed
    if sigma is None:
        cached = cache.get_gram(gram_key)
        if cached is not None:
            return cached

    # Get geodesic squared distances (cached)
    sq_dist = geodesic_squared_distances(X, backend)
    n = int(X.shape[0])

    # Compute sigma if not provided (median heuristic)
    computed_sigma = sigma
    if computed_sigma is None:
        # Median of non-zero squared distances (shared utility)
        median_val = compute_median_nonzero(sq_dist, backend)

        if median_val > 0:
            computed_sigma = sqrt_scalar(median_val / 2, backend)
        else:
            computed_sigma = 1.0
        computed_sigma = max(computed_sigma, regularization_epsilon(backend, X))

    # RBF kernel: K = exp(-D² / 2σ²)
    gram = backend.exp(-sq_dist / (2 * computed_sigma * computed_sigma))
    backend.eval(gram)

    # Only cache when using auto-computed sigma
    if sigma is None:
        cache.set_gram(gram_key, gram, 0.0)

    return gram


def rbf_gram_matrix_with_sigma(
    X: "Array",
    backend: "Backend",
    sigma: float | None = None,
) -> tuple["Array", float]:
    """
    Compute RBF Gram and return sigma used.
    
    Useful for reusing sigma across source/target comparisons.
    """
    cache = _cache()
    arr_key = cache.make_array_key(X, backend)
    
    # Get geodesic squared distances (cached)
    sq_dist = geodesic_squared_distances(X, backend)
    n = int(X.shape[0])
    
    # Compute sigma if not provided (median heuristic)
    if sigma is None:
        # Median of non-zero squared distances (shared utility)
        median_val = compute_median_nonzero(sq_dist, backend)

        if median_val > 0:
            sigma = sqrt_scalar(median_val / 2, backend)
        else:
            sigma = 1.0
        sigma = max(sigma, regularization_epsilon(backend, X))
    
    # RBF kernel
    gram = backend.exp(-sq_dist / (2 * sigma * sigma))
    backend.eval(gram)
    
    return gram, sigma


def _center_gram_matrix(
    gram: "Array",
    backend: "Backend",
    cache_key: str | None = None,
) -> "Array":
    """
    Center a Gram matrix: K_c = H @ K @ H where H = I - (1/n)11ᵀ.
    
    Efficient implementation without explicit H construction.
    Caches result if cache_key provided.
    """
    if cache_key is not None:
        cache = _cache()
        centered_key = cache.make_centered_gram_key(cache_key)
        cached = cache.get_centered_gram(centered_key)
        if cached is not None:
            return cached
    
    n = gram.shape[0]
    
    # K_c = K - row_mean - col_mean + grand_mean
    row_mean = backend.mean(gram, axis=1, keepdims=True)
    col_mean = backend.mean(gram, axis=0, keepdims=True)
    grand_mean = backend.mean(gram)
    
    centered = gram - row_mean - col_mean + grand_mean
    backend.eval(centered)
    
    if cache_key is not None:
        cache.set_centered_gram(centered_key, centered, 0.0)
    
    return centered


# =============================================================================
# HSIC COMPUTATION
# =============================================================================

def _hsic_from_centered(
    centered_a: "Array",
    centered_b: "Array",
    backend: "Backend",
) -> float:
    """
    Compute HSIC from pre-centered Gram matrices.
    
    HSIC(K, L) = trace(K_c @ L_c) / (n-1)²
    
    Uses element-wise multiply + sum for efficiency (equivalent to trace of product).
    """
    n = int(centered_a.shape[0])
    if n <= 1:
        return 0.0
    
    # trace(A @ B) = sum(A * B) for symmetric matrices
    hsic_raw = backend.sum(centered_a * centered_b)
    backend.eval(hsic_raw)
    
    return float(backend.to_scalar(hsic_raw)) / ((n - 1) ** 2)


def _hsic_unbiased(
    gram_a: "Array",
    gram_b: "Array",
    backend: "Backend",
) -> float:
    """
    Unbiased HSIC estimator (Song et al. 2012).
    
    Required for high-dimensional (P >> N) settings.
    """
    n = int(gram_a.shape[0])
    if n < 4:
        return 0.0
    
    # Zero diagonal for K_tilde, L_tilde
    k_tilde = gram_a - backend.diag(backend.diag(gram_a))
    l_tilde = gram_b - backend.diag(backend.diag(gram_b))
    backend.eval(k_tilde, l_tilde)
    
    # Term 1: trace(K_tilde @ L_tilde)
    term1 = backend.sum(k_tilde * l_tilde)
    
    # Term 2: (1ᵀ K_tilde 1)(1ᵀ L_tilde 1) / ((n-1)(n-2))
    sum_k = backend.sum(k_tilde)
    sum_l = backend.sum(l_tilde)
    term2 = (sum_k * sum_l) / ((n - 1) * (n - 2))
    
    # Term 3: (2/(n-2)) * 1ᵀ K_tilde L_tilde 1
    kl = backend.matmul(k_tilde, l_tilde)
    term3 = (2.0 / (n - 2)) * backend.sum(kl)
    
    backend.eval(term1, term2, term3)
    
    hsic = (float(backend.to_scalar(term1)) + 
            float(backend.to_scalar(term2)) - 
            float(backend.to_scalar(term3))) / (n * (n - 3))
    
    return max(0.0, hsic)


# =============================================================================
# CKA COMPUTATION
# =============================================================================

def compute_cka(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
    estimator: HSICEstimator = HSICEstimator.BIASED,
    feature_bias_correction: bool = False,
) -> CKAResult:
    """
    Compute CKA between two activation matrices.
    
    This is the main entry point. Uses the full computation chain:
    activations → geodesic distances → RBF Gram → centered → CKA
    
    All intermediate results are cached.
    
    Args:
        activations_x: [n_samples, features_x]
        activations_y: [n_samples, features_y]
        backend: Backend protocol. If None, uses default.
        estimator: HSIC estimator type.
        feature_bias_correction: Apply Chun et al. 2025 correction.
    
    Returns:
        CKAResult with CKA similarity and HSIC values.
    """
    if backend is None:
        backend = get_default_backend()
    
    # Auto-convert lists to arrays
    if isinstance(activations_x, list):
        activations_x = backend.array(activations_x)
        activations_x = backend.astype(activations_x, "float32")
    if isinstance(activations_y, list):
        activations_y = backend.array(activations_y)
        activations_y = backend.astype(activations_y, "float32")
    
    n = int(activations_x.shape[0])
    if n <= 1:
        return CKAResult(0.0, 0.0, 0.0, 0.0, n)
    
    if activations_x.shape[0] != activations_y.shape[0]:
        return CKAResult(0.0, 0.0, 0.0, 0.0, n)
    
    # Get RBF Gram matrices (cached via geodesic distances)
    gram_x = rbf_gram_matrix(activations_x, backend)
    gram_y = rbf_gram_matrix(activations_y, backend)
    
    # Get centered Gram matrices (cached)
    cache = _cache()
    key_x = cache.make_gram_key(activations_x, backend, kernel_type="rbf")
    key_y = cache.make_gram_key(activations_y, backend, kernel_type="rbf")
    centered_x = _center_gram_matrix(gram_x, backend, key_x)
    centered_y = _center_gram_matrix(gram_y, backend, key_y)
    
    # Compute HSIC
    use_unbiased = (
        estimator == HSICEstimator.UNBIASED or
        (estimator == HSICEstimator.AUTO and 
         max(activations_x.shape[1], activations_y.shape[1]) > n and n >= 4)
    )
    
    if use_unbiased and n >= 4:
        hsic_xy = _hsic_unbiased(gram_x, gram_y, backend)
        hsic_xx = _hsic_unbiased(gram_x, gram_x, backend)
        hsic_yy = _hsic_unbiased(gram_y, gram_y, backend)
    else:
        hsic_xy = _hsic_from_centered(centered_x, centered_y, backend)
        hsic_xx = _hsic_from_centered(centered_x, centered_x, backend)
        hsic_yy = _hsic_from_centered(centered_y, centered_y, backend)
    
    # CKA = HSIC(x,y) / sqrt(HSIC(x,x) * HSIC(y,y))
    denom = sqrt_scalar(hsic_xx * hsic_yy, backend)
    eps = division_epsilon(backend, gram_x)
    
    if denom < eps:
        return CKAResult(0.0, hsic_xy, hsic_xx, hsic_yy, n)
    
    cka = hsic_xy / denom
    cka = max(0.0, min(1.0, cka))
    
    # Feature bias correction (Chun et al. 2025)
    cka_corrected = None
    correction_factor = None
    if feature_bias_correction:
        corr_x = _feature_sampling_correction(centered_x, int(activations_x.shape[1]), backend)
        corr_y = _feature_sampling_correction(centered_y, int(activations_y.shape[1]), backend)
        correction_factor = corr_x[0] * corr_y[0]
        if correction_factor > 0 and is_finite(correction_factor, backend):
            cka_corrected = min(1.0, cka * correction_factor)
    
    return CKAResult(
        cka=cka,
        hsic_xy=hsic_xy,
        hsic_xx=hsic_xx,
        hsic_yy=hsic_yy,
        sample_count=n,
        cka_corrected=cka_corrected,
        correction_factor=correction_factor,
    )


def compute_cka_from_grams(
    gram_a: "Array",
    gram_b: "Array",
    backend: "Backend | None" = None,
) -> float:
    """
    Compute CKA from pre-computed RBF Gram matrices.
    
    Fast path when Grams are already computed. Skips geodesic computation.
    
    Args:
        gram_a: RBF Gram matrix [n, n]
        gram_b: RBF Gram matrix [n, n]
        backend: Backend protocol.
    
    Returns:
        CKA similarity in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()
    
    n = int(gram_a.shape[0])
    if n <= 1 or gram_a.shape != gram_b.shape:
        return 0.0
    
    # Center
    centered_a = _center_gram_matrix(gram_a, backend)
    centered_b = _center_gram_matrix(gram_b, backend)
    
    # HSIC via trace
    hsic_ab = _hsic_from_centered(centered_a, centered_b, backend)
    hsic_aa = _hsic_from_centered(centered_a, centered_a, backend)
    hsic_bb = _hsic_from_centered(centered_b, centered_b, backend)
    
    # CKA
    denom = sqrt_scalar(hsic_aa * hsic_bb, backend)
    eps = division_epsilon(backend, gram_a)
    if denom < eps:
        return 0.0
    
    return max(0.0, min(1.0, hsic_ab / denom))


def compute_cka_from_centered_grams(
    centered_a: "Array",
    centered_b: "Array",
    backend: "Backend | None" = None,
) -> float:
    """
    Compute CKA from pre-centered Gram matrices.
    
    Fastest path - when centering is already done.
    """
    if backend is None:
        backend = get_default_backend()
    
    n = int(centered_a.shape[0])
    if n <= 1 or centered_a.shape != centered_b.shape:
        return 0.0
    
    # HSIC directly
    hsic_ab = _hsic_from_centered(centered_a, centered_b, backend)
    hsic_aa = _hsic_from_centered(centered_a, centered_a, backend)
    hsic_bb = _hsic_from_centered(centered_b, centered_b, backend)
    
    denom = sqrt_scalar(hsic_aa * hsic_bb, backend)
    eps = division_epsilon(backend, centered_a)
    if denom < eps:
        return 0.0
    
    return max(0.0, min(1.0, hsic_ab / denom))


def compute_linear_cka(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
) -> float:
    """
    Compute CKA using LINEAR Gram matrices: K = X @ X.T.
    
    This matches the Gram matrices used in solve_via_gram_alignment.
    For perfect alignment with linear Gram alignment, use this function
    (NOT compute_cka which uses RBF Gram).
    
    Linear CKA = HSIC(K_x, K_y) / sqrt(HSIC(K_x, K_x) * HSIC(K_y, K_y))
    where K_x = X @ X.T (linear Gram, NOT RBF)
    
    Args:
        activations_x: [n_samples, features_x]
        activations_y: [n_samples, features_y]
        backend: Backend protocol. If None, uses default.
    
    Returns:
        CKA similarity in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()
    
    n = int(activations_x.shape[0])
    if n <= 1:
        return 0.0
    if activations_x.shape[0] != activations_y.shape[0]:
        return 0.0
    
    # LINEAR Gram matrices: K = X @ X.T
    gram_x = backend.matmul(activations_x, backend.transpose(activations_x))
    gram_y = backend.matmul(activations_y, backend.transpose(activations_y))
    backend.eval(gram_x, gram_y)

    # Center the Gram matrices
    centered_x = _center_gram_matrix(gram_x, backend)
    centered_y = _center_gram_matrix(gram_y, backend)
    backend.eval(centered_x, centered_y)

    # Compute raw HSIC sums directly (no (n-1)^2 normalization needed)
    # CKA = hsic_xy / sqrt(hsic_xx * hsic_yy) and (n-1)^2 cancels out
    # Using raw sums avoids tiny values that fail epsilon thresholds
    hsic_sum_xy = backend.sum(centered_x * centered_y)
    hsic_sum_xx = backend.sum(centered_x * centered_x)
    hsic_sum_yy = backend.sum(centered_y * centered_y)
    backend.eval(hsic_sum_xy, hsic_sum_xx, hsic_sum_yy)

    hsic_xy = float(backend.to_scalar(hsic_sum_xy))
    hsic_xx = float(backend.to_scalar(hsic_sum_xx))
    hsic_yy = float(backend.to_scalar(hsic_sum_yy))

    # CKA = HSIC(x,y) / sqrt(HSIC(x,x) * HSIC(y,y))
    #
    # IMPORTANT: When both matrices are small (after normalization + centering),
    # the HSIC values can be very small (e.g., 1e-4), making denom_sq = 1e-8.
    # This is still valid if the matrices are proportional - CKA should be 1.0.
    # Only return 0.0 for truly degenerate cases (zero variance).
    #
    # Use the sqrt of machine_epsilon as threshold since we're checking denom_sq.
    # For float32: sqrt(1.19e-7) ≈ 3.5e-4, so denom_sq threshold is ~1e-7.
    # But if HSIC values are consistently small, we should still compute CKA.
    # The true degenerate case is when ONE of hsic_xx or hsic_yy is zero.
    eps = machine_epsilon(backend, gram_x)

    # Check for truly degenerate cases (zero self-similarity)
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0

    # Compute denominator - this is safe since hsic_xx, hsic_yy > 0
    denom_sq = hsic_xx * hsic_yy
    denom = sqrt_scalar(denom_sq, backend)

    # Only fail if denominator is actually zero (numerical underflow)
    if denom <= 0:
        return 0.0

    return max(0.0, min(1.0, hsic_xy / denom))


# =============================================================================
# FEATURE BIAS CORRECTION
# =============================================================================

def _participation_ratio(eigenvalues: "Array", backend: "Backend") -> float:
    """Participation ratio (effective rank) from eigenvalues."""
    zero = backend.zeros_like(eigenvalues)
    eigvals = backend.maximum(eigenvalues, zero)
    sum_vals = float(backend.to_scalar(backend.sum(eigvals)))
    sum_sq = float(backend.to_scalar(backend.sum(eigvals * eigvals)))
    
    if not is_finite(sum_vals, backend) or sum_sq <= 0:
        return 0.0
    return (sum_vals * sum_vals) / sum_sq


def _feature_sampling_correction(
    centered_gram: "Array",
    feature_dim: int,
    backend: "Backend",
) -> tuple[float, float]:
    """
    Feature-sampling correction (Chun et al. 2025).
    
    Returns (correction_factor, gamma).
    """
    if feature_dim <= 0:
        return 1.0, 0.0
    
    n = int(centered_gram.shape[0])
    eigvals, _ = power_iteration_eigh(backend, centered_gram, k=n)
    backend.eval(eigvals)
    
    gamma = _participation_ratio(eigvals, backend)
    if gamma <= 0:
        return 1.0, gamma
    
    gamma = max(gamma, 1.0)
    correction = sqrt_scalar(1.0 + (gamma - 1.0) / float(feature_dim), backend)
    
    if not is_finite(correction, backend) or correction <= 0:
        return 1.0, gamma
    
    return correction, gamma


# =============================================================================
# LEGACY COMPATIBILITY (deprecated - all use geodesic RBF now)
# =============================================================================

def compute_cka_backend(
    x: "Array",
    y: "Array",
    backend: "Backend",
    estimator: HSICEstimator = HSICEstimator.BIASED,
    feature_bias_correction: bool = False,
) -> float:
    """
    DEPRECATED: Use compute_cka() instead. This now uses geodesic RBF.
    
    Kept for API compatibility during transition.
    """
    result = compute_cka(x, y, backend, estimator, feature_bias_correction)
    return result.best if result.is_valid else 0.0


def compute_cka_from_lists(
    x: list[list[float]],
    y: list[list[float]],
    backend: "Backend | None" = None,
    estimator: HSICEstimator = HSICEstimator.BIASED,
    feature_bias_correction: bool = False,
) -> float:
    """Compute CKA from Python lists."""
    if backend is None:
        backend = get_default_backend()
    
    arr_x = backend.array(x)
    arr_y = backend.array(y)
    arr_x = backend.astype(arr_x, "float32")
    arr_y = backend.astype(arr_y, "float32")
    
    result = compute_cka(arr_x, arr_y, backend, estimator, feature_bias_correction)
    return result.best if result.is_valid else 0.0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "compute_cka",
    "compute_linear_cka",  # For linear Gram alignment validation
    "compute_cka_from_grams",
    "compute_cka_from_centered_grams",
    "rbf_gram_matrix",
    "rbf_gram_matrix_with_sigma",
    # Types
    "HSICEstimator",
    "CKAResult",
    # Internal (needed by gram_aligner)
    "_center_gram_matrix",
    "_feature_sampling_correction",
    # Legacy
    "compute_cka_backend",
    "compute_cka_from_lists",
]

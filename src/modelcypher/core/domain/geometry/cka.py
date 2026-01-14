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

Geodesic distances are cached. Gram/centering caches are used only when
per-array sigma is used; shared-sigma paths skip caching to avoid key collisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_pinv,
    is_finite,
    machine_epsilon,
    power_iteration_eigh,
    precision_dtype,
    regularization_epsilon,
    sqrt_scalar,
    tiny_value,
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

    # Compute sigma if not provided (data-derived gap scale)
    computed_sigma = sigma
    if computed_sigma is None:
        computed_sigma = _derive_rbf_sigma_from_values(sq_dist, backend)

    # RBF kernel: K = exp(-D² / 2σ²)
    gram = _rbf_gram_from_sq_distances(sq_dist, computed_sigma, backend)

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
    
    # Compute sigma if not provided (data-derived gap scale)
    if sigma is None:
        sigma = _derive_rbf_sigma_from_values(sq_dist, backend)
    
    # RBF kernel
    gram = _rbf_gram_from_sq_distances(sq_dist, sigma, backend)
    
    return gram, sigma


def _rbf_gram_from_sq_distances(
    sq_dist: "Array",
    sigma: float,
    backend: "Backend",
) -> "Array":
    """Compute RBF Gram from precomputed squared distances."""
    gram = backend.exp(-sq_dist / (2 * sigma * sigma))
    backend.eval(gram)
    return gram


def _shared_rbf_sigma(
    sq_dist_x: "Array",
    sq_dist_y: "Array",
    backend: "Backend",
) -> float:
    """Compute a shared RBF sigma from combined distance statistics."""
    flat_x = backend.reshape(sq_dist_x, (-1,))
    flat_y = backend.reshape(sq_dist_y, (-1,))
    combined = backend.concatenate([flat_x, flat_y], axis=0)
    return _derive_rbf_sigma_from_values(combined, backend)


def _derive_rbf_sigma_from_values(
    values: "Array",
    backend: "Backend",
) -> float:
    """Derive RBF sigma from the data distribution (no fixed heuristics)."""
    flat = backend.reshape(values, (-1,))
    backend.eval(flat)

    eps = division_epsilon(backend, flat)
    mask = flat > eps
    count_arr = backend.sum(backend.astype(mask, "int32"))
    backend.eval(count_arr)
    count = int(backend.to_scalar(count_arr))
    if count <= 0:
        return regularization_epsilon(backend, values)

    inf = backend.full(flat.shape, float("inf"))
    filtered = backend.where(mask, flat, inf)
    sorted_vals = backend.sort(filtered)
    backend.eval(sorted_vals)
    vals = backend.take(sorted_vals, backend.arange(count), axis=0)
    backend.eval(vals)

    scale = None
    if count >= 2:
        curr = vals[:-1]
        next_vals = vals[1:]
        denom = backend.where(curr > eps, curr, backend.ones_like(curr))
        rel_gap = (next_vals - curr) / denom
        max_gap_arr = backend.max(rel_gap)
        gap_idx_arr = backend.argmax(rel_gap)
        backend.eval(max_gap_arr, gap_idx_arr)
        max_gap = float(backend.to_scalar(max_gap_arr))
        if max_gap > eps:
            idx = int(backend.to_scalar(gap_idx_arr))
            threshold_arr = vals[idx : idx + 1]
            backend.eval(threshold_arr)
            scale = float(backend.to_scalar(threshold_arr))

    if scale is None:
        sum_vals = backend.sum(vals)
        backend.eval(sum_vals)
        scale = float(backend.to_scalar(sum_vals)) / float(count)

    if scale <= 0.0:
        scale = eps

    sigma = sqrt_scalar(scale / 2.0, backend)
    return max(sigma, regularization_epsilon(backend, values))


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
    
    Geodesic distances are cached. Grams use a shared sigma for precision.
    
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
        activations_x = backend.astype(
            activations_x, precision_dtype(backend, reference=activations_x)
        )
    if isinstance(activations_y, list):
        activations_y = backend.array(activations_y)
        activations_y = backend.astype(
            activations_y, precision_dtype(backend, reference=activations_y)
        )
    
    n = int(activations_x.shape[0])
    if n <= 1:
        return CKAResult(0.0, 0.0, 0.0, 0.0, n)
    
    if activations_x.shape[0] != activations_y.shape[0]:
        return CKAResult(0.0, 0.0, 0.0, 0.0, n)
    
    # Get geodesic squared distances (cached) and use a shared sigma.
    # Shared bandwidth removes sigma skew between X/Y and improves precision.
    sq_dist_x = geodesic_squared_distances(activations_x, backend)
    sq_dist_y = geodesic_squared_distances(activations_y, backend)
    sigma = _shared_rbf_sigma(sq_dist_x, sq_dist_y, backend)

    # Build RBF Gram matrices from shared sigma.
    gram_x = _rbf_gram_from_sq_distances(sq_dist_x, sigma, backend)
    gram_y = _rbf_gram_from_sq_distances(sq_dist_y, sigma, backend)
    
    # Get centered Gram matrices (do not cache: sigma differs from per-array default).
    centered_x = _center_gram_matrix(gram_x, backend, cache_key=None)
    centered_y = _center_gram_matrix(gram_y, backend, cache_key=None)
    
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

    # Epsilon for division safety based on HSIC scale, not Gram scale.
    # HSIC values are normalized by n² and can be legitimately small for large n.
    # Use machine epsilon times the larger HSIC self-similarity as threshold.
    hsic_scale = max(hsic_xx, hsic_yy, tiny_value(backend, gram_x))
    eps = machine_epsilon(backend, gram_x) * sqrt_scalar(hsic_scale, backend)

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

    # Epsilon based on HSIC scale, not Gram scale
    hsic_scale = max(hsic_aa, hsic_bb, tiny_value(backend, gram_a))
    eps = machine_epsilon(backend, gram_a) * sqrt_scalar(hsic_scale, backend)
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

    # Epsilon based on HSIC scale, not centered Gram scale
    hsic_scale = max(hsic_aa, hsic_bb, tiny_value(backend, centered_a))
    eps = machine_epsilon(backend, centered_a) * sqrt_scalar(hsic_scale, backend)
    if denom < eps:
        return 0.0

    return max(0.0, min(1.0, hsic_ab / denom))


def compute_geodesic_cka(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
) -> float:
    """
    Compute CKA using geodesic RBF Gram matrices.

    Uses geodesic distances (k-NN graph + shortest paths) to properly
    handle curved neural representation manifolds. RBF kernel converts
    geodesic distances to similarities.

    CKA = HSIC(K_x, K_y) / sqrt(HSIC(K_x, K_x) * HSIC(K_y, K_y))
    where K_x, K_y are geodesic RBF Gram matrices.

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

    # Geodesic RBF Gram matrices: proper manifold distance
    sq_dist_x = geodesic_squared_distances(activations_x, backend)
    sq_dist_y = geodesic_squared_distances(activations_y, backend)
    sigma = _shared_rbf_sigma(sq_dist_x, sq_dist_y, backend)
    gram_x = _rbf_gram_from_sq_distances(sq_dist_x, sigma, backend)
    gram_y = _rbf_gram_from_sq_distances(sq_dist_y, sigma, backend)
    backend.eval(gram_x, gram_y)

    # Center the Gram matrices (don't cache - sigma is shared, not per-array)
    centered_x = _center_gram_matrix(gram_x, backend, cache_key=None)
    centered_y = _center_gram_matrix(gram_y, backend, cache_key=None)
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


def compute_linear_cka(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
) -> float:
    """DEPRECATED: use compute_geodesic_cka (geodesic RBF kernel)."""
    return compute_geodesic_cka(activations_x, activations_y, backend)


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
    arr_x = backend.astype(arr_x, precision_dtype(backend, reference=arr_x))
    arr_y = backend.astype(arr_y, precision_dtype(backend, reference=arr_y))
    
    result = compute_cka(arr_x, arr_y, backend, estimator, feature_bias_correction)
    return result.best if result.is_valid else 0.0


# =============================================================================
# SPLIT CKA: SHARED VS. NOVEL CONCEPTS
# =============================================================================


@dataclass
class SplitCKAResult:
    """Result of CKA computation split by shared vs. novel concepts.

    The invariant shape hypothesis says both models encode the same geometry.
    But they differ in:
    - Coverage: which concepts they encode (shared vs. novel)
    - Density: how precisely concepts are locked in

    For SHARED concepts (both models have), CKA should be high after alignment
    on the shared manifold (near 1.0).
    For NOVEL concepts (source has, target doesn't), CKA is undefined/low.
    """
    # CKA on shared concepts only - expected high after alignment
    shared_cka: float
    # CKA on novel concepts - expected to be low (target has no structure)
    novel_cka: float
    # Full CKA (on all samples) - blended measure
    full_cka: float
    # Fraction of samples that are shared vs. novel
    shared_fraction: float
    novel_fraction: float
    # Sample counts
    n_shared: int
    n_novel: int
    n_total: int
    # Response magnitudes (for debugging)
    source_response_mean: float
    target_response_mean: float


def compute_cka_split(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
    response_threshold: float | None = None,
    feature_transform: "Array | None" = None,
) -> SplitCKAResult:
    """Compute CKA separately for shared vs. novel concepts.

    Separates samples into:
    - SHARED: target lies in the aligned source column space (projection residual ~ 0)
    - NOVEL: target has residual outside the aligned source column space

    Shared/novel is derived from a closed-form projection residual:
    - Compute aligned target projection T_shared = S @ pinv(S) @ T
    - Residual R = T - T_shared is the novel component
    - Samples with residual norm <= precision-scaled target norm are "shared"
    - Samples with residual norm > precision-scaled target norm are "novel"

    Args:
        source_activations: [n_samples, d_source]
        target_activations: [n_samples, d_target]
        backend: Backend protocol. If None, uses default.
        response_threshold: Ignored (kept for backward compatibility).
        feature_transform: Optional alignment transform to reuse (source @ F ≈ target).

    Returns:
        SplitCKAResult with separate CKA for shared vs. novel concepts.
    """
    if backend is None:
        backend = get_default_backend()

    b = backend
    n = int(source_activations.shape[0])
    _ = response_threshold

    if n < 4:
        # Not enough samples to split
        return SplitCKAResult(
            shared_cka=0.0, novel_cka=0.0, full_cka=0.0,
            shared_fraction=0.0, novel_fraction=0.0,
            n_shared=0, n_novel=0, n_total=n,
            source_response_mean=0.0, target_response_mean=0.0,
        )

    if source_activations.shape[0] != target_activations.shape[0]:
        return SplitCKAResult(
            shared_cka=0.0, novel_cka=0.0, full_cka=0.0,
            shared_fraction=0.0, novel_fraction=0.0,
            n_shared=0, n_novel=0, n_total=n,
            source_response_mean=0.0, target_response_mean=0.0,
        )

    source_arr = b.astype(
        source_activations, precision_dtype(b, reference=source_activations)
    )
    target_arr = b.astype(
        target_activations, precision_dtype(b, reference=target_activations)
    )
    b.eval(source_arr, target_arr)

    if feature_transform is None:
        F = b.matmul(geodesic_pinv(b, source_arr), target_arr)
    else:
        F = feature_transform if hasattr(feature_transform, "dtype") else b.array(feature_transform)
        F = b.astype(F, precision_dtype(b, reference=F))
    b.eval(F)

    aligned = b.matmul(source_arr, F)
    residual = target_arr - aligned
    b.eval(aligned, residual)

    residual_norms = b.sqrt(b.sum(residual * residual, axis=1))
    target_norms = b.sqrt(b.sum(target_arr * target_arr, axis=1))
    aligned_norms = b.sqrt(b.sum(aligned * aligned, axis=1))
    b.eval(residual_norms, target_norms, aligned_norms)

    eps = machine_epsilon(b, target_arr)
    precision = sqrt_scalar(eps, b)
    precision_arr = b.array([precision], dtype=target_norms.dtype)
    threshold = precision_arr * (target_norms + precision_arr)
    shared_mask = residual_norms <= threshold
    novel_mask = residual_norms > threshold
    b.eval(shared_mask, novel_mask)

    # Count samples in each category
    count_dtype = precision_dtype(b, reference=target_norms)
    shared_count = int(b.to_scalar(b.sum(b.astype(shared_mask, count_dtype))))
    novel_count = int(b.to_scalar(b.sum(b.astype(novel_mask, count_dtype))))

    # Response means for debugging
    source_resp_mean = float(b.to_scalar(b.mean(aligned_norms)))
    target_resp_mean = float(b.to_scalar(b.mean(target_norms)))

    # Compute full CKA
    full_result = compute_cka(aligned, target_arr, backend)
    full_cka = full_result.cka if full_result.is_valid else 0.0

    # Compute CKA on shared samples only (if enough)
    shared_cka = 0.0
    if shared_count >= 4:
        # Use mask to select shared samples
        # Convert bool mask to indices
        shared_indices = b.nonzero(shared_mask)
        if len(shared_indices) > 0 and shared_indices[0].shape[0] >= 4:
            shared_idx = shared_indices[0]
            source_shared = b.take(aligned, shared_idx, axis=0)
            target_shared = b.take(target_arr, shared_idx, axis=0)
            b.eval(source_shared, target_shared)
            shared_result = compute_cka(source_shared, target_shared, backend)
            shared_cka = shared_result.cka if shared_result.is_valid else 0.0

    # Compute CKA on novel samples only (if enough)
    novel_cka = 0.0
    if novel_count >= 4:
        novel_indices = b.nonzero(novel_mask)
        if len(novel_indices) > 0 and novel_indices[0].shape[0] >= 4:
            novel_idx = novel_indices[0]
            source_novel = b.take(aligned, novel_idx, axis=0)
            target_novel = b.take(target_arr, novel_idx, axis=0)
            b.eval(source_novel, target_novel)
            novel_result = compute_cka(source_novel, target_novel, backend)
            novel_cka = novel_result.cka if novel_result.is_valid else 0.0

    return SplitCKAResult(
        shared_cka=shared_cka,
        novel_cka=novel_cka,
        full_cka=full_cka,
        shared_fraction=shared_count / n if n > 0 else 0.0,
        novel_fraction=novel_count / n if n > 0 else 0.0,
        n_shared=shared_count,
        n_novel=novel_count,
        n_total=n,
        source_response_mean=source_resp_mean,
        target_response_mean=target_resp_mean,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "compute_cka",
    "compute_geodesic_cka",
    "compute_linear_cka",
    "compute_cka_from_grams",
    "compute_cka_from_centered_grams",
    "rbf_gram_matrix",
    "rbf_gram_matrix_with_sigma",
    # Split CKA (shared vs. novel)
    "compute_cka_split",
    "SplitCKAResult",
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

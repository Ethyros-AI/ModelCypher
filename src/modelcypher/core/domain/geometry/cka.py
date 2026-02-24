# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Geodesic RBF CKA (Centered Kernel Alignment).

Implements CKA using an RBF kernel over geodesic distances, with optional
caching for distance and Gram computations.

References:
    - Kornblith et al. (2019). "Similarity of Neural Network Representations
      Revisited." arXiv:1905.00414
    - Tenenbaum et al. (2000). "Isomap" - geodesic distance via k-NN graph
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
    gpu_lstsq,
    is_finite,
    machine_epsilon,
    power_iteration_eigh,
    precision_dtype,
    regularization_epsilon,
    sqrt_scalar,
    tiny_value,
)
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry
from modelcypher.core.domain.geometry.subspace import (
    compute_subspace_overlap,
    project_to_subspace,
)

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
    int(X.shape[0])

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
    cache.make_array_key(X, backend)

    # Get geodesic squared distances (cached)
    sq_dist = geodesic_squared_distances(X, backend)
    int(X.shape[0])

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

    gram.shape[0]

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
    # Song et al. (2012, JMLR 13) unbiased HSIC has n(n-3) denominator (line 373).
    # n < 4 ⟹ n(n-3) ≤ 0 ⟹ division by zero or negative.
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
    # Unbiased HSIC (Song et al. 2012, JMLR 13) requires n ≥ 4 because
    # denominator is n(n-3). AUTO selects unbiased when P > N (high-dimensional).
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

    if denom <= 0.0 or denom < eps:
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


def compute_linear_cka_gram(
    K1: "Array",
    K2: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute linear CKA from pre-computed dot-product Gram matrices.

    This is the memory-efficient alternative to geodesic CKA. Uses O(n²) memory
    for the Gram matrices (which are already computed for alignment), rather than
    O(n²) memory for geodesic distance matrices.

    Linear CKA formula:
        CKA = HSIC(K1, K2) / sqrt(HSIC(K1, K1) × HSIC(K2, K2))
        where HSIC(K, L) = tr(K_c @ L_c) / (n-1)²

    Args:
        K1: First dot-product Gram matrix [n, n] where K1 = X @ X.T
        K2: Second dot-product Gram matrix [n, n] where K2 = Y @ Y.T
        backend: Backend protocol. If None, uses default.

    Returns:
        Linear CKA similarity in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()

    n = int(K1.shape[0])
    if n <= 1 or K1.shape != K2.shape:
        return 0.0

    # Center the Gram matrices: K_c = H @ K @ H where H = I - (1/n)11ᵀ
    K1_c = _center_gram_matrix(K1, backend)
    K2_c = _center_gram_matrix(K2, backend)
    backend.eval(K1_c, K2_c)

    # HSIC = tr(K1_c @ K2_c) - using element-wise multiply + sum for efficiency
    # trace(A @ B) = sum(A * B.T) = sum(A * B) for symmetric matrices
    hsic_12 = backend.sum(K1_c * K2_c)
    hsic_11 = backend.sum(K1_c * K1_c)
    hsic_22 = backend.sum(K2_c * K2_c)
    backend.eval(hsic_12, hsic_11, hsic_22)

    hsic_12_val = float(backend.to_scalar(hsic_12))
    hsic_11_val = float(backend.to_scalar(hsic_11))
    hsic_22_val = float(backend.to_scalar(hsic_22))

    # Check for degenerate cases (zero self-similarity)
    if hsic_11_val <= 0 or hsic_22_val <= 0:
        return 0.0

    denom = sqrt_scalar(hsic_11_val * hsic_22_val, backend)
    if denom <= 0:
        return 0.0

    return max(0.0, min(1.0, hsic_12_val / denom))


def compute_linear_cka_from_activations(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Compute linear CKA directly from activations without geodesic distances.

    This is the fast path for CKA when geodesic (manifold) structure is not needed.
    Uses linear kernel (dot-product Gram) instead of geodesic RBF kernel.

    Memory: O(n²) for Gram matrices, NOT O(n²) for geodesic distance matrices.
    Speed: O(n²d) for Gram computation, NOT O(n² × k × log(n)) for geodesic.

    Args:
        activations_x: [n_samples, features_x]
        activations_y: [n_samples, features_y]
        backend: Backend protocol. If None, uses default.

    Returns:
        Linear CKA similarity in [0, 1].
    """
    if backend is None:
        backend = get_default_backend()

    n = int(activations_x.shape[0])
    if n <= 1:
        return 0.0
    if activations_x.shape[0] != activations_y.shape[0]:
        return 0.0

    # Promote to higher precision for numerical stability
    activations_x = backend.astype(
        activations_x, precision_dtype(backend, reference=activations_x)
    )
    activations_y = backend.astype(
        activations_y, precision_dtype(backend, reference=activations_y)
    )
    backend.eval(activations_x, activations_y)

    # Compute dot-product Gram matrices: K = X @ X.T
    K1 = backend.matmul(activations_x, backend.transpose(activations_x))
    K2 = backend.matmul(activations_y, backend.transpose(activations_y))
    backend.eval(K1, K2)

    return compute_linear_cka_gram(K1, K2, backend)


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
    machine_epsilon(backend, gram_x)

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
# SPLIT CKA: SHARED VS. NOVEL CONCEPTS
# =============================================================================


@dataclass
class SplitCKAResult:
    """Result of CKA computation split by shared vs. novel concepts.

    Contains CKA metrics for shared/novel splits, plus sample counts and
    response summaries.
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
    feature_transform: "Array | None" = None,
) -> SplitCKAResult:
    """Compute CKA separately for shared vs. novel concepts.

    Separates samples into:
    - SHARED: target lies in the aligned source column space (projection residual ~ 0)
    - NOVEL: target has residual outside the aligned source column space

    Shared/novel is derived from a projection residual:
    - Compute aligned target projection T_shared = S @ pinv(S) @ T
    - Residual R = T - T_shared is the novel component
    - Samples with residual norm <= precision-scaled target norm are "shared"
    - Samples with residual norm > precision-scaled target norm are "novel"

    Args:
        source_activations: [n_samples, d_source]
        target_activations: [n_samples, d_target]
        backend: Backend protocol. If None, uses default.
        feature_transform: Optional alignment transform to reuse (source @ F ≈ target).

    Returns:
        SplitCKAResult with separate CKA for shared vs. novel concepts.
    """
    if backend is None:
        backend = get_default_backend()

    b = backend
    n = int(source_activations.shape[0])

    # Unbiased HSIC (Song et al. 2012, JMLR 13) requires n ≥ 4 (denominator n(n-3)).
    if n < 4:
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
        # Solve source @ F = target for F via closed-form normal equations
        F = gpu_lstsq(b, source_arr, target_arr)
    else:
        F = feature_transform if hasattr(feature_transform, "dtype") else b.array(feature_transform)
        F = b.astype(F, precision_dtype(b, reference=F))
    b.eval(F)

    aligned = b.matmul(source_arr, F)
    b.eval(aligned)

    # Compute response magnitudes for diagnostics
    aligned_norms = b.sqrt(b.sum(aligned * aligned, axis=1))
    target_norms = b.sqrt(b.sum(target_arr * target_arr, axis=1))
    b.eval(aligned_norms, target_norms)

    source_resp_mean = float(b.to_scalar(b.mean(aligned_norms)))
    target_resp_mean = float(b.to_scalar(b.mean(target_norms)))

    # Compute full CKA on all samples
    full_result = compute_cka(aligned, target_arr, backend)
    full_cka = full_result.cka if full_result.is_valid else 0.0

    # =================================================================
    # NEW: Rank-derived subspace splitting (replaces residual-ratio)
    # =================================================================
    # Use SVD-based principal angle analysis to identify which DIRECTIONS
    # are shared vs novel, not which SAMPLES. This is mathematically
    # grounded in the intrinsic geometry.
    #
    # The old approach used residual_ratio = ||target - aligned|| / ||target||
    # with gap-threshold detection. This failed when the distribution was
    # smooth/unimodal, causing catastrophic misclassification (e.g.,
    # n_shared=2, n_novel=815 when novel_cka=0.9999).
    # =================================================================

    try:
        subspace_result = compute_subspace_overlap(aligned, target_arr, b)

        # Get shared and novel ranks
        shared_rank = subspace_result.shared_rank
        source_rank = subspace_result.source_rank
        novel_rank = max(0, source_rank - shared_rank)

        # Overlap fraction is the fraction of source subspace that aligns with target
        shared_fraction = subspace_result.overlap_fraction
        novel_fraction = 1.0 - shared_fraction

        # Compute CKA on projections to shared/novel subspaces
        shared_cka = 0.0
        novel_cka = 0.0

        # Shared subspace CKA
        shared_basis_shape = b.shape(subspace_result.shared_basis)
        if int(shared_basis_shape[0]) > 0:
            # Project both aligned source and target onto shared subspace
            shared_src_proj = project_to_subspace(aligned, subspace_result.shared_basis, b)
            shared_tgt_proj = project_to_subspace(target_arr, subspace_result.shared_basis, b)
            b.eval(shared_src_proj, shared_tgt_proj)

            # CKA on shared subspace projections.
            # 1-feature projection → rank-1 Gram → CKA = 1.0 trivially (uninformative).
            if int(b.shape(shared_src_proj)[1]) >= 2:
                shared_result = compute_cka(shared_src_proj, shared_tgt_proj, backend)
                shared_cka = shared_result.cka if shared_result.is_valid else 0.0

        # Novel subspace CKA
        novel_basis_shape = b.shape(subspace_result.novel_basis)
        if int(novel_basis_shape[0]) > 0:
            # Project both onto novel subspace
            novel_src_proj = project_to_subspace(aligned, subspace_result.novel_basis, b)
            novel_tgt_proj = project_to_subspace(target_arr, subspace_result.novel_basis, b)
            b.eval(novel_src_proj, novel_tgt_proj)

            # CKA on novel subspace projections.
            # Low CKA here validates that these directions ARE truly novel.
            # 1-feature → rank-1 Gram → CKA degenerates (see shared subspace comment).
            if int(b.shape(novel_src_proj)[1]) >= 2:
                novel_result = compute_cka(novel_src_proj, novel_tgt_proj, backend)
                novel_cka = novel_result.cka if novel_result.is_valid else 0.0

        logger.debug(
            "Split CKA (rank-derived): shared_rank=%d, novel_rank=%d, "
            "shared_cka=%.4f, novel_cka=%.4f, overlap=%.2f%%",
            shared_rank,
            novel_rank,
            shared_cka,
            novel_cka,
            shared_fraction * 100,
        )

        return SplitCKAResult(
            shared_cka=shared_cka,
            novel_cka=novel_cka,
            full_cka=full_cka,
            shared_fraction=shared_fraction,
            novel_fraction=novel_fraction,
            n_shared=shared_rank,
            n_novel=novel_rank,
            n_total=source_rank,
            source_response_mean=source_resp_mean,
            target_response_mean=target_resp_mean,
        )

    except Exception as e:
        # Fallback: if subspace analysis fails, return degenerate result
        logger.warning("Subspace analysis failed, returning degenerate split: %s", e)
        return SplitCKAResult(
            shared_cka=full_cka,  # Use full CKA as shared
            novel_cka=0.0,
            full_cka=full_cka,
            shared_fraction=1.0,
            novel_fraction=0.0,
            n_shared=n,
            n_novel=0,
            n_total=n,
            source_response_mean=source_resp_mean,
            target_response_mean=target_resp_mean,
        )


# =============================================================================
# TOKEN-LEVEL CKA
# =============================================================================


@dataclass
class TokenCKAResult:
    """Result of token-level CKA computation.

    Provides both aggregate CKA and per-text CKA for analyzing
    token-level representation similarity with text boundary awareness.

    Attributes
    ----------
    aggregate_cka : float
        CKA computed over all tokens (ignoring text boundaries).
    per_text_cka : list[float]
        CKA computed separately for each text.
    text_lengths : list[int]
        Length of each text in tokens.
    min_text_cka : float
        Minimum per-text CKA.
    max_text_cka : float
        Maximum per-text CKA.
    mean_text_cka : float
        Mean of per-text CKA values.
    """

    aggregate_cka: float
    per_text_cka: list[float]
    text_lengths: list[int]
    min_text_cka: float
    max_text_cka: float
    mean_text_cka: float


def compute_token_cka(
    activations_x: "Array",
    activations_y: "Array",
    text_lengths_x: list[int],
    text_lengths_y: list[int],
    backend: "Backend | None" = None,
    alignment: str = "truncate",
) -> TokenCKAResult:
    """Compute CKA at token-level with text boundary awareness.

    This function handles the common case where two models may produce
    different numbers of tokens for the same input text. It provides
    multiple alignment strategies and computes CKA both aggregated
    and per-text.

    Implements token-level analysis from arXiv:2601.21571v1.

    Parameters
    ----------
    activations_x : Array
        First set of activations. Shape: [total_tokens_x, hidden_dim_x].
    activations_y : Array
        Second set of activations. Shape: [total_tokens_y, hidden_dim_y].
    text_lengths_x : list[int]
        Token count per text for activations_x.
    text_lengths_y : list[int]
        Token count per text for activations_y.
    backend : Backend, optional
        Computation backend. If None, uses default.
    alignment : str
        How to align texts with different token counts:
        - "truncate": Truncate longer text to match shorter (default).
        - "pad": Pad shorter text with zeros to match longer.
        - "dtw": Use dynamic time warping for soft alignment.

    Returns
    -------
    TokenCKAResult
        Token-level CKA results with aggregate and per-text metrics.
    """
    if backend is None:
        backend = get_default_backend()

    b = backend

    if len(text_lengths_x) != len(text_lengths_y):
        raise ValueError(
            f"Number of texts must match: {len(text_lengths_x)} vs {len(text_lengths_y)}"
        )

    n_texts = len(text_lengths_x)
    if n_texts == 0:
        return TokenCKAResult(
            aggregate_cka=0.0,
            per_text_cka=[],
            text_lengths=[],
            min_text_cka=0.0,
            max_text_cka=0.0,
            mean_text_cka=0.0,
        )

    activations_x = b.astype(activations_x, precision_dtype(b, reference=activations_x))
    activations_y = b.astype(activations_y, precision_dtype(b, reference=activations_y))
    b.eval(activations_x, activations_y)

    # Compute per-text CKA
    per_text_cka: list[float] = []
    aligned_lengths: list[int] = []

    offset_x = 0
    offset_y = 0

    for i in range(n_texts):
        len_x = text_lengths_x[i]
        len_y = text_lengths_y[i]

        # Extract text activations
        text_x = activations_x[offset_x : offset_x + len_x]
        text_y = activations_y[offset_y : offset_y + len_y]
        b.eval(text_x, text_y)

        # Align texts
        if alignment == "truncate":
            min_len = min(len_x, len_y)
            if min_len <= 1:
                per_text_cka.append(0.0)
                aligned_lengths.append(min_len)
            else:
                text_x = text_x[:min_len]
                text_y = text_y[:min_len]
                b.eval(text_x, text_y)
                cka_val = compute_geodesic_cka(text_x, text_y, backend)
                per_text_cka.append(cka_val)
                aligned_lengths.append(min_len)

        elif alignment == "pad":
            max_len = max(len_x, len_y)
            if max_len <= 1:
                per_text_cka.append(0.0)
                aligned_lengths.append(max_len)
            else:
                dim_x = int(text_x.shape[1]) if text_x.ndim > 1 else 1
                dim_y = int(text_y.shape[1]) if text_y.ndim > 1 else 1

                if len_x < max_len:
                    padding = b.zeros((max_len - len_x, dim_x))
                    text_x = b.concatenate([text_x, padding], axis=0)
                if len_y < max_len:
                    padding = b.zeros((max_len - len_y, dim_y))
                    text_y = b.concatenate([text_y, padding], axis=0)
                b.eval(text_x, text_y)
                cka_val = compute_geodesic_cka(text_x, text_y, backend)
                per_text_cka.append(cka_val)
                aligned_lengths.append(max_len)

        elif alignment == "dtw":
            # Dynamic time warping: find optimal alignment path
            cka_val = _dtw_aligned_cka(text_x, text_y, backend)
            per_text_cka.append(cka_val)
            aligned_lengths.append(max(len_x, len_y))

        else:
            raise ValueError(f"Unknown alignment method: {alignment}")

        offset_x += len_x
        offset_y += len_y

    # Compute aggregate CKA (using truncation for simplicity)
    # Collect all aligned texts
    all_x_aligned: list["Array"] = []
    all_y_aligned: list["Array"] = []

    offset_x = 0
    offset_y = 0
    for i in range(n_texts):
        len_x = text_lengths_x[i]
        len_y = text_lengths_y[i]
        min_len = min(len_x, len_y)

        if min_len > 0:
            all_x_aligned.append(activations_x[offset_x : offset_x + min_len])
            all_y_aligned.append(activations_y[offset_y : offset_y + min_len])

        offset_x += len_x
        offset_y += len_y

    if all_x_aligned:
        concat_x = b.concatenate(all_x_aligned, axis=0)
        concat_y = b.concatenate(all_y_aligned, axis=0)
        b.eval(concat_x, concat_y)
        aggregate_cka = compute_geodesic_cka(concat_x, concat_y, backend)
    else:
        aggregate_cka = 0.0

    # Summary statistics
    valid_cka = [c for c in per_text_cka if c > 0.0]
    if valid_cka:
        min_cka = min(valid_cka)
        max_cka = max(valid_cka)
        mean_cka = sum(valid_cka) / len(valid_cka)
    else:
        min_cka = 0.0
        max_cka = 0.0
        mean_cka = 0.0

    return TokenCKAResult(
        aggregate_cka=aggregate_cka,
        per_text_cka=per_text_cka,
        text_lengths=aligned_lengths,
        min_text_cka=min_cka,
        max_text_cka=max_cka,
        mean_text_cka=mean_cka,
    )


def _dtw_aligned_cka(
    text_x: "Array",
    text_y: "Array",
    backend: "Backend",
) -> float:
    """Compute CKA using DTW-aligned representations.

    Uses dynamic time warping to find the optimal alignment between
    two sequences of different lengths, then computes CKA on the
    aligned sequences.
    """
    b = backend
    n_x = int(text_x.shape[0])
    n_y = int(text_y.shape[0])

    if n_x == 0 or n_y == 0:
        return 0.0

    if n_x == n_y:
        return compute_geodesic_cka(text_x, text_y, backend)

    # Compute pairwise distances for DTW
    # Cost matrix: C[i,j] = ||x_i - y_j||^2
    # For efficiency, compute via (a-b)^2 = a^2 - 2ab + b^2

    text_x_f32 = b.astype(text_x, "float32")
    text_y_f32 = b.astype(text_y, "float32")

    # ||x||^2 and ||y||^2
    x_sq = b.sum(text_x_f32 * text_x_f32, axis=1, keepdims=True)  # [n_x, 1]
    y_sq = b.sum(text_y_f32 * text_y_f32, axis=1, keepdims=True)  # [n_y, 1]

    # -2 * x @ y.T
    xy = b.matmul(text_x_f32, b.transpose(text_y_f32))  # [n_x, n_y]

    # Cost matrix
    cost = x_sq - 2 * xy + b.transpose(y_sq)  # [n_x, n_y]
    b.eval(cost)

    # DTW path finding (simplified - greedy with look-ahead)
    # Full DTW would require O(n_x * n_y) memory for the DP table
    # Use a simplified greedy approach for efficiency
    cost_list = [[float(cost[i, j]) for j in range(n_y)] for i in range(n_x)]

    # Greedy DTW path
    path_x: list[int] = []
    path_y: list[int] = []
    i, j = 0, 0

    while i < n_x and j < n_y:
        path_x.append(i)
        path_y.append(j)

        if i == n_x - 1:
            j += 1
        elif j == n_y - 1:
            i += 1
        else:
            # Choose minimum cost move
            cost_i = cost_list[i + 1][j] if i + 1 < n_x else float("inf")
            cost_j = cost_list[i][j + 1] if j + 1 < n_y else float("inf")
            cost_ij = cost_list[i + 1][j + 1] if (i + 1 < n_x and j + 1 < n_y) else float("inf")

            min_cost = min(cost_i, cost_j, cost_ij)
            if min_cost == cost_ij:
                i += 1
                j += 1
            elif min_cost == cost_i:
                i += 1
            else:
                j += 1

    if len(path_x) < 2:
        return 0.0

    # Extract aligned sequences
    path_x_arr = b.array(path_x, dtype="int32")
    path_y_arr = b.array(path_y, dtype="int32")

    aligned_x = b.take(text_x, path_x_arr, axis=0)
    aligned_y = b.take(text_y, path_y_arr, axis=0)
    b.eval(aligned_x, aligned_y)

    return compute_geodesic_cka(aligned_x, aligned_y, backend)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "compute_cka",
    "compute_geodesic_cka",
    "compute_linear_cka_gram",
    "compute_linear_cka_from_activations",
    "compute_cka_from_grams",
    "compute_cka_from_centered_grams",
    "rbf_gram_matrix",
    "rbf_gram_matrix_with_sigma",
    # Split CKA (shared vs. novel)
    "compute_cka_split",
    "SplitCKAResult",
    # Token-level CKA
    "compute_token_cka",
    "TokenCKAResult",
    # Types
    "HSICEstimator",
    "CKAResult",
    # Internal (needed by gram_aligner)
    "_center_gram_matrix",
    "_feature_sampling_correction",
]

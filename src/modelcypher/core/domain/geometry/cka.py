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
Centered Kernel Alignment (CKA).

HSIC-based implementation for measuring neural network representation similarity.

References:
    - Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
    - Cristianini et al. (2002) "On Kernel-Target Alignment"

Mathematical Foundation:
    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    Where HSIC (Hilbert-Schmidt Independence Criterion):
    HSIC(K, L) = (1/(n-1)^2) * tr(K_c @ L_c^T)

    K_c = H @ K @ H  (centered Gram matrix)
    H = I - (1/n) * 1 @ 1^T  (centering matrix)

Properties:
    - Rotation invariant: CKA(X @ R, Y) = CKA(X, Y) for orthogonal R
    - Scale invariant: CKA(alpha * X, Y) = CKA(X, Y)
    - Permutation invariant: CKA(P @ X, P @ Y) = CKA(X, Y)
    - Range: [0, 1]
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.cache import ComputationCache
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    regularization_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)

# Session-scoped cache for Gram matrices and centered Gram matrices
_cache = ComputationCache.shared()


class HSICEstimator(str, Enum):
    """HSIC estimator type for CKA computation.

    BIASED: Standard (1/(n-1)^2) * tr(K_c @ L_c). Works with n >= 2.
            This is the original estimator from Gretton et al. (2005).
            Has O(1/n) bias that inflates CKA in high P>>N settings.

    UNBIASED: Song et al. (2012) debiased estimator. Requires n >= 4.
              Corrects for finite-sample bias through combinatorial adjustments.
              Use this when features >> samples to avoid false-positive similarity.

    AUTO: Automatically select based on feature/sample ratio.
          Uses UNBIASED when max(features) > samples AND n >= 4.
          Falls back to BIASED otherwise.
    """

    BIASED = "biased"
    UNBIASED = "unbiased"
    AUTO = "auto"


@dataclass(frozen=True)
class CKAResult:
    """Result of CKA computation."""

    cka: float  # CKA similarity [0, 1]
    hsic_xy: float  # HSIC between X and Y
    hsic_xx: float  # HSIC of X with itself
    hsic_yy: float  # HSIC of Y with itself
    sample_count: int

    @property
    def is_valid(self) -> bool:
        """Check if result is valid (not NaN/Inf)."""
        return math.isfinite(self.cka) and math.isfinite(self.hsic_xy) and 0.0 <= self.cka <= 1.0


def _compute_pairwise_squared_distances(
    X: "Array",
    backend: "Backend",
) -> "Array":
    """Compute pairwise squared geodesic distances.

    Parameters
    ----------
    X : Array
        Data matrix [n_samples, n_features].
    backend : Backend
        Backend protocol implementation.

    Returns
    -------
    Array
        Distance matrix [n_samples, n_samples] (squared geodesic distances).
    """
    # Use geodesic distances that account for manifold curvature.
    # geodesic_distances handles all cases including n <= 2 (where geodesic
    # equals Euclidean by construction - the k-NN graph has only one edge).
    from .riemannian_utils import RiemannianGeometry

    rg = RiemannianGeometry(backend)
    result = rg.geodesic_distances(X)
    # Square the geodesic distances for RBF kernel
    return result.distances * result.distances


def _rbf_gram_matrix(
    X: "Array",
    backend: "Backend",
    sigma: float | None = None,
) -> "Array":
    """
    Compute RBF (Gaussian) Gram matrix with geodesic distances.

    K(x_i, x_j) = exp(-d_geo(x_i, x_j)^2 / (2 * sigma^2))

    Uses geodesic distance because high-dimensional data lives on curved
    manifolds. The RBF kernel with geodesic distances is the heat kernel
    on the manifold - the correct construction.

    Args:
        X: Data matrix [n_samples, n_features]
        backend: Backend protocol implementation
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        RBF Gram matrix [n_samples, n_samples]
    """
    distances = _compute_pairwise_squared_distances(X, backend)
    n = X.shape[0]

    if sigma is None:
        # Median heuristic: sigma = median of non-zero distances
        # Flatten upper triangle and find median
        flat_dist = backend.reshape(distances, (-1,))
        sorted_dist = backend.sort(flat_dist)
        # Skip zeros (diagonal elements)
        mid_idx = (n * n) // 2
        median_dist = float(backend.to_numpy(sorted_dist[mid_idx]))
        if median_dist > 0:
            sigma = math.sqrt(median_dist / 2)
        else:
            sigma = 1.0
        # Use precision-aware minimum to avoid zero sigma
        sigma = max(sigma, regularization_epsilon(backend, X))

    # K = exp(-D / (2 * sigma^2))
    neg_dist_scaled = -distances / (2 * sigma**2)
    gram = backend.exp(neg_dist_scaled)

    return gram


def _center_gram_matrix(
    gram: "Array",
    backend: "Backend",
    cache_key: str | None = None,
) -> "Array":
    """
    Center a Gram matrix using the centering matrix H.

    H = I - (1/n) * 1 @ 1^T
    K_c = H @ K @ H

    Efficient implementation without explicit H construction.

    Args:
        gram: Gram matrix to center
        backend: Backend for computation
        cache_key: Optional cache key for the input Gram matrix (enables caching)

    Returns:
        Centered Gram matrix
    """
    # Check cache if key provided
    if cache_key is not None:
        centered_key = _cache.make_centered_gram_key(cache_key)
        cached = _cache.get_centered_gram(centered_key)
        if cached is not None:
            return cached

    n = gram.shape[0]
    if n == 0:
        return gram

    start = time.perf_counter()

    # Column means
    col_mean = backend.mean(gram, axis=0, keepdims=True)
    # Row means
    row_mean = backend.mean(gram, axis=1, keepdims=True)
    # Grand mean
    grand_mean = backend.mean(gram)

    # H @ K @ H = K - col_mean - row_mean + grand_mean
    centered = gram - col_mean - row_mean + grand_mean

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Cache result if key provided
    if cache_key is not None:
        centered_key = _cache.make_centered_gram_key(cache_key)
        _cache.set_centered_gram(centered_key, centered, elapsed_ms)

    return centered


def _compute_hsic(
    gram_x: "Array",
    gram_y: "Array",
    backend: "Backend",
    centered_x: "Array | None" = None,
    centered_y: "Array | None" = None,
) -> float:
    """
    Compute HSIC between two Gram matrices.

    HSIC(K, L) = (1/(n-1)^2) * tr(K_c @ L_c^T)

    For symmetric Gram matrices, L_c^T = L_c, so:
    HSIC(K, L) = (1/(n-1)^2) * tr(K_c @ L_c)
    """
    n = gram_x.shape[0]
    if n <= 1:
        return 0.0

    # Center if not already centered
    if centered_x is None:
        centered_x = _center_gram_matrix(gram_x, backend)
    if centered_y is None:
        centered_y = _center_gram_matrix(gram_y, backend)

    # Normalize before multiplication to avoid overflow
    x_norm = backend.norm(centered_x)
    y_norm = backend.norm(centered_y)
    backend.eval(x_norm, y_norm)

    x_norm_val = float(backend.to_numpy(x_norm))
    y_norm_val = float(backend.to_numpy(y_norm))

    # Use precision-aware threshold for near-zero detection
    eps = division_epsilon(backend, centered_x)
    if x_norm_val < eps or y_norm_val < eps:
        return 0.0

    centered_x_normalized = centered_x / x_norm
    centered_y_normalized = centered_y / y_norm

    # Compute trace of normalized product using element-wise multiply and sum
    trace_product = backend.sum(centered_x_normalized * centered_y_normalized)
    backend.eval(trace_product)

    # Scale back
    trace_val = float(backend.to_numpy(trace_product)) * x_norm_val * y_norm_val

    # Normalize by (n-1)^2
    hsic = trace_val / ((n - 1) ** 2)

    if not math.isfinite(hsic):
        return 0.0

    return hsic


def _compute_hsic_unbiased(
    gram_x: "Array",
    gram_y: "Array",
    backend: "Backend",
) -> float:
    """
    Compute unbiased HSIC using Song et al. (2012) estimator.

    This estimator corrects for finite-sample bias that causes the standard
    HSIC to inflate in high P>>N (features >> samples) settings.

    Formula (Song et al. 2012, Equation 5):
        HSIC_u = 1/(n(n-3)) * [
            tr(K_tilde @ L_tilde) +
            (1^T K_tilde 1 * 1^T L_tilde 1) / ((n-1)(n-2)) -
            (2/(n-2)) * 1^T K_tilde L_tilde 1
        ]

    Where K_tilde, L_tilde are Gram matrices with diagonal set to 0.

    Requires n >= 4 (returns 0.0 for smaller samples).

    Args:
        gram_x: Gram matrix for X [n, n]
        gram_y: Gram matrix for Y [n, n]
        backend: Backend protocol implementation

    Returns:
        Unbiased HSIC estimate (non-negative)
    """
    n = gram_x.shape[0]
    if n < 4:
        # Unbiased estimator requires n >= 4 (denominator n*(n-3))
        return 0.0

    # Zero out diagonals: K_tilde[i,i] = 0
    diag_mask = 1.0 - backend.eye(n)
    k_tilde = gram_x * diag_mask
    l_tilde = gram_y * diag_mask
    backend.eval(k_tilde, l_tilde)

    # Term 1: tr(K_tilde @ L_tilde)
    # Efficient via element-wise product: tr(A @ B) = sum(A * B^T)
    # For symmetric matrices: sum(K_tilde * L_tilde)
    term1 = backend.sum(k_tilde * l_tilde)

    # Term 2: (1^T K_tilde 1 * 1^T L_tilde 1) / ((n-1)(n-2))
    # 1^T K_tilde 1 = sum of all elements
    k_sum = backend.sum(k_tilde)
    l_sum = backend.sum(l_tilde)
    term2 = (k_sum * l_sum) / ((n - 1) * (n - 2))

    # Term 3: (2/(n-2)) * 1^T K_tilde L_tilde 1
    # = (2/(n-2)) * sum of all elements of K_tilde @ L_tilde
    kl_product = backend.matmul(k_tilde, l_tilde)
    term3 = (2.0 / (n - 2)) * backend.sum(kl_product)

    # Combine: HSIC_u = (term1 + term2 - term3) / (n * (n-3))
    hsic = (term1 + term2 - term3) / (n * (n - 3))
    backend.eval(hsic)

    result = float(backend.to_numpy(hsic))

    # HSIC should be non-negative; clamp to avoid numerical issues
    return max(0.0, result) if math.isfinite(result) else 0.0


def _compute_hsic_dispatch(
    gram_x: "Array",
    gram_y: "Array",
    backend: "Backend",
    estimator: HSICEstimator,
    n_features_x: int | None = None,
    n_features_y: int | None = None,
    centered_x: "Array | None" = None,
    centered_y: "Array | None" = None,
) -> float:
    """
    Dispatch to appropriate HSIC estimator based on configuration.

    Args:
        gram_x, gram_y: Gram matrices
        backend: Backend protocol implementation
        estimator: Which HSIC estimator to use
        n_features_x, n_features_y: Feature dimensions (for AUTO mode)
        centered_x, centered_y: Pre-centered Gram matrices (for biased mode)

    Returns:
        HSIC estimate
    """
    n = gram_x.shape[0]

    if estimator == HSICEstimator.AUTO:
        # Use unbiased when max(features) > samples (high-dimensional)
        max_features = max(n_features_x or 0, n_features_y or 0)
        if max_features > n and n >= 4:
            estimator = HSICEstimator.UNBIASED
        else:
            estimator = HSICEstimator.BIASED

    if estimator == HSICEstimator.UNBIASED:
        if n < 4:
            # Fall back to biased for insufficient samples
            return _compute_hsic(gram_x, gram_y, backend, centered_x, centered_y)
        return _compute_hsic_unbiased(gram_x, gram_y, backend)
    else:
        return _compute_hsic(gram_x, gram_y, backend, centered_x, centered_y)


def compute_cka(
    activations_x: "Array",
    activations_y: "Array",
    backend: "Backend | None" = None,
    use_linear_kernel: bool = True,
    estimator: HSICEstimator = HSICEstimator.BIASED,
) -> CKAResult:
    """
    Compute CKA between two activation matrices.

    Uses session-scoped caching for Gram matrices and centered Gram matrices
    to avoid redundant computation when the same activations are used multiple
    times (e.g., comparing one source model against multiple targets).

    Args:
        activations_x: Activations from model X [n_samples, n_features_x]
        activations_y: Activations from model Y [n_samples, n_features_y]
        backend: Backend protocol implementation. If None, uses default.
        use_linear_kernel: If True, use linear kernel (X @ X^T).
                          If False, use RBF kernel.
        estimator: Which HSIC estimator to use (BIASED, UNBIASED, or AUTO).
                   BIASED is the default for backward compatibility.
                   Use UNBIASED or AUTO when features >> samples.

    Returns:
        CKAResult with CKA similarity and HSIC values
    """
    if backend is None:
        backend = get_default_backend()

    # Convert to backend arrays
    activations_x = backend.array(activations_x)
    activations_y = backend.array(activations_y)

    # Validate inputs
    if activations_x.shape[0] != activations_y.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {activations_x.shape[0]} vs {activations_y.shape[0]}"
        )

    n_samples = activations_x.shape[0]
    if n_samples < 2:
        return CKAResult(
            cka=0.0,
            hsic_xy=0.0,
            hsic_xx=0.0,
            hsic_yy=0.0,
            sample_count=n_samples,
        )

    kernel_type = "linear" if use_linear_kernel else "rbf"

    if use_linear_kernel:
        # Use cached Gram matrices for linear kernel
        gram_key_x = _cache.make_gram_key(activations_x, backend, kernel_type)
        gram_key_y = _cache.make_gram_key(activations_y, backend, kernel_type)

        gram_x = _cache.get_gram(gram_key_x)
        if gram_x is None:
            start = time.perf_counter()
            gram_x = backend.matmul(activations_x, backend.transpose(activations_x))
            backend.eval(gram_x)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _cache.set_gram(gram_key_x, gram_x, elapsed_ms)

        gram_y = _cache.get_gram(gram_key_y)
        if gram_y is None:
            start = time.perf_counter()
            gram_y = backend.matmul(activations_y, backend.transpose(activations_y))
            backend.eval(gram_y)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _cache.set_gram(gram_key_y, gram_y, elapsed_ms)
    else:
        # RBF kernel - compute directly (less frequently reused)
        gram_key_x = _cache.make_gram_key(activations_x, backend, kernel_type)
        gram_key_y = _cache.make_gram_key(activations_y, backend, kernel_type)
        gram_x = _rbf_gram_matrix(activations_x, backend)
        gram_y = _rbf_gram_matrix(activations_y, backend)

    # Center Gram matrices (with caching) - needed for biased estimator
    centered_x = _center_gram_matrix(gram_x, backend, gram_key_x)
    centered_y = _center_gram_matrix(gram_y, backend, gram_key_y)

    # Get feature dimensions for AUTO mode
    n_features_x = activations_x.shape[1] if len(activations_x.shape) > 1 else 1
    n_features_y = activations_y.shape[1] if len(activations_y.shape) > 1 else 1

    # Compute HSIC values using selected estimator
    hsic_xy = _compute_hsic_dispatch(
        gram_x, gram_y, backend, estimator,
        n_features_x, n_features_y, centered_x, centered_y
    )
    hsic_xx = _compute_hsic_dispatch(
        gram_x, gram_x, backend, estimator,
        n_features_x, n_features_x, centered_x, centered_x
    )
    hsic_yy = _compute_hsic_dispatch(
        gram_y, gram_y, backend, estimator,
        n_features_y, n_features_y, centered_y, centered_y
    )

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denominator = math.sqrt(hsic_xx * hsic_yy)

    # Use precision-aware threshold for near-zero detection
    eps = division_epsilon(backend, activations_x)
    if denominator < eps:
        cka = 0.0
    else:
        cka = hsic_xy / denominator

    # Clamp to [0, 1] (can exceed due to numerical issues)
    cka = max(0.0, min(1.0, cka))

    return CKAResult(
        cka=cka,
        hsic_xy=hsic_xy,
        hsic_xx=hsic_xx,
        hsic_yy=hsic_yy,
        sample_count=n_samples,
    )


def compute_cka_matrix(
    source_activations: dict[str, "Array"],
    target_activations: dict[str, "Array"],
    backend: "Backend | None" = None,
) -> tuple["Array", list[str], list[str]]:
    """
    Compute pairwise CKA matrix between all activation sets.

    Args:
        source_activations: Dict mapping probe_id -> activations [n_samples, hidden_dim]
        target_activations: Dict mapping probe_id -> activations [n_samples, hidden_dim]
        backend: Backend protocol implementation. If None, uses default.

    Returns:
        Tuple of (CKA matrix, source probe IDs, target probe IDs)
    """
    if backend is None:
        backend = get_default_backend()

    source_ids = sorted(source_activations.keys())
    target_ids = sorted(target_activations.keys())

    if not source_ids or not target_ids:
        return backend.zeros((0, 0)), [], []

    matrix = backend.zeros((len(source_ids), len(target_ids)))
    matrix_list: list[list[float]] = []

    for i, s_id in enumerate(source_ids):
        row: list[float] = []
        for j, t_id in enumerate(target_ids):
            s_act = source_activations[s_id]
            t_act = target_activations[t_id]

            # Ensure same sample count
            min_samples = min(s_act.shape[0], t_act.shape[0])
            if min_samples < 2:
                row.append(0.0)
                continue

            result = compute_cka(s_act[:min_samples], t_act[:min_samples], backend)
            row.append(result.cka)
        matrix_list.append(row)

    # Convert to backend array
    matrix = backend.array(matrix_list)
    return matrix, source_ids, target_ids


def compute_layer_cka(
    source_weights: "Array",
    target_weights: "Array",
    backend: "Backend | None" = None,
) -> CKAResult:
    """
    Compute CKA between weight matrices directly.

    Treats weight rows as "samples" and columns as "features".
    This measures how similarly the two weight matrices structure
    the same input space.

    Args:
        source_weights: Source weight matrix [out_dim, in_dim]
        target_weights: Target weight matrix [out_dim, in_dim]
        backend: Backend protocol implementation. If None, uses default.

    Returns:
        CKAResult
    """
    if backend is None:
        backend = get_default_backend()

    # If shapes differ, try to align
    if source_weights.shape != target_weights.shape:
        min_out = min(source_weights.shape[0], target_weights.shape[0])
        min_in = min(source_weights.shape[1], target_weights.shape[1])
        source_weights = source_weights[:min_out, :min_in]
        target_weights = target_weights[:min_out, :min_in]

    return compute_cka(source_weights, target_weights, backend)


def compute_cka_backend(
    x: "Array",
    y: "Array",
    backend: "Backend",
) -> float:
    """
    Compute linear CKA using the Backend protocol for MLX/JAX/CUDA.

    This is the canonical Backend-aware implementation. Use this in hot paths
    where tensor operations should stay on-device (GPU/Metal).

    Uses session-scoped caching for Gram matrices to avoid redundant computation.

    Mathematical steps:
        1. Compute Gram matrices: K = X @ X^T, L = Y @ Y^T
        2. Compute HSIC via Frobenius inner product: HSIC(K,L) = sum(K * L)
        3. CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))

    Note: Uses uncentered HSIC (Frobenius inner product) which is equivalent
    to centered HSIC for CKA ratio computation. See Kornblith et al. (2019).

    Args:
        x: Activation matrix [n_samples, n_features_x]
        y: Activation matrix [n_samples, n_features_y]
        backend: Backend protocol implementation (MLX, JAX, CUDA)

    Returns:
        CKA similarity value in [0, 1]
    """
    # Use cached Gram matrices
    gram_x = _cache.get_or_compute_gram(x, backend, kernel_type="linear")
    gram_y = _cache.get_or_compute_gram(y, backend, kernel_type="linear")

    # HSIC via Frobenius inner product: sum(K * L)
    hsic_xy_arr = backend.sum(gram_x * gram_y)
    hsic_xx_arr = backend.sum(gram_x * gram_x)
    hsic_yy_arr = backend.sum(gram_y * gram_y)

    # Force evaluation for lazy backends (MLX)
    backend.eval(hsic_xy_arr, hsic_xx_arr, hsic_yy_arr)

    # Convert to Python floats
    hsic_xy = float(backend.to_numpy(hsic_xy_arr).item())
    hsic_xx = float(backend.to_numpy(hsic_xx_arr).item())
    hsic_yy = float(backend.to_numpy(hsic_yy_arr).item())

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denom = math.sqrt(hsic_xx * hsic_yy)
    # Use precision-aware threshold for near-zero detection
    eps = division_epsilon(backend, x)
    if denom < eps:
        return 0.0

    cka = hsic_xy / denom
    return max(0.0, min(1.0, cka))


def compute_cka_from_lists(
    x: list[list[float]],
    y: list[list[float]],
    backend: "Backend | None" = None,
) -> float:
    """
    Compute linear CKA from Python lists.

    Convenience wrapper that converts lists to backend arrays and calls compute_cka.
    Use this when working with list-based APIs.

    Args:
        x: Activation matrix as nested lists [n_samples][n_features_x]
        y: Activation matrix as nested lists [n_samples][n_features_y]
        backend: Backend protocol implementation. If None, uses default.

    Returns:
        CKA similarity value in [0, 1]
    """
    if backend is None:
        backend = get_default_backend()

    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x_arr = backend.array(x[:n])
    y_arr = backend.array(y[:n])

    result = compute_cka(x_arr, y_arr, backend, use_linear_kernel=True)
    return result.cka if result.is_valid else 0.0


def compute_cka_from_grams(
    gram_a: list[float] | "Array",
    gram_b: list[float] | "Array",
    n: int | None = None,
    backend: "Backend | None" = None,
    estimator: HSICEstimator = HSICEstimator.BIASED,
) -> float:
    """
    Compute CKA from pre-computed Gram matrices.

    This is the canonical implementation for working with pre-computed Gram matrices.
    Supports both flattened lists (n*n elements) and 2D arrays.

    IMPORTANT: This is THE method for cross-dimensional comparison.
    Gram matrices are [n,n] regardless of original feature dimension,
    making this work across ANY dimensions.

    Args:
        gram_a: Gram matrix for representation A. Either flattened [n*n] or [n, n].
        gram_b: Gram matrix for representation B. Either flattened [n*n] or [n, n].
        n: Matrix dimension (required if gram matrices are flattened lists).
        backend: Backend protocol implementation. If None, uses default.
        estimator: Which HSIC estimator to use (BIASED, UNBIASED, or AUTO).

    Returns:
        CKA similarity value in [0, 1]
    """
    if backend is None:
        backend = get_default_backend()

    # Convert to backend arrays
    if isinstance(gram_a, list):
        arr_a = backend.array(gram_a)
    else:
        arr_a = gram_a

    if isinstance(gram_b, list):
        arr_b = backend.array(gram_b)
    else:
        arr_b = gram_b

    # Handle flattened inputs
    if len(arr_a.shape) == 1:
        if n is None:
            n = int(math.sqrt(arr_a.shape[0]))
            if n * n != arr_a.shape[0]:
                return 0.0
        arr_a = backend.reshape(arr_a, (n, n))

    if len(arr_b.shape) == 1:
        if n is None:
            n = int(math.sqrt(arr_b.shape[0]))
            if n * n != arr_b.shape[0]:
                return 0.0
        arr_b = backend.reshape(arr_b, (n, n))

    # Validate dimensions
    if arr_a.shape != arr_b.shape or arr_a.shape[0] != arr_a.shape[1]:
        return 0.0

    n = arr_a.shape[0]
    if n <= 1:
        return 0.0

    # Center gram matrices (needed for biased estimator)
    centered_a = _center_gram_matrix(arr_a, backend)
    centered_b = _center_gram_matrix(arr_b, backend)

    # Compute HSIC values using selected estimator
    # Note: For Gram matrices, we don't know original feature dims, so AUTO
    # will use the same as BIASED unless explicitly set to UNBIASED
    hsic_ab = _compute_hsic_dispatch(arr_a, arr_b, backend, estimator, None, None, centered_a, centered_b)
    hsic_aa = _compute_hsic_dispatch(arr_a, arr_a, backend, estimator, None, None, centered_a, centered_a)
    hsic_bb = _compute_hsic_dispatch(arr_b, arr_b, backend, estimator, None, None, centered_b, centered_b)

    # CKA = HSIC(A,B) / sqrt(HSIC(A,A) * HSIC(B,B))
    denom = math.sqrt(hsic_aa * hsic_bb)
    # Use precision-aware threshold for near-zero detection
    eps = division_epsilon(backend, arr_a)
    if denom < eps:
        return 0.0

    cka = hsic_ab / denom
    return max(0.0, min(1.0, cka))


def ensemble_similarity(
    jaccard: float,
    cka: float,
    cosine: float,
    jaccard_weight: float = 0.5,
    cka_weight: float = 0.5,
) -> float:
    """
    Compute ensemble similarity score combining multiple metrics.

    Formula:
        score = w_jaccard * jaccard + w_cka * CKA + cosine_gate
        cosine_gate = max(0, cosine)  # Avoid anti-correlation penalty

    Args:
        jaccard: Weighted Jaccard similarity [0, 1]
        cka: CKA similarity [0, 1]
        cosine: Cosine similarity [-1, 1]
        jaccard_weight: Weight for Jaccard (default 0.5)
        cka_weight: Weight for CKA (default 0.5)

    Returns:
        Ensemble similarity score
    """
    # Cosine gate: only add positive contribution
    cosine_gate = max(0.0, cosine)

    # Weighted combination
    score = jaccard_weight * jaccard + cka_weight * cka + cosine_gate

    # Normalize to [0, 1] approximately
    # Max possible is jaccard_weight + cka_weight + 1.0 = 2.0
    # But we typically want a score in [0, 1]
    # Scale by expected max
    max_score = jaccard_weight + cka_weight + 1.0

    return score / max_score


class CKAComputer:
    """Configurable CKA computation with HSIC estimator selection.

    Use this class when you need:
    - Consistent estimator selection across multiple computations
    - Object-oriented interface for CKA operations
    - Integration with services that expect a stateful computer

    Example:
        >>> cka = CKAComputer(backend, estimator=HSICEstimator.AUTO)
        >>> similarity = cka.linear_cka(activations_x, activations_y)
        >>> print(f"CKA: {similarity:.4f}")

    For most use cases, the module-level functions (compute_cka, compute_cka_backend)
    are sufficient. This class is primarily for service integration patterns.
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        estimator: HSICEstimator = HSICEstimator.BIASED,
    ) -> None:
        """Initialize CKA computer with configuration.

        Args:
            backend: Backend protocol implementation. If None, uses default.
            estimator: Which HSIC estimator to use. BIASED is default for
                      backward compatibility; use UNBIASED or AUTO for
                      high-dimensional (P>>N) settings.
        """
        self._backend = backend or get_default_backend()
        self._estimator = estimator

    @property
    def backend(self) -> "Backend":
        """Get the backend protocol implementation."""
        return self._backend

    @property
    def estimator(self) -> HSICEstimator:
        """Get the HSIC estimator type."""
        return self._estimator

    def linear_cka(self, x: "Array", y: "Array") -> float:
        """Compute linear kernel CKA between activation matrices.

        Args:
            x: Activation matrix [n_samples, n_features_x]
            y: Activation matrix [n_samples, n_features_y]

        Returns:
            CKA similarity value in [0, 1]
        """
        result = compute_cka(
            x, y, self._backend,
            use_linear_kernel=True,
            estimator=self._estimator,
        )
        return result.cka

    def rbf_cka(self, x: "Array", y: "Array") -> float:
        """Compute RBF kernel CKA between activation matrices.

        Args:
            x: Activation matrix [n_samples, n_features_x]
            y: Activation matrix [n_samples, n_features_y]

        Returns:
            CKA similarity value in [0, 1]
        """
        result = compute_cka(
            x, y, self._backend,
            use_linear_kernel=False,
            estimator=self._estimator,
        )
        return result.cka

    def full(self, x: "Array", y: "Array", use_linear: bool = True) -> CKAResult:
        """Full CKA computation returning CKAResult with all HSIC values.

        Args:
            x: Activation matrix [n_samples, n_features_x]
            y: Activation matrix [n_samples, n_features_y]
            use_linear: If True, use linear kernel; if False, use RBF kernel.

        Returns:
            CKAResult with CKA similarity and HSIC values
        """
        return compute_cka(
            x, y, self._backend,
            use_linear_kernel=use_linear,
            estimator=self._estimator,
        )

    def from_grams(self, gram_a: "Array", gram_b: "Array") -> float:
        """Compute CKA from pre-computed Gram matrices.

        This is useful for cross-dimensional comparison where you have
        Gram matrices but not the original activations.

        Args:
            gram_a: Gram matrix for representation A [n, n]
            gram_b: Gram matrix for representation B [n, n]

        Returns:
            CKA similarity value in [0, 1]
        """
        return compute_cka_from_grams(
            gram_a, gram_b,
            backend=self._backend,
            estimator=self._estimator,
        )

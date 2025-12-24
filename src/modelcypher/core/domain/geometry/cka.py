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
from dataclasses import dataclass


import numpy as np

logger = logging.getLogger(__name__)


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
        return (
            np.isfinite(self.cka)
            and np.isfinite(self.hsic_xy)
            and 0.0 <= self.cka <= 1.0
        )


def _compute_pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances.

    D[i,j] = ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T @ x_j

    Args:
        X: Data matrix [n_samples, n_features]

    Returns:
        Distance matrix [n_samples, n_samples]
    """
    # Compute squared norms for each sample
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # [n, 1]

    # D[i,j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T @ x_j
    distances = sq_norms + sq_norms.T - 2 * (X @ X.T)

    # Ensure non-negative (numerical issues can cause tiny negatives)
    distances = np.maximum(distances, 0.0)

    return distances


def _rbf_gram_matrix(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    Compute RBF (Gaussian) Gram matrix.

    K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))

    Args:
        X: Data matrix [n_samples, n_features]
        sigma: RBF bandwidth. If None, uses median heuristic.

    Returns:
        RBF Gram matrix [n_samples, n_samples]
    """
    distances = _compute_pairwise_squared_distances(X.astype(np.float64))

    if sigma is None:
        # Median heuristic: sigma = median of non-zero distances
        # Extract upper triangle (excluding diagonal)
        upper_tri = distances[np.triu_indices_from(distances, k=1)]
        if len(upper_tri) > 0 and np.any(upper_tri > 0):
            median_dist = np.median(upper_tri[upper_tri > 0])
            sigma = np.sqrt(median_dist / 2)  # Convert squared dist to sigma
            sigma = max(sigma, 1e-6)  # Avoid zero sigma
        else:
            sigma = 1.0  # Default if all distances are zero

    # K = exp(-D / (2 * sigma^2))
    gram = np.exp(-distances / (2 * sigma ** 2))

    return gram.astype(np.float32)


def _center_gram_matrix(gram: np.ndarray) -> np.ndarray:
    """
    Center a Gram matrix using the centering matrix H.

    H = I - (1/n) * 1 @ 1^T
    K_c = H @ K @ H

    Efficient implementation without explicit H construction.
    """
    n = gram.shape[0]
    if n == 0:
        return gram

    # Column means
    col_mean = np.mean(gram, axis=0, keepdims=True)
    # Row means
    row_mean = np.mean(gram, axis=1, keepdims=True)
    # Grand mean
    grand_mean = np.mean(gram)

    # H @ K @ H = K - col_mean - row_mean + grand_mean
    centered = gram - col_mean - row_mean + grand_mean

    return centered


def _compute_hsic(
    gram_x: np.ndarray,
    gram_y: np.ndarray,
    centered_x: np.ndarray | None = None,
    centered_y: np.ndarray | None = None,
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
        centered_x = _center_gram_matrix(gram_x)
    if centered_y is None:
        centered_y = _center_gram_matrix(gram_y)

    # Use float64 to avoid overflow
    centered_x = centered_x.astype(np.float64)
    centered_y = centered_y.astype(np.float64)

    # Normalize before multiplication to avoid overflow
    x_norm = np.linalg.norm(centered_x)
    y_norm = np.linalg.norm(centered_y)
    if x_norm < 1e-10 or y_norm < 1e-10:
        return 0.0

    centered_x_normalized = centered_x / x_norm
    centered_y_normalized = centered_y / y_norm

    # Compute trace of normalized product
    trace_product = np.sum(centered_x_normalized * centered_y_normalized)

    # Scale back
    trace_product = trace_product * x_norm * y_norm

    # Normalize by (n-1)^2
    hsic = trace_product / ((n - 1) ** 2)

    if not np.isfinite(hsic):
        return 0.0

    return float(hsic)


def compute_cka(
    activations_x: np.ndarray,
    activations_y: np.ndarray,
    use_linear_kernel: bool = True,
) -> CKAResult:
    """
    Compute CKA between two activation matrices.

    Args:
        activations_x: Activations from model X [n_samples, n_features_x]
        activations_y: Activations from model Y [n_samples, n_features_y]
        use_linear_kernel: If True, use linear kernel (X @ X^T).
                          If False, use RBF kernel.

    Returns:
        CKAResult with CKA similarity and HSIC values
    """
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

    # Ensure float32 for numerical stability
    x = activations_x.astype(np.float32)
    y = activations_y.astype(np.float32)

    if use_linear_kernel:
        # Linear kernel: K = X @ X^T
        gram_x = x @ x.T
        gram_y = y @ y.T
    else:
        # RBF kernel with median heuristic for bandwidth
        gram_x = _rbf_gram_matrix(x)
        gram_y = _rbf_gram_matrix(y)

    # Center Gram matrices
    centered_x = _center_gram_matrix(gram_x)
    centered_y = _center_gram_matrix(gram_y)

    # Compute HSIC values
    hsic_xy = _compute_hsic(gram_x, gram_y, centered_x, centered_y)
    hsic_xx = _compute_hsic(gram_x, gram_x, centered_x, centered_x)
    hsic_yy = _compute_hsic(gram_y, gram_y, centered_y, centered_y)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denominator = np.sqrt(hsic_xx * hsic_yy)

    if denominator < 1e-10:
        cka = 0.0
    else:
        cka = hsic_xy / denominator

    # Clamp to [0, 1] (can exceed due to numerical issues)
    cka = float(np.clip(cka, 0.0, 1.0))

    return CKAResult(
        cka=cka,
        hsic_xy=float(hsic_xy),
        hsic_xx=float(hsic_xx),
        hsic_yy=float(hsic_yy),
        sample_count=n_samples,
    )


def compute_cka_matrix(
    source_activations: dict[str, np.ndarray],
    target_activations: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Compute pairwise CKA matrix between all activation sets.

    Args:
        source_activations: Dict mapping probe_id -> activations [n_samples, hidden_dim]
        target_activations: Dict mapping probe_id -> activations [n_samples, hidden_dim]

    Returns:
        Tuple of (CKA matrix, source probe IDs, target probe IDs)
    """
    source_ids = sorted(source_activations.keys())
    target_ids = sorted(target_activations.keys())

    if not source_ids or not target_ids:
        return np.array([]), [], []

    matrix = np.zeros((len(source_ids), len(target_ids)), dtype=np.float32)

    for i, s_id in enumerate(source_ids):
        for j, t_id in enumerate(target_ids):
            s_act = source_activations[s_id]
            t_act = target_activations[t_id]

            # Ensure same sample count
            min_samples = min(s_act.shape[0], t_act.shape[0])
            if min_samples < 2:
                continue

            result = compute_cka(s_act[:min_samples], t_act[:min_samples])
            matrix[i, j] = result.cka

    return matrix, source_ids, target_ids


def compute_layer_cka(
    source_weights: np.ndarray,
    target_weights: np.ndarray,
) -> CKAResult:
    """
    Compute CKA between weight matrices directly.

    Treats weight rows as "samples" and columns as "features".
    This measures how similarly the two weight matrices structure
    the same input space.

    Args:
        source_weights: Source weight matrix [out_dim, in_dim]
        target_weights: Target weight matrix [out_dim, in_dim]

    Returns:
        CKAResult
    """
    # Weights are [out_dim, in_dim]
    # Treat out_dim as samples, in_dim as features
    # This measures if the same input dimensions are used similarly

    if source_weights.shape != target_weights.shape:
        # If shapes differ, try to align
        min_out = min(source_weights.shape[0], target_weights.shape[0])
        min_in = min(source_weights.shape[1], target_weights.shape[1])
        source_weights = source_weights[:min_out, :min_in]
        target_weights = target_weights[:min_out, :min_in]

    return compute_cka(source_weights, target_weights)


def compute_cka_backend(
    x: "Array",
    y: "Array",
    backend: "Backend",
) -> float:
    """
    Compute linear CKA using the Backend protocol for MLX/JAX/CUDA.

    This is the canonical Backend-aware implementation. Use this in hot paths
    where tensor operations should stay on-device (GPU/Metal).

    Mathematical steps:
        1. Compute Gram matrices: K = X @ X^T, L = Y @ Y^T
        2. Compute HSIC via Frobenius inner product: HSIC(K,L) = sum(K * L)
        3. CKA = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))

    Note: Uses uncentered HSIC (Frobenius inner product) which is equivalent
    to centered HSIC for CKA ratio computation. See Kornblith et al. (2019).

    Args:
        x: Activation matrix [n_samples, n_features_x]
        y: Activation matrix [n_samples, n_features_y]
        backend: Backend protocol implementation (MLX, JAX, CUDA, or NumPy)

    Returns:
        CKA similarity value in [0, 1]
    """
    from modelcypher.ports.backend import Backend as BackendProtocol

    # Gram matrices: K = X @ X^T, L = Y @ Y^T
    gram_x = backend.matmul(x, backend.transpose(x))
    gram_y = backend.matmul(y, backend.transpose(y))

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
    import math
    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    cka = hsic_xy / denom
    return max(0.0, min(1.0, cka))


def compute_cka_from_lists(
    x: list[list[float]],
    y: list[list[float]],
) -> float:
    """
    Compute linear CKA from Python lists.

    Convenience wrapper that converts lists to numpy arrays and calls compute_cka.
    Use this when working with list-based APIs.

    Args:
        x: Activation matrix as nested lists [n_samples][n_features_x]
        y: Activation matrix as nested lists [n_samples][n_features_y]

    Returns:
        CKA similarity value in [0, 1]
    """
    n = min(len(x), len(y))
    if n < 2:
        return 0.0

    x_arr = np.array(x[:n], dtype=np.float32)
    y_arr = np.array(y[:n], dtype=np.float32)

    result = compute_cka(x_arr, y_arr, use_linear_kernel=True)
    return result.cka if result.is_valid else 0.0


def ensemble_similarity(
    jaccard: float,
    cka: float,
    cosine: float,
    jaccard_weight: float = 0.6,
    cka_weight: float = 0.4,
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
        jaccard_weight: Weight for Jaccard (default 0.6)
        cka_weight: Weight for CKA (default 0.4)

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

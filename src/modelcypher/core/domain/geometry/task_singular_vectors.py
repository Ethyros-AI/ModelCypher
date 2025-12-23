"""
Task Singular Vectors for Model Merging.

Decomposes task vectors (W_source - W_target) via SVD to identify principal
skill directions vs structural noise. Applies different alpha values to
high-rank (skill) vs low-rank (structure) components.

Reference:
- "Task Singular Vectors: Reducing Task Interference in Model Merging" (CVPR 2025)
- TrainingCypher/UnifiedManifoldMerger.swift (lines 1180-1230)

Mathematical Foundation
-----------------------
Given task vector Δ = W_source - W_target:

1. SVD decomposition: Δ = U @ diag(S) @ Vt

2. Partition singular values:
   - High-rank (top k): Principal skill directions, S[:k]
   - Low-rank (remaining): Structural/noise, S[k:]

3. Blend with different alpha:
   - High-rank: Use lower alpha (trust source skills more)
   - Low-rank: Use higher alpha (trust target structure more)

4. Reconstruct: W_merged = W_target + α_high * Δ_high + α_low * Δ_low

This reduces task interference by preserving skill-specific singular
components while maintaining target model stability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskVectorDecomposition:
    """SVD decomposition of a task vector."""

    # Left singular vectors [out_dim, k]
    U: np.ndarray

    # Singular values [k]
    S: np.ndarray

    # Right singular vectors (transposed) [k, in_dim]
    Vt: np.ndarray

    # Fraction of variance captured by kept components
    variance_captured: float

    # Effective rank (number of significant singular values)
    effective_rank: int

    # Original shape of the weight matrix
    original_shape: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        """Check if decomposition is valid for blending."""
        return len(self.S) > 0 and self.variance_captured > 0

    def reconstruct(self, alpha: float = 1.0) -> np.ndarray:
        """Reconstruct the task vector with optional scaling."""
        if not self.is_valid:
            return np.zeros(self.original_shape, dtype=np.float32)

        delta = self.U @ np.diag(self.S) @ self.Vt
        return (alpha * delta).astype(np.float32)


@dataclass(frozen=True)
class SVDBlendConfig:
    """Configuration for SVD-aware blending."""

    # Fraction of singular values to keep as "high-rank"
    rank_ratio: float = 0.1

    # Alternative: keep components capturing this fraction of variance
    variance_threshold: float = 0.9

    # Use rank_ratio if True, else use variance_threshold
    use_rank_ratio: bool = True

    # Alpha for high-rank components (skills) - trust source more
    high_rank_alpha: float = 0.3

    # Alpha for low-rank components (structure) - trust target more
    low_rank_alpha: float = 0.7

    # Minimum singular value ratio to consider (filters noise)
    min_sv_ratio: float = 1e-4

    # Epsilon for numerical stability
    epsilon: float = 1e-8

    @classmethod
    def default(cls) -> SVDBlendConfig:
        """Default configuration."""
        return cls()

    @classmethod
    def skill_preserving(cls) -> SVDBlendConfig:
        """Preserve skills aggressively."""
        return cls(
            rank_ratio=0.2,
            high_rank_alpha=0.2,
            low_rank_alpha=0.8,
        )

    @classmethod
    def structure_preserving(cls) -> SVDBlendConfig:
        """Preserve structure aggressively."""
        return cls(
            rank_ratio=0.05,
            high_rank_alpha=0.4,
            low_rank_alpha=0.6,
        )


def decompose_task_vector(
    source_weight: np.ndarray,
    target_weight: np.ndarray,
    config: Optional[SVDBlendConfig] = None,
) -> TaskVectorDecomposition:
    """
    Decompose task vector Δ = W_source - W_target via SVD.

    Args:
        source_weight: Source model weight matrix
        target_weight: Target model weight matrix (same shape)
        config: SVD blending configuration

    Returns:
        TaskVectorDecomposition with U, S, Vt and metadata
    """
    if config is None:
        config = SVDBlendConfig.default()

    original_shape = source_weight.shape

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, treat as single "singular value" = norm of difference
        delta = source_weight - target_weight
        delta_norm = float(np.linalg.norm(delta))

        if delta_norm < config.epsilon:
            return TaskVectorDecomposition(
                U=np.zeros((len(delta), 1), dtype=np.float32),
                S=np.array([0.0], dtype=np.float32),
                Vt=np.zeros((1, 1), dtype=np.float32),
                variance_captured=0.0,
                effective_rank=0,
                original_shape=original_shape,
            )

        # Normalize to create unit "singular vector"
        u = (delta / delta_norm).reshape(-1, 1)

        return TaskVectorDecomposition(
            U=u.astype(np.float32),
            S=np.array([delta_norm], dtype=np.float32),
            Vt=np.array([[1.0]], dtype=np.float32),
            variance_captured=1.0,
            effective_rank=1,
            original_shape=original_shape,
        )

    # Compute task vector
    source_np = np.asarray(source_weight, dtype=np.float32)
    target_np = np.asarray(target_weight, dtype=np.float32)
    delta = source_np - target_np

    # Full SVD
    try:
        U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed, returning zero decomposition")
        return TaskVectorDecomposition(
            U=np.zeros((delta.shape[0], 1), dtype=np.float32),
            S=np.array([0.0], dtype=np.float32),
            Vt=np.zeros((1, delta.shape[1]), dtype=np.float32),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Filter noise: keep only significant singular values
    if len(S) > 0:
        sv_threshold = S[0] * config.min_sv_ratio
        significant_mask = S >= sv_threshold
        num_significant = int(np.sum(significant_mask))
    else:
        num_significant = 0

    if num_significant == 0:
        return TaskVectorDecomposition(
            U=U[:, :1],
            S=np.array([0.0], dtype=np.float32),
            Vt=Vt[:1, :],
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Compute effective rank
    total_variance = float(np.sum(S**2))
    if total_variance < config.epsilon:
        effective_rank = 0
    else:
        cumulative_variance = np.cumsum(S**2) / total_variance
        effective_rank = int(np.searchsorted(cumulative_variance, 0.99)) + 1

    # Determine how many components to keep
    if config.use_rank_ratio:
        k = max(1, int(len(S) * config.rank_ratio))
    else:
        # Keep enough to capture variance_threshold
        cumulative = np.cumsum(S**2) / max(total_variance, config.epsilon)
        k = int(np.searchsorted(cumulative, config.variance_threshold)) + 1

    k = min(k, num_significant, len(S))

    # Compute variance captured by top-k
    if total_variance > config.epsilon:
        variance_captured = float(np.sum(S[:k] ** 2) / total_variance)
    else:
        variance_captured = 0.0

    return TaskVectorDecomposition(
        U=U.astype(np.float32),
        S=S.astype(np.float32),
        Vt=Vt.astype(np.float32),
        variance_captured=variance_captured,
        effective_rank=effective_rank,
        original_shape=original_shape,
    )


def blend_with_svd_awareness(
    source_weight: np.ndarray,
    target_weight: np.ndarray,
    base_alpha: float,
    config: Optional[SVDBlendConfig] = None,
) -> np.ndarray:
    """
    Blend weights using SVD-aware alpha for different singular components.

    High-rank components (skills): Use lower alpha to preserve source skills
    Low-rank components (structure): Use higher alpha to maintain target stability

    Args:
        source_weight: Source model weight matrix
        target_weight: Target model weight matrix (same shape)
        base_alpha: Base alpha value (used to modulate high/low rank alphas)
        config: SVD blending configuration

    Returns:
        Blended weight matrix
    """
    if config is None:
        config = SVDBlendConfig.default()

    # Handle 1D weights simply
    if source_weight.ndim == 1:
        # For 1D, use weighted average with base_alpha
        return (
            (1.0 - base_alpha) * source_weight + base_alpha * target_weight
        ).astype(np.float32)

    source_np = np.asarray(source_weight, dtype=np.float32)
    target_np = np.asarray(target_weight, dtype=np.float32)

    # Decompose task vector
    decomp = decompose_task_vector(source_np, target_np, config)

    if not decomp.is_valid:
        # Fallback to simple linear blend
        return (
            (1.0 - base_alpha) * source_np + base_alpha * target_np
        ).astype(np.float32)

    # Determine cutoff for high vs low rank
    if config.use_rank_ratio:
        k = max(1, int(len(decomp.S) * config.rank_ratio))
    else:
        total_var = float(np.sum(decomp.S**2))
        if total_var > config.epsilon:
            cumulative = np.cumsum(decomp.S**2) / total_var
            k = int(np.searchsorted(cumulative, config.variance_threshold)) + 1
        else:
            k = 1

    k = min(k, len(decomp.S))

    # Modulate alphas based on base_alpha
    # When base_alpha is 0.5, use configured high/low alphas
    # When base_alpha deviates, adjust proportionally
    alpha_scale = 2.0 * base_alpha  # 0 when base=0, 1 when base=0.5, 2 when base=1

    # High-rank alpha: lower → trust source skills
    # Interpolate between 0 and configured high_rank_alpha
    high_alpha = config.high_rank_alpha * alpha_scale
    high_alpha = max(0.0, min(1.0, high_alpha))

    # Low-rank alpha: higher → trust target structure
    # Interpolate between configured low_rank_alpha and 1.0
    low_alpha = config.low_rank_alpha + (1.0 - config.low_rank_alpha) * (alpha_scale - 1.0)
    low_alpha = max(0.0, min(1.0, low_alpha))

    # Reconstruct high-rank component (skills)
    U_high = decomp.U[:, :k]
    S_high = decomp.S[:k]
    Vt_high = decomp.Vt[:k, :]
    delta_high = U_high @ np.diag(S_high) @ Vt_high

    # Reconstruct low-rank component (structure)
    if k < len(decomp.S):
        U_low = decomp.U[:, k:]
        S_low = decomp.S[k:]
        Vt_low = decomp.Vt[k:, :]
        delta_low = U_low @ np.diag(S_low) @ Vt_low
    else:
        delta_low = np.zeros_like(delta_high)

    # Blend: W_merged = W_target + (1 - α_high) * Δ_high + (1 - α_low) * Δ_low
    # Note: α closer to 1 → trust target more → add less of delta
    merged = (
        target_np
        + (1.0 - high_alpha) * delta_high
        + (1.0 - low_alpha) * delta_low
    )

    return merged.astype(np.float32)


def compute_task_vector_similarity(
    decomp1: TaskVectorDecomposition,
    decomp2: TaskVectorDecomposition,
) -> float:
    """
    Compute similarity between two task vectors based on their singular subspaces.

    Uses principal angles between the column spaces of U matrices.

    Args:
        decomp1: First task vector decomposition
        decomp2: Second task vector decomposition

    Returns:
        Similarity score in [0, 1], higher = more similar
    """
    if not decomp1.is_valid or not decomp2.is_valid:
        return 0.0

    # Use top-k singular vectors for comparison
    k1 = max(1, int(len(decomp1.S) * 0.1))
    k2 = max(1, int(len(decomp2.S) * 0.1))
    k = min(k1, k2)

    U1 = decomp1.U[:, :k]
    U2 = decomp2.U[:, :k]

    # Compute principal angles via SVD of U1.T @ U2
    # Cosines of principal angles are singular values of this product
    try:
        _, cosines, _ = np.linalg.svd(U1.T @ U2, full_matrices=False)
        # Mean cosine as similarity
        similarity = float(np.mean(np.clip(cosines, 0, 1)))
    except np.linalg.LinAlgError:
        similarity = 0.0

    return similarity


def detect_task_interference(
    decompositions: dict[str, TaskVectorDecomposition],
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """
    Detect pairs of weight matrices with high task interference.

    High interference = similar singular subspaces = competition for same directions.

    Args:
        decompositions: Per-weight task vector decompositions
        threshold: Similarity threshold above which to flag interference

    Returns:
        List of (name1, name2, similarity) tuples for interfering pairs
    """
    interfering_pairs = []
    names = list(decompositions.keys())

    for i, name1 in enumerate(names):
        for name2 in names[i + 1 :]:
            decomp1 = decompositions[name1]
            decomp2 = decompositions[name2]

            similarity = compute_task_vector_similarity(decomp1, decomp2)

            if similarity > threshold:
                interfering_pairs.append((name1, name2, similarity))

    return sorted(interfering_pairs, key=lambda x: -x[2])


def svd_blend_weights(
    source_weights: dict[str, np.ndarray],
    target_weights: dict[str, np.ndarray],
    base_alphas: dict[str, float],
    config: Optional[SVDBlendConfig] = None,
) -> tuple[dict[str, np.ndarray], dict[str, TaskVectorDecomposition]]:
    """
    Apply SVD-aware blending to all weight matrices.

    Args:
        source_weights: Source model weights by name
        target_weights: Target model weights by name
        base_alphas: Base alpha per weight
        config: SVD blending configuration

    Returns:
        Tuple of (blended_weights, decompositions)
    """
    if config is None:
        config = SVDBlendConfig.default()

    blended: dict[str, np.ndarray] = {}
    decomps: dict[str, TaskVectorDecomposition] = {}

    for name, base_alpha in base_alphas.items():
        if name not in source_weights or name not in target_weights:
            continue

        source_w = source_weights[name]
        target_w = target_weights[name]

        if source_w.shape != target_w.shape:
            logger.warning(
                "Shape mismatch for %s: %s vs %s",
                name,
                source_w.shape,
                target_w.shape,
            )
            continue

        # Decompose and blend
        decomp = decompose_task_vector(source_w, target_w, config)
        decomps[name] = decomp

        merged = blend_with_svd_awareness(source_w, target_w, base_alpha, config)
        blended[name] = merged

        if decomp.effective_rank > 0:
            logger.debug(
                "%s: effective_rank=%d, variance=%.2f%%",
                name,
                decomp.effective_rank,
                decomp.variance_captured * 100,
            )

    return blended, decomps


def svd_summary(decompositions: dict[str, TaskVectorDecomposition]) -> dict:
    """
    Summarize SVD decompositions across all weight matrices.

    Args:
        decompositions: Per-weight task vector decompositions

    Returns:
        Summary statistics
    """
    if not decompositions:
        return {
            "total_weights": 0,
            "mean_effective_rank": 0.0,
            "mean_variance_captured": 0.0,
            "high_variance_count": 0,
        }

    effective_ranks = [d.effective_rank for d in decompositions.values() if d.is_valid]
    variances = [d.variance_captured for d in decompositions.values() if d.is_valid]

    return {
        "total_weights": len(decompositions),
        "valid_decompositions": len(effective_ranks),
        "mean_effective_rank": float(np.mean(effective_ranks)) if effective_ranks else 0.0,
        "max_effective_rank": int(np.max(effective_ranks)) if effective_ranks else 0,
        "mean_variance_captured": float(np.mean(variances)) if variances else 0.0,
        "min_variance_captured": float(np.min(variances)) if variances else 0.0,
        "high_variance_count": sum(1 for v in variances if v > 0.9),
    }

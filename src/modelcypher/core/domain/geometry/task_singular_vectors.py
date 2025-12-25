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
Task Singular Vectors for Model Merging.

Decomposes task vectors (W_source - W_target) via SVD to identify principal
skill directions vs structural noise. Applies different alpha values to
high-rank (skill) vs low-rank (structure) components.

Reference:
- "Task Singular Vectors: Reducing Task Interference in Model Merging" (CVPR 2025)

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
import math
from dataclasses import dataclass

from modelcypher.core.domain._backend import get_default_backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskVectorDecomposition:
    """SVD decomposition of a task vector."""

    # Left singular vectors [out_dim, k]
    U: "Array"

    # Singular values [k]
    S: "Array"

    # Right singular vectors (transposed) [k, in_dim]
    Vt: "Array"

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

    def reconstruct(self, alpha: float = 1.0) -> "Array":
        """Reconstruct the task vector with optional scaling."""
        backend = get_default_backend()
        if not self.is_valid:
            return backend.zeros(self.original_shape)

        # U @ diag(S) @ Vt
        scaled_U = self.U * self.S  # Broadcasting diag(S)
        delta = scaled_U @ self.Vt
        return alpha * delta


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
    source_weight: "Array",
    target_weight: "Array",
    config: SVDBlendConfig | None = None,
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

    backend = get_default_backend()
    original_shape = source_weight.shape

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, treat as single "singular value" = norm of difference
        delta = source_weight - target_weight
        delta_norm = float(backend.to_numpy(backend.norm(delta)))

        if delta_norm < config.epsilon:
            return TaskVectorDecomposition(
                U=backend.zeros((len(delta), 1)),
                S=backend.array([0.0]),
                Vt=backend.zeros((1, 1)),
                variance_captured=0.0,
                effective_rank=0,
                original_shape=original_shape,
            )

        # Normalize to create unit "singular vector"
        u = (delta / delta_norm).reshape(-1, 1)

        return TaskVectorDecomposition(
            U=u,
            S=backend.array([delta_norm]),
            Vt=backend.array([[1.0]]),
            variance_captured=1.0,
            effective_rank=1,
            original_shape=original_shape,
        )

    # Compute task vector
    delta = source_weight - target_weight

    # Full SVD
    try:
        U, S, Vt = backend.svd(delta, full_matrices=False)
    except Exception:
        logger.warning("SVD failed, returning zero decomposition")
        return TaskVectorDecomposition(
            U=backend.zeros((delta.shape[0], 1)),
            S=backend.array([0.0]),
            Vt=backend.zeros((1, delta.shape[1])),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Filter noise: keep only significant singular values
    S_np = backend.to_numpy(S)
    if len(S_np) > 0:
        sv_threshold = S_np[0] * config.min_sv_ratio
        significant_mask = S_np >= sv_threshold
        num_significant = int(sum(significant_mask))
    else:
        num_significant = 0

    if num_significant == 0:
        return TaskVectorDecomposition(
            U=U[:, :1],
            S=backend.array([0.0]),
            Vt=Vt[:1, :],
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Compute effective rank
    S_squared = S_np**2
    total_variance = float(sum(S_squared))
    if total_variance < config.epsilon:
        effective_rank = 0
    else:
        cumulative_variance = []
        cumsum = 0.0
        for s2 in S_squared:
            cumsum += s2
            cumulative_variance.append(cumsum / total_variance)

        effective_rank = 0
        for i, cv in enumerate(cumulative_variance):
            if cv >= 0.99:
                effective_rank = i + 1
                break
        if effective_rank == 0:
            effective_rank = len(S_np)

    # Determine how many components to keep
    if config.use_rank_ratio:
        k = max(1, int(len(S_np) * config.rank_ratio))
    else:
        # Keep enough to capture variance_threshold
        cumulative = []
        cumsum = 0.0
        for s2 in S_squared:
            cumsum += s2
            cumulative.append(cumsum / max(total_variance, config.epsilon))

        k = 0
        for i, cv in enumerate(cumulative):
            if cv >= config.variance_threshold:
                k = i + 1
                break
        if k == 0:
            k = len(S_np)

    k = min(k, num_significant, len(S_np))

    # Compute variance captured by top-k
    if total_variance > config.epsilon:
        variance_captured = float(sum(S_squared[:k]) / total_variance)
    else:
        variance_captured = 0.0

    return TaskVectorDecomposition(
        U=U,
        S=S,
        Vt=Vt,
        variance_captured=variance_captured,
        effective_rank=effective_rank,
        original_shape=original_shape,
    )


def blend_with_svd_awareness(
    source_weight: "Array",
    target_weight: "Array",
    base_alpha: float,
    config: SVDBlendConfig | None = None,
) -> "Array":
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

    backend = get_default_backend()

    # Handle 1D weights simply
    if source_weight.ndim == 1:
        # For 1D, use weighted average with base_alpha
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    # Decompose task vector
    decomp = decompose_task_vector(source_weight, target_weight, config)

    if not decomp.is_valid:
        # Fallback to simple linear blend
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    # Determine cutoff for high vs low rank
    S_np = backend.to_numpy(decomp.S)
    if config.use_rank_ratio:
        k = max(1, int(len(S_np) * config.rank_ratio))
    else:
        S_squared = S_np**2
        total_var = float(sum(S_squared))
        if total_var > config.epsilon:
            cumulative = []
            cumsum = 0.0
            for s2 in S_squared:
                cumsum += s2
                cumulative.append(cumsum / total_var)
            k = 0
            for i, cv in enumerate(cumulative):
                if cv >= config.variance_threshold:
                    k = i + 1
                    break
            if k == 0:
                k = len(S_np)
        else:
            k = 1

    k = min(k, len(S_np))

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
    # Matrix multiply: U_high @ diag(S_high) @ Vt_high
    scaled_U_high = U_high * S_high  # Broadcasting
    delta_high = scaled_U_high @ Vt_high

    # Reconstruct low-rank component (structure)
    if k < len(S_np):
        U_low = decomp.U[:, k:]
        S_low = decomp.S[k:]
        Vt_low = decomp.Vt[k:, :]
        scaled_U_low = U_low * S_low
        delta_low = scaled_U_low @ Vt_low
    else:
        delta_low = backend.zeros_like(delta_high)

    # Blend: W_merged = W_target + (1 - α_high) * Δ_high + (1 - α_low) * Δ_low
    # Note: α closer to 1 → trust target more → add less of delta
    merged = target_weight + (1.0 - high_alpha) * delta_high + (1.0 - low_alpha) * delta_low

    return merged


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

    backend = get_default_backend()

    # Use top-k singular vectors for comparison
    k1 = max(1, int(len(decomp1.S) * 0.1))
    k2 = max(1, int(len(decomp2.S) * 0.1))
    k = min(k1, k2)

    U1 = decomp1.U[:, :k]
    U2 = decomp2.U[:, :k]

    # Compute principal angles via SVD of U1.T @ U2
    # Cosines of principal angles are singular values of this product
    try:
        product = backend.transpose(U1) @ U2
        _, cosines, _ = backend.svd(product, full_matrices=False)
        # Mean cosine as similarity
        cosines_np = backend.to_numpy(cosines)
        cosines_clipped = [max(0.0, min(1.0, c)) for c in cosines_np]
        similarity = float(sum(cosines_clipped) / len(cosines_clipped))
    except Exception:
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
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    base_alphas: dict[str, float],
    config: SVDBlendConfig | None = None,
) -> tuple[dict[str, "Array"], dict[str, TaskVectorDecomposition]]:
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

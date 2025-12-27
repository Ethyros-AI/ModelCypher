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
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

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
    """Configuration for SVD-aware blending.

    IMPORTANT: This config contains ONLY numerical stability parameters.
    All blend decisions are derived from the SVD spectrum itself:

    - Rank cutoff: Determined by spectral gap (largest drop in singular values)
    - Alpha per component: Proportional to variance explained (high variance = trust source)
    - Noise filtering: Based on condition number of the matrix

    NO ARBITRARY PRESETS. The geometry tells us what to do.
    """

    # Numerical stability - derived from dtype if None
    epsilon: float | None = None

    # Condition number threshold - derived from dtype precision if None
    # Components with sv < sv_max / condition_threshold are numerical noise
    condition_threshold: float | None = None


def _get_epsilon(config: SVDBlendConfig, backend: "Backend", array: "Array") -> float:
    """Get epsilon from config or derive from dtype."""
    if config.epsilon is not None:
        return config.epsilon
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon, svd_via_eigh

    return machine_epsilon(backend, array)


def _get_condition_threshold(config: SVDBlendConfig, backend: "Backend", array: "Array") -> float:
    """Get condition threshold from config or derive from dtype."""
    if config.condition_threshold is not None:
        return config.condition_threshold
    from modelcypher.core.domain.geometry.numerical_stability import condition_threshold

    return condition_threshold(backend, array)


def _find_spectral_gap(singular_values: list[float], epsilon: float) -> int:
    """Find the rank cutoff from spectral gap (largest relative drop in singular values).

    The spectral gap is the natural boundary between "signal" (high-rank skill components)
    and "structure" (low-rank structural components). This is geometry, not tuning.

    Returns:
        Index where the gap occurs (components before this are "high-rank")
    """
    if len(singular_values) < 2:
        return len(singular_values)

    # Compute relative drops: (s[i] - s[i+1]) / s[i]
    # This measures the fractional decrease at each step
    max_gap = 0.0
    gap_index = 1  # Default to keeping at least 1 component

    for i in range(len(singular_values) - 1):
        if singular_values[i] < epsilon:
            break
        relative_drop = (singular_values[i] - singular_values[i + 1]) / singular_values[i]
        if relative_drop > max_gap:
            max_gap = relative_drop
            gap_index = i + 1

    return gap_index


def decompose_task_vector(
    source_weight: "Array",
    target_weight: "Array",
    config: SVDBlendConfig | None = None,
) -> TaskVectorDecomposition:
    """
    Decompose task vector Δ = W_source - W_target via SVD.

    All parameters are derived from the SVD spectrum itself:
    - Noise threshold: condition number based filtering
    - Effective rank: where 99% variance is captured
    - No arbitrary ratios or presets

    Args:
        source_weight: Source model weight matrix
        target_weight: Target model weight matrix (same shape)
        config: Numerical stability configuration

    Returns:
        TaskVectorDecomposition with U, S, Vt and metadata
    """
    if config is None:
        config = SVDBlendConfig()

    backend = get_default_backend()
    original_shape = source_weight.shape

    # Get epsilon derived from dtype
    epsilon = _get_epsilon(config, backend, source_weight)
    cond_threshold = _get_condition_threshold(config, backend, source_weight)

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        delta = source_weight - target_weight
        delta_norm = float(backend.to_numpy(backend.norm(delta)))

        if delta_norm < epsilon:
            return TaskVectorDecomposition(
                U=backend.zeros((len(delta), 1)),
                S=backend.array([0.0]),
                Vt=backend.zeros((1, 1)),
                variance_captured=0.0,
                effective_rank=0,
                original_shape=original_shape,
            )

        u = (delta / delta_norm).reshape(-1, 1)
        return TaskVectorDecomposition(
            U=u,
            S=backend.array([delta_norm]),
            Vt=backend.array([[1.0]]),
            variance_captured=1.0,
            effective_rank=1,
            original_shape=original_shape,
        )

    # Compute task vector and SVD
    # Cast to float32 for SVD (bfloat16 not supported)
    source_f32 = backend.astype(source_weight, "float32")
    target_f32 = backend.astype(target_weight, "float32")
    delta = source_f32 - target_f32

    try:
        U, S, Vt = svd_via_eigh(backend, delta, full_matrices=False)
    except Exception:
        logger.warning("Eigendecomposition SVD failed, returning zero decomposition")
        return TaskVectorDecomposition(
            U=backend.zeros((delta.shape[0], 1)),
            S=backend.array([0.0]),
            Vt=backend.zeros((1, delta.shape[1])),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    S_np = backend.to_numpy(S)
    if len(S_np) == 0 or S_np[0] < epsilon:
        return TaskVectorDecomposition(
            U=U[:, :1] if U.shape[1] > 0 else backend.zeros((delta.shape[0], 1)),
            S=backend.array([0.0]),
            Vt=Vt[:1, :] if Vt.shape[0] > 0 else backend.zeros((1, delta.shape[1])),
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Filter numerical noise using condition number
    # Components with sv < sv_max / condition_threshold are noise
    sv_threshold = S_np[0] / cond_threshold
    S_np_filtered = [s for s in S_np if s >= sv_threshold]
    num_significant = len(S_np_filtered)

    if num_significant == 0:
        return TaskVectorDecomposition(
            U=U[:, :1],
            S=backend.array([0.0]),
            Vt=Vt[:1, :],
            variance_captured=0.0,
            effective_rank=0,
            original_shape=original_shape,
        )

    # Effective rank: where cumulative variance reaches 99%
    # This is a statistical definition, not arbitrary
    S_squared = [s * s for s in S_np_filtered]
    total_variance = sum(S_squared)

    effective_rank = num_significant
    if total_variance > epsilon:
        cumsum = 0.0
        for i, s2 in enumerate(S_squared):
            cumsum += s2
            if cumsum / total_variance >= 0.99:
                effective_rank = i + 1
                break

    # Variance captured by significant components
    variance_captured = sum(S_squared) / max(total_variance, epsilon)

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
    Blend weights using SVD-aware alpha derived from the spectrum itself.

    The alpha for each singular component is determined by its variance contribution:
    - High variance components (skills): lower alpha = trust source more
    - Low variance components (structure): higher alpha = trust target more

    The spectral gap naturally separates skill from structure directions.
    No arbitrary presets - the geometry tells us what to do.

    Args:
        source_weight: Source model weight matrix
        target_weight: Target model weight matrix (same shape)
        base_alpha: Base alpha (scales the per-component alphas)
        config: Numerical stability configuration

    Returns:
        Blended weight matrix
    """
    if config is None:
        config = SVDBlendConfig()

    backend = get_default_backend()

    # Get epsilon derived from dtype
    epsilon = _get_epsilon(config, backend, source_weight)
    cond_threshold = _get_condition_threshold(config, backend, source_weight)

    # Handle 1D weights
    if source_weight.ndim == 1:
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    # Decompose task vector
    decomp = decompose_task_vector(source_weight, target_weight, config)

    if not decomp.is_valid:
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    S_np = backend.to_numpy(decomp.S)
    S_list = [float(s) for s in S_np]

    # Filter to significant components
    sv_threshold = S_list[0] / cond_threshold if S_list else 0
    significant_indices = [i for i, s in enumerate(S_list) if s >= sv_threshold]

    if not significant_indices:
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    # Find spectral gap - the natural boundary between skill and structure
    significant_svs = [S_list[i] for i in significant_indices]
    k = _find_spectral_gap(significant_svs, epsilon)

    # Compute per-component alpha based on variance contribution
    # Alpha_i = base_alpha * (1 - variance_fraction_i)
    # High variance components get lower alpha (preserve source skills)
    # Low variance components get higher alpha (trust target structure)
    S_squared = [s * s for s in significant_svs]
    total_variance = sum(S_squared)

    if total_variance < epsilon:
        return (1.0 - base_alpha) * source_weight + base_alpha * target_weight

    # Build the merged weight component by component
    merged = backend.zeros_like(target_weight)

    for i in significant_indices:
        s = S_list[i]
        variance_fraction = (s * s) / total_variance

        # Alpha inversely proportional to variance contribution
        # High variance (skill) → low alpha → keep source
        # Low variance (structure) → high alpha → trust target
        component_alpha = base_alpha * (1.0 - variance_fraction)
        component_alpha = max(0.0, min(1.0, component_alpha))

        # Reconstruct this singular component: u_i * s_i * v_i^T
        u_i = decomp.U[:, i : i + 1]  # Column vector
        v_i = decomp.Vt[i : i + 1, :]  # Row vector
        component = s * (u_i @ v_i)

        # Add weighted component: (1 - α) * component
        merged = merged + (1.0 - component_alpha) * component

    # Final result: target + weighted task vector components
    return target_weight + merged


def compute_task_vector_similarity(
    decomp1: TaskVectorDecomposition,
    decomp2: TaskVectorDecomposition,
    config: SVDBlendConfig | None = None,
) -> float:
    """
    Compute similarity between two task vectors based on their singular subspaces.

    Uses principal angles between the column spaces of U matrices.
    The number of components compared is the effective rank of each decomposition,
    not an arbitrary percentage.

    Args:
        decomp1: First task vector decomposition
        decomp2: Second task vector decomposition
        config: Numerical stability configuration

    Returns:
        Similarity score in [0, 1], higher = more similar
    """
    if not decomp1.is_valid or not decomp2.is_valid:
        return 0.0

    if config is None:
        config = SVDBlendConfig()

    backend = get_default_backend()

    # Use effective rank to determine how many components to compare
    # This is the geometrically meaningful number, not arbitrary 10%
    k1 = max(1, decomp1.effective_rank)
    k2 = max(1, decomp2.effective_rank)
    k = min(k1, k2, decomp1.U.shape[1], decomp2.U.shape[1])

    U1 = decomp1.U[:, :k]
    U2 = decomp2.U[:, :k]

    # Compute principal angles via SVD of U1.T @ U2
    # Cosines of principal angles are singular values of this product
    try:
        product = backend.transpose(U1) @ U2
        _, cosines, _ = svd_via_eigh(backend, product, full_matrices=False)
        # Mean cosine as similarity
        cosines_np = backend.to_numpy(cosines)
        cosines_clipped = [max(0.0, min(1.0, float(c))) for c in cosines_np]
        similarity = sum(cosines_clipped) / len(cosines_clipped) if cosines_clipped else 0.0
    except Exception:
        similarity = 0.0

    return similarity


def detect_task_interference(
    decompositions: dict[str, TaskVectorDecomposition],
) -> list[tuple[str, str, float]]:
    """
    Detect pairs of weight matrices with high task interference.

    High interference = similar singular subspaces = competition for same directions.

    Returns ALL pairs sorted by similarity (highest first). The caller decides
    what "high" means based on their context - no arbitrary threshold here.

    Args:
        decompositions: Per-weight task vector decompositions

    Returns:
        List of (name1, name2, similarity) tuples sorted by similarity descending
    """
    pairs = []
    names = list(decompositions.keys())

    for i, name1 in enumerate(names):
        for name2 in names[i + 1 :]:
            decomp1 = decompositions[name1]
            decomp2 = decompositions[name2]

            similarity = compute_task_vector_similarity(decomp1, decomp2)
            pairs.append((name1, name2, similarity))

    return sorted(pairs, key=lambda x: -x[2])


def svd_blend_weights(
    source_weights: dict[str, "Array"],
    target_weights: dict[str, "Array"],
    base_alphas: dict[str, float],
    config: SVDBlendConfig | None = None,
) -> tuple[dict[str, "Array"], dict[str, TaskVectorDecomposition]]:
    """
    Apply SVD-aware blending to all weight matrices.

    Per-component alphas are derived from the SVD spectrum of each weight matrix.

    Args:
        source_weights: Source model weights by name
        target_weights: Target model weights by name
        base_alphas: Base alpha per weight (scales the per-component alphas)
        config: Numerical stability configuration

    Returns:
        Tuple of (blended_weights, decompositions)
    """
    if config is None:
        config = SVDBlendConfig()

    blended: dict[str, "Array"] = {}
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
        "mean_effective_rank": float(sum(effective_ranks) / len(effective_ranks)) if effective_ranks else 0.0,
        "max_effective_rank": int(max(effective_ranks)) if effective_ranks else 0,
        "mean_variance_captured": float(sum(variances) / len(variances)) if variances else 0.0,
        "min_variance_captured": float(min(variances)) if variances else 0.0,
        "high_variance_count": sum(1 for v in variances if v > 0.9),
    }

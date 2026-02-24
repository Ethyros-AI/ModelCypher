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

"""Per-direction novelty analysis for model merging.

Computes per-dimension variance ratios to estimate source-vs-target novelty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class DirectionNoveltyResult:
    """Per-direction novelty analysis between source and target.

    Attributes:
        novelty_ratio: Per-direction novelty scores [d]
            Values in [0, 1]: 1.0 = source active / target dormant, 0.0 = opposite
        novel_mask: Boolean mask [d], True = novel direction
        shared_mask: Boolean mask [d], True = shared direction
        novel_count: Number of novel directions
        shared_count: Number of shared directions
        threshold: Data-derived novelty threshold
        source_variance: Per-direction variance in source [d]
        target_variance: Per-direction variance in target [d]
        mean_novelty: Mean novelty ratio across directions
        novel_indices: Indices of novel directions (sorted by novelty descending)
    """

    novelty_ratio: "Array"
    novel_mask: "Array"
    shared_mask: "Array"
    novel_count: int
    shared_count: int
    threshold: float
    source_variance: "Array"
    target_variance: "Array"
    mean_novelty: float
    novel_indices: list[int]


def compute_per_direction_novelty(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> DirectionNoveltyResult:
    """Identify directions that are 'activated' in source but 'dormant' in target.

    For each feature dimension, computes the variance ratio to determine
    whether source has more activity (novel) or target has more (preserve).

    Args:
        source_activations: Source activation matrix [n_samples, d]
        target_activations: Target activation matrix [n_samples, d]
        backend: Backend for tensor operations

    Returns:
        DirectionNoveltyResult with per-direction novelty analysis
    """
    b = backend or get_default_backend()

    shape_src = b.shape(source_activations)
    shape_tgt = b.shape(target_activations)
    n = int(shape_src[0])
    d = int(shape_src[1])

    if int(shape_tgt[0]) != n:
        raise ValueError(
            f"Sample count mismatch: source {shape_src[0]} vs target {shape_tgt[0]}"
        )
    if int(shape_tgt[1]) != d:
        raise ValueError(
            f"Dimension mismatch: source {shape_src[1]} vs target {shape_tgt[1]}"
        )

    # Promote to high precision for variance computation
    src_arr = b.astype(source_activations, precision_dtype(b, reference=source_activations))
    tgt_arr = b.astype(target_activations, precision_dtype(b, reference=target_activations))

    # Compute per-direction variance (column-wise)
    # var = mean((x - mean(x))^2) along axis 0
    src_mean = b.mean(src_arr, axis=0)  # [d]
    tgt_mean = b.mean(tgt_arr, axis=0)  # [d]
    b.eval(src_mean, tgt_mean)

    src_centered = src_arr - src_mean  # [n, d]
    tgt_centered = tgt_arr - tgt_mean  # [n, d]
    b.eval(src_centered, tgt_centered)

    source_variance = b.mean(src_centered * src_centered, axis=0)  # [d]
    target_variance = b.mean(tgt_centered * tgt_centered, axis=0)  # [d]
    b.eval(source_variance, target_variance)

    # Ensure non-negative (numerical precision)
    source_variance = b.maximum(source_variance, b.zeros_like(source_variance))
    target_variance = b.maximum(target_variance, b.zeros_like(target_variance))
    b.eval(source_variance, target_variance)

    # Compute novelty ratio: source_var / (source_var + target_var)
    # High ratio = source active, target dormant = NOVEL
    # Low ratio = source dormant, target active = PRESERVE
    eps = division_epsilon(b, source_variance)
    denom = source_variance + target_variance + eps
    novelty_ratio = source_variance / denom
    b.eval(novelty_ratio)

    # Clamp to [0, 1]
    novelty_ratio = b.clip(novelty_ratio, 0.0, 1.0)
    b.eval(novelty_ratio)

    # =========================================================================
    # GEOMETRY-DERIVED CLASSIFICATION (no arbitrary thresholds)
    # =========================================================================
    # The novelty ratio is src_var / (src_var + tgt_var), which is already
    # geometrically meaningful:
    #   - ratio = 1.0: source active, target dormant (fully novel)
    #   - ratio = 0.5: equal variance (shared)
    #   - ratio = 0.0: source dormant, target active (preserve)
    #
    # The threshold IS the geometry: 0.5 is the exact point where src_var = tgt_var.
    # This is not a heuristic - it's the mathematical definition of "novel".
    #
    # Novel: source has MORE variance than target (ratio > 0.5)
    # Shared: source has LESS OR EQUAL variance (ratio <= 0.5)
    # =========================================================================
    threshold = 0.5  # Exact geometric midpoint: src_var = tgt_var

    threshold_arr = b.array([threshold])
    novel_mask = novelty_ratio > threshold_arr
    shared_mask = novelty_ratio <= threshold_arr
    b.eval(novel_mask, shared_mask)

    # Count
    count_dtype = precision_dtype(b, reference=novelty_ratio)
    novel_count = int(b.to_scalar(b.sum(b.astype(novel_mask, count_dtype))))
    shared_count = int(b.to_scalar(b.sum(b.astype(shared_mask, count_dtype))))

    # Mean novelty
    mean_novelty = float(b.to_scalar(b.mean(novelty_ratio)))

    # Get indices of novel directions, sorted by novelty descending
    novel_indices_list: list[int] = []
    if novel_count > 0:
        # Get indices where novel_mask is True
        novel_idx_result = b.nonzero(novel_mask)
        if len(novel_idx_result) > 0 and novel_idx_result[0].shape[0] > 0:
            novel_idx_arr = novel_idx_result[0]
            # Get novelty values at these indices
            novel_values = b.take(novelty_ratio, novel_idx_arr, axis=0)
            b.eval(novel_values)
            # Sort by novelty descending
            sort_idx = b.argsort(novel_values)
            # Reverse to get descending order
            rev_idx = b.arange(int(b.shape(sort_idx)[0]) - 1, -1, -1)
            sort_idx_desc = b.take(sort_idx, rev_idx, axis=0)
            b.eval(sort_idx_desc)
            # Reorder indices
            sorted_novel_idx = b.take(novel_idx_arr, sort_idx_desc, axis=0)
            b.eval(sorted_novel_idx)
            # Convert to Python list
            novel_indices_list = [
                int(b.to_scalar(b.take(sorted_novel_idx, b.array([i]), axis=0)))
                for i in range(int(b.shape(sorted_novel_idx)[0]))
            ]

    logger.info(
        "Direction novelty: %d novel (%.1f%%), %d shared, threshold=%.4f, mean=%.4f",
        novel_count,
        novel_count / max(d, 1) * 100,
        shared_count,
        threshold,
        mean_novelty,
    )

    return DirectionNoveltyResult(
        novelty_ratio=novelty_ratio,
        novel_mask=novel_mask,
        shared_mask=shared_mask,
        novel_count=novel_count,
        shared_count=shared_count,
        threshold=threshold,
        source_variance=source_variance,
        target_variance=target_variance,
        mean_novelty=mean_novelty,
        novel_indices=novel_indices_list,
    )


def compute_direction_projector(
    novelty_result: DirectionNoveltyResult,
    backend: "Backend | None" = None,
    novel_only: bool = True,
) -> "Array":
    """Create a diagonal projector that filters directions by novelty.

    Returns a diagonal matrix that, when applied to a weight delta,
    only preserves the components in novel (or shared) directions.

    Args:
        novelty_result: Result from compute_per_direction_novelty
        backend: Backend for tensor operations
        novel_only: If True, project to novel directions. If False, to shared.

    Returns:
        Diagonal projector [d, d]
    """
    b = backend or get_default_backend()

    mask = novelty_result.novel_mask if novel_only else novelty_result.shared_mask
    int(b.shape(mask)[0])

    # Convert bool mask to float (0.0 or 1.0)
    count_dtype = precision_dtype(b, reference=novelty_result.novelty_ratio)
    mask_float = b.astype(mask, count_dtype)
    b.eval(mask_float)

    # Create diagonal matrix
    projector = b.diag(mask_float)
    b.eval(projector)

    return projector


def compute_weighted_direction_projector(
    novelty_result: DirectionNoveltyResult,
    backend: "Backend | None" = None,
) -> "Array":
    """Create a weighted projector based on novelty ratios.

    Instead of a binary mask, uses the novelty ratio itself as weights.
    This allows gradual transfer based on how "novel" each direction is.

    Args:
        novelty_result: Result from compute_per_direction_novelty
        backend: Backend for tensor operations

    Returns:
        Diagonal projector [d, d] with weights = novelty_ratio
    """
    b = backend or get_default_backend()

    # Use novelty ratio directly as weights
    # High novelty = high weight = transfer more
    weights = novelty_result.novelty_ratio
    b.eval(weights)

    # Create diagonal matrix
    projector = b.diag(weights)
    b.eval(projector)

    return projector


def diagnose_variance_distribution(
    novelty_result: DirectionNoveltyResult,
    backend: "Backend | None" = None,
) -> dict[str, float]:
    """Compute diagnostic statistics about the variance distribution.

    Useful for understanding why a merge might be failing.

    Args:
        novelty_result: Result from compute_per_direction_novelty
        backend: Backend for tensor operations

    Returns:
        Dictionary of diagnostic metrics
    """
    b = backend or get_default_backend()

    src_var = novelty_result.source_variance
    tgt_var = novelty_result.target_variance
    novelty = novelty_result.novelty_ratio

    d = int(b.shape(src_var)[0])

    # Total variance
    src_total_var = float(b.to_scalar(b.sum(src_var)))
    tgt_total_var = float(b.to_scalar(b.sum(tgt_var)))
    eps = division_epsilon(b, src_var)

    # Variance in novel vs shared directions
    novel_mask_float = b.astype(novelty_result.novel_mask, precision_dtype(b, reference=src_var))
    shared_mask_float = b.astype(novelty_result.shared_mask, precision_dtype(b, reference=src_var))
    b.eval(novel_mask_float, shared_mask_float)

    src_novel_var = float(b.to_scalar(b.sum(src_var * novel_mask_float)))
    src_shared_var = float(b.to_scalar(b.sum(src_var * shared_mask_float)))
    tgt_novel_var = float(b.to_scalar(b.sum(tgt_var * novel_mask_float)))
    tgt_shared_var = float(b.to_scalar(b.sum(tgt_var * shared_mask_float)))

    # Novelty distribution statistics
    novelty_std = float(b.to_scalar(b.std(novelty)))
    novelty_median_arr = b.sort(novelty, axis=0)
    novelty_median = float(b.to_scalar(b.take(novelty_median_arr, b.array([d // 2]), axis=0)))

    return {
        "source_total_variance": src_total_var,
        "target_total_variance": tgt_total_var,
        "source_novel_variance": src_novel_var,
        "source_shared_variance": src_shared_var,
        "target_novel_variance": tgt_novel_var,
        "target_shared_variance": tgt_shared_var,
        "novel_fraction_by_variance": src_novel_var / max(src_total_var, eps),
        "novelty_std": novelty_std,
        "novelty_median": novelty_median,
        "novelty_threshold": novelty_result.threshold,
    }


def compute_subspace_novelty(
    source_activations: "Array",
    target_activations: "Array",
    stitch: "Array | None" = None,
    backend: "Backend | None" = None,
) -> DirectionNoveltyResult:
    """Compute novelty using principal angle analysis.

    Args:
        source_activations: Source activation matrix [n, d_src]
        target_activations: Target activation matrix [n, d_tgt]
        stitch: Optional alignment matrix [d_tgt, d_src] for cross-arch.
            If provided, source is aligned to target space before analysis.
            The stitch is orthonormalized (U @ V^T) to remove scaling artifacts.
        backend: Backend for tensor operations

    Returns:
        DirectionNoveltyResult with novelty classification
    """
    from modelcypher.core.domain.geometry.subspace import compute_subspace_overlap

    b = backend or get_default_backend()

    src_arr = b.array(source_activations)
    tgt_arr = b.array(target_activations)

    shape_src = b.shape(src_arr)
    shape_tgt = b.shape(tgt_arr)
    n_src = int(shape_src[0])
    n_tgt = int(shape_tgt[0])
    d_src = int(shape_src[1])
    d_tgt = int(shape_tgt[1])

    if n_src != n_tgt:
        raise ValueError(f"Sample count mismatch: source {n_src} vs target {n_tgt}")

    compute_dtype = precision_dtype(b, reference=src_arr)
    src_arr = b.astype(src_arr, compute_dtype)
    tgt_arr = b.astype(tgt_arr, compute_dtype)

    # Handle cross-architecture: align source to target dimension
    if d_src != d_tgt:
        if stitch is None:
            raise ValueError(
                f"Dimension mismatch (src={d_src}, tgt={d_tgt}) requires stitch matrix"
            )
        stitch_arr = b.astype(b.array(stitch), compute_dtype)
        stitch_shape = b.shape(stitch_arr)

        # Verify stitch dimensions
        if int(stitch_shape[0]) != d_tgt or int(stitch_shape[1]) != d_src:
            raise ValueError(
                f"Stitch shape {stitch_shape} incompatible with dims (tgt={d_tgt}, src={d_src})"
            )

        # Orthonormalize stitch via SVD: U @ V^T removes scaling artifacts
        # while preserving the rotation/alignment
        U, S, Vt = b.svd(stitch_arr, compute_uv=True)
        b.eval(U, S, Vt)

        # Handle full vs thin SVD: for [m, n] matrix with k = min(m, n),
        # full SVD gives U [m, m], Vt [n, n] - we need U[:, :k] @ Vt[:k, :]
        k = min(d_tgt, d_src)
        U_thin = U[:, :k]  # [d_tgt, k]
        Vt_thin = Vt[:k, :]  # [k, d_src]
        ortho_stitch = b.matmul(U_thin, Vt_thin)
        b.eval(ortho_stitch)

        # Align source: src_aligned = src @ stitch^T
        src_aligned = b.matmul(src_arr, b.transpose(ortho_stitch))
        b.eval(src_aligned)

        logger.info(
            "SUBSPACE NOVELTY: Cross-arch alignment %d → %d (orthonormalized stitch)",
            d_src, d_tgt
        )
    else:
        src_aligned = src_arr

    # Compute subspace overlap using principal angle analysis
    subspace_result = compute_subspace_overlap(src_aligned, tgt_arr, backend=b)

    # Build novelty mask in target dimension space
    # Novel directions are those in source's subspace but orthogonal to target's
    novel_basis = subspace_result.novel_basis  # [k_novel, d_tgt]
    shared_basis = subspace_result.shared_basis  # [k_shared, d_tgt]

    n_novel = int(b.shape(novel_basis)[0])
    int(b.shape(shared_basis)[0])

    # Create per-direction novelty by projecting onto novel subspace
    # For each direction i, novelty[i] = how much of direction i is in novel subspace
    if n_novel > 0:
        # Novel projector: P_novel = V_novel^T @ V_novel
        novel_projector = b.matmul(b.transpose(novel_basis), novel_basis)
        b.eval(novel_projector)

        # Novelty score for each direction = diagonal of projector
        # (how much of direction i overlaps with novel subspace)
        novelty_ratio = b.diag(novel_projector)
        b.eval(novelty_ratio)
    else:
        novelty_ratio = b.zeros((d_tgt,))
        b.eval(novelty_ratio)

    # Clamp to [0, 1]
    novelty_ratio = b.clip(novelty_ratio, 0.0, 1.0)
    b.eval(novelty_ratio)

    # Threshold is geometry-derived: 0.5 means equal projection to novel and shared
    threshold = 0.5
    threshold_arr = b.array([threshold])
    novel_mask = novelty_ratio > threshold_arr
    shared_mask = novelty_ratio <= threshold_arr
    b.eval(novel_mask, shared_mask)

    # Count
    count_dtype = precision_dtype(b, reference=novelty_ratio)
    novel_count = int(b.to_scalar(b.sum(b.astype(novel_mask, count_dtype))))
    shared_count = int(b.to_scalar(b.sum(b.astype(shared_mask, count_dtype))))
    mean_novelty = float(b.to_scalar(b.mean(novelty_ratio)))

    # Compute variance for diagnostics (not used for classification)
    src_mean = b.mean(src_aligned, axis=0)
    tgt_mean = b.mean(tgt_arr, axis=0)
    src_centered = src_aligned - src_mean
    tgt_centered = tgt_arr - tgt_mean
    source_variance = b.mean(src_centered * src_centered, axis=0)
    target_variance = b.mean(tgt_centered * tgt_centered, axis=0)
    b.eval(source_variance, target_variance)

    # Get indices of novel directions
    novel_indices_list: list[int] = []
    if novel_count > 0:
        novel_idx_result = b.nonzero(novel_mask)
        if len(novel_idx_result) > 0 and novel_idx_result[0].shape[0] > 0:
            novel_idx_arr = novel_idx_result[0]
            novel_values = b.take(novelty_ratio, novel_idx_arr, axis=0)
            b.eval(novel_values)
            sort_idx = b.argsort(novel_values)
            rev_idx = b.arange(int(b.shape(sort_idx)[0]) - 1, -1, -1)
            sort_idx_desc = b.take(sort_idx, rev_idx, axis=0)
            sorted_novel_idx = b.take(novel_idx_arr, sort_idx_desc, axis=0)
            b.eval(sorted_novel_idx)
            novel_indices_list = [
                int(b.to_scalar(b.take(sorted_novel_idx, b.array([i]), axis=0)))
                for i in range(int(b.shape(sorted_novel_idx)[0]))
            ]

    logger.info(
        "SUBSPACE NOVELTY: %d novel (%.1f%%), %d shared, subspace_overlap=%.1f%%, mean=%.4f",
        novel_count,
        novel_count / max(d_tgt, 1) * 100,
        shared_count,
        subspace_result.overlap_fraction * 100,
        mean_novelty,
    )

    return DirectionNoveltyResult(
        novelty_ratio=novelty_ratio,
        novel_mask=novel_mask,
        shared_mask=shared_mask,
        novel_count=novel_count,
        shared_count=shared_count,
        threshold=threshold,
        source_variance=source_variance,
        target_variance=target_variance,
        mean_novelty=mean_novelty,
        novel_indices=novel_indices_list,
    )


__all__ = [
    "DirectionNoveltyResult",
    "compute_per_direction_novelty",
    "compute_subspace_novelty",
    "compute_direction_projector",
    "compute_weighted_direction_projector",
    "diagnose_variance_distribution",
]

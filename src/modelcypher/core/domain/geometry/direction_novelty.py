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

Identifies which directions (neurons/features) are "activated" in source but
"dormant" in target. These are the directions where knowledge transfer is most
beneficial and least destructive.

Mathematical Foundation:
    For each direction i in the activation space:
        - source_var[i] = variance of source activations along direction i
        - target_var[i] = variance of target activations along direction i
        - novelty_ratio[i] = source_var[i] / (source_var[i] + target_var[i])

    Interpretation:
        - novelty_ratio ≈ 1.0: Source uses this direction heavily, target doesn't
          → NOVEL: Safe to transfer knowledge here
        - novelty_ratio ≈ 0.5: Both use this direction similarly
          → SHARED: Be careful, might interfere
        - novelty_ratio ≈ 0.0: Target uses heavily, source doesn't
          → PRESERVE: Don't touch, target's knowledge

Why This Matters:
    The current merge algorithm uses density-weighted null-space projection,
    but doesn't explicitly identify which directions are "activated in source,
    dormant in target." This analysis provides that missing piece, allowing
    us to be surgical about knowledge transfer.

Integration:
    Combined with subspace analysis, we can:
    1. Identify shared subspace (from principal angles) - don't disturb
    2. Within the "novel" subspace, identify specific directions with high novelty
    3. Only inject delta in those high-novelty directions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    machine_epsilon,
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
    d = int(b.shape(mask)[0])

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
        "novel_fraction_by_variance": src_novel_var / max(src_total_var, 1e-10),
        "novelty_std": novelty_std,
        "novelty_median": novelty_median,
        "novelty_threshold": novelty_result.threshold,
    }


__all__ = [
    "DirectionNoveltyResult",
    "compute_per_direction_novelty",
    "compute_direction_projector",
    "compute_weighted_direction_projector",
    "diagnose_variance_distribution",
]

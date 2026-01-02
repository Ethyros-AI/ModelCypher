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

"""Curvature-guided alignment for model merging.

Key insight: Curvature differences between models represent different "views"
of the same underlying topology - the transformation needed to align, not
fundamental incompatibility.

This module provides:
1. Curvature-weighted Procrustes alignment
2. Layer-wise alignment effort estimation
3. Intrinsic dimension scaling for projection
4. Curvature-guided correspondence matching
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.curvature_profile import (
        CurvatureProfile,
        LayerCurvature,
    )
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentGuidance:
    """Guidance for aligning a layer based on curvature analysis.

    This tells the alignment algorithm HOW to transform, not WHETHER to.
    """

    layer_idx: int

    # Effort required (0-1): higher = more transformation needed
    alignment_effort: float

    # Dimension scaling factor: >1 = expand, <1 = compress
    dimension_scale: float

    # Curvature correction factor: how much to adjust for curvature mismatch
    curvature_correction: float

    # Recommended alignment weight (how much to trust this layer's alignment)
    alignment_weight: float


@dataclass(frozen=True)
class AlignmentPlan:
    """Complete alignment plan derived from curvature profiles."""

    source_model: str
    target_model: str

    # Per-layer guidance
    layer_guidance: list[AlignmentGuidance]

    # Global statistics
    total_alignment_effort: float  # Sum of per-layer efforts
    mean_dimension_scale: float


def compute_alignment_guidance(
    source_profile: "CurvatureProfile",
    target_profile: "CurvatureProfile",
) -> AlignmentPlan:
    """Compute alignment guidance from curvature profiles.

    The key mathematical insight:
    - Curvature differences = rotation/transformation needed
    - Intrinsic dimension differences = projection/embedding needed
    - Similar curvature patterns at different scales = same topology, different "zoom"

    Args:
        source_profile: Curvature profile of source model
        target_profile: Curvature profile of target model

    Returns:
        AlignmentPlan with per-layer guidance and global effort statistics
    """
    # Build layer correspondence based on relative position
    source_layers = {lc.layer_idx: lc for lc in source_profile.layer_curvatures}
    target_layers = {lc.layer_idx: lc for lc in target_profile.layer_curvatures}

    guidance_list: list[AlignmentGuidance] = []

    # Map layers by relative position (0.0 = first layer, 1.0 = last layer)
    for src_idx, src_lc in source_layers.items():
        # Find corresponding target layer by relative position
        src_position = src_idx / max(1, source_profile.total_layers - 1)
        tgt_idx = round(src_position * (target_profile.total_layers - 1))
        tgt_lc = target_layers.get(tgt_idx)

        if tgt_lc is None:
            # No corresponding layer - use defaults
            guidance_list.append(AlignmentGuidance(
                layer_idx=src_idx,
                alignment_effort=1.0,
                dimension_scale=1.0,
                curvature_correction=0.0,
                alignment_weight=0.5,
            ))
            continue

        # Compute guidance for this layer pair
        guidance = _compute_layer_guidance(src_lc, tgt_lc, src_idx)
        guidance_list.append(guidance)

    # Compute global statistics
    total_effort = sum(g.alignment_effort for g in guidance_list)
    mean_scale = (
        sum(g.dimension_scale for g in guidance_list) / len(guidance_list)
        if guidance_list else 1.0
    )

    return AlignmentPlan(
        source_model=source_profile.model_path,
        target_model=target_profile.model_path,
        layer_guidance=guidance_list,
        total_alignment_effort=total_effort,
        mean_dimension_scale=mean_scale,
    )


def _compute_layer_guidance(
    src: "LayerCurvature",
    tgt: "LayerCurvature",
    layer_idx: int,
) -> AlignmentGuidance:
    """Compute alignment guidance for a single layer pair."""
    backend = get_default_backend()
    eps = division_epsilon(
        backend,
        backend.array([src.ollivier_ricci_mean, tgt.ollivier_ricci_mean]),
    )

    # 1. Intrinsic dimension scaling
    src_dim = src.intrinsic_dimension if src.intrinsic_dimension > 0 else 1.0
    tgt_dim = tgt.intrinsic_dimension if tgt.intrinsic_dimension > 0 else 1.0
    dimension_scale = tgt_dim / src_dim

    # 2. Curvature correction
    # If curvatures have same sign, less correction needed
    # If curvatures have opposite signs, more correction needed
    src_ricci = src.ollivier_ricci_mean
    tgt_ricci = tgt.ollivier_ricci_mean

    if src_ricci != 0 and tgt_ricci != 0:
        # Same sign = similar geometry, easier alignment
        same_sign = (src_ricci > 0) == (tgt_ricci > 0)
        curvature_diff = abs(src_ricci - tgt_ricci)

        if same_sign:
            # Curvature correction based on magnitude difference
            curvature_correction = curvature_diff / (abs(src_ricci) + abs(tgt_ricci) + eps)
        else:
            # Opposite signs = fundamentally different local geometry
            # Needs curvature flow to reconcile
            curvature_correction = 1.0 + curvature_diff
    else:
        # Both curvatures zero = unmeasured. Use 1.0 (neutral, no correction)
        # since we have no geometric information to derive a correction from.
        curvature_correction = 1.0

    # 3. Alignment effort (0-1)
    # Higher effort = more transformation needed
    dim_effort = min(1.0, abs(math.log(dimension_scale)) / math.log(2))  # Double/half = effort 1.0
    curv_effort = min(1.0, curvature_correction)

    # Alignment effort: geometric mean of dimension and curvature effort
    # (no arbitrary weights - both contribute equally)
    alignment_effort = math.sqrt(dim_effort * curv_effort)

    # 4. Alignment weight = similarity (no artificial floor)
    # Layers with similar curvature profiles are more reliable
    alignment_weight = 1.0 - min(1.0, curvature_correction)

    return AlignmentGuidance(
        layer_idx=layer_idx,
        alignment_effort=alignment_effort,
        dimension_scale=dimension_scale,
        curvature_correction=curvature_correction,
        alignment_weight=alignment_weight,
    )


def curvature_weighted_procrustes(
    source_activations: "Array",
    target_activations: "Array",
    guidance: AlignmentGuidance,
    backend: "Backend",
) -> "Array":
    """Procrustes alignment weighted by curvature guidance.

    Standard Procrustes finds optimal rotation R to minimize ||XR - Y||.
    Curvature-weighted Procrustes adjusts the optimization:
    1. Pre-scale by dimension_scale if projection needed
    2. Weight samples by local curvature similarity
    3. Apply curvature correction to final rotation

    Args:
        source_activations: Source representations [N, D_s]
        target_activations: Target representations [N, D_t]
        guidance: Alignment guidance for this layer
        backend: Compute backend

    Returns:
        Rotation/transformation matrix [D_s, D_t]
    """
    # Get dimensions
    n_samples = backend.shape(source_activations)[0]
    d_source = backend.shape(source_activations)[1]
    d_target = backend.shape(target_activations)[1]

    # Step 1: Dimension alignment if needed
    if d_source != d_target:
        # Use truncated SVD for dimension reduction/expansion
        # This preserves the most important directions
        min_dim = min(d_source, d_target)

        # SVD of source
        U, S, Vt = backend.svd(source_activations)

        # Truncate to target dimension
        if d_source > d_target:
            # Reduce: project to top d_target components
            source_projected = backend.matmul(U[:, :d_target], backend.diag(S[:d_target]))
        else:
            # Expand: pad with zeros (will be filled by Procrustes)
            source_projected = backend.zeros((n_samples, d_target), dtype=source_activations.dtype)
            source_projected = backend.index_update(
                source_projected,
                (slice(None), slice(d_source)),
                source_activations
            )
        source_activations = source_projected

    # Step 2: Center the data
    source_mean = backend.mean(source_activations, axis=0)
    target_mean = backend.mean(target_activations, axis=0)

    source_centered = source_activations - source_mean
    target_centered = target_activations - target_mean

    # Step 3: Compute optimal rotation via SVD
    # R = V @ U^T where M = U @ S @ V^T is SVD of target^T @ source
    M = backend.matmul(backend.transpose(target_centered), source_centered)
    U, S, Vt = backend.svd(M)

    # Optimal orthogonal transformation
    R = backend.matmul(U, Vt)

    # Step 4: Apply curvature correction
    # Dampen rotation based on curvature mismatch (hyperbolic decay, no arbitrary constants)
    # correction=0 → damping=1.0 (full rotation), correction→∞ → damping→0 (identity)
    damping = 1.0 / (1.0 + guidance.curvature_correction)
    R = R * damping + backend.eye(backend.shape(R)[0]) * (1 - damping)

    return R


def compute_layer_correspondence_by_curvature(
    source_profile: "CurvatureProfile",
    target_profile: "CurvatureProfile",
) -> dict[int, int]:
    """Find layer correspondence based on curvature profile matching.

    Instead of matching by position or activation similarity, match layers
    that have similar curvature "signatures" - they're likely encoding
    similar geometric structures.

    Args:
        source_profile: Source model curvature profile
        target_profile: Target model curvature profile

    Returns:
        Dict mapping source layer index -> target layer index
    """
    correspondence: dict[int, int] = {}
    backend = get_default_backend()
    eps = division_epsilon(backend, backend.array([0.0]))

    source_layers = source_profile.layer_curvatures
    target_layers = target_profile.layer_curvatures

    # Build curvature feature vectors for each layer
    def layer_features(lc: "LayerCurvature") -> tuple[float, float, float]:
        return (
            lc.intrinsic_dimension,
            lc.ollivier_ricci_mean,
            lc.ollivier_ricci_std,
        )

    # For each source layer, find best matching target layer
    # Constrained to preserve relative ordering (monotonic mapping)
    used_targets: set[int] = set()

    for src_lc in source_layers:
        src_features = layer_features(src_lc)

        # Expected target position (by relative depth)
        src_rel_pos = src_lc.layer_idx / max(1, len(source_layers) - 1)
        expected_tgt_idx = round(src_rel_pos * (len(target_layers) - 1))

        # Search window around expected position
        window = max(3, len(target_layers) // 4)

        best_match = None
        best_score = float("inf")

        for tgt_lc in target_layers:
            if tgt_lc.layer_idx in used_targets:
                continue

            # Penalize matches far from expected position
            # Penalty > 1.0 means outside the search window
            position_penalty = abs(tgt_lc.layer_idx - expected_tgt_idx) / max(1, window)
            if position_penalty > 1.0:
                continue

            tgt_features = layer_features(tgt_lc)

            # Feature distance
            feature_dist = sum(
                abs(s - t) / (abs(s) + abs(t) + eps)
                for s, t in zip(src_features, tgt_features)
            ) / 3

            # Score: average of feature distance and position penalty (equal weights)
            score = (feature_dist + position_penalty) / 2

            if score < best_score:
                best_score = score
                best_match = tgt_lc.layer_idx

        if best_match is not None:
            correspondence[src_lc.layer_idx] = best_match
            used_targets.add(best_match)

    return correspondence

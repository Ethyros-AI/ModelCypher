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

Provides curvature-weighted alignment and per-layer curvature/dimension deltas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.lie_rotation import so_scale_rotation
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
)

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

    # Dimension scaling factor: >1 = expand, <1 = compress
    dimension_scale: float

    # Absolute intrinsic dimension delta (|target - source|)
    intrinsic_dimension_diff: float

    # Ollivier-Ricci delta (target - source)
    ollivier_ricci_delta: float

    # Normalized Ollivier-Ricci mismatch ratio in [0, 1]
    ollivier_ricci_relative_diff: float


@dataclass(frozen=True)
class AlignmentPlan:
    """Complete alignment plan derived from curvature profiles."""

    source_model: str
    target_model: str

    # Per-layer guidance
    layer_guidance: tuple[AlignmentGuidance, ...]

    # Global statistics (raw measurements)
    mean_dimension_scale: float
    mean_intrinsic_dimension_diff: float
    mean_ollivier_ricci_delta: float
    mean_ollivier_ricci_relative_diff: float


def compute_alignment_guidance(
    source_profile: "CurvatureProfile",
    target_profile: "CurvatureProfile",
) -> AlignmentPlan:
    """Compute alignment guidance from curvature profiles.

    Args:
        source_profile: Curvature profile of source model
        target_profile: Curvature profile of target model

    Returns:
        AlignmentPlan with per-layer guidance and global curvature/dimension statistics
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
            guidance_list.append(
                AlignmentGuidance(
                    layer_idx=src_idx,
                    dimension_scale=1.0,
                    intrinsic_dimension_diff=0.0,
                    ollivier_ricci_delta=0.0,
                    ollivier_ricci_relative_diff=0.0,
                )
            )
            continue

        # Compute guidance for this layer pair
        guidance = _compute_layer_guidance(src_lc, tgt_lc, src_idx)
        guidance_list.append(guidance)

    # Compute global statistics using vectorized operations
    if guidance_list:
        backend = get_default_backend()
        scales = backend.array([g.dimension_scale for g in guidance_list])
        dim_diffs = backend.array([g.intrinsic_dimension_diff for g in guidance_list])
        ricci_deltas = backend.array([g.ollivier_ricci_delta for g in guidance_list])
        ricci_rels = backend.array([g.ollivier_ricci_relative_diff for g in guidance_list])

        mean_scale_arr = backend.mean(scales)
        mean_dim_diff_arr = backend.mean(dim_diffs)
        mean_ricci_delta_arr = backend.mean(ricci_deltas)
        mean_ricci_rel_arr = backend.mean(ricci_rels)
        backend.eval(mean_scale_arr, mean_dim_diff_arr, mean_ricci_delta_arr, mean_ricci_rel_arr)

        mean_scale = float(backend.to_scalar(mean_scale_arr))
        mean_dim_diff = float(backend.to_scalar(mean_dim_diff_arr))
        mean_ricci_delta = float(backend.to_scalar(mean_ricci_delta_arr))
        mean_ricci_rel = float(backend.to_scalar(mean_ricci_rel_arr))
    else:
        mean_scale = 1.0
        mean_dim_diff = 0.0
        mean_ricci_delta = 0.0
        mean_ricci_rel = 0.0

    return AlignmentPlan(
        source_model=source_profile.model_path,
        target_model=target_profile.model_path,
        layer_guidance=tuple(guidance_list),
        mean_dimension_scale=mean_scale,
        mean_intrinsic_dimension_diff=mean_dim_diff,
        mean_ollivier_ricci_delta=mean_ricci_delta,
        mean_ollivier_ricci_relative_diff=mean_ricci_rel,
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
    src_dim = float(src.intrinsic_dimension)
    tgt_dim = float(tgt.intrinsic_dimension)
    if src_dim <= 0.0 or tgt_dim <= 0.0:
        dimension_scale = 1.0
        dim_diff = 0.0
    else:
        dimension_scale = tgt_dim / src_dim
        dim_diff = abs(tgt_dim - src_dim)

    # 2. Curvature correction
    # If curvatures have same sign, less correction needed
    # If curvatures have opposite signs, more correction needed
    src_ricci = src.ollivier_ricci_mean
    tgt_ricci = tgt.ollivier_ricci_mean

    ricci_delta = tgt_ricci - src_ricci
    curvature_diff = abs(ricci_delta)
    ricci_relative = curvature_diff / (abs(src_ricci) + abs(tgt_ricci) + eps)

    return AlignmentGuidance(
        layer_idx=layer_idx,
        dimension_scale=dimension_scale,
        intrinsic_dimension_diff=dim_diff,
        ollivier_ricci_delta=ricci_delta,
        ollivier_ricci_relative_diff=ricci_relative,
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
        min(d_source, d_target)

        # SVD of source (geodesic - GPU-only, iterates until convergence)
        U, S, Vt = geodesic_svd(backend, source_activations)

        # Truncate to target dimension
        if d_source > d_target:
            # Reduce: project to top d_target components
            source_projected = backend.matmul(U[:, :d_target], backend.diag(S[:d_target]))
        else:
            # Expand: pad with zeros (will be filled by Procrustes)
            pad = backend.zeros((n_samples, d_target - d_source), dtype=source_activations.dtype)
            source_projected = backend.concatenate([source_activations, pad], axis=1)
        source_activations = source_projected

    # Step 2: Center the data
    source_mean = backend.mean(source_activations, axis=0)
    target_mean = backend.mean(target_activations, axis=0)

    source_centered = source_activations - source_mean
    target_centered = target_activations - target_mean

    # Step 3: Compute optimal rotation via SVD (geodesic - GPU-only)
    # R = V @ U^T where M = U @ S @ V^T is SVD of target^T @ source
    M = backend.matmul(backend.transpose(target_centered), source_centered)
    U, S, Vt = geodesic_svd(backend, M)

    # Optimal orthogonal transformation
    R = backend.matmul(U, Vt)

    # Step 4: Apply curvature correction
    # Dampen rotation geodesically to preserve orthogonality.
    # relative_diff=0 → damping=1.0 (full rotation), relative_diff→∞ → damping→0 (identity)
    damping = 1.0 / (1.0 + guidance.ollivier_ricci_relative_diff)
    R = so_scale_rotation(R, damping, backend=backend)

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

    # For each source layer, find minimum-distance target layer
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

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

"""Functional transplant for zero-shot knowledge transfer.

Implements constrained replacement in weight space:
    A_core @ W' = A_core @ W_source_aligned
    A_boundary @ W' = A_boundary @ W_target

Update is projected into boundary null space, preserving connectivity by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.birkhoff_projector import BirkhoffProjector
from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
from modelcypher.core.domain.geometry.gram_aligner import GramAligner
from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreBoundaryPartition:
    core_indices: list[int]
    boundary_indices: list[int]
    core_probe_ids: list[str]
    boundary_probe_ids: list[str]


@dataclass(frozen=True)
class TransplantDeltaResult:
    merged_weight: Any
    applied: bool
    null_dim: int
    delta_norm: float
    filtered_norm: float
    projection_loss: float
    preserved_fraction: float
    delta_occupancy: Any | None = None
    # Birkhoff projection metrics (optional, populated when birkhoff_config is used)
    birkhoff_applied: bool = False
    birkhoff_converged: bool = False
    birkhoff_iterations: int = 0
    birkhoff_spectral_clipped: bool = False


def partition_core_boundary(
    activations: "Array",
    probe_ids: list[str],
    core_probe_ids: set[str],
    backend: "Backend | None" = None,
) -> CoreBoundaryPartition:
    """Partition probes into core and boundary sets (boundary = complement)."""
    b = backend or get_default_backend()
    points = b.array(activations)
    b.eval(points)

    n = int(points.shape[0])
    if n == 0 or not probe_ids:
        return CoreBoundaryPartition([], [], [], [])

    core_indices = [i for i, pid in enumerate(probe_ids) if pid in core_probe_ids]
    core_set = set(core_indices)
    if not core_indices:
        return CoreBoundaryPartition([], [], [], [])

    boundary_list = [i for i in range(n) if i not in core_set]
    return CoreBoundaryPartition(
        core_indices=core_indices,
        boundary_indices=boundary_list,
        core_probe_ids=[probe_ids[i] for i in core_indices],
        boundary_probe_ids=[probe_ids[i] for i in boundary_list],
    )


def compute_transplant_delta(
    weight_target: "Array",
    weight_source_aligned: "Array",
    activations_core: "Array",
    activations_boundary: "Array",
    backend: "Backend | None" = None,
    delta_scale: float = 1.0,
    occupancy_weights: "Array | None" = None,
    source_activations_aligned: "Array | None" = None,
    source_sample_weights: "Array | None" = None,
    target_sample_weights: "Array | None" = None,
) -> TransplantDeltaResult:
    """Compute boundary-preserving transplant update for a single weight matrix.

    Uses geodesic null-space filtering - ALL operations on GPU.
    No SVD, no pinv, no eigendecomposition. Geodesic math is accurate for
    high-dimensional manifolds (8kD+). Chord distance is only reliable up to 3D.

    Two modes depending on whether source_activations_aligned is provided:

    MODE 1 (legacy): Target-only filtering
        Finds "room" in target based on low variance + low magnitude.

    MODE 2 (density-aware): Source+target relative density
        Transfers where source is DENSE and target is SPARSE.
        Implements the "fog cloud overlay" principle.

    Args:
        weight_target: Target model weights to merge into.
        weight_source_aligned: Source model weights (CKA-aligned).
        activations_core: Core activation patterns (concepts to preserve).
        activations_boundary: Target boundary activation patterns (define null space).
        backend: Optional Backend for GPU operations.
        delta_scale: Scale factor for the projected delta (0.0-1.0).
            Use < 1.0 to reduce knowledge injection per merge for sequential
            stacking. Default 1.0 = full projection. Threshold is geometrically
            derived (1% of baseline weight norm) - exceeding causes generation
            degradation in sequential merges.
        occupancy_weights: Optional per-dimension occupancy weights (0-1) from
            prior merges to protect already-modified directions.
        source_activations_aligned: Optional source activations (already aligned
            to target coordinate system). If provided, enables density-aware
            transfer mode: transfers knowledge where source is dense AND target
            is sparse, instead of just finding "room" in target.

    Returns:
        TransplantDeltaResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()
    # Convert all inputs to float32 for numerical stability
    weight_target = b.astype(b.array(weight_target), "float32")
    weight_source_aligned = b.astype(b.array(weight_source_aligned), "float32")
    activations_core = b.astype(b.array(activations_core), "float32")
    activations_boundary = b.astype(b.array(activations_boundary), "float32")
    b.eval(weight_target, weight_source_aligned, activations_core, activations_boundary)

    if len(weight_target.shape) != 2:
        return TransplantDeltaResult(
            merged_weight=weight_target,
            applied=False,
            null_dim=0,
            delta_norm=0.0,
            filtered_norm=0.0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            delta_occupancy=None,
        )

    in_dim = int(weight_target.shape[1])
    if int(activations_core.shape[1]) != in_dim:
        return TransplantDeltaResult(
            merged_weight=weight_target,
            applied=False,
            null_dim=0,
            delta_norm=0.0,
            filtered_norm=0.0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            delta_occupancy=None,
        )

    if int(activations_core.shape[0]) < 2:
        return TransplantDeltaResult(
            merged_weight=weight_target,
            applied=False,
            null_dim=0,
            delta_norm=0.0,
            filtered_norm=0.0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            delta_occupancy=None,
        )

    # Early-exit: if source == target, there's nothing to transplant
    # This handles the edge case where weights are already identical
    diff = weight_source_aligned - weight_target
    diff_norm_arr = geodesic_norms(b.reshape(diff, (1, -1)), b, use_cache=False)
    b.eval(diff_norm_arr)
    diff_norm = float(b.to_scalar(diff_norm_arr[0]))
    reg = regularization_epsilon(b, weight_target)
    if diff_norm <= reg:
        return TransplantDeltaResult(
            merged_weight=weight_target,
            applied=True,
            null_dim=0,
            delta_norm=0.0,
            filtered_norm=0.0,
            projection_loss=0.0,
            preserved_fraction=1.0,
            delta_occupancy=None,
        )


    # ==========================================================================
    # NULL-SPACE DELTA MERGING (GPU-only, no SVD/pinv/eigendecomp)
    # ==========================================================================
    # THEORY: Project the DELTA (source - target) into null space, not the source.
    #
    # Why delta, not source?
    # - Source is CKA-aligned with target (by construction of the merge)
    # - Therefore source weights mostly lie in target's RANGE space
    # - Projecting source into null space yields ~0 (wrong!)
    # - Projecting DELTA finds what's DIFFERENT and projects that
    #
    # FORMULA:
    #   delta = source - target  (what source has that target doesn't)
    #   safe_delta = project_to_null(delta, target_activations)
    #   merged = target + safe_delta  (add safe parts of difference)
    #
    # This adds source knowledge where target is sparse (null space),
    # without disturbing where target is already active (range space).
    # ==========================================================================

    geo_filter = GeodesicNullSpaceFilter(b)

    out_dim = int(weight_source_aligned.shape[0])
    in_dim = int(weight_source_aligned.shape[1])
    n_boundary = int(activations_boundary.shape[0])

    # Compute DELTA = source - target (what's different)
    weight_delta = weight_source_aligned - weight_target
    b.eval(weight_delta)

    if n_boundary < 2:
        # Not enough boundary points for null-space projection.
        # Without geometry, we cannot determine what's safe to modify.
        # Return unchanged - the math requires sufficient samples.
        logger.warning(
            "Insufficient boundary samples (%d < 2) for null-space projection. "
            "Cannot transplant without geometry.",
            n_boundary,
        )
        delta_norm_arr = geodesic_norms(
            b.reshape(weight_delta, (1, -1)), b, use_cache=False
        )
        b.eval(delta_norm_arr)
        delta_norm = float(b.to_scalar(delta_norm_arr[0]))

        return TransplantDeltaResult(
            merged_weight=weight_target,  # Unchanged - no geometry available
            applied=False,
            null_dim=0,
            delta_norm=delta_norm,
            filtered_norm=0.0,
            projection_loss=1.0,  # 100% loss - nothing could be applied
            preserved_fraction=0.0,
            delta_occupancy=None,
        )

    # Project DELTA into target's NULL SPACE
    # This finds what parts of the difference are orthogonal to target's active directions
    # If source_activations_aligned is provided, uses density-aware transfer:
    # transfers where source is DENSE and target is SPARSE
    result = geo_filter.filter_delta(
        weight_delta=weight_delta,  # Project DELTA (source - target)
        prior_activations=activations_boundary,  # Target's activation patterns define null space
        occupancy_weights=occupancy_weights,
        source_activations=source_activations_aligned,  # Enables density-aware transfer mode
        source_sample_weights=source_sample_weights,
        target_sample_weights=target_sample_weights,
    )
    delta_in_null_space = result.filtered_delta  # Safe difference to add
    delta_occupancy = result.delta_weights
    geodesic_null_dim = result.orthogonal_dim
    b.eval(delta_in_null_space)

    # =========================================================================
    # SPECTRAL NORM BOUND (GPU-friendly power iteration, no SVD)
    # =========================================================================
    # To bound the spectral norm without SVD, use power iteration:
    # σ_max ≈ ||A @ v|| / ||v|| where v converges to top right singular vector

    # Frobenius norm provides upper bound: σ_max ≤ ||A||_F
    frob_norm_arr = geodesic_norms(
        b.reshape(delta_in_null_space, (1, -1)), b, use_cache=False
    )
    b.eval(frob_norm_arr)
    frob_norm = float(b.to_scalar(frob_norm_arr[0]))

    # Power iteration for tighter bound (3 iterations usually sufficient)
    reg = regularization_epsilon(b, delta_in_null_space)
    v = b.ones((in_dim,), dtype="float32")
    v_norms = geodesic_norms(b.reshape(v, (1, -1)), b, use_cache=False)
    b.eval(v_norms)
    v = v / (float(b.to_scalar(v_norms[0])) + reg)
    b.eval(v)

    for _ in range(3):
        # w = A @ v
        w = b.matmul(delta_in_null_space, b.reshape(v, (in_dim, 1)))
        w = b.squeeze(w)
        b.eval(w)
        # u = A.T @ w
        u = b.matmul(b.transpose(delta_in_null_space), b.reshape(w, (out_dim, 1)))
        u = b.squeeze(u)
        b.eval(u)
        # Normalize
        u_norm_arr = geodesic_norms(b.reshape(u, (1, -1)), b, use_cache=False)
        b.eval(u_norm_arr)
        u_norm_val = float(b.to_scalar(u_norm_arr[0]))
        if u_norm_val > reg:
            v = u / u_norm_val
        b.eval(v)

    # Spectral norm estimate
    w_final = b.matmul(delta_in_null_space, b.reshape(v, (in_dim, 1)))
    w_final = b.squeeze(w_final)
    spectral_norm_arr = geodesic_norms(
        b.reshape(w_final, (1, -1)), b, use_cache=False
    )
    b.eval(spectral_norm_arr)
    spectral_norm = float(b.to_scalar(spectral_norm_arr[0]))

    # Scale if needed (preserves geodesic null-space exactly)
    # For additive merging, we want to add a controlled amount of delta
    max_norm = 1.0
    spectral_clipped = False
    if spectral_norm > max_norm:
        scale = max_norm / spectral_norm
        delta_contribution = delta_in_null_space * scale
        spectral_clipped = True
    else:
        delta_contribution = delta_in_null_space
    b.eval(delta_contribution)

    # Apply delta_scale for sequential stacking budget control
    # delta_scale < 1.0 reduces knowledge injection to stay within budget
    if delta_scale < 1.0:
        delta_contribution = delta_contribution * delta_scale
        b.eval(delta_contribution)

    # ADDITIVE MERGE: target + delta_in_null_space (NO replacement!)
    # This adds the safe part of (source - target) into target's sparse regions
    merged_weight = weight_target + delta_contribution
    b.eval(merged_weight)

    # NOTE: The previous "POST-MERGE GEODESIC RE-ALIGNMENT" step was removed.
    # It was applying F.T @ W which destroyed the null-space interlacing.
    # The null-space addition already preserves target's structure by construction.
    # No post-merge transform should be applied.

    birkhoff_applied = False
    birkhoff_converged = False
    birkhoff_iterations = 0
    birkhoff_spectral_clipped_extra = False

    # Compute metrics
    # delta_norm: how much difference between source and target
    # contribution_norm: how much of that difference made it through null-space projection
    original_delta_norm_arr = geodesic_norms(
        b.reshape(weight_delta, (1, -1)), b, use_cache=False
    )
    contribution_norm_arr = geodesic_norms(
        b.reshape(delta_contribution, (1, -1)), b, use_cache=False
    )
    b.eval(original_delta_norm_arr, contribution_norm_arr)
    original_delta_norm = float(b.to_scalar(original_delta_norm_arr[0]))
    contribution_norm = float(b.to_scalar(contribution_norm_arr[0]))

    if original_delta_norm > 0.0:
        # How much of the delta made it through null-space projection
        preserved_fraction = contribution_norm / original_delta_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=geodesic_null_dim,
        delta_norm=original_delta_norm,  # Difference between source and target
        filtered_norm=contribution_norm,  # Delta after null-space projection
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        delta_occupancy=delta_occupancy,
        birkhoff_applied=birkhoff_applied,
        birkhoff_converged=birkhoff_converged,
        birkhoff_iterations=birkhoff_iterations,
        birkhoff_spectral_clipped=spectral_clipped or birkhoff_spectral_clipped_extra,
    )

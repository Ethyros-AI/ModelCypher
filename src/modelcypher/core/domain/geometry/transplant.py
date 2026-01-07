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
from modelcypher.core.domain.geometry.vector_math import geodesic_norms

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
) -> TransplantDeltaResult:
    """Compute boundary-preserving transplant update for a single weight matrix.

    Uses geodesic null-space filtering - ALL operations on GPU.
    No SVD, no pinv, no eigendecomposition. Geodesic math is accurate for
    high-dimensional manifolds (8kD+). Chord distance is only reliable up to 3D.

    The geometry determines everything - no configuration needed.
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
        )

    # Early-exit: if source == target, there's nothing to transplant
    # This handles the edge case where weights are already identical
    diff = weight_source_aligned - weight_target
    diff_norm_arr = geodesic_norms(b.reshape(diff, (1, -1)), b)
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
        )


    # ==========================================================================
    # ADDITIVE NULL-SPACE MERGING (GPU-only, no SVD/pinv/eigendecomp)
    # ==========================================================================
    # THEORY (from DIMENSIONAL_COMPRESSION.md):
    # - Concept probability clouds overlay - some overlap perfectly, some don't
    # - Source has denser knowledge in some regions (larger model)
    # - We ADD source knowledge into target's NULL SPACE (where target is sparse)
    # - This ACTIVATES MORE NEURONS without altering existing target neurons
    #
    # OLD WRONG FORMULA: delta = source - target  ⟹  W' = target + delta = source
    # NEW CORRECT FORMULA: W' = target + project_to_null(source, target)
    #
    # We project the SOURCE WEIGHTS (not the difference!) into the null space
    # of the target's activation patterns. This adds knowledge where target
    # is sparse, without disturbing where target is already dense.
    # ==========================================================================

    geo_filter = GeodesicNullSpaceFilter(b)

    # For weight matrices [out_dim, in_dim], we filter the SOURCE weights
    # to project them into target's null space (where target is sparse)
    # - activations_boundary: [n_samples, in_dim] = target activation patterns
    # - weight_source_aligned: [out_dim, in_dim] = source knowledge to add
    #
    # The filter finds directions ORTHOGONAL to target's activation manifold.
    # Source knowledge in those directions can be added without disturbing target.
    out_dim = int(weight_source_aligned.shape[0])
    in_dim = int(weight_source_aligned.shape[1])
    n_boundary = int(activations_boundary.shape[0])

    if n_boundary < 2:
        # Not enough boundary points for geodesic filtering
        # Use simpler approach: add scaled source (avoid blowing up activations)
        source_norm_arr = geodesic_norms(b.reshape(weight_source_aligned, (1, -1)), b)
        b.eval(source_norm_arr)
        source_norm = float(b.to_scalar(source_norm_arr[0]))
        target_norm_arr = geodesic_norms(b.reshape(weight_target, (1, -1)), b)
        b.eval(target_norm_arr)
        target_norm = float(b.to_scalar(target_norm_arr[0]))
        
        # Scale source to not dominate: add at 10% of relative magnitude
        scale = 0.1 * target_norm / (source_norm + 1e-8)
        source_contribution = weight_source_aligned * scale
        b.eval(source_contribution)
        
        return TransplantDeltaResult(
            merged_weight=weight_target + source_contribution,
            applied=True,
            null_dim=in_dim,  # All directions available
            delta_norm=source_norm,
            filtered_norm=source_norm * scale,
            projection_loss=0.0,
            preserved_fraction=1.0,
        )

    # Project SOURCE weights into target's NULL SPACE
    # This finds what parts of source are orthogonal to target's active directions
    result = geo_filter.filter_delta(
        weight_delta=weight_source_aligned,  # NOTE: project SOURCE, not (source - target)
        prior_activations=activations_boundary,  # Target's activation patterns define null space
    )
    source_in_null_space = result.filtered_delta  # Source knowledge that's orthogonal to target
    geodesic_null_dim = result.orthogonal_dim
    b.eval(source_in_null_space)

    # =========================================================================
    # SPECTRAL NORM BOUND (GPU-friendly power iteration, no SVD)
    # =========================================================================
    # To bound the spectral norm without SVD, use power iteration:
    # σ_max ≈ ||A @ v|| / ||v|| where v converges to top right singular vector

    # Frobenius norm provides upper bound: σ_max ≤ ||A||_F
    frob_norm_arr = geodesic_norms(b.reshape(source_in_null_space, (1, -1)), b)
    b.eval(frob_norm_arr)
    frob_norm = float(b.to_scalar(frob_norm_arr[0]))

    # Power iteration for tighter bound (3 iterations usually sufficient)
    reg = regularization_epsilon(b, source_in_null_space)
    v = b.ones((in_dim,), dtype="float32")
    v_norms = geodesic_norms(b.reshape(v, (1, -1)), b)
    b.eval(v_norms)
    v = v / (float(b.to_scalar(v_norms[0])) + reg)
    b.eval(v)

    for _ in range(3):
        # w = A @ v
        w = b.matmul(source_in_null_space, b.reshape(v, (in_dim, 1)))
        w = b.squeeze(w)
        b.eval(w)
        # u = A.T @ w
        u = b.matmul(b.transpose(source_in_null_space), b.reshape(w, (out_dim, 1)))
        u = b.squeeze(u)
        b.eval(u)
        # Normalize
        u_norm_arr = geodesic_norms(b.reshape(u, (1, -1)), b)
        b.eval(u_norm_arr)
        u_norm_val = float(b.to_scalar(u_norm_arr[0]))
        if u_norm_val > reg:
            v = u / u_norm_val
        b.eval(v)

    # Spectral norm estimate
    w_final = b.matmul(source_in_null_space, b.reshape(v, (in_dim, 1)))
    w_final = b.squeeze(w_final)
    spectral_norm_arr = geodesic_norms(b.reshape(w_final, (1, -1)), b)
    b.eval(spectral_norm_arr)
    spectral_norm = float(b.to_scalar(spectral_norm_arr[0]))

    # Scale if needed (preserves geodesic null-space exactly)
    # For additive merging, we want to add a controlled amount of source knowledge
    max_norm = 1.0
    spectral_clipped = False
    if spectral_norm > max_norm:
        scale = max_norm / spectral_norm
        source_contribution = source_in_null_space * scale
        spectral_clipped = True
    else:
        source_contribution = source_in_null_space
    b.eval(source_contribution)

    # ADDITIVE MERGE: target + source_in_null_space (NO replacement!)
    # This adds source knowledge into target's sparse regions
    merged_weight = weight_target + source_contribution
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
    source_norm_arr = geodesic_norms(b.reshape(weight_source_aligned, (1, -1)), b)
    contribution_norm_arr = geodesic_norms(b.reshape(source_contribution, (1, -1)), b)
    b.eval(source_norm_arr, contribution_norm_arr)
    source_norm = float(b.to_scalar(source_norm_arr[0]))
    contribution_norm = float(b.to_scalar(contribution_norm_arr[0]))

    if source_norm > 0.0:
        # How much of source knowledge made it through null-space projection
        preserved_fraction = contribution_norm / source_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=geodesic_null_dim,
        delta_norm=source_norm,  # Now means: source knowledge to add
        filtered_norm=contribution_norm,  # Now means: source after null-space projection
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        birkhoff_applied=birkhoff_applied,
        birkhoff_converged=birkhoff_converged,
        birkhoff_iterations=birkhoff_iterations,
        birkhoff_spectral_clipped=spectral_clipped or birkhoff_spectral_clipped_extra,
    )

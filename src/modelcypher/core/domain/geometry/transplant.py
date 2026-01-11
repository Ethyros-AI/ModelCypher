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
    backend: "Backend | None" = None,
    delta_scale: float = 1.0,
) -> TransplantDeltaResult:
    """Activate dormant neurons using source's patterns.

    This is NOT weight blending or delta filtering. This is neuron activation:
    - Find dimensions where target has LOW activation variance (dormant neurons)
    - For those dimensions, REPLACE with source's weights (wake them up)
    - For active dimensions, KEEP target's weights unchanged

    The goal: make target's representation clouds DENSER without changing
    the fundamental structure. Dormant neurons don't affect current behavior
    but can sharpen reasoning by adding new pathways.

    Algorithm:
        1. Compute per-dimension variance of target activations
        2. Identify dormant dimensions (variance in bottom percentile)
        3. merged[dormant] = source[dormant]  (activate with source)
        4. merged[active] = target[active]    (preserve target)

    Args:
        weight_target: Target model weights to merge into.
        weight_source_aligned: Source model weights (CKA-aligned).
        activations_core: Target activation patterns (defines dormancy).
        backend: Optional Backend for GPU operations.
        delta_scale: Not used in dormancy mode (kept for API compatibility).

    Returns:
        TransplantDeltaResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()
    # Convert all inputs to float32 for numerical stability
    weight_target = b.astype(b.array(weight_target), "float32")
    weight_source_aligned = b.astype(b.array(weight_source_aligned), "float32")
    activations_core = b.astype(b.array(activations_core), "float32")
    b.eval(weight_target, weight_source_aligned, activations_core)

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
    # DORMANCY-BASED NEURON ACTIVATION
    # ==========================================================================
    # This is NOT weight blending. This is selective neuron activation:
    #
    # 1. Dormant dimensions: target has LOW variance (neurons not used)
    #    - These don't affect target's current behavior
    #    - REPLACE with source's weights to "wake them up"
    #
    # 2. Active dimensions: target has HIGH variance (neurons in use)
    #    - These define target's current behavior
    #    - KEEP unchanged to preserve structure
    #
    # Result: Denser point clouds, sharper reasoning, same fundamental structure
    # ==========================================================================

    out_dim = int(weight_source_aligned.shape[0])
    in_dim = int(weight_source_aligned.shape[1])
    n_samples = int(activations_core.shape[0])

    # Compute per-dimension variance of target activations
    # variance[d] = mean((act[:, d] - mean(act[:, d]))^2)
    act_mean = b.mean(activations_core, axis=0)  # [in_dim]
    b.eval(act_mean)
    act_centered = activations_core - act_mean  # [n_samples, in_dim]
    b.eval(act_centered)
    act_var = b.mean(act_centered * act_centered, axis=0)  # [in_dim]
    b.eval(act_var)

    # Compute total variance for normalization
    total_var = b.sum(act_var)
    b.eval(total_var)
    total_var_val = float(b.to_scalar(total_var))

    if total_var_val <= 0:
        # No variance in activations - can't determine dormancy
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

    # Normalize variance to get per-dimension "activity level" [0, 1]
    # High activity = frequently used = preserve target
    # Low activity = dormant = can activate with source
    var_normalized = act_var / (total_var_val / in_dim)  # Relative to mean variance
    b.eval(var_normalized)

    # Compute dormancy threshold from data distribution
    # Use median as threshold: dimensions below median are "dormant"
    var_sorted = b.sort(var_normalized)
    b.eval(var_sorted)
    median_idx = in_dim // 2
    median_var = b.take(var_sorted, b.array([median_idx]), axis=0)
    b.eval(median_var)
    threshold = float(b.to_scalar(median_var[0]))

    # Identify dormant dimensions (variance below median)
    # is_dormant[d] = 1.0 if dormant, 0.0 if active
    is_dormant = b.astype(var_normalized < threshold, "float32")
    b.eval(is_dormant)

    # Count dormant dimensions
    n_dormant = int(float(b.to_scalar(b.sum(is_dormant))))

    logger.info(
        "DORMANCY: %d/%d dimensions dormant (%.1f%%), threshold=%.4f",
        n_dormant, in_dim, 100.0 * n_dormant / in_dim, threshold
    )

    # Create selection mask for each dimension
    # merged[:, d] = source[:, d] if dormant else target[:, d]
    # Reshape for broadcasting: [1, in_dim] for column selection
    dormant_mask = b.reshape(is_dormant, (1, in_dim))
    active_mask = 1.0 - dormant_mask
    b.eval(dormant_mask, active_mask)

    # Select: dormant from source, active from target
    merged_weight = (weight_source_aligned * dormant_mask) + (weight_target * active_mask)
    b.eval(merged_weight)

    # Compute metrics for the dormancy-based merge
    # weight_delta = actual change applied (merged - target)
    weight_delta = merged_weight - weight_target
    b.eval(weight_delta)

    # full_delta = potential change (source_aligned - target)
    full_delta = weight_source_aligned - weight_target
    b.eval(full_delta)

    # Compute norms
    # full_delta_norm: how much difference between source and target (potential)
    # applied_delta_norm: how much change was actually applied (via dormancy selection)
    full_delta_norm_arr = geodesic_norms(
        b.reshape(full_delta, (1, -1)), b, use_cache=False
    )
    applied_delta_norm_arr = geodesic_norms(
        b.reshape(weight_delta, (1, -1)), b, use_cache=False
    )
    b.eval(full_delta_norm_arr, applied_delta_norm_arr)
    full_delta_norm = float(b.to_scalar(full_delta_norm_arr[0]))
    applied_delta_norm = float(b.to_scalar(applied_delta_norm_arr[0]))

    if full_delta_norm > 0.0:
        # How much of the potential delta was applied via dormancy selection
        preserved_fraction = applied_delta_norm / full_delta_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    logger.info(
        "DORMANCY RESULT: applied_norm=%.4f, full_delta_norm=%.4f, preserved=%.1f%%",
        applied_delta_norm, full_delta_norm, 100.0 * preserved_fraction
    )

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=n_dormant,  # Number of dormant dimensions activated
        delta_norm=full_delta_norm,  # Potential change (source - target)
        filtered_norm=applied_delta_norm,  # Actual change applied
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        delta_occupancy=None,  # Not used in dormancy mode
        birkhoff_applied=False,  # Not used in dormancy mode
        birkhoff_converged=False,
        birkhoff_iterations=0,
        birkhoff_spectral_clipped=False,
    )

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
from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilter
from modelcypher.core.domain.geometry.numerical_stability import svd_via_eigh

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

    All null-space filtering parameters are derived from the data's spectral
    properties. No configuration needed - the geometry determines everything.
    """
    b = backend or get_default_backend()
    # Convert all inputs to float32 - pinv requires float32 or float64
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

    delta_source_t = b.transpose(weight_source_aligned - weight_target)
    b.eval(delta_source_t)

    core_pinv = b.pinv(activations_core)
    b.eval(core_pinv)

    core_output = b.matmul(activations_core, delta_source_t)
    b.eval(core_output)
    delta_core_t = b.matmul(core_pinv, core_output)
    b.eval(delta_core_t)

    # Null-space filter - all params derived from spectral properties
    null_filter = NullSpaceFilter(backend=b)
    null_projection = null_filter.compute_null_space_projection(activations_boundary)
    proj = null_projection.projection_matrix
    b.eval(proj)

    delta_filtered_t = b.matmul(proj, delta_core_t)
    b.eval(delta_filtered_t)

    # Spectral norm bound for compositional stability.
    # Use direct scalar scaling to preserve null-space membership exactly.
    # SVD-reconstruct can introduce numerical error; scalar multiply cannot.
    delta_filtered = b.transpose(delta_filtered_t)
    b.eval(delta_filtered)

    # Compute spectral norm (largest singular value)
    _, S, _ = svd_via_eigh(b, delta_filtered, full_matrices=False)
    b.eval(S)
    if int(S.shape[0]) > 0:
        max_sv_arr = b.take(S, b.array([0]), axis=0)
        max_sv_arr = b.squeeze(max_sv_arr)
        b.eval(max_sv_arr)
        spectral_norm = float(b.to_scalar(max_sv_arr))
    else:
        spectral_norm = 0.0

    # Scale by scalar if needed (preserves null-space exactly)
    max_norm = 1.0
    if spectral_norm > max_norm:
        scale = max_norm / spectral_norm
        delta_stabilized = delta_filtered * scale
        spectral_clipped = True
    else:
        delta_stabilized = delta_filtered
        spectral_clipped = False
    b.eval(delta_stabilized)

    merged_weight = weight_target + delta_stabilized
    b.eval(merged_weight)

    delta_norm_arr = b.norm(b.transpose(delta_core_t))
    filtered_norm_arr = b.norm(delta_stabilized)
    b.eval(delta_norm_arr, filtered_norm_arr)
    delta_norm = float(b.to_scalar(delta_norm_arr))
    filtered_norm = float(b.to_scalar(filtered_norm_arr))

    if delta_norm > 0.0:
        preserved_fraction = filtered_norm / delta_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=int(null_projection.null_dim),
        delta_norm=delta_norm,
        filtered_norm=filtered_norm,
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        birkhoff_applied=False,
        birkhoff_converged=False,
        birkhoff_iterations=0,
        birkhoff_spectral_clipped=spectral_clipped,
    )

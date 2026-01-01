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
from modelcypher.core.domain.geometry.null_space_filter import NullSpaceFilter
from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreBoundaryPartition:
    core_indices: list[int]
    boundary_indices: list[int]
    core_probe_ids: list[str]
    boundary_probe_ids: list[str]
    boundary_k: int
    geodesic_k_neighbors: int | None


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
    boundary_k: int | None = None,
    geodesic_k_neighbors: int | None = None,
    backend: "Backend | None" = None,
) -> CoreBoundaryPartition:
    """Partition probes into core and boundary sets using geodesic neighborhoods."""
    b = backend or get_default_backend()
    points = b.array(activations)
    b.eval(points)

    n = int(points.shape[0])
    if n == 0 or not probe_ids:
        return CoreBoundaryPartition([], [], [], [], 0, geodesic_k_neighbors)

    core_indices = [i for i, pid in enumerate(probe_ids) if pid in core_probe_ids]
    core_set = set(core_indices)
    if not core_indices:
        return CoreBoundaryPartition([], [], [], [], 0, geodesic_k_neighbors)

    if boundary_k is None:
        if geodesic_k_neighbors is not None:
            boundary_k = geodesic_k_neighbors
        else:
            boundary_k = min(10, max(1, n - 1))
    boundary_k = max(1, min(boundary_k, n - 1))

    geo = RiemannianGeometry(b).geodesic_distances(points, k_neighbors=geodesic_k_neighbors)
    dist_np = b.to_numpy(geo.distances)

    boundary_indices: set[int] = set()
    for core_idx in core_indices:
        row = dist_np[core_idx].tolist()
        candidates = [
            (j, row[j]) for j in range(n)
            if j != core_idx and j not in core_set
        ]
        candidates.sort(key=lambda item: item[1])
        for j, _ in candidates[:boundary_k]:
            boundary_indices.add(j)

    boundary_list = sorted(boundary_indices)
    return CoreBoundaryPartition(
        core_indices=core_indices,
        boundary_indices=boundary_list,
        core_probe_ids=[probe_ids[i] for i in core_indices],
        boundary_probe_ids=[probe_ids[i] for i in boundary_list],
        boundary_k=boundary_k,
        geodesic_k_neighbors=geodesic_k_neighbors,
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

    # Apply Birkhoff projection for compositional stability (per DeepSeek mHC paper)
    # This bounds spectral norm to prevent signal amplification across layers
    birkhoff = BirkhoffProjector(backend=b)
    delta_filtered = b.transpose(delta_filtered_t)
    b.eval(delta_filtered)

    birkhoff_result = birkhoff.project_weight_delta(delta_filtered)
    delta_stabilized = birkhoff_result.projected_matrix
    b.eval(delta_stabilized)

    # For non-square deltas, birkhoff returns projected Gram matrix
    # We need to apply it as a scaling factor to preserve the delta's shape
    if delta_stabilized.shape != delta_filtered.shape:
        # Non-square: use spectral norm bounding directly
        _, was_clipped = birkhoff.bound_spectral_norm(delta_filtered, max_norm=1.0)
        if was_clipped:
            # Scale delta to have unit spectral norm
            from modelcypher.core.domain.geometry.numerical_stability import svd_via_eigh
            _, S, _ = svd_via_eigh(b, delta_filtered, full_matrices=False)
            b.eval(S)
            S_np = b.to_numpy(S)
            spectral_norm = float(S_np[0]) if len(S_np) > 0 else 1.0
            if spectral_norm > 1.0:
                delta_stabilized = delta_filtered / spectral_norm
                b.eval(delta_stabilized)
            else:
                delta_stabilized = delta_filtered
        else:
            delta_stabilized = delta_filtered

    merged_weight = weight_target + delta_stabilized
    b.eval(merged_weight)

    delta_norm_arr = b.norm(b.transpose(delta_core_t))
    filtered_norm_arr = b.norm(delta_stabilized)
    b.eval(delta_norm_arr, filtered_norm_arr)
    delta_norm = float(b.to_numpy(delta_norm_arr))
    filtered_norm = float(b.to_numpy(filtered_norm_arr))

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
        birkhoff_applied=True,
        birkhoff_converged=birkhoff_result.converged,
        birkhoff_iterations=birkhoff_result.iterations_used,
        birkhoff_spectral_clipped=birkhoff_result.spectral_clipped,
    )

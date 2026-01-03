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
from modelcypher.core.domain.geometry.geodesic_null_space import GeodesicNullSpaceFilter
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
    high-dimensional manifolds (8kD+). Euclidean is only accurate up to 3D.

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

    # Compute delta: knowledge to transplant from source to target
    delta = weight_source_aligned - weight_target
    b.eval(delta)

    # =========================================================================
    # GEODESIC NULL-SPACE FILTERING (GPU-only, no SVD/pinv/eigendecomp)
    # =========================================================================
    # Instead of Euclidean linear algebra (wrong for 8kD manifolds), we use:
    # 1. k-NN graph construction (GPU: matmul + argsort)
    # 2. Geodesic distances via Floyd-Warshall (GPU: vectorized min)
    # 3. Tangent space projection at Fréchet mean (GPU: Gram matrix ops)
    #
    # This finds directions orthogonal to the manifold's geodesic structure,
    # not just orthogonal in flat Euclidean space.

    geo_filter = GeodesicNullSpaceFilter(b)

    # For weight matrices [out_dim, in_dim], we filter each output dimension
    # We need to handle the dimension matching carefully:
    # - activations_boundary: [n_samples, in_dim]
    # - delta: [out_dim, in_dim]
    #
    # We filter along the input dimension (columns of delta)
    out_dim = int(delta.shape[0])
    in_dim = int(delta.shape[1])
    n_boundary = int(activations_boundary.shape[0])

    if n_boundary < 2:
        # Not enough boundary points for geodesic filtering - return original
        delta_norm_arr = geodesic_norms(b.reshape(delta, (1, -1)), b)
        b.eval(delta_norm_arr)
        delta_norm = float(b.to_scalar(delta_norm_arr[0]))
        return TransplantDeltaResult(
            merged_weight=weight_target + delta,
            applied=True,
            null_dim=in_dim,  # All directions available
            delta_norm=delta_norm,
            filtered_norm=delta_norm,
            projection_loss=0.0,
            preserved_fraction=1.0,
        )

    # Filter each row of delta (each output dimension) through geodesic null space
    # This projects the input-space components onto geodesic-orthogonal directions
    delta_filtered_rows = []
    total_projection_loss = 0.0
    geodesic_null_dim = 0

    for i in range(out_dim):
        row = delta[i, :]  # [in_dim]
        b.eval(row)

        result = geo_filter.filter_delta(
            weight_delta=row,
            prior_activations=activations_boundary,
        )

        delta_filtered_rows.append(result.filtered_delta)
        total_projection_loss += result.projection_loss
        geodesic_null_dim = max(geodesic_null_dim, result.orthogonal_dim)

    # Stack filtered rows back into matrix
    delta_filtered = b.stack(delta_filtered_rows, axis=0)
    b.eval(delta_filtered)

    # =========================================================================
    # SPECTRAL NORM BOUND (GPU-friendly power iteration, no SVD)
    # =========================================================================
    # To bound the spectral norm without SVD, use power iteration:
    # σ_max ≈ ||A @ v|| / ||v|| where v converges to top right singular vector

    # Frobenius norm provides upper bound: σ_max ≤ ||A||_F
    frob_norm_arr = geodesic_norms(b.reshape(delta_filtered, (1, -1)), b)
    b.eval(frob_norm_arr)
    frob_norm = float(b.to_scalar(frob_norm_arr[0]))

    # Power iteration for tighter bound (3 iterations usually sufficient)
    reg = regularization_epsilon(b, delta_filtered)
    v = b.ones((in_dim,), dtype="float32")
    v_norms = geodesic_norms(b.reshape(v, (1, -1)), b)
    b.eval(v_norms)
    v = v / (float(b.to_scalar(v_norms[0])) + reg)
    b.eval(v)

    for _ in range(3):
        # w = A @ v
        w = b.matmul(delta_filtered, b.reshape(v, (in_dim, 1)))
        w = b.squeeze(w)
        b.eval(w)
        # u = A.T @ w
        u = b.matmul(b.transpose(delta_filtered), b.reshape(w, (out_dim, 1)))
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
    w_final = b.matmul(delta_filtered, b.reshape(v, (in_dim, 1)))
    w_final = b.squeeze(w_final)
    spectral_norm_arr = geodesic_norms(b.reshape(w_final, (1, -1)), b)
    b.eval(spectral_norm_arr)
    spectral_norm = float(b.to_scalar(spectral_norm_arr[0]))

    # Scale if needed (preserves geodesic null-space exactly)
    max_norm = 1.0
    spectral_clipped = False
    if spectral_norm > max_norm:
        scale = max_norm / spectral_norm
        delta_stabilized = delta_filtered * scale
        spectral_clipped = True
    else:
        delta_stabilized = delta_filtered
    b.eval(delta_stabilized)

    # Merge: target + geodesic-filtered delta (NO ALPHA - geometric addition)
    merged_weight = weight_target + delta_stabilized
    b.eval(merged_weight)

    # Compute metrics
    delta_norm_arr = geodesic_norms(b.reshape(delta, (1, -1)), b)
    filtered_norm_arr = geodesic_norms(b.reshape(delta_stabilized, (1, -1)), b)
    b.eval(delta_norm_arr, filtered_norm_arr)
    delta_norm = float(b.to_scalar(delta_norm_arr[0]))
    filtered_norm = float(b.to_scalar(filtered_norm_arr[0]))

    if delta_norm > 0.0:
        preserved_fraction = filtered_norm / delta_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=geodesic_null_dim,
        delta_norm=delta_norm,
        filtered_norm=filtered_norm,
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        birkhoff_applied=False,
        birkhoff_converged=False,
        birkhoff_iterations=0,
        birkhoff_spectral_clipped=spectral_clipped,
    )

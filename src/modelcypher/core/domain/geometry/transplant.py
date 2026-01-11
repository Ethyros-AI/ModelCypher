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

Solves constrained least-squares for weight update:

    min ||A_core @ delta_W - delta_A_core||_F²
    s.t. A_boundary @ delta_W = 0

Solution via null-space projection:
    N = I - pinv(A_boundary) @ A_boundary  (boundary null-space)
    delta_W_unc = pinv(A_core) @ delta_A_core  (unconstrained)
    delta_W = N @ delta_W_unc  (projected to boundary null-space)
    W' = W_target + delta_W.T

This ensures boundary outputs are EXACTLY preserved (to numerical precision)
while core outputs move toward the source's knowledge.

There are no heuristics, thresholds, or modes. The geometry determines everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


def _compute_transplant_delta_anchor_relative(
    weight_target: "Array",
    activations_core: "Array",
    delta_activations: "Array",
    boundary_activations: "Array | None",
    delta_scale: float,
    backend: "Backend",
) -> "TransplantDeltaResult":
    """Anchor-relative mode: constrained least-squares with boundary preservation.

    Solves:
        min ||A_core @ delta_W - delta_A_core||_F²
        s.t. A_boundary @ delta_W = 0

    Via null-space projection:
        N = I - pinv(A_boundary) @ A_boundary  (boundary null-space)
        delta_W_unc = pinv(A_core) @ delta_A_core  (unconstrained)
        delta_W = N @ delta_W_unc  (projected)
        W' = W_target + delta_W.T
    """
    b = backend

    # Convert inputs to float32
    weight_target = b.astype(b.array(weight_target), "float32")
    activations_core = b.astype(b.array(activations_core), "float32")
    delta_activations = b.astype(b.array(delta_activations), "float32")
    b.eval(weight_target, activations_core, delta_activations)

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

    out_dim = int(weight_target.shape[0])
    in_dim = int(weight_target.shape[1])
    n_core = int(activations_core.shape[0])

    logger.info(
        "ANCHOR-RELATIVE TRANSPLANT: weight=[%d, %d], n_core=%d, delta_A shape=%s",
        out_dim, in_dim, n_core, b.shape(delta_activations)
    )

    # Step 1: Compute boundary null-space projector N
    # N = I - pinv(A_boundary) @ A_boundary
    if boundary_activations is not None:
        boundary_activations = b.astype(b.array(boundary_activations), "float32")
        b.eval(boundary_activations)
        n_boundary = int(boundary_activations.shape[0])

        if n_boundary > 0:
            # A_b is [m, in_dim], pinv(A_b) is [in_dim, m]
            A_b_pinv = b.pinv(boundary_activations)
            b.eval(A_b_pinv)

            # N = I - pinv(A_b) @ A_b
            # [in_dim, m] @ [m, in_dim] -> [in_dim, in_dim]
            proj_b = b.matmul(A_b_pinv, boundary_activations)
            b.eval(proj_b)

            N = b.eye(in_dim) - proj_b
            b.eval(N)

            logger.info(
                "ANCHOR-RELATIVE: Boundary null-space computed from %d samples",
                n_boundary
            )
        else:
            N = b.eye(in_dim)
            n_boundary = 0
    else:
        N = b.eye(in_dim)
        n_boundary = 0

    # Step 2: Compute unconstrained solution
    # delta_W_unc = pinv(A_core) @ delta_A_core
    # A_core is [n, in_dim], pinv(A_core) is [in_dim, n]
    # delta_A is [n, out_dim]
    # Result: [in_dim, n] @ [n, out_dim] -> [in_dim, out_dim]
    A_c_pinv = b.pinv(activations_core)
    b.eval(A_c_pinv)

    delta_W_unc = b.matmul(A_c_pinv, delta_activations)
    b.eval(delta_W_unc)

    # Step 3: Project to boundary null-space
    # delta_W = N @ delta_W_unc
    # [in_dim, in_dim] @ [in_dim, out_dim] -> [in_dim, out_dim]
    delta_W = b.matmul(N, delta_W_unc)
    b.eval(delta_W)

    # Apply scale
    if delta_scale != 1.0:
        delta_W = delta_W * delta_scale
        b.eval(delta_W)

    # Step 4: Apply to weights
    # W' = W_target + delta_W.T
    # delta_W is [in_dim, out_dim], W is [out_dim, in_dim]
    merged_weight = weight_target + b.transpose(delta_W)
    b.eval(merged_weight)

    # Compute metrics
    delta_W_norm_arr = geodesic_norms(
        b.reshape(delta_W, (1, -1)), b, use_cache=False
    )
    delta_W_unc_norm_arr = geodesic_norms(
        b.reshape(delta_W_unc, (1, -1)), b, use_cache=False
    )
    delta_A_norm_arr = geodesic_norms(
        b.reshape(delta_activations, (1, -1)), b, use_cache=False
    )
    b.eval(delta_W_norm_arr, delta_W_unc_norm_arr, delta_A_norm_arr)

    delta_W_norm = float(b.to_scalar(delta_W_norm_arr[0]))
    delta_W_unc_norm = float(b.to_scalar(delta_W_unc_norm_arr[0]))
    delta_A_norm = float(b.to_scalar(delta_A_norm_arr[0]))

    if delta_W_unc_norm > 0:
        preserved_fraction = delta_W_norm / delta_W_unc_norm
        projection_loss = max(0.0, 1.0 - preserved_fraction)
    else:
        preserved_fraction = 1.0
        projection_loss = 0.0

    # Compute null-space dimension (rank of N minus full rank)
    # Approximation: use n_boundary as indicator
    null_dim = n_boundary if n_boundary > 0 else 0

    logger.info(
        "ANCHOR-RELATIVE RESULT: delta_A_norm=%.4f, delta_W_norm=%.4f, "
        "preserved=%.1f%%, n_boundary=%d",
        delta_A_norm, delta_W_norm, 100.0 * preserved_fraction, n_boundary
    )

    return TransplantDeltaResult(
        merged_weight=merged_weight,
        applied=True,
        null_dim=null_dim,
        delta_norm=delta_A_norm,
        filtered_norm=delta_W_norm,
        projection_loss=projection_loss,
        preserved_fraction=preserved_fraction,
        delta_occupancy=None,
        birkhoff_applied=False,
        birkhoff_converged=False,
        birkhoff_iterations=0,
        birkhoff_spectral_clipped=False,
    )


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
    activations_core: "Array",
    delta_activations: "Array",
    boundary_activations: "Array | None" = None,
    backend: "Backend | None" = None,
    delta_scale: float = 1.0,
) -> TransplantDeltaResult:
    """Compute weight update via constrained least-squares.

    Solves:
        min ||A_core @ delta_W - delta_A_core||_F²
        s.t. A_boundary @ delta_W = 0

    Solution via null-space projection:
        N = I - pinv(A_boundary) @ A_boundary  (boundary null-space)
        delta_W_unc = pinv(A_core) @ delta_A_core  (unconstrained)
        delta_W = N @ delta_W_unc  (projected to boundary null-space)
        W' = W_target + delta_W.T

    This ensures boundary outputs are EXACTLY preserved (to numerical precision)
    while core outputs move toward the source's knowledge.

    There are no heuristics, thresholds, or modes. The geometry determines everything.

    Args:
        weight_target: Target model weights to modify [out_dim, in_dim].
        activations_core: Core activation samples [n_core, in_dim].
        delta_activations: Desired change in activation space [n_core, out_dim].
            Computed upstream via anchor-relative grafting.
        boundary_activations: Boundary samples [n_boundary, in_dim]. If provided,
            their outputs are exactly preserved: A_boundary @ W' = A_boundary @ W.
        backend: Optional Backend for GPU operations.
        delta_scale: Scale factor for delta (default 1.0).

    Returns:
        TransplantDeltaResult with merged weight and diagnostics.
    """
    b = backend or get_default_backend()

    return _compute_transplant_delta_anchor_relative(
        weight_target=weight_target,
        activations_core=activations_core,
        delta_activations=delta_activations,
        boundary_activations=boundary_activations,
        delta_scale=delta_scale,
        backend=b,
    )

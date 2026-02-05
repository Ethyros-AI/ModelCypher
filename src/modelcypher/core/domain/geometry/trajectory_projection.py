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

"""Trajectory-tangent null-space projection.

Key insight: Not all null-space directions are equal. Knowledge should be
injected into null-space directions that are TANGENT to the trajectory -
"along the road the model already uses."

Velocities show WHERE the trajectory is heading. Null-space directions
aligned with velocities are the "shoulders of the road" - unused capacity
that's still reachable by the model's natural information flow.

This is the difference between:
- Building a town in the middle of nowhere (random null-space)
- Building a town along the highway (trajectory-tangent null-space)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision_float32 as _promote_precision,
)

from modelcypher.core.domain.geometry.null_space import (
    VarianceNullSpaceResult,
)
from modelcypher.core.domain.geometry.trajectory_analysis import (
    TrajectoryResult,
    TrajectorySubspaceResult,
    compute_trajectory_subspace,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryTangentResult:
    """Result of trajectory-tangent null-space computation.

    Contains both the full null space and the trajectory-tangent subspace
    within it - directions where knowledge can be injected "along the road."
    """

    U_null: "Array"  # Full null space [hidden_dim, null_rank]
    U_tangent: "Array"  # Trajectory-tangent null space [hidden_dim, tangent_rank]
    null_rank: int  # Rank of full null space
    tangent_rank: int  # Rank of tangent subspace (tangent_rank <= null_rank)
    velocity_alignment: float  # Mean alignment of tangent with velocities (0-1)
    hidden_dim: int


def compute_trajectory_tangent_null_space(
    trajectories: list[TrajectoryResult],
    backend: "Backend",
    subspace_result: TrajectorySubspaceResult | None = None,
) -> TrajectoryTangentResult | None:
    """Compute null space directions that are tangent to trajectory flow.

    The key insight: velocities show which directions the model uses to FLOW
    between concepts. Null-space directions aligned with velocities are
    "adjacent to the road" - unused but reachable.

    Algorithm:
    1. Compute full trajectory subspace (positions + velocities)
    2. Get null space (orthogonal complement)
    3. Project velocities into null space to find "road extensions"
    4. SVD on projected velocities to get dominant tangent directions

    Args:
        trajectories: List of TrajectoryResult objects.
        backend: Backend for tensor operations.
        subspace_result: Optional precomputed TrajectorySubspaceResult to
            avoid recomputing the SVD.

    Returns:
        TrajectoryTangentResult with null space and tangent directions.
        Returns None if computation fails.
    """
    b = backend

    if not trajectories:
        logger.warning("TRAJECTORY TANGENT: No trajectories provided")
        return None

    logger.info(
        "TRAJECTORY TANGENT: Computing from %d trajectories",
        len(trajectories)
    )

    try:
        # Step 1: Compute trajectory subspace (positions + velocities)
        if subspace_result is None:
            subspace_result = compute_trajectory_subspace(
                trajectories=trajectories,
                backend=b,
                include_velocities=True,
                include_accelerations=False,
            )

            if subspace_result is None:
                logger.warning("TRAJECTORY TANGENT: Subspace computation failed")
                return None

        rank = subspace_result.rank
        hidden_dim = subspace_result.hidden_dim
        Vt = subspace_result.Vt

        null_rank = hidden_dim - rank
        if null_rank <= 0:
            logger.info("TRAJECTORY TANGENT: No null space (full rank)")
            return None

        # Step 2: Get null space basis
        # Null space is spanned by Vt[rank:].T
        U_null = b.transpose(Vt[rank:, :])  # [hidden_dim, null_rank]
        U_null = _promote_precision(U_null, b)
        b.eval(U_null)

        # Step 3: Stack all velocities
        all_velocities: list["Array"] = []
        for traj in trajectories:
            all_velocities.append(traj.velocities)

        velocities = b.concatenate(all_velocities, axis=0)  # [n_vel, hidden_dim]
        velocities = _promote_precision(velocities, b)
        b.eval(velocities)

        n_vel = int(b.shape(velocities)[0])

        # Step 4: Project velocities into null space
        # velocity_in_null[i] = U_null @ U_null.T @ velocity[i]
        # This gives the component of each velocity that lies in null space
        velocity_null_proj = b.matmul(
            b.matmul(velocities, U_null),  # [n_vel, null_rank]
            b.transpose(U_null)  # [null_rank, hidden_dim]
        )  # [n_vel, hidden_dim]
        b.eval(velocity_null_proj)

        # Step 5: SVD on velocity projections to find dominant tangent directions
        # These are the null-space directions most aligned with velocity flow
        U_tan, S_tan, Vt_tan = b.svd(velocity_null_proj, compute_uv=True)
        b.eval(U_tan, S_tan, Vt_tan)

        # Step 6: Determine tangent rank (significant velocity projections)
        eps = machine_epsilon(b, S_tan)
        threshold_factor = sqrt_scalar(eps, b)

        max_s_arr = b.max(S_tan)
        b.eval(max_s_arr)
        max_s = float(b.to_scalar(max_s_arr))

        if max_s < eps:
            logger.info(
                "TRAJECTORY TANGENT: Velocities don't project into null space "
                "(max_singular=%.4e). No tangent subspace.",
                max_s,
            )
            return None

        threshold = max_s * threshold_factor
        rank_mask = S_tan > threshold
        tangent_rank_arr = b.sum(b.astype(rank_mask, "int32"))
        b.eval(tangent_rank_arr)
        tangent_rank = int(b.to_scalar(tangent_rank_arr))

        if tangent_rank <= 0:
            logger.info(
                "TRAJECTORY TANGENT: No singular values above threshold (%.4e).",
                threshold,
            )
            return None

        # Clamp to null_rank
        tangent_rank = min(tangent_rank, null_rank)

        # Step 7: Extract tangent basis
        # Vt_tan contains right singular vectors - directions in hidden_dim space
        # The first tangent_rank rows of Vt_tan are the dominant tangent directions
        U_tangent = b.transpose(Vt_tan[:tangent_rank, :])  # [hidden_dim, tangent_rank]
        b.eval(U_tangent)

        # Step 8: Compute alignment metric
        # How well do the tangent directions capture velocity projections?
        total_var = b.sum(S_tan * S_tan)
        captured_var = b.sum(S_tan[:tangent_rank] * S_tan[:tangent_rank])
        b.eval(total_var, captured_var)

        total_var_val = float(b.to_scalar(total_var))
        captured_var_val = float(b.to_scalar(captured_var))

        if total_var_val > 0:
            velocity_alignment = captured_var_val / total_var_val
        else:
            velocity_alignment = 0.0

        logger.info(
            "TRAJECTORY TANGENT: null_rank=%d, tangent_rank=%d (%.1f%%), "
            "velocity_alignment=%.4f",
            null_rank, tangent_rank,
            100.0 * tangent_rank / null_rank if null_rank > 0 else 0,
            velocity_alignment
        )

        return TrajectoryTangentResult(
            U_null=U_null,
            U_tangent=U_tangent,
            null_rank=null_rank,
            tangent_rank=tangent_rank,
            velocity_alignment=velocity_alignment,
            hidden_dim=hidden_dim,
        )

    except Exception as e:
        logger.error("TRAJECTORY TANGENT: Computation failed: %s", e)
        import traceback
        logger.error("TRACEBACK:\n%s", traceback.format_exc())
        return None


def project_delta_to_trajectory_tangent(
    delta: "Array",
    tangent_result: TrajectoryTangentResult,
    backend: "Backend",
    use_full_null: bool = False,
) -> "Array":
    """Project weight delta into trajectory-tangent null space.

    This puts the delta "along the road" - in null-space directions that
    are aligned with the model's natural trajectory flow.

    Args:
        delta: Weight delta to project [out_dim, in_dim] or [in_dim].
        tangent_result: Result from compute_trajectory_tangent_null_space.
        backend: Backend for tensor operations.
        use_full_null: If True, use full null space instead of tangent subspace.
            Useful for comparison or fallback.

    Returns:
        Projected delta in the same shape as input.
    """
    b = backend

    delta = b.array(delta)
    delta = _promote_precision(delta, b)
    b.eval(delta)

    # Choose projection basis
    if use_full_null:
        U = tangent_result.U_null
        label = "null"
    else:
        U = tangent_result.U_tangent
        label = "tangent"

    U = _promote_precision(U, b)
    b.eval(U)

    original_shape = b.shape(delta)
    is_2d = len(original_shape) == 2

    if is_2d:
        # Weight matrix [out_dim, in_dim]
        # Project along input dimension: delta @ U @ U.T
        # This projects each row (output neuron's input weights) into tangent space
        out_dim = int(original_shape[0])
        in_dim = int(original_shape[1])

        if in_dim != tangent_result.hidden_dim:
            logger.warning(
                "TRAJECTORY TANGENT PROJECT: Dimension mismatch "
                "(delta in_dim=%d, hidden_dim=%d). Returning original.",
                in_dim, tangent_result.hidden_dim
            )
            return delta

        # P = U @ U.T is the projection matrix [hidden_dim, hidden_dim]
        # delta_proj = delta @ P = delta @ U @ U.T
        delta_proj = b.matmul(
            b.matmul(delta, U),  # [out_dim, rank]
            b.transpose(U)  # [rank, hidden_dim]
        )  # [out_dim, hidden_dim]
        b.eval(delta_proj)

    else:
        # Vector [in_dim]
        in_dim = int(original_shape[0])

        if in_dim != tangent_result.hidden_dim:
            logger.warning(
                "TRAJECTORY TANGENT PROJECT: Dimension mismatch "
                "(delta dim=%d, hidden_dim=%d). Returning original.",
                in_dim, tangent_result.hidden_dim
            )
            return delta

        # P @ delta = U @ U.T @ delta
        delta_proj = b.matmul(U, b.matmul(b.transpose(U), delta))
        b.eval(delta_proj)

    # Compute norms for logging
    original_norm = b.sqrt(b.sum(delta * delta))
    projected_norm = b.sqrt(b.sum(delta_proj * delta_proj))
    b.eval(original_norm, projected_norm)

    orig_val = float(b.to_scalar(original_norm))
    proj_val = float(b.to_scalar(projected_norm))

    if orig_val > 0:
        preserved = proj_val / orig_val
    else:
        preserved = 0.0

    logger.debug(
        "TRAJECTORY TANGENT PROJECT (%s): ||delta||=%.4f, ||projected||=%.4f, "
        "preserved=%.2f%%",
        label, orig_val, proj_val, preserved * 100
    )

    return delta_proj


def project_delta_to_variance_null_space(
    delta: "Array",
    variance_result: VarianceNullSpaceResult,
    backend: "Backend",
) -> "Array":
    """Project weight delta into low-variance (available) directions.

    This projects source delta into directions where the target has low
    activation variance - the "available" capacity for new knowledge.

    Based on LoRA-Null (AAAI 2026): "The null space of activations is more accurate."

    Args:
        delta: Weight delta to project [out_dim, in_dim] or [in_dim].
        variance_result: Result from compute_variance_null_space.
        backend: Backend for tensor operations.

    Returns:
        Projected delta in the same shape as input.
    """
    b = backend

    delta = b.array(delta)
    delta = _promote_precision(delta, b)
    b.eval(delta)

    U_available = variance_result.available_basis
    U_available = _promote_precision(U_available, b)
    b.eval(U_available)

    original_shape = b.shape(delta)
    is_2d = len(original_shape) == 2
    available_rank = variance_result.available_rank

    if available_rank == 0:
        logger.warning(
            "VARIANCE NULL SPACE PROJECT: No available directions (utilized=%d/%d). "
            "Returning zero delta.",
            variance_result.utilized_rank,
            variance_result.utilized_rank + variance_result.available_rank
        )
        return b.zeros_like(delta)

    if is_2d:
        # Weight matrix [out_dim, in_dim]
        out_dim = int(original_shape[0])
        in_dim = int(original_shape[1])
        hidden_dim = variance_result.utilized_rank + variance_result.available_rank

        if in_dim != hidden_dim:
            logger.warning(
                "VARIANCE NULL SPACE PROJECT: Dimension mismatch "
                "(delta in_dim=%d, hidden_dim=%d). Returning original.",
                in_dim, hidden_dim
            )
            return delta

        # P_available = U_available @ U_available.T is the projection matrix
        # delta_proj = delta @ P_available = delta @ U_available @ U_available.T
        delta_proj = b.matmul(
            b.matmul(delta, U_available),  # [out_dim, available_rank]
            b.transpose(U_available)  # [available_rank, hidden_dim]
        )  # [out_dim, hidden_dim]
        b.eval(delta_proj)

    else:
        # Vector [in_dim]
        in_dim = int(original_shape[0])
        hidden_dim = variance_result.utilized_rank + variance_result.available_rank

        if in_dim != hidden_dim:
            logger.warning(
                "VARIANCE NULL SPACE PROJECT: Dimension mismatch "
                "(delta dim=%d, hidden_dim=%d). Returning original.",
                in_dim, hidden_dim
            )
            return delta

        # P_available @ delta = U_available @ U_available.T @ delta
        delta_proj = b.matmul(U_available, b.matmul(b.transpose(U_available), delta))
        b.eval(delta_proj)

    # Compute norms for logging
    original_norm = b.sqrt(b.sum(delta * delta))
    projected_norm = b.sqrt(b.sum(delta_proj * delta_proj))
    b.eval(original_norm, projected_norm)

    orig_val = float(b.to_scalar(original_norm))
    proj_val = float(b.to_scalar(projected_norm))

    if orig_val > 0:
        preserved = proj_val / orig_val
    else:
        preserved = 0.0

    logger.debug(
        "VARIANCE NULL SPACE PROJECT: ||delta||=%.4f, ||projected||=%.4f, "
        "preserved=%.2f%% (available_rank=%d)",
        orig_val, proj_val, preserved * 100, available_rank
    )

    return delta_proj


__all__ = [
    "TrajectoryTangentResult",
    "compute_trajectory_tangent_null_space",
    "project_delta_to_trajectory_tangent",
    "project_delta_to_variance_null_space",
]

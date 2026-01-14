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

"""Constrained transplant validation utilities.

Provides verification that null-space constrained transplant preserves
boundary outputs as mathematically guaranteed:

    A_boundary @ W' = A_boundary @ W_target

This is the core invariant of the transplant approach, validated by
AlphaEdit (ICLR 2025 Outstanding Paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_paired_distances
from modelcypher.core.domain.geometry.transplant import compute_transplant_delta

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def verify_boundary_invariance(
    transplanted_weights: "Array",
    target_weights: "Array",
    boundary_activations: "Array",
    tolerance: float | None = None,
    backend: "Backend | None" = None,
) -> dict[str, Any]:
    """Verify that boundary outputs are preserved after transplant.

    The null-space constrained transplant guarantees:
        A_boundary @ W' = A_boundary @ W_target

    This function measures the relative difference and reports whether
    the guarantee holds within numerical tolerance.

    Args:
        transplanted_weights: Merged weights W' after transplant [out_dim, in_dim]
        target_weights: Original target weights W_target [out_dim, in_dim]
        boundary_activations: Boundary probe activations [n_boundary, in_dim]
        tolerance: Maximum allowed relative difference (derived from dtype if None)
        backend: Compute backend

    Returns:
        Dictionary with:
            passed: bool - whether boundary invariance holds
            max_relative_diff: float - maximum relative difference
            mean_relative_diff: float - mean relative difference
            boundary_samples: int - number of boundary samples checked
    """
    b = backend or get_default_backend()

    # Ensure arrays are backend arrays
    transplanted = b.array(transplanted_weights)
    target = b.array(target_weights)
    boundary = b.array(boundary_activations)
    b.eval(transplanted, target, boundary)

    n_boundary = int(boundary.shape[0])
    if n_boundary == 0:
        return {
            "passed": True,
            "max_relative_diff": 0.0,
            "mean_relative_diff": 0.0,
            "boundary_samples": 0,
        }

    # Compute outputs: A @ W^T (weight is [out, in], activation is [n, in])
    output_transplanted = b.matmul(boundary, b.transpose(transplanted))
    output_target = b.matmul(boundary, b.transpose(target))
    b.eval(output_transplanted, output_target)

    # Compute per-sample relative difference using geodesic distance.
    # Geodesic works in all dimensions (reduces to chord in flat spaces).
    # Chord distance systematically errs in high dimensions (4D+).
    diff_norms = geodesic_paired_distances(output_transplanted, output_target, b)
    origin = b.zeros_like(output_target)
    target_norms = geodesic_paired_distances(origin, output_target, b)
    b.eval(diff_norms, target_norms)

    eps = float(machine_epsilon(b, target))
    if tolerance is None:
        tolerance = float(division_epsilon(b, target))

    eps_arr = b.full(target_norms.shape, eps)
    mask_target = target_norms > eps
    mask_small = diff_norms <= eps
    relative_diffs = b.where(
        mask_target,
        diff_norms / b.maximum(target_norms, eps_arr),
        b.where(mask_small, b.zeros_like(diff_norms), b.full(diff_norms.shape, float("inf"))),
    )
    b.eval(relative_diffs)
    max_rel_diff_arr = b.max(relative_diffs)
    b.eval(max_rel_diff_arr)
    max_rel_diff = float(b.to_scalar(max_rel_diff_arr))
    inf_count = b.sum(
        b.astype(b.isinf(relative_diffs), precision_dtype(b, reference=relative_diffs))
    )
    b.eval(inf_count)
    if float(b.to_scalar(inf_count)) > 0:
        mean_rel_diff = float("inf")
    else:
        mean_rel_diff_arr = b.mean(relative_diffs)
        b.eval(mean_rel_diff_arr)
        mean_rel_diff = float(b.to_scalar(mean_rel_diff_arr))

    return {
        "passed": max_rel_diff < float(tolerance),
        "max_relative_diff": max_rel_diff,
        "mean_relative_diff": mean_rel_diff,
        "boundary_samples": n_boundary,
    }


@dataclass(frozen=True)
class CausalInterventionReport:
    """Causal measurements for null-space transplant interventions."""

    core_samples: int
    boundary_samples: int
    core_mean_shift: float
    core_max_shift: float
    core_residual_mean: float
    core_residual_max: float
    boundary_max_relative_diff: float
    boundary_mean_relative_diff: float
    boundary_tolerance: float
    preserved_fraction: float
    projection_loss: float
    null_dim: int


def causal_intervention_report(
    target_weights: "Array",
    activations_core: "Array",
    delta_activations: "Array",
    boundary_activations: "Array | None" = None,
    tolerance: float | None = None,
    backend: "Backend | None" = None,
) -> CausalInterventionReport:
    """Measure causal effect of a constrained transplant on core vs boundary."""
    b = backend or get_default_backend()

    target = b.array(target_weights)
    core = b.array(activations_core)
    delta = b.array(delta_activations)
    b.eval(target, core, delta)

    boundary = None
    if boundary_activations is not None:
        boundary = b.array(boundary_activations)
        b.eval(boundary)

    transplant = compute_transplant_delta(
        weight_target=target,
        activations_core=core,
        delta_activations=delta,
        boundary_activations=boundary,
        backend=b,
    )

    merged = b.array(transplant.merged_weight)
    b.eval(merged)

    # Core shifts
    output_target = b.matmul(core, b.transpose(target))
    output_merged = b.matmul(core, b.transpose(merged))
    b.eval(output_target, output_merged)

    core_shift = geodesic_paired_distances(output_merged, output_target, b)
    b.eval(core_shift)
    core_mean = float(b.to_scalar(b.mean(core_shift)))
    core_max = float(b.to_scalar(b.max(core_shift)))

    # Residual to desired delta
    actual_delta = output_merged - output_target
    residual = geodesic_paired_distances(actual_delta, delta, b)
    b.eval(residual)
    residual_mean = float(b.to_scalar(b.mean(residual)))
    residual_max = float(b.to_scalar(b.max(residual)))

    # Boundary invariance
    if boundary is None:
        boundary_report = {
            "max_relative_diff": 0.0,
            "mean_relative_diff": 0.0,
            "boundary_samples": 0,
        }
        if tolerance is None:
            tolerance = float(division_epsilon(b, target))
    else:
        boundary_report = verify_boundary_invariance(
            transplanted_weights=merged,
            target_weights=target,
            boundary_activations=boundary,
            tolerance=tolerance,
            backend=b,
        )
        if tolerance is None:
            tolerance = float(division_epsilon(b, target))

    return CausalInterventionReport(
        core_samples=int(core.shape[0]),
        boundary_samples=int(boundary_report["boundary_samples"]),
        core_mean_shift=core_mean,
        core_max_shift=core_max,
        core_residual_mean=residual_mean,
        core_residual_max=residual_max,
        boundary_max_relative_diff=float(boundary_report["max_relative_diff"]),
        boundary_mean_relative_diff=float(boundary_report["mean_relative_diff"]),
        boundary_tolerance=float(tolerance),
        preserved_fraction=float(transplant.preserved_fraction),
        projection_loss=float(transplant.projection_loss),
        null_dim=int(transplant.null_dim),
    )


__all__ = ["CausalInterventionReport", "causal_intervention_report", "verify_boundary_invariance"]

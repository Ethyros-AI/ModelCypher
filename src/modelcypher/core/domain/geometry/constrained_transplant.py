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

from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_paired_distances

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
    inf_count = b.sum(b.astype(b.isinf(relative_diffs), "float32"))
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


__all__ = ["verify_boundary_invariance"]

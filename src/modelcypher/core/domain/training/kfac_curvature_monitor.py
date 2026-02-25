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

"""K-FAC curvature diagnostic for training (Experiment 3).

Measures how much of the adapter update norm falls in high-curvature
directions of the K-FAC Gauss-Newton Hessian approximation.

DIAGNOSTIC ONLY — no gradient modification. If existing Cayley+MASS
constraints already steer updates away from high-curvature directions
(< 5% of norm in top-10%), no action needed. If > 20%, consider
gradient projection via the existing gradient_hook in train_loop().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)

if TYPE_CHECKING:
    from modelcypher.core.domain.geometry.kfac_projector import KFACFactors
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerCurvatureResult:
    """Curvature alignment diagnostic for one adapter layer."""

    layer_name: str
    delta_frobenius: float
    top_10pct_fraction: float
    top_25pct_fraction: float
    null_fraction: float
    n_total_directions: int
    n_top_10pct: int
    n_null: int
    kfac_gain_ratio: float


@dataclass(frozen=True)
class EpochCurvatureReport:
    """Aggregate curvature alignment for all adapter layers at one epoch."""

    epoch: int
    n_layers: int
    median_top_10pct_fraction: float
    max_top_10pct_fraction: float
    median_null_fraction: float
    per_layer: list[LayerCurvatureResult]


def compute_curvature_alignment(
    delta_weight: "Array",
    factors: "KFACFactors",
    *,
    layer_name: str = "",
    backend: "Backend | None" = None,
) -> LayerCurvatureResult:
    """Decompose adapter delta into K-FAC eigenbasis and measure curvature alignment.

    Transforms delta_weight into the Kronecker eigenbasis (U_s, U_a) and
    measures what fraction of ||delta||_F^2 falls in:
    - Top-10% curvature directions (highest Kronecker eigenvalues)
    - Top-25% curvature directions
    - Null directions (below kron_threshold)

    Args:
        delta_weight: Adapter weight delta [out_dim, in_dim].
        factors: K-FAC eigensystem factors from compute_kfac_factors().
        layer_name: Name for logging/reporting.
        backend: Backend for tensor operations.

    Returns:
        LayerCurvatureResult with energy fractions per curvature band.
    """
    b = backend or get_default_backend()

    delta = b.array(delta_weight)
    compute_dtype = precision_dtype(b, reference=delta)
    delta = b.astype(delta, compute_dtype)
    A_eigvecs = b.astype(factors.A_eigvecs, compute_dtype)
    S_eigvecs = b.astype(factors.S_eigvecs, compute_dtype)
    b.eval(delta, A_eigvecs, S_eigvecs)

    # Transform to Kronecker eigenbasis: Q[i,j] = component in (S_i, A_j) direction
    Q = b.matmul(b.transpose(S_eigvecs), delta)
    Q = b.matmul(Q, A_eigvecs)
    b.eval(Q)

    # Energy per direction (Frobenius norm is invariant under orthogonal transform)
    Q_sq = Q * Q
    total_energy_arr = b.sum(Q_sq)
    b.eval(total_energy_arr)
    total_energy = float(b.to_scalar(total_energy_arr))

    eps_div = float(division_epsilon(b, delta))
    safe_total = max(total_energy, eps_div)

    # Kronecker eigenvalue grid
    S_eigvals = b.astype(factors.S_eigvals, compute_dtype)
    A_eigvals = b.astype(factors.A_eigvals, compute_dtype)
    b.eval(S_eigvals, A_eigvals)

    S_col = b.reshape(S_eigvals, (factors.out_dim, 1))
    A_row = b.reshape(A_eigvals, (1, factors.in_dim))
    kron_eigvals = b.matmul(S_col, A_row)
    b.eval(kron_eigvals)

    n_total = factors.out_dim * factors.in_dim
    n_top_10 = max(1, n_total // 10)
    n_top_25 = max(1, n_total // 4)

    # Flatten and sort descending to find top-k% curvature threshold
    kron_flat = b.reshape(kron_eigvals, (-1,))
    kron_sorted = b.sort(kron_flat, axis=0)
    b.eval(kron_sorted)

    # b.sort is ascending; top-10% threshold is at index n_total - n_top_10
    threshold_10_idx = max(0, n_total - n_top_10)
    threshold_25_idx = max(0, n_total - n_top_25)
    threshold_10_arr = b.take(kron_sorted, b.array(threshold_10_idx), axis=0)
    threshold_25_arr = b.take(kron_sorted, b.array(threshold_25_idx), axis=0)
    b.eval(threshold_10_arr, threshold_25_arr)
    threshold_10 = float(b.to_scalar(threshold_10_arr))
    threshold_25 = float(b.to_scalar(threshold_25_arr))

    # Energy in top-10% curvature directions
    top_10_mask = b.astype(kron_eigvals >= threshold_10, compute_dtype)
    top_25_mask = b.astype(kron_eigvals >= threshold_25, compute_dtype)
    top_10_energy_arr = b.sum(Q_sq * top_10_mask)
    top_25_energy_arr = b.sum(Q_sq * top_25_mask)
    b.eval(top_10_energy_arr, top_25_energy_arr)
    top_10_energy = float(b.to_scalar(top_10_energy_arr))
    top_25_energy = float(b.to_scalar(top_25_energy_arr))

    # Energy in null directions (from K-FAC null mask)
    null_mask_float = b.astype(factors.null_mask, compute_dtype)
    null_energy_arr = b.sum(Q_sq * null_mask_float)
    null_count_arr = b.sum(null_mask_float)
    b.eval(null_energy_arr, null_count_arr)
    null_energy = float(b.to_scalar(null_energy_arr))
    n_null = int(round(float(b.to_scalar(null_count_arr))))

    return LayerCurvatureResult(
        layer_name=layer_name,
        delta_frobenius=math.sqrt(max(0.0, total_energy)),
        top_10pct_fraction=top_10_energy / safe_total,
        top_25pct_fraction=top_25_energy / safe_total,
        null_fraction=null_energy / safe_total,
        n_total_directions=n_total,
        n_top_10pct=n_top_10,
        n_null=n_null,
        kfac_gain_ratio=factors.gain_ratio,
    )


def aggregate_epoch_curvature(
    epoch: int,
    per_layer_results: list[LayerCurvatureResult],
) -> EpochCurvatureReport:
    """Aggregate per-layer curvature diagnostics into epoch-level report.

    Args:
        epoch: Current epoch number.
        per_layer_results: List of per-layer curvature diagnostics.

    Returns:
        EpochCurvatureReport with median/max statistics.
    """
    n = len(per_layer_results)
    if n == 0:
        return EpochCurvatureReport(
            epoch=epoch,
            n_layers=0,
            median_top_10pct_fraction=0.0,
            max_top_10pct_fraction=0.0,
            median_null_fraction=0.0,
            per_layer=[],
        )

    top_10_fracs = sorted(r.top_10pct_fraction for r in per_layer_results)
    null_fracs = sorted(r.null_fraction for r in per_layer_results)

    median_idx = n // 2
    median_top_10 = top_10_fracs[median_idx]
    max_top_10 = top_10_fracs[-1]
    median_null = null_fracs[median_idx]

    report = EpochCurvatureReport(
        epoch=epoch,
        n_layers=n,
        median_top_10pct_fraction=median_top_10,
        max_top_10pct_fraction=max_top_10,
        median_null_fraction=median_null,
        per_layer=per_layer_results,
    )

    logger.info(
        "K-FAC CURVATURE epoch=%d: median_top10=%.3f, max_top10=%.3f, "
        "median_null=%.3f (%d layers)",
        epoch,
        median_top_10,
        max_top_10,
        median_null,
        n,
    )

    return report


__all__ = [
    "EpochCurvatureReport",
    "LayerCurvatureResult",
    "aggregate_epoch_curvature",
    "compute_curvature_alignment",
]

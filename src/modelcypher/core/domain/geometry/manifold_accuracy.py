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

"""Accuracy reports for geodesic distance and curvature estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import (
    RiemannianGeometry,
    derive_k_neighbors,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class GeodesicAccuracyReport:
    sample_count: int
    k_neighbors: int
    mean_abs_error: float
    max_abs_error: float
    mean_relative_error: float
    connected: bool


@dataclass(frozen=True)
class CurvatureAccuracyReport:
    sample_count: int
    k_neighbors: int
    mean_curvature: float
    max_curvature: float
    analytic_curvature: float
    mean_abs_error: float
    max_abs_error: float
    mean_confidence: float


def geodesic_accuracy_report(
    points: "Array",
    analytic_distances: "Array",
    k_neighbors: int | None = None,
    backend: "Backend | None" = None,
) -> GeodesicAccuracyReport:
    """Compute geodesic approximation error against analytic distances."""
    b = backend or get_default_backend()
    points_arr = b.array(points)
    analytic_arr = b.array(analytic_distances)
    b.eval(points_arr, analytic_arr)

    if k_neighbors is None:
        k_neighbors = derive_k_neighbors(points_arr, b)

    rg = RiemannianGeometry(b)
    geo_result = rg.geodesic_distances(points_arr, k_neighbors=k_neighbors)
    approx = geo_result.distances
    b.eval(approx)

    diff = b.abs(approx - analytic_arr)
    mask = approx < geo_result.inf_value
    diff = b.where(mask, diff, b.zeros_like(diff))
    b.eval(diff)

    mask_f = b.astype(mask, precision_dtype(b, reference=diff))
    count = b.sum(mask_f)
    b.eval(count)
    count_val = float(b.to_scalar(count))
    denom = count_val if count_val > 0 else 1.0

    mean_abs = float(b.to_scalar(b.sum(diff))) / denom
    max_abs = float(b.to_scalar(b.max(diff)))

    eps = division_epsilon(b, analytic_arr)
    eps_arr = b.full(analytic_arr.shape, eps, dtype=analytic_arr.dtype)
    rel = diff / b.maximum(analytic_arr, eps_arr)
    rel = b.where(mask, rel, b.zeros_like(rel))
    b.eval(rel)
    mean_rel = float(b.to_scalar(b.sum(rel))) / denom

    return GeodesicAccuracyReport(
        sample_count=int(points_arr.shape[0]),
        k_neighbors=int(geo_result.k_neighbors),
        mean_abs_error=float(mean_abs),
        max_abs_error=float(max_abs),
        mean_relative_error=float(mean_rel),
        connected=bool(geo_result.connected),
    )


def curvature_accuracy_report(
    points: "Array",
    analytic_curvature: float,
    k_neighbors: int | None = None,
    center_indices: list[int] | None = None,
    backend: "Backend | None" = None,
) -> CurvatureAccuracyReport:
    """Estimate curvature and compare against analytic ground truth."""
    b = backend or get_default_backend()
    points_arr = b.array(points)
    b.eval(points_arr)

    n = int(points_arr.shape[0])
    if n == 0:
        return CurvatureAccuracyReport(
            sample_count=0,
            k_neighbors=0,
            mean_curvature=0.0,
            max_curvature=0.0,
            analytic_curvature=float(analytic_curvature),
            mean_abs_error=0.0,
            max_abs_error=0.0,
            mean_confidence=0.0,
        )

    if k_neighbors is None:
        k_neighbors = derive_k_neighbors(points_arr, b)

    if center_indices is None:
        center_indices = list(range(n))

    rg = RiemannianGeometry(b)
    curvatures: list[float] = []
    confidences: list[float] = []
    for idx in center_indices:
        estimate = rg.estimate_local_curvature(
            points_arr,
            center_idx=int(idx),
            k_neighbors=k_neighbors,
        )
        curvatures.append(float(estimate.sectional_curvature))
        confidences.append(float(estimate.confidence))

    mean_curv = sum(curvatures) / len(curvatures)
    max_curv = max(curvatures)
    abs_errors = [abs(c - analytic_curvature) for c in curvatures]
    mean_abs = sum(abs_errors) / len(abs_errors)
    max_abs = max(abs_errors)
    mean_conf = sum(confidences) / len(confidences)

    return CurvatureAccuracyReport(
        sample_count=int(points_arr.shape[0]),
        k_neighbors=int(k_neighbors),
        mean_curvature=float(mean_curv),
        max_curvature=float(max_curv),
        analytic_curvature=float(analytic_curvature),
        mean_abs_error=float(mean_abs),
        max_abs_error=float(max_abs),
        mean_confidence=float(mean_conf),
    )


__all__ = [
    "CurvatureAccuracyReport",
    "GeodesicAccuracyReport",
    "curvature_accuracy_report",
    "geodesic_accuracy_report",
]

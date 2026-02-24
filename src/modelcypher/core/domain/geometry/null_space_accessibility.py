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

"""Null-space accessibility diagnostics for training validation.

This module answers a specific geometric question:
"Is null space present but not being accessed?"

Diagnostics are strictly measurement-only. No thresholds, no gating logic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.gram_spectrum import (
    analyze_projection,
    compute_gram_spectrum,
)
from modelcypher.core.domain.geometry.null_space import compute_variance_null_space
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
    sqrt_scalar,
)
from modelcypher.core.domain.geometry.subspace import compute_grassmann_distance

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _extract_layer_index(module_key: str) -> int | None:
    parts = module_key.split(".")
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _safe_float(value: float | int | bool | None) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def _to_2d(delta_w: "Array", backend: "Backend") -> "Array":
    arr = backend.array(delta_w) if not hasattr(delta_w, "shape") else delta_w
    shape = backend.shape(arr)
    if len(shape) == 1:
        arr = backend.reshape(arr, (1, int(shape[0])))
    elif len(shape) > 2:
        first = int(shape[0])
        remaining = 1
        for dim in shape[1:]:
            remaining *= int(dim)
        arr = backend.reshape(arr, (first, remaining))
    return arr


def _row_space_basis(
    delta_w: "Array",
    backend: "Backend",
) -> tuple["Array", int, float]:
    """Compute orthonormal basis for row-space(delta_w) via right singular vectors."""
    b = backend
    delta_2d = _to_2d(delta_w, b)
    delta_2d = b.astype(delta_2d, precision_dtype(b, reference=delta_2d))
    b.eval(delta_2d)

    _u, s_vals, vt = b.svd(delta_2d, compute_uv=True)
    b.eval(s_vals, vt)

    n_sv = int(b.shape(s_vals)[0])
    in_dim = int(b.shape(delta_2d)[1])
    if n_sv == 0:
        return b.zeros((0, in_dim)), 0, 0.0

    max_sv_arr = b.max(s_vals)
    b.eval(max_sv_arr)
    max_sv = float(b.to_scalar(max_sv_arr))
    eps = machine_epsilon(b, s_vals)
    threshold = max_sv * sqrt_scalar(eps, b)
    threshold_arr = b.array([threshold], dtype=precision_dtype(b, reference=s_vals))
    rank_mask = s_vals > threshold_arr
    count_arr = b.sum(b.astype(rank_mask, "int32"))
    b.eval(count_arr)
    rank = int(b.to_scalar(count_arr))

    if rank <= 0:
        return b.zeros((0, in_dim)), 0, threshold

    keep_idx = b.arange(0, rank)
    basis = b.take(vt, keep_idx, axis=0)  # [rank, in_dim]
    b.eval(basis)
    return basis, rank, threshold


def _principal_angle_stats(principal_angles: "Array", backend: "Backend") -> tuple[int, float, float, float]:
    b = backend
    shape = b.shape(principal_angles)
    n = int(shape[0]) if len(shape) > 0 else 0
    if n <= 0:
        return 0, float("nan"), float("nan"), float("nan")
    min_arr = b.min(principal_angles)
    max_arr = b.max(principal_angles)
    mean_arr = b.mean(principal_angles)
    b.eval(min_arr, max_arr, mean_arr)
    return (
        n,
        float(b.to_scalar(min_arr)),
        float(b.to_scalar(max_arr)),
        float(b.to_scalar(mean_arr)),
    )


def analyze_layer_null_observability(
    activations: "Array",
    backend: "Backend | None" = None,
) -> dict[str, float | int]:
    """Measure per-layer null-space observability from activations only."""
    b = backend or get_default_backend()
    spectrum = compute_gram_spectrum(activations, backend=b)
    coverage_ratio = (
        spectrum.n_samples / spectrum.d_features
        if spectrum.d_features > 0
        else float("nan")
    )
    return {
        "n_samples": int(spectrum.n_samples),
        "hidden_dim": int(spectrum.d_features),
        "coverage_ratio": float(coverage_ratio),
        "available_rank": int(spectrum.null_rank),
        "used_rank": int(spectrum.numeric_rank),
        "condition_number": float(spectrum.condition_number),
        "intrinsic_dimension": float(spectrum.intrinsic_dimension),
        "energy_ratio_numeric_rank": float(spectrum.energy_ratio_numeric_rank),
        "energy_ratio_intrinsic_dim": float(spectrum.energy_ratio_intrinsic_dim),
        "spectral_gap": float(spectrum.spectral_gap),
        "rank_threshold": float(spectrum.rank_threshold),
        "total_variance": float(spectrum.total_variance),
        "max_eigenvalue": float(spectrum.max_eigenvalue),
        "min_eigenvalue": float(spectrum.min_eigenvalue),
    }


def analyze_module_null_accessibility(
    delta_w: "Array",
    activations: "Array",
    backend: "Backend | None" = None,
) -> dict[str, float | int]:
    """Measure whether a module delta is geometrically accessing available null-space."""
    b = backend or get_default_backend()
    activations = b.array(activations)
    delta_2d = _to_2d(delta_w, b)
    delta_2d = b.astype(delta_2d, precision_dtype(b, reference=delta_2d))
    activations = b.astype(activations, precision_dtype(b, reference=activations))
    b.eval(delta_2d, activations)

    obs = analyze_layer_null_observability(activations, b)
    out_dim = int(b.shape(delta_2d)[0])
    in_dim = int(b.shape(delta_2d)[1])
    hidden_dim = int(b.shape(activations)[1])
    dimension_compatible = int(in_dim == hidden_dim)

    row_basis, row_rank, row_rank_threshold = _row_space_basis(delta_2d, b)
    variance_split = compute_variance_null_space(activations, b)
    available_basis = variance_split.available_basis
    available_rank = int(variance_split.available_rank)

    projection_delta_norm = float("nan")
    projection_projected_norm = float("nan")
    projection_correction_norm = float("nan")
    behavioral_preserved_fraction = float("nan")
    behavioral_cosine_similarity = float("nan")
    if dimension_compatible == 1:
        projection = analyze_projection(delta_2d, activations, b)
        projection_delta_norm = float(projection.delta_norm)
        projection_projected_norm = float(projection.projected_norm)
        projection_correction_norm = float(projection.correction_norm)
        behavioral_preserved_fraction = float(projection.preserved_fraction)
        behavioral_cosine_similarity = float(projection.cosine_similarity)
        eps = machine_epsilon(b, activations)
        if projection_delta_norm <= eps and projection_projected_norm <= eps:
            # Degenerate zero-behavior case: projection neither removed nor added behavior.
            behavioral_preserved_fraction = 1.0
            behavioral_cosine_similarity = 1.0

    available_basis_rows = b.transpose(available_basis)
    b.eval(available_basis_rows)

    principal_angle_count = 0
    principal_angle_min = float("nan")
    principal_angle_max = float("nan")
    principal_angle_mean = float("nan")
    grassmann_geodesic_distance = float("nan")
    grassmann_chordal_distance = float("nan")
    if dimension_compatible == 1:
        grassmann = compute_grassmann_distance(row_basis, available_basis_rows, b)
        principal_angle_count, principal_angle_min, principal_angle_max, principal_angle_mean = (
            _principal_angle_stats(grassmann.principal_angles, b)
        )
        grassmann_geodesic_distance = float(grassmann.geodesic_distance)
        grassmann_chordal_distance = float(grassmann.chordal_distance)

    return {
        "out_dim": int(out_dim),
        "in_dim": int(in_dim),
        "hidden_dim": int(hidden_dim),
        "dimension_compatible": int(dimension_compatible),
        "coverage_ratio": float(obs["coverage_ratio"]),
        "available_rank": int(obs["available_rank"]),
        "used_rank": int(obs["used_rank"]),
        "condition_number": float(obs["condition_number"]),
        "variance_threshold": float(variance_split.variance_threshold),
        "variance_available_rank": int(available_rank),
        "variance_utilized_rank": int(variance_split.utilized_rank),
        "delta_row_rank": int(row_rank),
        "delta_row_rank_threshold": float(row_rank_threshold),
        "behavioral_delta_norm": float(projection_delta_norm),
        "behavioral_projected_norm": float(projection_projected_norm),
        "behavioral_correction_norm": float(projection_correction_norm),
        "behavioral_preserved_fraction": float(behavioral_preserved_fraction),
        "behavioral_cosine_similarity": float(behavioral_cosine_similarity),
        "principal_angle_count": int(principal_angle_count),
        "principal_angle_min": float(principal_angle_min),
        "principal_angle_max": float(principal_angle_max),
        "principal_angle_mean": float(principal_angle_mean),
        "grassmann_geodesic_distance": float(grassmann_geodesic_distance),
        "grassmann_chordal_distance": float(grassmann_chordal_distance),
    }


def aggregate_layer_accessibility(
    per_module_metrics: dict[str, dict[str, float | int]],
) -> dict[int, dict[str, float | int]]:
    """Aggregate module-level accessibility diagnostics into layer summaries."""
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for module_key, metrics in per_module_metrics.items():
        layer_idx = _extract_layer_index(module_key)
        if layer_idx is None:
            continue
        grouped.setdefault(layer_idx, []).append(metrics)

    out: dict[int, dict[str, float | int]] = {}
    for layer_idx, metrics_list in grouped.items():
        module_count = len(metrics_list)
        compatible_metrics = [
            m for m in metrics_list
            if int(_safe_float(m.get("dimension_compatible"))) == 1
        ]
        compatible_count = len(compatible_metrics)

        preserved = []
        preserved_weights = []
        principal_mean = []
        principal_weights = []
        available_ranks = []
        coverage_ratios = []
        conditions = []
        total_behavioral_delta_norm = 0.0
        for m in compatible_metrics:
            w = _safe_float(m.get("behavioral_delta_norm"))
            v_pres = _safe_float(m.get("behavioral_preserved_fraction"))
            v_angle = _safe_float(m.get("principal_angle_mean"))
            v_rank = _safe_float(m.get("available_rank"))
            v_cov = _safe_float(m.get("coverage_ratio"))
            v_cond = _safe_float(m.get("condition_number"))
            if math.isfinite(w) and w > 0.0:
                total_behavioral_delta_norm += w
            if math.isfinite(v_pres):
                preserved.append(v_pres)
                preserved_weights.append(w if math.isfinite(w) and w > 0.0 else 0.0)
            if math.isfinite(v_angle):
                principal_mean.append(v_angle)
                principal_weights.append(w if math.isfinite(w) and w > 0.0 else 0.0)
            if math.isfinite(v_rank):
                available_ranks.append(v_rank)
            if math.isfinite(v_cov):
                coverage_ratios.append(v_cov)
            if math.isfinite(v_cond):
                conditions.append(v_cond)

        def _weighted_mean(values: list[float], ws: list[float]) -> float:
            if not values:
                return float("nan")
            if len(values) != len(ws):
                return float("nan")
            weight_sum = sum(ws)
            if weight_sum > 0.0:
                return sum(v * w for v, w in zip(values, ws, strict=False)) / weight_sum
            return sum(values) / len(values)

        preserved_weighted = _weighted_mean(preserved, preserved_weights)
        principal_weighted = _weighted_mean(principal_mean, principal_weights)

        out[layer_idx] = {
            "module_count": int(module_count),
            "dimension_compatible_count": int(compatible_count),
            "dimension_compatible_fraction": (
                compatible_count / module_count if module_count > 0 else float("nan")
            ),
            "behavioral_preserved_fraction": float(preserved_weighted),
            "behavioral_preserved_fraction_min": (
                float(min(preserved)) if preserved else float("nan")
            ),
            "behavioral_preserved_fraction_max": (
                float(max(preserved)) if preserved else float("nan")
            ),
            "principal_angle_mean_weighted": float(principal_weighted),
            "available_rank_mean": (
                float(sum(available_ranks) / len(available_ranks))
                if available_ranks else float("nan")
            ),
            "coverage_ratio_mean": (
                float(sum(coverage_ratios) / len(coverage_ratios))
                if coverage_ratios else float("nan")
            ),
            "condition_number_mean": (
                float(sum(conditions) / len(conditions))
                if conditions else float("nan")
            ),
            "total_behavioral_delta_norm": float(total_behavioral_delta_norm),
        }

    return out


__all__ = [
    "analyze_layer_null_observability",
    "analyze_module_null_accessibility",
    "aggregate_layer_accessibility",
]

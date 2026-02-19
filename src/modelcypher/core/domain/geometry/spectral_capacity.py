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

"""Per-layer spectral capacity analysis for weight matrices.

All thresholds are derived from IEEE-754 machine precision.
No empirical constants or hand-tuned heuristics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


_EPS_F32 = math.ldexp(1.0, -23)  # IEEE-754 float32 machine epsilon
_EPS_F16 = math.ldexp(1.0, -10)  # IEEE-754 float16 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)
_SQRT_EPS_F16 = math.sqrt(_EPS_F16)


@dataclass(frozen=True)
class LayerCapacityReport:
    layer_name: str
    weight_shape: tuple[int, int]
    singular_values: list[float]
    spectral_norm: float
    nuclear_norm: float
    frobenius_norm: float
    effective_rank: float
    stable_rank: float
    numerical_rank_f32: int
    numerical_rank_f16: int
    null_space_dim_f32: int
    null_space_fraction: float
    recommended_rank: int
    spectral_gap_at_rank: float
    capacity_utilization: float
    computation_method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "layerName": self.layer_name,
            "weightShape": list(self.weight_shape),
            "singularValues": self.singular_values,
            "spectralNorm": self.spectral_norm,
            "nuclearNorm": self.nuclear_norm,
            "frobeniusNorm": self.frobenius_norm,
            "effectiveRank": self.effective_rank,
            "stableRank": self.stable_rank,
            "numericalRankF32": self.numerical_rank_f32,
            "numericalRankF16": self.numerical_rank_f16,
            "nullSpaceDimF32": self.null_space_dim_f32,
            "nullSpaceFraction": self.null_space_fraction,
            "recommendedRank": self.recommended_rank,
            "spectralGapAtRank": self.spectral_gap_at_rank,
            "capacityUtilization": self.capacity_utilization,
            "computationMethod": self.computation_method,
        }


class SpectralCapacityAnalyzer:
    """Compute layer-level capacity metrics from singular values."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def analyze(self, layer_name: str, weight: "Array") -> LayerCapacityReport:
        b = self._backend
        weight_arr = b.array(weight) if not hasattr(weight, "shape") else weight
        shape = tuple(int(dim) for dim in b.shape(weight_arr))
        if len(shape) != 2:
            raise ValueError(
                f"Spectral capacity requires a 2D matrix, got shape={shape} for layer={layer_name}"
            )

        m, n = shape
        min_dim = min(m, n)
        if min_dim == 0:
            return LayerCapacityReport(
                layer_name=layer_name,
                weight_shape=(m, n),
                singular_values=[],
                spectral_norm=0.0,
                nuclear_norm=0.0,
                frobenius_norm=0.0,
                effective_rank=0.0,
                stable_rank=0.0,
                numerical_rank_f32=0,
                numerical_rank_f16=0,
                null_space_dim_f32=0,
                null_space_fraction=0.0,
                recommended_rank=0,
                spectral_gap_at_rank=0.0,
                capacity_utilization=0.0,
                computation_method="degenerate",
            )

        sv, computation_method = _compute_singular_values(weight_arr, b)

        frobenius_norm = _frobenius_norm(weight_arr, b)
        frobenius_sq = frobenius_norm * frobenius_norm

        spectral_norm = sv[0] if sv else 0.0
        nuclear_norm = sum(sv)

        effective_rank = (
            (nuclear_norm * nuclear_norm) / frobenius_sq if frobenius_sq > 0.0 else 0.0
        )
        stable_rank = (
            frobenius_sq / (spectral_norm * spectral_norm) if spectral_norm > 0.0 else 0.0
        )

        threshold_f32 = spectral_norm * _SQRT_EPS_F32
        threshold_f16 = spectral_norm * _SQRT_EPS_F16
        numerical_rank_f32 = sum(1 for value in sv if value > threshold_f32)
        numerical_rank_f16 = sum(1 for value in sv if value > threshold_f16)

        null_space_dim_f32 = max(0, min_dim - numerical_rank_f32)
        null_space_fraction = (
            float(null_space_dim_f32) / float(min_dim) if min_dim > 0 else 0.0
        )

        recommended_rank, spectral_gap_at_rank = _largest_relative_gap_rank(sv)
        capacity_utilization = (
            effective_rank / float(min_dim) if min_dim > 0 else 0.0
        )

        return LayerCapacityReport(
            layer_name=layer_name,
            weight_shape=(m, n),
            singular_values=sv,
            spectral_norm=spectral_norm,
            nuclear_norm=nuclear_norm,
            frobenius_norm=frobenius_norm,
            effective_rank=effective_rank,
            stable_rank=stable_rank,
            numerical_rank_f32=numerical_rank_f32,
            numerical_rank_f16=numerical_rank_f16,
            null_space_dim_f32=null_space_dim_f32,
            null_space_fraction=null_space_fraction,
            recommended_rank=recommended_rank,
            spectral_gap_at_rank=spectral_gap_at_rank,
            capacity_utilization=capacity_utilization,
            computation_method=computation_method,
        )


def _compute_singular_values(weight: "Array", backend: "Backend") -> tuple[list[float], str]:
    b = backend
    weight_f32 = b.astype(weight, "float32")
    b.eval(weight_f32)

    try:
        singular_values = b.svd(weight_f32, compute_uv=False)
        b.eval(singular_values)
        values = b.tolist(singular_values)
        if isinstance(values, (int, float)):
            return [float(values)], "svd"
        return sorted((float(v) for v in values), reverse=True), "svd"
    except Exception:
        pass

    try:
        gram_values = _gram_eigh_singular_values(weight_f32, b)
        if gram_values:
            return gram_values, "gram_eigh"
    except Exception:
        pass

    iterative_values = _iterative_singular_values(weight_f32, b)
    if iterative_values:
        return iterative_values, "power_deflation"

    spectral_fallback = _spectral_norm_power_iteration(weight_f32, b)
    if spectral_fallback > 0.0:
        return [spectral_fallback], "power_iteration"

    raise RuntimeError("Singular spectrum computation failed across all methods.")


def _spectral_norm_power_iteration(
    matrix: "Array",
    backend: "Backend",
) -> float:
    """Estimate largest singular value via power iteration."""
    b = backend
    m, n = int(matrix.shape[0]), int(matrix.shape[1])
    if m == 0 or n == 0:
        return 0.0

    max_iters = max(2, min(m, n))
    sqrt_eps = _SQRT_EPS_F32

    v = b.ones((n,))
    norm_v = b.norm(v)
    b.eval(norm_v)
    v_norm = float(b.to_scalar(norm_v))
    if v_norm <= 0.0:
        return 0.0
    v = v / v_norm
    b.eval(v)

    sigma = 0.0
    prev_sigma = 0.0
    for _ in range(max_iters):
        u = b.matmul(matrix, b.reshape(v, (-1, 1)))
        u = b.reshape(u, (-1,))
        u_norm = b.norm(u)
        b.eval(u_norm)
        u_norm_val = float(b.to_scalar(u_norm))
        if u_norm_val <= 0.0:
            return 0.0
        u = u / u_norm_val
        b.eval(u)

        v = b.matmul(b.transpose(matrix), b.reshape(u, (-1, 1)))
        v = b.reshape(v, (-1,))
        v_norm = b.norm(v)
        b.eval(v_norm)
        sigma = float(b.to_scalar(v_norm))
        if sigma <= 0.0:
            return 0.0
        v = v / sigma
        b.eval(v)

        delta = abs(sigma - prev_sigma)
        tolerance = sqrt_eps * max(1.0, sigma)
        if delta <= tolerance:
            break
        prev_sigma = sigma

    return sigma


def _largest_relative_gap_rank(singular_values: list[float]) -> tuple[int, float]:
    n = len(singular_values)
    if n == 0:
        return 0, 0.0
    if n == 1:
        return 1, float("inf")

    best_rank = 1
    best_ratio = 0.0

    for i in range(n - 1):
        left = singular_values[i]
        right = singular_values[i + 1]
        if left <= 0.0:
            continue
        ratio = float("inf") if right <= 0.0 else left / right
        if ratio > best_ratio:
            best_ratio = ratio
            best_rank = i + 1

    if best_ratio == 0.0:
        return 1, 1.0
    return best_rank, best_ratio


def _gram_eigh_singular_values(weight: "Array", backend: "Backend") -> list[float]:
    b = backend
    m, n = int(weight.shape[0]), int(weight.shape[1])
    if m <= n:
        gram = b.matmul(weight, b.transpose(weight))
    else:
        gram = b.matmul(b.transpose(weight), weight)
    b.eval(gram)

    eigvals = b.eigvalsh(gram)
    b.eval(eigvals)
    eigvals_sorted = b.sort(eigvals)
    reverse_idx = b.arange(int(eigvals_sorted.shape[0]) - 1, -1, -1)
    eigvals_desc = b.take(eigvals_sorted, reverse_idx, axis=0)
    b.eval(eigvals_desc)
    eigvals_list = b.tolist(eigvals_desc)
    if isinstance(eigvals_list, (int, float)):
        eigvals_values = [float(eigvals_list)]
    else:
        eigvals_values = [float(v) for v in eigvals_list]

    singular_values: list[float] = []
    for eig in eigvals_values:
        if eig <= 0.0:
            singular_values.append(0.0)
        else:
            singular_values.append(math.sqrt(eig))
    return singular_values


def _iterative_singular_values(weight: "Array", backend: "Backend") -> list[float]:
    b = backend
    m, n = int(weight.shape[0]), int(weight.shape[1])
    min_dim = min(m, n)
    if min_dim == 0:
        return []

    residual = weight
    b.eval(residual)
    first_sigma = 0.0
    threshold = 0.0
    singular_values: list[float] = []

    for _ in range(min_dim):
        sigma, u, v = _top_singular_triplet(residual, b)
        if sigma <= 0.0:
            break
        if first_sigma <= 0.0:
            first_sigma = sigma
            threshold = first_sigma * _SQRT_EPS_F32
        if sigma <= threshold:
            break

        singular_values.append(sigma)

        uv_t = b.matmul(
            b.reshape(u, (-1, 1)),
            b.reshape(v, (1, -1)),
        )
        residual = residual - (sigma * uv_t)
        b.eval(residual)

    return singular_values


def _top_singular_triplet(
    matrix: "Array",
    backend: "Backend",
) -> tuple[float, "Array", "Array"]:
    b = backend
    m, n = int(matrix.shape[0]), int(matrix.shape[1])
    if m == 0 or n == 0:
        return 0.0, b.zeros((m,)), b.zeros((n,))

    max_iters = max(2, min(m, n))
    sqrt_eps = _SQRT_EPS_F32

    v = b.ones((n,))
    v_norm = b.norm(v)
    b.eval(v_norm)
    v_norm_val = float(b.to_scalar(v_norm))
    if v_norm_val <= 0.0:
        return 0.0, b.zeros((m,)), b.zeros((n,))
    v = v / v_norm_val
    b.eval(v)

    sigma = 0.0
    prev_sigma = 0.0
    u = b.zeros((m,))

    for _ in range(max_iters):
        u = b.matmul(matrix, b.reshape(v, (-1, 1)))
        u = b.reshape(u, (-1,))
        u_norm = b.norm(u)
        b.eval(u_norm)
        u_norm_val = float(b.to_scalar(u_norm))
        if u_norm_val <= 0.0:
            return 0.0, b.zeros((m,)), b.zeros((n,))
        u = u / u_norm_val
        b.eval(u)

        v = b.matmul(b.transpose(matrix), b.reshape(u, (-1, 1)))
        v = b.reshape(v, (-1,))
        v_norm = b.norm(v)
        b.eval(v_norm)
        sigma = float(b.to_scalar(v_norm))
        if sigma <= 0.0:
            return 0.0, b.zeros((m,)), b.zeros((n,))
        v = v / sigma
        b.eval(v)

        delta = abs(sigma - prev_sigma)
        tolerance = sqrt_eps * max(1.0, sigma)
        if delta <= tolerance:
            break
        prev_sigma = sigma

    return sigma, u, v


def _frobenius_norm(matrix: "Array", backend: "Backend") -> float:
    b = backend
    frob = b.norm(matrix)
    b.eval(frob)
    return float(b.to_scalar(frob))


__all__ = [
    "LayerCapacityReport",
    "SpectralCapacityAnalyzer",
]

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

Numerical thresholds (rank, gap, noise floor) are derived from IEEE-754
machine precision. Spectral decay classification uses convenience labels
with explicit cutoffs documented in classify_spectral_decay().
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


_EPS_F32 = math.ldexp(1.0, -23)  # IEEE-754 float32 machine epsilon
_EPS_F16 = math.ldexp(1.0, -10)  # IEEE-754 float16 machine epsilon
_SQRT_EPS_F32 = math.sqrt(_EPS_F32)
_SQRT_EPS_F16 = math.sqrt(_EPS_F16)

# Noise floor for 2nd finite difference of normalized energy curve from float32 SVD.
# Stencil L1 norm (4) × relative error from squaring SVs (2ε_f32) = 8ε_f32.
_D2_NOISE_F32 = 8.0 * _EPS_F32


class SpectralDecayType(str, Enum):
    """Classification of singular value decay shape.

    Derived from the spectral structure, not heuristics:
    - sharp_cliff: numerically significant gap at effective rank boundary
    - power_law: log-log linear decay (R^2 > 0.95)
    - gradual_slope: smooth decay, neither cliff nor power law
    - flat: nearly uniform spectrum (stable_rank/min_dim > 0.9)
    """

    SHARP_CLIFF = "sharp_cliff"
    POWER_LAW = "power_law"
    GRADUAL_SLOPE = "gradual_slope"
    FLAT = "flat"


@dataclass(frozen=True)
class EnergyFractions:
    """GRASP-style energy concentration metrics.

    Measures what fraction of total spectral energy (sum of sigma_i^2)
    is captured by the top K% of singular values.

    Reference: GRASP (EMNLP 2025) found top 10% retains 90% reasoning.
    """

    top_10pct: float  # energy fraction in top 10% of SVs
    top_20pct: float  # energy fraction in top 20% of SVs
    top_50pct: float  # energy fraction in top 50% of SVs

    def to_dict(self) -> dict[str, float]:
        return {
            "top10pct": self.top_10pct,
            "top20pct": self.top_20pct,
            "top50pct": self.top_50pct,
        }


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
    decay_type: SpectralDecayType | None = None
    energy_fractions: EnergyFractions | None = None
    power_law_exponent: float | None = None
    power_law_r_squared: float | None = None
    shannon_effective_rank: float | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
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
        if self.decay_type is not None:
            result["decayType"] = self.decay_type.value
        if self.energy_fractions is not None:
            result["energyFractions"] = self.energy_fractions.to_dict()
        if self.power_law_exponent is not None:
            result["powerLawExponent"] = self.power_law_exponent
        if self.power_law_r_squared is not None:
            result["powerLawRSquared"] = self.power_law_r_squared
        if self.shannon_effective_rank is not None:
            result["shannonEffectiveRank"] = self.shannon_effective_rank
        return result


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

        shannon_eff_rank = _shannon_effective_rank(sv) if len(sv) > 1 else float(len(sv))
        decay_type, pl_exponent, pl_r2 = classify_spectral_decay(
            sv, shannon_eff_rank, stable_rank, min_dim
        )
        energy = compute_energy_fractions(sv)

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
            decay_type=decay_type,
            energy_fractions=energy,
            power_law_exponent=pl_exponent,
            power_law_r_squared=pl_r2,
            shannon_effective_rank=shannon_eff_rank,
        )


def _shannon_effective_rank(singular_values: list[float]) -> float:
    """Shannon effective rank: exp(spectral entropy).

    H = -sum(p_i * log(p_i)) where p_i = sigma_i^2 / sum(sigma_j^2).
    Effective rank = exp(H).
    """
    total_energy = sum(s * s for s in singular_values if s > 0.0)
    if total_energy <= 0.0:
        return 0.0

    entropy = 0.0
    for s in singular_values:
        if s <= 0.0:
            continue
        p = (s * s) / total_energy
        if p > 0.0:
            entropy -= p * math.log(p)

    return math.exp(entropy)


def compute_energy_fractions(singular_values: list[float]) -> EnergyFractions:
    """Compute GRASP-style energy concentration metrics.

    Returns what fraction of total spectral energy (sum sigma_i^2)
    is captured by top 10%, 20%, 50% of singular values.

    Reference: GRASP (EMNLP 2025) — top 10% retains 90% reasoning performance.
    """
    n = len(singular_values)
    if n == 0:
        return EnergyFractions(top_10pct=0.0, top_20pct=0.0, top_50pct=0.0)

    energies = [s * s for s in singular_values]
    total = sum(energies)
    if total <= 0.0:
        return EnergyFractions(top_10pct=0.0, top_20pct=0.0, top_50pct=0.0)

    def _cumulative_fraction(pct: float) -> float:
        k = max(1, int(math.ceil(n * pct)))
        return sum(energies[:k]) / total

    return EnergyFractions(
        top_10pct=_cumulative_fraction(0.10),
        top_20pct=_cumulative_fraction(0.20),
        top_50pct=_cumulative_fraction(0.50),
    )


def compute_full_energy_curve(singular_values: list[float]) -> list[float]:
    """Cumulative energy curve: E(k) = sum(sigma_i^2 for i=0..k) / total.

    Returns a list of length len(singular_values) where curve[k] is the
    fraction of total spectral energy captured by the top k+1 singular values.
    Monotonically non-decreasing, converges to 1.0.
    """
    if not singular_values:
        return []
    energies = [s * s for s in singular_values]
    total = sum(energies)
    if total <= 0.0:
        return [0.0] * len(singular_values)
    cumsum = 0.0
    curve: list[float] = []
    for e in energies:
        cumsum += e
        curve.append(cumsum / total)
    return curve


def find_energy_inflection_points(
    energy_curve: list[float],
    # 10x noise floor treats rounding residuals as ~Gaussian around 0 and
    # drives two-sided false-positive tail mass to near-zero (|z|>=10).
    min_prominence: float = 10.0 * _D2_NOISE_F32,
) -> list[dict[str, object]]:
    """Find inflection points in an energy accumulation curve.

    Detects positions where the second derivative of E(k) has local
    extrema — i.e., where the rate of energy capture changes most rapidly.

    Args:
        energy_curve: Cumulative energy fractions from compute_full_energy_curve.
        min_prominence: Minimum |d2E| to report. Default is 10× the
            second-difference noise floor for float32 SVD inputs
            (10 × 8ε_f32 ≈ 9.5e-6), which suppresses rounding-noise
            false positives under a standard Gaussian-tail approximation.

    Returns:
        List of dicts sorted by prominence (largest |d2E| first):
        {"rank": int, "d2E": float, "abs_d2E": float, "energy_at_rank": float}
    """
    if len(energy_curve) < 4:
        return []

    # First derivative: energy contribution per additional SV
    dE = [energy_curve[i + 1] - energy_curve[i] for i in range(len(energy_curve) - 1)]

    # Second derivative: change in energy contribution rate
    d2E = [dE[i + 1] - dE[i] for i in range(len(dE) - 1)]

    # Find local extrema in |d2E|
    inflection_points: list[dict[str, object]] = []
    for i in range(1, len(d2E) - 1):
        abs_val = abs(d2E[i])
        if abs_val > abs(d2E[i - 1]) and abs_val > abs(d2E[i + 1]):
            if abs_val >= min_prominence:
                inflection_points.append(
                    {
                        "rank": i + 2,  # +2: d2E offset by 1, energy_curve is 0-indexed
                        "d2E": d2E[i],
                        "abs_d2E": abs_val,
                        "energy_at_rank": energy_curve[i + 1],
                    }
                )

    return sorted(inflection_points, key=lambda x: x["abs_d2E"], reverse=True)  # type: ignore[arg-type]


def classify_spectral_decay(
    singular_values: list[float],
    shannon_effective_rank: float,
    stable_rank: float,
    min_dim: int,
) -> tuple[SpectralDecayType, float | None, float | None]:
    """Classify spectral decay shape from singular value distribution.

    Returns (decay_type, power_law_exponent, power_law_r_squared).
    power_law_exponent and power_law_r_squared are None unless decay is POWER_LAW.

    Classification uses convenience labels for reporting. The raw measurements
    (stable_rank, shannon_effective_rank, power_law_r_squared) are always
    returned in LayerCapacityReport for callers to apply their own criteria.

    Label assignment rules:
    1. FLAT: stable_rank / min_dim > 1 - sqrt(eps_f32) (energy nearly uniform)
    2. SHARP_CLIFF: gap at Shannon rank boundary > sqrt(eps_f32) * sigma_max
    3. POWER_LAW: log-log OLS regression R^2 > 1 - sqrt(eps_f32)
    4. GRADUAL_SLOPE: none of the above
    """
    n = len(singular_values)
    if n < 2:
        return SpectralDecayType.FLAT, None, None

    # 1. Flat spectrum: stable_rank close to min_dim
    # Threshold: 1 - sqrt(eps_f32). Flat means "indistinguishable from uniform
    # within float32 precision" — the gap between stable_rank/min_dim and 1.0
    # is smaller than the relative precision of the SVD computation.
    if min_dim > 0 and stable_rank / float(min_dim) > 1.0 - _SQRT_EPS_F32:
        return SpectralDecayType.FLAT, None, None

    # 2. Sharp cliff at Shannon effective rank boundary
    k = int(math.floor(shannon_effective_rank))
    if 0 < k < n:
        sigma_max = singular_values[0] if singular_values[0] > 0.0 else 1.0
        gap = singular_values[k - 1] - singular_values[min(k, n - 1)]
        cliff_threshold = _SQRT_EPS_F32 * sigma_max
        if gap > cliff_threshold:
            return SpectralDecayType.SHARP_CLIFF, None, None

    # 3. Power law: log(sigma_i) = a - alpha * log(i)
    # R^2 threshold: 1 - sqrt(eps_f32). The residual from the fit must be
    # within the relative precision of float32 SVD outputs.
    exponent, r_squared = _power_law_fit(singular_values)
    if r_squared is not None and r_squared > 1.0 - _SQRT_EPS_F32:
        return SpectralDecayType.POWER_LAW, exponent, r_squared

    # 4. Gradual slope (default)
    return SpectralDecayType.GRADUAL_SLOPE, None, None


def _power_law_fit(singular_values: list[float]) -> tuple[float | None, float | None]:
    """Fit sigma_i ~ C * i^(-alpha) via OLS on log-log data.

    Returns (alpha, R^2). Both None if insufficient data.
    Uses only positive singular values. Pure math, no framework calls.
    """
    # Collect (log(index), log(sigma)) pairs for positive SVs
    log_x: list[float] = []
    log_y: list[float] = []
    for i, s in enumerate(singular_values):
        if s <= 0.0:
            break
        log_x.append(math.log(float(i + 1)))
        log_y.append(math.log(s))

    n = len(log_x)
    if n < 3:
        return None, None

    # OLS: y = a + b*x where b = -alpha
    sum_x = sum(log_x)
    sum_y = sum(log_y)
    sum_xy = sum(x * y for x, y in zip(log_x, log_y))
    sum_x2 = sum(x * x for x in log_x)

    denom = n * sum_x2 - sum_x * sum_x
    # Underflow guard: float64 machine epsilon ≈ 2.2e-16 (Python float is f64)
    _EPS_F64 = math.ldexp(1.0, -52)
    if abs(denom) < _EPS_F64:
        return None, None

    b = (n * sum_xy - sum_x * sum_y) / denom
    a = (sum_y - b * sum_x) / n

    # R^2 = 1 - SS_res / SS_tot
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in log_y)
    if ss_tot < _EPS_F64:
        return None, None

    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(log_x, log_y))
    r_squared = 1.0 - ss_res / ss_tot

    alpha = -b  # power law exponent (positive = decay)
    return alpha, r_squared


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
        logger.warning(
            "SVD failed (known MLX crash on ill-conditioned matrices). "
            "Falling back to gram eigenvalues — spectral accuracy degraded.",
            exc_info=True,
        )

    try:
        gram_values = _gram_eigh_singular_values(weight_f32, b)
        if gram_values:
            return gram_values, "gram_eigh"
    except Exception:
        logger.warning(
            "Gram eigenvalue fallback also failed. "
            "Falling back to power deflation — spectral accuracy severely degraded.",
            exc_info=True,
        )

    iterative_values = _iterative_singular_values(weight_f32, b)
    if iterative_values:
        logger.warning(
            "Using power deflation for singular values. "
            "Budget bounds computed from this method are unreliable."
        )
        return iterative_values, "power_deflation"

    spectral_fallback = _spectral_norm_power_iteration(weight_f32, b)
    if spectral_fallback > 0.0:
        logger.warning(
            "Using single power-iteration spectral norm. "
            "Only σ_max available — all spectral analysis is unreliable."
        )
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
    "EnergyFractions",
    "LayerCapacityReport",
    "SpectralCapacityAnalyzer",
    "SpectralDecayType",
    "classify_spectral_decay",
    "compute_energy_fractions",
]

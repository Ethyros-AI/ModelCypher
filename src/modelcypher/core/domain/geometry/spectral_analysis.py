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

"""Spectral analysis for model weight matrices.

Computes spectral metrics (condition numbers, singular value ratios) to assess
relationships between source and target weight matrices.

Also provides principal subspace angle measurement for comparing subspaces
before and after perturbation (Björck & Golub, 1973).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    condition_threshold,
    division_epsilon,
    geodesic_svd,
    precision_dtype,
)
from modelcypher.core.domain.geometry.riemannian_utils import geodesic_norms

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralMetrics:
    """Spectral metrics for a weight matrix pair."""

    # Condition number of the target weight matrix
    condition_number: float

    # Ratio of max singular values: source / target
    spectral_ratio: float

    # Symmetric ratio: min(ratio, 1/ratio) in [0, 1]
    spectral_ratio_symmetry: float

    # Max singular value of source
    source_spectral_norm: float

    # Max singular value of target
    target_spectral_norm: float

    # Frobenius norm of the difference
    delta_frobenius: float


# =============================================================================
# NO CONFIGURATION CLASSES
# =============================================================================
# All parameters are derived from data:
# - epsilon: derived from dtype machine epsilon
# - max_condition_number: derived from dtype precision
# - SVD computation: always computes all singular values (needed for condition number)
# =============================================================================


def _to_float(val: Any) -> float:
    """Convert any scalar (including backend scalars) to Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def compute_spectral_metrics(
    source_weight: "Array",
    target_weight: "Array",
    backend: "Backend | None" = None,
) -> SpectralMetrics:
    """
    Compute spectral metrics for a weight matrix pair.

    All parameters are derived from data:
    - epsilon: from dtype machine epsilon
    - max_condition_number: from dtype precision

    Args:
        source_weight: Source model weight matrix [out_dim, in_dim] or [dim]
        target_weight: Target model weight matrix (same shape)
        backend: Optional Backend for GPU-accelerated SVD.

    Returns:
        SpectralMetrics with condition number, spectral ratio, and ratio symmetry
    """

    b = backend or get_default_backend()
    # Derive all thresholds from dtype
    eps = division_epsilon(b, target_weight)
    max_condition_number = condition_threshold(b, target_weight)

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, use vector norms instead of singular values
        source_norm_arr = geodesic_norms(b.reshape(source_weight, (1, -1)), b)
        target_norm_arr = geodesic_norms(b.reshape(target_weight, (1, -1)), b)
        delta_norm_arr = geodesic_norms(
            b.reshape(source_weight - target_weight, (1, -1)), b
        )
        b.eval(source_norm_arr, target_norm_arr, delta_norm_arr)
        source_norm = float(b.to_scalar(source_norm_arr[0]))
        target_norm = float(b.to_scalar(target_norm_arr[0]))
        delta_norm = float(b.to_scalar(delta_norm_arr[0]))

        if target_norm < eps:
            target_norm = eps

        ratio = source_norm / target_norm
        ratio_symmetry = min(ratio, 1.0 / max(ratio, eps))

        return SpectralMetrics(
            condition_number=1.0,  # 1D vectors don't have condition numbers
            spectral_ratio=ratio,
            spectral_ratio_symmetry=ratio_symmetry,
            source_spectral_norm=source_norm,
            target_spectral_norm=target_norm,
            delta_frobenius=delta_norm,
        )

    # 2D weight matrices - use backend for GPU acceleration
    source_arr = b.array(source_weight) if not hasattr(source_weight, "shape") else source_weight
    target_arr = b.array(target_weight) if not hasattr(target_weight, "shape") else target_weight

    # Promote for SVD (bfloat16 not supported)
    source_f32 = b.astype(source_arr, precision_dtype(b, reference=source_arr))
    target_f32 = b.astype(target_arr, precision_dtype(b, reference=target_arr))

    # Geodesic SVD - GPU-only power iteration, iterates until convergence
    # For (vocab_size, hidden_dim) matrices, we compute the full SVD
    # We need all singular values for condition number (σ_max / σ_min)
    _, source_s, _ = geodesic_svd(b, source_f32)
    _, target_s, _ = geodesic_svd(b, target_f32)

    # Evaluate and extract values
    b.eval(source_s, target_s)

    # Extract values
    source_len = int(source_s.shape[0])
    target_len = int(target_s.shape[0])
    if source_len > 0:
        source_s0 = b.take(source_s, b.array([0]), axis=0)
        source_s0 = b.squeeze(source_s0)
        b.eval(source_s0)
        source_spectral = float(b.to_scalar(source_s0))
    else:
        source_spectral = eps

    if target_len > 0:
        target_s0 = b.take(target_s, b.array([0]), axis=0)
        target_s0 = b.squeeze(target_s0)
        b.eval(target_s0)
        target_spectral = float(b.to_scalar(target_s0))

        target_last = b.take(target_s, b.array([target_len - 1]), axis=0)
        target_last = b.squeeze(target_last)
        b.eval(target_last)
        target_min_s = float(b.to_scalar(target_last))
    else:
        target_spectral = eps
        target_min_s = eps

    # Delta Frobenius norm
    delta_arr = geodesic_norms(b.reshape(source_arr - target_arr, (1, -1)), b)
    b.eval(delta_arr)
    delta_frobenius = float(b.to_scalar(delta_arr[0]))

    # Condition number of target
    condition_number = target_spectral / max(target_min_s, eps)
    condition_number = min(condition_number, max_condition_number)

    # Spectral ratio
    spectral_ratio = source_spectral / max(target_spectral, eps)

    # Spectral ratio symmetry
    if spectral_ratio > 0:
        spectral_ratio_symmetry = min(spectral_ratio, 1.0 / spectral_ratio)
    else:
        spectral_ratio_symmetry = 0.0

    return SpectralMetrics(
        condition_number=condition_number,
        spectral_ratio=spectral_ratio,
        spectral_ratio_symmetry=spectral_ratio_symmetry,
        source_spectral_norm=source_spectral,
        target_spectral_norm=target_spectral,
        delta_frobenius=delta_frobenius,
    )


def spectral_summary(metrics: dict[str, SpectralMetrics]) -> dict:
    """
    Summarize spectral metrics across all weight matrices.

    Args:
        metrics: Per-weight spectral metrics

    Returns:
        Summary statistics
    """
    if not metrics:
        return {
            "total_weights": 0,
            "mean_ratio_symmetry": 0.0,
            "mean_condition_number": 0.0,
        }

    ratio_symmetry = [m.spectral_ratio_symmetry for m in metrics.values()]
    conditions = [m.condition_number for m in metrics.values()]

    return {
        "total_weights": len(metrics),
        "mean_ratio_symmetry": sum(ratio_symmetry) / len(ratio_symmetry),
        "min_ratio_symmetry": min(ratio_symmetry),
        "max_ratio_symmetry": max(ratio_symmetry),
        "mean_condition_number": sum(conditions) / len(conditions),
        "max_condition_number": max(conditions),
    }


def principal_subspace_angle(
    V_base: "Array",
    V_adapted: "Array",
    backend: "Backend | None" = None,
) -> float:
    """Largest principal angle between two k-dimensional subspaces.

    Given two sets of k orthonormal row vectors spanning k-dimensional
    subspaces, computes the largest principal angle between them.

    Björck & Golub (1973): the principal angles are arccos(σ_i) where
    σ_i are the singular values of V_base @ V_adapted^T.  The largest
    angle corresponds to the smallest singular value.

    Euclidean weight space — standard SVD, no geodesic machinery.

    Args:
        V_base: [k, n] — top-k right singular vectors of the base weight matrix.
            Rows must be orthonormal.
        V_adapted: [k, n] — top-k right singular vectors of the adapted weight
            matrix. Same k and n as V_base.
        backend: Backend for SVD computation.

    Returns:
        Largest principal angle in radians.  0.0 = identical subspaces,
        π/2 = orthogonal along at least one direction.
    """
    b = backend or get_default_backend()

    V_b = b.astype(V_base, precision_dtype(b, reference=V_base))
    V_a = b.astype(V_adapted, precision_dtype(b, reference=V_adapted))

    # M = V_base @ V_adapted^T  — shape [k, k]
    M = b.matmul(V_b, b.transpose(V_a))

    # SVD of the k×k cross-product matrix
    _, sigmas, _ = geodesic_svd(b, M)
    b.eval(sigmas)

    n_sigmas = int(sigmas.shape[0])
    if n_sigmas == 0:
        return math.pi / 2.0

    # Smallest singular value → largest principal angle
    sigma_min_arr = b.take(sigmas, b.array([n_sigmas - 1]), axis=0)
    sigma_min_arr = b.squeeze(sigma_min_arr)
    b.eval(sigma_min_arr)
    sigma_min = float(b.to_scalar(sigma_min_arr))

    # Clamp to [0, 1] for numerical safety before arccos
    sigma_min = max(0.0, min(1.0, sigma_min))

    return math.acos(sigma_min)

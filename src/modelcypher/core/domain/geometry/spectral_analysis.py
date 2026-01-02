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

"""
Spectral Analysis for Model Weight Matrices.

Computes spectral metrics (condition numbers, singular value ratios) to assess
the geometric relationship between source and target weight matrices.

This module provides RAW MEASUREMENTS only. No alpha blending, no interpretation.
Use these metrics to understand transformation effort - decisions are yours.

Mathematical Foundation
-----------------------
For weight matrices W_source and W_target:

1. Spectral ratio = σ_max(W_source) / σ_max(W_target)
   - σ_max = largest singular value
   - Ratio near 1.0 = similar representation scales

2. Spectral confidence = min(ratio, 1/ratio)
   - Symmetric: both 2.0 and 0.5 give same confidence
   - Range: [0, 1], higher = more similar scales

3. Condition number = σ_max / σ_min
   - Higher = more ill-conditioned
   - Derived threshold from dtype precision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    condition_threshold,
    division_epsilon,
    svd_via_eigh,
)

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

    # Symmetric confidence: min(ratio, 1/ratio) in [0, 1]
    spectral_confidence: float

    # Max singular value of source
    source_spectral_norm: float

    # Max singular value of target
    target_spectral_norm: float

    # Frobenius norm of the difference
    delta_frobenius: float


@dataclass(frozen=True)
class SpectralConfig:
    """Configuration for spectral analysis.

    All thresholds are derived from dtype precision when None.
    """

    # Epsilon for numerical stability (None = derived from dtype)
    epsilon: float | None = None

    # Maximum condition number before clamping (None = derived from dtype)
    max_condition_number: float | None = None

    # Whether to use full SVD (slower but more accurate) or just top-k
    use_full_svd: bool = False

    # Number of singular values to compute if not full
    top_k: int = 10


def _to_float(val: Any) -> float:
    """Convert any scalar (including MLX) to Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def compute_spectral_metrics(
    source_weight: "Array",
    target_weight: "Array",
    config: SpectralConfig,
    backend: "Backend | None" = None,
) -> SpectralMetrics:
    """
    Compute spectral metrics for a weight matrix pair.

    Args:
        source_weight: Source model weight matrix [out_dim, in_dim] or [dim]
        target_weight: Target model weight matrix (same shape)
        config: Spectral analysis configuration (use with_parameters() to create).
        backend: Optional Backend for GPU-accelerated SVD.

    Returns:
        SpectralMetrics with condition number, spectral ratio, and confidence
    """

    b = backend or get_default_backend()
    eps = config.epsilon if config.epsilon is not None else division_epsilon(b, target_weight)
    max_condition_number = (
        config.max_condition_number
        if config.max_condition_number is not None
        else condition_threshold(b, target_weight)
    )

    # Handle 1D weights (biases, layernorms)
    if source_weight.ndim == 1:
        # For 1D, use vector norms instead of singular values
        source_norm = _to_float(b.norm(source_weight))
        target_norm = _to_float(b.norm(target_weight))
        delta_norm = _to_float(b.norm(source_weight - target_weight))

        if target_norm < eps:
            target_norm = eps

        ratio = source_norm / target_norm
        confidence = min(ratio, 1.0 / max(ratio, eps))

        return SpectralMetrics(
            condition_number=1.0,  # 1D vectors don't have condition numbers
            spectral_ratio=ratio,
            spectral_confidence=confidence,
            source_spectral_norm=source_norm,
            target_spectral_norm=target_norm,
            delta_frobenius=delta_norm,
        )

    # 2D weight matrices - use backend for GPU acceleration
    source_arr = b.array(source_weight) if not hasattr(source_weight, "shape") else source_weight
    target_arr = b.array(target_weight) if not hasattr(target_weight, "shape") else target_weight

    # Cast to float32 for SVD (bfloat16 not supported)
    source_f32 = b.astype(source_arr, "float32")
    target_f32 = b.astype(target_arr, "float32")

    # SVD - compute only singular values (not U or Vt) to avoid 92GB allocation
    # For (vocab_size, hidden_dim) matrices, full U would be vocab_size^2
    _, source_s, _ = svd_via_eigh(b, source_f32, full_matrices=False)
    _, target_s, _ = svd_via_eigh(b, target_f32, full_matrices=False)

    # Limit to top_k if not using full SVD
    if not config.use_full_svd:
        source_s = source_s[: config.top_k]
        target_s = target_s[: config.top_k]

    # Evaluate and extract values
    b.eval(source_s, target_s)

    # Extract values
    source_s_np = b.to_numpy(source_s)
    target_s_np = b.to_numpy(target_s)

    source_spectral = float(source_s_np[0]) if len(source_s_np) > 0 else eps
    target_spectral = float(target_s_np[0]) if len(target_s_np) > 0 else eps
    target_min_s = float(target_s_np[-1]) if len(target_s_np) > 0 else eps

    # Delta Frobenius norm
    delta_arr = b.norm(source_arr - target_arr)
    b.eval(delta_arr)
    delta_frobenius = _to_float(delta_arr)

    # Condition number of target
    condition_number = target_spectral / max(target_min_s, eps)
    condition_number = min(condition_number, max_condition_number)

    # Spectral ratio
    spectral_ratio = source_spectral / max(target_spectral, eps)

    # Spectral confidence (symmetric)
    if spectral_ratio > 0:
        spectral_confidence = min(spectral_ratio, 1.0 / spectral_ratio)
    else:
        spectral_confidence = 0.0

    return SpectralMetrics(
        condition_number=condition_number,
        spectral_ratio=spectral_ratio,
        spectral_confidence=spectral_confidence,
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
            "mean_confidence": 0.0,
            "mean_condition_number": 0.0,
        }

    confidences = [m.spectral_confidence for m in metrics.values()]
    conditions = [m.condition_number for m in metrics.values()]

    return {
        "total_weights": len(metrics),
        "mean_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "mean_condition_number": sum(conditions) / len(conditions),
        "max_condition_number": max(conditions),
    }

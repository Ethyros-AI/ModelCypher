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
Output Layer Stability Analysis for Model Merging.

Measures numerical stability properties of weight matrices that affect generation quality.
Specifically designed to track changes in the output layer (lm_head) across merges.

Mathematical Foundation
-----------------------
For a weight matrix W with singular values σ₁ ≥ σ₂ ≥ ... ≥ σᵣ:

1. **Condition Number** κ(W) = σ₁ / σᵣ
   - Measures sensitivity of outputs to input perturbations
   - High κ = small input changes → large output changes = numerically unstable
   - Critical for generation: high κ means token predictions are volatile

2. **Spectral Entropy** H(W) = -∑ (σᵢ/∑σⱼ) * log(σᵢ/∑σⱼ)
   - Measures how "spread out" the singular value distribution is
   - High entropy = energy distributed across many dimensions
   - Low entropy = energy concentrated in few dimensions (potentially brittle)

3. **Effective Rank** r(W) = (∑σᵢ)² / ∑σᵢ²
   - Smooth measure of "how many dimensions matter"
   - Range: [1, min(m,n)]
   - Stable models have effective rank >> 1

These metrics track numerical stability - distinct from geometric alignment (CKA).
A merged model can have high geodesic CKA (geometry overlap) but degraded stability
(high condition number) leading to broken generation.

Usage:
    from modelcypher.core.domain.geometry.output_stability import (
        compute_output_stability,
        OutputStabilityMetrics,
    )

    # Analyze a weight matrix
    metrics = compute_output_stability(lm_head_weights, backend)
    print(f"Condition number: {metrics.condition_number}")
    print(f"Spectral entropy: {metrics.spectral_entropy}")

    # Compare before/after merge
    before = compute_output_stability(target_lm_head, backend)
    after = compute_output_stability(merged_lm_head, backend)
    delta = compare_stability(before, after)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    condition_threshold,
    division_epsilon,
    geodesic_svd,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputStabilityMetrics:
    """Stability metrics for a weight matrix."""

    # Condition number: σ_max / σ_min
    # Higher = more numerically unstable
    condition_number: float

    # Spectral entropy: -∑ p_i * log(p_i) where p_i = σ_i / ∑σ
    # Higher = more distributed singular values
    spectral_entropy: float

    # Effective rank: (∑σ)² / ∑σ²
    # Higher = more dimensions "matter"
    effective_rank: float

    # Max singular value (spectral norm)
    spectral_norm: float

    # Min singular value (smallest mode)
    min_singular_value: float

    # Frobenius norm of the weight matrix
    frobenius_norm: float

    # Number of singular values computed
    n_singular_values: int


@dataclass(frozen=True)
class StabilityComparison:
    """Comparison of stability metrics between two weight matrices."""

    # Ratio of condition numbers: after / before
    # >1.0 means stability degraded, <1.0 means improved
    condition_number_ratio: float

    # Change in spectral entropy: after - before
    spectral_entropy_delta: float

    # Ratio of effective ranks: after / before
    effective_rank_ratio: float

    # Ratio of spectral norms: after / before
    spectral_norm_ratio: float

    # Absolute metrics
    before: OutputStabilityMetrics
    after: OutputStabilityMetrics


def _to_float(val: Any) -> float:
    """Convert any scalar (including MLX) to Python float."""
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)


def compute_output_stability(
    weight_matrix: "Array",
    backend: "Backend | None" = None,
) -> OutputStabilityMetrics:
    """
    Compute stability metrics for a weight matrix.

    All parameters are derived from data - no arbitrary thresholds.

    Args:
        weight_matrix: Weight matrix [out_dim, in_dim] or [dim] (1D treated specially)
        backend: Optional Backend for GPU-accelerated SVD.

    Returns:
        OutputStabilityMetrics with condition number, spectral entropy, effective rank
    """
    b = backend or get_default_backend()

    # Derive thresholds from dtype
    eps = division_epsilon(b, weight_matrix)
    max_condition = condition_threshold(b, weight_matrix)
    log_eps = safe_log_epsilon(b, weight_matrix)

    # Handle 1D weights (biases, layernorms)
    if weight_matrix.ndim == 1:
        # For 1D, these metrics don't apply meaningfully
        norm_arr = b.sqrt(b.sum(weight_matrix * weight_matrix))
        b.eval(norm_arr)
        frob_norm = float(b.to_scalar(norm_arr))

        return OutputStabilityMetrics(
            condition_number=1.0,  # 1D has no condition number
            spectral_entropy=0.0,  # Single "dimension"
            effective_rank=1.0,
            spectral_norm=frob_norm,
            min_singular_value=frob_norm,
            frobenius_norm=frob_norm,
            n_singular_values=1,
        )

    # 2D weight matrices - compute SVD
    W = b.array(weight_matrix) if not hasattr(weight_matrix, "shape") else weight_matrix
    W_f32 = b.astype(W, "float32")

    # Compute full SVD (exact, no approximation)
    _, S, _ = geodesic_svd(b, W_f32)
    b.eval(S)

    n_sv = int(S.shape[0])
    if n_sv == 0:
        # Degenerate matrix
        return OutputStabilityMetrics(
            condition_number=float("inf"),
            spectral_entropy=0.0,
            effective_rank=0.0,
            spectral_norm=0.0,
            min_singular_value=0.0,
            frobenius_norm=0.0,
            n_singular_values=0,
        )

    # Extract singular values
    S_max_arr = b.take(S, b.array([0]), axis=0)
    S_min_arr = b.take(S, b.array([n_sv - 1]), axis=0)
    b.eval(S_max_arr, S_min_arr)

    sigma_max = float(b.to_scalar(b.squeeze(S_max_arr)))
    sigma_min = float(b.to_scalar(b.squeeze(S_min_arr)))

    # Condition number: σ_max / σ_min
    condition_number = sigma_max / max(sigma_min, eps)
    condition_number = min(condition_number, max_condition)

    # Frobenius norm: sqrt(∑σ²)
    S_sq = S * S
    frob_sq = b.sum(S_sq)
    b.eval(frob_sq)
    frob_norm = float(b.to_scalar(b.sqrt(frob_sq)))

    # Sum of singular values
    S_sum = b.sum(S)
    b.eval(S_sum)
    sigma_sum = float(b.to_scalar(S_sum))

    # Effective rank: (∑σ)² / ∑σ²
    frob_sq_val = float(b.to_scalar(frob_sq))
    if frob_sq_val > eps:
        effective_rank = (sigma_sum * sigma_sum) / frob_sq_val
    else:
        effective_rank = 0.0

    # Spectral entropy: -∑ p_i * log(p_i) where p_i = σ_i / ∑σ
    if sigma_sum > eps:
        # Normalize to probability distribution
        P = S / sigma_sum
        b.eval(P)

        # Compute -p * log(p) for each, handling zeros
        eps_arr = b.full(P.shape, log_eps, dtype=b.dtype(P))
        P_safe = b.where(P > log_eps, P, eps_arr)
        log_P = b.log(P_safe)
        entropy_terms = -P * log_P
        b.eval(entropy_terms)

        spectral_entropy = float(b.to_scalar(b.sum(entropy_terms)))

        # Normalize to [0, 1] range: max entropy is log(n_sv)
        max_entropy = float(b.to_scalar(b.log(b.array([float(n_sv)]))))
        if max_entropy > eps:
            spectral_entropy = spectral_entropy / max_entropy
    else:
        spectral_entropy = 0.0

    return OutputStabilityMetrics(
        condition_number=condition_number,
        spectral_entropy=spectral_entropy,
        effective_rank=effective_rank,
        spectral_norm=sigma_max,
        min_singular_value=sigma_min,
        frobenius_norm=frob_norm,
        n_singular_values=n_sv,
    )


def compare_stability(
    before: OutputStabilityMetrics,
    after: OutputStabilityMetrics,
) -> StabilityComparison:
    """
    Compare stability metrics between two states (e.g., before/after merge).

    Args:
        before: Stability metrics before the operation
        after: Stability metrics after the operation

    Returns:
        StabilityComparison with ratios and deltas
    """
    # Use sqrt(machine_epsilon) for safe division
    # float32 machine epsilon = 2^-23, sqrt = 2^-11.5 ≈ 3.45e-4
    import math
    eps = math.sqrt(2.0 ** -23)  # sqrt of float32 machine epsilon

    return StabilityComparison(
        condition_number_ratio=after.condition_number / max(before.condition_number, eps),
        spectral_entropy_delta=after.spectral_entropy - before.spectral_entropy,
        effective_rank_ratio=after.effective_rank / max(before.effective_rank, eps),
        spectral_norm_ratio=after.spectral_norm / max(before.spectral_norm, eps),
        before=before,
        after=after,
    )


def analyze_weight_stability(
    weights: dict[str, "Array"],
    backend: "Backend | None" = None,
    filter_pattern: str | None = None,
) -> dict[str, OutputStabilityMetrics]:
    """
    Analyze stability of multiple weight matrices.

    Args:
        weights: Dictionary of weight name -> weight array
        backend: Optional Backend for GPU operations
        filter_pattern: Optional regex pattern to filter weight names

    Returns:
        Dictionary of weight name -> OutputStabilityMetrics
    """
    import re

    b = backend or get_default_backend()
    results = {}

    for name, weight in weights.items():
        if filter_pattern and not re.search(filter_pattern, name):
            continue

        try:
            metrics = compute_output_stability(weight, b)
            results[name] = metrics
        except Exception as e:
            logger.warning(f"Failed to analyze {name}: {e}")

    return results


def stability_summary(metrics: dict[str, OutputStabilityMetrics]) -> dict:
    """
    Summarize stability metrics across weight matrices.

    Args:
        metrics: Dictionary of weight name -> OutputStabilityMetrics

    Returns:
        Summary statistics
    """
    if not metrics:
        return {
            "total_weights": 0,
            "mean_condition_number": 0.0,
            "max_condition_number": 0.0,
            "mean_spectral_entropy": 0.0,
            "mean_effective_rank": 0.0,
        }

    conditions = [m.condition_number for m in metrics.values()]
    entropies = [m.spectral_entropy for m in metrics.values()]
    ranks = [m.effective_rank for m in metrics.values()]

    # Find the weight with highest condition number (most unstable)
    max_cond_name = max(metrics.keys(), key=lambda k: metrics[k].condition_number)

    return {
        "total_weights": len(metrics),
        "mean_condition_number": sum(conditions) / len(conditions),
        "max_condition_number": max(conditions),
        "max_condition_weight": max_cond_name,
        "mean_spectral_entropy": sum(entropies) / len(entropies),
        "min_spectral_entropy": min(entropies),
        "mean_effective_rank": sum(ranks) / len(ranks),
        "min_effective_rank": min(ranks),
    }

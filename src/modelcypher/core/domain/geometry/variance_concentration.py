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

"""Variance concentration metrics for bottleneck layer detection.

TwoNN intrinsic dimension measures local manifold curvature but FAILS to detect
true 1D bottlenecks where 99%+ of variance is concentrated in a single direction.

Example: LFM2-350M layer 7 has 99.4% variance in top-1 singular value, but
TwoNN reports ID=6.39 (not even close to 1).

This module provides SVD-based metrics that correctly identify bottlenecks:
- Variance concentration: % of variance in top-k singular values
- Effective rank: Entropy-based measure of dimensionality

References:
    - Roy & Bhattacharya (2007) "Effective Rank: A measure of matrix information"
    - Intrinsic dimension estimation via variance is standard in PCA literature
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass
class VarianceConcentrationResult:
    """Result of variance concentration analysis for a single layer."""

    # Variance concentration (top-1 singular value % of total variance)
    # 1.0 = all variance in one direction (perfect 1D bottleneck)
    # 0.01 = uniform variance across 100 directions (high dimensional)
    var_top1: float

    # Variance in top-k directions (k=1,2,3,5,10)
    var_top_k: dict[int, float]

    # Effective rank = exp(entropy of normalized singular values)
    # Low effective rank = bottleneck (information concentrated)
    # High effective rank = distributed (information spread out)
    effective_rank: float

    # Number of singular values computed
    n_singular_values: int

    # Total variance (for debugging)
    total_variance: float


def compute_variance_concentration(
    activations: "Array",
    backend: "Backend | None" = None,
    top_k: list[int] | None = None,
) -> VarianceConcentrationResult:
    """Compute variance concentration and effective rank from activations.

    This is the correct way to detect bottleneck layers. TwoNN intrinsic
    dimension fails because it measures local manifold curvature, not
    variance distribution.

    Algorithm:
    1. Center activations (subtract mean)
    2. Compute SVD: A = U @ S @ V.T
    3. Variance in direction i = σᵢ² (squared singular value)
    4. var_top1 = σ₁² / Σσᵢ² (top-1 concentration)
    5. effective_rank = exp(-Σ pᵢ log(pᵢ)) where pᵢ = σᵢ² / Σσⱼ²

    Args:
        activations: [n_samples, hidden_dim] activation matrix
        backend: Computation backend (uses default if None)
        top_k: List of k values for top-k variance (default [1,2,3,5,10])

    Returns:
        VarianceConcentrationResult with variance metrics
    """
    b = backend or get_default_backend()

    if top_k is None:
        top_k = [1, 2, 3, 5, 10]

    # Ensure 2D
    if len(b.shape(activations)) == 1:
        activations = b.reshape(activations, (1, -1))

    n_samples, hidden_dim = b.shape(activations)

    # Center activations (subtract mean per dimension)
    mean = b.mean(activations, axis=0, keepdims=True)
    centered = activations - mean

    # Compute SVD - we only need singular values
    # For n_samples x hidden_dim matrix:
    # - If n_samples < hidden_dim: compute A @ A.T (n x n) eigenvalues
    # - If n_samples >= hidden_dim: compute A.T @ A (d x d) eigenvalues
    # Singular values = sqrt(eigenvalues)
    if n_samples < hidden_dim:
        # Small batch: use A @ A.T
        gram = b.matmul(centered, b.transpose(centered))
        # Add small regularization for numerical stability
        eps = machine_epsilon(b, gram)
        gram = gram + eps * b.eye(n_samples)
        eigenvalues = b.eigvalsh(gram)
        b.eval(eigenvalues)
        # Singular values squared = eigenvalues (already σ²)
        # Sort descending (eigvalsh returns ascending)
        singular_sq = b.sort(eigenvalues)[::-1]
    else:
        # Normal case: use A.T @ A
        gram = b.matmul(b.transpose(centered), centered)
        eps = machine_epsilon(b, gram)
        gram = gram + eps * b.eye(hidden_dim)
        eigenvalues = b.eigvalsh(gram)
        b.eval(eigenvalues)
        singular_sq = b.sort(eigenvalues)[::-1]

    # Ensure non-negative (numerical precision can give tiny negatives)
    singular_sq = b.maximum(singular_sq, b.zeros_like(singular_sq))

    # Total variance = sum of squared singular values
    total_var = b.sum(singular_sq)
    b.eval(total_var)
    total_var_scalar = float(b.to_scalar(total_var))

    if total_var_scalar <= 0:
        return VarianceConcentrationResult(
            var_top1=0.0,
            var_top_k={k: 0.0 for k in top_k},
            effective_rank=0.0,
            n_singular_values=int(min(n_samples, hidden_dim)),
            total_variance=0.0,
        )

    # Compute variance concentration for top-k
    var_top_k_dict: dict[int, float] = {}
    n_sv = int(singular_sq.shape[0])

    for k in top_k:
        k_actual = min(k, n_sv)
        if k_actual > 0:
            top_k_sum = b.sum(singular_sq[:k_actual])
            b.eval(top_k_sum)
            var_top_k_dict[k] = float(b.to_scalar(top_k_sum)) / total_var_scalar
        else:
            var_top_k_dict[k] = 0.0

    var_top1 = var_top_k_dict.get(1, 0.0)

    # Compute effective rank via entropy
    # p_i = σᵢ² / Σσⱼ² (normalized variance)
    # effective_rank = exp(-Σ pᵢ log(pᵢ))
    normalized = singular_sq / total_var
    # Avoid log(0) - mask zero entries
    log_eps = safe_log_epsilon(b, normalized)
    normalized_safe = b.maximum(normalized, b.full(normalized.shape, log_eps))
    log_p = b.log(normalized_safe)
    # Only include non-zero entries in entropy
    non_zero_mask = normalized > log_eps
    entropy_terms = b.where(non_zero_mask, -normalized * log_p, b.zeros_like(normalized))
    entropy = b.sum(entropy_terms)
    effective_rank_arr = b.exp(entropy)
    b.eval(effective_rank_arr)
    effective_rank = float(b.to_scalar(effective_rank_arr))

    return VarianceConcentrationResult(
        var_top1=var_top1,
        var_top_k=var_top_k_dict,
        effective_rank=effective_rank,
        n_singular_values=n_sv,
        total_variance=total_var_scalar,
    )


def identify_bottleneck_layers(
    layer_metrics: dict[int, VarianceConcentrationResult],
    var_threshold: float = 0.70,
    effective_rank_threshold: float | None = None,
) -> list[int]:
    """Identify bottleneck layers from variance concentration metrics.

    Bottleneck = layer where information is most compressed:
    - High variance concentration (top-1 singular value dominates)
    - Low effective rank (few dimensions carry the information)

    For LFM2-350M:
    - Layer 7: var_top1=99.4%, effective_rank=9.8 (PRIMARY bottleneck)
    - Layer 14: var_top1=70.9%, effective_rank=67.6 (secondary bottleneck)

    Args:
        layer_metrics: Dict of layer_idx -> VarianceConcentrationResult
        var_threshold: Minimum var_top1 to be considered bottleneck (default 0.70)
        effective_rank_threshold: Optional max effective_rank threshold
            If None, uses median effective rank as threshold

    Returns:
        List of layer indices that are bottlenecks, sorted by var_top1 descending
    """
    if not layer_metrics:
        return []

    bottlenecks = []

    # Compute median effective rank if no threshold given
    if effective_rank_threshold is None:
        ranks = [m.effective_rank for m in layer_metrics.values()]
        ranks.sort()
        mid = len(ranks) // 2
        if len(ranks) % 2 == 0 and len(ranks) > 1:
            effective_rank_threshold = (ranks[mid - 1] + ranks[mid]) / 2
        else:
            effective_rank_threshold = ranks[mid] if ranks else 100.0

    for layer_idx, metrics in layer_metrics.items():
        # Bottleneck = high variance concentration OR low effective rank
        is_high_variance = metrics.var_top1 >= var_threshold
        is_low_rank = metrics.effective_rank <= effective_rank_threshold

        if is_high_variance or is_low_rank:
            bottlenecks.append((layer_idx, metrics.var_top1))

    # Sort by variance concentration descending
    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    return [layer_idx for layer_idx, _ in bottlenecks]

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

    # Activation matrix shape used for this estimate
    n_samples: int
    hidden_dim: int

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

    n_samples_raw, hidden_dim_raw = b.shape(activations)
    n_samples = int(n_samples_raw)
    hidden_dim = int(hidden_dim_raw)

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
            n_samples=n_samples,
            hidden_dim=hidden_dim,
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
        n_samples=n_samples,
        hidden_dim=hidden_dim,
        total_variance=total_var_scalar,
    )


def identify_bottleneck_layers(
    layer_metrics: dict[int, VarianceConcentrationResult],
) -> list[int]:
    """Identify bottleneck layers from variance concentration metrics.

    Finds the natural gap in the sorted var_top1 distribution across layers.
    Layers above the largest gap are bottlenecks. If no significant gap exists,
    the model has no bottleneck layers (all layers have similar concentration).

    Measured on LFM2 models (350M, 700M, 1.2B):
    - 350M: clear bimodal gap 0.40 → 0.69 separating layers 7-13
    - 700M: no bimodal gap (continuous 0.16-0.60), no bottlenecks
    The gap is a geometric property of the model's spectral structure.

    Args:
        layer_metrics: Dict of layer_idx -> VarianceConcentrationResult

    Returns:
        List of layer indices that are bottlenecks, sorted by var_top1 descending
    """
    if not layer_metrics:
        raise ValueError("No variance measurements available for bottleneck detection")

    # Sort layers by var_top1
    sorted_items = sorted(layer_metrics.items(), key=lambda x: x[1].var_top1)
    n = len(sorted_items)
    if n < 2:
        return []

    # Find the largest gap in the sorted var_top1 distribution
    max_gap = 0.0
    max_gap_idx = 0
    for i in range(n - 1):
        gap = sorted_items[i + 1][1].var_top1 - sorted_items[i][1].var_top1
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i

    # The gap must be geometrically significant: larger than the typical
    # spacing between adjacent values. Use the median of all spacings
    # EXCLUDING the max gap itself — this gives the background spacing level.
    spacings = [
        sorted_items[i + 1][1].var_top1 - sorted_items[i][1].var_top1
        for i in range(n - 1)
        if i != max_gap_idx
    ]
    if not spacings:
        # Only 2 layers — can't determine background spacing
        return []
    spacings.sort()
    background_spacing = spacings[len(spacings) // 2]

    # Gap must exceed background spacing by an order of magnitude.
    # Measured: 350M bimodal gap = 14.4× median, 700M largest fluctuation = 3.3×.
    # Threshold at 5× separates real bimodal structure from continuous spectra.
    if background_spacing <= 0 or max_gap <= 5.0 * background_spacing:
        return []  # No significant gap — no bottlenecks

    # Layers above the gap are bottlenecks
    threshold = sorted_items[max_gap_idx][1].var_top1
    bottlenecks = [
        (layer_idx, metrics.var_top1)
        for layer_idx, metrics in layer_metrics.items()
        if metrics.var_top1 > threshold
    ]
    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    return [layer_idx for layer_idx, _ in bottlenecks]

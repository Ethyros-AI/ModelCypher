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

"""Fisher Information for merge compatibility prediction.

Provides empirical Fisher computation, diagonal approximations, and
compatibility scores derived from Fisher statistics.

References:
    - Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in
      neural networks" - Elastic Weight Consolidation (EWC)
    - Ritter et al. (2018) "A scalable Laplace approximation for neural
      networks"
    - Matena & Raffel (2022) "Merging Models with Fisher-Weighted Averaging"
      (TIES-Merging uses Fisher-weighted merging)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    find_magnitude_gap_threshold,
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class FisherInformationResult:
    """Result of Fisher Information Matrix analysis."""

    # Diagonal FIM approximation [d] - per-parameter importance
    diagonal_fim: "Array"

    # Effective rank (participation ratio of FIM eigenvalues)
    effective_rank: float

    # Total curvature (trace of FIM)
    trace_fim: float

    # Condition number of diagonal FIM (max/min ratio)
    condition_number: float

    # Mean FIM value (average parameter importance)
    mean_fim: float

    # Number of significant directions (data-derived threshold)
    n_significant: int

    # Data-derived significance threshold for the diagonal FIM
    significance_threshold: float


@dataclass
class FisherCompatibilityResult:
    """Result of Fisher-based merge compatibility analysis."""

    # Compatibility score (cosine similarity of diagonal FIM)
    compatibility_score: float

    # Cosine similarity of diagonal FIM vectors
    cosine_similarity: float

    # Correlation of FIM values
    correlation: float

    # Overlap ratio: fraction of dimensions where both have significant FIM
    overlap_ratio: float

    # Source FIM analysis
    source_fisher: FisherInformationResult

    # Target FIM analysis
    target_fisher: FisherInformationResult

    # No recommendation - raw metrics only
    recommendation: str


def _significance_threshold(diagonal_fim: "Array", backend: "Backend") -> float:
    """Derive a significance threshold from precision + spectral gap."""
    abs_fim = backend.abs(diagonal_fim)
    sorted_vals = backend.sort(abs_fim)
    backend.eval(sorted_vals)

    eps = machine_epsilon(backend, abs_fim)
    gap_threshold = find_magnitude_gap_threshold(sorted_vals, eps=eps, backend=backend)

    max_val_arr = backend.max(abs_fim)
    backend.eval(max_val_arr)
    max_val = float(backend.to_scalar(max_val_arr))
    noise_floor = max_val * eps

    return max(noise_floor, gap_threshold)


def compute_empirical_fisher_diagonal(
    activations: "Array",
    backend: "Backend | None" = None,
) -> FisherInformationResult:
    """Compute diagonal empirical Fisher Information from activations.

    The empirical Fisher Information Matrix is:
        F = E[g_i @ g_i.T]
    where g_i = d(loss)/d(weight) for sample i.

    For a linear layer y = x @ W.T, the gradient w.r.t. W is:
        d(loss)/d(W) = d(loss)/d(y) @ x

    The diagonal FIM captures per-dimension importance:
        F_diag[j] = E[(d(loss)/d(W[:,j]))^2]

    For activations (pre-nonlinearity), this simplifies to:
        F_diag[j] ≈ E[x_j^2]

    High F_diag[j] means dimension j strongly influences the output,
    so it must be preserved during merging (Fisher-weighted projections null it otherwise).

    Args:
        activations: Activation matrix [n_samples, n_features].
        backend: Optional backend for computation.

    Returns:
        FisherInformationResult with diagonal FIM and diagnostics.
    """
    b = backend or get_default_backend()

    A = b.astype(activations, precision_dtype(b, reference=activations))
    b.eval(A)

    n_samples = int(A.shape[0])
    n_features = int(A.shape[1])
    reg = regularization_epsilon(b, A)

    logger.debug(
        "FISHER: Computing diagonal FIM for [%d x %d]",
        n_samples,
        n_features,
    )

    # Compute per-dimension second moment: E[x_j^2]
    # This is proportional to diagonal FIM for linear layers
    second_moment = b.mean(A * A, axis=0)  # E[x_j^2]
    b.eval(second_moment)

    diagonal_fim = second_moment
    b.eval(diagonal_fim)

    # Compute diagnostics
    trace_fim_arr = b.sum(diagonal_fim)
    mean_fim_arr = b.mean(diagonal_fim)
    max_fim = b.max(diagonal_fim)
    min_fim_pos = b.min(b.where(diagonal_fim > reg, diagonal_fim, b.full(b.shape(diagonal_fim), float("inf"))))
    b.eval(trace_fim_arr, mean_fim_arr, max_fim, min_fim_pos)

    trace_fim = float(b.to_scalar(trace_fim_arr))
    mean_fim = float(b.to_scalar(mean_fim_arr))
    max_fim_val = float(b.to_scalar(max_fim))
    min_fim_val = float(b.to_scalar(min_fim_pos))

    # Condition number (max/min ratio of diagonal)
    if min_fim_val > reg and min_fim_val < float("inf"):
        condition_number = max_fim_val / min_fim_val
    else:
        condition_number = float("inf")

    # Effective rank via participation ratio: (sum F_ii)^2 / sum(F_ii^2)
    fim_sq = diagonal_fim * diagonal_fim
    sum_fim_sq = b.sum(fim_sq)
    b.eval(sum_fim_sq)
    sum_fim_sq_val = float(b.to_scalar(sum_fim_sq))

    if sum_fim_sq_val > reg:
        effective_rank = (trace_fim * trace_fim) / sum_fim_sq_val
    else:
        effective_rank = 0.0

    # Count significant dimensions (data-derived threshold)
    threshold = _significance_threshold(diagonal_fim, b)
    significant_mask = diagonal_fim > threshold
    n_significant_arr = b.sum(b.astype(significant_mask, "int32"))
    b.eval(n_significant_arr)
    n_significant = int(b.to_scalar(n_significant_arr))

    logger.info(
        "FISHER: trace=%.4f, mean=%.6f, eff_rank=%.1f, n_sig=%d/%d, cond=%.2e",
        trace_fim,
        mean_fim,
        effective_rank,
        n_significant,
        n_features,
        condition_number,
    )

    return FisherInformationResult(
        diagonal_fim=diagonal_fim,
        effective_rank=effective_rank,
        trace_fim=trace_fim,
        condition_number=condition_number,
        mean_fim=mean_fim,
        n_significant=n_significant,
        significance_threshold=threshold,
    )


def fisher_compatibility_score(
    source_activations: "Array",
    target_activations: "Array",
    backend: "Backend | None" = None,
) -> FisherCompatibilityResult:
    """Compute Fisher-based merge compatibility score.

    Two models are compatible for merging if they have similar loss landscape
    curvature. This is measured by comparing their diagonal Fisher Information.

    Returns raw curvature metrics. The compatibility_score is the cosine
    similarity of the diagonal FIM vectors.

    Args:
        source_activations: Source model activations [n_samples, d_source].
        target_activations: Target model activations [n_samples, d_target].
        backend: Optional backend for computation.

    Returns:
        FisherCompatibilityResult with compatibility score and diagnostics.

    """
    b = backend or get_default_backend()

    source = b.astype(source_activations, precision_dtype(b, reference=source_activations))
    target = b.astype(target_activations, precision_dtype(b, reference=target_activations))
    b.eval(source, target)

    d_source = int(source.shape[1])
    d_target = int(target.shape[1])
    eps = division_epsilon(b, source)

    # Compute Fisher Information for both
    source_fisher = compute_empirical_fisher_diagonal(source, b)
    target_fisher = compute_empirical_fisher_diagonal(target, b)

    # Handle dimension mismatch
    # If dimensions differ, compare the common dimensions or project
    min_dim = min(d_source, d_target)

    source_fim = source_fisher.diagonal_fim[:min_dim]
    target_fim = target_fisher.diagonal_fim[:min_dim]
    b.eval(source_fim, target_fim)

    # Compute cosine similarity: F_s · F_t / (||F_s|| ||F_t||)
    dot_product = b.sum(source_fim * target_fim)
    source_norm = b.sqrt(b.sum(source_fim * source_fim))
    target_norm = b.sqrt(b.sum(target_fim * target_fim))
    b.eval(dot_product, source_norm, target_norm)

    source_norm_val = float(b.to_scalar(source_norm))
    target_norm_val = float(b.to_scalar(target_norm))
    dot_val = float(b.to_scalar(dot_product))

    if source_norm_val > eps and target_norm_val > eps:
        cosine_similarity = dot_val / (source_norm_val * target_norm_val)
    else:
        cosine_similarity = 0.0
    cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

    # Compute Pearson correlation of FIM values
    source_mean = b.mean(source_fim)
    target_mean = b.mean(target_fim)
    source_centered = source_fim - source_mean
    target_centered = target_fim - target_mean
    b.eval(source_centered, target_centered)

    cov = b.sum(source_centered * target_centered)
    source_std = b.sqrt(b.sum(source_centered * source_centered))
    target_std = b.sqrt(b.sum(target_centered * target_centered))
    b.eval(cov, source_std, target_std)

    source_std_val = float(b.to_scalar(source_std))
    target_std_val = float(b.to_scalar(target_std))
    cov_val = float(b.to_scalar(cov))

    if source_std_val > eps and target_std_val > eps:
        correlation = cov_val / (source_std_val * target_std_val)
    else:
        correlation = 0.0
    correlation = max(-1.0, min(1.0, correlation))

    # Compute overlap ratio: fraction of dimensions significant in both
    source_threshold = _significance_threshold(source_fim, b)
    target_threshold = _significance_threshold(target_fim, b)

    source_sig = source_fim > source_threshold
    target_sig = target_fim > target_threshold
    both_sig = b.sum(b.astype(source_sig * target_sig, "int32"))
    either_sig = b.sum(b.astype(source_sig + target_sig > 0, "int32"))
    b.eval(both_sig, either_sig)

    both_sig_val = int(b.to_scalar(both_sig))
    either_sig_val = int(b.to_scalar(either_sig))

    if either_sig_val > 0:
        overlap_ratio = float(both_sig_val) / float(either_sig_val)
    else:
        overlap_ratio = 1.0  # Both have no significant dimensions

    # Scalar summary: cosine similarity of diagonal FIM vectors
    compatibility_score = cosine_similarity

    # NOTE: Fisher measures loss curvature, NOT semantic compatibility.
    # Low Fisher scores between different architectures are EXPECTED.
    # Do not interpret this as "incompatible models" - interpret as
    # "different loss landscapes" which is normal for cross-architecture merges.

    logger.info(
        "FISHER: score=%.3f, cos=%.3f, corr=%.3f, overlap=%.3f (measures loss curvature, not semantic structure)",
        compatibility_score,
        cosine_similarity,
        correlation,
        overlap_ratio,
    )

    return FisherCompatibilityResult(
        compatibility_score=compatibility_score,
        cosine_similarity=cosine_similarity,
        correlation=correlation,
        overlap_ratio=overlap_ratio,
        source_fisher=source_fisher,
        target_fisher=target_fisher,
        recommendation="",  # No recommendation - raw metrics only
    )


def fisher_weighted_merge(
    source_weights: "Array",
    target_weights: "Array",
    source_fisher: FisherInformationResult,
    target_fisher: FisherInformationResult,
    backend: "Backend | None" = None,
) -> "Array":
    """Merge weights using Fisher Information weighting.

    Instead of simple averaging, weight the contribution of each model
    by its Fisher Information (parameter importance). This is the TIES-Merging
    approach from Matena & Raffel (2022).

    Formula:
        W_merged = (F_s * W_s + F_t * W_t) / (F_s + F_t)

    Where F_s and F_t are the diagonal Fisher values for source and target.
    Dimensions with higher Fisher in source get more source contribution.

    Args:
        source_weights: Source model weights [out, in].
        target_weights: Target model weights [out, in].
        source_fisher: Source Fisher Information result.
        target_fisher: Target Fisher Information result.
        backend: Optional backend for computation.

    Returns:
        Fisher-weighted merged weights.
    """
    b = backend or get_default_backend()

    W_s = b.astype(source_weights, precision_dtype(b, reference=source_weights))
    W_t = b.astype(target_weights, precision_dtype(b, reference=target_weights))
    b.eval(W_s, W_t)

    # Get Fisher values (per input dimension)
    F_s = source_fisher.diagonal_fim
    F_t = target_fisher.diagonal_fim
    eps = regularization_epsilon(b, W_s)

    # Ensure dimensions match
    in_dim = int(W_s.shape[1])
    if int(F_s.shape[0]) != in_dim or int(F_t.shape[0]) != in_dim:
        raise ValueError("FISHER MERGE: Dimension mismatch for Fisher weighting")

    # Normalize Fisher values to sum to 1
    F_s_sum = b.sum(F_s)
    F_t_sum = b.sum(F_t)
    b.eval(F_s_sum, F_t_sum)

    F_s_norm = F_s / b.maximum(F_s_sum, b.array([eps]))
    F_t_norm = F_t / b.maximum(F_t_sum, b.array([eps]))
    b.eval(F_s_norm, F_t_norm)

    # Compute per-dimension weights
    # source_weight[j] = F_s[j] / (F_s[j] + F_t[j])
    total_F = F_s_norm + F_t_norm
    total_F = b.maximum(total_F, b.full(b.shape(total_F), eps))
    source_weight = F_s_norm / total_F
    target_weight = F_t_norm / total_F
    b.eval(source_weight, target_weight)

    # Expand to match weight dimensions [1, in]
    source_weight = b.reshape(source_weight, (1, in_dim))
    target_weight = b.reshape(target_weight, (1, in_dim))

    # Apply Fisher-weighted merge
    W_merged = source_weight * W_s + target_weight * W_t
    b.eval(W_merged)

    logger.info(
        "FISHER MERGE: mean_source_weight=%.3f, mean_target_weight=%.3f",
        float(b.to_scalar(b.mean(source_weight))),
        float(b.to_scalar(b.mean(target_weight))),
    )

    return W_merged


__all__ = [
    "FisherInformationResult",
    "FisherCompatibilityResult",
    "compute_empirical_fisher_diagonal",
    "fisher_compatibility_score",
    "fisher_weighted_merge",
]

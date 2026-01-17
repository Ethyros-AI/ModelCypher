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

"""Gram matrix spectrum analysis for null-space diagnostics.

Analyzes eigenvalues of the Gram matrix (A @ A.T) to derive rank and
stability metrics used in transplant workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    precision_dtype,
    svd_rank_threshold,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GramSpectrumResult:
    """Result of Gram matrix spectrum analysis.

    All values are raw measurements - no interpretation applied.

    Attributes:
        n_samples: Number of activation samples (rows of A).
        d_features: Number of features (columns of A).
        eigenvalues: Top eigenvalues of Gram matrix (sorted descending).
        total_variance: Sum of all eigenvalues (total energy).
        max_eigenvalue: Largest eigenvalue.
        min_eigenvalue: Smallest eigenvalue.
        condition_number: max_eig / min_nonzero_eig (stability indicator).
        numeric_rank: Number of eigenvalues above rank threshold.
        null_rank: d_features - numeric_rank (dimensions available for transplant).
        intrinsic_dimension: TwoNN estimate of manifold dimensionality.
        energy_50_dims: Dimensions needed for 50% variance.
        energy_90_dims: Dimensions needed for 90% variance.
        energy_99_dims: Dimensions needed for 99% variance.
        spectral_gap: Ratio of (eigenvalue at ID) / (max eigenvalue).
        rank_threshold: Computed threshold for numeric rank.
    """

    n_samples: int
    d_features: int
    eigenvalues: list[float]
    total_variance: float
    max_eigenvalue: float
    min_eigenvalue: float
    condition_number: float
    numeric_rank: int
    null_rank: int
    intrinsic_dimension: float
    energy_50_dims: int
    energy_90_dims: int
    energy_99_dims: int
    spectral_gap: float
    rank_threshold: float


def compute_gram_spectrum(
    activations: "Array",
    backend: "Backend | None" = None,
    max_eigenvalues_returned: int = 50,
) -> GramSpectrumResult:
    """Analyze the eigenvalue spectrum of the Gram matrix.

    Computes G = A @ A.T and analyzes its eigenvalues to understand
    the geometry available for null-space projection.

    Args:
        activations: Activation matrix [n_samples, d_features].
        backend: Optional backend (uses default if None).
        max_eigenvalues_returned: Max eigenvalues to include in result.

    Returns:
        GramSpectrumResult with all spectrum measurements.

    Raises:
        ValueError: If activations has zero samples or features.
    """
    b = backend or get_default_backend()

    activations = b.array(activations)
    shape = b.shape(activations)

    if len(shape) != 2:
        raise ValueError(f"Expected 2D activations, got shape {shape}")

    n_samples, d_features = shape[0], shape[1]

    if n_samples == 0 or d_features == 0:
        raise ValueError(f"Empty activations: shape {shape}")

    # Promote to computation dtype (bfloat16 not supported by eigvalsh)
    compute_dtype = precision_dtype(b, reference=activations)
    activations = b.astype(activations, compute_dtype)
    b.eval(activations)

    # Build Gram matrix: G = A @ A.T ∈ R^{n×n}
    AAt = b.matmul(activations, b.transpose(activations))
    b.eval(AAt)

    # Get eigenvalues (symmetric positive semi-definite)
    eigvals = b.eigvalsh(AAt)
    b.eval(eigvals)

    # Sort descending
    idx = b.argsort(-eigvals, axis=0)
    eigvals = b.take(eigvals, idx, axis=0)
    b.eval(eigvals)

    # Machine epsilon for stability
    eps = float(machine_epsilon(b, AAt))

    # Ensure non-negative (numerical precision)
    eigvals_pos = b.maximum(eigvals, b.array(0.0))
    b.eval(eigvals_pos)

    # Total variance
    total_var = b.sum(eigvals_pos)
    b.eval(total_var)
    total_var_val = float(b.to_scalar(total_var))

    # Max/min eigenvalues
    max_eig = float(b.to_scalar(eigvals_pos[0]))
    min_eig = float(b.to_scalar(eigvals_pos[-1]))

    # Condition number (max / min nonzero)
    # Find smallest eigenvalue above machine epsilon
    eigvals_np = []
    for i in range(int(eigvals_pos.shape[0])):
        val = float(b.to_scalar(eigvals_pos[i]))
        eigvals_np.append(val)

    nonzero_eigvals = [v for v in eigvals_np if v > eps]
    if len(nonzero_eigvals) > 0:
        condition_number = nonzero_eigvals[0] / nonzero_eigvals[-1]
    else:
        condition_number = float("inf")

    # Numeric rank using data-derived threshold
    rank_scale = svd_rank_threshold(b, eigvals_pos, d_features)
    rank_threshold = max(max_eig, eps) * rank_scale

    numeric_rank = sum(1 for v in eigvals_np if v > rank_threshold)
    null_rank = max(0, d_features - numeric_rank)

    # Intrinsic dimension (TwoNN)
    id_estimator = IntrinsicDimension(b)
    id_result = id_estimator.compute(activations)
    intrinsic_dim = id_result.intrinsic_dimension

    # Energy distribution (cumulative sum)
    cumsum = 0.0
    energy_50_dims = n_samples
    energy_90_dims = n_samples
    energy_99_dims = n_samples

    for i, val in enumerate(eigvals_np):
        cumsum += val
        if energy_50_dims == n_samples and cumsum >= 0.5 * total_var_val:
            energy_50_dims = i + 1
        if energy_90_dims == n_samples and cumsum >= 0.9 * total_var_val:
            energy_90_dims = i + 1
        if energy_99_dims == n_samples and cumsum >= 0.99 * total_var_val:
            energy_99_dims = i + 1

    # Spectral gap: eigenvalue at intrinsic_dim / max eigenvalue
    id_idx = min(max(0, int(round(intrinsic_dim)) - 1), len(eigvals_np) - 1)
    spectral_gap = eigvals_np[id_idx] / max(max_eig, eps) if max_eig > eps else 0.0

    return GramSpectrumResult(
        n_samples=n_samples,
        d_features=d_features,
        eigenvalues=eigvals_np[:max_eigenvalues_returned],
        total_variance=total_var_val,
        max_eigenvalue=max_eig,
        min_eigenvalue=min_eig,
        condition_number=condition_number,
        numeric_rank=numeric_rank,
        null_rank=null_rank,
        intrinsic_dimension=intrinsic_dim,
        energy_50_dims=energy_50_dims,
        energy_90_dims=energy_90_dims,
        energy_99_dims=energy_99_dims,
        spectral_gap=spectral_gap,
        rank_threshold=rank_threshold,
    )


@dataclass(frozen=True)
class ProjectionAnalysisResult:
    """Result of analyzing delta projection into null-space.

    Attributes:
        delta_norm: Frobenius norm of original delta.
        projected_norm: Frobenius norm after null-space projection.
        correction_norm: Frobenius norm of what was removed.
        preserved_fraction: projected_norm / delta_norm (1.0 = nothing removed).
        cosine_similarity: Angle between original and projected delta.
    """

    delta_norm: float
    projected_norm: float
    correction_norm: float
    preserved_fraction: float
    cosine_similarity: float


def analyze_projection(
    delta_W: "Array",
    input_activations: "Array",
    backend: "Backend | None" = None,
) -> ProjectionAnalysisResult:
    """Analyze what happens when delta is projected into null-space.

    Computes:
        delta_proj = delta_W - (delta_W @ A.T) @ (A @ A.T)^+ @ A

    And measures how much of delta survives the projection.

    Args:
        delta_W: Weight delta matrix [out_dim, in_dim].
        input_activations: Input activations [n_samples, in_dim].
        backend: Optional backend (uses default if None).

    Returns:
        ProjectionAnalysisResult with norm and angle measurements.
    """
    from modelcypher.core.domain.geometry.numerical_stability import geodesic_pinv

    b = backend or get_default_backend()

    delta_W = b.array(delta_W)
    input_activations = b.array(input_activations)

    # Compute precision
    compute_dtype = precision_dtype(b, reference=delta_W)
    delta_W = b.astype(delta_W, compute_dtype)
    input_activations = b.astype(input_activations, compute_dtype)
    b.eval(delta_W, input_activations)

    # Build Gram matrix and pseudoinverse
    AAt = b.matmul(input_activations, b.transpose(input_activations))
    b.eval(AAt)

    AAt_inv = geodesic_pinv(b, AAt)
    b.eval(AAt_inv)

    # Compute projection: delta_proj = delta - (delta @ A.T) @ (A @ A.T)^+ @ A
    delta_row = b.matmul(delta_W, b.transpose(input_activations))
    correction = b.matmul(delta_row, AAt_inv)
    correction = b.matmul(correction, input_activations)
    delta_proj = delta_W - correction
    b.eval(delta_proj, correction)

    # Compute norms
    delta_norm = float(b.to_scalar(b.sqrt(b.sum(delta_W * delta_W))))
    proj_norm = float(b.to_scalar(b.sqrt(b.sum(delta_proj * delta_proj))))
    correction_norm = float(b.to_scalar(b.sqrt(b.sum(correction * correction))))

    # Preserved fraction
    eps = float(machine_epsilon(b, delta_W))
    preserved_fraction = proj_norm / max(delta_norm, eps)

    # Cosine similarity
    dot_product = float(b.to_scalar(b.sum(delta_W * delta_proj)))
    if delta_norm > eps and proj_norm > eps:
        cosine_sim = dot_product / (delta_norm * proj_norm)
    else:
        cosine_sim = 0.0

    return ProjectionAnalysisResult(
        delta_norm=delta_norm,
        projected_norm=proj_norm,
        correction_norm=correction_norm,
        preserved_fraction=preserved_fraction,
        cosine_similarity=cosine_sim,
    )


def compute_geometry_derived_scale(
    gram_spectrum: GramSpectrumResult,
) -> float:
    """Derive delta_scale from activation geometry.

    This is an EXPERIMENTAL function that attempts to derive the appropriate
    delta_scale from the geometry rather than using heuristics.

    The intuition:
    - If condition_number is high, pseudoinverse is unstable → scale down
    - If null_rank is small, little capacity available → scale down
    - If spectral_gap is large, clear separation → scale up

    Current formula (subject to revision based on experiments):
        scale = (null_rank / d_features) * (1 / log10(condition_number + 10))

    Args:
        gram_spectrum: Result from compute_gram_spectrum.

    Returns:
        Recommended delta_scale in [0, 1].
    """
    import math

    # Fraction of dimensions that are "null" (available)
    null_fraction = gram_spectrum.null_rank / max(gram_spectrum.d_features, 1)

    # Condition number penalty (log scale to handle large values)
    # log10(10) = 1, log10(100) = 2, log10(1000) = 3, etc.
    condition_factor = 1.0 / math.log10(gram_spectrum.condition_number + 10)

    # Spectral gap bonus (if gap is large, directions are well-separated)
    gap_factor = min(1.0, gram_spectrum.spectral_gap * 10)

    # Combine factors
    scale = null_fraction * condition_factor * (0.5 + 0.5 * gap_factor)

    # Clamp to [0, 1]
    return max(0.0, min(1.0, scale))

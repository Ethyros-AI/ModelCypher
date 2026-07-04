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

"""Random Matrix Theory signal/noise separation for null-space projection.

Uses Marchenko-Pastur bulk edges to separate higher-variance components from
the bulk in activation covariance spectra. Provides data-derived thresholds
for null-space projection.

References:
    - Marchenko & Pastur (1967) "Distribution of eigenvalues for some sets
      of random matrices"
    - Couillet & Benaych-Georges (2016) "Random Matrix Methods for Machine
      Learning"
    - ICLR 2026 "From Memorization to Reasoning in the Spectrum of Loss
      Curvature" (spectral structure of neural networks)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.mp_noise_estimator import estimate_mp_noise
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median,
    division_epsilon,
    machine_epsilon,
    precision_dtype,
    regularization_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


@dataclass
class MPSignalNoiseResult:
    """Result of Marchenko-Pastur signal/noise separation."""

    # Eigenvalues above MP bulk edge (true signal)
    signal_eigenvalues: "Array"

    # Eigenvalues within MP bulk (noise - available for transfer)
    noise_eigenvalues: "Array"

    # Indices of signal eigenvalues in original sorted order
    signal_indices: "Array"

    # Indices of noise eigenvalues in original sorted order
    noise_indices: "Array"

    # Marchenko-Pastur bulk upper edge
    mp_upper_edge: float

    # Marchenko-Pastur bulk lower edge
    mp_lower_edge: float

    # Number of signal directions (eigenvalues above bulk)
    signal_rank: int

    # Number of noise directions (eigenvalues in bulk)
    noise_rank: int

    # Aspect ratio gamma = d / n
    aspect_ratio: float

    # Estimated noise variance sigma^2 from bulk
    noise_variance: float

    # Fraction of total variance captured by signal
    signal_variance_fraction: float


@dataclass
class RMTNullSpaceProjection:
    """Feature-space RMT projections for null-space filtering."""

    # Per-coordinate diagnostic fraction of each coordinate in the noise subspace.
    keep_weights: "Array"

    # Orthogonal projector onto MP-bulk eigenvectors: V_noise @ V_noise.T.
    noise_projection: "Array"

    # Orthogonal projector onto MP-signal eigenvectors: V_signal @ V_signal.T.
    signal_projection: "Array"

    # MP bulk separation diagnostics.
    mp_result: MPSignalNoiseResult


def marchenko_pastur_edges(
    n_samples: int,
    n_features: int,
    noise_variance: float,
    backend: "Backend | None" = None,
) -> tuple[float, float]:
    """Compute the Marchenko-Pastur bulk edges.

    For a random matrix X with n_samples rows and n_features columns,
    and entries with variance sigma^2, the eigenvalue density of (X^T X) / n
    has support on [lambda_-, lambda_+] where:

        lambda_- = sigma^2 * (1 - sqrt(gamma))^2
        lambda_+ = sigma^2 * (1 + sqrt(gamma))^2
        gamma = n_features / n_samples

    Eigenvalues outside this range are TRUE SIGNAL (not random noise).
    Eigenvalues inside this range are consistent with pure noise.

    Args:
        n_samples: Number of samples (rows) in the data matrix.
        n_features: Number of features (columns) in the data matrix.
        noise_variance: Estimated variance of noise (sigma^2).
        backend: Optional backend for computation.

    Returns:
        (lower_edge, upper_edge) of the Marchenko-Pastur bulk.
    """
    b = backend or get_default_backend()

    # Aspect ratio gamma = d / n
    # For underdetermined case (n < d), some eigenvalues are exactly 0
    gamma = float(n_features) / float(n_samples)

    # Compute edges
    sqrt_gamma = sqrt_scalar(gamma, b)
    lower_edge = noise_variance * (1.0 - sqrt_gamma) ** 2
    upper_edge = noise_variance * (1.0 + sqrt_gamma) ** 2

    # Handle edge cases
    if gamma > 1.0:
        # Underdetermined: (d - n) eigenvalues are exactly 0
        # The MP distribution has a point mass at 0
        lower_edge = 0.0

    return lower_edge, upper_edge


def estimate_noise_variance_from_bulk(
    eigenvalues: "Array",
    n_samples: int,
    n_features: int,
    backend: "Backend",
) -> float:
    """Estimate noise variance using the shared spike-robust MP estimator.

    Args:
        eigenvalues: Eigenvalues sorted in descending order.
        n_samples: Number of samples.
        n_features: Number of features.
        backend: Backend for computation.

    Returns:
        Estimated noise variance sigma^2.
    """
    return estimate_mp_noise(
        eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
        backend=backend,
    ).sigma_sq


def estimate_noise_variance_closed_form(
    eigenvalues: "Array",
    n_samples: int,
    n_features: int,
    backend: "Backend",
) -> float:
    """Compatibility wrapper for the shared spike-robust MP estimator.

    Args:
        eigenvalues: Eigenvalues (any order, will be sorted).
        n_samples: Number of samples.
        n_features: Number of features.
        backend: Backend for computation.

    Returns:
        Estimated noise variance sigma^2.

    References:
        - Marchenko & Pastur (1967)
        - The invariant "signal > bulk" follows from spiked covariance models
    """
    return estimate_mp_noise(
        eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
        backend=backend,
    ).sigma_sq


def separate_signal_noise(
    activations: "Array",
    backend: "Backend | None" = None,
    noise_estimation: str = "closed_form",
) -> MPSignalNoiseResult:
    """Separate signal from noise using Marchenko-Pastur distribution.

    This is the principled replacement for variance-based heuristics in
    null-space detection. Instead of arbitrary thresholds, we use the
    fundamental properties of random matrices.

    Algorithm:
        1. Center the data and compute sample covariance C = A.T @ A / (n-1)
        2. Compute eigenvalues of C
        3. Estimate noise variance from lower bulk (guaranteed noise)
        4. Compute MP bulk edges
        5. Eigenvalues > upper_edge are TRUE SIGNAL
        6. Eigenvalues <= upper_edge are NOISE (available for transfer)

    Args:
        activations: Activation matrix [n_samples, n_features].
        backend: Optional backend for computation.
        noise_estimation: Method for noise variance estimation.
            - "closed_form": Single-pass using lower-bulk invariant (default)
            - "median": Single-pass median-based estimation

    Returns:
        MPSignalNoiseResult with separated eigenvalues and diagnostics.

    Example:
        >>> result = separate_signal_noise(activations, backend)
        >>> # Use noise_indices for null-space projection
        >>> # These are the directions available for knowledge transfer
        >>> print(f"Signal rank: {result.signal_rank}, Noise rank: {result.noise_rank}")
    """
    b = backend or get_default_backend()

    A = b.astype(activations, precision_dtype(b, reference=activations))
    b.eval(A)

    n_samples = int(A.shape[0])
    n_features = int(A.shape[1])
    gamma = float(n_features) / float(n_samples)
    eps = machine_epsilon(b, A)
    reg = regularization_epsilon(b, A)

    logger.debug(
        "RMT: Separating signal/noise for [%d x %d] (gamma=%.3f)",
        n_samples,
        n_features,
        gamma,
    )

    # Center the data
    mean_A = b.mean(A, axis=0, keepdims=True)
    A_centered = A - mean_A
    b.eval(A_centered)

    # Compute sample covariance in feature space: C = A.T @ A / (n-1)
    # This gives us d eigenvalues (one per feature dimension)
    n_denom = max(float(n_samples - 1), 1.0)
    C = b.matmul(b.transpose(A_centered), A_centered) / n_denom
    b.eval(C)

    # Add small regularization for numerical stability
    C = C + reg * b.eye(n_features)
    b.eval(C)

    # Compute eigenvalues (symmetric positive semidefinite)
    eigenvalues = b.eigvalsh(C)
    b.eval(eigenvalues)

    # Sort in descending order (eigvalsh returns ascending)
    n_eigs = int(eigenvalues.shape[0])  # This is now n_features
    desc_indices = b.arange(n_eigs - 1, -1, -1)
    eigenvalues_sorted = b.take(eigenvalues, desc_indices, axis=0)
    b.eval(eigenvalues_sorted)

    estimate = estimate_mp_noise(
        eigenvalues_sorted,
        n_samples=n_samples,
        n_features=n_features,
        backend=b,
    )
    sigma_sq = estimate.sigma_sq
    lower_edge = estimate.lower_edge
    upper_edge = estimate.upper_edge

    logger.debug(
        "RMT: sigma^2=%.6f, MP edges=[%.6f, %.6f]",
        sigma_sq,
        lower_edge,
        upper_edge,
    )

    # Separate signal from noise
    # Signal: eigenvalues significantly above upper edge
    # We use a small margin (1 + sqrt(eps)) to avoid edge effects
    margin = 1.0 + sqrt_scalar(eps, b)
    signal_threshold = upper_edge * margin

    signal_mask = eigenvalues_sorted > signal_threshold
    noise_mask = eigenvalues_sorted <= signal_threshold
    b.eval(signal_mask, noise_mask)

    # Count signal and noise dimensions
    signal_count = b.sum(b.astype(signal_mask, "int32"))
    noise_count = b.sum(b.astype(noise_mask, "int32"))
    b.eval(signal_count, noise_count)
    signal_rank = int(b.to_scalar(signal_count))
    noise_rank = int(b.to_scalar(noise_count))

    # Extract indices
    all_indices = b.arange(n_eigs)

    # Get signal indices (where signal_mask is True)
    signal_indices_raw = b.where(
        signal_mask,
        all_indices,
        b.full(all_indices.shape, -1),  # Use -1 as placeholder
    )
    signal_indices_raw = b.astype(signal_indices_raw, "int32")
    b.eval(signal_indices_raw)

    # Filter to only valid indices
    if signal_rank > 0:
        # Sort to get valid indices first
        sorted_signal = b.sort(signal_indices_raw)
        b.eval(sorted_signal)
        # Take from the end (where valid indices are after sorting ascending)
        signal_indices = sorted_signal[-signal_rank:]
        b.eval(signal_indices)
    else:
        signal_indices = b.zeros((0,), dtype="int32")
        b.eval(signal_indices)

    # Get noise indices similarly
    noise_indices_raw = b.where(
        noise_mask,
        all_indices,
        b.full(all_indices.shape, n_eigs),  # Use n_eigs as placeholder
    )
    noise_indices_raw = b.astype(noise_indices_raw, "int32")
    b.eval(noise_indices_raw)

    if noise_rank > 0:
        sorted_noise = b.sort(noise_indices_raw)
        b.eval(sorted_noise)
        # Take from the beginning (where valid indices are)
        noise_indices = sorted_noise[:noise_rank]
        b.eval(noise_indices)
    else:
        noise_indices = b.zeros((0,), dtype="int32")
        b.eval(noise_indices)

    # Extract eigenvalue arrays
    if signal_rank > 0:
        signal_eigenvalues = b.take(eigenvalues_sorted, signal_indices, axis=0)
    else:
        signal_eigenvalues = b.zeros((0,))
    if noise_rank > 0:
        noise_eigenvalues = b.take(eigenvalues_sorted, noise_indices, axis=0)
    else:
        noise_eigenvalues = b.zeros((0,))
    b.eval(signal_eigenvalues, noise_eigenvalues)

    # Compute signal variance fraction
    total_variance = b.sum(eigenvalues_sorted)
    signal_variance = b.sum(signal_eigenvalues) if signal_rank > 0 else b.array([0.0])
    b.eval(total_variance, signal_variance)

    total_var_val = float(b.to_scalar(total_variance))
    signal_var_val = float(b.to_scalar(signal_variance))

    if total_var_val > eps:
        signal_variance_fraction = signal_var_val / total_var_val
    else:
        signal_variance_fraction = 0.0

    logger.info(
        "RMT: signal_rank=%d, noise_rank=%d, signal_var=%.1f%%, MP_edge=%.6f",
        signal_rank,
        noise_rank,
        100.0 * signal_variance_fraction,
        upper_edge,
    )

    return MPSignalNoiseResult(
        signal_eigenvalues=signal_eigenvalues,
        noise_eigenvalues=noise_eigenvalues,
        signal_indices=signal_indices,
        noise_indices=noise_indices,
        mp_upper_edge=upper_edge,
        mp_lower_edge=lower_edge,
        signal_rank=signal_rank,
        noise_rank=noise_rank,
        aspect_ratio=gamma,
        noise_variance=sigma_sq,
        signal_variance_fraction=signal_variance_fraction,
    )


def compute_rmt_null_space_projection(
    activations: "Array",
    backend: "Backend | None" = None,
) -> RMTNullSpaceProjection:
    """Compute RMT feature-space projectors for null-space filtering.

    This is the direct replacement for coordinate-wise variance heuristics.
    Instead of scaling each feature independently, use the Marchenko-Pastur
    distribution to identify signal/noise eigenvector subspaces and build the
    orthogonal projector onto the noise subspace.

    Algorithm:
        1. Compute covariance eigenvectors
        2. Separate signal from noise using MP distribution
        3. Build P_noise = V_noise @ V_noise.T
        4. Report diag(P_noise)-equivalent keep weights for diagnostics

    For each dimension d:
        - If d's variance projects mainly onto noise eigenvectors: keep_weight ≈ 1
        - If d's variance projects mainly onto signal eigenvectors: keep_weight ≈ 0

    Args:
        activations: Activation matrix [n_samples, n_features].
        backend: Optional backend for computation.

    Returns:
        RMTNullSpaceProjection with projector matrices and MP diagnostics.
    """
    b = backend or get_default_backend()

    A = b.astype(activations, precision_dtype(b, reference=activations))
    b.eval(A)

    n_samples = int(A.shape[0])
    n_features = int(A.shape[1])
    eps = division_epsilon(b, A)
    reg = regularization_epsilon(b, A)

    # Separate signal from noise
    mp_result = separate_signal_noise(A, backend=b)

    # Compute centered activations
    mean_A = b.mean(A, axis=0, keepdims=True)
    A_centered = A - mean_A
    b.eval(A_centered)

    # Compute covariance matrix C = A^T @ A / (n - 1)
    # This gives us eigenvectors in feature space
    C = b.matmul(b.transpose(A_centered), A_centered) / max(float(n_samples - 1), 1.0)
    C = C + reg * b.eye(n_features)
    b.eval(C)

    # Compute eigendecomposition
    eigenvalues, eigenvectors = b.eigh(C)
    b.eval(eigenvalues, eigenvectors)

    # Sort in descending order
    desc_indices = b.argsort(-eigenvalues)
    eigenvalues = b.take(eigenvalues, desc_indices, axis=0)
    eigenvectors = b.take(eigenvectors, desc_indices, axis=1)
    b.eval(eigenvalues, eigenvectors)

    # Determine which eigenvectors correspond to signal vs noise
    # Use the MP result to set the threshold
    n_signal = mp_result.signal_rank
    n_noise = mp_result.noise_rank

    # Create binary mask for noise eigenvectors (1 = noise, 0 = signal)
    n_eigs = int(eigenvalues.shape[0])
    noise_mask = b.arange(n_eigs) >= n_signal  # Eigenvectors after signal are noise
    noise_mask = b.astype(noise_mask, eigenvectors.dtype)
    b.eval(noise_mask)
    signal_mask = 1.0 - noise_mask
    b.eval(signal_mask)

    noise_vectors = eigenvectors * b.reshape(noise_mask, (1, -1))
    signal_vectors = eigenvectors * b.reshape(signal_mask, (1, -1))
    noise_projection = b.matmul(noise_vectors, b.transpose(noise_vectors))
    signal_projection = b.matmul(signal_vectors, b.transpose(signal_vectors))
    b.eval(noise_projection, signal_projection)

    # Compute per-dimension projection onto noise vs signal subspace
    # Each dimension's "noise fraction" = sum of squared loadings on noise eigenvectors
    # Loading = eigenvector component
    eigvec_sq = eigenvectors * eigenvectors  # [n_features, n_eigs]

    # Weight by noise_mask to get noise contribution
    noise_contributions = b.matmul(eigvec_sq, b.reshape(noise_mask, (-1, 1)))
    noise_contributions = b.squeeze(noise_contributions)  # [n_features]
    b.eval(noise_contributions)

    # Total contributions (should sum to 1 for each dimension in exact arithmetic)
    total_contributions = b.sum(eigvec_sq, axis=1)
    b.eval(total_contributions)

    # Noise fraction = noise_contributions / total_contributions
    # This is our "keep weight" - how much of this dimension is available
    eps_arr = b.full(b.shape(total_contributions), eps)
    keep_weights = noise_contributions / b.maximum(total_contributions, eps_arr)
    keep_weights = b.clip(keep_weights, 0.0, 1.0)
    b.eval(keep_weights)

    # Log statistics
    mean_keep = float(b.to_scalar(b.mean(keep_weights)))
    min_keep = float(b.to_scalar(b.min(keep_weights)))
    max_keep = float(b.to_scalar(b.max(keep_weights)))

    logger.info(
        "RMT weights: mean=%.3f, range=[%.3f, %.3f], signal_dim=%d, noise_dim=%d",
        mean_keep,
        min_keep,
        max_keep,
        n_signal,
        n_noise,
    )

    return RMTNullSpaceProjection(
        keep_weights=keep_weights,
        noise_projection=noise_projection,
        signal_projection=signal_projection,
        mp_result=mp_result,
    )


def compute_rmt_null_space_weights(
    activations: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", MPSignalNoiseResult]:
    """Compute per-dimension diagnostic weights for null-space projection.

    The production filter should use ``compute_rmt_null_space_projection`` so
    rotated signal subspaces remain rotated. This wrapper is retained for
    callers and tests that need the historical coordinate diagnostics.
    """
    projection = compute_rmt_null_space_projection(activations, backend=backend)
    return projection.keep_weights, projection.mp_result


@dataclass
class SignalRankResult:
    """Result of signal rank computation from pre-computed singular values.

    This is a lightweight result for when only signal rank is needed,
    avoiding the overhead of full eigenvalue/index computation.
    """

    signal_rank: int
    noise_rank: int
    mp_upper_edge: float
    signal_variance_fraction: float
    # Shannon effective rank of the activation covariance spectrum.
    # This is the concentration-aware budget; signal_rank remains the
    # distinguishability ceiling from MP noise separation.
    effective_rank: float = 0.0


def compute_signal_rank_from_singular_values(
    singular_values: "Array",
    n_samples: int,
    n_features: int,
    backend: "Backend | None" = None,
    center_correction: bool = True,
) -> SignalRankResult:
    """Compute signal rank from pre-computed singular values using Marchenko-Pastur.

    This is an optimization for when SVD is already computed elsewhere.
    The singular values of A correspond to sqrt(eigenvalues of A.T @ A).

    Key relationship:
        SVD: A = U @ diag(S) @ V.T
        Covariance: C = A.T @ A / (n-1) = V @ diag(S^2 / (n-1)) @ V.T
        Therefore: eigenvalues(C) = S^2 / (n-1)

    Args:
        singular_values: Pre-computed singular values from SVD, sorted descending.
        n_samples: Number of samples (rows in original matrix).
        n_features: Number of features (columns in original matrix).
        backend: Optional backend for computation.
        center_correction: If True, apply (n-1) denominator for sample covariance.
            Set False if singular values are from already-centered data with
            different normalization.

    Returns:
        SignalRankResult with signal/noise rank and MP diagnostics.

    Example:
        >>> U, S, Vt = geodesic_svd(backend, activations)
        >>> result = compute_signal_rank_from_singular_values(
        ...     S, n_samples=activations.shape[0], n_features=activations.shape[1], backend=backend
        ... )
        >>> intrinsic_rank = result.signal_rank
    """
    b = backend or get_default_backend()

    S = b.array(singular_values)
    b.eval(S)

    eps = machine_epsilon(b, S)
    n_sv = int(S.shape[0])

    # Convert singular values to covariance eigenvalues
    # eigenvalues(A.T @ A / (n-1)) = S^2 / (n-1)
    if center_correction and n_samples > 1:
        denom = float(n_samples - 1)
    else:
        denom = 1.0

    eigenvalues = (S * S) / denom
    b.eval(eigenvalues)

    # Ensure descending order (SVD singular values should already be descending)
    # but eigenvalues derived from them maintain that order

    # Compute effective dimensions for MP distribution
    # gamma = smaller_dim / larger_dim (always <= 1)
    if n_samples >= n_features:
        n_eff = n_samples
        d_eff = n_features
    else:
        n_eff = n_features
        d_eff = n_samples

    gamma = float(d_eff) / float(n_eff)

    # Estimate noise variance from median of eigenvalues
    # median(eigenvalues) ≈ sigma^2 * (1 + gamma) for MP distribution
    median_eig = compute_median(eigenvalues, b)
    sigma_sq = median_eig / (1.0 + gamma) if gamma > 0 else median_eig

    # Compute MP upper edge
    sqrt_gamma = sqrt_scalar(gamma, b)
    upper_edge = sigma_sq * (1.0 + sqrt_gamma) ** 2

    # Signal threshold with small margin to avoid edge effects
    margin = 1.0 + sqrt_scalar(eps, b)
    signal_threshold = upper_edge * margin

    # Count signal eigenvalues (above threshold)
    signal_mask = eigenvalues > signal_threshold
    signal_count = b.sum(b.astype(signal_mask, "int32"))
    b.eval(signal_count)
    signal_rank = int(b.to_scalar(signal_count))
    noise_rank = n_sv - signal_rank

    # Compute signal variance fraction
    total_variance = b.sum(eigenvalues)
    b.eval(total_variance)
    total_var_val = float(b.to_scalar(total_variance))

    if signal_rank > 0 and total_var_val > eps:
        signal_eigenvalues = eigenvalues[:signal_rank]
        signal_variance = b.sum(signal_eigenvalues)
        b.eval(signal_variance)
        signal_variance_fraction = float(b.to_scalar(signal_variance)) / total_var_val
    else:
        signal_variance_fraction = 0.0

    if signal_rank > 0 and total_var_val > eps:
        signal_total_var = float(b.to_scalar(signal_variance))
        eig_list = b.tolist(signal_eigenvalues)
        if isinstance(eig_list, (int, float)):
            eig_values = [float(eig_list)]
        else:
            eig_values = [float(v) for v in eig_list]
        entropy = 0.0
        for eig in eig_values:
            if eig <= 0.0:
                continue
            prob = eig / max(signal_total_var, eps)
            entropy -= prob * math.log(prob)
        effective_rank = math.exp(entropy) if entropy > 0.0 else float(bool(eig_values))
    else:
        effective_rank = 0.0

    logger.debug(
        "RMT (from SVD): signal_rank=%d, noise_rank=%d, MP_edge=%.6f, "
        "signal_var=%.1f%%, erank=%.2f",
        signal_rank,
        noise_rank,
        upper_edge,
        100.0 * signal_variance_fraction,
        effective_rank,
    )

    return SignalRankResult(
        signal_rank=signal_rank,
        noise_rank=noise_rank,
        mp_upper_edge=upper_edge,
        signal_variance_fraction=signal_variance_fraction,
        effective_rank=effective_rank,
    )


__all__ = [
    "MPSignalNoiseResult",
    "SignalRankResult",
    "marchenko_pastur_edges",
    "estimate_noise_variance_from_bulk",
    "estimate_noise_variance_closed_form",
    "separate_signal_noise",
    "compute_rmt_null_space_weights",
    "compute_signal_rank_from_singular_values",
]

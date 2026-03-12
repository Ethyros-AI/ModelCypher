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
    """Estimate noise variance from the bulk of the eigenvalue spectrum.

    Uses the median of eigenvalues as a robust estimator of the bulk center,
    then derives sigma^2 from the Marchenko-Pastur distribution.

    For MP distribution with aspect ratio gamma = d/n:
        median ≈ sigma^2 * (1 - sqrt(gamma))^2 + sigma^2 * gamma / 2
                (approximate, exact formula involves incomplete beta)

    We use the approximation:
        sigma^2 ≈ median / (1 + gamma)

    This is robust to outlier signal eigenvalues at the top.

    Args:
        eigenvalues: Eigenvalues sorted in descending order.
        n_samples: Number of samples.
        n_features: Number of features.
        backend: Backend for computation.

    Returns:
        Estimated noise variance sigma^2.
    """
    b = backend
    eigs = b.astype(eigenvalues, precision_dtype(b, reference=eigenvalues))
    b.eval(eigs)

    n_eigs = int(eigs.shape[0])
    if n_eigs == 0:
        return 1.0

    gamma = float(n_features) / float(n_samples)
    eps = division_epsilon(b, eigs)

    # Use median as robust estimate of bulk center
    # Sort eigenvalues (they should already be sorted, but ensure it)
    sorted_eigs = b.sort(eigs)
    b.eval(sorted_eigs)

    # Compute median
    mid = n_eigs // 2
    if n_eigs % 2 == 1:
        median_val = float(b.to_scalar(sorted_eigs[mid]))
    else:
        lower = float(b.to_scalar(sorted_eigs[mid - 1]))
        upper = float(b.to_scalar(sorted_eigs[mid]))
        median_val = (lower + upper) / 2.0

    # Estimate sigma^2 from median
    # For MP distribution: E[lambda] = sigma^2 * (1 + gamma)
    # Median is slightly lower, so this slightly overestimates sigma^2
    # which overestimates sigma^2, raising the MP edge (fewer false signals)
    sigma_sq = median_val / max(1.0 + gamma, eps)

    return max(sigma_sq, eps)


def estimate_noise_variance_closed_form(
    eigenvalues: "Array",
    n_samples: int,
    n_features: int,
    backend: "Backend",
) -> float:
    """Noise variance estimator using the lower MP bulk.

    Uses the lower half of eigenvalues to estimate sigma^2 based on MP
    distribution relationships. Assumes signal components lie above the bulk
    in the spiked model.

    Mathematical derivation:
        - Lower edge: lambda_- = sigma^2 * (1 - sqrt(gamma))^2
        - Q1 approximation: Q1 ≈ sigma^2 * (1 + gamma - sqrt(gamma))
        - Mean of lower half ≈ (lower_edge + Q1) / 2
        - Solve for sigma^2 = lower_mean / expected_factor

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
    b = backend
    eigs = b.astype(eigenvalues, precision_dtype(b, reference=eigenvalues))
    b.eval(eigs)

    n_eigs = int(eigs.shape[0])
    if n_eigs == 0:
        return 1.0

    gamma = float(n_features) / float(n_samples)
    eps = division_epsilon(b, eigs)

    # Sort eigenvalues ascending (lower eigenvalues first)
    sorted_eigs = b.sort(eigs)
    b.eval(sorted_eigs)

    # Take lower half - these are GUARANTEED to be noise
    # Signal eigenvalues are always ABOVE the bulk, never below
    lower_half_count = max(n_eigs // 2, 1)
    lower_indices = b.arange(lower_half_count)
    lower_eigs = b.take(sorted_eigs, lower_indices, axis=0)
    b.eval(lower_eigs)

    # Compute mean of lower half
    lower_mean = b.mean(lower_eigs)
    b.eval(lower_mean)
    lower_mean_val = float(b.to_scalar(lower_mean))

    # Compute expected mean of lower half of MP distribution
    # E[lambda | lambda < median] / sigma^2 = g(gamma)
    #
    # For MP distribution on [a, b] where:
    #   a = sigma^2 * (1 - sqrt(gamma))^2  (lower edge)
    #   b = sigma^2 * (1 + sqrt(gamma))^2  (upper edge)
    #
    # The lower half has conditional expectation approximately:
    #   E[lambda | lambda < median] ≈ sigma^2 * (lower_edge_factor + q1_factor) / 2
    #
    # where:
    #   lower_edge_factor = (1 - sqrt(gamma))^2
    #   q1_factor ≈ 1 + gamma - sqrt(gamma)  (Q1 approximation from MP CDF)
    sqrt_gamma = sqrt_scalar(gamma, b)

    # Lower edge factor: (1 - sqrt(gamma))^2
    lower_edge_factor = (1.0 - sqrt_gamma) ** 2

    # Q1 approximation: derived from MP CDF behavior
    # The 25th percentile lies at approximately sigma^2 * (1 + gamma - sqrt(gamma))
    q1_factor = 1.0 + gamma - sqrt_gamma

    # Expected mean of lower half ≈ midpoint of [lower_edge, Q1]
    expected_lower_mean_factor = (lower_edge_factor + q1_factor) / 2.0

    # Edge case: when gamma is very small or very large
    if expected_lower_mean_factor < eps:
        # Fall back to simple mean relationship
        expected_lower_mean_factor = 0.5 * (1.0 + gamma)

    # Solve for sigma^2
    sigma_sq = lower_mean_val / max(expected_lower_mean_factor, eps)

    logger.debug(
        "RMT closed-form: gamma=%.4f, lower_mean=%.6f, factor=%.4f, sigma^2=%.6f",
        gamma,
        lower_mean_val,
        expected_lower_mean_factor,
        sigma_sq,
    )

    return max(sigma_sq, eps)


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

    # For MP distribution, use effective sample size
    # gamma_eff = d / n for overdetermined, n / d for underdetermined
    # The MP distribution always uses the smaller dimension in numerator
    if n_samples >= n_features:
        # Overdetermined: gamma = d/n < 1
        n_eff = n_samples
        d_eff = n_features
    else:
        # Underdetermined: gamma = n/d < 1
        n_eff = n_features
        d_eff = n_samples

    # Estimate noise variance from bulk
    # Default: closed-form using lower-bulk invariant (signal > bulk, so lower = noise)
    if noise_estimation == "closed_form":
        sigma_sq = estimate_noise_variance_closed_form(
            eigenvalues_sorted, n_eff, d_eff, b
        )
    else:  # "median" fallback
        sigma_sq = estimate_noise_variance_from_bulk(
            eigenvalues_sorted, n_eff, d_eff, b
        )

    # Compute MP edges with effective dimensions
    lower_edge, upper_edge = marchenko_pastur_edges(
        n_eff, d_eff, sigma_sq, b
    )

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


def compute_rmt_null_space_weights(
    activations: "Array",
    backend: "Backend | None" = None,
) -> tuple["Array", MPSignalNoiseResult]:
    """Compute per-dimension weights for null-space projection using RMT.

    This is the direct replacement for variance-based heuristics. Instead of
    normalizing variance to [0, 1] and using (1 - variance) as keep weights,
    we use the Marchenko-Pastur distribution to determine which directions
    are truly signal vs noise.

    Algorithm:
        1. Compute covariance eigenvectors
        2. Separate signal from noise using MP distribution
        3. Project variance onto eigenvector basis
        4. Weight dimensions by their contribution to noise vs signal

    For each dimension d:
        - If d's variance projects mainly onto noise eigenvectors: keep_weight ≈ 1
        - If d's variance projects mainly onto signal eigenvectors: keep_weight ≈ 0

    Args:
        activations: Activation matrix [n_samples, n_features].
        backend: Optional backend for computation.

    Returns:
        (keep_weights, mp_result) where keep_weights[i] ∈ [0, 1] indicates
        how much of dimension i is available for transfer.
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

    return keep_weights, mp_result


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

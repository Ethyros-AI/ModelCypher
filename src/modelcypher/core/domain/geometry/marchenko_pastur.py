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

"""Marchenko-Pastur noise edge and Tikhonov shrinkage weights.

Provides principled signal/noise separation for sample covariance eigenspectra.
For a data matrix X of shape [N, D], the sample covariance C = X^T X / N has
eigenvalues that split into signal (population structure) and noise (finite-
sample artifact). The Marchenko-Pastur law (1967) gives the upper edge of the
noise bulk:

    sigma_sq = robust MP bulk mean with signal spikes excluded
    gamma = D / N  (aspect ratio, columns / rows)
    alpha = sigma_sq * (1 + sqrt(gamma))^2

Eigenvalues below alpha are indistinguishable from sampling noise.
Eigenvalues above alpha contain population signal.

Tikhonov shrinkage weights w_i = lambda_i / (lambda_i + alpha) provide a
continuous, derived weighting: directions with strong signal get w_i -> 1,
directions at the noise floor get w_i -> 0.

Citation: Marchenko, V. A. & Pastur, L. A. (1967). Distribution of eigenvalues
for some sets of random matrices. *Matematicheskii Sbornik*, 114(4), 507-536.

Experimental status: Module is production-ready (pure math, no heuristics).
Applications using this module may be experimental (see caller docstrings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.mp_noise_estimator import estimate_mp_noise

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class MarchenkoPasturResult:
    """Result of Marchenko-Pastur noise edge computation."""

    sigma_sq: float
    """Estimated MP bulk mean with exact zeros and signal spikes excluded."""

    aspect_ratio: float
    """D / N (features / samples)."""

    noise_edge: float
    """Upper edge of the MP noise bulk: sigma_sq * (1 + sqrt(gamma))^2."""

    effective_rank: float
    """Sum of Tikhonov weights (continuous effective rank)."""


def marchenko_pastur_noise_edge(
    eigenvalues: list[float] | tuple[float, ...],
    n_features: int,
    n_samples: int,
    backend: "Backend | None" = None,
) -> float:
    """Compute the spike-robust Marchenko-Pastur upper noise edge.

    Args:
        eigenvalues: Eigenvalues of the sample covariance.
        n_features: D, dimensionality of each sample.
        n_samples: N, number of samples (rows of the data matrix X).
        backend: Optional source backend used to derive the eigenvalue dtype
            precision. Omit only for Python-float spectra.

    Returns:
        alpha: The noise edge. Eigenvalues below this are sampling noise.

    The estimator excludes exact rank-deficiency zeros and estimates sigma_sq
    from the remaining MP bulk after removing signal spikes.
    """
    estimate = estimate_mp_noise(
        eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
        backend=backend,
    )
    return estimate.upper_edge


def tikhonov_weights_from_eigenvalues(
    eigenvalues: list[float] | tuple[float, ...],
    alpha: float,
) -> list[float]:
    """Compute Tikhonov shrinkage weights: w_i = lambda_i / (lambda_i + alpha).

    Args:
        eigenvalues: Eigenvalues of the sample covariance (non-negative).
        alpha: Regularization parameter (typically the MP noise edge).

    Returns:
        List of weights in [0, 1]. Eigenvalues >> alpha get w_i -> 1.
        Eigenvalues << alpha get w_i -> 0. Continuous, no integer rank.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    return [max(0.0, lam / (lam + alpha)) for lam in eigenvalues]


def tikhonov_effective_rank(
    eigenvalues: list[float] | tuple[float, ...],
    alpha: float,
) -> float:
    """Sum of Tikhonov weights — a continuous effective rank.

    Unlike integer rank (hard cutoff) or participation ratio (purely
    spectral), this incorporates the finite-sample noise floor via alpha.

    Returns a float in [0, D]. Higher means more directions are above
    the noise floor.
    """
    weights = tikhonov_weights_from_eigenvalues(eigenvalues, alpha)
    return sum(weights)


def compute_marchenko_pastur_profile(
    eigenvalues: list[float] | tuple[float, ...],
    n_features: int,
    n_samples: int,
) -> MarchenkoPasturResult:
    """Full MP analysis: noise edge + Tikhonov effective rank.

    Convenience function that computes everything in one call.

    Args:
        eigenvalues: Eigenvalues of sample covariance, in any order.
        n_features: D, dimensionality.
        n_samples: N, number of samples.

    Returns:
        MarchenkoPasturResult with sigma_sq, aspect_ratio, noise_edge,
        and effective_rank (sum of Tikhonov weights).
    """
    estimate = estimate_mp_noise(
        eigenvalues,
        n_samples=n_samples,
        n_features=n_features,
    )
    noise_edge = estimate.upper_edge
    eff_rank = tikhonov_effective_rank(eigenvalues, noise_edge)
    return MarchenkoPasturResult(
        sigma_sq=estimate.sigma_sq,
        aspect_ratio=estimate.aspect_ratio,
        noise_edge=noise_edge,
        effective_rank=eff_rank,
    )

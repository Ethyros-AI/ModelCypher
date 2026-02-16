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

"""Spectral analysis utilities for prime geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    _promote_precision,
    division_epsilon,
    machine_epsilon,
    power_iteration_eigh,
)

from .prime_geometry_types import EigenvalueDistribution, SpectralComparison
from .prime_geometry_utils import _array_to_list

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def analyze_eigenvalues(
    gram: "Array",
    backend: "Backend | None" = None,
) -> EigenvalueDistribution:
    """Analyze the eigenvalue distribution of a Gram matrix.

    Args:
        gram: Symmetric Gram matrix [n, n].
        backend: Compute backend.

    Returns:
        EigenvalueDistribution with spectral metrics.
    """
    backend = backend or get_default_backend()
    gram = _promote_precision(gram, backend)

    # Compute eigenvalues (geodesic - GPU-only)
    n_gram = int(gram.shape[0])
    eigenvalues, _ = power_iteration_eigh(backend, gram, k=n_gram)

    # power_iteration_eigh returns eigenvalues in descending order.

    # Filter positive eigenvalues for stability
    eps = machine_epsilon(backend, eigenvalues)
    pos_mask = eigenvalues > eps
    pos_count_arr = backend.sum(backend.astype(pos_mask, "int32"))
    backend.eval(pos_mask, pos_count_arr)
    pos_count = int(backend.to_scalar(pos_count_arr))

    if pos_count < 2:
        # Degenerate case
        return EigenvalueDistribution(
            eigenvalues=eigenvalues,
            participation_ratio=1.0,
            spectral_entropy=0.0,
            condition_number=1.0,
            top_k_ratio=1.0,
        )

    pos_ev = eigenvalues[:pos_count]

    # Participation ratio: (sum(λ))^2 / sum(λ^2)
    # Measures effective number of significant eigenvalues
    sum_ev_arr = backend.sum(pos_ev)
    sum_ev_sq_arr = backend.sum(pos_ev * pos_ev)
    backend.eval(sum_ev_arr, sum_ev_sq_arr)
    sum_ev = float(backend.to_scalar(sum_ev_arr))
    sum_ev_sq = float(backend.to_scalar(sum_ev_sq_arr))
    participation_ratio = (sum_ev * sum_ev) / sum_ev_sq if sum_ev_sq > eps else 1.0

    # Spectral entropy: -sum(p * log(p)) where p = λ/sum(λ)
    # Measures how spread out the spectrum is
    p = pos_ev / sum_ev
    log_p = backend.where(p > eps, backend.log(p), backend.zeros_like(p))
    entropy_arr = -backend.sum(p * log_p)
    backend.eval(entropy_arr)
    spectral_entropy = float(backend.to_scalar(entropy_arr))

    # Condition number
    first_ev = backend.take(pos_ev, backend.array([0]), axis=0)
    last_ev = backend.take(pos_ev, backend.array([pos_count - 1]), axis=0)
    backend.eval(first_ev, last_ev)
    first_ev_val = float(backend.to_scalar(first_ev))
    last_ev_val = float(backend.to_scalar(last_ev))
    denom_eps = division_epsilon(backend, pos_ev)
    condition_number = first_ev_val / max(last_ev_val, denom_eps)

    # Top-k ratio (top 10 or all if fewer)
    k = min(10, pos_count)
    top_k_sum_arr = backend.sum(pos_ev[:k])
    backend.eval(top_k_sum_arr)
    top_k_sum = float(backend.to_scalar(top_k_sum_arr))
    top_k_ratio = top_k_sum / sum_ev if sum_ev > eps else 0.0

    return EigenvalueDistribution(
        eigenvalues=eigenvalues,
        participation_ratio=participation_ratio,
        spectral_entropy=spectral_entropy,
        condition_number=condition_number,
        top_k_ratio=top_k_ratio,
    )


def compare_distributions(
    dist1: EigenvalueDistribution,
    dist2: EigenvalueDistribution,
    label1: str,
    label2: str,
    backend: "Backend | None" = None,
) -> SpectralComparison:
    """Compare two eigenvalue distributions.

    Args:
        dist1, dist2: Eigenvalue distributions to compare.
        label1, label2: Labels for the distributions.
        backend: Compute backend.

    Returns:
        SpectralComparison with distance metrics.
    """
    backend = backend or get_default_backend()

    # Normalize eigenvalues to simplex weights
    ev1 = _array_to_list(backend, dist1.eigenvalues)
    ev2 = _array_to_list(backend, dist2.eigenvalues)

    eps = machine_epsilon(backend, dist1.eigenvalues)
    ev1_pos = [e for e in ev1 if e > eps]
    ev2_pos = [e for e in ev2 if e > eps]

    sum1 = sum(ev1_pos)
    sum2 = sum(ev2_pos)

    p1 = [e / sum1 for e in ev1_pos] if sum1 > 0 else ev1_pos
    p2 = [e / sum2 for e in ev2_pos] if sum2 > 0 else ev2_pos

    # Pad to same length for comparison
    max_len = max(len(p1), len(p2))
    p1 = p1 + [0.0] * (max_len - len(p1))
    p2 = p2 + [0.0] * (max_len - len(p2))

    # Wasserstein-1 distance (Earth Mover's Distance for 1D)
    # W1 = integral |F1(x) - F2(x)| dx where F is the CDF
    cdf1 = [sum(p1[: i + 1]) for i in range(len(p1))]
    cdf2 = [sum(p2[: i + 1]) for i in range(len(p2))]
    wasserstein = sum(abs(c1 - c2) for c1, c2 in zip(cdf1, cdf2)) / len(cdf1)

    # Kolmogorov-Smirnov statistic: max |F1(x) - F2(x)|
    ks_stat = max(abs(c1 - c2) for c1, c2 in zip(cdf1, cdf2))

    return SpectralComparison(
        source_label=label1,
        target_label=label2,
        participation_ratio_diff=dist1.participation_ratio - dist2.participation_ratio,
        spectral_entropy_diff=dist1.spectral_entropy - dist2.spectral_entropy,
        wasserstein_distance=wasserstein,
        ks_statistic=ks_stat,
    )

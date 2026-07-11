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

"""Spike-robust Marchenko-Pastur noise-bulk estimation."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend


@dataclass(frozen=True)
class MPNoiseEstimate:
    """Shared Marchenko-Pastur bulk estimate."""

    sigma_sq: float
    aspect_ratio: float
    lower_edge: float
    upper_edge: float
    positive_eigenvalue_count: int
    bulk_eigenvalue_count: int
    zero_eigenvalue_count: int


def _as_float_list(
    eigenvalues: Iterable[float] | object,
    backend: "Backend | None",
) -> list[float]:
    if backend is not None and not isinstance(eigenvalues, (list, tuple)):
        values = backend.tolist(eigenvalues)
    elif hasattr(eigenvalues, "tolist"):
        values = eigenvalues.tolist()  # type: ignore[union-attr]
    else:
        values = eigenvalues
    if isinstance(values, (float, int)):
        return [float(values)]
    return [float(value) for value in values]  # type: ignore[arg-type]


def _machine_epsilon(backend: "Backend | None") -> float:
    if backend is None:
        return sys.float_info.epsilon
    return float(backend.finfo().eps)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n_values = len(ordered)
    mid = n_values // 2
    if n_values % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mp_edges(sigma_sq: float, aspect_ratio: float) -> tuple[float, float]:
    sqrt_gamma = math.sqrt(aspect_ratio)
    lower = sigma_sq * (1.0 - sqrt_gamma) ** 2
    upper = sigma_sq * (1.0 + sqrt_gamma) ** 2
    if aspect_ratio > 1.0:
        lower = 0.0
    return lower, upper


def estimate_mp_noise(
    eigenvalues: Iterable[float] | object,
    *,
    n_samples: int,
    n_features: int,
    backend: "Backend | None" = None,
) -> MPNoiseEstimate:
    """Estimate MP noise variance without letting signal spikes set the edge.

    The MP bulk mean is ``sigma_sq``.  Signal spikes are above that bulk, so the
    estimator repeatedly fits the bulk from the current non-spike set and removes
    eigenvalues above the implied upper edge.  Exact rank-deficiency zeros are
    excluded before estimation; otherwise the ``N << D`` activation-probe regime
    estimates a zero-dominated bulk and can lose every measured signal direction.
    """
    if n_samples <= 0 or n_features <= 0:
        raise ValueError(
            f"n_samples={n_samples} and n_features={n_features} must be positive"
        )

    raw_values = _as_float_list(eigenvalues, backend)
    eps = _machine_epsilon(backend)
    finite_values = [max(0.0, value) for value in raw_values if math.isfinite(value)]
    if not finite_values:
        sigma_sq = eps
        lower, upper = _mp_edges(sigma_sq, float(n_features) / float(n_samples))
        return MPNoiseEstimate(sigma_sq, float(n_features) / float(n_samples), lower, upper, 0, 0, 0)

    max_eigenvalue = max(finite_values)
    zero_cutoff = max_eigenvalue * eps
    positive = [value for value in finite_values if value > zero_cutoff]
    zero_count = len(finite_values) - len(positive)
    aspect_ratio = float(n_features) / float(n_samples)

    if not positive:
        sigma_sq = max(zero_cutoff, eps)
        lower, upper = _mp_edges(sigma_sq, aspect_ratio)
        return MPNoiseEstimate(sigma_sq, aspect_ratio, lower, upper, 0, 0, zero_count)

    mean_positive = sum(positive) / len(positive)
    variance_positive = sum((value - mean_positive) ** 2 for value in positive) / len(
        positive
    )
    coefficient_of_variation = math.sqrt(max(variance_positive, 0.0)) / max(
        mean_positive, eps
    )
    if coefficient_of_variation <= math.sqrt(eps):
        sigma_sq = max(max_eigenvalue * eps, eps)
        lower, upper = _mp_edges(sigma_sq, aspect_ratio)
        return MPNoiseEstimate(
            sigma_sq=sigma_sq,
            aspect_ratio=aspect_ratio,
            lower_edge=lower,
            upper_edge=upper,
            positive_eigenvalue_count=len(positive),
            bulk_eigenvalue_count=0,
            zero_eigenvalue_count=zero_count,
        )

    candidates = sorted(positive, reverse=True)
    bulk = list(candidates)
    previous_bulk_count = -1
    upper = 0.0
    lower = 0.0

    while len(bulk) != previous_bulk_count:
        previous_bulk_count = len(bulk)
        median_sigma = max(_median(bulk), eps)
        lower, upper = _mp_edges(median_sigma, aspect_ratio)
        bulk = [value for value in bulk if value <= upper]
        if not bulk:
            break

    if bulk:
        sigma_sq = max(sum(bulk) / len(bulk), eps)
    else:
        sigma_sq = max(min(positive) * eps, eps)
    lower, upper = _mp_edges(sigma_sq, aspect_ratio)
    return MPNoiseEstimate(
        sigma_sq=sigma_sq,
        aspect_ratio=aspect_ratio,
        lower_edge=lower,
        upper_edge=upper,
        positive_eigenvalue_count=len(positive),
        bulk_eigenvalue_count=len(bulk),
        zero_eigenvalue_count=zero_count,
    )


__all__ = ["MPNoiseEstimate", "estimate_mp_noise"]

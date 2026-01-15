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

"""Effective rank diagnostics for activation manifolds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    geodesic_svd,
    safe_log_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class EffectiveRankResult:
    """Effective rank statistics for a point cloud of activations."""

    renyi_effective_rank: float
    shannon_effective_rank: float
    spectral_entropy: float
    sample_count: int
    feature_dim: int
    n_singular_values: int


class EffectiveRank:
    """Compute effective rank of activation manifolds from centered activations."""

    def __init__(self, backend: "Backend | None" = None) -> None:
        self._backend = backend or get_default_backend()

    def compute(self, activations: "Array") -> EffectiveRankResult:
        """Compute Renyi and Shannon effective rank from centered activations."""
        b = self._backend
        arr = b.array(activations) if not hasattr(activations, "shape") else activations
        shape = arr.shape

        if len(shape) == 0:
            return EffectiveRankResult(
                renyi_effective_rank=0.0,
                shannon_effective_rank=0.0,
                spectral_entropy=0.0,
                sample_count=0,
                feature_dim=0,
                n_singular_values=0,
            )

        if len(shape) == 1:
            feature_dim = int(shape[0])
            sample_count = 1
        else:
            feature_dim = int(shape[-1])
            sample_count = 1
            for dim in shape[:-1]:
                sample_count *= int(dim)

        if feature_dim == 0 or sample_count == 0:
            return EffectiveRankResult(
                renyi_effective_rank=0.0,
                shannon_effective_rank=0.0,
                spectral_entropy=0.0,
                sample_count=sample_count,
                feature_dim=feature_dim,
                n_singular_values=0,
            )

        arr_2d = b.reshape(arr, (sample_count, feature_dim))
        mean = b.mean(arr_2d, axis=0)
        centered = arr_2d - mean
        b.eval(centered)

        _, singular_values, _ = geodesic_svd(b, centered)
        b.eval(singular_values)

        n_sv = int(singular_values.shape[0])
        if n_sv == 0:
            return EffectiveRankResult(
                renyi_effective_rank=0.0,
                shannon_effective_rank=0.0,
                spectral_entropy=0.0,
                sample_count=sample_count,
                feature_dim=feature_dim,
                n_singular_values=0,
            )

        eigvals = singular_values * singular_values
        sum_eig = b.sum(eigvals)
        sum_eig_sq = b.sum(eigvals * eigvals)
        b.eval(sum_eig, sum_eig_sq)

        sum_eig_val = float(b.to_scalar(sum_eig))
        sum_eig_sq_val = float(b.to_scalar(sum_eig_sq))
        eps = division_epsilon(b, eigvals)

        if sum_eig_sq_val > eps:
            renyi_rank = (sum_eig_val * sum_eig_val) / sum_eig_sq_val
        else:
            renyi_rank = 0.0

        if sum_eig_val > eps:
            p = eigvals / sum_eig_val
            log_eps = safe_log_epsilon(b, eigvals)
            eps_arr = b.full(p.shape, log_eps, dtype=p.dtype)
            p_safe = b.where(p > log_eps, p, eps_arr)
            entropy_terms = -p * b.log(p_safe)
            spectral_entropy_arr = b.sum(entropy_terms)
            b.eval(spectral_entropy_arr)
            spectral_entropy = float(b.to_scalar(spectral_entropy_arr))

            shannon_rank_arr = b.exp(spectral_entropy_arr)
            b.eval(shannon_rank_arr)
            shannon_rank = float(b.to_scalar(shannon_rank_arr))
        else:
            spectral_entropy = 0.0
            shannon_rank = 0.0

        return EffectiveRankResult(
            renyi_effective_rank=renyi_rank,
            shannon_effective_rank=shannon_rank,
            spectral_entropy=spectral_entropy,
            sample_count=sample_count,
            feature_dim=feature_dim,
            n_singular_values=n_sv,
        )

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

"""Jacobian rank diagnostics from gradient probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class JacobianRankResult:
    renyi_effective_rank: float
    shannon_effective_rank: float
    spectral_entropy: float
    projection_count: int
    parameter_dim: int
    n_singular_values: int


def estimate_jacobian_rank(
    gradients: "Array",
    *,
    backend: "Backend | None" = None,
) -> JacobianRankResult:
    """Estimate Jacobian rank from projected gradient samples."""
    b = backend or get_default_backend()
    grads = b.array(gradients) if not hasattr(gradients, "shape") else gradients
    b.eval(grads)

    shape = grads.shape
    if len(shape) == 0:
        return JacobianRankResult(0.0, 0.0, 0.0, 0, 0, 0)

    if len(shape) == 1:
        projection_count = 1
        parameter_dim = int(shape[0])
    else:
        parameter_dim = int(shape[-1])
        projection_count = 1
        for dim in shape[:-1]:
            projection_count *= int(dim)

    if projection_count == 0 or parameter_dim == 0:
        return JacobianRankResult(0.0, 0.0, 0.0, projection_count, parameter_dim, 0)

    grads_2d = b.reshape(grads, (projection_count, parameter_dim))
    er = EffectiveRank(b).compute(grads_2d)

    return JacobianRankResult(
        renyi_effective_rank=er.renyi_effective_rank,
        shannon_effective_rank=er.shannon_effective_rank,
        spectral_entropy=er.spectral_entropy,
        projection_count=projection_count,
        parameter_dim=parameter_dim,
        n_singular_values=er.n_singular_values,
    )


__all__ = ["JacobianRankResult", "estimate_jacobian_rank"]

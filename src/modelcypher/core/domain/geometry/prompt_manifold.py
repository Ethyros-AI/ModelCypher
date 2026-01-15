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

"""Prompt-manifold basis derivation for local geometry probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.effective_rank import EffectiveRank, EffectiveRankResult
from modelcypher.core.domain.geometry.numerical_stability import (
    compute_median_nonzero,
    geodesic_svd,
    machine_epsilon,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class PromptManifoldBasis:
    mean: "Array"
    basis: "Array"
    basis_rank: int
    effective_rank: EffectiveRankResult
    sample_count: int
    feature_dim: int
    scale: float


def _empty_basis(
    backend: "Backend",
    dtype: object,
    sample_count: int,
    feature_dim: int,
) -> PromptManifoldBasis:
    zero_rank = 0
    return PromptManifoldBasis(
        mean=backend.zeros((feature_dim,), dtype=dtype),
        basis=backend.zeros((0, feature_dim), dtype=dtype),
        basis_rank=zero_rank,
        effective_rank=EffectiveRankResult(
            renyi_effective_rank=0.0,
            shannon_effective_rank=0.0,
            spectral_entropy=0.0,
            sample_count=sample_count,
            feature_dim=feature_dim,
            n_singular_values=0,
        ),
        sample_count=sample_count,
        feature_dim=feature_dim,
        scale=0.0,
    )


def derive_prompt_manifold_basis(
    embeddings: "Array",
    *,
    basis_rank: int | None = None,
    backend: "Backend | None" = None,
) -> PromptManifoldBasis:
    """Derive a prompt-manifold basis from pooled prompt embeddings."""
    b = backend or get_default_backend()
    arr = b.array(embeddings) if not hasattr(embeddings, "shape") else embeddings
    b.eval(arr)

    shape = arr.shape
    if len(shape) == 0:
        return _empty_basis(b, arr.dtype, 0, 0)

    if len(shape) == 1:
        feature_dim = int(shape[0])
        sample_count = 1
    else:
        feature_dim = int(shape[-1])
        sample_count = 1
        for dim in shape[:-1]:
            sample_count *= int(dim)

    if feature_dim == 0 or sample_count == 0:
        return _empty_basis(b, arr.dtype, sample_count, feature_dim)

    arr_2d = b.reshape(arr, (sample_count, feature_dim))
    mean = b.mean(arr_2d, axis=0)
    centered = arr_2d - mean
    b.eval(mean, centered)

    er = EffectiveRank(b).compute(arr_2d)
    max_rank = min(sample_count, feature_dim)
    if max_rank == 0:
        return _empty_basis(b, arr.dtype, sample_count, feature_dim)

    if basis_rank is None:
        eps = machine_epsilon(b, arr_2d)
        basis_rank = int(round(er.renyi_effective_rank + eps))
    else:
        basis_rank = int(basis_rank)

    if basis_rank < 0:
        basis_rank = 0
    if basis_rank > max_rank:
        basis_rank = max_rank

    if basis_rank == 0:
        return PromptManifoldBasis(
            mean=mean,
            basis=b.zeros((0, feature_dim), dtype=arr.dtype),
            basis_rank=0,
            effective_rank=er,
            sample_count=sample_count,
            feature_dim=feature_dim,
            scale=0.0,
        )

    _, _, v_t = geodesic_svd(b, centered, k=basis_rank)
    b.eval(v_t)

    norms = b.sqrt(b.sum(centered * centered, axis=1))
    scale = compute_median_nonzero(norms, b)

    return PromptManifoldBasis(
        mean=mean,
        basis=v_t,
        basis_rank=basis_rank,
        effective_rank=er,
        sample_count=sample_count,
        feature_dim=feature_dim,
        scale=scale,
    )


def apply_prompt_basis(
    base_embeddings: "Array",
    basis: "Array",
    coefficients: "Array",
    *,
    backend: "Backend | None" = None,
) -> "Array":
    """Apply a prompt-manifold basis to a base embedding sequence."""
    b = backend or get_default_backend()
    base = b.array(base_embeddings) if not hasattr(base_embeddings, "shape") else base_embeddings
    basis_arr = b.array(basis) if not hasattr(basis, "shape") else basis
    coeffs = b.array(coefficients) if not hasattr(coefficients, "shape") else coefficients

    if len(basis_arr.shape) < 2 or int(basis_arr.shape[0]) == 0:
        return base

    basis_rank = int(basis_arr.shape[0])
    feature_dim = int(basis_arr.shape[-1])
    coeffs_vec = b.reshape(coeffs, (1, basis_rank))
    delta = b.matmul(coeffs_vec, basis_arr)
    delta = b.reshape(delta, (1, feature_dim))

    base_shape = base.shape
    if len(base_shape) == 1:
        delta_vec = b.reshape(delta, (feature_dim,))
        return base + delta_vec

    expanded_shape = (1,) * (len(base_shape) - 2) + (1, feature_dim)
    delta_expanded = b.reshape(delta, expanded_shape)
    delta_broadcast = b.broadcast_to(delta_expanded, base_shape)
    return base + delta_broadcast


__all__ = [
    "PromptManifoldBasis",
    "derive_prompt_manifold_basis",
    "apply_prompt_basis",
]

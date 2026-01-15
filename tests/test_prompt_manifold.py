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

from __future__ import annotations

from modelcypher.core.domain.geometry.jacobian_rank import estimate_jacobian_rank
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.prompt_manifold import (
    apply_prompt_basis,
    derive_prompt_manifold_basis,
)


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def test_prompt_manifold_basis_rank(any_backend):
    backend = any_backend
    points = backend.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )

    basis = derive_prompt_manifold_basis(points, backend=backend)
    eps = _eps(backend, basis.effective_rank.renyi_effective_rank)

    assert basis.sample_count == 4
    assert basis.feature_dim == 2
    assert abs(basis.effective_rank.renyi_effective_rank - 2.0) <= eps
    assert basis.basis_rank == 2
    assert basis.basis.shape == (2, 2)
    assert abs(basis.scale - 1.0) <= eps


def test_apply_prompt_basis_adds_direction(any_backend):
    backend = any_backend
    base = backend.array([[0.0, 0.0], [1.0, 1.0]])
    basis = backend.array([[1.0, 0.0], [0.0, 1.0]])
    coeffs = backend.array([2.0, -1.0])

    result = apply_prompt_basis(base, basis, coeffs, backend=backend)
    backend.eval(result)
    eps = _eps(backend)

    expected = backend.array([[2.0, -1.0], [3.0, 0.0]])
    backend.eval(expected)

    diff = backend.abs(result - expected)
    max_diff = backend.max(diff)
    backend.eval(max_diff)
    assert float(backend.to_scalar(max_diff)) <= eps


def test_jacobian_rank_balanced(any_backend):
    backend = any_backend
    grads = backend.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )

    result = estimate_jacobian_rank(grads, backend=backend)
    eps = _eps(backend, result.renyi_effective_rank, result.shannon_effective_rank)

    assert result.projection_count == 4
    assert result.parameter_dim == 2
    assert abs(result.renyi_effective_rank - 2.0) <= eps
    assert abs(result.shannon_effective_rank - 2.0) <= eps

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

from modelcypher.core.domain.geometry.effective_rank import EffectiveRank
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(backend, *values: float) -> float:
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))


def test_effective_rank_balanced(any_backend):
    backend = any_backend
    points = backend.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )

    result = EffectiveRank(backend).compute(points)
    eps = _eps(backend, result.renyi_effective_rank, result.shannon_effective_rank)

    assert result.sample_count == 4
    assert result.feature_dim == 2
    assert abs(result.renyi_effective_rank - 2.0) <= eps
    assert abs(result.shannon_effective_rank - 2.0) <= eps
    assert result.spectral_entropy >= -eps


def test_effective_rank_rank_one(any_backend):
    backend = any_backend
    points = backend.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [2.0, 0.0],
            [-2.0, 0.0],
        ]
    )

    result = EffectiveRank(backend).compute(points)
    eps = _eps(backend, result.renyi_effective_rank, result.shannon_effective_rank)

    assert abs(result.renyi_effective_rank - 1.0) <= eps
    assert abs(result.shannon_effective_rank - 1.0) <= eps

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

import pytest

from modelcypher.core.domain.geometry.gromov_wasserstein import Config, GromovWassersteinDistance


def test_gw_identity_distance() -> None:
    """Self-comparison should give zero GW distance with uniform coupling."""
    gw = GromovWassersteinDistance()
    points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    distances = gw.compute_pairwise_distances(points)
    result = gw.compute(distances, distances)
    assert result.distance == 0.0
    assert result.converged is True
    assert result.iterations == 0
    assert len(result.coupling) == 3
    # Use larger tolerance for float32 precision, convert to float for comparison
    assert float(result.coupling[0][0]) == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert float(result.coupling[1][1]) == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert float(result.coupling[2][2]) == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_gw_permutation_distance_small() -> None:
    """Permuted points should have near-zero GW distance (same shape)."""
    gw = GromovWassersteinDistance()
    points_a = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    permutation = [2, 0, 1]
    points_b = [points_a[idx] for idx in permutation]
    dist_a = gw.compute_pairwise_distances(points_a)
    dist_b = gw.compute_pairwise_distances(points_b)
    # Use default config which now uses Frank-Wolfe with permutation search
    config = Config()
    result = gw.compute(dist_a, dist_b, config=config)
    assert result.distance < 0.02
    # Verify coupling marginals sum to uniform distribution
    row_mass = [sum(row) for row in result.coupling]
    col_mass = [
        sum(result.coupling[i][j] for i in range(len(result.coupling)))
        for j in range(len(result.coupling[0]))
    ]
    assert max(abs(value - 1.0 / 3.0) for value in row_mass) < 0.02
    assert max(abs(value - 1.0 / 3.0) for value in col_mass) < 0.02

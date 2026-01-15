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

"""Parity tests for backend shortest path implementations."""

from __future__ import annotations

from modelcypher.core.domain.geometry.numerical_stability import regularization_epsilon


def _max_abs(backend, array) -> float:
    diff = backend.max(backend.abs(array))
    backend.eval(diff)
    return float(backend.to_scalar(diff))


def test_single_source_matches_floyd_warshall(any_backend) -> None:
    """Single-source shortest paths should match Floyd-Warshall row."""
    b = any_backend
    dist = b.array(
        [
            [0.0, 2.0, 9.0, 10.0],
            [2.0, 0.0, 6.0, 4.0],
            [9.0, 6.0, 0.0, 3.0],
            [10.0, 4.0, 3.0, 0.0],
        ]
    )

    fw = b.floyd_warshall(dist)
    sssp = b.single_source_shortest_paths(dist, 0)
    b.eval(fw, sssp)

    row0 = fw[0]
    tol = regularization_epsilon(b, fw)
    assert _max_abs(b, row0 - sssp) <= tol

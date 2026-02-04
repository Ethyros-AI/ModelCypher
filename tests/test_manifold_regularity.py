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

"""Manifold regularity tests (requires backend)."""

import pytest

from modelcypher.core.domain.geometry.manifold_dimensionality import ManifoldDimensionality
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def test_manifold_regularity_intrinsic_dimension(any_backend):
    """Test intrinsic dimension regularity."""
    import random

    random.seed(42)
    # Points on a 2D manifold embedded in 10D
    # Use sufficient noise (0.3) to break grid structure - TwoNN requires
    # continuous manifold data, not discrete grids. With small noise (0.01),
    # all nearest neighbor ratios mu ~= 1, making regression unstable.
    n = 200
    points = [
        [float(i % 14) + random.gauss(0, 0.3), float(i // 14) + random.gauss(0, 0.3)]
        + [random.gauss(0, 0.01)] * 8
        for i in range(n)
    ]

    summary = ManifoldDimensionality.estimate_id(points)

    eps = division_epsilon(any_backend, any_backend.array([1.0]))
    assert summary.intrinsic_dimension >= eps
    assert summary.intrinsic_dimension <= len(points[0]) + eps

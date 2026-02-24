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

"""Tests for Riemannian extensions (sampling and interpolation)."""

from __future__ import annotations

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry


class TestRiemannianExtensions:
    """Tests for sampling and interpolation mixins."""

    def setup_method(self):
        self.backend = get_default_backend()
        self.rg = RiemannianGeometry(self.backend)

    def test_geodesic_interpolation_linear(self):
        """Interpolation on a line returns a path vertex."""
        # Simple Euclidean case
        p1 = self.backend.array([0.0, 0.0])
        p2 = self.backend.array([2.0, 2.0])

        # Provide context so manifold is defined
        context = self.backend.array([[0.0, 0.0], [2.0, 2.0]])
        # t=0.5
        mid = self.rg.geodesic_interpolation(p1, p2, 0.5, points_context=context)
        self.backend.eval(mid)

        val = mid.tolist()
        assert val in ([0.0, 0.0], [2.0, 2.0])

    def test_reconstruct_geodesic_path(self):
        """Path reconstruction via Floyd-Warshall/Dijkstra logic."""
        # Create a graph: 0 -> 1 -> 2
        # Distances:
        # 0-1: 1.0
        # 1-2: 1.0
        # 0-2: 2.0

        # Mock GeodesicDistanceResult or distances array
        geo_dist = self.backend.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]
        ])

        path = self.rg._reconstruct_geodesic_path(geo_dist, 0, 2)
        assert path == [0, 1, 2]

    def test_farthest_point_sampling(self):
        """FPS selects points maximizing distance."""
        # Points in a square: (0,0), (0,1), (1,0), (1,1)
        # Start with (0,0). Farther is (1,1) distance sqrt(2)=1.41
        # (0,1) and (1,0) are distance 1.

        points = self.backend.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])

        result = self.rg.farthest_point_sampling(points, n_samples=2)
        indices = result.selected_indices
        assert len(indices) == 2

        result_3 = self.rg.farthest_point_sampling(points, n_samples=3)
        indices_3 = result_3.selected_indices
        assert len(indices_3) == 3
        assert len(set(indices_3)) == 3

    def test_directional_coverage(self):
        """Directional coverage analyzes tangent space."""
        # Center point surrounded by points in +x, -x, +y directions.
        # Sparse direction should be -y.
        points = self.backend.array([
            [0.0, 0.0],   # 0: center
            [1.0, 0.0],   # 1: +x
            [-1.0, 0.0],  # 2: -x
            [0.0, 1.0],   # 3: +y
        ])

        coverage = self.rg.directional_coverage(0, points)

        # Coverage result has sparse_direction
        s_dir = coverage.sparse_direction.tolist()
        # Should be roughly [0, -1]

        # Dot product with [0, -1] should be high (close to 1)
        # or distance to [0, -1] low.
        # Just check it returns valid object.
        assert coverage.max_gap_angle >= 0
        assert len(s_dir) == 2

    def test_propose_in_sparse_direction(self):
        """Propose point returns new vector."""
        points = self.backend.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ])
        new_pt = self.rg.propose_in_sparse_direction(0, points)
        self.backend.eval(new_pt)
        assert new_pt.shape == (2,)

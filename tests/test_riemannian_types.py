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

"""Tests for riemannian_types.py - Result dataclasses."""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from modelcypher.core.domain.geometry.riemannian_types import (
    CurvatureEstimate,
    DirectionalCoverage,
    FarthestPointSamplingResult,
    FrechetMeanResult,
    GeodesicDistanceResult,
)


class TestRiemannianTypes:
    """Tests for Riemannian geometry result dataclasses."""

    def test_frechet_mean_result(self):
        """FrechetMeanResult stores fields correctly."""
        mean_mock = Mock()
        result = FrechetMeanResult(
            mean=mean_mock,
            iterations=5,
            converged=True,
            final_variance=10.0,
        )
        assert result.mean is mean_mock
        assert result.iterations == 5
        assert result.converged is True
        assert result.final_variance == 10.0

    def test_geodesic_distance_result(self):
        """GeodesicDistanceResult stores fields correctly."""
        dist_mock = Mock()
        adj_mock = Mock()
        result = GeodesicDistanceResult(
            distances=dist_mock,
            adjacency=adj_mock,
            inf_value=float("inf"),
            k_neighbors=10,
            connected=True,
        )
        assert result.distances is dist_mock
        assert result.k_neighbors == 10

    def test_curvature_estimate(self):
        """CurvatureEstimate stores fields correctly."""
        result = CurvatureEstimate(
            sectional_curvature=0.5,
            is_positive=True,
            is_negative=False,
            confidence=0.8,
        )
        assert result.sectional_curvature == 0.5
        assert result.is_positive is True

    def test_directional_coverage(self):
        """DirectionalCoverage stores fields correctly."""
        sparse_dir = Mock()
        neighbors = Mock()
        result = DirectionalCoverage(
            sparse_direction=sparse_dir,
            max_gap_angle=0.5,
            coverage_variance=0.8,
            neighbor_directions=neighbors,
            point_idx=1,
        )
        assert result.sparse_direction is sparse_dir
        assert result.point_idx == 1

    def test_farthest_point_sampling_result(self):
        """FarthestPointSamplingResult stores fields correctly."""
        min_dists = Mock()
        result = FarthestPointSamplingResult(
            selected_indices=[0, 1, 5],
            min_distances=min_dists,
            coverage_radius=2.5,
        )
        assert result.selected_indices == [0, 1, 5]
        assert result.coverage_radius == 2.5

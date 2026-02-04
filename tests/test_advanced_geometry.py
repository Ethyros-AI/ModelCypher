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

"""Advanced geometry tests requiring MLX (Apple Silicon)."""

import pytest

from tests.conftest import HAS_MLX

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon
from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension
from modelcypher.core.domain.geometry.manifold_clusterer import ManifoldClusterer, ManifoldPoint


def _eps() -> float:
    backend = get_default_backend()
    return division_epsilon(backend, backend.array([1.0]))


def test_intrinsic_dimension_estimator_mle():
    # Compare 1D vs 2D manifolds embedded in 10D space
    backend = get_default_backend()
    N = 200
    D = 10

    x1 = backend.random_normal((N, 1))
    line_points = backend.zeros((N, D))
    line_points[:, :1] = x1 * 10.0

    x2 = backend.random_normal((N, 2))
    plane_points = backend.zeros((N, D))
    plane_points[:, :2] = x2 * 10.0

    estimator = IntrinsicDimension()
    line_est = estimator.compute(line_points)
    plane_est = estimator.compute(plane_points)

    eps = _eps()
    assert plane_est.intrinsic_dimension - line_est.intrinsic_dimension > eps


def test_manifold_clusterer_simple():
    # Creates two distinct clusters of ManifoldPoints

    def create_point(base_entropy, mean_gate):
        return ManifoldPoint(
            mean_entropy=base_entropy,
            entropy_variance=0.1,
            first_token_entropy=base_entropy,
            gate_count=5,
            mean_gate_similarity=mean_gate,
            dominant_gate_category=0,
            entropy_path_correlation=0.5,
            assessment_strength=0.5,
            prompt_hash="hash",
        )

    cluster1 = [create_point(1.0, 0.9) for _ in range(5)]
    cluster2 = [create_point(5.0, 0.2) for _ in range(5)]

    all_points = cluster1 + cluster2

    clusterer = ManifoldClusterer()
    result = clusterer.cluster(all_points)

    # Expect 2 regions
    assert len(result.regions) == 2
    assert result.noise_points == ()

    # Check region centroids
    centroids = sorted([r.centroid.mean_entropy for r in result.regions])
    eps = _eps()
    assert abs(centroids[0] - 1.0) <= eps
    assert abs(centroids[1] - 5.0) <= eps


def test_manifold_clusterer_noise():
    # 5 points in cluster, 1 outlier far away
    def fn(e):
        return ManifoldPoint(
            mean_entropy=e,
            entropy_variance=0.0,
            first_token_entropy=e,
            gate_count=0,
            mean_gate_similarity=0.0,
            dominant_gate_category=0.0,
            entropy_path_correlation=0.0,
            assessment_strength=0.0,
            prompt_hash="h",
        )
    points = [fn(1.0) for _ in range(5)]
    outlier = fn(100.0)

    clusterer = ManifoldClusterer()
    result = clusterer.cluster(points + [outlier])

    assert len(result.regions) == 1
    assert len(result.noise_points) == 1
    assert abs(result.noise_points[0].mean_entropy - 100.0) <= _eps()

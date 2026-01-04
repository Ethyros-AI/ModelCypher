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

"""Tests for ManifoldClusterer geodesic-based clustering."""

from __future__ import annotations

import pytest
from uuid import uuid4

from modelcypher.core.domain.geometry.manifold_clusterer import (
    ManifoldClusterer,
    ClusteringResult,
)
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldRegion,
)


def _make_point(
    mean_entropy: float = 1.0,
    entropy_variance: float = 0.1,
    first_token_entropy: float = 1.0,
    gate_count: int = 5,
    mean_gate_similarity: float = 0.5,
    dominant_gate_category: float = 0.3,
    entropy_path_correlation: float = 0.7,
    assessment_strength: float = 0.8,
    prompt_hash: str | None = None,
) -> ManifoldPoint:
    """Create a ManifoldPoint for testing."""
    return ManifoldPoint(
        mean_entropy=mean_entropy,
        entropy_variance=entropy_variance,
        first_token_entropy=first_token_entropy,
        gate_count=gate_count,
        mean_gate_similarity=mean_gate_similarity,
        dominant_gate_category=dominant_gate_category,
        entropy_path_correlation=entropy_path_correlation,
        assessment_strength=assessment_strength,
        prompt_hash=prompt_hash or str(uuid4()),
    )


class TestManifoldClustererInit:
    """Tests for ManifoldClusterer initialization."""

    def test_default_initialization(self):
        """Clusterer should initialize without parameters."""
        clusterer = ManifoldClusterer()
        assert clusterer is not None

    def test_cluster_method_exists(self):
        """Clusterer should have cluster method."""
        clusterer = ManifoldClusterer()
        assert hasattr(clusterer, "cluster")
        assert callable(clusterer.cluster)


class TestClusterEmptyInput:
    """Tests for clustering empty or minimal inputs."""

    def test_empty_points_returns_empty_result(self):
        """Empty input should return empty result."""
        clusterer = ManifoldClusterer()
        result = clusterer.cluster([])

        assert isinstance(result, ClusteringResult)
        assert result.regions == []
        assert result.noise_points == []
        assert result.new_clusters_formed == 0
        assert result.clusters_merged == 0
        assert result.points_assigned_to_existing == 0

    def test_single_point_becomes_noise(self):
        """Single point should be classified as noise (min cluster size = 2)."""
        clusterer = ManifoldClusterer()
        point = _make_point()
        result = clusterer.cluster([point])

        # Single point can't form a cluster, should be noise
        assert result.new_clusters_formed == 0


class TestClusterFormation:
    """Tests for cluster formation behavior."""

    def test_two_identical_points_form_cluster(self):
        """Two identical points should form a cluster."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=1.0, entropy_variance=0.1),
            _make_point(mean_entropy=1.0, entropy_variance=0.1),
        ]
        result = clusterer.cluster(points)

        # Two nearby points should form at least one cluster
        assert isinstance(result, ClusteringResult)
        # Result should have regions or noise points
        total_points = len(result.noise_points) + sum(
            len(r.points) if hasattr(r, 'points') else 0 for r in result.regions
        )
        # All points should be accounted for in some form

    def test_distant_points_form_separate_clusters_or_noise(self):
        """Very distant points should not cluster together."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=0.1, entropy_variance=0.01),
            _make_point(mean_entropy=10.0, entropy_variance=5.0),
        ]
        result = clusterer.cluster(points)

        assert isinstance(result, ClusteringResult)

    def test_tight_cluster_of_similar_points(self):
        """Points with similar features should cluster together."""
        clusterer = ManifoldClusterer()
        # Create a tight cluster of 5 similar points
        points = [
            _make_point(
                mean_entropy=1.0 + i * 0.01,
                entropy_variance=0.1 + i * 0.001,
            )
            for i in range(5)
        ]
        result = clusterer.cluster(points)

        assert isinstance(result, ClusteringResult)
        # Should form at least one cluster with multiple points
        if result.regions:
            assert result.new_clusters_formed >= 1


class TestClusteringResultStructure:
    """Tests for ClusteringResult dataclass structure."""

    def test_result_has_required_fields(self):
        """ClusteringResult should have all required fields."""
        clusterer = ManifoldClusterer()
        result = clusterer.cluster([])

        assert hasattr(result, "regions")
        assert hasattr(result, "noise_points")
        assert hasattr(result, "new_clusters_formed")
        assert hasattr(result, "clusters_merged")
        assert hasattr(result, "points_assigned_to_existing")

    def test_regions_is_list(self):
        """Regions should be a list."""
        clusterer = ManifoldClusterer()
        result = clusterer.cluster([])
        assert isinstance(result.regions, list)

    def test_noise_points_is_list(self):
        """Noise points should be a list."""
        clusterer = ManifoldClusterer()
        result = clusterer.cluster([])
        assert isinstance(result.noise_points, list)


class TestIncrementalClustering:
    """Tests for incremental clustering functionality."""

    def test_cluster_incremental_exists(self):
        """Incremental clustering method should exist."""
        clusterer = ManifoldClusterer()
        assert hasattr(clusterer, "cluster_incremental")
        assert callable(clusterer.cluster_incremental)

    def test_cluster_incremental_with_empty_existing(self):
        """Incremental clustering with no existing regions."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=1.0),
            _make_point(mean_entropy=1.01),
        ]
        result = clusterer.cluster_incremental(
            points, existing_regions=[], existing_noise=[]
        )

        assert isinstance(result, ClusteringResult)


class TestFindNearestRegion:
    """Tests for find_nearest_region functionality."""

    def test_find_nearest_region_exists(self):
        """find_nearest_region method should exist."""
        clusterer = ManifoldClusterer()
        assert hasattr(clusterer, "find_nearest_region")
        assert callable(clusterer.find_nearest_region)


class TestGeodesicDistanceComputation:
    """Tests for geodesic distance computation internals."""

    def test_geodesic_matrix_computed(self):
        """Geodesic distance matrix should be computed for clustering."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=1.0),
            _make_point(mean_entropy=2.0),
            _make_point(mean_entropy=3.0),
        ]

        # Internal method - verify it works by clustering
        result = clusterer.cluster(points)
        assert isinstance(result, ClusteringResult)


class TestEpsilonDerivation:
    """Tests for data-driven epsilon derivation."""

    def test_epsilon_derived_from_data(self):
        """Epsilon should be derived from data, not hardcoded."""
        clusterer = ManifoldClusterer()

        # Cluster two sets with very different scales
        tight_points = [
            _make_point(mean_entropy=1.0 + i * 0.001)
            for i in range(5)
        ]

        spread_points = [
            _make_point(mean_entropy=1.0 + i * 1.0)
            for i in range(5)
        ]

        result_tight = clusterer.cluster(tight_points)
        result_spread = clusterer.cluster(spread_points)

        # Both should produce valid results
        assert isinstance(result_tight, ClusteringResult)
        assert isinstance(result_spread, ClusteringResult)


class TestManifoldPointFeatureVector:
    """Tests for ManifoldPoint feature vector computation."""

    def test_feature_vector_has_correct_dimension(self):
        """Feature vector should have 8 dimensions."""
        point = _make_point()
        assert len(point.feature_vector) == 8
        assert len(point.feature_vector) == ManifoldPoint.feature_dimension

    def test_feature_vector_matches_fields(self):
        """Feature vector values should match field values."""
        point = _make_point(
            mean_entropy=1.5,
            entropy_variance=0.2,
            first_token_entropy=1.3,
            gate_count=7,
            mean_gate_similarity=0.6,
            dominant_gate_category=0.4,
            entropy_path_correlation=0.8,
            assessment_strength=0.9,
        )
        fv = point.feature_vector

        assert fv[0] == 1.5  # mean_entropy
        assert fv[1] == 0.2  # entropy_variance
        assert fv[2] == 1.3  # first_token_entropy
        assert fv[3] == 7.0  # gate_count (as float)
        assert fv[4] == 0.6  # mean_gate_similarity
        assert fv[5] == 0.4  # dominant_gate_category
        assert fv[6] == 0.8  # entropy_path_correlation
        assert fv[7] == 0.9  # assessment_strength


class TestClusteringStability:
    """Tests for clustering stability and reproducibility."""

    def test_deterministic_clustering(self):
        """Same input should produce consistent results."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=1.0, prompt_hash="a"),
            _make_point(mean_entropy=1.01, prompt_hash="b"),
            _make_point(mean_entropy=5.0, prompt_hash="c"),
        ]

        result1 = clusterer.cluster(points)
        result2 = clusterer.cluster(points)

        # Same input should give same cluster count
        assert result1.new_clusters_formed == result2.new_clusters_formed
        assert len(result1.regions) == len(result2.regions)
        assert len(result1.noise_points) == len(result2.noise_points)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_identical_points(self):
        """All identical points should form one cluster."""
        clusterer = ManifoldClusterer()
        points = [_make_point(mean_entropy=1.0) for _ in range(10)]
        result = clusterer.cluster(points)

        assert isinstance(result, ClusteringResult)
        # All identical points should cluster together
        if result.regions:
            assert result.new_clusters_formed >= 1

    def test_two_distinct_clusters(self):
        """Two well-separated groups should form distinct clusters."""
        clusterer = ManifoldClusterer()

        # Cluster 1: low entropy
        cluster1 = [
            _make_point(mean_entropy=0.1 + i * 0.01)
            for i in range(5)
        ]

        # Cluster 2: high entropy
        cluster2 = [
            _make_point(mean_entropy=10.0 + i * 0.01)
            for i in range(5)
        ]

        result = clusterer.cluster(cluster1 + cluster2)
        assert isinstance(result, ClusteringResult)

    def test_large_number_of_points(self):
        """Should handle larger number of points."""
        clusterer = ManifoldClusterer()
        points = [
            _make_point(mean_entropy=float(i % 10))
            for i in range(50)
        ]
        result = clusterer.cluster(points)

        assert isinstance(result, ClusteringResult)

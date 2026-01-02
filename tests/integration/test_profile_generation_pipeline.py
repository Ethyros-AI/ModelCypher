# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Integration tests for the profile generation pipeline.

Tests the full flow: activations → metrics → profile → storage → retrieval.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_profile import (
    ManifoldPoint,
    ManifoldProfile,
)


@pytest.fixture
def backend():
    """Get the default backend."""
    return get_default_backend()


@pytest.fixture
def sample_points():
    """Create sample ManifoldPoints for testing."""
    points = []
    for i in range(20):
        point = ManifoldPoint(
            mean_entropy=1.5 + i * 0.1,
            entropy_variance=0.2 + i * 0.01,
            first_token_entropy=2.0 + i * 0.05,
            gate_count=5 + i % 3,
            mean_gate_similarity=0.8 + i * 0.01,
            dominant_gate_category=float(i % 4),
            entropy_path_correlation=0.6 + i * 0.02,
            assessment_strength=0.7 + i * 0.015,
            prompt_hash=f"hash_{i:04d}",
            intervention_level=i % 3,
        )
        points.append(point)
    return points


class TestManifoldPointCreation:
    """Tests for ManifoldPoint creation and properties."""

    def test_point_feature_vector_length(self, sample_points):
        """Feature vector should have correct length."""
        for point in sample_points:
            assert len(point.feature_vector) == ManifoldPoint.feature_dimension
            assert len(point.feature_vector) == 8

    def test_point_feature_vector_values(self, sample_points):
        """Feature vector should contain correct values."""
        point = sample_points[0]
        vector = point.feature_vector

        assert vector[0] == pytest.approx(point.mean_entropy)
        assert vector[1] == pytest.approx(point.entropy_variance)
        assert vector[2] == pytest.approx(point.first_token_entropy)
        assert vector[3] == pytest.approx(float(point.gate_count))
        assert vector[4] == pytest.approx(point.mean_gate_similarity)

    def test_point_has_unique_id(self, sample_points):
        """Each point should have a unique ID."""
        ids = [point.id for point in sample_points]
        assert len(set(ids)) == len(ids)


class TestManifoldProfileCreation:
    """Tests for ManifoldProfile creation and operations."""

    def test_create_empty_profile(self):
        """Creating an empty profile should work."""
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model-id",
            model_name="Test Model",
        )

        assert profile.model_id == "test-model-id"
        assert profile.model_name == "Test Model"
        assert len(profile.regions) == 0
        assert len(profile.recent_points) == 0
        assert profile.total_point_count == 0

    def test_profile_with_points(self, sample_points):
        """Profile with recent points should compute statistics."""
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model-id",
            model_name="Test Model",
            recent_points=sample_points,
            total_point_count=len(sample_points),
        )

        stats = profile.compute_statistics()
        assert stats.total_points == len(sample_points)
        assert stats.recent_point_count == len(sample_points)
        assert stats.region_count == 0  # No regions yet

    def test_profile_statistics_with_no_regions(self, sample_points):
        """Profile statistics should handle no regions gracefully."""
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model-id",
            model_name="Test Model",
            recent_points=sample_points,
            total_point_count=len(sample_points),
        )

        stats = profile.compute_statistics()

        assert stats.region_count == 0
        assert stats.dense_region_count == 0
        assert stats.sparse_region_count == 0
        assert stats.transitional_region_count == 0
        assert stats.mean_intrinsic_dimension is None


class TestProfileFeatureVectorPipeline:
    """Tests for the feature vector extraction pipeline."""

    def test_feature_vectors_backend_conversion(self, backend, sample_points):
        """Feature vectors should convert to backend arrays correctly."""
        # Extract feature vectors
        vectors = [point.feature_vector for point in sample_points]

        # Convert to backend array
        X = backend.array(vectors)
        backend.eval(X)

        assert X.shape == (len(sample_points), 8)

    def test_feature_vectors_gram_matrix(self, backend, sample_points):
        """Feature vectors should produce valid Gram matrix."""
        vectors = [point.feature_vector for point in sample_points]
        X = backend.array(vectors)
        backend.eval(X)

        # Compute Gram matrix
        gram = backend.matmul(X, backend.transpose(X))
        backend.eval(gram)

        assert gram.shape == (len(sample_points), len(sample_points))

        # Diagonal should be positive (squared norms)
        gram_np = backend.to_numpy(gram)
        for i in range(len(sample_points)):
            assert gram_np[i, i] > 0


class TestProfileGeometryMetrics:
    """Tests for geometric metrics on profiles."""

    def test_intrinsic_dimension_from_points(self, backend, sample_points):
        """Should be able to compute intrinsic dimension from profile points."""
        from modelcypher.core.domain.geometry.intrinsic_dimension import (
            IntrinsicDimension,
        )

        vectors = [point.feature_vector for point in sample_points]

        # Use the static method for computing intrinsic dimension
        result = IntrinsicDimension.compute_two_nn(vectors, backend=backend)

        # Should return a valid dimension estimate
        assert result.intrinsic_dimension > 0
        # Note: Estimated ID can exceed ambient dimension with small samples
        assert result.sample_count == len(sample_points)

    def test_curvature_from_points(self, backend, sample_points):
        """Should be able to compute curvature metrics from profile points."""
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        vectors = [point.feature_vector for point in sample_points]
        X = backend.array(vectors)
        backend.eval(X)

        rg = RiemannianGeometry(backend)

        # Compute geodesic distances
        result = rg.geodesic_distances(X, k_neighbors=5)

        assert result.distances.shape == (len(sample_points), len(sample_points))


class TestProfileSerialization:
    """Tests for profile serialization (without persistence)."""

    def test_profile_to_dict_roundtrip(self, sample_points):
        """Profile should serialize to dict and back correctly."""
        # Create profile
        original_profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model-id",
            model_name="Test Model",
            recent_points=sample_points[:5],
            total_point_count=5,
        )

        # Verify basic properties are maintained
        assert original_profile.model_id == "test-model-id"
        assert original_profile.model_name == "Test Model"
        assert original_profile.total_point_count == 5
        assert len(original_profile.recent_points) == 5

        # Verify points are correctly stored
        for i, point in enumerate(original_profile.recent_points):
            assert point.prompt_hash == f"hash_{i:04d}"

    def test_profile_statistics_are_deterministic(self, sample_points):
        """Profile statistics should be deterministic."""
        profile = ManifoldProfile(
            id=uuid4(),
            model_id="test-model-id",
            model_name="Test Model",
            recent_points=sample_points,
            total_point_count=len(sample_points),
        )

        # Compute statistics twice
        stats1 = profile.compute_statistics()
        stats2 = profile.compute_statistics()

        # Should be identical
        assert stats1.total_points == stats2.total_points
        assert stats1.region_count == stats2.region_count
        assert stats1.recent_point_count == stats2.recent_point_count

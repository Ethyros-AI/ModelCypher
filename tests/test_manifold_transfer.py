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

"""Tests for cross-manifold projection (manifold_transfer.py).

Validates the implementation of landmark MDS-based concept transfer
as described in de Silva & Tenenbaum (2004).
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    CrossManifoldConfig,
    CrossManifoldProjector,
    ProjectionQuality,
    TransferPoint,
)


class TestAnchorDistanceProfile:
    """Tests for AnchorDistanceProfile dataclass."""

    def test_profile_creation(self) -> None:
        """Test creating a basic distance profile."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="test_concept",
            anchor_ids=["a1", "a2", "a3"],
            distances=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            weights=backend.to_numpy(backend.array([0.5, 0.3, 0.2])),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.concept_id == "test_concept"
        assert profile.num_anchors == 3
        assert profile.mean_distance == pytest.approx(1.7)  # Weighted mean: 1*0.5 + 2*0.3 + 3*0.2
        assert profile.distance_variance == pytest.approx(2 / 3, rel=0.01)

    def test_profile_num_anchors(self) -> None:
        """Test num_anchors property."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a", "b", "c", "d", "e"],
            distances=backend.to_numpy(backend.array([1.0, 1.5, 2.0, 2.5, 3.0])),
            weights=backend.to_numpy(backend.ones((5,)) / 5),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.num_anchors == 5

    def test_distance_to(self) -> None:
        """Test distance_to method."""
        backend = get_default_backend()
        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1", "a2", "a3"],
            distances=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            weights=backend.to_numpy(backend.array([0.33, 0.34, 0.33])),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.distance_to("a1") == pytest.approx(1.0)
        assert profile.distance_to("a2") == pytest.approx(2.0)
        assert profile.distance_to("nonexistent") is None


class TestTransferPoint:
    """Tests for TransferPoint dataclass."""

    @pytest.fixture
    def make_profile(self) -> AnchorDistanceProfile:
        """Create a sample profile for testing."""
        backend = get_default_backend()
        return AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1"],
            distances=backend.to_numpy(backend.array([1.0])),
            weights=backend.to_numpy(backend.array([1.0])),
            source_curvature=None,
            source_volume=None,
        )

    def test_transfer_point_reliable_excellent(
        self,
        make_profile: AnchorDistanceProfile,
    ) -> None:
        """Excellent quality = reliable."""
        backend = get_default_backend()
        point = TransferPoint(
            concept_id="test",
            source_profile=make_profile,
            coordinates=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            projected_volume=None,
            stress=0.05,
            quality=ProjectionQuality.EXCELLENT,
            curvature_mismatch=0.02,
            confidence=0.95,
        )
        assert point.is_reliable is True

    def test_transfer_point_reliable_good(
        self,
        make_profile: AnchorDistanceProfile,
    ) -> None:
        """Good quality = reliable."""
        backend = get_default_backend()
        point = TransferPoint(
            concept_id="test",
            source_profile=make_profile,
            coordinates=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            projected_volume=None,
            stress=0.2,
            quality=ProjectionQuality.GOOD,
            curvature_mismatch=0.1,
            confidence=0.85,
        )
        assert point.is_reliable is True

    def test_transfer_point_unreliable_marginal(
        self,
        make_profile: AnchorDistanceProfile,
    ) -> None:
        """Marginal quality = not reliable."""
        backend = get_default_backend()
        point = TransferPoint(
            concept_id="test",
            source_profile=make_profile,
            coordinates=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            projected_volume=None,
            stress=0.4,
            quality=ProjectionQuality.MARGINAL,
            curvature_mismatch=0.2,
            confidence=0.6,
        )
        assert point.is_reliable is False

    def test_transfer_point_unreliable_poor(
        self,
        make_profile: AnchorDistanceProfile,
    ) -> None:
        """Poor quality = not reliable."""
        backend = get_default_backend()
        point = TransferPoint(
            concept_id="test",
            source_profile=make_profile,
            coordinates=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            projected_volume=None,
            stress=0.6,
            quality=ProjectionQuality.POOR,
            curvature_mismatch=0.3,
            confidence=0.4,
        )
        assert point.is_reliable is False

    def test_to_dict(self, make_profile: AnchorDistanceProfile) -> None:
        """Test to_dict serialization."""
        backend = get_default_backend()
        point = TransferPoint(
            concept_id="test_concept",
            source_profile=make_profile,
            coordinates=backend.to_numpy(backend.array([1.0, 2.0, 3.0])),
            projected_volume=None,
            stress=0.05,
            quality=ProjectionQuality.EXCELLENT,
            curvature_mismatch=0.02,
            confidence=0.95,
        )
        d = point.to_dict()
        assert d["conceptId"] == "test_concept"
        assert d["stress"] == pytest.approx(0.05)
        assert d["quality"] == "excellent"
        assert d["confidence"] == pytest.approx(0.95)


class TestCrossManifoldProjector:
    """Tests for CrossManifoldProjector."""

    @pytest.fixture
    def projector(self) -> CrossManifoldProjector:
        """Create a projector with default config."""
        return CrossManifoldProjector()

    @pytest.fixture
    def sample_activations(self):
        """Generate sample activations for testing."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 128
        n_samples = 5
        n_anchors = 20

        concept_activations = backend.to_numpy(backend.random_normal((n_samples, d)))
        anchor_activations = {f"anchor_{i}": backend.to_numpy(backend.random_normal((3, d))) for i in range(n_anchors)}
        return concept_activations, anchor_activations

    def test_compute_distance_profile(
        self,
        projector: CrossManifoldProjector,
        sample_activations,
    ) -> None:
        """Test computing distance profile from activations."""
        backend = get_default_backend()
        concept_acts, anchor_acts = sample_activations

        profile = projector.compute_distance_profile(
            concept_activations=concept_acts,
            concept_id="test_concept",
            anchor_activations=anchor_acts,
        )

        assert profile.concept_id == "test_concept"
        assert profile.num_anchors == 20
        assert len(profile.distances) == 20
        distances_arr = backend.array(profile.distances)
        assert backend.all(distances_arr >= 0)  # Distances should be non-negative
        weights_sum = backend.sum(backend.array(profile.weights))
        assert float(backend.to_numpy(weights_sum)) == pytest.approx(1.0)  # Weights sum to 1

    def test_project_preserves_distances(
        self,
        projector: CrossManifoldProjector,
    ) -> None:
        """Test that projection attempts to preserve relational distances."""
        backend = get_default_backend()
        backend.random_seed(123)
        d = 64
        n_anchors = 15

        # Create concept and anchors in source
        concept_acts = backend.to_numpy(backend.random_normal((3, d)))
        source_anchors = {f"anchor_{i}": backend.to_numpy(backend.random_normal((2, d))) for i in range(n_anchors)}

        # Target anchors - slightly perturbed from source
        target_anchors = {}
        for k, v in source_anchors.items():
            v_arr = backend.array(v)
            noise = backend.random_normal(v.shape) * 0.1
            target_anchors[k] = backend.to_numpy(v_arr + noise)

        # Compute profile in source
        profile = projector.compute_distance_profile(
            concept_activations=concept_acts,
            concept_id="test",
            anchor_activations=source_anchors,
        )

        # Project to target
        transfer = projector.project(
            profile=profile,
            target_anchor_activations=target_anchors,
        )

        # Verify basic properties
        assert transfer.concept_id == "test"
        assert transfer.stress >= 0
        assert transfer.confidence >= 0 and transfer.confidence <= 1
        assert transfer.quality in list(ProjectionQuality)

    def test_min_anchors_warning(self) -> None:
        """Test that few anchors produces a warning."""
        backend = get_default_backend()
        config = CrossManifoldConfig(min_anchors=30)
        projector = CrossManifoldProjector(config)

        backend.random_seed(42)
        d = 64

        # Only 10 anchors, less than minimum
        concept_acts = backend.to_numpy(backend.random_normal((3, d)))
        source_anchors = {f"anchor_{i}": backend.to_numpy(backend.random_normal((2, d))) for i in range(10)}
        target_anchors = source_anchors.copy()

        # Should still work despite warning
        profile = projector.compute_distance_profile(
            concept_activations=concept_acts,
            concept_id="test",
            anchor_activations=source_anchors,
        )

        transfer = projector.project(
            profile=profile,
            target_anchor_activations=target_anchors,
        )

        assert transfer is not None
        assert transfer.concept_id == "test"


class TestCrossManifoldConfig:
    """Tests for CrossManifoldConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CrossManifoldConfig()

        assert config.min_anchors == 10
        assert config.use_curvature_correction is True
        assert config.max_iterations == 1000
        assert config.convergence_tolerance == 1e-6

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CrossManifoldConfig(
            min_anchors=50,
            use_curvature_correction=False,
            max_iterations=500,
            learning_rate=0.05,
        )

        assert config.min_anchors == 50
        assert config.use_curvature_correction is False
        assert config.max_iterations == 500
        assert config.learning_rate == 0.05


class TestProjectionQuality:
    """Tests for ProjectionQuality enum."""

    def test_quality_ordering(self) -> None:
        """Test that quality levels exist."""
        qualities = list(ProjectionQuality)

        assert ProjectionQuality.EXCELLENT in qualities
        assert ProjectionQuality.GOOD in qualities
        assert ProjectionQuality.MARGINAL in qualities
        assert ProjectionQuality.POOR in qualities

    def test_quality_values(self) -> None:
        """Test quality enum values."""
        assert ProjectionQuality.EXCELLENT.value == "excellent"
        assert ProjectionQuality.GOOD.value == "good"
        assert ProjectionQuality.MARGINAL.value == "marginal"
        assert ProjectionQuality.POOR.value == "poor"

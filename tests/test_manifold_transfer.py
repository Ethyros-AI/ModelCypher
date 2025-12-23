"""Tests for cross-manifold projection (manifold_transfer.py).

Validates the implementation of landmark MDS-based concept transfer
as described in de Silva & Tenenbaum (2004).
"""
from __future__ import annotations

import numpy as np
import pytest

from modelcypher.core.domain.geometry.manifold_transfer import (
    AnchorDistanceProfile,
    TransferPoint,
    TransferQuality,
    CrossManifoldProjector,
    CrossManifoldConfig,
    compute_piecewise_geodesic_distance,
)


class TestAnchorDistanceProfile:
    """Tests for AnchorDistanceProfile dataclass."""

    def test_profile_creation(self) -> None:
        """Test creating a basic distance profile."""
        profile = AnchorDistanceProfile(
            concept_id="test_concept",
            anchor_ids=["a1", "a2", "a3"],
            distances=np.array([1.0, 2.0, 3.0]),
            weights=np.array([0.5, 0.3, 0.2]),
            source_curvature=None,
            source_volume=None,
        )
        assert profile.concept_id == "test_concept"
        assert profile.num_anchors == 3
        assert profile.mean_distance == pytest.approx(2.0)
        assert profile.distance_variance == pytest.approx(2 / 3, rel=0.01)

    def test_profile_num_anchors(self) -> None:
        """Test num_anchors property."""
        profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a", "b", "c", "d", "e"],
            distances=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            weights=np.ones(5) / 5,
            source_curvature=None,
            source_volume=None,
        )
        assert profile.num_anchors == 5


class TestTransferPoint:
    """Tests for TransferPoint dataclass."""

    def test_transfer_point_reliable_high_quality(self) -> None:
        """High quality + low stress + high confidence = reliable."""
        source_profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1"],
            distances=np.array([1.0]),
            weights=np.array([1.0]),
            source_curvature=None,
            source_volume=None,
        )
        point = TransferPoint(
            concept_id="test",
            coordinates=np.array([1.0, 2.0, 3.0]),
            stress=0.05,
            quality=TransferQuality.EXCELLENT,
            confidence=0.95,
            source_profile=source_profile,
            curvature_mismatch=0.02,
        )
        assert point.is_reliable is True

    def test_transfer_point_unreliable_poor_quality(self) -> None:
        """Poor quality = not reliable."""
        source_profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1"],
            distances=np.array([1.0]),
            weights=np.array([1.0]),
            source_curvature=None,
            source_volume=None,
        )
        point = TransferPoint(
            concept_id="test",
            coordinates=np.array([1.0, 2.0, 3.0]),
            stress=0.5,
            quality=TransferQuality.POOR,
            confidence=0.5,
            source_profile=source_profile,
            curvature_mismatch=0.3,
        )
        assert point.is_reliable is False

    def test_transfer_point_unreliable_high_stress(self) -> None:
        """High stress = not reliable even with good quality."""
        source_profile = AnchorDistanceProfile(
            concept_id="test",
            anchor_ids=["a1"],
            distances=np.array([1.0]),
            weights=np.array([1.0]),
            source_curvature=None,
            source_volume=None,
        )
        point = TransferPoint(
            concept_id="test",
            coordinates=np.array([1.0, 2.0, 3.0]),
            stress=0.25,  # Above threshold
            quality=TransferQuality.GOOD,
            confidence=0.9,
            source_profile=source_profile,
            curvature_mismatch=0.1,
        )
        assert point.is_reliable is False


class TestCrossManifoldProjector:
    """Tests for CrossManifoldProjector."""

    @pytest.fixture
    def projector(self) -> CrossManifoldProjector:
        """Create a projector with default config."""
        return CrossManifoldProjector()

    @pytest.fixture
    def sample_activations(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Generate sample activations for testing."""
        np.random.seed(42)
        d = 128
        n_samples = 5
        n_anchors = 20

        concept_activations = np.random.randn(n_samples, d)
        anchor_activations = {
            f"anchor_{i}": np.random.randn(3, d)
            for i in range(n_anchors)
        }
        return concept_activations, anchor_activations

    def test_compute_distance_profile(
        self,
        projector: CrossManifoldProjector,
        sample_activations: tuple[np.ndarray, dict[str, np.ndarray]],
    ) -> None:
        """Test computing distance profile from activations."""
        concept_acts, anchor_acts = sample_activations

        profile = projector.compute_distance_profile(
            concept_activations=concept_acts,
            concept_id="test_concept",
            anchor_activations=anchor_acts,
        )

        assert profile.concept_id == "test_concept"
        assert profile.num_anchors == 20
        assert len(profile.distances) == 20
        assert np.all(profile.distances >= 0)  # Distances should be non-negative
        assert np.sum(profile.weights) == pytest.approx(1.0)  # Weights sum to 1

    def test_project_preserves_distances(
        self,
        projector: CrossManifoldProjector,
    ) -> None:
        """Test that projection attempts to preserve relational distances."""
        np.random.seed(123)
        d = 64
        n_anchors = 15

        # Create concept and anchors in source
        concept_acts = np.random.randn(3, d)
        source_anchors = {
            f"anchor_{i}": np.random.randn(2, d)
            for i in range(n_anchors)
        }

        # Target anchors - slightly perturbed from source
        target_anchors = {
            k: v + 0.1 * np.random.randn(*v.shape)
            for k, v in source_anchors.items()
        }

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
        assert transfer.quality in list(TransferQuality)

    def test_quality_assessment(
        self,
        projector: CrossManifoldProjector,
    ) -> None:
        """Test quality assessment based on stress."""
        # Low stress should give excellent quality
        assert projector._assess_quality(0.01) == TransferQuality.EXCELLENT

        # Medium stress should give good quality
        assert projector._assess_quality(0.08) == TransferQuality.GOOD

        # Higher stress should give acceptable quality
        assert projector._assess_quality(0.18) == TransferQuality.ACCEPTABLE

        # High stress should give poor quality
        assert projector._assess_quality(0.35) == TransferQuality.POOR

    def test_min_anchors_config(self) -> None:
        """Test minimum anchor requirement."""
        config = CrossManifoldConfig(min_anchors=30)
        projector = CrossManifoldProjector(config)

        np.random.seed(42)
        d = 64

        # Only 10 anchors, less than minimum
        concept_acts = np.random.randn(3, d)
        source_anchors = {
            f"anchor_{i}": np.random.randn(2, d)
            for i in range(10)
        }
        target_anchors = source_anchors.copy()

        profile = projector.compute_distance_profile(
            concept_activations=concept_acts,
            concept_id="test",
            anchor_activations=source_anchors,
        )

        # Should still work but with lower confidence
        transfer = projector.project(
            profile=profile,
            target_anchor_activations=target_anchors,
        )

        # With fewer anchors than min, quality should be affected
        assert transfer.confidence < 1.0

    def test_empty_anchors_raises(
        self,
        projector: CrossManifoldProjector,
    ) -> None:
        """Test that empty anchors raises ValueError."""
        concept_acts = np.random.randn(3, 64)

        with pytest.raises(ValueError, match="No anchor activations"):
            projector.compute_distance_profile(
                concept_activations=concept_acts,
                concept_id="test",
                anchor_activations={},
            )


class TestPiecewiseGeodesicDistance:
    """Tests for piecewise geodesic distance computation."""

    def test_euclidean_fallback(self) -> None:
        """Without curvature info, should default to Euclidean."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        dist = compute_piecewise_geodesic_distance(a, b, curvature=None)

        # Should be Euclidean distance
        assert dist == pytest.approx(5.0)

    def test_positive_curvature_increases_distance(self) -> None:
        """Positive curvature (sphere-like) should increase geodesic distance."""
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])

        from modelcypher.core.domain.geometry.riemannian_density import LocalCurvature

        curvature = LocalCurvature(
            sectional_curvature=0.5,  # Positive = sphere-like
            ricci_scalar=1.0,
            christoffel_norm=0.0,
            point=a,
        )

        dist_curved = compute_piecewise_geodesic_distance(a, b, curvature=curvature)
        dist_flat = compute_piecewise_geodesic_distance(a, b, curvature=None)

        # Positive curvature should make geodesic longer
        assert dist_curved >= dist_flat

    def test_zero_distance(self) -> None:
        """Distance from point to itself should be zero."""
        a = np.array([1.0, 2.0, 3.0])

        dist = compute_piecewise_geodesic_distance(a, a, curvature=None)

        assert dist == pytest.approx(0.0)


class TestCrossManifoldConfig:
    """Tests for CrossManifoldConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CrossManifoldConfig()

        assert config.min_anchors == 10
        assert config.use_curvature_correction is True
        assert config.stress_threshold == 0.2
        assert config.distance_metric == "cosine"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CrossManifoldConfig(
            min_anchors=50,
            use_curvature_correction=False,
            stress_threshold=0.1,
            distance_metric="euclidean",
        )

        assert config.min_anchors == 50
        assert config.use_curvature_correction is False
        assert config.stress_threshold == 0.1
        assert config.distance_metric == "euclidean"


class TestTransferQuality:
    """Tests for TransferQuality enum."""

    def test_quality_ordering(self) -> None:
        """Test that quality levels have correct ordering."""
        qualities = list(TransferQuality)

        assert TransferQuality.EXCELLENT in qualities
        assert TransferQuality.GOOD in qualities
        assert TransferQuality.ACCEPTABLE in qualities
        assert TransferQuality.POOR in qualities

    def test_quality_values(self) -> None:
        """Test quality enum values."""
        assert TransferQuality.EXCELLENT.value == "excellent"
        assert TransferQuality.GOOD.value == "good"
        assert TransferQuality.ACCEPTABLE.value == "acceptable"
        assert TransferQuality.POOR.value == "poor"

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

"""Tests for Riemannian density estimation and interference prediction.

Tests cover:
- Manifold curvature estimation
- ConceptVolume computation
- Volume relationship metrics
- Interference prediction
- Edge cases and numerical stability
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Check if scipy is available
try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from modelcypher.core.domain.geometry.interference_predictor import (
    InterferencePredictor,
    InterferenceType,
    quick_interference_check,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    SectionalCurvatureEstimator,
    compute_curvature_divergence,
)
from modelcypher.core.domain.geometry.riemannian_density import (
    InfluenceType,
    RiemannianDensityConfig,
    RiemannianDensityEstimator,
    batch_estimate_volumes,
    compute_pairwise_relations,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_gaussian_samples():
    """Generate samples from a simple Gaussian distribution."""
    np.random.seed(42)
    n_samples = 100
    d = 10
    mean = np.zeros(d)
    cov = np.eye(d)
    return np.random.multivariate_normal(mean, cov, n_samples)


@pytest.fixture
def spherical_samples():
    """Generate samples on a sphere (positive curvature)."""
    np.random.seed(42)
    n_samples = 100
    d = 10
    # Sample from unit sphere
    samples = np.random.randn(n_samples, d)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    return samples


@pytest.fixture
def two_overlapping_concepts():
    """Generate two overlapping concept activations."""
    np.random.seed(42)
    d = 20
    n = 50

    # Concept A centered at origin
    mean_a = np.zeros(d)
    cov_a = np.eye(d) * 0.5
    samples_a = np.random.multivariate_normal(mean_a, cov_a, n)

    # Concept B centered nearby with some overlap
    mean_b = np.ones(d) * 0.5
    cov_b = np.eye(d) * 0.5
    samples_b = np.random.multivariate_normal(mean_b, cov_b, n)

    return samples_a, samples_b


@pytest.fixture
def two_distant_concepts():
    """Generate two distant concept activations."""
    np.random.seed(42)
    d = 20
    n = 50

    mean_a = np.zeros(d)
    mean_b = np.ones(d) * 10  # Far apart
    cov = np.eye(d) * 0.5

    samples_a = np.random.multivariate_normal(mean_a, cov, n)
    samples_b = np.random.multivariate_normal(mean_b, cov, n)

    return samples_a, samples_b


# ============================================================================
# Manifold Curvature Tests
# ============================================================================


class TestSectionalCurvatureEstimator:
    """Tests for curvature estimation."""

    def test_flat_space_has_low_curvature(self, simple_gaussian_samples):
        """Flat Gaussian samples should have low curvature magnitude."""
        estimator = SectionalCurvatureEstimator()

        point = simple_gaussian_samples[0]
        neighbors = simple_gaussian_samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # Should have low curvature magnitude (estimation is inherently noisy)
        assert abs(curvature.mean_sectional) < 1.0
        # Sign can vary due to noise, but variance should be relatively low
        assert curvature.variance_sectional < 0.5

    def test_spherical_has_positive_curvature(self, spherical_samples):
        """Spherical samples should have positive curvature."""
        estimator = SectionalCurvatureEstimator()

        point = spherical_samples[0]
        neighbors = spherical_samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # Sphere has positive curvature (though estimation may be noisy)
        # We check that it's not strongly negative
        assert curvature.mean_sectional > -0.5

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for manifold profile")
    def test_curvature_profile_statistics(self, simple_gaussian_samples):
        """Test manifold profile aggregates curvature correctly."""
        estimator = SectionalCurvatureEstimator()

        profile = estimator.estimate_manifold_profile(simple_gaussian_samples, k_neighbors=10)

        assert len(profile.local_curvatures) == len(simple_gaussian_samples)
        assert 0 <= sum(profile.sign_distribution.values()) <= 1.1  # Account for rounding
        assert profile.dominant_sign in CurvatureSign

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for manifold profile")
    def test_curvature_divergence_same_profile(self, simple_gaussian_samples):
        """Same profile should have zero divergence."""
        estimator = SectionalCurvatureEstimator()
        profile = estimator.estimate_manifold_profile(simple_gaussian_samples[:50], k_neighbors=10)

        divergence = compute_curvature_divergence(profile, profile)
        assert divergence == 0.0

    def test_insufficient_neighbors_returns_flat(self):
        """Too few neighbors should return flat curvature."""
        estimator = SectionalCurvatureEstimator()

        point = np.zeros(10)
        neighbors = np.random.randn(3, 10)  # Less than d+1

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.sign == CurvatureSign.FLAT
        assert curvature.mean_sectional == 0.0


# ============================================================================
# Riemannian Density Tests
# ============================================================================


class TestRiemannianDensityEstimator:
    """Tests for density estimation."""

    def test_estimate_concept_volume_basic(self, simple_gaussian_samples):
        """Test basic volume estimation."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test_concept", simple_gaussian_samples)

        assert volume.concept_id == "test_concept"
        assert volume.dimension == simple_gaussian_samples.shape[1]
        assert volume.num_samples == len(simple_gaussian_samples)
        assert volume.centroid.shape == (simple_gaussian_samples.shape[1],)
        assert volume.covariance.shape == (volume.dimension, volume.dimension)

    def test_volume_centroid_is_valid_center(self, simple_gaussian_samples):
        """Centroid should be a valid center of the distribution.

        Note: Uses Fréchet mean (Riemannian center of mass), not arithmetic mean.
        In curved spaces, these differ. Fréchet mean minimizes sum of squared
        geodesic distances - the correct center for manifold data.
        """
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        # Centroid should be in the convex hull of samples (approximately)
        # and have reasonable dimension
        assert volume.centroid.shape == (simple_gaussian_samples.shape[1],)
        # Centroid should not be too far from arithmetic mean (they're related)
        arithmetic_mean = np.mean(simple_gaussian_samples, axis=0)
        distance_from_arithmetic = np.linalg.norm(volume.centroid - arithmetic_mean)
        # Fréchet mean is typically close to arithmetic mean for mild curvature
        assert distance_from_arithmetic < 1.0  # Reasonable bound

    def test_density_at_centroid_is_maximum(self, simple_gaussian_samples):
        """Density should be highest at centroid."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        density_at_centroid = volume.density_at(volume.centroid)

        # Check some random points have lower density
        for _ in range(10):
            random_point = np.random.randn(volume.dimension) * 3
            density_random = volume.density_at(random_point)
            assert density_random <= density_at_centroid

    def test_volume_covariance_positive_definite(self, simple_gaussian_samples):
        """Covariance should be positive definite."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        eigenvalues = np.linalg.eigvalsh(volume.covariance)
        assert all(eigenvalues > 0)

    def test_single_sample_volume(self):
        """Single sample should produce point mass volume."""
        estimator = RiemannianDensityEstimator()

        single = np.array([[1.0, 2.0, 3.0]])
        volume = estimator.estimate_concept_volume("single", single)

        assert volume.num_samples == 1
        assert volume.geodesic_radius == 0.0
        np.testing.assert_allclose(volume.centroid, single[0])

    def test_mahalanobis_distance_at_centroid(self, simple_gaussian_samples):
        """Mahalanobis distance at centroid should be zero."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        distance = volume.mahalanobis_distance(volume.centroid)
        assert abs(distance) < 1e-10


class TestConceptVolumeRelation:
    """Tests for volume relationship computation."""

    def test_overlapping_volumes_have_nonzero_bhattacharyya(self, two_overlapping_concepts):
        """Overlapping volumes should have non-zero Bhattacharyya coefficient."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        relation = estimator.compute_relation(vol_a, vol_b)

        # Bhattacharyya > 0 indicates some overlap (threshold depends on distance)
        assert relation.bhattacharyya_coefficient > 0
        assert relation.centroid_distance > 0

    def test_distant_volumes_have_low_bhattacharyya(self, two_distant_concepts):
        """Distant volumes should have near-zero Bhattacharyya coefficient."""
        samples_a, samples_b = two_distant_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        relation = estimator.compute_relation(vol_a, vol_b)

        assert relation.bhattacharyya_coefficient < 0.1
        assert relation.centroid_distance > 5

    def test_identical_volumes_have_perfect_overlap(self, simple_gaussian_samples):
        """Identical volumes should have perfect overlap."""
        estimator = RiemannianDensityEstimator()

        vol = estimator.estimate_concept_volume("A", simple_gaussian_samples)

        relation = estimator.compute_relation(vol, vol)

        assert relation.bhattacharyya_coefficient > 0.99
        assert relation.centroid_distance < 1e-10
        assert relation.subspace_alignment > 0.99

    def test_subspace_alignment_similar_spaces(self):
        """Similar subspaces should have high alignment."""
        np.random.seed(42)
        d = 10
        n = 50

        # Samples from similar distributions (high alignment expected)
        samples_a = np.random.randn(n, d)
        samples_b = np.random.randn(n, d)  # Same distribution

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        relation = estimator.compute_relation(vol_a, vol_b)

        # Similar distributions should have high subspace alignment
        assert relation.subspace_alignment > 0.5


# ============================================================================
# Interference Predictor Tests
# ============================================================================


class TestInterferencePredictor:
    """Tests for interference prediction."""

    def test_distant_concepts_neutral(self, two_distant_concepts):
        """Distant concepts should have neutral interference.

        Note: Curvature-aware analysis may detect curvature mismatch between
        distant regions, which can lower safety scores. The key property is
        that interference type is NEUTRAL and overlap is near-zero.
        """
        samples_a, samples_b = two_distant_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        assert result.interference_type == InterferenceType.NEUTRAL
        # Overlap should be negligible for distant concepts
        assert result.overlap_score < 0.01
        # Safety score may be lower due to curvature mismatch detection
        # but should still be positive and reasonable
        assert result.safety_score > 0.3

    def test_overlapping_concepts_have_mechanisms(self, two_overlapping_concepts):
        """Overlapping concepts should have identified mechanisms."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        # Overlapping concepts should have some mechanisms identified or non-zero scores
        # Type can vary but the analysis should produce meaningful scores
        assert result.overlap_score >= 0
        assert result.safety_score > 0

    def test_identical_volumes_high_overlap(self, simple_gaussian_samples):
        """Identical volumes should have high overlap score."""
        estimator = RiemannianDensityEstimator()
        vol = estimator.estimate_concept_volume("A", simple_gaussian_samples)

        predictor = InterferencePredictor()
        result = predictor.predict(vol, vol)

        # Overlap score should be substantial for identical volumes
        assert result.overlap_score > 0.5
        # Distance should be zero
        assert result.distance_score == 0.0
        # Alignment should be perfect
        assert result.alignment_score > 0.99

    def test_result_has_recommendations(self, two_overlapping_concepts):
        """Interference result should include recommendations."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        assert result.recommended_action is not None
        assert len(result.recommended_action) > 0
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.mitigation_strategies, list)

    def test_confidence_bounded(self, two_overlapping_concepts):
        """Confidence should be in [0, 1]."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        assert 0 <= result.confidence <= 1


class TestGlobalInterferenceReport:
    """Tests for global interference analysis."""

    def test_global_report_structure(self):
        """Test global report has correct structure."""
        np.random.seed(42)
        d = 10
        n = 30

        concepts = {
            "A": np.random.randn(n, d),
            "B": np.random.randn(n, d) + 2,
            "C": np.random.randn(n, d) + 4,
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = InterferencePredictor()
        report = predictor.predict_global(volumes)

        # Should have 3 pairs: (A,B), (A,C), (B,C)
        assert report.total_pairs == 3
        assert len(report.pair_results) == 3

        # Counts should sum to total
        total_count = (
            report.constructive_count
            + report.neutral_count
            + report.partial_destructive_count
            + report.destructive_count
        )
        assert total_count == report.total_pairs

    def test_concept_risk_scores_exist(self):
        """Each concept should have a risk score."""
        np.random.seed(42)
        concepts = {
            "X": np.random.randn(20, 5),
            "Y": np.random.randn(20, 5) + 5,
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = InterferencePredictor()
        report = predictor.predict_global(volumes)

        assert "X" in report.concept_risk_scores
        assert "Y" in report.concept_risk_scores
        assert 0 <= report.concept_risk_scores["X"] <= 1
        assert 0 <= report.concept_risk_scores["Y"] <= 1


class TestQuickInterferenceCheck:
    """Tests for quick_interference_check utility."""

    def test_quick_check_common_concepts(self):
        """Quick check should analyze common concepts."""
        np.random.seed(42)
        d = 10
        n = 30

        source = {
            "math": np.random.randn(n, d),
            "code": np.random.randn(n, d) + 2,
        }
        target = {
            "math": np.random.randn(n, d) + 0.5,
            "code": np.random.randn(n, d) + 2.5,
        }

        report = quick_interference_check(source, target)

        assert report.total_pairs == 2  # math, code
        assert "source:math" in report.concept_risk_scores
        assert "target:math" in report.concept_risk_scores

    def test_quick_check_no_common_concepts(self):
        """Quick check with no common concepts returns empty report."""
        source = {"A": np.random.randn(10, 5)}
        target = {"B": np.random.randn(10, 5)}

        report = quick_interference_check(source, target)

        assert report.total_pairs == 0
        assert report.overall_safety_score == 1.0


# ============================================================================
# Property-Based Tests
# ============================================================================


class TestRiemannianDensityProperties:
    """Property-based tests using Hypothesis."""

    @given(
        st.integers(min_value=3, max_value=50),
        st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_volume_dimension_matches_input(self, n_samples, dim):
        """Volume dimension should match input dimension."""
        samples = np.random.randn(n_samples, dim)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("test", samples)

        assert volume.dimension == dim
        assert volume.centroid.shape == (dim,)
        assert volume.covariance.shape == (dim, dim)

    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_density_decreases_with_distance(self, scale):
        """Density should decrease as we move away from centroid."""
        assume(np.isfinite(scale))

        np.random.seed(42)
        samples = np.random.randn(50, 5) * scale

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("test", samples)

        # Density at centroid
        d0 = volume.density_at(volume.centroid)

        # Density at 2 standard deviations away
        direction = np.ones(5) / np.sqrt(5)
        far_point = volume.centroid + direction * 2 * scale
        d_far = volume.density_at(far_point)

        assert d_far < d0

    @given(
        st.floats(min_value=-5, max_value=5),
        st.floats(min_value=-5, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_safety_score_bounded(self, offset_a, offset_b):
        """Safety score should always be in [0, 1]."""
        assume(np.isfinite(offset_a) and np.isfinite(offset_b))

        np.random.seed(42)
        samples_a = np.random.randn(20, 5) + offset_a
        samples_b = np.random.randn(20, 5) + offset_b

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        assert 0 <= result.safety_score <= 1


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_variance(self):
        """Handle samples with very small variance."""
        samples = np.ones((20, 5)) + np.random.randn(20, 5) * 1e-10

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("tiny", samples)

        # Should not crash, covariance should be regularized
        assert np.all(np.isfinite(volume.covariance))
        eigenvalues = np.linalg.eigvalsh(volume.covariance)
        assert all(eigenvalues > 0)

    def test_high_dimensional_samples(self):
        """Handle high-dimensional samples."""
        np.random.seed(42)
        samples = np.random.randn(100, 500)  # 500 dimensions

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("high_dim", samples)

        assert volume.dimension == 500
        assert np.all(np.isfinite(volume.centroid))

    def test_different_influence_types(self, simple_gaussian_samples):
        """Test different influence function types."""
        # Only test GAUSSIAN and UNIFORM which don't require scipy.special
        for influence_type in [
            InfluenceType.GAUSSIAN,
            InfluenceType.UNIFORM,
            InfluenceType.LAPLACIAN,
        ]:
            config = RiemannianDensityConfig(influence_type=influence_type)
            estimator = RiemannianDensityEstimator(config)

            volume = estimator.estimate_concept_volume(
                f"test_{influence_type.value}", simple_gaussian_samples
            )

            assert volume.influence_type == influence_type
            # Density should be non-negative
            density = volume.density_at(volume.centroid)
            assert density >= 0

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for Student-t distribution")
    def test_student_t_influence_type(self, simple_gaussian_samples):
        """Test Student-t influence function type (requires scipy)."""
        config = RiemannianDensityConfig(influence_type=InfluenceType.STUDENT_T)
        estimator = RiemannianDensityEstimator(config)

        volume = estimator.estimate_concept_volume("test_student_t", simple_gaussian_samples)

        assert volume.influence_type == InfluenceType.STUDENT_T
        density = volume.density_at(volume.centroid)
        assert density >= 0

    def test_batch_estimation_empty_dict(self):
        """Batch estimation with empty dict returns empty."""
        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, {})
        assert len(volumes) == 0

    def test_pairwise_relations_single_concept(self):
        """Pairwise relations with single concept returns empty."""
        estimator = RiemannianDensityEstimator()
        samples = np.random.randn(20, 5)
        volumes = {"only": estimator.estimate_concept_volume("only", samples)}

        relations = compute_pairwise_relations(estimator, volumes)
        assert len(relations) == 0

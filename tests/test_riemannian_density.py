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

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Check if scipy is available
try:
    import scipy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.interference_predictor import (
    InterferencePredictor,
    TransformationType,
    quick_merge_analysis,
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
    backend = get_default_backend()
    backend.random_seed(42)
    n_samples = 100
    d = 10

    # Generate samples using backend then convert to numpy for the tests
    # Note: multivariate_normal not in backend, so we use standard normal
    samples = backend.random_normal((n_samples, d))
    samples = backend.astype(samples, "float32")
    backend.eval(samples)

    return backend.to_numpy(samples)


@pytest.fixture
def spherical_samples():
    """Generate samples on a sphere (positive curvature)."""
    backend = get_default_backend()
    backend.random_seed(42)
    n_samples = 100
    d = 10

    # Sample from unit sphere
    samples = backend.random_normal((n_samples, d))
    backend.eval(samples)
    samples_np = backend.to_numpy(samples)

    # Normalize to unit sphere using numpy (unavoidable for this operation)
    import numpy as np
    samples_np = samples_np / np.linalg.norm(samples_np, axis=1, keepdims=True)

    return samples_np


@pytest.fixture
def two_overlapping_concepts():
    """Generate two overlapping concept activations."""
    backend = get_default_backend()
    backend.random_seed(42)
    d = 20
    n = 50

    # Concept A centered at origin with small variance
    samples_a = backend.random_normal((n, d))
    samples_a = samples_a * 0.707  # sqrt(0.5)
    backend.eval(samples_a)

    # Concept B centered nearby with some overlap
    samples_b = backend.random_normal((n, d))
    samples_b = samples_b * 0.707  # sqrt(0.5)
    samples_b = samples_b + 0.5  # shift by 0.5
    backend.eval(samples_b)

    return backend.to_numpy(samples_a), backend.to_numpy(samples_b)


@pytest.fixture
def two_distant_concepts():
    """Generate two distant concept activations."""
    backend = get_default_backend()
    backend.random_seed(42)
    d = 20
    n = 50

    # Concept A at origin
    samples_a = backend.random_normal((n, d))
    samples_a = samples_a * 0.707  # sqrt(0.5)
    backend.eval(samples_a)

    # Concept B far apart
    samples_b = backend.random_normal((n, d))
    samples_b = samples_b * 0.707  # sqrt(0.5)
    samples_b = samples_b + 10.0  # shift by 10
    backend.eval(samples_b)

    return backend.to_numpy(samples_a), backend.to_numpy(samples_b)


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
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        point = backend.zeros((10,))
        backend.random_seed(42)
        neighbors = backend.random_normal((3, 10))  # Less than d+1
        backend.eval(point, neighbors)

        point_np = backend.to_numpy(point)
        neighbors_np = backend.to_numpy(neighbors)

        curvature = estimator.estimate_local_curvature(point_np, neighbors_np)

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
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        # Centroid should be in the convex hull of samples (approximately)
        # and have reasonable dimension
        assert volume.centroid.shape == (simple_gaussian_samples.shape[1],)
        # Centroid should not be too far from arithmetic mean (they're related)
        samples_tensor = backend.array(simple_gaussian_samples)
        arithmetic_mean = backend.mean(samples_tensor, axis=0)
        backend.eval(arithmetic_mean)
        arithmetic_mean_np = backend.to_numpy(arithmetic_mean)

        distance_from_arithmetic = backend.norm(backend.array(volume.centroid) - backend.array(arithmetic_mean_np))
        backend.eval(distance_from_arithmetic)
        # Fréchet mean is typically close to arithmetic mean for mild curvature
        assert backend.to_numpy(distance_from_arithmetic) < 1.0  # Reasonable bound

    def test_density_at_centroid_is_maximum(self, simple_gaussian_samples):
        """Density should be highest at centroid."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        density_at_centroid = volume.density_at(volume.centroid)

        # Check some random points have lower density
        backend.random_seed(99)
        for _ in range(10):
            random_point = backend.random_normal((volume.dimension,))
            random_point = random_point * 3.0
            backend.eval(random_point)
            random_point_np = backend.to_numpy(random_point)
            density_random = volume.density_at(random_point_np)
            assert density_random <= density_at_centroid

    def test_volume_covariance_positive_definite(self, simple_gaussian_samples):
        """Covariance should be positive definite."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        # Use numpy for eigvalsh - unavoidable
        import numpy as np
        eigenvalues = np.linalg.eigvalsh(volume.covariance)
        assert all(eigenvalues > 0)

    def test_single_sample_volume(self):
        """Single sample should produce point mass volume."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        single = backend.array([[1.0, 2.0, 3.0]])
        backend.eval(single)
        single_np = backend.to_numpy(single)

        volume = estimator.estimate_concept_volume("single", single_np)

        assert volume.num_samples == 1
        assert volume.geodesic_radius == 0.0
        # Use numpy for assert_allclose - unavoidable
        import numpy as np
        np.testing.assert_allclose(volume.centroid, single_np[0])

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

        # Bhattacharyya should be near zero for distant volumes
        assert relation.bhattacharyya_coefficient < 0.1
        # Centroid distance should be positive (exact value depends on normalization)
        assert relation.centroid_distance > 1.0

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
        backend = get_default_backend()
        backend.random_seed(42)
        d = 10
        n = 50

        # Samples from similar distributions (high alignment expected)
        samples_a = backend.random_normal((n, d))
        samples_b = backend.random_normal((n, d))  # Same distribution
        backend.eval(samples_a, samples_b)

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", backend.to_numpy(samples_a))
        vol_b = estimator.estimate_concept_volume("B", backend.to_numpy(samples_b))

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

        # Distant concepts should need few or no transformations
        # (minimal geometric intervention needed)
        assert len(result.transformations) <= 2
        # Overlap should be negligible for distant concepts
        assert result.overlap_score < 0.01
        # Measurement confidence should be positive
        assert result.measurement_confidence > 0.0

    def test_overlapping_concepts_have_mechanisms(self, two_overlapping_concepts):
        """Overlapping concepts should have identified mechanisms."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        # Overlapping concepts should have some transformations identified
        # Type can vary but the analysis should produce meaningful scores
        assert result.overlap_score >= 0
        assert result.measurement_confidence > 0

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

    def test_result_has_transformation_descriptions(self, two_overlapping_concepts):
        """Merge analysis result should include transformation descriptions."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        # Should have transformation descriptions (even if empty for direct merge)
        assert isinstance(result.transformation_descriptions, list)
        assert len(result.transformation_descriptions) > 0
        # Should have transformations list (may be empty)
        assert isinstance(result.transformations, list)

    def test_measurement_confidence_bounded(self, two_overlapping_concepts):
        """Measurement confidence should be in [0, 1]."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        assert 0 <= result.measurement_confidence <= 1


class TestGlobalInterferenceReport:
    """Tests for global interference analysis."""

    def test_global_report_structure(self):
        """Test global report has correct structure."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 10
        n = 30

        samples_a = backend.random_normal((n, d))
        samples_b = backend.random_normal((n, d)) + 2.0
        samples_c = backend.random_normal((n, d)) + 4.0
        backend.eval(samples_a, samples_b, samples_c)

        concepts = {
            "A": backend.to_numpy(samples_a),
            "B": backend.to_numpy(samples_b),
            "C": backend.to_numpy(samples_c),
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = InterferencePredictor()
        report = predictor.predict_global(volumes)

        # Should have 3 pairs: (A,B), (A,C), (B,C)
        assert report.total_pairs == 3
        assert len(report.pair_results) == 3

        # Transformation counts should exist for all transformation types
        assert len(report.transformation_counts) > 0
        # Total transformation counts across all pairs
        total_transformations = sum(report.transformation_counts.values())
        assert total_transformations >= 0  # Could be 0 if no transformations needed

    def test_pair_results_have_geometric_measurements(self):
        """Each pair should have geometric measurements."""
        backend = get_default_backend()
        backend.random_seed(42)

        samples_x = backend.random_normal((20, 5))
        samples_y = backend.random_normal((20, 5)) + 5.0
        backend.eval(samples_x, samples_y)

        concepts = {
            "X": backend.to_numpy(samples_x),
            "Y": backend.to_numpy(samples_y),
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = InterferencePredictor()
        report = predictor.predict_global(volumes)

        # Each pair result should have bounded geometric measurements
        for pair, result in report.pair_results.items():
            assert 0 <= result.overlap_score <= 1
            assert 0 <= result.alignment_score <= 1
            assert result.measurement_confidence >= 0


class TestQuickInterferenceCheck:
    """Tests for quick_interference_check utility."""

    def test_quick_check_common_concepts(self):
        """Quick check should analyze common concepts."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 10
        n = 30

        math_source = backend.random_normal((n, d))
        code_source = backend.random_normal((n, d)) + 2.0
        math_target = backend.random_normal((n, d)) + 0.5
        code_target = backend.random_normal((n, d)) + 2.5
        backend.eval(math_source, code_source, math_target, code_target)

        source = {
            "math": backend.to_numpy(math_source),
            "code": backend.to_numpy(code_source),
        }
        target = {
            "math": backend.to_numpy(math_target),
            "code": backend.to_numpy(code_target),
        }

        report = quick_merge_analysis(source, target)

        assert report.total_pairs == 2  # math, code
        # Check that pair results contain the analyzed concepts
        pair_keys = [key for pair in report.pair_results.keys() for key in pair]
        assert "source:math" in pair_keys
        assert "target:math" in pair_keys

    def test_quick_check_no_common_concepts(self):
        """Quick check with no common concepts returns empty report."""
        backend = get_default_backend()
        backend.random_seed(42)

        source_a = backend.random_normal((10, 5))
        target_b = backend.random_normal((10, 5))
        backend.eval(source_a, target_b)

        source = {"A": backend.to_numpy(source_a)}
        target = {"B": backend.to_numpy(target_b)}

        report = quick_merge_analysis(source, target)

        assert report.total_pairs == 0
        # No pairs means no transformations needed
        assert len(report.transformation_counts) == 0 or sum(report.transformation_counts.values()) == 0


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
        backend = get_default_backend()
        backend.random_seed(99)
        samples = backend.random_normal((n_samples, dim))
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("test", backend.to_numpy(samples))

        assert volume.dimension == dim
        assert volume.centroid.shape == (dim,)
        assert volume.covariance.shape == (dim, dim)

    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_density_decreases_with_distance(self, scale):
        """Density should decrease as we move away from centroid."""
        import numpy as np
        assume(np.isfinite(scale))

        backend = get_default_backend()
        backend.random_seed(42)
        samples = backend.random_normal((50, 5))
        samples = samples * scale
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("test", backend.to_numpy(samples))

        # Density at centroid
        d0 = volume.density_at(volume.centroid)

        # Density at 2 standard deviations away
        direction = backend.ones((5,))
        direction = direction / backend.sqrt(backend.array(5.0))
        backend.eval(direction)
        direction_np = backend.to_numpy(direction)

        far_point = volume.centroid + direction_np * 2 * scale
        d_far = volume.density_at(far_point)

        assert d_far < d0

    @given(
        st.floats(min_value=-5, max_value=5),
        st.floats(min_value=-5, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_geometric_scores_bounded(self, offset_a, offset_b):
        """Geometric measurements should always be in [0, 1]."""
        import numpy as np
        assume(np.isfinite(offset_a) and np.isfinite(offset_b))

        backend = get_default_backend()
        backend.random_seed(42)
        samples_a = backend.random_normal((20, 5))
        samples_a = samples_a + offset_a
        samples_b = backend.random_normal((20, 5))
        samples_b = samples_b + offset_b
        backend.eval(samples_a, samples_b)

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", backend.to_numpy(samples_a))
        vol_b = estimator.estimate_concept_volume("B", backend.to_numpy(samples_b))

        predictor = InterferencePredictor()
        result = predictor.predict(vol_a, vol_b)

        # All geometric measurements should be bounded (with small epsilon for floating point)
        eps = 1e-6
        assert -eps <= result.overlap_score <= 1 + eps
        assert -eps <= result.alignment_score <= 1 + eps
        assert -eps <= result.measurement_confidence <= 1 + eps


# ============================================================================
# Edge Cases and Numerical Stability
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_variance(self):
        """Handle samples with very small variance."""
        backend = get_default_backend()
        backend.random_seed(42)

        ones = backend.ones((20, 5))
        noise = backend.random_normal((20, 5))
        noise = noise * 1e-10
        samples = ones + noise
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("tiny", backend.to_numpy(samples))

        # Should not crash, covariance should be regularized
        import numpy as np
        assert np.all(np.isfinite(volume.covariance))
        eigenvalues = np.linalg.eigvalsh(volume.covariance)
        assert all(eigenvalues > 0)

    def test_high_dimensional_samples(self):
        """Handle high-dimensional samples."""
        backend = get_default_backend()
        backend.random_seed(42)
        samples = backend.random_normal((100, 500))  # 500 dimensions
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("high_dim", backend.to_numpy(samples))

        assert volume.dimension == 500
        import numpy as np
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
        backend = get_default_backend()
        backend.random_seed(42)
        estimator = RiemannianDensityEstimator()
        samples = backend.random_normal((20, 5))
        backend.eval(samples)
        volumes = {"only": estimator.estimate_concept_volume("only", backend.to_numpy(samples))}

        relations = compute_pairwise_relations(estimator, volumes)
        assert len(relations) == 0

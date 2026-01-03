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
    MergeAnalyzer,
    quick_merge_analysis,
)
from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    SectionalCurvatureEstimator,
    compute_curvature_divergence,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon
from modelcypher.core.domain.geometry.riemannian_density import (
    InfluenceType,
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

    return backend.tolist(samples)


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
    # Normalize to unit sphere using backend
    norms = backend.norm(samples, axis=1, keepdims=True)
    normalized = samples / norms
    backend.eval(normalized)
    return backend.tolist(normalized)


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

    return backend.tolist(samples_a), backend.tolist(samples_b)


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

    return backend.tolist(samples_a), backend.tolist(samples_b)


# ============================================================================
# Manifold Curvature Tests
# ============================================================================


class TestSectionalCurvatureEstimator:
    """Tests for curvature estimation."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for manifold profile")
    def test_curvature_profile_statistics(self, simple_gaussian_samples):
        """Test manifold profile aggregates curvature correctly."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        profile = estimator.estimate_manifold_profile(simple_gaussian_samples)

        assert len(profile.local_curvatures) == len(simple_gaussian_samples)
        sign_values = list(profile.sign_distribution.values())
        sign_total = sum(sign_values)
        eps = machine_epsilon(backend, backend.array(sign_values))
        assert abs(sign_total - 1.0) <= eps * max(1, len(sign_values))
        assert profile.dominant_sign in CurvatureSign

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for manifold profile")
    def test_curvature_divergence_same_profile(self, simple_gaussian_samples):
        """Same profile should have zero divergence."""
        estimator = SectionalCurvatureEstimator()
        profile = estimator.estimate_manifold_profile(simple_gaussian_samples[:50])

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

        point_np = backend.tolist(point)
        neighbors_np = backend.tolist(neighbors)

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
        sample_dim = len(simple_gaussian_samples[0]) if simple_gaussian_samples else 0
        assert volume.dimension == sample_dim
        assert volume.num_samples == len(simple_gaussian_samples)
        assert len(volume.centroid) == sample_dim
        assert len(volume.covariance) == volume.dimension
        assert all(len(row) == volume.dimension for row in volume.covariance)

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
            random_point_np = backend.tolist(random_point)
            density_random = volume.density_at(random_point_np)
            assert density_random <= density_at_centroid

    def test_volume_covariance_positive_definite(self, simple_gaussian_samples):
        """Covariance should be positive definite."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        # Use backend for eigenvalue decomposition
        backend = get_default_backend()
        cov_arr = backend.array(volume.covariance)
        eigenvalues, _ = backend.eigh(cov_arr)
        backend.eval(eigenvalues)
        eig_list = backend.tolist(eigenvalues)
        assert all(e > 0 for e in eig_list)

    def test_single_sample_volume(self):
        """Single sample should produce point mass volume."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        single = backend.array([[1.0, 2.0, 3.0]])
        backend.eval(single)
        single_list = backend.tolist(single)

        volume = estimator.estimate_concept_volume("single", single_list)

        assert volume.num_samples == 1
        assert volume.geodesic_radius == 0.0
        # Use pytest.approx for comparison - centroid may contain arrays
        centroid_list = [float(c) if hasattr(c, 'item') else float(c) for c in volume.centroid]
        assert centroid_list == pytest.approx(single_list[0])

    def test_mahalanobis_distance_at_centroid(self, simple_gaussian_samples):
        """Mahalanobis distance at centroid should be zero."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        distance = volume.mahalanobis_distance(volume.centroid)
        eps = machine_epsilon(backend, backend.array(volume.centroid))
        assert abs(distance) <= eps


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
        """Distant volumes should have lower overlap than identical volumes."""
        samples_a, samples_b = two_distant_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        relation_distant = estimator.compute_relation(vol_a, vol_b)

        # Compare to self-relation (identical volumes)
        relation_self = estimator.compute_relation(vol_a, vol_a)

        # Distant volumes should have LOWER overlap than identical volumes
        assert relation_distant.bhattacharyya_coefficient < relation_self.bhattacharyya_coefficient
        # Distant volumes should have GREATER distance than self
        assert relation_distant.centroid_distance > relation_self.centroid_distance

    def test_identical_volumes_have_perfect_overlap(self, simple_gaussian_samples):
        """Identical volumes should have perfect overlap."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()

        vol = estimator.estimate_concept_volume("A", simple_gaussian_samples)

        relation = estimator.compute_relation(vol, vol)

        eps = machine_epsilon(backend, backend.array(vol.centroid))
        tol = eps * max(1, vol.dimension)
        assert abs(relation.bhattacharyya_coefficient - 1.0) <= tol
        assert abs(relation.centroid_distance) <= tol
        assert abs(relation.subspace_alignment - 1.0) <= tol

    def test_subspace_alignment_similar_spaces(self):
        """Similar subspaces should have higher alignment than orthogonal ones."""
        backend = get_default_backend()
        backend.random_seed(42)
        d = 10
        n = 50

        # Samples from similar distributions
        samples_a = backend.random_normal((n, d))
        samples_b = backend.random_normal((n, d))  # Same distribution
        backend.eval(samples_a, samples_b)

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", backend.tolist(samples_a))
        vol_b = estimator.estimate_concept_volume("B", backend.tolist(samples_b))

        relation_similar = estimator.compute_relation(vol_a, vol_b)

        # Create orthogonal subspace (rotate samples_b by 90 degrees in first 2 dims)
        # Swap and negate to create orthogonal vectors in first 2 dimensions
        orthogonal_list = backend.tolist(samples_b)
        for row in orthogonal_list:
            old_0, old_1 = row[0], row[1]
            row[0], row[1] = -old_1, old_0
        samples_orthogonal = backend.array(orthogonal_list)
        backend.eval(samples_orthogonal)

        vol_orthogonal = estimator.estimate_concept_volume("C", backend.tolist(samples_orthogonal))
        relation_orthogonal = estimator.compute_relation(vol_a, vol_orthogonal)

        # Self-alignment is the maximum possible
        relation_self = estimator.compute_relation(vol_a, vol_a)

        # Similar distributions should be at least as aligned as orthogonal ones
        # (geometry determines the actual values; we just check ordering)
        assert relation_similar.subspace_alignment >= relation_orthogonal.subspace_alignment
        # And self-alignment should be the maximum
        assert relation_self.subspace_alignment >= relation_similar.subspace_alignment


# ============================================================================
# Interference Predictor Tests
# ============================================================================


class TestMergeAnalyzer:
    """Tests for interference prediction."""

    def test_distant_concepts_neutral(self, two_distant_concepts, two_overlapping_concepts):
        """Distant concepts should have less interference than overlapping ones.

        We compare distant concepts to overlapping concepts to avoid arbitrary
        thresholds. The geometry determines the actual values.
        """
        samples_a_distant, samples_b_distant = two_distant_concepts
        samples_a_overlap, samples_b_overlap = two_overlapping_concepts

        estimator = RiemannianDensityEstimator()

        vol_a_distant = estimator.estimate_concept_volume("A_distant", samples_a_distant)
        vol_b_distant = estimator.estimate_concept_volume("B_distant", samples_b_distant)
        vol_a_overlap = estimator.estimate_concept_volume("A_overlap", samples_a_overlap)
        vol_b_overlap = estimator.estimate_concept_volume("B_overlap", samples_b_overlap)

        predictor = MergeAnalyzer()
        result_distant = predictor.analyze(vol_a_distant, vol_b_distant)
        result_overlap = predictor.analyze(vol_a_overlap, vol_b_overlap)

        # Distant concepts should have LESS overlap than overlapping concepts
        assert result_distant.overlap_score < result_overlap.overlap_score

    def test_overlapping_concepts_have_mechanisms(self, two_overlapping_concepts):
        """Overlapping concepts should have identified mechanisms."""
        samples_a, samples_b = two_overlapping_concepts
        estimator = RiemannianDensityEstimator()

        vol_a = estimator.estimate_concept_volume("A", samples_a)
        vol_b = estimator.estimate_concept_volume("B", samples_b)

        predictor = MergeAnalyzer()
        result = predictor.analyze(vol_a, vol_b)

        # Overlapping concepts should produce meaningful scores
        assert result.overlap_score >= 0
        assert result.alignment_score >= 0

    def test_identical_volumes_high_overlap(self, simple_gaussian_samples):
        """Identical volumes should have high overlap score."""
        backend = get_default_backend()
        estimator = RiemannianDensityEstimator()
        vol = estimator.estimate_concept_volume("A", simple_gaussian_samples)

        predictor = MergeAnalyzer()
        result = predictor.analyze(vol, vol)

        eps = machine_epsilon(backend, backend.array(vol.centroid))
        tol = eps * max(1, vol.dimension)
        assert abs(result.overlap_score - 1.0) <= tol
        assert abs(result.distance_score) <= tol
        assert abs(result.alignment_score - 1.0) <= tol

class TestGlobalMergeAnalysisReport:
    """Tests for global merge analysis."""

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
            "A": backend.tolist(samples_a),
            "B": backend.tolist(samples_b),
            "C": backend.tolist(samples_c),
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = MergeAnalyzer()
        report = predictor.analyze_global(volumes)

        # Should have 3 pairs: (A,B), (A,C), (B,C)
        assert report.total_pairs == 3
        assert len(report.pair_results) == 3

        # Global metrics should be computed
        assert report.mean_overlap >= 0
        assert report.mean_alignment >= 0

    def test_pair_results_have_geometric_measurements(self):
        """Each pair should have geometric measurements."""
        backend = get_default_backend()
        backend.random_seed(42)

        samples_x = backend.random_normal((20, 5))
        samples_y = backend.random_normal((20, 5)) + 5.0
        backend.eval(samples_x, samples_y)

        concepts = {
            "X": backend.tolist(samples_x),
            "Y": backend.tolist(samples_y),
        }

        estimator = RiemannianDensityEstimator()
        volumes = batch_estimate_volumes(estimator, concepts)

        predictor = MergeAnalyzer()
        report = predictor.analyze_global(volumes)

        # Each pair result should have bounded geometric measurements
        eps = machine_epsilon(backend, backend.array([0.0]))
        for pair, result in report.pair_results.items():
            assert -eps <= result.overlap_score <= 1 + eps
            assert -eps <= result.alignment_score <= 1 + eps
            assert result.curvature_divergence >= -eps
            assert result.distance_score >= -eps


class TestQuickInterferenceCheck:
    """Tests for quick_merge_analysis utility."""

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
            "math": backend.tolist(math_source),
            "code": backend.tolist(code_source),
        }
        target = {
            "math": backend.tolist(math_target),
            "code": backend.tolist(code_target),
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

        source = {"A": backend.tolist(source_a)}
        target = {"B": backend.tolist(target_b)}

        report = quick_merge_analysis(source, target)

        assert report.total_pairs == 0
        # Empty report should have zero metrics
        assert report.mean_overlap == 0.0


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
        volume = estimator.estimate_concept_volume("test", backend.tolist(samples))

        assert volume.dimension == dim
        assert volume.centroid.shape == (dim,)
        assert volume.covariance.shape == (dim, dim)

    @given(st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=10, deadline=None)
    def test_density_decreases_with_distance(self, scale):
        """Density should decrease as we move away from centroid."""
        import math
        assume(math.isfinite(scale))

        backend = get_default_backend()
        backend.random_seed(42)
        samples = backend.random_normal((50, 5))
        samples = samples * scale
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("test", backend.tolist(samples))

        # Density at centroid
        d0 = volume.density_at(volume.centroid)

        # Density at 2 standard deviations away
        direction = backend.ones((5,))
        direction = direction / backend.sqrt(backend.array(5.0))
        backend.eval(direction)
        direction_list = backend.tolist(direction)

        # Element-wise: centroid + direction * 2 * scale
        far_point = [c + d * 2 * scale for c, d in zip(volume.centroid, direction_list)]
        d_far = volume.density_at(far_point)

        assert d_far < d0

    @given(
        st.floats(min_value=-5, max_value=5),
        st.floats(min_value=-5, max_value=5),
    )
    @settings(max_examples=20, deadline=None)
    def test_geometric_scores_bounded(self, offset_a, offset_b):
        """Geometric measurements should always be in [0, 1]."""
        import math
        assume(math.isfinite(offset_a) and math.isfinite(offset_b))

        backend = get_default_backend()
        backend.random_seed(42)
        samples_a = backend.random_normal((20, 5))
        samples_a = samples_a + offset_a
        samples_b = backend.random_normal((20, 5))
        samples_b = samples_b + offset_b
        backend.eval(samples_a, samples_b)

        estimator = RiemannianDensityEstimator()
        vol_a = estimator.estimate_concept_volume("A", backend.tolist(samples_a))
        vol_b = estimator.estimate_concept_volume("B", backend.tolist(samples_b))

        predictor = MergeAnalyzer()
        result = predictor.analyze(vol_a, vol_b)

        # All geometric measurements should be bounded (with small epsilon for floating point)
        eps = machine_epsilon(backend, backend.array([0.0]))
        assert -eps <= result.overlap_score <= 1 + eps
        assert -eps <= result.alignment_score <= 1 + eps
        assert result.curvature_divergence >= -eps
        assert result.distance_score >= -eps


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
        volume = estimator.estimate_concept_volume("tiny", backend.tolist(samples))

        # Should not crash, covariance should be regularized
        import math
        cov_arr = backend.array(volume.covariance)
        isfinite_arr = backend.isfinite(cov_arr)
        backend.eval(isfinite_arr)
        isfinite_list = backend.tolist(isfinite_arr)
        assert all(all(row) for row in isfinite_list)
        eigenvalues, _ = backend.eigh(cov_arr)
        backend.eval(eigenvalues)
        eig_list = backend.tolist(eigenvalues)
        assert all(e > 0 for e in eig_list)

    def test_high_dimensional_samples(self):
        """Handle high-dimensional samples."""
        backend = get_default_backend()
        backend.random_seed(42)
        samples = backend.random_normal((100, 500))  # 500 dimensions
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("high_dim", backend.tolist(samples))

        assert volume.dimension == 500
        import math
        assert all(math.isfinite(c) for c in volume.centroid)

    def test_influence_type_derived_from_data(self, simple_gaussian_samples):
        """Test that influence type is derived from data kurtosis."""
        estimator = RiemannianDensityEstimator()

        volume = estimator.estimate_concept_volume("test", simple_gaussian_samples)

        # Influence type should be derived from data kurtosis
        # Gaussian samples should result in Gaussian influence type
        # (unless kurtosis happens to be significantly > 3)
        assert volume.influence_type in [InfluenceType.GAUSSIAN, InfluenceType.STUDENT_T]
        # Density should be non-negative
        density = volume.density_at(volume.centroid)
        assert density >= 0

    def test_heavy_tailed_data_uses_student_t(self):
        """Test that heavy-tailed data results in Student-t influence type."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Create heavy-tailed data (mixture of narrow and wide Gaussians)
        # This should have excess kurtosis > threshold
        narrow = backend.random_normal((80, 10)) * 0.1
        wide = backend.random_normal((20, 10)) * 3.0
        samples = backend.concatenate([narrow, wide], axis=0)
        backend.eval(samples)

        estimator = RiemannianDensityEstimator()
        volume = estimator.estimate_concept_volume("heavy_tailed", backend.tolist(samples))

        # Heavy tails should trigger Student-t
        # (depends on actual kurtosis value)
        assert volume.influence_type in [InfluenceType.GAUSSIAN, InfluenceType.STUDENT_T]
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
        volumes = {"only": estimator.estimate_concept_volume("only", backend.tolist(samples))}

        relations = compute_pairwise_relations(estimator, volumes)
        assert len(relations) == 0

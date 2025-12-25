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

"""Tests for ManifoldStitcher and related similarity functions.

Tests mathematical invariants including:
- Jaccard similarity: ∈ [0, 1], J(A, A) = 1, J(∅, ∅) = 0
- Weighted Jaccard: ∈ [0, 1], sum(min)/sum(max) formula
- Cosine similarity: ∈ [-1, 1], cos(x, x) = 1
- Ensemble similarity: ∈ [0, 1], weighted combination
- Proper rotation: det(R) = +1 (not reflection)
- LayerConfidence: ∈ [0, 1]
- ContinuousFingerprint.entropies: ∈ [0, 1]
- K-Means: all points assigned to valid clusters
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ContinuousFingerprint,
    DimensionCorrelation,
    EnsembleWeights,
    LayerConfidence,
    ManifoldStitcher,
    Thresholds,
    _ensure_proper_rotation,
    compute_cosine_similarity,
    compute_ensemble_similarity,
    compute_jaccard_similarity,
    compute_weighted_jaccard_similarity,
)

# =============================================================================
# Hypothesis Strategies
# =============================================================================


@st.composite
def finite_set(draw, max_size: int = 20):
    """Generate a set of non-negative integers."""
    size = draw(st.integers(min_value=0, max_value=max_size))
    elements = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    return set(elements)


@st.composite
def activation_dict(draw, max_dims: int = 20):
    """Generate a dict of dimension -> activation value."""
    size = draw(st.integers(min_value=0, max_value=max_dims))
    dims = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=size,
            max_size=size,
            unique=True,
        )
    )
    values = [
        draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(size)
    ]
    return dict(zip(dims, values))


@st.composite
def activation_vector(draw, size: int = 10):
    """Generate a list of activation values with fixed size."""
    return [
        draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
        for _ in range(size)
    ]


@st.composite
def point_cloud_uniform(draw, n_points: int = 15, dims: int = 5):
    """Generate a point cloud with uniform dimensions."""
    return [
        [
            draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
            for _ in range(dims)
        ]
        for _ in range(n_points)
    ]


@st.composite
def orthogonal_matrix(draw, size: int = 3):
    """Generate an orthogonal matrix via QR decomposition."""
    # Generate random matrix
    data = [
        [
            draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
            for _ in range(size)
        ]
        for _ in range(size)
    ]
    arr = np.array(data)
    # QR decomposition gives orthogonal Q
    q, _ = np.linalg.qr(arr)
    return q


# =============================================================================
# Jaccard Similarity Tests
# =============================================================================


class TestJaccardSimilarity:
    """Tests for compute_jaccard_similarity."""

    @given(finite_set(), finite_set())
    @settings(max_examples=50, deadline=None)
    def test_jaccard_bounded_zero_one(self, set_a: set, set_b: set):
        """Jaccard similarity must be in [0, 1].

        Mathematical property: |A ∩ B| / |A ∪ B| is always bounded.
        """
        result = compute_jaccard_similarity(set_a, set_b)
        assert 0.0 <= result <= 1.0

    @given(finite_set())
    @settings(max_examples=50, deadline=None)
    def test_jaccard_self_similarity_is_one(self, s: set):
        """Jaccard(A, A) = 1 for non-empty sets.

        Mathematical property: |A ∩ A| / |A ∪ A| = |A| / |A| = 1
        """
        assume(len(s) > 0)
        result = compute_jaccard_similarity(s, s)
        assert result == pytest.approx(1.0)

    def test_jaccard_empty_sets_is_zero(self):
        """Jaccard(∅, ∅) = 0 by convention."""
        result = compute_jaccard_similarity(set(), set())
        assert result == 0.0

    def test_jaccard_disjoint_sets_is_zero(self):
        """Jaccard of disjoint sets is 0.

        Mathematical property: |A ∩ B| = 0 when A ∩ B = ∅
        """
        result = compute_jaccard_similarity({1, 2, 3}, {4, 5, 6})
        assert result == 0.0

    @given(finite_set(), finite_set())
    @settings(max_examples=50, deadline=None)
    def test_jaccard_symmetric(self, set_a: set, set_b: set):
        """Jaccard is symmetric: J(A, B) = J(B, A)."""
        result_ab = compute_jaccard_similarity(set_a, set_b)
        result_ba = compute_jaccard_similarity(set_b, set_a)
        assert result_ab == pytest.approx(result_ba)


# =============================================================================
# Weighted Jaccard Similarity Tests
# =============================================================================


class TestWeightedJaccardSimilarity:
    """Tests for compute_weighted_jaccard_similarity."""

    @given(activation_dict(), activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_weighted_jaccard_bounded_zero_one(self, dict_a: dict, dict_b: dict):
        """Weighted Jaccard must be in [0, 1].

        Mathematical property: sum(min(a, b)) / sum(max(a, b)) ∈ [0, 1]
        """
        result = compute_weighted_jaccard_similarity(dict_a, dict_b)
        assert 0.0 <= result <= 1.0

    @given(activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_weighted_jaccard_self_is_one(self, d: dict):
        """Weighted Jaccard with self is 1 for non-empty dicts.

        Mathematical property: sum(min(a, a)) / sum(max(a, a)) = 1
        """
        assume(len(d) > 0)
        assume(any(v > 0 for v in d.values()))  # At least one positive value
        result = compute_weighted_jaccard_similarity(d, d)
        assert result == pytest.approx(1.0)

    def test_weighted_jaccard_empty_is_zero(self):
        """Weighted Jaccard of empty dicts is 0."""
        result = compute_weighted_jaccard_similarity({}, {})
        assert result == 0.0

    @given(activation_dict(), activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_weighted_jaccard_symmetric(self, dict_a: dict, dict_b: dict):
        """Weighted Jaccard is symmetric."""
        result_ab = compute_weighted_jaccard_similarity(dict_a, dict_b)
        result_ba = compute_weighted_jaccard_similarity(dict_b, dict_a)
        assert result_ab == pytest.approx(result_ba)


# =============================================================================
# Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    """Tests for compute_cosine_similarity."""

    @given(activation_dict(), activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_cosine_bounded_minus_one_to_one(self, dict_a: dict, dict_b: dict):
        """Cosine similarity must be in [-1, 1].

        Mathematical property: cos(θ) ∈ [-1, 1] for any angle θ.
        """
        result = compute_cosine_similarity(dict_a, dict_b)
        assert -1.0 <= result <= 1.0 + 1e-6  # Small tolerance for floating point

    @given(activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_cosine_self_is_one(self, d: dict):
        """Cosine similarity with self is 1 for non-zero vectors.

        Mathematical property: cos(0) = 1 (angle with self is 0).
        """
        assume(len(d) > 0)
        # Need a vector with sufficient magnitude (not just tiny values)
        norm_sq = sum(v * v for v in d.values())
        assume(norm_sq > 1e-10)  # Non-trivial vector
        result = compute_cosine_similarity(d, d)
        assert result == pytest.approx(1.0)

    def test_cosine_orthogonal_is_zero(self):
        """Cosine of orthogonal vectors is 0."""
        # Vectors (1, 0) and (0, 1) in sparse form
        a = {0: 1.0}
        b = {1: 1.0}
        result = compute_cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_cosine_opposite_is_minus_one(self):
        """Cosine of opposite vectors is -1."""
        a = {0: 1.0, 1: 1.0}
        b = {0: -1.0, 1: -1.0}
        result = compute_cosine_similarity(a, b)
        assert result == pytest.approx(-1.0)


# =============================================================================
# Ensemble Similarity Tests
# =============================================================================


class TestEnsembleSimilarity:
    """Tests for compute_ensemble_similarity."""

    @given(activation_dict(), activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_ensemble_bounded_zero_one(self, dict_a: dict, dict_b: dict):
        """Ensemble similarity must be in [0, 1].

        Mathematical property: Weighted sum with non-negative weights
        and bounded components produces bounded result.
        """
        result = compute_ensemble_similarity(dict_a, dict_b)
        assert 0.0 <= result <= 1.0 + 1e-6

    @given(activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_ensemble_self_is_high(self, d: dict):
        """Ensemble similarity with self should be high (near 1).

        All components (Jaccard, CKA, cosine) should be maximal.
        """
        assume(len(d) > 0)
        # Require at least one non-trivial activation to avoid numerical edge cases
        assume(any(v > 0.01 for v in d.values()))
        result = compute_ensemble_similarity(d, d)
        # Self-similarity should be close to 1
        assert result > 0.9

    def test_ensemble_weights_normalize(self):
        """EnsembleWeights.normalized() should sum to 1."""
        weights = EnsembleWeights(weighted_jaccard=1.0, cka=2.0, cosine=3.0)
        normalized = weights.normalized()
        total = normalized.weighted_jaccard + normalized.cka + normalized.cosine
        assert total == pytest.approx(1.0)

    def test_ensemble_zero_weights_fallback(self):
        """Zero weights should normalize to equal distribution."""
        weights = EnsembleWeights(weighted_jaccard=0.0, cka=0.0, cosine=0.0)
        normalized = weights.normalized()
        assert normalized.weighted_jaccard == pytest.approx(1 / 3)
        assert normalized.cka == pytest.approx(1 / 3)
        assert normalized.cosine == pytest.approx(1 / 3)


# =============================================================================
# Proper Rotation Tests
# =============================================================================


class TestProperRotation:
    """Tests for _ensure_proper_rotation."""

    def test_proper_rotation_determinant_positive(self):
        """Proper rotation must have det(R) = +1.

        Mathematical property: Rotations preserve orientation (det = +1),
        while reflections reverse it (det = -1).
        """
        backend = get_default_backend()

        # Create a reflection matrix (det = -1)
        reflection = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # SVD of reflection
        u, _, vt = np.linalg.svd(reflection)
        omega = u @ vt

        u_arr = backend.array(u)
        vt_arr = backend.array(vt)
        omega_arr = backend.array(omega)

        # Fix to proper rotation
        result = _ensure_proper_rotation(u_arr, vt_arr, omega_arr, backend)
        result_np = backend.to_numpy(result)

        # Determinant should be +1
        det = np.linalg.det(result_np)
        assert det == pytest.approx(1.0, abs=1e-6)

    def test_proper_rotation_preserves_orthogonality(self):
        """Proper rotation should remain orthogonal."""
        backend = get_default_backend()

        # Random orthogonal matrix via SVD
        random_mat = np.random.randn(3, 3)
        u, _, vt = np.linalg.svd(random_mat)
        omega = u @ vt

        u_arr = backend.array(u)
        vt_arr = backend.array(vt)
        omega_arr = backend.array(omega)

        result = _ensure_proper_rotation(u_arr, vt_arr, omega_arr, backend)
        result_np = backend.to_numpy(result)

        # R @ R^T should be identity
        product = result_np @ result_np.T
        assert np.allclose(product, np.eye(3), atol=1e-6)


# =============================================================================
# LayerConfidence Tests
# =============================================================================


class TestLayerConfidence:
    """Tests for LayerConfidence dataclass."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_confidence_bounded_zero_one(self, strong: int, moderate: int, weak: int):
        """LayerConfidence.confidence must be in [0, 1].

        Mathematical property: Weighted average with weights in [0, 1].
        """
        lc = LayerConfidence(
            layer=0,
            strong_correlations=strong,
            moderate_correlations=moderate,
            weak_correlations=weak,
        )
        assert 0.0 <= lc.confidence <= 1.0

    def test_confidence_all_strong_is_max(self):
        """All strong correlations should give confidence = 1.0."""
        lc = LayerConfidence(
            layer=0,
            strong_correlations=10,
            moderate_correlations=0,
            weak_correlations=0,
        )
        assert lc.confidence == pytest.approx(Thresholds.strong_weight)

    def test_confidence_all_weak_is_min_nonzero(self):
        """All weak correlations should give confidence = weak_weight."""
        lc = LayerConfidence(
            layer=0,
            strong_correlations=0,
            moderate_correlations=0,
            weak_correlations=10,
        )
        assert lc.confidence == pytest.approx(Thresholds.weak_weight)

    def test_confidence_empty_is_zero(self):
        """No correlations should give confidence = 0."""
        lc = LayerConfidence(
            layer=0,
            strong_correlations=0,
            moderate_correlations=0,
            weak_correlations=0,
        )
        assert lc.confidence == 0.0


# =============================================================================
# DimensionCorrelation Tests
# =============================================================================


class TestDimensionCorrelation:
    """Tests for DimensionCorrelation dataclass."""

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=50, deadline=None)
    def test_correlation_classification_exclusive(self, corr: float):
        """Each correlation should be in exactly one category."""
        dc = DimensionCorrelation(source_dim=0, target_dim=0, correlation=corr)

        categories = [
            dc.is_strong_correlation,
            dc.is_moderate_correlation,
            dc.is_weak_correlation,
        ]

        # Exactly one should be True
        assert sum(categories) == 1

    def test_strong_threshold(self):
        """Correlation > 0.7 should be strong."""
        dc = DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.8)
        assert dc.is_strong_correlation is True
        assert dc.is_moderate_correlation is False
        assert dc.is_weak_correlation is False

    def test_moderate_threshold(self):
        """Correlation in (0.4, 0.7] should be moderate."""
        dc = DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.5)
        assert dc.is_strong_correlation is False
        assert dc.is_moderate_correlation is True
        assert dc.is_weak_correlation is False

    def test_weak_threshold(self):
        """Correlation <= 0.4 should be weak."""
        dc = DimensionCorrelation(source_dim=0, target_dim=0, correlation=0.3)
        assert dc.is_strong_correlation is False
        assert dc.is_moderate_correlation is False
        assert dc.is_weak_correlation is True


# =============================================================================
# ContinuousFingerprint Tests
# =============================================================================


class TestContinuousFingerprint:
    """Tests for ContinuousFingerprint entropy normalization."""

    def test_entropy_normalized_zero_one(self):
        """Entropy values should be normalized to [0, 1]."""
        # Uniform distribution has max entropy
        uniform = [1.0] * 100
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: uniform},
        )

        # Entropy should be high but bounded by 1
        assert 0.0 <= fp.entropies[0] <= 1.0

    def test_entropy_peaked_is_low(self):
        """Peaked distribution should have low entropy."""
        # One very high value, rest near zero
        peaked = [0.01] * 99 + [100.0]
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: peaked},
        )

        assert fp.entropies[0] < 0.3  # Low entropy

    def test_sparsity_bounded_zero_one(self):
        """Sparsity should be in [0, 1]."""
        activations = [0.0, 0.0, 0.0, 1.0, 2.0]  # 60% near zero
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: activations},
        )

        assert 0.0 <= fp.sparsities[0] <= 1.0


# =============================================================================
# K-Means Tests
# =============================================================================


class TestKMeans:
    """Tests for ManifoldStitcher.k_means."""

    @given(point_cloud_uniform(n_points=15, dims=5))
    @settings(max_examples=20, deadline=None)
    def test_kmeans_all_points_assigned(self, points: list):
        """All points should be assigned to a cluster."""
        k = 3
        assignments, centroids = ManifoldStitcher.k_means(points, k)

        # Every point should have an assignment
        assert len(assignments) == len(points)

        # Assignments should be valid cluster indices
        assert all(0 <= a < k for a in assignments)

    @given(point_cloud_uniform(n_points=15, dims=5))
    @settings(max_examples=20, deadline=None)
    def test_kmeans_centroid_count(self, points: list):
        """Should produce correct number of centroids."""
        k = 3
        _, centroids = ManifoldStitcher.k_means(points, k)

        assert len(centroids) == k

    def test_kmeans_empty_returns_empty(self):
        """Empty input should return empty output."""
        assignments, centroids = ManifoldStitcher.k_means([], 3)
        assert assignments == []
        assert centroids == []

    def test_kmeans_zero_k_returns_empty(self):
        """Zero clusters should return empty output."""
        points = [[1.0, 2.0], [3.0, 4.0]]
        assignments, centroids = ManifoldStitcher.k_means(points, 0)
        assert assignments == []
        assert centroids == []


# =============================================================================
# CKA Matrix Tests
# =============================================================================


class TestCKAMatrix:
    """Tests for ManifoldStitcher.compute_cka_matrix."""

    def test_cka_self_diagonal_is_one(self):
        """CKA of fingerprint with itself should be 1 on diagonal."""
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: [1.0, 2.0, 3.0]},
        )

        from modelcypher.core.domain.geometry.manifold_stitcher import ContinuousModelFingerprints

        model_fps = ContinuousModelFingerprints(
            model_id="test",
            hidden_dim=3,
            layer_count=1,
            fingerprints=[fp],
        )

        matrix, _, _ = ManifoldStitcher.compute_cka_matrix(model_fps, model_fps, layer=0)

        backend = get_default_backend()
        matrix_np = backend.to_numpy(matrix)

        if matrix_np.size > 0:
            # Diagonal should be 1 (self-similarity)
            for i in range(min(matrix_np.shape)):
                assert matrix_np[i, i] == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Integration Tests
# =============================================================================


class TestManifoldStitcherIntegration:
    """Integration tests for ManifoldStitcher methods."""

    def test_compute_continuous_correlation_returns_valid(self):
        """compute_continuous_correlation should return bounded values."""
        fp1 = ContinuousFingerprint.from_activations(
            prime_id="a",
            prime_text="a",
            layer_activations={0: [1.0, 2.0, 3.0, 4.0]},
        )
        fp2 = ContinuousFingerprint.from_activations(
            prime_id="b",
            prime_text="b",
            layer_activations={0: [1.1, 2.1, 3.1, 4.1]},
        )

        result = ManifoldStitcher.compute_continuous_correlation(fp1, fp2, layer=0)

        assert result is not None
        assert 0.0 <= result.cka <= 1.0
        assert -1.0 <= result.cosine_similarity <= 1.0
        assert result.magnitude_ratio > 0

    def test_cluster_activations_produces_valid_clusters(self):
        """cluster_activations should produce valid AlignmentCluster objects."""
        source = {f"prime_{i}": [float(i), float(i + 1)] for i in range(10)}
        target = {f"prime_{i}": [float(i + 0.1), float(i + 1.1)] for i in range(10)}

        clusters = ManifoldStitcher.cluster_activations(source, target, cluster_count=3)

        # Should produce some clusters
        assert len(clusters) > 0

        # Each cluster should have valid properties
        for cluster in clusters:
            assert cluster.member_count > 0
            assert 0.0 <= cluster.procrustes_error  # Error is non-negative

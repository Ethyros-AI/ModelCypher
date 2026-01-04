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
- Proper rotation: det(R) = +1 (not reflection)
- LayerConfidence: ∈ [0, 1]
- ContinuousFingerprint.entropies: ∈ [0, 1]
- K-Means: all points assigned to valid clusters
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.intersection_similarity import (
    compute_cosine_similarity,
    compute_jaccard_similarity,
    compute_weighted_jaccard_similarity,
)
from modelcypher.core.domain.geometry.manifold_stitcher import (
    ContinuousFingerprint,
    ManifoldStitcher,
    _ensure_proper_rotation,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon


def _eps(*values: float) -> float:
    backend = get_default_backend()
    return machine_epsilon(backend, backend.array(list(values) or [1.0]))

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
    backend = get_default_backend()
    # Generate random matrix
    data = [
        [
            draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
            for _ in range(size)
        ]
        for _ in range(size)
    ]
    arr = backend.array(data)
    # QR decomposition gives orthogonal Q
    q, _ = backend.qr(arr)
    return backend.tolist(q)


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
        eps = _eps(result)
        assert -eps <= result <= 1.0 + eps

    @given(finite_set())
    @settings(max_examples=50, deadline=None)
    def test_jaccard_self_similarity_is_one(self, s: set):
        """Jaccard(A, A) = 1 for non-empty sets.

        Mathematical property: |A ∩ A| / |A ∪ A| = |A| / |A| = 1
        """
        assume(len(s) > 0)
        result = compute_jaccard_similarity(s, s)
        eps = _eps(result)
        assert abs(result - 1.0) <= eps

    def test_jaccard_empty_sets_is_zero(self):
        """Jaccard(∅, ∅) = 0 by convention."""
        result = compute_jaccard_similarity(set(), set())
        eps = _eps(result)
        assert abs(result - 0.0) <= eps

    def test_jaccard_disjoint_sets_is_zero(self):
        """Jaccard of disjoint sets is 0.

        Mathematical property: |A ∩ B| = 0 when A ∩ B = ∅
        """
        result = compute_jaccard_similarity({1, 2, 3}, {4, 5, 6})
        eps = _eps(result)
        assert abs(result - 0.0) <= eps

    @given(finite_set(), finite_set())
    @settings(max_examples=50, deadline=None)
    def test_jaccard_symmetric(self, set_a: set, set_b: set):
        """Jaccard is symmetric: J(A, B) = J(B, A)."""
        result_ab = compute_jaccard_similarity(set_a, set_b)
        result_ba = compute_jaccard_similarity(set_b, set_a)
        eps = _eps(result_ab, result_ba)
        assert abs(result_ab - result_ba) <= eps


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
        eps = _eps(result)
        assert -eps <= result <= 1.0 + eps

    @given(activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_weighted_jaccard_self_is_one(self, d: dict):
        """Weighted Jaccard with self is 1 for non-empty dicts.

        Mathematical property: sum(min(a, a)) / sum(max(a, a)) = 1
        """
        assume(len(d) > 0)
        assume(any(v > 0 for v in d.values()))  # At least one positive value
        result = compute_weighted_jaccard_similarity(d, d)
        eps = _eps(result)
        assert abs(result - 1.0) <= eps

    def test_weighted_jaccard_empty_is_zero(self):
        """Weighted Jaccard of empty dicts is 0."""
        result = compute_weighted_jaccard_similarity({}, {})
        eps = _eps(result)
        assert abs(result - 0.0) <= eps

    @given(activation_dict(), activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_weighted_jaccard_symmetric(self, dict_a: dict, dict_b: dict):
        """Weighted Jaccard is symmetric."""
        result_ab = compute_weighted_jaccard_similarity(dict_a, dict_b)
        result_ba = compute_weighted_jaccard_similarity(dict_b, dict_a)
        eps = _eps(result_ab, result_ba)
        assert abs(result_ab - result_ba) <= eps


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
        eps = _eps(result)
        assert -1.0 - eps <= result <= 1.0 + eps

    @given(activation_dict())
    @settings(max_examples=50, deadline=None)
    def test_cosine_self_is_one(self, d: dict):
        """Cosine similarity with self is 1 for non-zero vectors.

        Mathematical property: cos(0) = 1 (angle with self is 0).
        """
        assume(len(d) > 0)
        # Need a vector with sufficient magnitude (not just tiny values)
        # Use a higher threshold to avoid numerical precision issues
        norm_sq = sum(v * v for v in d.values())
        eps = _eps(norm_sq)
        assume(norm_sq > eps)  # Non-trivial vector
        result = compute_cosine_similarity(d, d)
        # Use 3 * machine_epsilon for tolerance - cosine involves multiple
        # floating-point operations (dot, norm, division) that accumulate error
        eps = 3 * _eps(result)
        assert abs(result - 1.0) <= eps

    def test_cosine_orthogonal_is_zero(self):
        """Cosine of orthogonal vectors is 0."""
        # Vectors (1, 0) and (0, 1) in sparse form
        a = {0: 1.0}
        b = {1: 1.0}
        result = compute_cosine_similarity(a, b)
        eps = _eps(result)
        assert abs(result - 0.0) <= eps

    def test_cosine_opposite_is_minus_one(self):
        """Cosine of opposite vectors is -1."""
        a = {0: 1.0, 1: 1.0}
        b = {0: -1.0, 1: -1.0}
        result = compute_cosine_similarity(a, b)
        eps = _eps(result)
        assert abs(result - (-1.0)) <= eps



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
        reflection = backend.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # SVD of reflection
        u, _, vt = backend.svd(reflection)
        omega = backend.matmul(u, vt)

        # Fix to proper rotation
        result = _ensure_proper_rotation(u, vt, omega, backend)

        # Determinant should be +1
        det = backend.det(result)
        backend.eval(det)
        det_scalar = float(backend.to_scalar(det))
        eps = _eps(det_scalar)
        assert abs(det_scalar - 1.0) <= eps

    def test_proper_rotation_preserves_orthogonality(self):
        """Proper rotation should remain orthogonal."""
        backend = get_default_backend()

        # Random orthogonal matrix via SVD
        backend.random_seed(42)
        random_mat = backend.random_normal((3, 3))
        u, _, vt = backend.svd(random_mat)
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)

        # R @ R^T should be identity
        # Tolerance: n * eps for n×n matrix operations (error accumulates)
        n = 3
        product = backend.matmul(result, backend.transpose(result))
        expected = backend.eye(n)
        backend.eval(product, expected)
        diff = backend.abs(product - expected)
        backend.eval(diff)
        diff_val = float(backend.max(diff))
        eps = _eps(diff_val)
        assert diff_val <= n * eps


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
        eps = _eps(fp.entropies[0])
        assert -eps <= fp.entropies[0] <= 1.0 + eps

    def test_entropy_peaked_is_low(self):
        """Peaked distribution should have low entropy."""
        # One very high value, rest near zero
        peaked = [0.01] * 99 + [100.0]
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: peaked},
        )

        uniform = [1.0] * 100
        uniform_fp = ContinuousFingerprint.from_activations(
            prime_id="uniform",
            prime_text="uniform",
            layer_activations={0: uniform},
        )
        eps = _eps(fp.entropies[0], uniform_fp.entropies[0])
        assert fp.entropies[0] <= uniform_fp.entropies[0] + eps

    def test_sparsity_bounded_zero_one(self):
        """Sparsity should be in [0, 1]."""
        activations = [0.0, 0.0, 0.0, 1.0, 2.0]  # 60% near zero
        fp = ContinuousFingerprint.from_activations(
            prime_id="test",
            prime_text="test",
            layer_activations={0: activations},
        )

        eps = _eps(fp.sparsities[0])
        assert -eps <= fp.sparsities[0] <= 1.0 + eps


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
        shape = backend.shape(matrix)

        if shape[0] > 0:
            # Diagonal should be 1 (self-similarity)
            diag = backend.diag(matrix)
            backend.eval(diag)
            for i in range(min(shape)):
                val = float(backend.to_scalar(diag[i]))
                eps = _eps(val)
                assert abs(val - 1.0) <= eps


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
        eps = _eps(result.cka, result.cosine_similarity, result.magnitude_ratio)
        assert -eps <= result.cka <= 1.0 + eps
        assert -1.0 - eps <= result.cosine_similarity <= 1.0 + eps
        assert result.magnitude_ratio >= eps

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
            eps = _eps(cluster.procrustes_error)
            assert cluster.procrustes_error >= -eps  # Error is non-negative

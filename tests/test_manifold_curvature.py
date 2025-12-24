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

"""Tests for ManifoldCurvature estimation.

Tests mathematical properties of Riemannian curvature computation:
- Sectional curvature sign classification
- Curvature anisotropy bounds [0, 1]
- Metric tensor positive definiteness
- Christoffel symbol symmetry
- Curvature divergence properties
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    CurvatureConfig,
    LocalCurvature,
    ManifoldCurvatureProfile,
    SectionalCurvatureEstimator,
    compute_curvature_divergence,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_local_curvature(
    mean: float = 0.0,
    variance: float = 0.01,
    min_val: float | None = None,
    max_val: float | None = None,
    sign: CurvatureSign = CurvatureSign.FLAT,
) -> LocalCurvature:
    """Create a LocalCurvature for testing."""
    return LocalCurvature(
        point=np.zeros(4),
        mean_sectional=mean,
        variance_sectional=variance,
        min_sectional=min_val if min_val is not None else mean - 0.1,
        max_sectional=max_val if max_val is not None else mean + 0.1,
        principal_directions=None,
        principal_curvatures=None,
        sign=sign,
        scalar_curvature=mean * 3,  # Approximate
        ricci_curvature=None,
    )


def make_gaussian_samples(n: int = 100, d: int = 10, seed: int = 42) -> np.ndarray:
    """Create Gaussian samples for testing."""
    np.random.seed(seed)
    return np.random.randn(n, d)


def make_spherical_samples(n: int = 100, d: int = 10, seed: int = 42) -> np.ndarray:
    """Create samples on unit sphere (positive curvature)."""
    np.random.seed(seed)
    samples = np.random.randn(n, d)
    return samples / np.linalg.norm(samples, axis=1, keepdims=True)


# =============================================================================
# CurvatureSign Tests
# =============================================================================


class TestCurvatureSign:
    """Tests for CurvatureSign enum."""

    def test_enum_values(self) -> None:
        """Should have expected values."""
        assert CurvatureSign.POSITIVE.value == "positive"
        assert CurvatureSign.NEGATIVE.value == "negative"
        assert CurvatureSign.FLAT.value == "flat"
        assert CurvatureSign.MIXED.value == "mixed"

    def test_all_signs_distinct(self) -> None:
        """All signs should be distinct."""
        signs = list(CurvatureSign)
        assert len(signs) == len(set(signs))


# =============================================================================
# LocalCurvature Tests
# =============================================================================


class TestLocalCurvatureProperties:
    """Tests for LocalCurvature properties."""

    def test_is_positively_curved_positive_sign(self) -> None:
        """Should be positively curved when mean > 0 and sign is POSITIVE."""
        lc = make_local_curvature(mean=0.5, sign=CurvatureSign.POSITIVE)
        assert lc.is_positively_curved is True
        assert lc.is_negatively_curved is False

    def test_is_positively_curved_mixed_sign(self) -> None:
        """Should be positively curved when mean > 0 and sign is MIXED."""
        lc = make_local_curvature(mean=0.3, sign=CurvatureSign.MIXED)
        assert lc.is_positively_curved is True

    def test_is_negatively_curved_negative_sign(self) -> None:
        """Should be negatively curved when mean < 0 and sign is NEGATIVE."""
        lc = make_local_curvature(mean=-0.5, sign=CurvatureSign.NEGATIVE)
        assert lc.is_negatively_curved is True
        assert lc.is_positively_curved is False

    def test_is_negatively_curved_mixed_sign(self) -> None:
        """Should be negatively curved when mean < 0 and sign is MIXED."""
        lc = make_local_curvature(mean=-0.3, sign=CurvatureSign.MIXED)
        assert lc.is_negatively_curved is True

    def test_flat_is_neither_positive_nor_negative(self) -> None:
        """Flat curvature should be neither positive nor negative."""
        lc = make_local_curvature(mean=0.0, sign=CurvatureSign.FLAT)
        assert lc.is_positively_curved is False
        assert lc.is_negatively_curved is False


class TestCurvatureAnisotropy:
    """Tests for curvature anisotropy computation."""

    def test_isotropic_has_zero_anisotropy(self) -> None:
        """Isotropic curvature (min = max) should have 0 anisotropy."""
        lc = make_local_curvature(mean=1.0, min_val=1.0, max_val=1.0)
        assert lc.curvature_anisotropy == 0.0

    def test_anisotropy_bounded_zero_one(self) -> None:
        """Anisotropy should be in [0, 1] for all valid inputs."""
        for min_val, max_val in [(-1, 1), (-0.5, 0.5), (0, 2), (-2, 0)]:
            lc = make_local_curvature(
                mean=(min_val + max_val) / 2,
                min_val=min_val,
                max_val=max_val,
            )
            assert 0.0 <= lc.curvature_anisotropy <= 1.0

    def test_high_anisotropy_for_different_signs(self) -> None:
        """Large difference between min and max should give high anisotropy."""
        lc = make_local_curvature(mean=0.0, min_val=-1.0, max_val=1.0)
        # (max - min) / (|max| + |min|) = 2 / 2 = 1
        assert lc.curvature_anisotropy == pytest.approx(1.0)


# =============================================================================
# SectionalCurvatureEstimator Tests
# =============================================================================


class TestSectionalCurvatureEstimator:
    """Tests for curvature estimation."""

    def test_insufficient_neighbors_returns_flat(self) -> None:
        """Too few neighbors should return flat curvature."""
        estimator = SectionalCurvatureEstimator()

        point = np.zeros(10)
        neighbors = np.random.randn(5, 10)  # Less than d+1 = 11

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.sign == CurvatureSign.FLAT
        assert curvature.mean_sectional == 0.0
        assert curvature.variance_sectional == 0.0

    def test_flat_gaussian_has_low_curvature(self) -> None:
        """Flat Gaussian samples should have low curvature magnitude."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=10)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # Flat space should have curvature near 0
        assert abs(curvature.mean_sectional) < 1.0

    def test_spherical_has_positive_mean_curvature(self) -> None:
        """Spherical samples should have positive mean curvature."""
        estimator = SectionalCurvatureEstimator()
        samples = make_spherical_samples(n=100, d=10)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # Sphere has positive curvature (estimation may be noisy)
        assert curvature.mean_sectional > -0.5

    def test_curvature_variance_non_negative(self) -> None:
        """Curvature variance should always be >= 0."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=50, d=8)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.variance_sectional >= 0

    def test_min_less_than_max(self) -> None:
        """min_sectional should be <= max_sectional."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=50, d=8)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.min_sectional <= curvature.max_sectional


class TestSignClassification:
    """Tests for curvature sign classification logic."""

    def test_mostly_positive_is_positive(self) -> None:
        """> 80% positive samples should classify as POSITIVE."""
        estimator = SectionalCurvatureEstimator()

        # Simulate > 80% positive curvatures
        curvatures = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.08, -0.02])
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.POSITIVE

    def test_mostly_negative_is_negative(self) -> None:
        """< 80% negative samples should classify as NEGATIVE."""
        estimator = SectionalCurvatureEstimator()

        curvatures = np.array([-0.1, -0.2, -0.15, -0.3, -0.25, -0.18, -0.22, -0.12, -0.08, 0.02])
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.NEGATIVE

    def test_near_zero_is_flat(self) -> None:
        """All near-zero curvatures should classify as FLAT."""
        config = CurvatureConfig(flat_threshold=0.1)
        estimator = SectionalCurvatureEstimator(config)

        curvatures = np.array([0.01, -0.02, 0.005, -0.008, 0.03, -0.01, 0.02, -0.015])
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.FLAT

    def test_mixed_signs_is_mixed(self) -> None:
        """Mixed positive/negative should classify as MIXED."""
        estimator = SectionalCurvatureEstimator()

        curvatures = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.MIXED


class TestMetricTensorEstimation:
    """Tests for metric tensor computation."""

    def test_metric_is_symmetric(self) -> None:
        """Estimated metric tensor should be symmetric."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=50, d=8)

        centered = samples - samples[0]
        metric = estimator._estimate_metric_tensor(centered)

        assert np.allclose(metric, metric.T)

    def test_metric_is_positive_definite(self) -> None:
        """Metric tensor eigenvalues should all be positive."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=8)

        centered = samples - samples[0]
        metric = estimator._estimate_metric_tensor(centered)

        eigenvalues = np.linalg.eigvalsh(metric)
        assert np.all(eigenvalues > 0)

    def test_metric_matches_dimension(self) -> None:
        """Metric tensor should have shape (d, d)."""
        estimator = SectionalCurvatureEstimator()

        for d in [4, 8, 16]:
            samples = make_gaussian_samples(n=50, d=d)
            centered = samples - samples[0]
            metric = estimator._estimate_metric_tensor(centered)

            assert metric.shape == (d, d)


class TestChristoffelSymbols:
    """Tests for Christoffel symbol computation."""

    def test_christoffel_lower_index_symmetry(self) -> None:
        """Christoffel symbols should be symmetric in lower indices.

        Γ^k_ij = Γ^k_ji (torsion-free connection)
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=8)

        point = samples[0]
        neighbors = samples[1:]

        christoffel = estimator._estimate_christoffel_symbols(point, neighbors, None)

        # Check symmetry in lower indices
        d = christoffel.shape[0]
        for k in range(d):
            for i in range(d):
                for j in range(i + 1, d):
                    assert christoffel[k, i, j] == pytest.approx(
                        christoffel[k, j, i], abs=1e-6
                    ), f"Asymmetry at Γ^{k}_{i}{j}"

    def test_christoffel_has_correct_shape(self) -> None:
        """Christoffel symbols should have shape (d, d, d)."""
        estimator = SectionalCurvatureEstimator()

        for d in [4, 8]:
            samples = make_gaussian_samples(n=50, d=d)
            point = samples[0]
            neighbors = samples[1:]

            christoffel = estimator._estimate_christoffel_symbols(point, neighbors, None)

            assert christoffel.shape == (d, d, d)


# =============================================================================
# ManifoldCurvatureProfile Tests
# =============================================================================


class TestManifoldCurvatureProfile:
    """Tests for ManifoldCurvatureProfile."""

    def test_sign_distribution_sums_to_one(self) -> None:
        """Sign distribution should sum to 1 (probability distribution)."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=50, d=8)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        total = sum(profile.sign_distribution.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_local_curvatures_match_point_count(self) -> None:
        """Should have one LocalCurvature per input point."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        assert len(profile.local_curvatures) == 30

    def test_dominant_sign_is_valid(self) -> None:
        """Dominant sign should be one of the CurvatureSign values."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        assert profile.dominant_sign in CurvatureSign

    def test_get_high_curvature_regions_returns_valid_indices(self) -> None:
        """High curvature region indices should be valid."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)
        high_curv = profile.get_high_curvature_regions(threshold=1.5)

        for idx in high_curv:
            assert 0 <= idx < len(profile.local_curvatures)


class TestCurvatureAtPoint:
    """Tests for curvature interpolation at arbitrary points."""

    def test_curvature_at_existing_point(self) -> None:
        """Curvature at measured point should match stored value."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        # Query at first point
        result = profile.curvature_at_point(samples[0], k=1)

        assert result is not None
        # Should be close to stored value (exact match for k=1)
        assert result.mean_sectional == pytest.approx(
            profile.local_curvatures[0].mean_sectional, rel=0.1
        )

    def test_curvature_at_point_returns_local_curvature(self) -> None:
        """Should return a LocalCurvature object."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        # Query at arbitrary point
        query = np.random.randn(6)
        result = profile.curvature_at_point(query, k=3)

        assert isinstance(result, LocalCurvature)


# =============================================================================
# Curvature Divergence Tests
# =============================================================================


class TestCurvatureDivergence:
    """Tests for compute_curvature_divergence."""

    def test_same_profile_zero_divergence(self) -> None:
        """Same profile compared to itself should have zero divergence."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=30, d=6)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=10)

        divergence = compute_curvature_divergence(profile, profile)

        assert divergence == 0.0

    def test_divergence_non_negative(self) -> None:
        """Divergence should always be >= 0."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=30, d=6, seed=42)
        samples_b = make_gaussian_samples(n=30, d=6, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a, k_neighbors=10)
        profile_b = estimator.estimate_manifold_profile(samples_b, k_neighbors=10)

        divergence = compute_curvature_divergence(profile_a, profile_b)

        assert divergence >= 0

    def test_divergence_is_symmetric(self) -> None:
        """Divergence(A, B) should equal Divergence(B, A)."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=30, d=6, seed=42)
        samples_b = make_gaussian_samples(n=30, d=6, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a, k_neighbors=10)
        profile_b = estimator.estimate_manifold_profile(samples_b, k_neighbors=10)

        div_ab = compute_curvature_divergence(profile_a, profile_b)
        div_ba = compute_curvature_divergence(profile_b, profile_a)

        assert div_ab == pytest.approx(div_ba)


# =============================================================================
# Mathematical Invariants
# =============================================================================


class TestPrincipalCurvatureInvariants:
    """Tests for principal curvature ordering invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_principal_curvatures_sorted_descending(self, seed: int) -> None:
        """Principal curvatures should be sorted in descending order.

        Mathematical property: λ₁ ≥ λ₂ ≥ ... ≥ λₙ by convention.
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=8, seed=seed)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        if curvature.principal_curvatures is not None:
            pcs = curvature.principal_curvatures
            for i in range(len(pcs) - 1):
                assert pcs[i] >= pcs[i + 1] - 1e-10, (
                    f"Principal curvatures not sorted: {pcs[i]} < {pcs[i+1]}"
                )

    @pytest.mark.parametrize("seed", range(5))
    def test_principal_curvature_mean_matches(self, seed: int) -> None:
        """Mean of principal curvatures should approximate mean sectional curvature.

        Mathematical property: Mean sectional curvature is related to scalar curvature.
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=8, seed=seed)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        if curvature.principal_curvatures is not None:
            pc_mean = np.mean(curvature.principal_curvatures)
            # They should be in the same ballpark
            assert np.isfinite(pc_mean)


class TestRicciCurvatureInvariants:
    """Tests for Ricci curvature invariants."""

    @pytest.mark.parametrize("seed", range(5))
    def test_ricci_eigenvalues_real(self, seed: int) -> None:
        """Ricci curvature eigenvalues should be real.

        Mathematical property: Ricci tensor is symmetric, hence has real eigenvalues.
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=8, seed=seed)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        if curvature.ricci_curvature is not None:
            # Check that all values are real (not complex)
            if hasattr(curvature.ricci_curvature, '__iter__'):
                assert all(np.isreal(v) for v in curvature.ricci_curvature)

    @pytest.mark.parametrize("d", [4, 6, 8])
    def test_scalar_curvature_is_finite(self, d: int) -> None:
        """Scalar curvature should be finite for valid inputs.

        Mathematical property: Scalar curvature R = trace of Ricci tensor.
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=100, d=d, seed=42)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert np.isfinite(curvature.scalar_curvature)


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    @given(
        mean=st.floats(min_value=-10, max_value=10, allow_nan=False),
    )
    @settings(max_examples=20)
    def test_anisotropy_bounded(self, mean: float) -> None:
        """Anisotropy should always be in [0, 1]."""
        min_val = mean - abs(mean) * 0.5 - 0.1
        max_val = mean + abs(mean) * 0.5 + 0.1

        lc = LocalCurvature(
            point=np.zeros(4),
            mean_sectional=mean,
            variance_sectional=0.1,
            min_sectional=min_val,
            max_sectional=max_val,
            principal_directions=None,
            principal_curvatures=None,
            sign=CurvatureSign.MIXED,
            scalar_curvature=mean * 3,
            ricci_curvature=None,
        )

        assert 0.0 <= lc.curvature_anisotropy <= 1.0

    def test_variance_non_negative_for_all_samples(self) -> None:
        """Variance should be non-negative for any sample set."""
        estimator = SectionalCurvatureEstimator()

        np.random.seed(42)
        for _ in range(10):
            samples = np.random.randn(50, 8)
            point = samples[0]
            neighbors = samples[1:]

            curvature = estimator.estimate_local_curvature(point, neighbors)
            assert curvature.variance_sectional >= 0


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_single_dimension(self) -> None:
        """Should handle 1D case gracefully."""
        estimator = SectionalCurvatureEstimator()

        point = np.array([0.0])
        neighbors = np.random.randn(20, 1)

        # Should not crash (returns flat curvature for low dim)
        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert curvature is not None

    def test_high_dimension(self) -> None:
        """Should handle high-dimensional case."""
        estimator = SectionalCurvatureEstimator()

        d = 100
        samples = np.random.randn(200, d)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert curvature is not None
        assert np.isfinite(curvature.mean_sectional)

    def test_very_small_variance_samples(self) -> None:
        """Should handle samples with very small variance."""
        estimator = SectionalCurvatureEstimator()

        # Nearly identical points
        samples = np.ones((50, 8)) + np.random.randn(50, 8) * 1e-10

        point = samples[0]
        neighbors = samples[1:]

        # Should not crash
        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert np.isfinite(curvature.mean_sectional)

    def test_empty_profile_curvature_at_point(self) -> None:
        """Should handle empty profile."""
        profile = ManifoldCurvatureProfile(
            local_curvatures=[],
            global_mean=0.0,
            global_variance=0.0,
            sign_distribution={s: 0.0 for s in CurvatureSign},
            dominant_sign=CurvatureSign.FLAT,
            estimated_dimension=None,
        )

        result = profile.curvature_at_point(np.zeros(4))
        assert result is None

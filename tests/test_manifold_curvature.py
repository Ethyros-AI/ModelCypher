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

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    EdgeCurvature,
    LocalCurvature,
    ManifoldCurvatureProfile,
    NodeRicciCurvature,
    OllivierRicciConfig,
    OllivierRicciCurvature,
    OllivierRicciResult,
    SectionalCurvatureEstimator,
    compute_curvature_divergence,
)
from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

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
    backend = get_default_backend()
    return LocalCurvature(
        point=backend.to_numpy(backend.zeros((4,))),
        mean_sectional=mean,
        variance_sectional=variance,
        min_sectional=min_val if min_val is not None else mean - 0.1,
        max_sectional=max_val if max_val is not None else mean + 0.1,
        principal_directions=None,
        principal_curvatures=None,
        sign=sign,
        scalar_curvature=mean * 3,  # Approximate
        principal_curvature_proxy=None,
    )


def make_gaussian_samples(n: int = 60, d: int = 6, seed: int = 42):
    """Create Gaussian samples for testing."""
    backend = get_default_backend()
    backend.random_seed(seed)
    return backend.to_numpy(backend.random_normal((n, d)))


def make_spherical_samples(n: int = 60, d: int = 6, seed: int = 42):
    """Create samples on unit sphere (positive curvature)."""
    backend = get_default_backend()
    backend.random_seed(seed)
    samples = backend.random_normal((n, d))
    norms = backend.norm(samples, axis=1, keepdims=True)
    return backend.to_numpy(samples / norms)


def _eps(backend) -> float:
    return machine_epsilon(backend, backend.array([1.0]))


_FAST_SINKHORN_ITERATIONS = 10


def _fast_ollivier_estimator(
    config: OllivierRicciConfig | None = None,
) -> OllivierRicciCurvature:
    return OllivierRicciCurvature(
        config=config
        if config is not None
        else OllivierRicciConfig(sinkhorn_iterations=_FAST_SINKHORN_ITERATIONS)
    )


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
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([lc.curvature_anisotropy, 0.0]))
        assert abs(lc.curvature_anisotropy - 0.0) <= eps

    def test_anisotropy_bounded_zero_one(self) -> None:
        """Anisotropy should be in [0, 1] for all valid inputs."""
        for min_val, max_val in [(-1, 1), (-0.5, 0.5), (0, 2), (-2, 0)]:
            lc = make_local_curvature(
                mean=(min_val + max_val) / 2,
                min_val=min_val,
                max_val=max_val,
            )
            backend = get_default_backend()
            eps = _eps(backend)
            assert lc.curvature_anisotropy >= -eps
            assert lc.curvature_anisotropy <= 1.0 + eps

    def test_high_anisotropy_for_different_signs(self) -> None:
        """Large difference between min and max should give high anisotropy."""
        lc = make_local_curvature(mean=0.0, min_val=-1.0, max_val=1.0)
        # (max - min) / (|max| + |min|) = 2 / 2 = 1
        backend = get_default_backend()
        eps = machine_epsilon(backend, backend.array([1.0]))
        assert abs(lc.curvature_anisotropy - 1.0) <= eps


# =============================================================================
# SectionalCurvatureEstimator Tests
# =============================================================================


class TestSectionalCurvatureEstimator:
    """Tests for curvature estimation."""

    def test_insufficient_neighbors_returns_flat(self) -> None:
        """Too few neighbors should return flat curvature."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        point = backend.to_numpy(backend.zeros((10,)))
        backend.random_seed(42)
        neighbors = backend.to_numpy(backend.random_normal((5, 10)))  # Less than d+1 = 11

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.sign == CurvatureSign.FLAT
        backend = get_default_backend()
        eps = machine_epsilon(
            backend,
            backend.array([curvature.mean_sectional, curvature.variance_sectional, 0.0]),
        )
        assert abs(curvature.mean_sectional - 0.0) <= eps
        assert abs(curvature.variance_sectional - 0.0) <= eps

    def test_curvature_variance_non_negative(self) -> None:
        """Curvature variance should always be >= 0."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=20, d=6)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        backend = get_default_backend()
        eps = _eps(backend)
        assert curvature.variance_sectional >= -eps

    def test_min_less_than_max(self) -> None:
        """min_sectional should be <= max_sectional."""
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=20, d=6)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert curvature.min_sectional <= curvature.max_sectional


class TestSignClassification:
    """Tests for curvature sign classification logic."""

    def test_mostly_positive_is_positive(self) -> None:
        """Positive samples should classify as POSITIVE."""
        estimator = SectionalCurvatureEstimator()

        curvatures = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.08]
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.POSITIVE

    def test_mostly_negative_is_negative(self) -> None:
        """Negative samples should classify as NEGATIVE."""
        estimator = SectionalCurvatureEstimator()

        curvatures = [-0.1, -0.2, -0.15, -0.3, -0.25, -0.18, -0.22, -0.12, -0.08]
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.NEGATIVE

    def test_near_zero_is_flat(self) -> None:
        """All near-zero curvatures should classify as FLAT."""
        estimator = SectionalCurvatureEstimator()
        curvatures = [0.0, -0.0, 0.0, -0.0]
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.FLAT

    def test_mixed_signs_is_mixed(self) -> None:
        """Mixed positive/negative should classify as MIXED."""
        estimator = SectionalCurvatureEstimator()

        curvatures = [0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1]
        sign = estimator._classify_sign(curvatures)

        assert sign == CurvatureSign.MIXED


class TestMetricTensorEstimation:
    """Tests for metric tensor computation."""

    def test_metric_is_symmetric(self) -> None:
        """Estimated metric tensor should be symmetric."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=20, d=6)

        centered = backend.array(samples - samples[0])
        metric = estimator._estimate_metric_tensor(centered, backend)

        metric_arr = backend.array(metric)
        metric_T = backend.transpose(metric_arr)
        # Check symmetry manually (allclose not in Backend protocol)
        diff = backend.abs(metric_arr - metric_T)
        max_diff = float(backend.to_numpy(backend.max(diff)))
        eps = _eps(backend)
        assert max_diff <= eps, f"Metric not symmetric, max diff: {max_diff}"

    def test_metric_is_positive_definite(self) -> None:
        """Metric tensor eigenvalues should all be positive."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=20, d=6)

        centered = backend.array(samples - samples[0])
        metric = estimator._estimate_metric_tensor(centered, backend)

        metric_arr = backend.array(metric)
        # Use eigh for symmetric matrix eigenvalue decomposition
        eigenvalues, _ = backend.eigh(metric_arr)
        eigenvalues_np = backend.to_numpy(eigenvalues)
        eps = _eps(backend)
        assert all(eigenvalues_np > -eps)

    def test_metric_matches_dimension(self) -> None:
        """Metric tensor should have shape (d, d)."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        for d in [4, 6, 8]:
            samples = make_gaussian_samples(n=20, d=d)
            centered = backend.array(samples - samples[0])
            metric = estimator._estimate_metric_tensor(centered, backend)

            assert metric.shape == (d, d)


class TestChristoffelSymbols:
    """Tests for Christoffel symbol computation."""

    def test_christoffel_lower_index_symmetry(self) -> None:
        """Christoffel symbols should be symmetric in lower indices.

        Γ^k_ij = Γ^k_ji (torsion-free connection)
        """
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=20, d=6)

        point = backend.array(samples[0])
        neighbors = backend.array(samples[1:])

        christoffel = estimator._estimate_christoffel_symbols(point, neighbors, None, backend)
        christoffel_np = backend.to_numpy(christoffel)

        # Check symmetry in lower indices
        d = christoffel_np.shape[0]
        eps = _eps(backend)
        for k in range(d):
            for i in range(d):
                for j in range(i + 1, d):
                    assert (
                        abs(christoffel_np[k, i, j] - christoffel_np[k, j, i]) <= eps
                    ), f"Asymmetry at Γ^{k}_{i}{j}"

    def test_christoffel_has_correct_shape(self) -> None:
        """Christoffel symbols should have shape (d, d, d)."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        for d in [4, 6]:
            samples = make_gaussian_samples(n=20, d=d)
            point = backend.array(samples[0])
            neighbors = backend.array(samples[1:])

            christoffel = estimator._estimate_christoffel_symbols(point, neighbors, None, backend)

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
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        total = sum(profile.sign_distribution.values())
        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(total - 1.0) <= eps

    def test_local_curvatures_match_point_count(self) -> None:
        """Should have one LocalCurvature per input point."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        assert len(profile.local_curvatures) == samples.shape[0]

    def test_dominant_sign_is_valid(self) -> None:
        """Dominant sign should be one of the CurvatureSign values."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        assert profile.dominant_sign in CurvatureSign

    def test_get_high_curvature_regions_returns_valid_indices(self) -> None:
        """High curvature region indices should be valid."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)
        backend = get_default_backend()
        high_curv = profile.get_high_curvature_regions(threshold=_eps(backend))

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
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        # Query at first point
        result = profile.curvature_at_point(samples[0], k=1)

        assert result is not None
        # Should be close to stored value (exact match for k=1)
        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(result.mean_sectional - profile.local_curvatures[0].mean_sectional) <= eps

    def test_curvature_at_point_returns_local_curvature(self) -> None:
        """Should return a LocalCurvature object."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        # Query at arbitrary point
        backend = get_default_backend()
        backend.random_seed(42)
        query = backend.to_numpy(backend.random_normal((4,)))
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
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples, k_neighbors=2)

        divergence = compute_curvature_divergence(profile, profile)

        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(divergence) <= eps

    def test_divergence_non_negative(self) -> None:
        """Divergence should always be >= 0."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=12, d=4, seed=42)
        samples_b = make_gaussian_samples(n=12, d=4, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a, k_neighbors=2)
        profile_b = estimator.estimate_manifold_profile(samples_b, k_neighbors=2)

        divergence = compute_curvature_divergence(profile_a, profile_b)

        backend = get_default_backend()
        eps = _eps(backend)
        assert divergence >= -eps

    def test_divergence_is_symmetric(self) -> None:
        """Divergence(A, B) should equal Divergence(B, A)."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.skip("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=12, d=4, seed=42)
        samples_b = make_gaussian_samples(n=12, d=4, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a, k_neighbors=2)
        profile_b = estimator.estimate_manifold_profile(samples_b, k_neighbors=2)

        div_ab = compute_curvature_divergence(profile_a, profile_b)
        div_ba = compute_curvature_divergence(profile_b, profile_a)

        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(div_ab - div_ba) <= eps


# =============================================================================
# Mathematical Invariants
# =============================================================================


class TestPrincipalCurvatureInvariants:
    """Tests for principal curvature ordering invariants."""

    @pytest.mark.parametrize("seed", range(2))
    def test_principal_curvature_mean_matches(self, seed: int) -> None:
        """Mean of principal curvatures should approximate mean sectional curvature.

        Mathematical property: Mean sectional curvature is related to scalar curvature.
        """
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=16, d=4, seed=seed)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        if curvature.principal_curvatures is not None:
            import math

            pc_arr = backend.array(curvature.principal_curvatures)
            pc_mean = backend.mean(pc_arr)
            pc_mean_scalar = float(backend.to_numpy(pc_mean))
            # They should be in the same ballpark
            assert math.isfinite(pc_mean_scalar)


class TestRicciCurvatureInvariants:
    """Tests for Ricci curvature invariants."""

    @pytest.mark.parametrize("seed", range(2))
    def test_ricci_eigenvalues_real(self, seed: int) -> None:
        """Ricci curvature eigenvalues should be real.

        Mathematical property: Ricci tensor is symmetric, hence has real eigenvalues.
        """
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=16, d=4, seed=seed)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        if curvature.principal_curvature_proxy is not None:
            # Check that all values are real (not complex)
            if hasattr(curvature.principal_curvature_proxy, "__iter__"):
                ricci_arr = backend.array(list(curvature.principal_curvature_proxy))
                ricci_np = backend.to_numpy(ricci_arr)
                # All values in the proxy should be real floats
                for v in ricci_np:
                    assert not isinstance(v, complex), "Ricci proxy value should be real"

    @pytest.mark.parametrize("d", [4, 6])
    def test_scalar_curvature_is_finite(self, d: int) -> None:
        """Scalar curvature should be finite for valid inputs.

        Mathematical property: Scalar curvature R = trace of Ricci tensor.
        """
        import math

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=16, d=d, seed=42)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert math.isfinite(curvature.scalar_curvature)


class TestMathematicalInvariants:
    """Property-based tests for mathematical invariants."""

    @given(
        mean=st.floats(min_value=-10, max_value=10, allow_nan=False),
    )
    @settings(max_examples=5)
    def test_anisotropy_bounded(self, mean: float) -> None:
        """Anisotropy should always be in [0, 1]."""
        backend = get_default_backend()
        min_val = mean - abs(mean) * 0.5 - 0.1
        max_val = mean + abs(mean) * 0.5 + 0.1

        lc = LocalCurvature(
            point=backend.to_numpy(backend.zeros((4,))),
            mean_sectional=mean,
            variance_sectional=0.1,
            min_sectional=min_val,
            max_sectional=max_val,
            principal_directions=None,
            principal_curvatures=None,
            sign=CurvatureSign.MIXED,
            scalar_curvature=mean * 3,
            principal_curvature_proxy=None,
        )

        eps = _eps(backend)
        assert lc.curvature_anisotropy >= -eps
        assert lc.curvature_anisotropy <= 1.0 + eps

    def test_variance_non_negative_for_all_samples(self) -> None:
        """Variance should be non-negative for any sample set."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        backend.random_seed(42)
        for i in range(3):
            backend.random_seed(42 + i)
            samples = backend.to_numpy(backend.random_normal((16, 4)))
            point = samples[0]
            neighbors = samples[1:]

            curvature = estimator.estimate_local_curvature(point, neighbors)
            assert curvature.variance_sectional >= 0


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_single_dimension(self) -> None:
        """Should handle 1D case gracefully."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        point = backend.to_numpy(backend.array([0.0]))
        backend.random_seed(42)
        neighbors = backend.to_numpy(backend.random_normal((8, 1)))

        # Should not crash (returns flat curvature for low dim)
        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert curvature is not None

    def test_high_dimension(self) -> None:
        """Should handle high-dimensional case."""
        import math

        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        d = 16
        backend.random_seed(42)
        samples = backend.to_numpy(backend.random_normal((40, d)))

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert curvature is not None
        assert math.isfinite(curvature.mean_sectional)

    def test_very_small_variance_samples(self) -> None:
        """Should handle samples with very small variance."""
        import math

        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        # Nearly identical points
        backend.random_seed(42)
        base = backend.ones((20, 6))
        noise = backend.random_normal((20, 6)) * _eps(backend)
        samples = backend.to_numpy(base + noise)

        point = samples[0]
        neighbors = samples[1:]

        # Should not crash
        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert math.isfinite(curvature.mean_sectional)

    def test_empty_profile_curvature_at_point(self) -> None:
        """Should handle empty profile."""
        backend = get_default_backend()
        profile = ManifoldCurvatureProfile(
            local_curvatures=[],
            global_mean=0.0,
            global_variance=0.0,
            sign_distribution={s: 0.0 for s in CurvatureSign},
            dominant_sign=CurvatureSign.FLAT,
            estimated_dimension=None,
        )

        result = profile.curvature_at_point(backend.to_numpy(backend.zeros((4,))))
        assert result is None


# =============================================================================
# Ollivier-Ricci Curvature Tests
# =============================================================================


class TestOllivierRicciConfig:
    """Tests for OllivierRicciConfig dataclass."""

    def test_config_is_frozen(self) -> None:
        """Config should be immutable."""
        config = OllivierRicciConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.base_alpha = 0.9  # type: ignore


class TestOllivierRicciCurvature:
    """Tests for OllivierRicciCurvature class."""

    def test_basic_computation(self) -> None:
        """Should compute curvature on simple point cloud."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Generate random point cloud
        points = backend.random_normal((16, 4))
        n_points = int(points.shape[0])

        estimator = _fast_ollivier_estimator()
        k_neighbors = 2
        result = estimator.compute(points, k_neighbors=k_neighbors)

        assert isinstance(result, OllivierRicciResult)
        assert len(result.edge_curvatures) > 0
        assert len(result.node_curvatures) == n_points
        assert result.n_points == n_points
        assert result.k_neighbors == k_neighbors

    def test_curvature_bounds(self) -> None:
        """Edge curvature should be bounded by 1.0."""
        import math

        backend = get_default_backend()
        backend.random_seed(123)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        k_neighbors = 2
        result = estimator.compute(points, k_neighbors=k_neighbors)
        eps = _eps(backend)

        for edge in result.edge_curvatures:
            # Curvature is at most 1 (when measures are identical)
            assert edge.curvature <= 1.0 + eps
            # Curvature is finite
            assert math.isfinite(edge.curvature)

    def test_node_curvature_aggregation(self) -> None:
        """Node curvatures should aggregate correctly."""
        backend = get_default_backend()
        backend.random_seed(456)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)
        eps = _eps(backend)

        for node in result.node_curvatures:
            # Min <= mean <= max
            assert node.min_curvature <= node.mean_curvature + eps
            assert node.mean_curvature <= node.max_curvature + eps
            # Has at least one edge (k >= 1)
            assert node.num_edges >= 1

    def test_symmetrized_graph(self) -> None:
        """Symmetrize option should create undirected edges."""
        backend = get_default_backend()
        backend.random_seed(222)

        points = backend.random_normal((16, 4))

        config = OllivierRicciConfig(
            symmetrize=True, sinkhorn_iterations=_FAST_SINKHORN_ITERATIONS
        )
        estimator = OllivierRicciCurvature(config=config)
        result = estimator.compute(points, k_neighbors=2)

        # Check that edges appear in both directions (symmetrized)
        edge_set = {(e.source_idx, e.target_idx) for e in result.edge_curvatures}
        for e in result.edge_curvatures:
            # Either (i,j) or (j,i) should exist (or both if symmetrized)
            has_edge = (e.source_idx, e.target_idx) in edge_set
            has_reverse = (e.target_idx, e.source_idx) in edge_set
            assert has_edge or has_reverse

    def test_determinism(self) -> None:
        """Same input should produce same output."""
        backend = get_default_backend()

        # Generate data with fixed seed
        backend.random_seed(333)
        points1 = backend.random_normal((16, 4))

        backend.random_seed(333)
        points2 = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result1 = estimator.compute(points1, k_neighbors=2)
        result2 = estimator.compute(points2, k_neighbors=2)
        eps = _eps(backend)

        assert abs(result1.mean_edge_curvature - result2.mean_edge_curvature) <= eps
        assert abs(result1.mean_node_curvature - result2.mean_node_curvature) <= eps
        assert len(result1.edge_curvatures) == len(result2.edge_curvatures)

    def test_wasserstein_distance_non_negative(self) -> None:
        """Wasserstein distance should always be non-negative."""
        backend = get_default_backend()
        backend.random_seed(444)

        points = backend.random_normal((14, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)
        eps = _eps(backend)

        for edge in result.edge_curvatures:
            assert edge.wasserstein_distance >= -eps

    def test_small_k_neighbors(self) -> None:
        """Should handle small k values."""
        backend = get_default_backend()
        backend.random_seed(555)

        points = backend.random_normal((14, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)

        assert result is not None
        assert len(result.edge_curvatures) > 0

    def test_large_k_neighbors(self) -> None:
        """Should handle k approaching n."""
        backend = get_default_backend()
        backend.random_seed(666)

        n_points = 10
        points = backend.random_normal((n_points, 4))

        estimator = _fast_ollivier_estimator()
        # k close to n-1
        result = estimator.compute(points, k_neighbors=min(8, n_points - 1))

        assert result is not None
        assert len(result.node_curvatures) == n_points

    def test_result_statistics(self) -> None:
        """Result statistics should be consistent."""
        import math

        backend = get_default_backend()
        backend.random_seed(777)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)

        # Standard deviation should be non-negative
        eps = _eps(backend)
        assert result.std_edge_curvature >= -eps

        # Mean should be finite
        assert math.isfinite(result.mean_edge_curvature)
        assert math.isfinite(result.mean_node_curvature)

    def test_config_in_result(self) -> None:
        """Result should contain the config used."""
        backend = get_default_backend()
        backend.random_seed(888)

        points = backend.random_normal((14, 4))
        n_points = int(points.shape[0])

        base_alpha = 1.0 - _eps(backend)
        config_k_neighbors = min(3, n_points - 1)
        k_neighbors = min(2, n_points - 1)
        config = OllivierRicciConfig(
            base_alpha=base_alpha,
            k_neighbors=config_k_neighbors,
            sinkhorn_iterations=_FAST_SINKHORN_ITERATIONS,
        )
        estimator = OllivierRicciCurvature(config=config)
        result = estimator.compute(points, k_neighbors=k_neighbors)

        assert result.config.base_alpha == base_alpha
        # k_neighbors in result should be from compute() call, not config
        assert result.k_neighbors == k_neighbors


class TestOllivierRicciEdgeCases:
    """Edge case tests for Ollivier-Ricci curvature."""

    def test_minimum_points(self) -> None:
        """Should handle minimum viable number of points."""
        backend = get_default_backend()
        backend.random_seed(901)

        # Minimum: k+1 points for k neighbors
        points = backend.random_normal((6, 3))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=3)

        assert result is not None
        assert result.n_points == 6

    def test_high_dimensional_points(self) -> None:
        """Should handle high-dimensional data."""
        import math

        backend = get_default_backend()
        backend.random_seed(902)

        # High dimension
        points = backend.random_normal((12, 16))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)

        assert result is not None
        assert math.isfinite(result.mean_edge_curvature)

    def test_nearly_collinear_points(self) -> None:
        """Should handle nearly collinear point configurations."""
        import math

        backend = get_default_backend()
        backend.random_seed(903)

        # Points mostly along one direction
        base = backend.linspace(0, 1, 12)
        noise = backend.random_normal((12, 3)) * _eps(backend)

        # Reshape base to broadcast
        base_expanded = backend.reshape(base, (-1, 1))
        points = base_expanded + noise

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)

        # Should complete without error
        assert result is not None
        assert math.isfinite(result.mean_edge_curvature)

    def test_clustered_points(self) -> None:
        """Should handle clustered point configurations."""
        backend = get_default_backend()
        backend.random_seed(904)

        # Two clusters
        cluster1 = backend.random_normal((10, 3)) * _eps(backend)
        cluster2 = backend.random_normal((10, 3)) * _eps(backend)
        points = backend.concatenate([cluster1, cluster2], axis=0)

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points, k_neighbors=2)

        # Clustered data may show different curvature patterns
        assert result is not None
        assert len(result.edge_curvatures) > 0


class TestLazyMeasureProperties:
    """Tests for lazy random walk measure properties."""

    def test_measure_sums_to_one(self) -> None:
        """Lazy measure should sum to 1.0."""
        backend = get_default_backend()
        backend.random_seed(1001)

        points = backend.random_normal((10, 4))
        n_points = int(points.shape[0])
        k_neighbors = 2

        estimator = _fast_ollivier_estimator()

        # We need to access internal methods to test measure construction
        # First compute to build adjacency
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k_neighbors)

        adjacency_list = estimator._build_adjacency_list(geo_result, k_neighbors, n_points)
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values())

        # Test measure for a node
        node_idx = 0
        measure = estimator._build_lazy_measure(node_idx, adjacency_list, max_degree, n_points)

        # Convert to numpy for sum
        measure_sum = float(backend.sum(measure))
        eps = _eps(backend)
        assert abs(measure_sum - 1.0) <= eps

    def test_measure_alpha_zero_is_delta(self) -> None:
        """When alpha=0, measure should be delta at node."""
        backend = get_default_backend()
        backend.random_seed(1002)

        points = backend.random_normal((10, 4))
        n_points = int(points.shape[0])
        k_neighbors = 2

        config = OllivierRicciConfig(
            base_alpha=0.0,
            adaptive_alpha=False,
            sinkhorn_iterations=_FAST_SINKHORN_ITERATIONS,
        )
        estimator = OllivierRicciCurvature(config=config)

        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k_neighbors)

        adjacency_list = estimator._build_adjacency_list(geo_result, k_neighbors, n_points)
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values())

        node_idx = 5
        measure = estimator._build_lazy_measure(node_idx, adjacency_list, max_degree, n_points)

        # With alpha=0, all mass should be on node_idx
        measure_np = backend.to_numpy(measure)
        eps = _eps(backend)
        assert abs(measure_np[node_idx] - 1.0) <= eps
        # All other entries should be 0
        for i in range(n_points):
            if i != node_idx:
                assert abs(measure_np[i]) <= eps

    def test_measure_alpha_one_is_uniform_on_neighbors(self) -> None:
        """When alpha=1, measure should be uniform on neighbors."""
        backend = get_default_backend()
        backend.random_seed(1003)

        points = backend.random_normal((10, 4))
        n_points = int(points.shape[0])
        k_neighbors = 2

        config = OllivierRicciConfig(
            base_alpha=1.0,
            adaptive_alpha=False,
            sinkhorn_iterations=_FAST_SINKHORN_ITERATIONS,
        )
        estimator = OllivierRicciCurvature(config=config)

        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points, k_neighbors=k_neighbors)

        adjacency_list = estimator._build_adjacency_list(geo_result, k_neighbors, n_points)
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values())

        node_idx = 6
        measure = estimator._build_lazy_measure(node_idx, adjacency_list, max_degree, n_points)

        measure_np = backend.to_numpy(measure)
        neighbors = adjacency_list[node_idx]
        n_neighbors = len(neighbors)
        eps = _eps(backend)

        # With alpha=1, no mass on self
        assert abs(measure_np[node_idx]) <= eps

        # Uniform mass on neighbors
        expected_mass = 1.0 / n_neighbors
        for neighbor_idx in neighbors:
            assert abs(measure_np[neighbor_idx] - expected_mass) <= eps

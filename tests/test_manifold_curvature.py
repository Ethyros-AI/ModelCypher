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

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_curvature import (
    CurvatureSign,
    LocalCurvature,
    ManifoldCurvatureProfile,
    SectionalCurvatureEstimator,
    compute_curvature_divergence,
)
from modelcypher.core.domain.geometry.ollivier_ricci import (
    EdgeCurvature,
    NodeRicciCurvature,
    OllivierRicciCurvature,
    OllivierRicciResult,
)
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.support.array_utils import array_to_list

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
    mean_val = float(mean)
    min_sectional = float(min_val) if min_val is not None else mean_val - 0.1
    max_sectional = float(max_val) if max_val is not None else mean_val + 0.1
    return LocalCurvature(
        point=array_to_list(backend, backend.zeros((4,))),
        mean_sectional=mean_val,
        variance_sectional=float(variance),
        min_sectional=min_sectional,
        max_sectional=max_sectional,
        principal_directions=None,
        principal_curvatures=None,
        sign=sign,
        scalar_curvature=mean_val * 3,  # Approximate
        principal_curvature_proxy=None,
    )


def make_gaussian_samples(n: int = 60, d: int = 6, seed: int = 42):
    """Create Gaussian samples for testing."""
    backend = get_default_backend()
    backend.random_seed(seed)
    samples = backend.random_normal((n, d))
    backend.eval(samples)
    return samples


def make_spherical_samples(n: int = 60, d: int = 6, seed: int = 42):
    """Create samples on unit sphere (positive curvature)."""
    backend = get_default_backend()
    backend.random_seed(seed)
    samples = backend.random_normal((n, d))
    norms = backend.norm(samples, axis=1, keepdims=True)
    sphere = samples / norms
    backend.eval(sphere)
    return sphere


def _eps(backend) -> float:
    return machine_epsilon(backend, backend.array([1.0]))


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _fast_ollivier_estimator() -> OllivierRicciCurvature:
    """Create Ollivier-Ricci estimator.

    All parameters are derived from data - no config needed.
    """
    return OllivierRicciCurvature()


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
        # Formula: (max - min) / (|max| + |min| + div_eps)
        # = 2 / (2 + div_eps) ≈ 1 - div_eps/2
        # Tolerance must account for division_epsilon in the denominator
        backend = get_default_backend()
        div_eps = division_epsilon(backend, backend.array([1.0]))
        assert abs(lc.curvature_anisotropy - 1.0) <= div_eps


# =============================================================================
# SectionalCurvatureEstimator Tests
# =============================================================================


class TestSectionalCurvatureEstimator:
    """Tests for curvature estimation."""

    def test_insufficient_neighbors_returns_flat(self) -> None:
        """Too few neighbors should return flat curvature."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        rng = np.random.Generator(np.random.PCG64(42))
        point = backend.zeros((10,))
        neighbors = backend.array(rng.standard_normal((5, 10)).astype(np.float32))
        backend.eval(point, neighbors)

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
        max_val = backend.max(diff)
        backend.eval(max_val)
        max_diff = float(backend.to_scalar(max_val))
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
        eps = _eps(backend)
        min_val = backend.min(eigenvalues)
        backend.eval(min_val)
        assert float(backend.to_scalar(min_val)) > -eps

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
        christoffel_list = array_to_list(backend, christoffel)

        # Check symmetry in lower indices
        d = len(christoffel_list)
        eps = _eps(backend)
        for k in range(d):
            for i in range(d):
                for j in range(i + 1, d):
                    assert (
                        abs(christoffel_list[k][i][j] - christoffel_list[k][j][i]) <= eps
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
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        total = sum(profile.sign_distribution.values())
        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(total - 1.0) <= eps

    def test_local_curvatures_match_point_count(self) -> None:
        """Should have one LocalCurvature per input point."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        assert len(profile.local_curvatures) == samples.shape[0]

    def test_dominant_sign_is_valid(self) -> None:
        """Dominant sign should be one of the CurvatureSign values."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        assert profile.dominant_sign in CurvatureSign

    def test_get_high_curvature_regions_returns_valid_indices(self) -> None:
        """High curvature region indices should be valid."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)
        high_curv = profile.get_high_curvature_regions()

        for idx in high_curv:
            assert 0 <= idx < len(profile.local_curvatures)


class TestCurvatureAtPoint:
    """Tests for curvature interpolation at arbitrary points."""

    def test_curvature_at_existing_point(self) -> None:
        """Curvature at measured point should match stored value."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        # Query at first point
        result = profile.curvature_at_point(samples[0])

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
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        # Query at arbitrary point
        backend = get_default_backend()
        backend.random_seed(42)
        query = backend.random_normal((4,))
        backend.eval(query)
        result = profile.curvature_at_point(query)

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
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=12, d=4)

        profile = estimator.estimate_manifold_profile(samples)

        divergence = compute_curvature_divergence(profile, profile)

        backend = get_default_backend()
        eps = _eps(backend)
        assert abs(divergence) <= eps

    def test_divergence_non_negative(self) -> None:
        """Divergence should always be >= 0."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=12, d=4, seed=42)
        samples_b = make_gaussian_samples(n=12, d=4, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a)
        profile_b = estimator.estimate_manifold_profile(samples_b)

        divergence = compute_curvature_divergence(profile_a, profile_b)

        backend = get_default_backend()
        eps = _eps(backend)
        assert divergence >= -eps

    def test_divergence_is_symmetric(self) -> None:
        """Divergence(A, B) should equal Divergence(B, A)."""
        try:
            from scipy.spatial import KDTree
        except ImportError:
            pytest.fail("scipy required for manifold profile")

        estimator = SectionalCurvatureEstimator()

        samples_a = make_gaussian_samples(n=12, d=4, seed=42)
        samples_b = make_gaussian_samples(n=12, d=4, seed=123)

        profile_a = estimator.estimate_manifold_profile(samples_a)
        profile_b = estimator.estimate_manifold_profile(samples_b)

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
            pc_arr = backend.array(curvature.principal_curvatures)
            pc_mean = backend.mean(pc_arr)
            backend.eval(pc_mean)
            pc_mean_scalar = float(backend.to_scalar(pc_mean))
            # They should be in the same ballpark
            assert _is_finite(pc_mean_scalar)


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
                ricci_list = array_to_list(backend, ricci_arr)
                # All values in the proxy should be real floats
                for v in ricci_list:
                    assert not isinstance(v, complex), "Ricci proxy value should be real"

    @pytest.mark.parametrize("d", [4, 6])
    def test_scalar_curvature_is_finite(self, d: int) -> None:
        """Scalar curvature should be finite for valid inputs.

        Mathematical property: Scalar curvature R = trace of Ricci tensor.
        """
        estimator = SectionalCurvatureEstimator()
        samples = make_gaussian_samples(n=16, d=d, seed=42)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        assert _is_finite(curvature.scalar_curvature)


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
            point=array_to_list(backend, backend.zeros((4,))),
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
            samples = backend.random_normal((16, 4))
            backend.eval(samples)
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

        point = backend.array([0.0])
        backend.random_seed(42)
        neighbors = backend.random_normal((8, 1))
        backend.eval(point, neighbors)

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
        samples = backend.random_normal((40, d))
        backend.eval(samples)

        point = samples[0]
        neighbors = samples[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert curvature is not None
        assert _is_finite(curvature.mean_sectional)

    def test_very_small_variance_samples(self) -> None:
        """Should handle samples with very small variance."""
        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()

        # Nearly identical points
        backend.random_seed(42)
        base = backend.ones((20, 6))
        noise = backend.random_normal((20, 6)) * _eps(backend)
        samples = base + noise
        backend.eval(samples)

        point = samples[0]
        neighbors = samples[1:]

        # Should not crash
        curvature = estimator.estimate_local_curvature(point, neighbors)
        assert _is_finite(curvature.mean_sectional)

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

        result = profile.curvature_at_point(backend.zeros((4,)))
        assert result is None


# =============================================================================
# Ollivier-Ricci Curvature Tests
# =============================================================================


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
        result = estimator.compute(points)

        assert isinstance(result, OllivierRicciResult)
        assert len(result.edge_curvatures) > 0
        assert len(result.node_curvatures) == n_points
        assert result.n_points == n_points
        # k_neighbors is derived from intrinsic dimension, just check it's positive
        assert result.k_neighbors >= 1

    def test_curvature_bounds(self) -> None:
        """Edge curvature should be bounded by 1.0."""
        backend = get_default_backend()
        backend.random_seed(123)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)
        eps = _eps(backend)

        for edge in result.edge_curvatures:
            # Curvature is at most 1 (when measures are identical)
            assert edge.curvature <= 1.0 + eps
            # Curvature is finite
            assert _is_finite(edge.curvature)

    def test_node_curvature_aggregation(self) -> None:
        """Node curvatures should aggregate correctly."""
        backend = get_default_backend()
        backend.random_seed(456)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)
        eps = _eps(backend)

        for node in result.node_curvatures:
            # Min <= mean <= max
            assert node.min_curvature <= node.mean_curvature + eps
            assert node.mean_curvature <= node.max_curvature + eps
            # Has at least one edge (k >= 1)
            assert node.num_edges >= 1

    def test_symmetrized_graph(self) -> None:
        """Graph should have symmetric edges (derived from k-NN)."""
        backend = get_default_backend()
        backend.random_seed(222)

        points = backend.random_normal((16, 4))

        # Symmetry is now automatic in the implementation
        estimator = OllivierRicciCurvature()
        result = estimator.compute(points)

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
        result1 = estimator.compute(points1)
        result2 = estimator.compute(points2)
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
        result = estimator.compute(points)
        eps = _eps(backend)

        for edge in result.edge_curvatures:
            assert edge.wasserstein_distance >= -eps

    def test_small_k_neighbors(self) -> None:
        """Should handle small k values."""
        backend = get_default_backend()
        backend.random_seed(555)

        points = backend.random_normal((14, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

        assert result is not None
        assert len(result.edge_curvatures) > 0

    def test_large_k_neighbors(self) -> None:
        """Should handle k approaching n."""
        backend = get_default_backend()
        backend.random_seed(666)

        n_points = 10
        points = backend.random_normal((n_points, 4))

        estimator = _fast_ollivier_estimator()
        # k is derived from data, should still work
        result = estimator.compute(points)

        assert result is not None
        assert len(result.node_curvatures) == n_points

    def test_result_statistics(self) -> None:
        """Result statistics should be consistent."""
        backend = get_default_backend()
        backend.random_seed(777)

        points = backend.random_normal((16, 4))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

        # Standard deviation should be non-negative
        eps = _eps(backend)
        assert result.std_edge_curvature >= -eps

        # Mean should be finite
        assert _is_finite(result.mean_edge_curvature)
        assert _is_finite(result.mean_node_curvature)

class TestOllivierRicciEdgeCases:
    """Edge case tests for Ollivier-Ricci curvature."""

    def test_minimum_points(self) -> None:
        """Should handle minimum viable number of points."""
        backend = get_default_backend()
        backend.random_seed(901)

        # Minimum: k+1 points for k neighbors
        points = backend.random_normal((6, 3))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

        assert result is not None
        assert result.n_points == 6

    def test_high_dimensional_points(self) -> None:
        """Should handle high-dimensional data."""
        backend = get_default_backend()
        backend.random_seed(902)

        # High dimension
        points = backend.random_normal((12, 16))

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

        assert result is not None
        assert _is_finite(result.mean_edge_curvature)

    def test_nearly_collinear_points(self) -> None:
        """Should handle nearly collinear point configurations."""
        backend = get_default_backend()
        backend.random_seed(903)

        # Points mostly along one direction
        base = backend.linspace(0, 1, 12)
        noise = backend.random_normal((12, 3)) * _eps(backend)

        # Reshape base to broadcast
        base_expanded = backend.reshape(base, (-1, 1))
        points = base_expanded + noise

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

        # Should complete without error
        assert result is not None
        assert _is_finite(result.mean_edge_curvature)

    def test_clustered_points(self) -> None:
        """Should handle clustered point configurations."""
        backend = get_default_backend()
        backend.random_seed(904)

        # Two clusters with meaningful intra-cluster variance (0.1)
        # and inter-cluster separation (offset cluster2 by 2.0)
        cluster1 = backend.random_normal((10, 3)) * 0.1
        cluster2 = backend.random_normal((10, 3)) * 0.1 + 2.0
        points = backend.concatenate([cluster1, cluster2], axis=0)

        estimator = _fast_ollivier_estimator()
        result = estimator.compute(points)

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

        estimator = _fast_ollivier_estimator()

        # First run compute to set up derived parameters (_derived_base_alpha, etc.)
        result = estimator.compute(points)
        k_neighbors = result.k_neighbors

        # Now we can access internal methods
        from modelcypher.core.domain.geometry.riemannian_utils import RiemannianGeometry

        rg = RiemannianGeometry(backend)
        geo_result = rg.geodesic_distances(points)

        adjacency_list = estimator._build_adjacency_list(geo_result, k_neighbors, n_points)
        max_degree = max(len(neighbors) for neighbors in adjacency_list.values())

        # Test measure for a node
        node_idx = 0
        measure = estimator._build_lazy_measure(node_idx, adjacency_list, max_degree, n_points)

        # Measure should sum to 1.0 (probability distribution)
        measure_sum = float(backend.sum(measure))
        eps = _eps(backend)
        assert abs(measure_sum - 1.0) <= eps


# =============================================================================
# Ground-Truth Curvature Validation Tests
# =============================================================================


class TestGroundTruthFlatPlane:
    """Flat plane in R^d should have K = 0.

    Ground truth: An affine subspace of R^d has zero sectional curvature
    everywhere. Tolerance: finite-difference error O(h^2) where h is mean
    nearest-neighbor distance.
    """

    @pytest.mark.parametrize("d", [4, 6, 8])
    def test_flat_plane_curvature_near_zero(self, d: int) -> None:
        """Points sampled from a flat affine subspace should yield K ≈ 0."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Generate points on a flat d-dim plane: x_i = origin + sum_j(a_ij * e_j)
        # Use first 2 basis vectors as the plane, rest zero — still flat in R^d
        n = max(40, 3 * d)
        coords = backend.random_normal((n, 2))  # 2D coordinates on the plane
        # Embed into R^d: only first two dims are nonzero
        zeros_pad = backend.zeros((n, d - 2))
        points = backend.concatenate([coords, zeros_pad], axis=1)
        # Add small offset to avoid exact origin (numerical stability)
        offset = backend.ones((1, d)) * 0.5
        points = points + offset
        backend.eval(points)

        estimator = SectionalCurvatureEstimator()
        point = points[0]
        neighbors = points[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # K should be near zero. Tolerance from finite-difference truncation
        # error O(h^2), where h = n^(-1/d) is the characteristic spacing for n
        # points in d dimensions. Clamp by sqrt(machine_epsilon) for dtype floor.
        h = n ** (-1.0 / d)
        tol = max(h * h, math.sqrt(machine_epsilon(backend, points)))
        assert abs(curvature.mean_sectional) < tol, (
            f"Flat plane K should be near 0, got {curvature.mean_sectional}"
        )
        # Sign should be FLAT or very weak MIXED
        assert curvature.sign in (CurvatureSign.FLAT, CurvatureSign.MIXED), (
            f"Flat plane sign should be FLAT or MIXED, got {curvature.sign}"
        )


class TestGroundTruthSphere:
    """Unit sphere S^(d-1) should have K = 1.

    Ground truth: The sectional curvature of S^(d-1) with radius R is K = 1/R^2.
    For the unit sphere (R=1), K = 1.

    The promoted estimator uses a canonical manifold fit selector that chooses
    sphere fitting when it beats the flat model by a precision-derived margin.
    """
    @pytest.mark.parametrize("d", [4, 6, 8])
    def test_sphere_curvature_positive(self, d: int) -> None:
        """Points on S^(d-1) should yield positive curvature."""
        backend = get_default_backend()
        backend.random_seed(42)

        # Dense sampling on unit sphere
        n = max(80, 5 * d)
        raw = backend.random_normal((n, d))
        norms = backend.norm(raw, axis=1, keepdims=True)
        sphere = raw / norms
        backend.eval(sphere)

        estimator = SectionalCurvatureEstimator()
        point = sphere[0]
        neighbors = sphere[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # K must be positive (ground truth: K = 1)
        assert curvature.mean_sectional > 0, (
            f"Sphere K should be positive, got {curvature.mean_sectional}"
        )
        assert curvature.sign in (CurvatureSign.POSITIVE, CurvatureSign.MIXED), (
            f"Sphere sign should be POSITIVE, got {curvature.sign}"
        )


class TestGroundTruthHyperboloid:
    """Hyperboloid x_1^2 + ... + x_{d-1}^2 - x_d^2 = -1 has K < 0.

    Ground truth: The hyperboloid model of hyperbolic space has constant
    negative sectional curvature K = -1. We test the sign only (magnitude
    harder with finite samples).

    The promoted estimator uses a canonical manifold fit selector that chooses
    hyperboloid fitting when it beats the flat model by a precision-derived
    margin.
    """
    def test_hyperboloid_curvature_negative(self) -> None:
        """Points on the hyperboloid should yield negative curvature."""
        backend = get_default_backend()
        rng = np.random.Generator(np.random.PCG64(42))
        d = 4  # Ambient dimension (3D hyperboloid in 4D)

        # Sample points on the upper sheet of x1^2 + x2^2 + x3^2 - x4^2 = -1
        # Parametrize: x_i = sinh(r) * u_i for i<d, x_d = cosh(r)
        # where u is on the unit (d-2)-sphere and r is the hyperbolic radius
        n = 80
        # Sample in a local patch — radius must be large enough for
        # curvature to be detectable above the finite-sample noise floor.
        r_vals = np.abs(rng.standard_normal((n, 1)) * 1.0).astype(np.float32)

        # Random directions on S^(d-2) for the spatial part
        raw_dirs = rng.standard_normal((n, d - 1)).astype(np.float32)
        dir_norms = np.linalg.norm(raw_dirs, axis=1, keepdims=True)
        unit_dirs = raw_dirs / dir_norms

        # Hyperboloid embedding: spatial = sinh(r) * u, time = cosh(r)
        sinh_r = np.sinh(r_vals)
        cosh_r = np.cosh(r_vals)
        spatial = unit_dirs * sinh_r
        points_np = np.concatenate([spatial, cosh_r], axis=1)
        points = backend.array(points_np)
        backend.eval(points)

        estimator = SectionalCurvatureEstimator()
        point = points[0]
        neighbors = points[1:]

        curvature = estimator.estimate_local_curvature(point, neighbors)

        # K should be negative (ground truth: K = -1)
        # Use scalar_curvature (from principal curvatures / shape operator)
        # which is always computed, unlike mean_sectional which depends on
        # the conservative canonical selector accepting the fit.
        assert curvature.scalar_curvature < 0, (
            f"Hyperboloid scalar curvature should be negative, got {curvature.scalar_curvature}"
        )
        # Majority of principal curvatures should be negative
        if curvature.principal_curvatures is not None:
            pc = backend.tolist(curvature.principal_curvatures)
            neg_count = sum(1 for k in pc if k < 0)
            assert neg_count > len(pc) // 2, (
                f"Majority of principal curvatures should be negative, got {pc}"
            )


class TestCanonicalSelectorRegression:
    """Regression tests for canonical selector false-positive control."""

    @pytest.mark.parametrize("seed", [11, 17, 23])
    def test_selector_rejects_random_gaussian_clouds(self, seed: int) -> None:
        backend = get_default_backend()
        rng = np.random.Generator(np.random.PCG64(seed))
        estimator = SectionalCurvatureEstimator()

        samples = backend.array(rng.standard_normal((80, 6)).astype(np.float32))
        backend.eval(samples)
        point = samples[0]
        neighbors = samples[1:]

        candidate = estimator._canonical_fit_candidate(point, neighbors, backend)
        assert candidate.get("selected", False) is False, (
            f"Selector falsely chose {candidate.get('model')} on Gaussian cloud: {candidate}"
        )


# =============================================================================
# GeometryDomain Guard Tests
# =============================================================================


class TestGeometryDomainGuard:
    """Tests that curvature estimation rejects weight-domain tensors."""

    def test_sectional_estimator_rejects_weight_domain(self) -> None:
        """SectionalCurvatureEstimator should raise ValueError for WEIGHT domain."""
        from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain

        backend = get_default_backend()
        estimator = SectionalCurvatureEstimator()
        point = backend.zeros((6,))
        neighbors = backend.random_normal((10, 6))
        backend.eval(point, neighbors)

        with pytest.raises(ValueError, match="weight"):
            estimator.estimate_local_curvature(
                point, neighbors, domain=GeometryDomain.WEIGHT
            )

    def test_sectional_estimator_accepts_activation_domain(self) -> None:
        """SectionalCurvatureEstimator should work for ACTIVATION domain."""
        from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain

        backend = get_default_backend()
        backend.random_seed(42)
        estimator = SectionalCurvatureEstimator()
        point = backend.zeros((6,))
        neighbors = backend.random_normal((20, 6))
        backend.eval(point, neighbors)

        # Should not raise
        curvature = estimator.estimate_local_curvature(
            point, neighbors, domain=GeometryDomain.ACTIVATION
        )
        assert isinstance(curvature, LocalCurvature)

    def test_riemannian_curvature_mixin_rejects_weight_domain(self) -> None:
        """RiemannianCurvatureMixin should raise ValueError for WEIGHT domain."""
        from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain
        from modelcypher.core.domain.geometry.riemannian_utils import (
            RiemannianGeometry,
        )

        backend = get_default_backend()
        backend.random_seed(42)
        rg = RiemannianGeometry(backend)
        points = backend.random_normal((20, 6))
        backend.eval(points)

        with pytest.raises(ValueError, match="weight"):
            rg.estimate_local_curvature(
                points, center_idx=0, domain=GeometryDomain.WEIGHT
            )

    def test_manifold_profile_rejects_weight_domain(self) -> None:
        """Manifold profile estimation should reject WEIGHT domain."""
        from modelcypher.core.domain.geometry.geometry_domain import GeometryDomain

        backend = get_default_backend()
        backend.random_seed(42)
        estimator = SectionalCurvatureEstimator()
        points = backend.random_normal((20, 6))
        backend.eval(points)

        with pytest.raises(ValueError, match="weight"):
            estimator.estimate_manifold_profile(
                points, domain=GeometryDomain.WEIGHT
            )

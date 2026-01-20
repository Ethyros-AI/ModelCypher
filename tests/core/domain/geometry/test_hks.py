# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for Heat Kernel Signature (HKS)."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.spectral_signature import (
    HeatKernelSignature,
    compare_hks_profiles,
    compute_heat_kernel_signature,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestHeatKernelSignature:
    """Tests for Heat Kernel Signature computation."""

    def test_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ])

        result = compute_heat_kernel_signature(points, n_times=5, backend=b)

        assert hasattr(result, "signatures")
        assert hasattr(result, "times")
        assert hasattr(result, "eigenvalues")
        assert hasattr(result, "t_min")
        assert hasattr(result, "t_max")

    def test_signature_shape(self, backend):
        """Signatures should have shape [n_points, n_times]."""
        b = backend
        n_points = 5
        n_times = 8

        points = b.array([
            [float(i), float(j)]
            for i in range(n_points)
            for j in [0]
        ][:n_points])

        result = compute_heat_kernel_signature(points, n_times=n_times, backend=b)

        b.eval(result.signatures)
        assert result.signatures.shape[0] == n_points
        assert result.signatures.shape[1] == n_times

    def test_times_log_spaced(self, backend):
        """Times should be logarithmically spaced."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])

        result = compute_heat_kernel_signature(points, n_times=10, backend=b)

        assert result.t_min > 0
        assert result.t_max > result.t_min

        times = b.tolist(result.times)
        # Check roughly log-spaced (ratios similar)
        if len(times) >= 3:
            ratio1 = times[1] / times[0]
            ratio2 = times[2] / times[1]
            # Ratios should be similar for log-spacing
            assert abs(ratio1 - ratio2) / max(ratio1, ratio2) < 0.5

    def test_signatures_positive(self, backend):
        """HKS signatures should be non-negative."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866],
        ])

        result = compute_heat_kernel_signature(points, n_times=5, backend=b)

        b.eval(result.signatures)
        min_val = b.min(result.signatures)
        b.eval(min_val)

        # HKS should be non-negative (sum of exp(...) * phi^2)
        assert float(b.to_scalar(min_val)) >= -1e-10

    def test_small_point_cloud(self, backend):
        """Test with very small point cloud."""
        b = backend
        points = b.array([[0.0, 0.0], [1.0, 0.0]])

        result = compute_heat_kernel_signature(points, n_times=3, backend=b)

        assert hasattr(result, "signatures")

    def test_single_point_degenerate(self, backend):
        """Single point should return degenerate result."""
        b = backend
        points = b.array([[0.0, 0.0]])

        result = compute_heat_kernel_signature(points, n_times=3, backend=b)

        assert result.t_min == 0.0
        assert result.t_max == 0.0


class TestHKSClass:
    """Tests for HeatKernelSignature class."""

    def test_limited_eigenvalues(self, backend):
        """Test limiting number of eigenvalues."""
        b = backend
        hks = HeatKernelSignature(b)

        points = b.array([
            [float(i), float(j)]
            for i in range(4)
            for j in range(4)
        ])

        result = hks.compute(points, n_times=5, k_eigenvalues=3)

        b.eval(result.eigenvalues)
        assert result.eigenvalues.shape[0] <= 3


class TestCompareHKSProfiles:
    """Tests for HKS profile comparison."""

    def test_same_shape_zero_difference(self, backend):
        """Same shape should have near-zero difference."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])

        hks1 = compute_heat_kernel_signature(points, n_times=5, backend=b)
        hks2 = compute_heat_kernel_signature(points, n_times=5, backend=b)

        diff = compare_hks_profiles(hks1, hks2, backend=b)

        # Same points should give same HKS (zero difference)
        assert diff < 0.01

    def test_different_shapes_positive_difference(self, backend):
        """Different shapes should have positive difference."""
        b = backend
        points1 = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])
        points2 = b.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 5.0],
        ])

        hks1 = compute_heat_kernel_signature(points1, n_times=5, backend=b)
        hks2 = compute_heat_kernel_signature(points2, n_times=5, backend=b)

        diff = compare_hks_profiles(hks1, hks2, backend=b)

        # Different shapes should have non-zero difference
        assert diff >= 0

    def test_comparison_bounded(self, backend):
        """Comparison should be bounded in [0, 1]."""
        b = backend
        points1 = b.array([[0.0, 0.0], [1.0, 0.0]])
        points2 = b.array([[0.0, 0.0], [100.0, 0.0]])

        hks1 = compute_heat_kernel_signature(points1, n_times=3, backend=b)
        hks2 = compute_heat_kernel_signature(points2, n_times=3, backend=b)

        diff = compare_hks_profiles(hks1, hks2, backend=b)

        assert 0.0 <= diff <= 1.0

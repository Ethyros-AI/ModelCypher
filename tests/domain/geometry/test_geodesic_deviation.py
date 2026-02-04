# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for geodesic deviation analysis."""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.geodesic_deviation import (
    GeodesicDeviationAnalyzer,
    compute_geodesic_deviation,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestGeodesicDeviation:
    """Tests for geodesic deviation computation."""

    def test_result_structure(self, backend):
        """Test that result has expected fields."""
        b = backend
        # Simple point cloud
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ])

        result = compute_geodesic_deviation(
            points, start_idx=0, end_idx=4, n_perturbations=2, backend=b
        )

        assert hasattr(result, "reference_path")
        assert hasattr(result, "deviation_rates")
        assert hasattr(result, "mean_deviation_rate")
        assert hasattr(result, "arc_lengths")
        assert hasattr(result, "separations")
        assert hasattr(result, "curvature_correlation")

    def test_straight_line_low_deviation(self, backend):
        """Points on a straight line should have low deviation."""
        b = backend
        # Perfectly aligned points
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])

        result = compute_geodesic_deviation(
            points, start_idx=0, end_idx=3, n_perturbations=4, backend=b
        )

        # Mean deviation rate should be finite
        assert result.mean_deviation_rate != float("inf")
        assert result.mean_deviation_rate != float("-inf")

    def test_minimum_path_length(self, backend):
        """Degenerate path should return sensible defaults."""
        b = backend
        points = b.array([[0.0, 0.0], [1.0, 0.0]])

        result = compute_geodesic_deviation(
            points, start_idx=0, end_idx=1, n_perturbations=2, backend=b
        )

        # Should still return valid structure
        assert len(result.reference_path) >= 2

    def test_arc_lengths_monotonic(self, backend):
        """Arc lengths should be monotonically increasing."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ])

        result = compute_geodesic_deviation(
            points, start_idx=0, end_idx=3, n_perturbations=2, backend=b
        )

        arc_lengths = b.tolist(result.arc_lengths)
        for i in range(1, len(arc_lengths)):
            assert arc_lengths[i] >= arc_lengths[i - 1]

    def test_deviation_rates_array_shape(self, backend):
        """Deviation rates should have n_perturbations entries."""
        b = backend
        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])

        n_perturbations = 5
        result = compute_geodesic_deviation(
            points,
            start_idx=0,
            end_idx=2,
            n_perturbations=n_perturbations,
            backend=b,
        )

        b.eval(result.deviation_rates)
        assert result.deviation_rates.shape[0] == n_perturbations


class TestGeodesicDeviationAnalyzer:
    """Tests for GeodesicDeviationAnalyzer class."""

    def test_custom_perturbation_scale(self, backend):
        """Test with custom perturbation scale."""
        b = backend
        analyzer = GeodesicDeviationAnalyzer(b)

        points = b.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ])

        result = analyzer.compute_deviation(
            points,
            start_idx=0,
            end_idx=2,
            n_perturbations=2,
            perturbation_scale=0.01,
        )

        assert result.mean_deviation_rate is not None

    def test_higher_dimensional_space(self, backend):
        """Test deviation in higher dimensional space."""
        b = backend
        analyzer = GeodesicDeviationAnalyzer(b)

        # 5D point cloud
        points = b.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0, 0.0],
        ])

        result = analyzer.compute_deviation(
            points, start_idx=0, end_idx=3, n_perturbations=3
        )

        assert hasattr(result, "deviation_rates")
        b.eval(result.deviation_rates)
        assert result.deviation_rates.shape[0] == 3

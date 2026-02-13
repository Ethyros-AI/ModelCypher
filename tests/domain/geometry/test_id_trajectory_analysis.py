# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for id_trajectory_analysis module.

Covers IDTrajectoryAnalysis dataclass and analyze_id_trajectory function
with various input patterns including edge cases.
"""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.id_trajectory_analysis import (
    IDTrajectoryAnalysis,
    analyze_id_trajectory,
)


class TestIDTrajectoryAnalysis:
    """Tests for the IDTrajectoryAnalysis frozen dataclass."""

    def test_instantiation(self):
        a = IDTrajectoryAnalysis(
            expansion_ratio=1.5,
            peak_layer=2,
            peak_dim=30.0,
            final_dim=20.0,
            smoothness=0.8,
        )
        assert a.expansion_ratio == 1.5
        assert a.peak_layer == 2
        assert a.peak_dim == 30.0
        assert a.final_dim == 20.0
        assert a.smoothness == 0.8

    def test_frozen(self):
        a = IDTrajectoryAnalysis(1.0, 0, 10.0, 10.0, 0.5)
        with pytest.raises(AttributeError):
            a.expansion_ratio = 2.0  # type: ignore[misc]


class TestAnalyzeIDTrajectory:
    """Tests for the analyze_id_trajectory function."""

    def test_simple_expand_contract(self):
        """[10, 20, 15]: peak=20 at layer 1, final=15, ratio=20/15."""
        result = analyze_id_trajectory([10.0, 20.0, 15.0])
        assert result.peak_dim == 20.0
        assert result.peak_layer == 1
        assert result.final_dim == 15.0
        assert result.expansion_ratio == pytest.approx(20.0 / 15.0)

    def test_flat_trajectory(self):
        """Constant values: expansion_ratio should be 1.0."""
        result = analyze_id_trajectory([10.0, 10.0, 10.0])
        assert result.expansion_ratio == pytest.approx(1.0)
        assert result.peak_dim == 10.0
        assert result.final_dim == 10.0

    def test_monotonically_increasing(self):
        """[5, 10, 15, 20]: peak=20 at layer 3, final=20, ratio=1.0."""
        result = analyze_id_trajectory([5.0, 10.0, 15.0, 20.0])
        assert result.peak_dim == 20.0
        assert result.peak_layer == 3
        assert result.final_dim == 20.0
        assert result.expansion_ratio == pytest.approx(1.0)

    def test_monotonically_decreasing(self):
        """[20, 15, 10, 5]: peak=20 at layer 0, final=5, ratio=4.0."""
        result = analyze_id_trajectory([20.0, 15.0, 10.0, 5.0])
        assert result.peak_dim == 20.0
        assert result.peak_layer == 0
        assert result.final_dim == 5.0
        assert result.expansion_ratio == pytest.approx(4.0)

    def test_empty_trajectory(self):
        """Empty list should return defaults."""
        result = analyze_id_trajectory([])
        assert result.peak_dim == 0.0
        assert result.peak_layer == -1
        assert result.final_dim == 0.0
        assert math.isnan(result.expansion_ratio)
        assert result.smoothness == 0.0

    def test_single_element(self):
        """Single element is < 2 valid, returns defaults."""
        result = analyze_id_trajectory([5.0])
        assert result.peak_dim == 0.0
        assert result.peak_layer == -1
        assert math.isnan(result.expansion_ratio)
        assert result.smoothness == 0.0

    def test_nan_values_filtered(self):
        """NaN values should be filtered before analysis."""
        result = analyze_id_trajectory([float("nan"), 10.0, 20.0, float("nan"), 15.0])
        # Valid values: [10, 20, 15] -> peak=20, final=15
        assert result.peak_dim == 20.0
        assert result.final_dim == 15.0
        assert result.expansion_ratio == pytest.approx(20.0 / 15.0)

    def test_all_nan(self):
        """All NaN values should return defaults (no valid values)."""
        result = analyze_id_trajectory([float("nan"), float("nan")])
        assert result.peak_dim == 0.0
        assert result.peak_layer == -1
        assert math.isnan(result.expansion_ratio)
        assert result.smoothness == 0.0

    def test_nan_with_one_valid(self):
        """One valid value among NaNs is still insufficient (< 2)."""
        result = analyze_id_trajectory([float("nan"), 5.0, float("nan")])
        assert result.peak_dim == 0.0
        assert result.peak_layer == -1
        assert math.isnan(result.expansion_ratio)

    def test_two_elements(self):
        """Two elements: sufficient for analysis."""
        result = analyze_id_trajectory([5.0, 10.0])
        assert result.peak_dim == 10.0
        assert result.peak_layer == 1
        assert result.final_dim == 10.0
        assert result.expansion_ratio == pytest.approx(1.0)

    def test_peak_at_first_layer(self):
        """Peak at layer 0."""
        result = analyze_id_trajectory([100.0, 50.0, 30.0])
        assert result.peak_layer == 0
        assert result.peak_dim == 100.0

    def test_peak_at_last_layer(self):
        """Peak at the last layer means ratio = 1.0."""
        result = analyze_id_trajectory([5.0, 10.0, 50.0])
        assert result.peak_layer == 2
        assert result.peak_dim == 50.0
        assert result.final_dim == 50.0
        assert result.expansion_ratio == pytest.approx(1.0)

    def test_smoothness_monotonic_increasing(self):
        """Monotonically increasing trajectory should have high smoothness (Spearman ~ 1)."""
        result = analyze_id_trajectory([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.smoothness == pytest.approx(1.0)

    def test_smoothness_monotonic_decreasing(self):
        """Monotonically decreasing trajectory should have smoothness ~ -1."""
        result = analyze_id_trajectory([5.0, 4.0, 3.0, 2.0, 1.0])
        assert result.smoothness == pytest.approx(-1.0)

    def test_smoothness_non_monotonic(self):
        """Non-monotonic trajectory should have smoothness between -1 and 1."""
        result = analyze_id_trajectory([1.0, 5.0, 2.0, 4.0, 3.0])
        assert -1.0 <= result.smoothness <= 1.0

    def test_zero_final_dim(self):
        """If final_dim is 0 (all zeros except peak), ratio should be NaN."""
        # Zero values are filtered out (d > 0 check), so [0, 0] gives < 2 valid
        result = analyze_id_trajectory([0.0, 0.0, 5.0, 0.0])
        # Only one valid value (5.0), so defaults
        assert result.peak_dim == 0.0
        assert math.isnan(result.expansion_ratio)

    def test_large_trajectory(self):
        """Test with a longer trajectory to verify no issues at scale."""
        trajectory = [float(i) + 1.0 for i in range(50)]
        result = analyze_id_trajectory(trajectory)
        assert result.peak_dim == 50.0
        assert result.peak_layer == 49
        assert result.final_dim == 50.0
        assert result.expansion_ratio == pytest.approx(1.0)
        assert result.smoothness == pytest.approx(1.0)

    def test_negative_values_filtered(self):
        """Negative values should be filtered (d > 0 check)."""
        result = analyze_id_trajectory([-5.0, 10.0, 20.0])
        # Valid: [10, 20], peak=20 at layer 2, final=20
        assert result.peak_dim == 20.0
        assert result.final_dim == 20.0

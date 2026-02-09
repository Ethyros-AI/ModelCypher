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

"""Tests for core.domain.geometry.id_trajectory_analysis."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.id_trajectory_analysis import (
    IDTrajectoryAnalysis,
    analyze_id_trajectory,
)


class TestAnalyzeIDTrajectory:
    def test_bell_shaped(self):
        """Bell-shaped trajectory → peak in middle, expansion_ratio > 1."""
        traj = [2.0, 4.0, 8.0, 6.0, 3.0]
        result = analyze_id_trajectory(traj)

        assert isinstance(result, IDTrajectoryAnalysis)
        assert result.peak_dim == pytest.approx(8.0)
        assert result.peak_layer == 2
        assert result.final_dim == pytest.approx(3.0)
        assert result.expansion_ratio == pytest.approx(8.0 / 3.0)
        # Not monotonic, so smoothness should be modest
        assert -1.0 <= result.smoothness <= 1.0

    def test_monotonic_increasing(self):
        """Monotonically increasing → peak at end."""
        traj = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyze_id_trajectory(traj)

        assert result.peak_dim == pytest.approx(5.0)
        assert result.peak_layer == 4
        assert result.final_dim == pytest.approx(5.0)
        assert result.expansion_ratio == pytest.approx(1.0)
        assert result.smoothness == pytest.approx(1.0)

    def test_all_nan(self):
        """All NaN → sensible defaults."""
        traj = [float("nan")] * 5
        result = analyze_id_trajectory(traj)

        assert result.peak_dim == 0.0
        assert result.peak_layer == -1
        assert result.final_dim == 0.0
        assert math.isnan(result.expansion_ratio)
        assert result.smoothness == 0.0

    def test_single_valid(self):
        """Only one valid value → defaults (need >= 2)."""
        traj = [float("nan"), 3.0, float("nan")]
        result = analyze_id_trajectory(traj)

        assert result.peak_dim == 0.0
        assert result.peak_layer == -1

    def test_with_nan_mixed(self):
        """Some NaN values mixed in."""
        traj = [1.0, float("nan"), 5.0, 3.0]
        result = analyze_id_trajectory(traj)

        assert result.peak_dim == pytest.approx(5.0)
        assert result.peak_layer == 2
        assert result.final_dim == pytest.approx(3.0)

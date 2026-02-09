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

"""Tests for core.domain.geometry.velocity_decomposition."""

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.velocity_decomposition import (
    VelocityDecompositionResult,
    decompose_velocity,
)


class TestDecomposeVelocity:
    def test_constant_norms(self):
        """Constant norms with known velocity → verify radial/angular split."""
        # n_l = n_l1 = 10, vel = 0.5 → step = 5
        # cos_theta = (100 + 100 - 25) / 200 = 0.875
        # rad = 0.875 * 10/10 - 1 = -0.125
        # ang = sqrt(0.25 - 0.015625) = sqrt(0.234375) ≈ 0.4841
        norms = [10.0, 10.0, 10.0, 10.0]
        velocities = [0.5, 0.5, 0.5]

        result = decompose_velocity(norms, velocities)

        assert isinstance(result, VelocityDecompositionResult)
        assert len(result.radial_trajectory) == 3
        assert len(result.angular_trajectory) == 3

        for i in range(3):
            assert result.cos_theta_trajectory[i] == pytest.approx(0.875)
            assert result.radial_trajectory[i] == pytest.approx(-0.125)
            assert result.angular_trajectory[i] == pytest.approx(0.484123, rel=1e-3)

    def test_known_triangle_geometry(self):
        """Right-angle triangle: n_l=3, n_l1=4, step=5 → cos_theta=0."""
        # vel = step / n_l = 5/3
        norms = [3.0, 4.0]
        velocities = [5.0 / 3.0]

        result = decompose_velocity(norms, velocities)

        # cos_theta = (16 + 9 - 25) / (2*3*4) = 0/24 = 0
        assert result.cos_theta_trajectory[0] == pytest.approx(0.0)

        # radial = cos_theta * n_l1 / n_l - 1 = 0 * 4/3 - 1 = -1
        assert result.radial_trajectory[0] == pytest.approx(-1.0)

        # angular = sqrt(vel² - rad²) = sqrt(25/9 - 1) = sqrt(16/9) = 4/3
        assert result.angular_trajectory[0] == pytest.approx(4.0 / 3.0)

    def test_near_zero_norms(self):
        """Near-zero norms → zero components, cos_theta=1."""
        norms = [1e-20, 1e-20]
        velocities = [0.1]

        result = decompose_velocity(norms, velocities)

        assert result.radial_trajectory[0] == 0.0
        assert result.angular_trajectory[0] == 0.0
        assert result.cos_theta_trajectory[0] == 1.0

    def test_empty(self):
        result = decompose_velocity([10.0], [])
        assert result.radial_trajectory == []
        assert result.angular_trajectory == []
        assert result.cos_theta_trajectory == []

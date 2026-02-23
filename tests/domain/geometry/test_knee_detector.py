# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math

import pytest

from modelcypher.core.domain.geometry.knee_detector import (
    KneeResult,
    detect_knee,
)


class TestDetectKnee:
    def test_exponential_decay_knee(self) -> None:
        """Exponential decay y = exp(-x) → knee near the transition."""
        x = [0.1 * i for i in range(20)]
        y = [math.exp(-xi) for xi in x]
        result = detect_knee(x, y, seed=42)

        # Knee of exponential decay is near x=0 where curvature is maximal
        assert result.x_knee < 0.5
        assert result.curvature != 0.0
        assert result.n_points == 20

    def test_hockey_stick_curve(self) -> None:
        """Flat then steep: y = max(0, x - 5) → knee at x=5."""
        x = [float(i) for i in range(11)]
        y = [max(0.0, xi - 5.0) for xi in x]
        result = detect_knee(x, y, seed=42)

        # Knee should be at or near x=5
        assert 4.0 <= result.x_knee <= 6.0

    def test_l_shaped_curve(self) -> None:
        """Sharp L-shape: steep descent then flat."""
        x = [float(i) for i in range(10)]
        y = [10.0, 2.0, 1.0, 0.8, 0.7, 0.65, 0.6, 0.58, 0.56, 0.55]
        result = detect_knee(x, y, seed=42)

        # Knee should be near the transition (index 1-2)
        assert result.x_knee <= 3.0

    def test_straight_line_no_knee(self) -> None:
        """Perfectly linear → knee exists but curvature is near zero."""
        x = [float(i) for i in range(10)]
        y = [2.0 * xi + 1.0 for xi in x]
        result = detect_knee(x, y, seed=42)

        assert abs(result.curvature) < 0.1

    def test_minimum_points(self) -> None:
        """Exactly 4 points — minimum viable."""
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 5.0, 2.0, 1.5]
        result = detect_knee(x, y, seed=42)
        assert isinstance(result.x_knee, float)
        assert result.n_points == 4

    def test_too_few_raises(self) -> None:
        with pytest.raises(ValueError, match="Need >= 4"):
            detect_knee([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Need >= 4"):
            detect_knee([0.0, 1.0, 2.0, 3.0], [1.0, 2.0])

    def test_ci_bounds(self) -> None:
        """CI bounds should be within the x range."""
        x = [float(i) for i in range(20)]
        y = [1.0 / (1.0 + math.exp(-(xi - 10.0))) for xi in x]  # Sigmoid
        result = detect_knee(x, y, seed=42)

        assert result.ci_lower >= x[0]
        assert result.ci_upper <= x[-1]
        assert result.ci_lower <= result.ci_upper

    def test_non_uniform_spacing(self) -> None:
        """Non-uniform x spacing handled correctly."""
        x = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        y = [10.0, 9.0, 5.0, 3.0, 2.5, 2.3, 2.2]
        result = detect_knee(x, y, seed=42)
        assert isinstance(result.x_knee, float)

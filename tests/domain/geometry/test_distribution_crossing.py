# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import pytest

from modelcypher.core.domain.geometry.distribution_crossing import (
    CrossingResult,
    find_distribution_crossing,
)


class TestFindDistributionCrossing:
    def test_well_separated_groups(self) -> None:
        """Clearly separated groups → stable boundary between them.

        Uses 20 samples per group for reliable bootstrap stability.
        """
        group_a = [float(i) for i in range(1, 21)]       # 1..20
        group_b = [float(i) for i in range(30, 50)]       # 30..49
        result = find_distribution_crossing(group_a, group_b, seed=42)

        assert result.boundary >= 20.0
        assert result.boundary <= 30.0
        assert result.is_stable is True
        assert result.auroc > 0.9
        assert result.false_alarm_rate < 0.5
        assert result.miss_rate < 0.5
        assert result.n_a == 20
        assert result.n_b == 20

    def test_overlapping_groups(self) -> None:
        """Heavily overlapping groups → boundary exists but AUROC near 0.5."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [2.0, 3.0, 4.0, 5.0, 6.0]
        result = find_distribution_crossing(group_a, group_b, seed=42)

        assert isinstance(result.boundary, float)
        assert result.auroc < 0.9  # Not well separated
        assert result.n_a == 5
        assert result.n_b == 5

    def test_identical_groups(self) -> None:
        """Identical groups → boundary at some midpoint, AUROC = 0.5."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = find_distribution_crossing(group_a, group_b, seed=42)

        assert result.auroc == pytest.approx(0.5, abs=0.05)

    def test_minimum_samples(self) -> None:
        """Two samples per group — minimum viable input."""
        result = find_distribution_crossing([1.0, 2.0], [5.0, 6.0], seed=42)
        assert result.boundary >= 2.0
        assert result.boundary <= 5.0

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="Both groups need >= 2"):
            find_distribution_crossing([1.0], [2.0, 3.0])
        with pytest.raises(ValueError, match="Both groups need >= 2"):
            find_distribution_crossing([1.0, 2.0], [3.0])

    def test_perfect_separation_auroc(self) -> None:
        """Non-overlapping groups → AUROC = 1.0."""
        group_a = list(range(10))
        group_b = list(range(100, 110))
        result = find_distribution_crossing(
            [float(x) for x in group_a],
            [float(x) for x in group_b],
            seed=42,
        )
        assert result.auroc == pytest.approx(1.0)

    def test_bootstrap_ci_covers_point_estimate(self) -> None:
        """CI should generally contain the point estimate."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        group_b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        result = find_distribution_crossing(group_a, group_b, seed=42)

        assert result.ci_lower <= result.boundary
        assert result.ci_upper >= result.boundary

    def test_reversed_groups(self) -> None:
        """group_b < group_a — boundary still found."""
        group_a = [10.0, 11.0, 12.0, 13.0, 14.0]
        group_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = find_distribution_crossing(group_a, group_b, seed=42)

        assert isinstance(result.boundary, float)
        assert result.auroc < 0.1  # B < A → AUROC near 0 (convention: B > A)

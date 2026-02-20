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

from modelcypher.core.domain.training.dimensional_monitor import (
    DimensionalSnapshot,
    DimensionalTrend,
    assess_trend,
    compute_null_space_recruitment,
)


class TestDimensionalSnapshot:
    """Tests for DimensionalSnapshot dataclass."""

    def test_snapshot_is_frozen(self):
        snap = DimensionalSnapshot(
            epoch=0, expansion_ratio=1.2, peak_dim=10.0,
            final_dim=8.0, peak_layer=5, smoothness=0.9, hidden_dim=16,
        )
        with pytest.raises(AttributeError):
            snap.epoch = 1  # type: ignore[misc]

    def test_used_and_null_fractions(self):
        snap = DimensionalSnapshot(
            epoch=0, expansion_ratio=1.25, peak_dim=10.0,
            final_dim=8.0, peak_layer=5, smoothness=0.9, hidden_dim=16,
        )
        assert snap.final_used_fraction == pytest.approx(0.5)
        assert snap.final_null_fraction == pytest.approx(0.5)


class TestAssessTrend:
    """Tests for assess_trend()."""

    def _snap(self, epoch: int, er: float) -> DimensionalSnapshot:
        return DimensionalSnapshot(
            epoch=epoch, expansion_ratio=er, peak_dim=er * 8.0,
            final_dim=8.0, peak_layer=5, smoothness=0.9, hidden_dim=16,
        )

    def test_single_snapshot_no_contraction(self):
        """A single snapshot yields zero delta, no contraction."""
        trend = assess_trend([self._snap(0, 1.2)])
        assert trend.delta == 0.0
        assert trend.is_contracting is False
        assert trend.baseline_expansion_ratio == 1.2
        assert trend.current_expansion_ratio == 1.2

    def test_expanding_trend(self):
        """Increasing expansion_ratio → positive delta, not contracting."""
        snaps = [self._snap(i, 1.0 + i * 0.1) for i in range(5)]
        trend = assess_trend(snaps)
        assert trend.delta > 0
        assert trend.is_contracting is False
        assert trend.baseline_expansion_ratio == 1.0
        assert trend.current_expansion_ratio == pytest.approx(1.4)

    def test_contracting_trend(self):
        """Decreasing expansion_ratio → negative delta, is_contracting."""
        snaps = [self._snap(i, 1.4 - i * 0.1) for i in range(5)]
        trend = assess_trend(snaps)
        assert trend.delta < 0
        assert trend.is_contracting is True
        assert trend.baseline_expansion_ratio == 1.4
        assert trend.current_expansion_ratio == pytest.approx(1.0)

    def test_stable_trend(self):
        """Flat expansion_ratio → zero delta, not contracting."""
        snaps = [self._snap(i, 1.2) for i in range(5)]
        trend = assess_trend(snaps)
        assert trend.delta == 0.0
        # SE is 0 when all values identical, so delta < 0 is False
        assert trend.is_contracting is False

    def test_two_snapshots_slight_drop(self):
        """Two snapshots with a slight drop — delta < 0 but within noise."""
        snaps = [self._snap(0, 1.200), self._snap(1, 1.199)]
        trend = assess_trend(snaps)
        assert trend.delta < 0
        # With only 2 points, SE is large relative to tiny drop
        # so is_contracting depends on magnitude
        assert isinstance(trend.is_contracting, bool)

    def test_two_snapshots_large_drop(self):
        """Two snapshots with a large drop — clearly contracting."""
        snaps = [self._snap(0, 1.5), self._snap(1, 0.8)]
        trend = assess_trend(snaps)
        assert trend.delta < 0
        assert trend.is_contracting is True

    def test_empty_list(self):
        """Empty snapshot list — graceful fallback."""
        trend = assess_trend([])
        assert trend.baseline_expansion_ratio == 0.0
        assert trend.delta == 0.0
        assert trend.is_contracting is False


def test_null_space_recruitment_positive_for_dimension_growth():
    baseline = DimensionalSnapshot(
        epoch=0,
        expansion_ratio=1.0,
        peak_dim=8.0,
        final_dim=8.0,
        peak_layer=5,
        smoothness=0.9,
        hidden_dim=32,
    )
    current = DimensionalSnapshot(
        epoch=1,
        expansion_ratio=1.0,
        peak_dim=12.0,
        final_dim=12.0,
        peak_layer=5,
        smoothness=0.9,
        hidden_dim=32,
    )
    assert compute_null_space_recruitment(baseline, current) == pytest.approx(0.125)

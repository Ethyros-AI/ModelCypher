# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for entropy_math.py — EntropyMath contract tests.

Covers the canonical entropy math used in training regime selection:
  EM1 — Empty trajectory handling.
  EM2 — Mean is arithmetic mean.
  EM3 — Variance is Bessel-corrected sample variance (ddof=1).
  EM4 — first/last token entropy and delta.
  EM5 — Cooling/heating direction.
  EM6 — compute_delta_h.
  EM7 — sample_mean.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain.entropy.entropy_math import EntropyMath

# ---------------------------------------------------------------------------
# EM1: Empty Trajectory
# ---------------------------------------------------------------------------

class TestEmptyTrajectory:
    def test_empty_returns_invalid_stats(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([])
        assert not stats.is_valid
        assert stats.trajectory_length == 0
        assert stats.mean_entropy == pytest.approx(0.0)

    def test_empty_with_fallback(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([], fallback_entropy=5.0)
        assert stats.mean_entropy == pytest.approx(5.0)
        assert stats.first_token_entropy == pytest.approx(5.0)
        assert stats.last_token_entropy == pytest.approx(5.0)
        assert stats.trajectory_length == 0


# ---------------------------------------------------------------------------
# EM2: Mean Computation
# ---------------------------------------------------------------------------

class TestMeanComputation:
    def test_mean_of_two_values(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([1.0, 3.0])
        assert stats.mean_entropy == pytest.approx(2.0, abs=1e-5)

    def test_mean_of_constant_trajectory(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([2.0, 2.0, 2.0])
        assert stats.mean_entropy == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# EM3: Variance Computation (Bessel-corrected, ddof=1)
# ---------------------------------------------------------------------------

class TestVarianceComputation:
    def test_variance_two_values(self) -> None:
        """[1, 3]: mean=2, deviations [-1, 1], Bessel-corrected: sum/1 = 2.0."""
        stats = EntropyMath.calculate_trajectory_stats([1.0, 3.0])
        assert stats.entropy_variance == pytest.approx(2.0, abs=1e-5)

    def test_variance_constant_is_zero(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([5.0, 5.0, 5.0])
        assert stats.entropy_variance == pytest.approx(0.0, abs=1e-7)

    def test_variance_single_element_is_zero(self) -> None:
        """Single element → ddof=1 path returns 0.0 (n-1=0 guard)."""
        stats = EntropyMath.calculate_trajectory_stats([3.0])
        assert stats.entropy_variance == pytest.approx(0.0, abs=1e-7)

    def test_sample_variance_ddof0_is_population(self) -> None:
        """ddof=0: [1, 3] → ((1-2)²+(3-2)²)/2 = 1.0 (population variance)."""
        v = EntropyMath.sample_variance([1.0, 3.0], ddof=0)
        assert v == pytest.approx(1.0, abs=1e-5)

    def test_sample_variance_ddof1_matches_trajectory_stats(self) -> None:
        """sample_variance(ddof=1) and calculate_trajectory_stats agree."""
        data = [1.0, 2.0, 4.0, 3.0]
        stats = EntropyMath.calculate_trajectory_stats(data)
        direct = EntropyMath.sample_variance(data, ddof=1)
        assert direct == pytest.approx(stats.entropy_variance, abs=1e-5)


# ---------------------------------------------------------------------------
# EM4: First/Last Token Entropy and Delta
# ---------------------------------------------------------------------------

class TestFirstLastToken:
    def test_first_last_token(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([1.0, 2.0, 3.0])
        assert stats.first_token_entropy == pytest.approx(1.0)
        assert stats.last_token_entropy == pytest.approx(3.0)
        assert stats.entropy_delta == pytest.approx(2.0, abs=1e-5)

    def test_single_element_delta_zero(self) -> None:
        """length < 2 → entropy_delta = 0.0 (guard in property)."""
        stats = EntropyMath.calculate_trajectory_stats([5.0])
        assert stats.entropy_delta == pytest.approx(0.0)

    def test_trajectory_length_matches_input(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.trajectory_length == 5


# ---------------------------------------------------------------------------
# EM5: Cooling / Heating Direction
# ---------------------------------------------------------------------------

class TestCoolingHeating:
    def test_decreasing_is_cooling(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([3.0, 2.0, 1.0])
        assert stats.is_cooling is True
        assert stats.is_heating is False

    def test_increasing_is_heating(self) -> None:
        stats = EntropyMath.calculate_trajectory_stats([1.0, 2.0, 3.0])
        assert stats.is_heating is True
        assert stats.is_cooling is False

    def test_flat_is_neither(self) -> None:
        """Constant entropy: delta=0, not cooling, not heating."""
        stats = EntropyMath.calculate_trajectory_stats([2.0, 2.0])
        assert stats.is_cooling is False
        assert stats.is_heating is False


# ---------------------------------------------------------------------------
# EM6: compute_delta_h
# ---------------------------------------------------------------------------

class TestDeltaH:
    def test_no_baseline_returns_none(self) -> None:
        assert EntropyMath.compute_delta_h(5.0, None) is None

    def test_delta_h_value(self) -> None:
        result = EntropyMath.compute_delta_h(5.0, 3.0)
        assert result == pytest.approx(2.0)

    def test_delta_h_negative(self) -> None:
        result = EntropyMath.compute_delta_h(1.0, 4.0)
        assert result == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# EM7: sample_mean
# ---------------------------------------------------------------------------

class TestSampleMean:
    def test_empty_mean_is_zero(self) -> None:
        assert EntropyMath.sample_mean([]) == pytest.approx(0.0)

    def test_mean_known_values(self) -> None:
        assert EntropyMath.sample_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0, abs=1e-5)

    def test_mean_single_element(self) -> None:
        assert EntropyMath.sample_mean([7.5]) == pytest.approx(7.5, abs=1e-6)

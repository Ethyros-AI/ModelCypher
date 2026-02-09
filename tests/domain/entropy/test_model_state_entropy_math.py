# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from datetime import datetime

import pytest

from modelcypher.core.domain.entropy.entropy_math import EntropyMath
from modelcypher.core.domain.entropy.model_state import ModelEntropyBaseline, StateEntropyTransition


def test_model_entropy_baseline_z_score_and_normalized() -> None:
    baseline = ModelEntropyBaseline(mean=10.0, std=2.0, max_theoretical=20.0, model_name="demo")

    assert baseline.z_score(14.0) == pytest.approx(2.0)
    assert baseline.normalized(10.0) == pytest.approx(0.5)


def test_model_entropy_baseline_handles_near_zero_denominators() -> None:
    zero_std = ModelEntropyBaseline(mean=10.0, std=0.0, max_theoretical=20.0)
    zero_max = ModelEntropyBaseline(mean=10.0, std=2.0, max_theoretical=0.0)

    assert zero_std.z_score(20.0) == 0.0
    assert zero_max.normalized(10.0) == 0.0


def test_state_entropy_transition_deltas_zscore_and_description() -> None:
    baseline = ModelEntropyBaseline(mean=4.0, std=2.0, max_theoretical=10.0)
    transition = StateEntropyTransition(
        from_entropy=5.0,
        from_variance=1.5,
        to_entropy=7.0,
        to_variance=2.0,
        token_index=9,
        timestamp=datetime.utcnow(),
        reason="test",
    )

    assert transition.entropy_delta == pytest.approx(2.0)
    assert transition.variance_delta == pytest.approx(0.5)
    assert transition.z_score_delta(baseline) == pytest.approx(1.0)
    assert "increased" in transition.description
    assert "token 9" in transition.description


def test_entropy_math_calculate_trajectory_stats_handles_empty_and_non_empty() -> None:
    empty = EntropyMath.calculate_trajectory_stats([], fallback_entropy=3.0)
    stats = EntropyMath.calculate_trajectory_stats([1.0, 2.0, 3.0])
    singleton = EntropyMath.calculate_trajectory_stats([5.0])

    assert empty.mean_entropy == 3.0
    assert empty.entropy_variance == 0.0
    assert empty.trajectory_length == 0

    assert stats.mean_entropy == pytest.approx(2.0)
    assert stats.entropy_variance == pytest.approx(1.0)
    assert stats.entropy_delta == pytest.approx(2.0)
    assert stats.is_heating is True
    assert stats.is_cooling is False
    assert stats.is_valid is True

    assert singleton.entropy_delta == 0.0
    assert singleton.entropy_variance == 0.0


def test_entropy_math_sample_statistics_and_delta_h() -> None:
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    assert EntropyMath.sample_mean(values) == pytest.approx(5.0)
    assert EntropyMath.sample_variance(values, ddof=1) == pytest.approx(32.0 / 7.0)
    assert EntropyMath.sample_std(values, ddof=1) == pytest.approx((32.0 / 7.0) ** 0.5)
    assert EntropyMath.compute_delta_h(6.0, 5.5) == pytest.approx(0.5)
    assert EntropyMath.compute_delta_h(6.0, None) is None


def test_entropy_math_percentiles() -> None:
    values = [1.0, 5.0, 9.0, 13.0]

    assert EntropyMath.percentile(values, 0) == 1.0
    assert EntropyMath.percentile(values, 50) == 9.0
    assert EntropyMath.percentile(values, 95) == 13.0
    assert EntropyMath.percentile([], 50) == 0.0

    summary = EntropyMath.entropy_percentiles(values)
    assert set(summary.keys()) == {25, 50, 75, 95}
    assert summary[50] == 9.0


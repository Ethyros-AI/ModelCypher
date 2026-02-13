from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from modelcypher.core.domain.entropy.model_state import (
    ModelEntropyBaseline,
    StateEntropyTransition,
)


class TestStateEntropyTransitionInstantiation:
    """Verify StateEntropyTransition can be created with required and optional fields."""

    def test_required_fields_only(self) -> None:
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=1.0,
            to_entropy=5.0,
            to_variance=2.0,
            token_index=7,
        )
        assert t.from_entropy == 3.0
        assert t.from_variance == 1.0
        assert t.to_entropy == 5.0
        assert t.to_variance == 2.0
        assert t.token_index == 7
        assert t.reason is None
        # timestamp is auto-populated
        assert isinstance(t.timestamp, datetime)

    def test_with_explicit_timestamp_and_reason(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, 0)
        t = StateEntropyTransition(
            from_entropy=2.0,
            from_variance=0.5,
            to_entropy=4.0,
            to_variance=1.5,
            token_index=3,
            timestamp=ts,
            reason="spike detected",
        )
        assert t.timestamp == ts
        assert t.reason == "spike detected"


class TestStateEntropyTransitionEntropyDelta:
    """Verify entropy_delta property (to_entropy - from_entropy)."""

    def test_positive_delta_increasing_uncertainty(self) -> None:
        t = StateEntropyTransition(
            from_entropy=2.0,
            from_variance=1.0,
            to_entropy=6.0,
            to_variance=1.0,
            token_index=0,
        )
        assert t.entropy_delta == pytest.approx(4.0)

    def test_negative_delta_decreasing_uncertainty(self) -> None:
        t = StateEntropyTransition(
            from_entropy=8.0,
            from_variance=1.0,
            to_entropy=3.0,
            to_variance=1.0,
            token_index=1,
        )
        assert t.entropy_delta == pytest.approx(-5.0)

    def test_zero_delta_no_change(self) -> None:
        t = StateEntropyTransition(
            from_entropy=5.0,
            from_variance=1.0,
            to_entropy=5.0,
            to_variance=1.0,
            token_index=2,
        )
        assert t.entropy_delta == pytest.approx(0.0)


class TestStateEntropyTransitionVarianceDelta:
    """Verify variance_delta property (to_variance - from_variance)."""

    def test_variance_delta_positive(self) -> None:
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=1.0,
            to_entropy=5.0,
            to_variance=3.5,
            token_index=0,
        )
        assert t.variance_delta == pytest.approx(2.5)

    def test_variance_delta_negative(self) -> None:
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=4.0,
            to_entropy=5.0,
            to_variance=1.0,
            token_index=0,
        )
        assert t.variance_delta == pytest.approx(-3.0)

    def test_variance_delta_zero(self) -> None:
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=2.0,
            to_entropy=5.0,
            to_variance=2.0,
            token_index=0,
        )
        assert t.variance_delta == pytest.approx(0.0)


class TestStateEntropyTransitionDescription:
    """Verify description property contains direction words and token index."""

    def test_increased_description(self) -> None:
        t = StateEntropyTransition(
            from_entropy=2.0,
            from_variance=1.0,
            to_entropy=5.0,
            to_variance=1.0,
            token_index=12,
        )
        desc = t.description
        assert "increased" in desc
        assert "token 12" in desc

    def test_decreased_description(self) -> None:
        t = StateEntropyTransition(
            from_entropy=7.0,
            from_variance=1.0,
            to_entropy=3.0,
            to_variance=1.0,
            token_index=5,
        )
        desc = t.description
        assert "decreased" in desc
        assert "token 5" in desc

    def test_unchanged_description(self) -> None:
        t = StateEntropyTransition(
            from_entropy=4.0,
            from_variance=1.0,
            to_entropy=4.0,
            to_variance=1.0,
            token_index=0,
        )
        desc = t.description
        assert "unchanged" in desc
        assert "token 0" in desc


class TestStateEntropyTransitionZScoreDelta:
    """Verify z_score_delta divides entropy_delta by baseline std."""

    def test_z_score_delta_with_normal_baseline(self) -> None:
        baseline = ModelEntropyBaseline(mean=5.0, std=2.0, max_theoretical=15.0)
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=1.0,
            to_entropy=7.0,
            to_variance=1.0,
            token_index=0,
        )
        # entropy_delta = 4.0, std = 2.0 -> z_score_delta = 2.0
        assert t.z_score_delta(baseline) == pytest.approx(2.0)

    def test_z_score_delta_negative(self) -> None:
        baseline = ModelEntropyBaseline(mean=5.0, std=2.5, max_theoretical=15.0)
        t = StateEntropyTransition(
            from_entropy=8.0,
            from_variance=1.0,
            to_entropy=3.0,
            to_variance=1.0,
            token_index=0,
        )
        # entropy_delta = -5.0, std = 2.5 -> z_score_delta = -2.0
        assert t.z_score_delta(baseline) == pytest.approx(-2.0)

    def test_z_score_delta_near_zero_std_returns_zero(self) -> None:
        baseline = ModelEntropyBaseline(mean=5.0, std=0.0, max_theoretical=15.0)
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=1.0,
            to_entropy=10.0,
            to_variance=1.0,
            token_index=0,
        )
        assert t.z_score_delta(baseline) == 0.0

    def test_z_score_delta_with_tiny_std_returns_zero(self) -> None:
        baseline = ModelEntropyBaseline(mean=5.0, std=1e-20, max_theoretical=15.0)
        t = StateEntropyTransition(
            from_entropy=3.0,
            from_variance=1.0,
            to_entropy=10.0,
            to_variance=1.0,
            token_index=0,
        )
        # std < eps, should return 0.0
        assert t.z_score_delta(baseline) == 0.0


class TestStateEntropyTransitionReasonField:
    """Verify optional reason field."""

    def test_reason_none_by_default(self) -> None:
        t = StateEntropyTransition(
            from_entropy=1.0,
            from_variance=0.5,
            to_entropy=2.0,
            to_variance=0.5,
            token_index=0,
        )
        assert t.reason is None

    def test_reason_stores_string(self) -> None:
        t = StateEntropyTransition(
            from_entropy=1.0,
            from_variance=0.5,
            to_entropy=2.0,
            to_variance=0.5,
            token_index=0,
            reason="calibration drift",
        )
        assert t.reason == "calibration drift"


class TestStateEntropyTransitionFrozen:
    """Verify frozen dataclass cannot be mutated."""

    def test_cannot_mutate_field(self) -> None:
        t = StateEntropyTransition(
            from_entropy=1.0,
            from_variance=0.5,
            to_entropy=2.0,
            to_variance=0.5,
            token_index=0,
        )
        with pytest.raises(FrozenInstanceError):
            t.from_entropy = 99.0  # type: ignore[misc]

    def test_cannot_mutate_token_index(self) -> None:
        t = StateEntropyTransition(
            from_entropy=1.0,
            from_variance=0.5,
            to_entropy=2.0,
            to_variance=0.5,
            token_index=0,
        )
        with pytest.raises(FrozenInstanceError):
            t.token_index = 42  # type: ignore[misc]

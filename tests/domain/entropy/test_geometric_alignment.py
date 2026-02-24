from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelcypher.core.domain.entropy.geometric_alignment import (
    AlignmentDecision,
    OscillationPattern,
    SentinelSample,
    SessionTelemetry,
)

# =============================================================================
# SentinelSample
# =============================================================================


class TestSentinelSampleProperties:
    """Verify SentinelSample computed properties for dip classification."""

    def test_true_dip_negative_delta_and_below_ceiling(self) -> None:
        s = SentinelSample(
            token_index=0,
            entropy=3.0,
            delta_h=-1.0,
            is_spike=False,
            is_negative_delta=True,
            is_below_ceiling=True,
        )
        assert s.is_true_dip is True
        assert s.is_pseudo_dip is False
        assert s.is_any_dip is True

    def test_pseudo_dip_negative_delta_above_ceiling(self) -> None:
        s = SentinelSample(
            token_index=1,
            entropy=8.0,
            delta_h=-2.0,
            is_spike=True,
            is_negative_delta=True,
            is_below_ceiling=False,
        )
        assert s.is_true_dip is False
        assert s.is_pseudo_dip is True
        assert s.is_any_dip is True

    def test_no_dip_positive_delta(self) -> None:
        s = SentinelSample(
            token_index=2,
            entropy=5.0,
            delta_h=1.5,
            is_spike=False,
            is_negative_delta=False,
            is_below_ceiling=True,
        )
        assert s.is_true_dip is False
        assert s.is_pseudo_dip is False
        assert s.is_any_dip is False

    def test_no_dip_positive_delta_above_ceiling(self) -> None:
        s = SentinelSample(
            token_index=3,
            entropy=10.0,
            delta_h=0.5,
            is_spike=False,
            is_negative_delta=False,
            is_below_ceiling=False,
        )
        assert s.is_true_dip is False
        assert s.is_pseudo_dip is False
        assert s.is_any_dip is False

    def test_no_dip_zero_delta_below_ceiling(self) -> None:
        s = SentinelSample(
            token_index=4,
            entropy=2.0,
            delta_h=0.0,
            is_spike=False,
            is_negative_delta=False,
            is_below_ceiling=True,
        )
        assert s.is_true_dip is False
        assert s.is_pseudo_dip is False
        assert s.is_any_dip is False

    def test_spike_flag_independent_of_dip(self) -> None:
        s = SentinelSample(
            token_index=5,
            entropy=4.0,
            delta_h=3.0,
            is_spike=True,
            is_negative_delta=False,
            is_below_ceiling=True,
        )
        assert s.is_spike is True
        assert s.is_any_dip is False


# =============================================================================
# OscillationPattern
# =============================================================================


class TestOscillationPatternIsUnstable:
    """Verify is_unstable property with different signal combinations."""

    def test_all_zeros_is_stable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=0,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.0,
        )
        assert p.is_unstable is False

    def test_sign_changes_makes_unstable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=3,
            window_spike_count=0,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.3,
        )
        assert p.is_unstable is True

    def test_spike_count_makes_unstable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=1,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.1,
        )
        assert p.is_unstable is True

    def test_w_shape_makes_unstable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=0,
            w_shape_count=2,
            failed_deflections=0,
            severity=0.2,
        )
        assert p.is_unstable is True

    def test_failed_deflections_makes_unstable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=0,
            w_shape_count=0,
            failed_deflections=1,
            severity=0.1,
        )
        assert p.is_unstable is True

    def test_multiple_signals_all_unstable(self) -> None:
        p = OscillationPattern(
            window_sign_changes=2,
            window_spike_count=3,
            w_shape_count=1,
            failed_deflections=1,
            severity=0.8,
        )
        assert p.is_unstable is True

    def test_severity_alone_does_not_make_unstable(self) -> None:
        """severity is a continuous measure; is_unstable checks discrete signals."""
        p = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=0,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.99,
        )
        assert p.is_unstable is False


# =============================================================================
# AlignmentDecision
# =============================================================================


class TestAlignmentDecision:
    """Verify AlignmentDecision instantiation with component objects."""

    def test_instantiation(self) -> None:
        sentinel = SentinelSample(
            token_index=0,
            entropy=5.0,
            delta_h=0.5,
            is_spike=False,
            is_negative_delta=False,
            is_below_ceiling=True,
        )
        pattern = OscillationPattern(
            window_sign_changes=0,
            window_spike_count=0,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.0,
        )
        decision = AlignmentDecision(
            sentinel=sentinel,
            pattern=pattern,
            consecutive_oscillations=0,
            above_ceiling=False,
        )
        assert decision.sentinel is sentinel
        assert decision.pattern is pattern
        assert decision.consecutive_oscillations == 0
        assert decision.above_ceiling is False

    def test_above_ceiling_flag(self) -> None:
        sentinel = SentinelSample(
            token_index=10,
            entropy=12.0,
            delta_h=2.0,
            is_spike=True,
            is_negative_delta=False,
            is_below_ceiling=False,
        )
        pattern = OscillationPattern(
            window_sign_changes=1,
            window_spike_count=1,
            w_shape_count=0,
            failed_deflections=0,
            severity=0.5,
        )
        decision = AlignmentDecision(
            sentinel=sentinel,
            pattern=pattern,
            consecutive_oscillations=3,
            above_ceiling=True,
        )
        assert decision.above_ceiling is True
        assert decision.consecutive_oscillations == 3


# =============================================================================
# SessionTelemetry
# =============================================================================


class TestSessionTelemetryInstantiation:
    """Verify SessionTelemetry frozen dataclass."""

    def test_all_fields(self) -> None:
        t = SessionTelemetry(
            tokens_observed=100,
            tokens_above_ceiling=15,
            spike_count=7,
            max_severity=0.65,
            consecutive_oscillations=2,
        )
        assert t.tokens_observed == 100
        assert t.tokens_above_ceiling == 15
        assert t.spike_count == 7
        assert t.max_severity == pytest.approx(0.65)
        assert t.consecutive_oscillations == 2


class TestSessionTelemetryToDict:
    """Verify to_dict produces correct dictionary representation."""

    def test_to_dict_keys(self) -> None:
        t = SessionTelemetry(
            tokens_observed=50,
            tokens_above_ceiling=5,
            spike_count=3,
            max_severity=0.4,
            consecutive_oscillations=1,
        )
        d = t.to_dict()
        expected_keys = {
            "tokens_observed",
            "tokens_above_ceiling",
            "spike_count",
            "max_severity",
            "consecutive_oscillations",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_fields(self) -> None:
        t = SessionTelemetry(
            tokens_observed=200,
            tokens_above_ceiling=20,
            spike_count=10,
            max_severity=0.85,
            consecutive_oscillations=4,
        )
        d = t.to_dict()
        assert d["tokens_observed"] == 200
        assert d["tokens_above_ceiling"] == 20
        assert d["spike_count"] == 10
        assert d["max_severity"] == pytest.approx(0.85)
        assert d["consecutive_oscillations"] == 4

    def test_to_dict_round_trip_values(self) -> None:
        t = SessionTelemetry(
            tokens_observed=0,
            tokens_above_ceiling=0,
            spike_count=0,
            max_severity=0.0,
            consecutive_oscillations=0,
        )
        d = t.to_dict()
        t2 = SessionTelemetry(**d)
        assert t2 == t


class TestSessionTelemetryFrozen:
    """Verify SessionTelemetry cannot be mutated."""

    def test_cannot_mutate_tokens_observed(self) -> None:
        t = SessionTelemetry(
            tokens_observed=50,
            tokens_above_ceiling=5,
            spike_count=3,
            max_severity=0.4,
            consecutive_oscillations=1,
        )
        with pytest.raises(FrozenInstanceError):
            t.tokens_observed = 999  # type: ignore[misc]

    def test_cannot_mutate_max_severity(self) -> None:
        t = SessionTelemetry(
            tokens_observed=50,
            tokens_above_ceiling=5,
            spike_count=3,
            max_severity=0.4,
            consecutive_oscillations=1,
        )
        with pytest.raises(FrozenInstanceError):
            t.max_severity = 1.0  # type: ignore[misc]

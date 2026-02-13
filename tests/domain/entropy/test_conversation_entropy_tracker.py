# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

from uuid import UUID

import pytest

from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
    ConversationAssessment,
    ConversationEntropyBaseline,
    ConversationEntropyTracker,
    TurnSummary,
)


class TestConversationEntropyBaseline:
    """Tests for ConversationEntropyBaseline dataclass and classmethods."""

    def test_from_samples_with_valid_data(self) -> None:
        """from_samples computes mean and std from 2+ samples."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])

        assert baseline.delta_mean == pytest.approx(3.0, abs=1e-5)
        assert baseline.delta_std_dev > 0.0

    def test_from_samples_minimum_two(self) -> None:
        """from_samples works with exactly 2 samples."""
        baseline = ConversationEntropyBaseline.from_samples([10.0, 20.0])

        assert baseline.delta_mean == pytest.approx(15.0, abs=1e-5)
        assert baseline.delta_std_dev > 0.0

    def test_from_samples_less_than_two_raises_value_error(self) -> None:
        """from_samples with fewer than 2 samples raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            ConversationEntropyBaseline.from_samples([1.0])

    def test_from_samples_empty_raises_value_error(self) -> None:
        """from_samples with empty list raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            ConversationEntropyBaseline.from_samples([])

    def test_z_score_with_normal_data(self) -> None:
        """z_score computes correct z-score for values relative to baseline."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 3.0, 5.0, 7.0, 9.0])
        # Mean = 5.0, std > 0

        z_at_mean = baseline.z_score(5.0)
        assert z_at_mean == pytest.approx(0.0, abs=1e-4)

        z_above = baseline.z_score(10.0)
        assert z_above > 0.0

        z_below = baseline.z_score(0.0)
        assert z_below < 0.0

    def test_z_score_with_near_zero_std(self) -> None:
        """z_score returns 0.0 when std is near zero and value equals mean."""
        # Identical samples produce zero std
        baseline = ConversationEntropyBaseline.from_samples([5.0, 5.0, 5.0, 5.0])

        assert baseline.delta_std_dev == pytest.approx(0.0, abs=1e-6)
        z = baseline.z_score(5.0)
        assert z == pytest.approx(0.0, abs=1e-5)

    def test_frozen_dataclass(self) -> None:
        """ConversationEntropyBaseline is frozen (immutable)."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 2.0, 3.0])
        with pytest.raises(AttributeError):
            baseline.delta_mean = 999.0  # type: ignore[misc]


class TestConversationEntropyTrackerInitialState:
    """Tests for ConversationEntropyTracker initial state."""

    def test_initial_turn_count_is_zero(self) -> None:
        """New tracker has turn_count of 0."""
        tracker = ConversationEntropyTracker()
        assert tracker.current_turn_count == 0

    def test_initial_conversation_id_is_none(self) -> None:
        """New tracker has no conversation_id."""
        tracker = ConversationEntropyTracker()
        assert tracker.current_conversation_id is None

    def test_initial_turn_summaries_empty(self) -> None:
        """New tracker has no turn summaries."""
        tracker = ConversationEntropyTracker()
        assert tracker.all_turn_summaries == []

    def test_construction_with_baseline(self) -> None:
        """Tracker can be constructed with a baseline."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 2.0, 3.0])
        tracker = ConversationEntropyTracker(baseline=baseline)
        assert tracker.current_turn_count == 0


class TestRecordTurn:
    """Tests for the record_turn method."""

    def test_first_turn_creates_conversation_id(self) -> None:
        """First record_turn assigns a conversation_id."""
        tracker = ConversationEntropyTracker()
        assert tracker.current_conversation_id is None

        tracker.record_turn(
            token_count=100,
            avg_delta=0.5,
            max_anomaly_score=0.1,
            anomaly_count=0,
        )

        assert tracker.current_conversation_id is not None
        assert isinstance(tracker.current_conversation_id, UUID)

    def test_returns_conversation_assessment(self) -> None:
        """record_turn returns a ConversationAssessment."""
        tracker = ConversationEntropyTracker()

        assessment = tracker.record_turn(
            token_count=50,
            avg_delta=0.3,
            max_anomaly_score=0.0,
            anomaly_count=0,
        )

        assert isinstance(assessment, ConversationAssessment)

    def test_assessment_turn_count_matches(self) -> None:
        """ConversationAssessment turn_count reflects recorded turns."""
        tracker = ConversationEntropyTracker()

        assessment1 = tracker.record_turn(
            token_count=50,
            avg_delta=0.3,
            max_anomaly_score=0.0,
            anomaly_count=0,
        )
        assert assessment1.turn_count == 1

        assessment2 = tracker.record_turn(
            token_count=75,
            avg_delta=0.4,
            max_anomaly_score=0.1,
            anomaly_count=1,
        )
        assert assessment2.turn_count == 2

    def test_multiple_turns_accumulate_summaries(self) -> None:
        """Multiple record_turn calls accumulate turn summaries."""
        tracker = ConversationEntropyTracker()

        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        tracker.record_turn(token_count=20, avg_delta=0.2, max_anomaly_score=0.5, anomaly_count=1)
        tracker.record_turn(token_count=30, avg_delta=0.3, max_anomaly_score=0.8, anomaly_count=2)

        assert tracker.current_turn_count == 3
        summaries = tracker.all_turn_summaries
        assert len(summaries) == 3
        assert all(isinstance(s, TurnSummary) for s in summaries)

    def test_conversation_id_stable_across_turns(self) -> None:
        """conversation_id remains the same across multiple turns."""
        tracker = ConversationEntropyTracker()

        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        id_after_first = tracker.current_conversation_id

        tracker.record_turn(token_count=20, avg_delta=0.2, max_anomaly_score=0.0, anomaly_count=0)
        id_after_second = tracker.current_conversation_id

        assert id_after_first == id_after_second

    def test_assessment_anomaly_fields(self) -> None:
        """ConversationAssessment correctly aggregates anomaly fields."""
        tracker = ConversationEntropyTracker()

        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.2, anomaly_count=1)
        assessment = tracker.record_turn(
            token_count=20, avg_delta=0.3, max_anomaly_score=0.9, anomaly_count=3
        )

        assert assessment.anomaly_count == 4  # 1 + 3
        assert assessment.max_anomaly_score == pytest.approx(0.9, abs=1e-5)
        assert assessment.anomaly_rate == pytest.approx(4.0 / 2.0, abs=1e-5)


class TestConversationAssessmentFields:
    """Tests for ConversationAssessment field population."""

    def test_all_fields_populated_single_turn(self) -> None:
        """All ConversationAssessment fields are populated after one turn."""
        tracker = ConversationEntropyTracker()

        assessment = tracker.record_turn(
            token_count=100,
            avg_delta=0.5,
            max_anomaly_score=0.1,
            anomaly_count=0,
        )

        assert assessment.conversation_id is not None
        assert assessment.turn_count == 1
        assert isinstance(assessment.mean_delta, float)
        assert isinstance(assessment.std_delta, float)
        assert isinstance(assessment.oscillation_amplitude, float)
        assert isinstance(assessment.oscillation_frequency, float)
        assert isinstance(assessment.cumulative_drift, float)
        assert isinstance(assessment.anomaly_count, int)
        assert isinstance(assessment.anomaly_rate, float)
        assert isinstance(assessment.max_anomaly_score, float)
        assert isinstance(assessment.delta_change_mean, float)
        assert isinstance(assessment.delta_change_std, float)

    def test_all_fields_populated_multiple_turns(self) -> None:
        """All ConversationAssessment fields are populated after multiple turns."""
        tracker = ConversationEntropyTracker()

        for i in range(5):
            assessment = tracker.record_turn(
                token_count=50 + i * 10,
                avg_delta=0.1 * (i + 1),
                max_anomaly_score=0.05 * i,
                anomaly_count=i % 2,
            )

        assert assessment.turn_count == 5
        assert assessment.mean_delta > 0.0
        assert assessment.conversation_id is not None

    def test_delta_change_stats_require_multiple_turns(self) -> None:
        """delta_change_mean and delta_change_std are 0 with a single turn."""
        tracker = ConversationEntropyTracker()

        assessment = tracker.record_turn(
            token_count=50, avg_delta=0.5, max_anomaly_score=0.0, anomaly_count=0
        )

        assert assessment.delta_change_mean == pytest.approx(0.0)
        assert assessment.delta_change_std == pytest.approx(0.0)

    def test_delta_change_stats_with_two_turns(self) -> None:
        """delta_change_mean computed correctly with two turns."""
        tracker = ConversationEntropyTracker()

        tracker.record_turn(token_count=50, avg_delta=1.0, max_anomaly_score=0.0, anomaly_count=0)
        assessment = tracker.record_turn(
            token_count=50, avg_delta=3.0, max_anomaly_score=0.0, anomaly_count=0
        )

        # Change from 1.0 to 3.0 = delta of 2.0
        assert assessment.delta_change_mean == pytest.approx(2.0, abs=1e-4)

    def test_frozen_assessment(self) -> None:
        """ConversationAssessment is frozen (immutable)."""
        tracker = ConversationEntropyTracker()
        assessment = tracker.record_turn(
            token_count=50, avg_delta=0.5, max_anomaly_score=0.0, anomaly_count=0
        )
        with pytest.raises(AttributeError):
            assessment.turn_count = 999  # type: ignore[misc]


class TestReset:
    """Tests for the reset method."""

    def test_reset_clears_turn_count(self) -> None:
        """reset brings turn_count back to 0."""
        tracker = ConversationEntropyTracker()
        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        assert tracker.current_turn_count == 1

        tracker.reset()

        assert tracker.current_turn_count == 0

    def test_reset_clears_conversation_id(self) -> None:
        """reset clears the conversation_id to None."""
        tracker = ConversationEntropyTracker()
        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        assert tracker.current_conversation_id is not None

        tracker.reset()

        assert tracker.current_conversation_id is None

    def test_reset_clears_summaries(self) -> None:
        """reset clears all accumulated turn summaries."""
        tracker = ConversationEntropyTracker()
        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        tracker.record_turn(token_count=20, avg_delta=0.2, max_anomaly_score=0.0, anomaly_count=0)
        assert len(tracker.all_turn_summaries) == 2

        tracker.reset()

        assert tracker.all_turn_summaries == []

    def test_record_after_reset_creates_new_conversation(self) -> None:
        """Recording after reset creates a fresh conversation_id."""
        tracker = ConversationEntropyTracker()
        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        old_id = tracker.current_conversation_id

        tracker.reset()
        tracker.record_turn(token_count=10, avg_delta=0.1, max_anomaly_score=0.0, anomaly_count=0)
        new_id = tracker.current_conversation_id

        assert new_id is not None
        assert new_id != old_id


class TestTrackerWithBaseline:
    """Tests for tracker behavior when a baseline is provided."""

    def test_cumulative_drift_uses_baseline_z_score(self) -> None:
        """When baseline is provided, cumulative_drift uses baseline z_score."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        tracker = ConversationEntropyTracker(baseline=baseline)

        # Record a turn with avg_delta at the baseline mean
        assessment = tracker.record_turn(
            token_count=100,
            avg_delta=baseline.delta_mean,
            max_anomaly_score=0.0,
            anomaly_count=0,
        )

        # Drift should be ~0 since we are at the baseline mean
        assert assessment.cumulative_drift == pytest.approx(0.0, abs=1e-3)

    def test_cumulative_drift_nonzero_for_deviated_delta(self) -> None:
        """cumulative_drift is non-zero when avg_delta deviates from baseline."""
        baseline = ConversationEntropyBaseline.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        tracker = ConversationEntropyTracker(baseline=baseline)

        # Record a turn with avg_delta far from baseline mean
        assessment = tracker.record_turn(
            token_count=100,
            avg_delta=100.0,
            max_anomaly_score=0.0,
            anomaly_count=0,
        )

        assert abs(assessment.cumulative_drift) > 1.0


class TestTurnSummaryDataclass:
    """Tests for TurnSummary dataclass."""

    def test_turn_summary_fields(self) -> None:
        """TurnSummary stores turn-level measurements."""
        tracker = ConversationEntropyTracker()
        tracker.record_turn(
            token_count=42,
            avg_delta=0.77,
            max_anomaly_score=0.33,
            anomaly_count=2,
        )

        summaries = tracker.all_turn_summaries
        assert len(summaries) == 1

        ts = summaries[0]
        assert ts.turn_index == 0
        assert ts.token_count == 42
        assert ts.avg_delta == pytest.approx(0.77)
        assert ts.max_anomaly_score == pytest.approx(0.33)
        assert ts.anomaly_count == 2
        assert ts.timestamp is not None

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

"""Integration tests for entropy workflow components.

Validates that entropy measurement components operate together without
threshold-based classification.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.conversation_entropy_tracker import (
    ConversationEntropyBaseline,
    ConversationEntropyTracker,
)
from modelcypher.core.domain.entropy.entropy_window import (
    EntropyWindow,
    EntropyWindowConfig,
)
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
)
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


# =============================================================================
# Entropy Calculation Integration
# =============================================================================


class TestEntropyCalculationIntegration:
    """Tests for entropy calculation integration."""

    def test_calculate_entropy_from_logits(self) -> None:
        """Entropy calculation should work on typical logit distributions."""
        backend = get_default_backend()
        calculator = LogitEntropyCalculator(top_k=10, backend=backend)

        logits = backend.array([2.0, 1.5, 0.5, -0.5, -1.0])
        entropy, variance = calculator.compute(logits)

        assert entropy >= 0
        assert variance >= 0

    def test_entropy_tracks_uncertainty(self) -> None:
        """Higher uncertainty distributions should have higher entropy."""
        backend = get_default_backend()
        calculator = LogitEntropyCalculator(top_k=10, backend=backend)

        low_uncertainty = backend.array([10.0, 0.0, 0.0, 0.0, 0.0])
        high_uncertainty = backend.array([1.0, 1.0, 1.0, 1.0, 1.0])

        low_entropy, _ = calculator.compute(low_uncertainty)
        high_entropy, _ = calculator.compute(high_uncertainty)

        assert high_entropy > low_entropy

    @pytest.mark.parametrize("seed", range(5))
    def test_entropy_always_non_negative(self, seed: int) -> None:
        """Entropy should always be non-negative."""
        backend = get_default_backend()
        calculator = LogitEntropyCalculator(top_k=10, backend=backend)

        backend.random_seed(seed)
        logits = backend.random_normal((20,))
        backend.eval(logits)
        entropy, variance = calculator.compute(logits)

        assert entropy >= 0
        assert variance >= 0


# =============================================================================
# Entropy Window Integration
# =============================================================================


class TestEntropyWindowIntegration:
    """Tests for entropy window tracking integration."""

    def test_window_tracks_entropy_over_time(self) -> None:
        """Window should track entropy samples over time."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))

        config = EntropyWindowConfig(window_size=5)
        window = EntropyWindow(config)

        samples = [(1.0 + 0.1 * i, 0.2 + 0.01 * i, i) for i in range(8)]
        for entropy, variance, token_index in samples:
            window.add(entropy, variance, token_index)

        status = window.status()
        tail = samples[-5:]
        expected_mean = sum(s[0] for s in tail) / len(tail)

        assert status.sample_count == 5
        assert abs(status.moving_average - expected_mean) <= eps
        assert status.token_start == tail[0][2]
        assert status.token_end == tail[-1][2]

    def test_window_empty_status(self) -> None:
        """Empty window returns zeroed measurements."""
        window = EntropyWindow(EntropyWindowConfig(window_size=4))
        status = window.status()

        assert status.sample_count == 0
        assert status.current_entropy == 0.0
        assert status.moving_average == 0.0


# =============================================================================
# Conversation Tracking Integration
# =============================================================================


class TestConversationTrackingIntegration:
    """Tests for conversation entropy tracking integration."""

    def test_tracker_records_turns_and_metrics(self) -> None:
        """Tracker should compute raw measurements from turns."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))

        tracker = ConversationEntropyTracker()
        timestamp = datetime.now(timezone.utc)

        tracker.record_turn(
            token_count=50,
            avg_delta=0.1,
            max_anomaly_score=0.2,
            anomaly_count=0,
            timestamp=timestamp,
        )
        assessment = tracker.record_turn(
            token_count=60,
            avg_delta=-0.1,
            max_anomaly_score=0.3,
            anomaly_count=1,
            timestamp=timestamp,
        )
        assessment = tracker.record_turn(
            token_count=55,
            avg_delta=0.1,
            max_anomaly_score=0.15,
            anomaly_count=0,
            timestamp=timestamp,
        )

        assert assessment.turn_count == 3
        expected_mean = (0.1 - 0.1 + 0.1) / 3.0
        assert abs(assessment.mean_delta - expected_mean) <= eps
        assert assessment.anomaly_count == 1
        assert assessment.max_anomaly_score == 0.3
        assert assessment.oscillation_frequency == 1.0

    def test_tracker_baseline_drift(self) -> None:
        """Baseline drift should use baseline z-score of mean delta."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))

        baseline = ConversationEntropyBaseline(delta_mean=0.0, delta_std_dev=0.5)
        tracker = ConversationEntropyTracker(baseline=baseline)
        timestamp = datetime.now(timezone.utc)

        tracker.record_turn(
            token_count=40,
            avg_delta=0.2,
            max_anomaly_score=0.1,
            anomaly_count=0,
            timestamp=timestamp,
        )
        assessment = tracker.record_turn(
            token_count=55,
            avg_delta=0.2,
            max_anomaly_score=0.1,
            anomaly_count=0,
            timestamp=timestamp,
        )

        expected_z = baseline.z_score(assessment.mean_delta)
        assert abs(assessment.cumulative_drift - expected_z) <= eps

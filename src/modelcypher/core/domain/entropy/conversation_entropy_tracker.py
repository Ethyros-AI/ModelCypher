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

"""Conversation entropy tracker for multi-turn measurements.

Tracks entropy dynamics across conversation turns and reports raw measurements.
No internal classification or thresholds are applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon

@dataclass(frozen=True)
class ConversationEntropyBaseline:
    """Baseline entropy statistics for comparison."""

    delta_mean: float
    delta_std_dev: float

    @classmethod
    def from_samples(cls, delta_samples: list[float]) -> "ConversationEntropyBaseline":
        """Derive baseline from entropy delta samples."""
        if len(delta_samples) < 2:
            raise ValueError("Need at least 2 samples for baseline")

        mean = sum(delta_samples) / len(delta_samples)
        variance = sum((d - mean) ** 2 for d in delta_samples) / len(delta_samples)
        return cls(delta_mean=mean, delta_std_dev=math.sqrt(variance))

    def z_score(self, value: float) -> float:
        """Compute z-score relative to the baseline mean/std."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        if self.delta_std_dev < eps:
            return 0.0 if abs(value - self.delta_mean) < eps else float("inf")
        return (value - self.delta_mean) / self.delta_std_dev


@dataclass(frozen=True)
class TurnSummary:
    """Summary of a single conversation turn."""

    turn_index: int
    timestamp: datetime
    token_count: int
    avg_delta: float
    max_anomaly_score: float
    anomaly_count: int


@dataclass(frozen=True)
class ConversationAssessment:
    """Conversation-level entropy measurements."""

    conversation_id: UUID | None
    turn_count: int
    mean_delta: float
    std_delta: float
    oscillation_amplitude: float
    oscillation_frequency: float
    cumulative_drift: float
    anomaly_count: int
    anomaly_rate: float
    max_anomaly_score: float
    delta_change_mean: float
    delta_change_std: float


class ConversationEntropyTracker:
    """Tracks entropy patterns across conversation turns and reports measurements."""

    def __init__(self, baseline: ConversationEntropyBaseline | None = None) -> None:
        self._baseline = baseline
        self._turn_summaries: list[TurnSummary] = []
        self._conversation_start: datetime | None = None
        self._conversation_id: UUID | None = None

    def record_turn(
        self,
        token_count: int,
        avg_delta: float,
        max_anomaly_score: float,
        anomaly_count: int,
        timestamp: datetime | None = None,
    ) -> ConversationAssessment:
        """Record a completed generation turn and return conversation assessment."""
        timestamp = timestamp or datetime.utcnow()

        if self._conversation_start is None:
            self._conversation_start = timestamp
            self._conversation_id = uuid4()

        turn_index = len(self._turn_summaries)
        summary = TurnSummary(
            turn_index=turn_index,
            timestamp=timestamp,
            token_count=token_count,
            avg_delta=avg_delta,
            max_anomaly_score=max_anomaly_score,
            anomaly_count=anomaly_count,
        )
        self._turn_summaries.append(summary)

        return self._compute_assessment()

    def reset(self) -> None:
        """Reset the conversation tracker for a new conversation."""
        self._turn_summaries = []
        self._conversation_start = None
        self._conversation_id = None

    @property
    def current_turn_count(self) -> int:
        """Current turn count."""
        return len(self._turn_summaries)

    @property
    def all_turn_summaries(self) -> list[TurnSummary]:
        """All turn summaries for export/analysis."""
        return list(self._turn_summaries)

    @property
    def current_conversation_id(self) -> UUID | None:
        """Current conversation ID."""
        return self._conversation_id

    def _compute_assessment(self) -> ConversationAssessment:
        """Compute current conversation measurements."""
        turn_count = len(self._turn_summaries)

        if turn_count == 0:
            return ConversationAssessment(
                conversation_id=self._conversation_id,
                turn_count=0,
                mean_delta=0.0,
                std_delta=0.0,
                oscillation_amplitude=0.0,
                oscillation_frequency=0.0,
                cumulative_drift=0.0,
                anomaly_count=0,
                anomaly_rate=0.0,
                max_anomaly_score=0.0,
                delta_change_mean=0.0,
                delta_change_std=0.0,
            )

        deltas = [t.avg_delta for t in self._turn_summaries]
        mean_delta = sum(deltas) / turn_count
        std_delta = self._compute_std(deltas)
        oscillation_amplitude = std_delta
        oscillation_frequency = self._compute_oscillation_frequency(deltas)
        cumulative_drift = self._compute_cumulative_drift(mean_delta, std_delta)

        anomaly_count = sum(t.anomaly_count for t in self._turn_summaries)
        anomaly_rate = anomaly_count / turn_count
        max_anomaly_score = max((t.max_anomaly_score for t in self._turn_summaries), default=0.0)

        delta_changes = [
            deltas[i] - deltas[i - 1]
            for i in range(1, len(deltas))
        ]
        delta_change_mean = sum(delta_changes) / len(delta_changes) if delta_changes else 0.0
        delta_change_std = self._compute_std(delta_changes) if delta_changes else 0.0

        return ConversationAssessment(
            conversation_id=self._conversation_id,
            turn_count=turn_count,
            mean_delta=mean_delta,
            std_delta=std_delta,
            oscillation_amplitude=oscillation_amplitude,
            oscillation_frequency=oscillation_frequency,
            cumulative_drift=cumulative_drift,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate,
            max_anomaly_score=max_anomaly_score,
            delta_change_mean=delta_change_mean,
            delta_change_std=delta_change_std,
        )

    def _compute_std(self, values: list[float]) -> float:
        """Compute population standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)

    def _compute_oscillation_frequency(self, deltas: list[float]) -> float:
        """Compute oscillation frequency (sign changes in delta differences)."""
        if len(deltas) < 3:
            return 0.0

        sign_changes = 0
        previous_diff: float | None = None

        for i in range(1, len(deltas)):
            diff = deltas[i] - deltas[i - 1]
            if previous_diff is not None:
                if (previous_diff > 0 and diff < 0) or (previous_diff < 0 and diff > 0):
                    sign_changes += 1
            previous_diff = diff

        max_changes = len(deltas) - 2
        return sign_changes / max_changes if max_changes > 0 else 0.0

    def _compute_cumulative_drift(self, mean_delta: float, std_delta: float) -> float:
        """Compute cumulative drift from baseline or conversation start."""
        if self._baseline is not None:
            return self._baseline.z_score(mean_delta)

        if not self._turn_summaries:
            return 0.0

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        first_delta = self._turn_summaries[0].avg_delta
        return (mean_delta - first_delta) / max(std_delta, eps)

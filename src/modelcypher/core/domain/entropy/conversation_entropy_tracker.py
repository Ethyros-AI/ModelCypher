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

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    inf_value,
)


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

        backend = get_default_backend()
        samples_arr = backend.array(delta_samples)
        mean_arr = backend.mean(samples_arr)
        variance_arr = backend.var(samples_arr)
        std_arr = backend.sqrt(variance_arr)
        backend.eval(mean_arr, std_arr)
        return cls(
            delta_mean=float(backend.to_scalar(mean_arr)),
            delta_std_dev=float(backend.to_scalar(std_arr)),
        )

    def z_score(self, value: float) -> float:
        """Compute z-score relative to the baseline mean/std."""
        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        val_arr = backend.array([value])
        mean_arr = backend.array([self.delta_mean])
        std_arr = backend.array([self.delta_std_dev])
        diff = val_arr - mean_arr
        abs_diff = backend.abs(diff)
        eps_arr = backend.array([eps])
        std_small = std_arr < eps_arr
        within = abs_diff < eps_arr
        zero_arr = backend.zeros_like(val_arr)
        inf_arr = backend.full(val_arr.shape, inf_value(backend))
        z_arr = diff / std_arr
        result = backend.where(std_small, backend.where(within, zero_arr, inf_arr), z_arr)
        backend.eval(result)
        return float(backend.to_scalar(result))


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

        backend = get_default_backend()
        deltas = [t.avg_delta for t in self._turn_summaries]
        deltas_arr = backend.array(deltas)
        mean_arr = backend.mean(deltas_arr)
        backend.eval(mean_arr)
        mean_delta = float(backend.to_scalar(mean_arr))
        std_delta = self._compute_std(deltas)
        oscillation_amplitude = std_delta
        oscillation_frequency = self._compute_oscillation_frequency(deltas_arr)
        cumulative_drift = self._compute_cumulative_drift(mean_delta, std_delta)

        anomaly_arr = backend.array([t.anomaly_count for t in self._turn_summaries])
        anomaly_count_arr = backend.sum(anomaly_arr)
        max_anomaly_arr = backend.max(
            backend.array([t.max_anomaly_score for t in self._turn_summaries])
        )
        anomaly_rate_arr = anomaly_count_arr / backend.array([float(turn_count)])
        backend.eval(anomaly_count_arr, max_anomaly_arr, anomaly_rate_arr)
        anomaly_count = int(backend.to_scalar(anomaly_count_arr))
        anomaly_rate = float(backend.to_scalar(anomaly_rate_arr))
        max_anomaly_score = float(backend.to_scalar(max_anomaly_arr))

        if turn_count > 1:
            idx_prev = backend.arange(0, turn_count - 1)
            idx_next = backend.arange(1, turn_count)
            prev_vals = backend.take(deltas_arr, idx_prev, axis=0)
            next_vals = backend.take(deltas_arr, idx_next, axis=0)
            changes_arr = next_vals - prev_vals
            change_mean_arr = backend.mean(changes_arr)
            backend.eval(change_mean_arr)
            delta_change_mean = float(backend.to_scalar(change_mean_arr))
            # Stay on backend - avoid tolist() conversion
            delta_change_std = self._compute_std_arr(changes_arr, backend)
        else:
            delta_change_mean = 0.0
            delta_change_std = 0.0

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
        backend = get_default_backend()
        values_arr = backend.array(values)
        return self._compute_std_arr(values_arr, backend)

    def _compute_std_arr(self, values_arr, backend=None) -> float:
        """Compute population standard deviation from backend array."""
        count = int(values_arr.shape[0])
        if count < 2:
            return 0.0
        if backend is None:
            backend = get_default_backend()
        mean_arr = backend.mean(values_arr)
        diff = values_arr - mean_arr
        # Use epsilon guard for division
        eps = division_epsilon(backend, values_arr)
        safe_count = max(float(count), eps)
        variance_arr = backend.sum(diff * diff) / safe_count
        std_arr = backend.sqrt(variance_arr)
        backend.eval(std_arr)
        return float(backend.to_scalar(std_arr))

    def _compute_oscillation_frequency(self, deltas_arr) -> float:
        """Compute oscillation frequency (sign changes in delta differences)."""
        count = int(deltas_arr.shape[0])
        if count < 3:
            return 0.0

        backend = get_default_backend()
        idx_prev = backend.arange(0, count - 1)
        idx_next = backend.arange(1, count)
        prev_vals = backend.take(deltas_arr, idx_prev, axis=0)
        next_vals = backend.take(deltas_arr, idx_next, axis=0)
        diffs = next_vals - prev_vals
        signs = backend.sign(diffs)
        if int(signs.shape[0]) < 2:
            return 0.0

        idx_prev_s = backend.arange(0, int(signs.shape[0]) - 1)
        idx_next_s = backend.arange(1, int(signs.shape[0]))
        prev_sign = backend.take(signs, idx_prev_s, axis=0)
        next_sign = backend.take(signs, idx_next_s, axis=0)
        sign_change = (prev_sign * next_sign) < 0
        count_arr = backend.sum(backend.astype(sign_change, "float32"))
        # Use epsilon guard for division
        eps = division_epsilon(backend, count_arr)
        max_changes = max(float(count - 2), eps)
        freq_arr = count_arr / max_changes
        backend.eval(freq_arr)
        return float(backend.to_scalar(freq_arr))

    def _compute_cumulative_drift(self, mean_delta: float, std_delta: float) -> float:
        """Compute cumulative drift from baseline or conversation start."""
        if self._baseline is not None:
            return self._baseline.z_score(mean_delta)

        if not self._turn_summaries:
            return 0.0

        backend = get_default_backend()
        eps = division_epsilon(backend, backend.array([0.0]))
        first_delta = self._turn_summaries[0].avg_delta
        mean_arr = backend.array([mean_delta])
        first_arr = backend.array([first_delta])
        std_arr = backend.array([std_delta])
        denom = backend.maximum(std_arr, backend.array([eps]))
        drift_arr = (mean_arr - first_arr) / denom
        backend.eval(drift_arr)
        return float(backend.to_scalar(drift_arr))

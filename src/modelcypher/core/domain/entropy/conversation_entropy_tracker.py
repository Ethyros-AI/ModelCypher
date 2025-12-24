"""Conversation entropy tracker for multi-turn manipulation detection.

Tracks entropy patterns across conversation turns to detect:
- Oscillation patterns: The sawtooth signature of manipulation attempts
- Cumulative drift: Gradual shift from baseline over extended interactions
- Turn-level anomalies: Sudden behavioral changes between turns

Key Insight (Jailbreak Detection):
Legitimate conversations show settling entropy - users ask, get answers, move on.
Manipulation attempts show sustained oscillation as the attacker repeatedly
probes boundaries, retreats, and probes again. The oscillation frequency and
amplitude over turns is the manipulation signature.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class ConversationPattern(str, Enum):
    """Conversation-level pattern classification."""

    INSUFFICIENT = "insufficient"
    """Not enough turns for analysis."""

    SETTLING = "settling"
    """Normal conversation - variance decreasing over time."""

    OSCILLATING = "oscillating"
    """High-frequency delta changes - manipulation signature."""

    DRIFTING = "drifting"
    """Gradual shift from baseline - possible slow manipulation."""

    UNSTABLE = "unstable"
    """Erratic behavior - frequent anomalies."""


class ConversationRecommendation(str, Enum):
    """Recommended action based on conversation analysis."""

    CONTINUE = "continue"
    """Continue normally."""

    MONITOR = "monitor"
    """Increase monitoring/logging."""

    INTERVENE = "intervene"
    """Intervene (e.g., change topic, add guardrails)."""

    HALT = "halt"
    """Halt conversation."""


@dataclass(frozen=True)
class ConversationEntropyConfiguration:
    """Configuration for conversation entropy tracking."""

    oscillation_window_size: int = 5
    """Sliding window size for oscillation detection (number of turns)."""

    minimum_turns_for_analysis: int = 3
    """Minimum turns before oscillation analysis is valid."""

    oscillation_threshold: float = 0.8
    """Oscillation amplitude threshold (std dev of deltas over window)."""

    drift_threshold: float = 1.5
    """Cumulative drift threshold from baseline before flagging."""

    turn_spike_threshold: float = 0.5
    """Turn-over-turn delta change threshold for spike detection."""

    recency_decay: float = 0.9
    """Confidence decay factor for older turns (0.0-1.0)."""

    alert_threshold: float = 0.6
    """Threshold for emitting manipulation alert signals (composite score)."""

    @classmethod
    def default(cls) -> ConversationEntropyConfiguration:
        """Create default configuration."""
        return cls()


@dataclass(frozen=True)
class TurnSummary:
    """Summary of a single conversation turn."""

    turn_index: int
    """Index of this turn in the conversation."""

    timestamp: datetime
    """When this turn completed."""

    token_count: int
    """Number of tokens generated in this turn."""

    avg_delta: float
    """Average entropy delta for this turn."""

    max_anomaly_score: float
    """Maximum anomaly score observed."""

    anomaly_count: int
    """Number of anomalies detected."""

    backdoor_signature_count: int
    """Number of backdoor signatures detected."""

    circuit_breaker_tripped: bool
    """Whether the circuit breaker was tripped."""

    security_assessment: str
    """Security assessment from the turn."""


@dataclass(frozen=True)
class ConversationAssessment:
    """Comprehensive conversation-level assessment."""

    conversation_id: UUID | None
    """Unique identifier for this conversation."""

    turn_count: int
    """Number of turns in this conversation."""

    oscillation_amplitude: float
    """Standard deviation of entropy deltas over recent window."""

    oscillation_frequency: float
    """Frequency of sign changes in delta differences (0-1)."""

    cumulative_drift: float
    """Z-score drift from baseline or conversation start."""

    manipulation_signal: float
    """Composite manipulation signal (0 = benign, 1 = likely manipulation)."""

    assessment_confidence: float
    """Confidence in assessment (based on turn count)."""

    pattern: ConversationPattern
    """Classified conversation pattern."""

    recommendation: ConversationRecommendation
    """Recommended action."""

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        signal_pct = int(self.manipulation_signal * 100)
        return (
            f"Turn {self.turn_count}: {self.pattern.value} pattern, "
            f"{signal_pct}% manipulation signal, recommend: {self.recommendation.value}"
        )


@dataclass
class EntropyBaseline:
    """Baseline entropy statistics for comparison."""

    delta_mean: float = 0.0
    """Mean entropy delta."""

    delta_std_dev: float = 0.1
    """Standard deviation of entropy deltas."""

    oscillation_threshold: float = 0.8
    """Threshold for excessive oscillation."""

    def is_oscillation_excessive(self, amplitude: float) -> bool:
        """Check if oscillation amplitude is excessive."""
        return amplitude > self.oscillation_threshold


class ConversationEntropyTracker:
    """Tracks entropy patterns across conversation turns for manipulation detection.

    While per-turn trackers monitor within a single generation, this tracker
    maintains state across multiple conversation turns to detect patterns
    that emerge over extended interactions.
    """

    def __init__(
        self,
        baseline: EntropyBaseline | None = None,
        configuration: ConversationEntropyConfiguration | None = None,
    ) -> None:
        """Create a conversation entropy tracker.

        Args:
            baseline: Optional entropy baseline for drift detection.
            configuration: Tracker configuration.
        """
        self._config = configuration or ConversationEntropyConfiguration.default()
        self._baseline = baseline
        self._turn_summaries: list[TurnSummary] = []
        self._conversation_start: datetime | None = None
        self._conversation_id: UUID | None = None

    def record_turn(
        self,
        token_count: int,
        avg_delta: float,
        max_anomaly_score: float = 0.0,
        anomaly_count: int = 0,
        backdoor_signature_count: int = 0,
        circuit_breaker_tripped: bool = False,
        security_assessment: str = "nominal",
        timestamp: datetime | None = None,
    ) -> ConversationAssessment:
        """Record a completed generation turn and return conversation-level assessment.

        Args:
            token_count: Number of tokens generated.
            avg_delta: Average entropy delta for this turn.
            max_anomaly_score: Maximum anomaly score observed.
            anomaly_count: Number of anomalies detected.
            backdoor_signature_count: Number of backdoor signatures detected.
            circuit_breaker_tripped: Whether circuit breaker was tripped.
            security_assessment: Security assessment string.
            timestamp: Turn completion time. Defaults to now.

        Returns:
            Current conversation assessment including manipulation signals.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Initialize conversation if needed
        if self._conversation_start is None:
            self._conversation_start = timestamp
            self._conversation_id = uuid4()

        # Create turn summary
        turn_index = len(self._turn_summaries)
        summary = TurnSummary(
            turn_index=turn_index,
            timestamp=timestamp,
            token_count=token_count,
            avg_delta=avg_delta,
            max_anomaly_score=max_anomaly_score,
            anomaly_count=anomaly_count,
            backdoor_signature_count=backdoor_signature_count,
            circuit_breaker_tripped=circuit_breaker_tripped,
            security_assessment=security_assessment,
        )
        self._turn_summaries.append(summary)

        # Compute conversation-level metrics
        assessment = self._compute_assessment()

        # Log significant events
        if assessment.manipulation_signal > 0.5:
            logger.warning(
                "Manipulation signal elevated: %.2f at turn %d",
                assessment.manipulation_signal,
                turn_index,
            )

        return assessment

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
        """Compute current conversation assessment."""
        turn_count = len(self._turn_summaries)

        if turn_count < self._config.minimum_turns_for_analysis:
            return ConversationAssessment(
                conversation_id=self._conversation_id,
                turn_count=turn_count,
                oscillation_amplitude=0.0,
                oscillation_frequency=0.0,
                cumulative_drift=0.0,
                manipulation_signal=0.0,
                assessment_confidence=turn_count / self._config.minimum_turns_for_analysis,
                pattern=ConversationPattern.INSUFFICIENT,
                recommendation=ConversationRecommendation.CONTINUE,
            )

        # Compute oscillation metrics over sliding window
        window_turns = self._turn_summaries[-self._config.oscillation_window_size :]
        deltas = [t.avg_delta for t in window_turns]

        oscillation_amplitude = self._compute_oscillation_amplitude(deltas)
        oscillation_frequency = self._compute_oscillation_frequency(deltas)

        # Compute cumulative drift from baseline
        cumulative_drift = self._compute_cumulative_drift()

        # Compute manipulation signal (composite score)
        manipulation_signal = self._compute_manipulation_signal(
            oscillation_amplitude=oscillation_amplitude,
            oscillation_frequency=oscillation_frequency,
            cumulative_drift=cumulative_drift,
            window_turns=window_turns,
        )

        # Classify pattern
        recent_anomalies = sum(t.anomaly_count for t in window_turns)
        pattern = self._classify_pattern(
            oscillation_amplitude=oscillation_amplitude,
            cumulative_drift=cumulative_drift,
            recent_anomalies=recent_anomalies,
        )

        # Determine recommendation
        recommendation = self._determine_recommendation(
            manipulation_signal=manipulation_signal,
            pattern=pattern,
        )

        return ConversationAssessment(
            conversation_id=self._conversation_id,
            turn_count=turn_count,
            oscillation_amplitude=oscillation_amplitude,
            oscillation_frequency=oscillation_frequency,
            cumulative_drift=cumulative_drift,
            manipulation_signal=manipulation_signal,
            assessment_confidence=min(
                1.0, turn_count / self._config.oscillation_window_size
            ),
            pattern=pattern,
            recommendation=recommendation,
        )

    def _compute_oscillation_amplitude(self, deltas: list[float]) -> float:
        """Compute oscillation amplitude (standard deviation of deltas)."""
        if len(deltas) < 2:
            return 0.0

        mean = sum(deltas) / len(deltas)
        variance = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
        return math.sqrt(variance)

    def _compute_oscillation_frequency(self, deltas: list[float]) -> float:
        """Compute oscillation frequency (number of sign changes in delta differences)."""
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

        # Normalize by maximum possible sign changes
        max_changes = len(deltas) - 2
        return sign_changes / max_changes if max_changes > 0 else 0.0

    def _compute_cumulative_drift(self) -> float:
        """Compute cumulative drift from baseline or conversation start."""
        if not self._turn_summaries:
            return 0.0

        recent_deltas = [t.avg_delta for t in self._turn_summaries[-3:]]
        recent_mean = sum(recent_deltas) / len(recent_deltas)

        if self._baseline is not None:
            # Drift from declared baseline
            return abs(recent_mean - self._baseline.delta_mean) / max(
                self._baseline.delta_std_dev, 0.001
            )
        else:
            # Drift from conversation start
            initial_deltas = [t.avg_delta for t in self._turn_summaries[:3]]
            initial_mean = sum(initial_deltas) / len(initial_deltas)
            all_deltas = [t.avg_delta for t in self._turn_summaries]
            conversation_std = self._compute_oscillation_amplitude(all_deltas)
            return abs(recent_mean - initial_mean) / max(conversation_std, 0.001)

    def _compute_manipulation_signal(
        self,
        oscillation_amplitude: float,
        oscillation_frequency: float,
        cumulative_drift: float,
        window_turns: list[TurnSummary],
    ) -> float:
        """Compute composite manipulation signal (0.0 = benign, 1.0 = likely manipulation)."""
        # Normalized components
        osc_amp_score = min(1.0, oscillation_amplitude / self._config.oscillation_threshold)
        osc_freq_score = oscillation_frequency  # Already 0-1
        drift_score = min(1.0, cumulative_drift / self._config.drift_threshold)

        # Apply recency-weighted analysis
        weighted_anomaly_score = 0.0
        weighted_backdoor_score = 0.0
        has_circuit_breaker_trip = False
        spike_count = 0
        total_weight = 0.0

        for index, turn in enumerate(window_turns):
            # Recency weight: more recent turns weighted higher
            recency_weight = self._config.recency_decay ** (len(window_turns) - 1 - index)
            total_weight += recency_weight

            weighted_anomaly_score += turn.anomaly_count * recency_weight

            if turn.backdoor_signature_count > 0:
                weighted_backdoor_score += turn.backdoor_signature_count * recency_weight

            if turn.circuit_breaker_tripped:
                has_circuit_breaker_trip = True

            if index > 0:
                previous_delta = window_turns[index - 1].avg_delta
                delta_change = abs(turn.avg_delta - previous_delta)
                if delta_change > self._config.turn_spike_threshold:
                    spike_count += 1

        # Normalize weighted scores
        normalized_anomaly = (
            min(1.0, weighted_anomaly_score / (5.0 * total_weight))
            if total_weight > 0
            else 0.0
        )
        normalized_backdoor = (
            min(1.0, weighted_backdoor_score / total_weight) if total_weight > 0 else 0.0
        )
        spike_score = (
            min(1.0, spike_count / (len(window_turns) - 1))
            if len(window_turns) > 1
            else 0.0
        )

        # Weighted combination
        signal = (
            0.30 * osc_amp_score
            + 0.20 * osc_freq_score
            + 0.15 * drift_score
            + 0.10 * normalized_anomaly
            + 0.15 * normalized_backdoor
            + 0.10 * spike_score
        )

        # Circuit breaker trips immediately elevate to high alert
        if has_circuit_breaker_trip:
            signal = max(signal, 0.8)

        # Use baseline thresholds if available
        if self._baseline is not None:
            if self._baseline.is_oscillation_excessive(oscillation_amplitude):
                signal = max(signal, 0.7)

        return min(1.0, signal)

    def _classify_pattern(
        self,
        oscillation_amplitude: float,
        cumulative_drift: float,
        recent_anomalies: int,
    ) -> ConversationPattern:
        """Classify conversation pattern."""
        if oscillation_amplitude > self._config.oscillation_threshold:
            return ConversationPattern.OSCILLATING

        if cumulative_drift > self._config.drift_threshold:
            return ConversationPattern.DRIFTING

        if recent_anomalies > 3:
            return ConversationPattern.UNSTABLE

        return ConversationPattern.SETTLING

    def _determine_recommendation(
        self,
        manipulation_signal: float,
        pattern: ConversationPattern,
    ) -> ConversationRecommendation:
        """Determine recommended action."""
        if manipulation_signal > 0.8 or pattern == ConversationPattern.OSCILLATING:
            return ConversationRecommendation.HALT

        if manipulation_signal > 0.6 or pattern == ConversationPattern.UNSTABLE:
            return ConversationRecommendation.INTERVENE

        if manipulation_signal > 0.4 or pattern == ConversationPattern.DRIFTING:
            return ConversationRecommendation.MONITOR

        return ConversationRecommendation.CONTINUE

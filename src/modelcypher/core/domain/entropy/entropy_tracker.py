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

"""
Entropy Tracking for Cognitive State Analysis.

Raw geometric measurements. The entropy and variance values ARE the cognitive state.
No classification enums that destroy information.

Information-theoretic thresholds (not arbitrary):
- Confident: entropy < ln(e²) ≈ 2.0 (probability mass concentrated)
- Uncertain: entropy > 3.0 (high uncertainty)
- Distress signature: high entropy + low variance

Research Basis:
- Anthropic "Signs of introspection in LLMs" (Oct 2025)
- arXiv:2406.15927 - Semantic Entropy Probes
- arXiv:2502.21239 - Semantic Volume
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# EntropyTransition
# =============================================================================


@dataclass(frozen=True)
class EntropyTransition:
    """Records an entropy transition during generation.

    Raw entropy/variance values before and after. The delta IS the severity change.
    """

    from_entropy: float
    from_variance: float
    to_entropy: float
    to_variance: float
    token_index: int
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str | None = None

    @property
    def entropy_delta(self) -> float:
        """Change in entropy. Positive = increasing uncertainty."""
        return self.to_entropy - self.from_entropy

    @property
    def variance_delta(self) -> float:
        """Change in variance."""
        return self.to_variance - self.from_variance

    @property
    def is_escalation(self) -> bool:
        """Entropy increased significantly (getting more uncertain)."""
        return self.entropy_delta > 0.5

    @property
    def is_recovery(self) -> bool:
        """Entropy decreased significantly (getting more confident)."""
        return self.entropy_delta < -0.5


# Backward compatibility alias
StateTransition = EntropyTransition


@dataclass
class EntropySample:
    """
    Semantic entropy measurement from a generation window.

    The entropy and variance values ARE the cognitive state.
    Multi-tier entropy metrics:
    - logit_entropy: Fast Shannon entropy from full-vocab logits
    - sep_entropy: SEP probe prediction (optional)
    - semantic_volume: Multi-sample geometric metric (optional)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_start: int = 0
    token_end: int = 0

    # Plan 1: Logit Entropy (always available)
    logit_entropy: float = 0.0
    top_k_variance: float = 0.0

    # Plan 2: SEP Probe (optional)
    sep_entropy: float | None = None
    sep_layers: list[int] | None = None
    sep_confidence: float | None = None

    # Plan 3: Semantic Volume (optional, expensive)
    semantic_volume: float | None = None
    sample_count: int | None = None
    pca_dimensions: int | None = None

    # Metadata
    computed_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    source: str | None = None
    correlation_id: str | None = None

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start + 1

    @property
    def best_entropy_estimate(self) -> float:
        """Best available entropy: prefer SEP > logit."""
        return self.sep_entropy if self.sep_entropy is not None else self.logit_entropy

    @property
    def is_low_entropy(self) -> bool:
        """Entropy below confident threshold (< 2.0)."""
        return self.best_entropy_estimate < 2.0

    @property
    def is_high_entropy(self) -> bool:
        """Entropy above uncertain threshold (> 3.0)."""
        return self.best_entropy_estimate > 3.0

    def should_trip_circuit_breaker(self, threshold: float = 4.0) -> bool:
        """Whether entropy exceeds circuit breaker threshold."""
        return self.best_entropy_estimate >= threshold


# =============================================================================
# DistressDetection
# =============================================================================


@dataclass(frozen=True)
class DistressDetection:
    """Detection result from pattern analysis."""

    detected: bool
    confidence: float
    token_index: int
    mean_entropy: float
    mean_variance: float
    reason: str


# =============================================================================
# EntropyWindow
# =============================================================================


@dataclass
class EntropyWindowStatus:
    """Status of the sliding entropy window."""

    window_id: str
    sample_count: int
    current_entropy: float
    moving_average: float
    entropy_std_dev: float
    token_start: int
    token_end: int
    consecutive_high_count: int
    should_trip_circuit_breaker: bool


class EntropyWindow:
    """Sliding window for entropy statistics."""

    def __init__(self, window_size: int = 20, high_threshold: float = 3.0):
        self.window_id = str(uuid.uuid4())
        self.window_size = window_size
        self.high_threshold = high_threshold
        self.samples: list[tuple[float, float, int]] = []  # (entropy, variance, tokenIndex)
        self.consecutive_high_count = 0
        self.circuit_breaker_tripped = False

    def add(self, entropy: float, variance: float, token_index: int) -> EntropyWindowStatus:
        """Add a sample to the window."""
        self.samples.append((entropy, variance, token_index))
        if len(self.samples) > self.window_size:
            self.samples.pop(0)

        # Track consecutive high entropy
        if entropy >= self.high_threshold:
            self.consecutive_high_count += 1
        else:
            self.consecutive_high_count = 0

        # Trip circuit breaker on sustained high entropy
        if self.consecutive_high_count >= 5 or entropy >= 4.5:
            self.circuit_breaker_tripped = True

        return self.status()

    def status(self) -> EntropyWindowStatus:
        """Get current window status."""
        if not self.samples:
            return EntropyWindowStatus(
                window_id=self.window_id,
                sample_count=0,
                current_entropy=0.0,
                moving_average=0.0,
                entropy_std_dev=0.0,
                token_start=0,
                token_end=0,
                consecutive_high_count=0,
                should_trip_circuit_breaker=False,
            )

        entropies = [s[0] for s in self.samples]
        current = entropies[-1]
        avg = sum(entropies) / len(entropies)
        variance = sum((e - avg) ** 2 for e in entropies) / len(entropies)
        std_dev = math.sqrt(variance)

        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self.samples),
            current_entropy=current,
            moving_average=avg,
            entropy_std_dev=std_dev,
            token_start=self.samples[0][2],
            token_end=self.samples[-1][2],
            consecutive_high_count=self.consecutive_high_count,
            should_trip_circuit_breaker=self.circuit_breaker_tripped,
        )

    def to_entropy_sample(self, source: str, correlation_id: str | None = None) -> EntropySample:
        """Create an EntropySample from current window state."""
        status = self.status()
        avg_variance = sum(s[1] for s in self.samples) / max(len(self.samples), 1)

        return EntropySample(
            window_id=self.window_id,
            token_start=status.token_start,
            token_end=status.token_end,
            logit_entropy=status.moving_average,
            top_k_variance=avg_variance,
            source=source,
            correlation_id=correlation_id,
        )


# =============================================================================
# Entropy State Analysis (no classification enums)
# =============================================================================


@dataclass
class EntropyStateThresholds:
    """Information-theoretic thresholds for entropy analysis.

    These are not arbitrary - they're derived from probability theory:
    - Confident: entropy < ln(e²) ≈ 2.0 means probability mass concentrated
    - Uncertain: entropy > 3.0 means high uncertainty
    - Distress signature: high entropy + low variance = normative uncertainty
    """

    confident_entropy_max: float = 2.0
    uncertain_entropy_min: float = 3.0
    distress_entropy_min: float = 3.5
    distress_variance_max: float = 0.2

    @classmethod
    def default(cls) -> "EntropyStateThresholds":
        return cls()


# Backward compatibility alias
ClassifierThresholds = EntropyStateThresholds


class ModelStateClassifier:
    """Analyzes model cognitive state from raw entropy and variance.

    Returns boolean flags for state conditions. The raw values ARE the state.
    """

    def __init__(self, thresholds: EntropyStateThresholds | None = None):
        self.thresholds = thresholds or EntropyStateThresholds.default()

    def is_confident(self, entropy: float, variance: float) -> bool:
        """Check if model is confident (low entropy)."""
        return entropy < self.thresholds.confident_entropy_max

    def is_uncertain(self, entropy: float, variance: float) -> bool:
        """Check if model is uncertain (high entropy)."""
        return entropy >= self.thresholds.uncertain_entropy_min

    def is_distressed(self, entropy: float, variance: float) -> bool:
        """Check if model shows distress signature (high entropy + low variance)."""
        return (
            entropy >= self.thresholds.distress_entropy_min
            and variance <= self.thresholds.distress_variance_max
        )

    def requires_caution(self, entropy: float, variance: float) -> bool:
        """Check if current state warrants caution."""
        return self.is_uncertain(entropy, variance) or self.is_distressed(entropy, variance)


# =============================================================================
# LogitEntropyCalculator
# =============================================================================


class LogitEntropyCalculator:
    """Computes entropy and variance from logits."""

    def __init__(self, top_k: int = 10, backend: "Backend | None" = None) -> None:
        self.top_k = top_k
        self._backend = backend or get_default_backend()

    def compute(self, logits: "Array") -> tuple[float, float]:
        """
        Compute entropy and top-K variance from logits.

        Args:
            logits: Array of shape [..., vocab_size]

        Returns:
            Tuple of (entropy, top_k_variance)
        """
        b = self._backend

        # Flatten if needed
        if logits.ndim > 1:
            logits = b.reshape(logits, (-1,))

        logits_f32 = b.astype(logits, "float32")

        # Full-vocab entropy: H = -sum(p * log(p))
        probs = b.softmax(logits_f32)
        log_probs = b.log(probs + 1e-10)
        entropy = -b.sum(probs * log_probs)
        b.eval(entropy)

        # Top-K variance
        k = min(self.top_k, logits.shape[-1])
        top_k_logits = b.sort(logits_f32)[-k:]
        variance = b.var(top_k_logits)
        b.eval(variance)

        entropy_np = b.to_numpy(entropy)
        variance_np = b.to_numpy(variance)
        return float(entropy_np.item()), float(variance_np.item())

    def create_sample(
        self,
        entropy: float,
        variance: float,
        window_id: str,
        token_start: int,
        token_end: int,
        latency_ms: float,
        source: str,
        correlation_id: str | None = None,
    ) -> EntropySample:
        """Create an EntropySample from computed values."""
        return EntropySample(
            window_id=window_id,
            token_start=token_start,
            token_end=token_end,
            logit_entropy=entropy,
            top_k_variance=variance,
            latency_ms=latency_ms,
            source=source,
            correlation_id=correlation_id,
        )


# =============================================================================
# EntropyPatternDetector
# =============================================================================


@dataclass
class PatternConfig:
    """Configuration for entropy pattern detection."""

    min_samples: int = 5
    high_entropy_threshold: float = 3.5
    low_variance_threshold: float = 0.2
    sustained_count: int = 3


class EntropyPatternDetector:
    """Detects distress patterns from entropy history."""

    def __init__(self, config: PatternConfig | None = None):
        self.config = config or PatternConfig()

    def detect_distress(
        self,
        samples: list[tuple[float, float]],  # (entropy, variance)
        token_index: int = 0,
    ) -> DistressDetection | None:
        """
        Detect distress pattern: sustained high entropy + low variance.

        Returns DistressDetection if distress detected, None otherwise.
        """
        if len(samples) < self.config.min_samples:
            return None

        recent = samples[-self.config.sustained_count :]
        high_entropy_count = sum(
            1
            for e, v in recent
            if e >= self.config.high_entropy_threshold and v <= self.config.low_variance_threshold
        )

        if high_entropy_count >= self.config.sustained_count:
            mean_e = sum(e for e, _ in recent) / len(recent)
            mean_v = sum(v for _, v in recent) / len(recent)
            confidence = high_entropy_count / len(recent)

            return DistressDetection(
                detected=True,
                confidence=confidence,
                token_index=token_index,
                mean_entropy=mean_e,
                mean_variance=mean_v,
                reason=f"Sustained high entropy ({mean_e:.2f}) with low variance ({mean_v:.2f})",
            )

        return None


# =============================================================================
# EntropyTracker
# =============================================================================


@dataclass
class EntropyTrackerConfig:
    """Configuration for EntropyTracker."""

    top_k: int = 10
    window_size: int = 20
    emit_interval: int = 1
    source: str = "EntropyTracker"

    @classmethod
    def default(cls) -> "EntropyTrackerConfig":
        return cls()


class EntropyTracker:
    """
    Coordinates entropy tracking for cognitive state analysis.

    Tracks raw entropy and variance values over token generation.
    The values ARE the cognitive state - no classification needed.

    Usage:
        tracker = EntropyTracker()
        tracker.start_session()

        # In generation loop:
        await tracker.record_logits(logits, token_index)

        sample = tracker.end_session()
    """

    def __init__(self, config: EntropyTrackerConfig | None = None):
        self.config = config or EntropyTrackerConfig.default()
        self.calculator = LogitEntropyCalculator(top_k=self.config.top_k)
        self.classifier = ModelStateClassifier()
        self.pattern_detector = EntropyPatternDetector()

        # Session state
        self._window: EntropyWindow | None = None
        self._correlation_id: str | None = None
        self._token_count: int = 0
        self._session_start: datetime | None = None

        # State tracking (raw values)
        self._current_entropy: float = 0.0
        self._current_variance: float = 0.0
        self._transition_history: list[EntropyTransition] = []
        self._sample_history: list[tuple[float, float]] = []
        self._trajectory_buffer: list[tuple[float, float, int]] = []
        self._last_distress_check: int = 0
        self._last_sample: EntropySample | None = None

        # Callbacks (updated signatures for raw values)
        self.on_entropy_sample: Callable[[EntropySample], None] | None = None
        self.on_entropy_changed: Callable[[EntropyTransition], None] | None = None
        self.on_distress_detected: Callable[[DistressDetection], None] | None = None
        self.on_circuit_breaker_tripped: Callable[[EntropyWindowStatus], None] | None = None

    def start_session(self, correlation_id: str | None = None):
        """Start a new tracking session."""
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._window = EntropyWindow(window_size=self.config.window_size)
        self._token_count = 0
        self._session_start = datetime.now()
        self._current_entropy = 0.0
        self._current_variance = 0.0
        self._transition_history = []
        self._sample_history = []
        self._trajectory_buffer = []
        self._last_distress_check = 0

    def end_session(self) -> EntropySample | None:
        """End the tracking session and return final sample."""
        if self._window is None:
            return None

        status = self._window.status()
        if status.sample_count == 0:
            self._window = None
            return None

        sample = self._window.to_entropy_sample(
            source=self.config.source,
            correlation_id=self._correlation_id,
        )

        self._last_sample = sample
        self._window = None
        self._correlation_id = None

        return sample

    async def record_logits(self, logits: "Array", token_index: int) -> float:
        """Record logits from a generation step."""
        if self._window is None:
            return 0.0

        start = time.time()
        entropy, variance = self.calculator.compute(logits)
        latency_ms = (time.time() - start) * 1000

        return await self._record(entropy, variance, token_index, latency_ms)

    async def record_entropy(
        self,
        entropy: float,
        variance: float = 0.0,
        token_index: int = 0,
    ) -> float:
        """Record pre-computed entropy value."""
        if self._window is None:
            return 0.0

        return await self._record(entropy, variance, token_index, 0.0)

    async def _record(
        self,
        entropy: float,
        variance: float,
        token_index: int,
        latency_ms: float,
    ) -> float:
        """Internal recording logic."""
        if self._window is None:
            return entropy

        status = self._window.add(entropy, variance, token_index)
        self._token_count += 1

        # Check circuit breaker
        if status.should_trip_circuit_breaker and self.on_circuit_breaker_tripped:
            self.on_circuit_breaker_tripped(status)

        # Emit periodic samples
        if self._token_count % self.config.emit_interval == 0:
            sample = self.calculator.create_sample(
                entropy=entropy,
                variance=variance,
                window_id=status.window_id,
                token_start=status.token_start,
                token_end=status.token_end,
                latency_ms=latency_ms,
                source=self.config.source,
                correlation_id=self._correlation_id,
            )
            if self.on_entropy_sample:
                self.on_entropy_sample(sample)

        # Track history
        self._sample_history.append((entropy, variance))
        if len(self._sample_history) > self.config.window_size:
            self._sample_history.pop(0)

        self._trajectory_buffer.append((entropy, variance, token_index))

        # Check for significant entropy change (> 0.5 delta)
        entropy_delta = abs(entropy - self._current_entropy)
        if entropy_delta > 0.5 and self._current_entropy != 0.0:
            transition = EntropyTransition(
                from_entropy=self._current_entropy,
                from_variance=self._current_variance,
                to_entropy=entropy,
                to_variance=variance,
                token_index=token_index,
            )
            self._transition_history.append(transition)
            if len(self._transition_history) > self.config.window_size:
                self._transition_history.pop(0)

            if self.on_entropy_changed:
                self.on_entropy_changed(transition)

        self._current_entropy = entropy
        self._current_variance = variance

        # Periodic distress check
        if token_index - self._last_distress_check >= 5:
            self._last_distress_check = token_index
            distress = self.pattern_detector.detect_distress(
                self._sample_history,
                token_index,
            )
            if distress and self.on_distress_detected:
                self.on_distress_detected(distress)

        return entropy

    @property
    def is_session_active(self) -> bool:
        return self._window is not None

    @property
    def current_token_count(self) -> int:
        return self._token_count

    @property
    def current_entropy(self) -> float:
        """Current entropy value. This IS the cognitive state."""
        return self._current_entropy

    @property
    def current_variance(self) -> float:
        """Current variance value."""
        return self._current_variance

    @property
    def transition_history(self) -> list[EntropyTransition]:
        """History of significant entropy transitions."""
        return self._transition_history.copy()

    @property
    def last_sample(self) -> EntropySample | None:
        return self._last_sample

    @property
    def last_trajectory(self) -> list[tuple[float, float, int]]:
        return self._trajectory_buffer.copy()

    @property
    def requires_caution(self) -> bool:
        """Check if current entropy/variance warrants caution."""
        return self.classifier.requires_caution(self._current_entropy, self._current_variance)

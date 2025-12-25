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

Raw geometric measurements. No arbitrary magic numbers - all thresholds derived
from calibration.

Notes
-----
Requires calibrated baseline from EntropyCalibrationService.
Use z-scores (standard deviations from mean) for all comparisons.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend

# Import calibration-based classes from model_state_classifier
from modelcypher.core.domain.entropy.model_state_classifier import (
    CalibratedBaseline,
    ClassificationResult,
    ClassificationSnapshot,
    EntropyStateThresholds,
    ModelStateClassifier,
    ModelStateSignals,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# EntropyTransition (baseline-relative)
# =============================================================================


@dataclass(frozen=True)
class EntropyTransition:
    """Records an entropy transition during generation.

    Uses z-score delta for significance testing. No magic numbers.
    """

    from_entropy: float
    from_variance: float
    to_entropy: float
    to_variance: float
    from_z_score: float
    to_z_score: float
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
    def z_score_delta(self) -> float:
        """Change in z-score. THE key metric for significance."""
        return self.to_z_score - self.from_z_score

    @property
    def is_escalation(self) -> bool:
        """Z-score increased by more than 1σ (statistically significant)."""
        return self.z_score_delta > 1.0

    @property
    def is_recovery(self) -> bool:
        """Z-score decreased by more than 1σ (statistically significant)."""
        return self.z_score_delta < -1.0

    @property
    def is_significant(self) -> bool:
        """Z-score changed by more than 1σ in either direction."""
        return abs(self.z_score_delta) > 1.0


# Backward compatibility alias
StateTransition = EntropyTransition


@dataclass
class EntropySample:
    """Semantic entropy measurement from a generation window.

    Attributes
    ----------
    id : str
        Unique sample identifier.
    window_id : str
        Window identifier.
    token_start : int
        Starting token index.
    token_end : int
        Ending token index.
    logit_entropy : float
        Entropy from logits (always available).
    top_k_variance : float
        Variance of top-K logits.
    z_score : float, optional
        Z-score relative to baseline (REQUIRED for meaningful comparison).
    sep_entropy : float, optional
        Entropy from SEP probe.
    sep_layers : list of int, optional
        Layers used for SEP probe.
    sep_confidence : float, optional
        Confidence of SEP probe prediction.
    semantic_volume : float, optional
        Semantic volume (expensive computation).
    sample_count : int, optional
        Number of samples used for volume.
    pca_dimensions : int, optional
        PCA dimensions for volume.
    computed_at : datetime
        Timestamp of computation.
    latency_ms : float
        Computation latency in milliseconds.
    source : str, optional
        Source identifier.
    correlation_id : str, optional
        Correlation ID for tracking.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_start: int = 0
    token_end: int = 0

    # Plan 1: Logit Entropy (always available)
    logit_entropy: float = 0.0
    top_k_variance: float = 0.0

    # Z-score relative to baseline (REQUIRED for meaningful comparison)
    z_score: float | None = None

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

    def is_low_entropy(self, baseline: CalibratedBaseline) -> bool:
        """Check if entropy is low relative to calibrated baseline."""
        return baseline.is_low_entropy(self.best_entropy_estimate)

    def is_high_entropy(self, baseline: CalibratedBaseline) -> bool:
        """Check if entropy is high relative to calibrated baseline."""
        return baseline.is_high_entropy(self.best_entropy_estimate)

    def should_trip_circuit_breaker(self, baseline: CalibratedBaseline) -> bool:
        """Check if entropy exceeds calibrated circuit breaker threshold."""
        return baseline.should_trip_circuit_breaker(self.best_entropy_estimate)

    def get_z_score(self, baseline: CalibratedBaseline) -> float:
        """Compute z-score relative to baseline."""
        return baseline.z_score(self.best_entropy_estimate)


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
    mean_z_score: float
    reason: str


# =============================================================================
# EntropyWindow (baseline-relative)
# =============================================================================


@dataclass
class EntropyWindowStatus:
    """Status of the sliding entropy window."""

    window_id: str
    sample_count: int
    current_entropy: float
    current_z_score: float
    moving_average: float
    entropy_std_dev: float
    token_start: int
    token_end: int
    consecutive_high_count: int
    should_trip_circuit_breaker: bool


class EntropyWindow:
    """Sliding window for entropy statistics.

    REQUIRES a calibrated baseline. Uses z-scores for all thresholds.
    """

    def __init__(self, baseline: CalibratedBaseline, window_size: int = 20):
        """Create entropy window.

        Args:
            baseline: Calibrated baseline from EntropyCalibrationService. REQUIRED.
            window_size: Number of samples to track.
        """
        self.window_id = str(uuid.uuid4())
        self.window_size = window_size
        self._baseline = baseline
        self.samples: list[tuple[float, float, float, int]] = []  # (entropy, variance, z_score, tokenIndex)
        self.consecutive_high_count = 0
        self.circuit_breaker_tripped = False

    def add(self, entropy: float, variance: float, token_index: int) -> EntropyWindowStatus:
        """Add a sample to the window."""
        z_score = self._baseline.z_score(entropy)
        self.samples.append((entropy, variance, z_score, token_index))
        if len(self.samples) > self.window_size:
            self.samples.pop(0)

        # Track consecutive high entropy (z > 1.5)
        if z_score > 1.5:
            self.consecutive_high_count += 1
        else:
            self.consecutive_high_count = 0

        # Trip circuit breaker on sustained high entropy or extreme outlier
        if self.consecutive_high_count >= 5 or z_score > 3.0:
            self.circuit_breaker_tripped = True

        # Also check absolute threshold from calibration
        if entropy >= self._baseline.percentile_95:
            self.circuit_breaker_tripped = True

        return self.status()

    def status(self) -> EntropyWindowStatus:
        """Get current window status."""
        if not self.samples:
            return EntropyWindowStatus(
                window_id=self.window_id,
                sample_count=0,
                current_entropy=0.0,
                current_z_score=0.0,
                moving_average=0.0,
                entropy_std_dev=0.0,
                token_start=0,
                token_end=0,
                consecutive_high_count=0,
                should_trip_circuit_breaker=False,
            )

        entropies = [s[0] for s in self.samples]
        z_scores = [s[2] for s in self.samples]
        current = entropies[-1]
        current_z = z_scores[-1]
        avg = sum(entropies) / len(entropies)
        variance = sum((e - avg) ** 2 for e in entropies) / len(entropies)
        std_dev = math.sqrt(variance)

        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self.samples),
            current_entropy=current,
            current_z_score=current_z,
            moving_average=avg,
            entropy_std_dev=std_dev,
            token_start=self.samples[0][3],
            token_end=self.samples[-1][3],
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
            z_score=status.current_z_score,
            source=source,
            correlation_id=correlation_id,
        )


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
        z_score: float | None = None,
    ) -> EntropySample:
        """Create an EntropySample from computed values."""
        return EntropySample(
            window_id=window_id,
            token_start=token_start,
            token_end=token_end,
            logit_entropy=entropy,
            top_k_variance=variance,
            z_score=z_score,
            latency_ms=latency_ms,
            source=source,
            correlation_id=correlation_id,
        )


# =============================================================================
# EntropyPatternDetector (baseline-relative)
# =============================================================================


@dataclass
class PatternConfig:
    """Configuration for entropy pattern detection.

    Uses z-scores for thresholds. No magic numbers.
    """

    min_samples: int = 5
    high_z_score_threshold: float = 1.5
    """Z-score threshold for high entropy (1.5σ above mean)."""

    low_variance_threshold: float = 0.2
    """Variance threshold (scale-independent, kept as is)."""

    sustained_count: int = 3


class EntropyPatternDetector:
    """Detects distress patterns from entropy history.

    REQUIRES a calibrated baseline for z-score computation.
    """

    def __init__(self, baseline: CalibratedBaseline, config: PatternConfig | None = None):
        """Create pattern detector.

        Args:
            baseline: Calibrated baseline from EntropyCalibrationService. REQUIRED.
            config: Optional pattern configuration.
        """
        self._baseline = baseline
        self.config = config or PatternConfig()

    def detect_distress(
        self,
        samples: list[tuple[float, float]],  # (entropy, variance)
        token_index: int = 0,
    ) -> DistressDetection | None:
        """
        Detect distress pattern: sustained high z-score + low variance.

        Returns DistressDetection if distress detected, None otherwise.
        """
        if len(samples) < self.config.min_samples:
            return None

        recent = samples[-self.config.sustained_count :]

        # Check for sustained high z-score + low variance
        distress_count = 0
        z_scores = []
        for e, v in recent:
            z = self._baseline.z_score(e)
            z_scores.append(z)
            if z >= self.config.high_z_score_threshold and v <= self.config.low_variance_threshold:
                distress_count += 1

        if distress_count >= self.config.sustained_count:
            mean_e = sum(e for e, _ in recent) / len(recent)
            mean_v = sum(v for _, v in recent) / len(recent)
            mean_z = sum(z_scores) / len(z_scores)
            confidence = distress_count / len(recent)

            return DistressDetection(
                detected=True,
                confidence=confidence,
                token_index=token_index,
                mean_entropy=mean_e,
                mean_variance=mean_v,
                mean_z_score=mean_z,
                reason=f"Sustained high z-score ({mean_z:+.2f}σ) with low variance ({mean_v:.2f})",
            )

        return None


# =============================================================================
# EntropyTracker (baseline-relative)
# =============================================================================


@dataclass
class EntropyTrackerConfig:
    """Configuration for EntropyTracker."""

    top_k: int = 10
    window_size: int = 20
    emit_interval: int = 1
    source: str = "EntropyTracker"
    z_score_change_threshold: float = 1.0
    """Z-score change threshold for significant transitions (1σ)."""


class EntropyTracker:
    """
    Coordinates entropy tracking for cognitive state analysis.

    REQUIRES a calibrated baseline. No magic numbers.
    Tracks raw entropy/variance with z-scores relative to baseline.

    Usage:
        baseline = calibration_service.load_calibration("model_calibration.json")
        tracker = EntropyTracker(baseline)
        tracker.start_session()

        # In generation loop:
        await tracker.record_logits(logits, token_index)

        sample = tracker.end_session()
    """

    def __init__(
        self,
        baseline: CalibratedBaseline,
        config: EntropyTrackerConfig | None = None,
    ):
        """Create entropy tracker.

        Args:
            baseline: Calibrated baseline from EntropyCalibrationService. REQUIRED.
            config: Optional tracker configuration.
        """
        self._baseline = baseline
        self.config = config or EntropyTrackerConfig()
        self.calculator = LogitEntropyCalculator(top_k=self.config.top_k)
        self.classifier = ModelStateClassifier(baseline)
        self.pattern_detector = EntropyPatternDetector(baseline)

        # Session state
        self._window: EntropyWindow | None = None
        self._correlation_id: str | None = None
        self._token_count: int = 0
        self._session_start: datetime | None = None

        # State tracking (raw values + z-scores)
        self._current_entropy: float = 0.0
        self._current_variance: float = 0.0
        self._current_z_score: float = 0.0
        self._transition_history: list[EntropyTransition] = []
        self._sample_history: list[tuple[float, float]] = []
        self._trajectory_buffer: list[tuple[float, float, float, int]] = []  # Added z_score
        self._last_distress_check: int = 0
        self._last_sample: EntropySample | None = None

        # Callbacks
        self.on_entropy_sample: Callable[[EntropySample], None] | None = None
        self.on_entropy_changed: Callable[[EntropyTransition], None] | None = None
        self.on_distress_detected: Callable[[DistressDetection], None] | None = None
        self.on_circuit_breaker_tripped: Callable[[EntropyWindowStatus], None] | None = None

    @property
    def baseline(self) -> CalibratedBaseline:
        """Get the calibrated baseline."""
        return self._baseline

    def start_session(self, correlation_id: str | None = None):
        """Start a new tracking session."""
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._window = EntropyWindow(self._baseline, window_size=self.config.window_size)
        self._token_count = 0
        self._session_start = datetime.now()
        self._current_entropy = 0.0
        self._current_variance = 0.0
        self._current_z_score = 0.0
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
        z_score = self._baseline.z_score(entropy)
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
                z_score=z_score,
            )
            if self.on_entropy_sample:
                self.on_entropy_sample(sample)

        # Track history
        self._sample_history.append((entropy, variance))
        if len(self._sample_history) > self.config.window_size:
            self._sample_history.pop(0)

        self._trajectory_buffer.append((entropy, variance, z_score, token_index))

        # Check for significant z-score change (> 1σ delta)
        z_score_delta = abs(z_score - self._current_z_score)
        if z_score_delta > self.config.z_score_change_threshold and self._current_z_score != 0.0:
            transition = EntropyTransition(
                from_entropy=self._current_entropy,
                from_variance=self._current_variance,
                to_entropy=entropy,
                to_variance=variance,
                from_z_score=self._current_z_score,
                to_z_score=z_score,
                token_index=token_index,
            )
            self._transition_history.append(transition)
            if len(self._transition_history) > self.config.window_size:
                self._transition_history.pop(0)

            if self.on_entropy_changed:
                self.on_entropy_changed(transition)

        self._current_entropy = entropy
        self._current_variance = variance
        self._current_z_score = z_score

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
        """Current entropy value. Raw measurement."""
        return self._current_entropy

    @property
    def current_variance(self) -> float:
        """Current variance value."""
        return self._current_variance

    @property
    def current_z_score(self) -> float:
        """Current z-score relative to baseline. THE key metric."""
        return self._current_z_score

    @property
    def transition_history(self) -> list[EntropyTransition]:
        """History of significant entropy transitions."""
        return self._transition_history.copy()

    @property
    def last_sample(self) -> EntropySample | None:
        return self._last_sample

    @property
    def last_trajectory(self) -> list[tuple[float, float, float, int]]:
        """Trajectory buffer: (entropy, variance, z_score, token_index)."""
        return self._trajectory_buffer.copy()

    @property
    def requires_caution(self) -> bool:
        """Check if current z-score warrants caution (> 1.5σ)."""
        return self._current_z_score > 1.5 or self.classifier.requires_caution(
            self._current_entropy, self._current_variance
        )

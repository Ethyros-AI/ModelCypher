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

Raw geometric measurements only. No thresholds or classification.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_entropy_calculator import LogitEntropyCalculator
from modelcypher.core.domain.entropy.model_state_classifier import CalibratedBaseline
from modelcypher.core.domain.geometry.numerical_stability import sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array


# =============================================================================
# EntropyTransition (baseline-relative)
# =============================================================================


@dataclass(frozen=True)
class EntropyTransition:
    """Records an entropy transition during generation."""

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
        """Change in z-score."""
        return self.to_z_score - self.from_z_score


@dataclass
class EntropySample:
    """Semantic entropy measurement from a generation window."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    token_start: int = 0
    token_end: int = 0

    # Plan 1: Logit Entropy (always available)
    logit_entropy: float = 0.0
    top_k_variance: float = 0.0

    # Z-score relative to baseline (optional, computed on demand)
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
        """Preferred entropy: SEP if available, else logit."""
        return self.sep_entropy if self.sep_entropy is not None else self.logit_entropy

    def get_z_score(self, baseline: CalibratedBaseline) -> float:
        """Compute z-score relative to baseline."""
        return baseline.z_score(self.best_entropy_estimate)


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


class EntropyWindow:
    """Sliding window for entropy statistics."""

    def __init__(
        self,
        baseline: CalibratedBaseline,
        *,
        window_size: int,
    ):
        """Create entropy window."""
        self.window_id = str(uuid.uuid4())
        self.window_size = window_size
        self._baseline = baseline
        self.samples: list[tuple[float, float, float, int]] = []  # (entropy, variance, z_score, tokenIndex)

    def add(self, entropy: float, variance: float, token_index: int) -> EntropyWindowStatus:
        """Add a sample to the window."""
        z_score = self._baseline.z_score(entropy)
        self.samples.append((entropy, variance, z_score, token_index))
        if len(self.samples) > self.window_size:
            self.samples.pop(0)

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
            )

        entropies = [s[0] for s in self.samples]
        z_scores = [s[2] for s in self.samples]
        current = entropies[-1]
        current_z = z_scores[-1]
        _b = get_default_backend()
        entropy_arr = _b.array(entropies)
        avg_arr = _b.mean(entropy_arr)
        std_arr = _b.std(entropy_arr)
        _b.eval(avg_arr, std_arr)
        avg = float(_b.to_scalar(avg_arr))
        std_dev = float(_b.to_scalar(std_arr)) if len(entropies) > 1 else 0.0

        return EntropyWindowStatus(
            window_id=self.window_id,
            sample_count=len(self.samples),
            current_entropy=current,
            current_z_score=current_z,
            moving_average=avg,
            entropy_std_dev=std_dev,
            token_start=self.samples[0][3],
            token_end=self.samples[-1][3],
        )

    def to_entropy_sample(self, source: str, correlation_id: str | None = None) -> EntropySample:
        """Create an EntropySample from current window state."""
        status = self.status()
        if self.samples:
            _b = get_default_backend()
            variance_arr = _b.array([s[1] for s in self.samples])
            avg_var_arr = _b.mean(variance_arr)
            _b.eval(avg_var_arr)
            avg_variance = float(_b.to_scalar(avg_var_arr))
        else:
            avg_variance = 0.0

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
# EntropyTracker
# =============================================================================


class EntropyTracker:
    """Coordinates entropy tracking for cognitive state analysis.

    All parameters are derived from the calibrated baseline:
    - window_size: sqrt(baseline sample count)
    - emit_interval: sqrt(window_size)
    - variance: full-vocabulary logit variance (no top-k truncation)
    """

    def __init__(
        self,
        baseline: CalibratedBaseline,
        source: str = "entropy_tracker",
    ):
        """Create entropy tracker.

        Args:
            baseline: Calibrated baseline for z-score computation.
            source: Source identifier for sample metadata.
        """
        self._baseline = baseline
        self._source = source

        if not hasattr(baseline, "sample_count"):
            raise ValueError("baseline.sample_count required for derived entropy windows")
        baseline_n = int(baseline.sample_count)
        if baseline_n <= 0:
            raise ValueError("baseline.sample_count must be positive for derived windows")
        backend = get_default_backend()
        self._window_size = int(sqrt_scalar(float(baseline_n), backend))
        if self._window_size <= 0:
            raise ValueError("derived window_size must be positive")

        # Derive emit_interval from window_size without arbitrary ratios
        self._emit_interval = int(sqrt_scalar(float(self._window_size), backend))
        if self._emit_interval <= 0:
            raise ValueError("derived emit_interval must be positive")

        # Use full-vocabulary entropy/variance (no top-k truncation)
        self.calculator = LogitEntropyCalculator()

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
        self._trajectory_buffer: list[tuple[float, float, float, int]] = []
        self._last_sample: EntropySample | None = None

        # Callbacks
        self.on_entropy_sample: Callable[[EntropySample], None] | None = None
        self.on_entropy_changed: Callable[[EntropyTransition], None] | None = None

    @property
    def baseline(self) -> CalibratedBaseline:
        """Get the calibrated baseline."""
        return self._baseline

    def start_session(self, correlation_id: str | None = None):
        """Start a new tracking session."""
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._window = EntropyWindow(
            self._baseline,
            window_size=self._window_size,
        )
        self._token_count = 0
        self._session_start = datetime.now()
        self._current_entropy = 0.0
        self._current_variance = 0.0
        self._current_z_score = 0.0
        self._transition_history = []
        self._sample_history = []
        self._trajectory_buffer = []

    def end_session(self) -> EntropySample | None:
        """End the tracking session and return final sample."""
        if self._window is None:
            return None

        status = self._window.status()
        if status.sample_count == 0:
            self._window = None
            return None

        sample = self._window.to_entropy_sample(
            source=self._source,
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
        has_previous = self._token_count > 0
        self._token_count += 1

        # Emit periodic samples
        if self._token_count % self._emit_interval == 0:
            sample = EntropySample(
                window_id=status.window_id,
                token_start=status.token_start,
                token_end=status.token_end,
                logit_entropy=entropy,
                top_k_variance=variance,
                z_score=z_score,
                latency_ms=latency_ms,
                source=self._source,
                correlation_id=self._correlation_id,
            )
            if self.on_entropy_sample:
                self.on_entropy_sample(sample)

        # Track history
        self._sample_history.append((entropy, variance))
        if len(self._sample_history) > self._window_size:
            self._sample_history.pop(0)

        self._trajectory_buffer.append((entropy, variance, z_score, token_index))

        # Record transition for every step after the first
        if has_previous:
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
            if len(self._transition_history) > self._window_size:
                self._transition_history.pop(0)

            if self.on_entropy_changed:
                self.on_entropy_changed(transition)

        self._current_entropy = entropy
        self._current_variance = variance
        self._current_z_score = z_score

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
        """Current z-score relative to baseline."""
        return self._current_z_score

    @property
    def transition_history(self) -> list[EntropyTransition]:
        """History of entropy transitions."""
        return self._transition_history.copy()

    @property
    def last_sample(self) -> EntropySample | None:
        return self._last_sample

    @property
    def last_trajectory(self) -> list[tuple[float, float, float, int]]:
        """Trajectory buffer: (entropy, variance, z_score, token_index)."""
        return self._trajectory_buffer.copy()

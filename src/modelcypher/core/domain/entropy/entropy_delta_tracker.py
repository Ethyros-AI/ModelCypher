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
Entropy Delta Tracker for Dual-Path Security Analysis.

Ported 1:1 from the reference Swift implementation.

Compares entropy between base model (no adapter) and adapter-modified model
at each token to detect potential backdoor behavior. High anomaly scores
(base uncertain + adapter confident) signal potential security issues.

Research Hypothesis:
Legitimate adapters narrow distributions within domains the base model understands.
Malicious backdoors force navigation to unexpected regions, creating detectable
entropy disagreement.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Awaitable
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.entropy_tracker import (
    LogitEntropyCalculator,
    ModelStateClassifier,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis
from modelcypher.core.domain.entropy.model_state import ModelState

logger = logging.getLogger(__name__)


@dataclass
class EntropyDeltaTrackerConfig:
    """Configuration for EntropyDeltaTracker."""

    # Top-K used for variance calculation (entropy is full-vocab).
    top_k: int = 10

    # Anomaly score threshold for alerts.
    anomaly_threshold: float = 0.6

    # Consecutive high-anomaly samples before circuit breaker trips.
    consecutive_anomaly_count: int = 3

    # Whether to compute variance (slightly more expensive).
    compute_variance: bool = True

    # Source identifier for emitted signals.
    source: str = "EntropyDeltaTracker"

    @classmethod
    def default(cls) -> "EntropyDeltaTrackerConfig":
        return cls()

    @classmethod
    def aggressive(cls) -> "EntropyDeltaTrackerConfig":
        """Aggressive config for high-security scenarios."""
        return cls(
            anomaly_threshold=0.4,
            consecutive_anomaly_count=2,
        )

    @classmethod
    def relaxed(cls) -> "EntropyDeltaTrackerConfig":
        """Relaxed config for trusted adapters."""
        return cls(
            anomaly_threshold=0.8,
            consecutive_anomaly_count=5,
        )


@dataclass
class PendingEntropyData:
    """Pre-computed entropy data to avoid MLXArray transfer across async boundaries."""

    token_index: int
    generated_token: int
    base_entropy: float
    base_top_k_variance: float
    base_state: ModelState
    base_top_token: int
    adapter_entropy: float
    adapter_top_k_variance: float
    adapter_state: ModelState
    adapter_top_token: int
    base_surprisal: float | None = None
    base_approval_probability: float | None = None
    normalized_approval_score: float | None = None
    base_approved_top_k: bool | None = None
    kl_divergence_adapter_to_base: float | None = None
    latency_ms: float = 0.0


class EntropyDeltaTracker:
    """
    Coordinates dual-path entropy tracking for LoRA adapter security analysis.

    Compares entropy between base model (no adapter) and adapter-modified model
    at each token to detect potential backdoor behavior. High anomaly scores
    (base uncertain + adapter confident) signal potential security issues.

    Usage:
        tracker = EntropyDeltaTracker()
        tracker.start_session(correlation_id=generation_id)

        # In dual-path generation loop:
        sample = await tracker.record_dual_entropy(
            base_logits=base_logits,
            adapter_logits=adapter_logits,
            token_index=i,
            generated_token=token_id
        )

        result = tracker.end_session()
    """

    def __init__(
        self,
        config: EntropyDeltaTrackerConfig | None = None,
        classifier: ModelStateClassifier | None = None,
        backend: "Backend | None" = None,
    ) -> None:
        self.config = config or EntropyDeltaTrackerConfig.default()
        self._backend = backend or get_default_backend()
        self.calculator = LogitEntropyCalculator(top_k=self.config.top_k, backend=self._backend)
        self.classifier = classifier or ModelStateClassifier()

        # Session state
        self._session_active: bool = False
        self._correlation_id: UUID | None = None
        self._session_start: datetime | None = None
        self._samples: list[EntropyDeltaSample] = []
        self._consecutive_anomalies: int = 0
        self._circuit_breaker_tripped: bool = False
        self._circuit_breaker_trip_index: int | None = None

        # Callbacks
        self.on_delta_sample: Callable[[EntropyDeltaSample], Awaitable[None]] | None = None
        self.on_anomaly_detected: Callable[[EntropyDeltaSample], Awaitable[None]] | None = None
        self.on_circuit_breaker_tripped: Callable[[list[EntropyDeltaSample]], Awaitable[None]] | None = None

    def start_session(self, correlation_id: UUID | None = None) -> None:
        """
        Start a new tracking session.

        Args:
            correlation_id: Optional ID for tracing related signals.
        """
        self._session_active = True
        self._correlation_id = correlation_id or uuid4()
        self._session_start = datetime.utcnow()
        self._samples = []
        self._consecutive_anomalies = 0
        self._circuit_breaker_tripped = False
        self._circuit_breaker_trip_index = None

        logger.info(f"Started security scan session: {self._correlation_id}")

    async def record_dual_entropy(
        self,
        base_logits: "Array",
        adapter_logits: "Array",
        token_index: int,
        generated_token: int,
    ) -> EntropyDeltaSample:
        """
        Record dual entropy from base and adapter logits.

        Computes entropy for both paths, creates a delta sample, and checks
        for anomalies. May trigger circuit breaker if consecutive anomalies
        exceed the configured threshold.

        Args:
            base_logits: Logits from base model (no adapter).
            adapter_logits: Logits from adapter-modified model.
            token_index: Current token position in generation.
            generated_token: The token ID that was actually generated.

        Returns:
            The computed delta sample.
        """
        start_time = time.perf_counter()

        # Compute entropy for both paths
        base_entropy, base_variance = self.calculator.compute(base_logits)
        adapter_entropy, adapter_variance = self.calculator.compute(adapter_logits)

        # Get top token predictions
        base_top_token = self._get_top_token(base_logits)
        adapter_top_token = self._get_top_token(adapter_logits)

        # Classify states
        base_state = self.classifier.classify(base_entropy, base_variance)
        adapter_state = self.classifier.classify(adapter_entropy, adapter_variance)

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Create sample
        sample = EntropyDeltaSample.create(
            token_index=token_index,
            generated_token=generated_token,
            base_entropy=base_entropy,
            base_top_k_variance=base_variance,
            base_state=base_state,
            base_top_token=base_top_token,
            adapter_entropy=adapter_entropy,
            adapter_top_k_variance=adapter_variance,
            adapter_state=adapter_state,
            adapter_top_token=adapter_top_token,
            latency_ms=latency_ms,
            correlation_id=self._correlation_id,
            source=self.config.source,
        )

        self._samples.append(sample)

        # Check for anomalies
        await self._check_anomalies(sample)

        # Invoke callback
        if self.on_delta_sample:
            await self.on_delta_sample(sample)

        return sample

    async def record_entropy_from_data(self, data: PendingEntropyData) -> EntropyDeltaSample:
        """
        Record entropy from pre-computed data.

        Avoids MLXArray transfer across async boundaries.

        Args:
            data: Pre-computed entropy data.

        Returns:
            The created delta sample.
        """
        sample = EntropyDeltaSample.create(
            token_index=data.token_index,
            generated_token=data.generated_token,
            base_entropy=data.base_entropy,
            base_top_k_variance=data.base_top_k_variance,
            base_state=data.base_state,
            base_top_token=data.base_top_token,
            adapter_entropy=data.adapter_entropy,
            adapter_top_k_variance=data.adapter_top_k_variance,
            adapter_state=data.adapter_state,
            adapter_top_token=data.adapter_top_token,
            base_surprisal=data.base_surprisal,
            base_approval_probability=data.base_approval_probability,
            normalized_approval_score=data.normalized_approval_score,
            base_approved_top_k=data.base_approved_top_k,
            kl_divergence_adapter_to_base=data.kl_divergence_adapter_to_base,
            latency_ms=data.latency_ms,
            correlation_id=self._correlation_id,
            source=self.config.source,
        )

        self._samples.append(sample)

        # Check for anomalies
        await self._check_anomalies(sample)

        # Invoke callback
        if self.on_delta_sample:
            await self.on_delta_sample(sample)

        return sample

    def end_session(self) -> EntropyDeltaSessionResult:
        """
        End the tracking session and return results.

        Returns:
            Summary of the security scan session.
        """
        if not self._session_active:
            logger.warning("end_session called without active session")
            return self._create_empty_result()

        self._session_active = False
        session_end = datetime.utcnow()

        # Compute statistics
        total_tokens = len(self._samples)
        anomaly_count = sum(
            1 for s in self._samples if s.anomaly_score >= self.config.anomaly_threshold
        )
        max_anomaly_score = max((s.anomaly_score for s in self._samples), default=0.0)
        avg_delta = (
            sum(s.delta for s in self._samples) / total_tokens if total_tokens > 0 else 0.0
        )
        disagreement_count = sum(1 for s in self._samples if s.top_token_disagreement)
        disagreement_rate = disagreement_count / total_tokens if total_tokens > 0 else 0.0
        backdoor_signature_count = sum(1 for s in self._samples if s.has_backdoor_signature)

        # Approval-based statistics
        approval_anomaly_count = sum(1 for s in self._samples if s.has_approval_anomaly)
        surprisal_values = [s.base_surprisal for s in self._samples if s.base_surprisal is not None]
        avg_base_surprisal = sum(surprisal_values) / len(surprisal_values) if surprisal_values else None
        max_base_surprisal = max(surprisal_values) if surprisal_values else None

        # Compute conflict analysis
        kl_divergences = [s.kl_divergence_adapter_to_base for s in self._samples]
        base_approved_top_k = [s.base_approved_top_k for s in self._samples]
        conflict_analysis = ConflictAnalysis.compute(kl_divergences, base_approved_top_k)

        result = EntropyDeltaSessionResult(
            session_id=self._correlation_id or uuid4(),
            correlation_id=self._correlation_id,
            session_start=self._session_start or session_end,
            session_end=session_end,
            total_tokens=total_tokens,
            anomaly_count=anomaly_count,
            max_anomaly_score=max_anomaly_score,
            avg_delta=avg_delta,
            disagreement_rate=disagreement_rate,
            backdoor_signature_count=backdoor_signature_count,
            approval_anomaly_count=approval_anomaly_count,
            avg_base_surprisal=avg_base_surprisal,
            max_base_surprisal=max_base_surprisal,
            conflict_analysis=conflict_analysis,
            circuit_breaker_tripped=self._circuit_breaker_tripped,
            circuit_breaker_trip_index=self._circuit_breaker_trip_index,
            samples=self._samples.copy(),
        )

        logger.info(
            f"Security scan complete: {total_tokens} tokens, {anomaly_count} anomalies, "
            f"max score: {max_anomaly_score:.2f}, assessment: {result.security_assessment.value}"
        )

        return result

    async def _check_anomalies(self, sample: EntropyDeltaSample) -> None:
        """Check for anomalies and manage circuit breaker."""
        is_anomaly = sample.anomaly_score >= self.config.anomaly_threshold

        if is_anomaly:
            self._consecutive_anomalies += 1

            # Invoke callback
            if self.on_anomaly_detected:
                await self.on_anomaly_detected(sample)

            logger.warning(
                f"Anomaly detected at token {sample.token_index}: "
                f"score={sample.anomaly_score:.2f}, delta={sample.delta:.2f}, "
                f"baseState={sample.base_state.value}, adapterState={sample.adapter_state.value}"
            )

            # Check circuit breaker
            if (
                not self._circuit_breaker_tripped
                and self._consecutive_anomalies >= self.config.consecutive_anomaly_count
            ):
                await self._trip_circuit_breaker(sample.token_index)
        else:
            # Reset consecutive count on non-anomaly
            self._consecutive_anomalies = 0

    async def _trip_circuit_breaker(self, token_index: int) -> None:
        """Trip the security circuit breaker."""
        self._circuit_breaker_tripped = True
        self._circuit_breaker_trip_index = token_index

        recent_samples = self._samples[-self.config.consecutive_anomaly_count :]

        # Invoke callback
        if self.on_circuit_breaker_tripped:
            await self.on_circuit_breaker_tripped(recent_samples)

        logger.error(
            f"Security circuit breaker TRIPPED at token {token_index}: "
            f"{self.config.consecutive_anomaly_count} consecutive anomalies detected"
        )

    def _get_top_token(self, logits: "Array") -> int:
        """Get the top predicted token from logits."""
        b = self._backend
        # Get the last token's logits if multi-dimensional
        if logits.ndim > 1:
            if logits.ndim == 3:
                last_logits = logits[0, -1, :]
            else:
                last_logits = logits[-1, :]
        else:
            last_logits = logits

        # Find argmax
        top_index = b.argmax(last_logits)
        b.eval(top_index)

        top_index_np = b.to_numpy(top_index)
        return int(top_index_np.item())

    def _create_empty_result(self) -> EntropyDeltaSessionResult:
        """Create an empty session result."""
        now = datetime.utcnow()
        return EntropyDeltaSessionResult(
            session_id=uuid4(),
            correlation_id=None,
            session_start=now,
            session_end=now,
            total_tokens=0,
            anomaly_count=0,
            max_anomaly_score=0.0,
            avg_delta=0.0,
            disagreement_rate=0.0,
            backdoor_signature_count=0,
            circuit_breaker_tripped=False,
        )

    # State accessors

    @property
    def is_session_active(self) -> bool:
        """Whether a session is currently active."""
        return self._session_active

    @property
    def is_circuit_breaker_tripped(self) -> bool:
        """Whether the circuit breaker has tripped in the current session."""
        return self._circuit_breaker_tripped

    @property
    def current_sample_count(self) -> int:
        """Current sample count in the active session."""
        return len(self._samples)

    @property
    def current_consecutive_anomalies(self) -> int:
        """Current consecutive anomaly count."""
        return self._consecutive_anomalies

    @property
    def correlation_id(self) -> UUID | None:
        """Current session correlation ID."""
        return self._correlation_id

    # Convenience constructors

    @classmethod
    def standalone(cls) -> "EntropyDeltaTracker":
        """Create a tracker with default configuration."""
        return cls()

    @classmethod
    def monitor_only(cls) -> "EntropyDeltaTracker":
        """Create a tracker that only logs anomalies (no circuit breaker)."""
        config = EntropyDeltaTrackerConfig(
            anomaly_threshold=0.6,
            consecutive_anomaly_count=999999,  # Never trip
            source="EntropyDeltaTracker.monitor",
        )
        return cls(config=config)

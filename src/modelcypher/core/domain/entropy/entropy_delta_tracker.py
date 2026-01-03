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

Notes
-----
Legitimate adapters narrow distributions within domains the base model understands.
Malicious backdoors force navigation to unexpected regions, creating detectable
entropy disagreement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Awaitable, Callable
from uuid import UUID, uuid4

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_entropy_calculator import LogitEntropyCalculator

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend
from modelcypher.core.domain.entropy.conflict_score import ConflictAnalysis
from modelcypher.core.domain.entropy.entropy_delta_sample import (
    EntropyDeltaSample,
    EntropyDeltaSessionResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntropyDeltaCalibration:
    """Calibration derived from baseline anomaly scores."""

    anomaly_threshold: float
    source: str

    @classmethod
    def from_baseline_distribution(
        cls,
        anomaly_score_samples: list[float],
        *,
        source: str,
    ) -> "EntropyDeltaCalibration":
        """Derive threshold from baseline anomaly score distribution.

        Args:
            anomaly_score_samples: Anomaly scores from baseline model runs.
        """
        if not anomaly_score_samples:
            raise ValueError("anomaly_score_samples required for calibration")

        from modelcypher.core.domain.geometry.numerical_stability import (
            division_epsilon,
            find_magnitude_gap_threshold,
        )
        backend = get_default_backend()
        samples_arr = backend.array(anomaly_score_samples)
        sorted_samples_arr = backend.sort(samples_arr)
        backend.eval(sorted_samples_arr)
        sorted_samples = [float(v) for v in backend.tolist(sorted_samples_arr)]
        eps = division_epsilon(backend, backend.array([0.0]))
        threshold = float(find_magnitude_gap_threshold(sorted_samples, eps=eps))

        return cls(
            anomaly_threshold=threshold,
            source=source,
        )


@dataclass
class PendingEntropyData:
    """Pre-computed entropy data to avoid MLXArray transfer across async boundaries.

    Attributes
    ----------
    token_index : int
        Token position in sequence.
    generated_token : int
        The generated token ID.
    base_entropy : float
        Entropy from base model.
    base_top_k_variance : float
        Variance of base model top-K logits.
    base_top_token : int
        Top predicted token from base model.
    adapter_entropy : float
        Entropy from adapter model.
    adapter_top_k_variance : float
        Variance of adapter model top-K logits.
    adapter_top_token : int
        Top predicted token from adapter model.
    base_logit_margin : float, optional
        Margin between base top logit and generated token logit.
    base_token_logit : float, optional
        Base model logit for generated token.
    base_rank_fraction : float, optional
        Normalized rank fraction for the generated token.
    base_frontier_hit : bool, optional
        Whether token is in base model frontier.
    kl_divergence_adapter_to_base : float, optional
        KL divergence from adapter to base distribution.
    latency_ms : float
        Computation latency in milliseconds.
    """

    token_index: int
    generated_token: int
    base_entropy: float
    base_top_k_variance: float
    base_top_token: int
    adapter_entropy: float
    adapter_top_k_variance: float
    adapter_top_token: int
    base_logit_margin: float | None = None
    base_token_logit: float | None = None
    base_rank_fraction: float | None = None
    base_frontier_hit: bool | None = None
    kl_divergence_adapter_to_base: float | None = None
    latency_ms: float = 0.0


class EntropyDeltaTracker:
    """
    Coordinates dual-path entropy tracking for LoRA adapter security analysis.

    Compares entropy between base model (no adapter) and adapter-modified model
    at each token to detect potential backdoor behavior. High anomaly scores
    (base uncertain + adapter confident) signal potential security issues.

    Usage:
        calibration = EntropyDeltaCalibration.from_baseline_distribution(
            baseline_scores,
            source="entropy_delta_tracker",
        )
        tracker = EntropyDeltaTracker(calibration)
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
        calibration: EntropyDeltaCalibration,
        backend: "Backend | None" = None,
    ) -> None:
        self.calibration = calibration
        self._backend = backend or get_default_backend()
        self.calculator = LogitEntropyCalculator(backend=self._backend)

        # Session state
        self._session_active: bool = False
        self._correlation_id: UUID | None = None
        self._session_start: datetime | None = None
        self._samples: list[EntropyDeltaSample] = []

        # Callbacks
        self.on_delta_sample: Callable[[EntropyDeltaSample], Awaitable[None]] | None = None
        self.on_anomaly_detected: Callable[[EntropyDeltaSample], Awaitable[None]] | None = None

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
        exceed the calibrated threshold.

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

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Create sample from raw entropy values
        sample = EntropyDeltaSample.create(
            token_index=token_index,
            generated_token=generated_token,
            base_entropy=base_entropy,
            base_top_k_variance=base_variance,
            base_top_token=base_top_token,
            adapter_entropy=adapter_entropy,
            adapter_top_k_variance=adapter_variance,
            adapter_top_token=adapter_top_token,
            latency_ms=latency_ms,
            correlation_id=self._correlation_id,
            source=self.calibration.source,
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
            base_top_token=data.base_top_token,
            adapter_entropy=data.adapter_entropy,
            adapter_top_k_variance=data.adapter_top_k_variance,
            adapter_top_token=data.adapter_top_token,
            base_logit_margin=data.base_logit_margin,
            base_token_logit=data.base_token_logit,
            base_rank_fraction=data.base_rank_fraction,
            base_frontier_hit=data.base_frontier_hit,
            kl_divergence_adapter_to_base=data.kl_divergence_adapter_to_base,
            latency_ms=data.latency_ms,
            correlation_id=self._correlation_id,
            source=self.calibration.source,
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
        b = self._backend
        if total_tokens > 0:
            scores_arr = b.array([s.anomaly_score for s in self._samples])
            anomaly_mask = scores_arr >= self.calibration.anomaly_threshold
            anomaly_count_arr = b.sum(b.astype(anomaly_mask, scores_arr.dtype))
            max_anomaly_arr = b.max(scores_arr)

            delta_arr = b.array([s.delta for s in self._samples])
            avg_delta_arr = b.mean(delta_arr)

            disagreement_arr = b.array(
                [1.0 if s.top_token_disagreement else 0.0 for s in self._samples]
            )
            disagreement_count_arr = b.sum(disagreement_arr)
            total_arr = b.array([float(total_tokens)])
            disagreement_rate_arr = disagreement_count_arr / total_arr

            b.eval(
                anomaly_count_arr,
                max_anomaly_arr,
                avg_delta_arr,
                disagreement_count_arr,
                disagreement_rate_arr,
            )
            anomaly_count = int(b.to_scalar(anomaly_count_arr))
            max_anomaly_score = float(b.to_scalar(max_anomaly_arr))
            avg_delta = float(b.to_scalar(avg_delta_arr))
            disagreement_count = int(b.to_scalar(disagreement_count_arr))
            disagreement_rate = float(b.to_scalar(disagreement_rate_arr))
        else:
            anomaly_count = 0
            max_anomaly_score = 0.0
            avg_delta = 0.0
            disagreement_count = 0
            disagreement_rate = 0.0
        # Logit margin statistics
        margin_values = [
            s.base_logit_margin for s in self._samples if s.base_logit_margin is not None
        ]
        if margin_values:
            margin_arr = b.array(margin_values)
            avg_margin_arr = b.mean(margin_arr)
            max_margin_arr = b.max(margin_arr)
            b.eval(avg_margin_arr, max_margin_arr)
            avg_base_logit_margin = float(b.to_scalar(avg_margin_arr))
            max_base_logit_margin = float(b.to_scalar(max_margin_arr))
        else:
            avg_base_logit_margin = None
            max_base_logit_margin = None

        # Compute conflict analysis
        kl_divergences = [s.kl_divergence_adapter_to_base for s in self._samples]
        base_frontier_hit = [s.base_frontier_hit for s in self._samples]
        conflict_analysis = ConflictAnalysis.compute(kl_divergences, base_frontier_hit)

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
            avg_base_logit_margin=avg_base_logit_margin,
            max_base_logit_margin=max_base_logit_margin,
            conflict_analysis=conflict_analysis,
            samples=self._samples.copy(),
        )

        logger.info(
            f"Security scan complete: {total_tokens} tokens, {anomaly_count} anomalies, "
            f"max score: {max_anomaly_score:.2f}"
        )

        return result

    async def _check_anomalies(self, sample: EntropyDeltaSample) -> None:
        """Check for anomalies and emit callbacks."""
        is_anomaly = sample.anomaly_score >= self.calibration.anomaly_threshold

        if is_anomaly:
            # Invoke callback
            if self.on_anomaly_detected:
                await self.on_anomaly_detected(sample)

            logger.warning(
                f"Anomaly detected at token {sample.token_index}: "
                f"score={sample.anomaly_score:.2f}, delta={sample.delta:.2f}, "
                f"baseEntropy={sample.base_entropy:.2f}, adapterEntropy={sample.adapter_entropy:.2f}"
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

        return int(b.to_scalar(top_index))

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
        )

    # State accessors

    @property
    def is_session_active(self) -> bool:
        """Whether a session is currently active."""
        return self._session_active

    @property
    def current_sample_count(self) -> int:
        """Current sample count in the active session."""
        return len(self._samples)

    @property
    def correlation_id(self) -> UUID | None:
        """Current session correlation ID."""
        return self._correlation_id

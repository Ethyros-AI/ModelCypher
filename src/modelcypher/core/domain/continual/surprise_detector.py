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
Surprise Detector - Identify novel information for knowledge encoding.

This module detects "surprising" events during inference - situations where
the model's prediction significantly differs from the actual outcome. These
surprises are candidates for knowledge encoding because they represent
information the model doesn't already know.

The key insight (from Titans/Hope architecture): Use gradient as a surprise
signal. High gradient = model would update significantly = high surprise.

Math:
    surprise = -log P(actual_token | context)  # Cross-entropy loss
    surprise_normalized = surprise / mean_surprise  # Relative to baseline

We use multiple surprise signals:
1. **Token surprise**: Cross-entropy of actual vs predicted token
2. **Rank surprise**: How far from top-1 was the actual token?
3. **Activation surprise**: How different is this activation from recent history?

Events that trigger encoding:
- Token surprise > threshold (model was wrong)
- Information is novel (not seen in recent context)
- User provides correction or new information

References:
    - Titans: Learning to Memorize at Test Time (arXiv:2501.00663)
    - Nested Learning (Google Research, NeurIPS 2025)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class SurpriseEvent:
    """A detected surprise event with raw measurements.

    Returns raw metrics without hardcoded thresholds. The caller decides
    whether to encode based on precision-derived or empirical constraints.

    Attributes:
        timestep: When the surprise occurred.
        token_id: The actual token that caused surprise.
        predicted_token_id: The model's top-1 prediction.
        token_surprise: Cross-entropy surprise (-log P(actual)).
        token_surprise_baseline: Mean token surprise over observed history.
        token_surprise_zscore: Z-score of token_surprise vs baseline.
        rank_surprise: Rank of actual token in predictions (0 = top-1).
        rank_log: log(rank + 1) for log-scale comparison.
        activation_surprise: L2 distance from recent activation mean, normalized.
        percentile: Where this event falls in recent surprise distribution [0, 1].
        context_tokens: Recent context for encoding reference.
    """

    timestep: int
    token_id: int
    predicted_token_id: int
    token_surprise: float
    token_surprise_baseline: float
    token_surprise_zscore: float
    rank_surprise: int
    rank_log: float
    activation_surprise: float
    percentile: float
    context_tokens: tuple[int, ...]

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestep": self.timestep,
            "token_id": self.token_id,
            "predicted_token_id": self.predicted_token_id,
            "token_surprise": self.token_surprise,
            "token_surprise_baseline": self.token_surprise_baseline,
            "token_surprise_zscore": self.token_surprise_zscore,
            "rank_surprise": self.rank_surprise,
            "rank_log": self.rank_log,
            "activation_surprise": self.activation_surprise,
            "percentile": self.percentile,
            "context_tokens": list(self.context_tokens),
        }


class SurpriseDetector:
    """Detects surprising events during inference for knowledge encoding.

    Maintains baselines of recent predictions and activations to identify
    when the model encounters genuinely novel information.

    Returns raw metrics without hardcoded thresholds. The caller decides
    encoding thresholds based on z-scores, percentiles, or other criteria.

    Usage:
        detector = SurpriseDetector()

        for logits, actual_token, hidden_state in inference_loop:
            event = detector.detect(logits, actual_token, hidden_state)

            # Caller decides threshold based on raw metrics
            # Options: token_surprise_zscore, percentile, activation_surprise
            if should_encode(event):  # Application-specific decision
                encoder.encode(event, hidden_state, weights)
    """

    def __init__(
        self,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the surprise detector.

        Args:
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        # Surprise history for percentile computation (unbounded by design)
        self._surprise_history: list[float] = []
        self._surprise_count = 0
        self._surprise_mean = 0.0
        self._surprise_m2 = 0.0

        # Token history for context
        self._token_history: deque[int] = deque()

        # Running activation statistics for activation surprise
        self._activation_mean: Array | None = None
        self._activation_count = 0

        self._timestep = 0

    def detect(
        self,
        logits: Array,
        actual_token_id: int,
        hidden_state: Array | None = None,
    ) -> SurpriseEvent:
        """Detect surprise metrics for this prediction.

        Returns raw measurements without hardcoded thresholds.
        The caller decides encoding thresholds based on:
        - Machine precision constraints
        - Empirical baselines from this session
        - Application-specific requirements

        Args:
            logits: Model output logits [vocab_size] or [..., vocab_size].
            actual_token_id: The actual next token.
            hidden_state: Optional hidden state for activation surprise.

        Returns:
            SurpriseEvent with raw surprise metrics.
        """
        b = self._backend

        # Flatten logits to 1D
        flat_logits = self._flatten_logits(logits)

        # Compute token surprise (cross-entropy)
        token_surprise = self._compute_token_surprise(flat_logits, actual_token_id)

        # Compute rank surprise
        predicted_token_id, rank_surprise = self._compute_rank_surprise(
            flat_logits, actual_token_id
        )

        # Compute activation surprise if hidden state provided
        if hidden_state is not None:
            activation_surprise = self._compute_activation_surprise(hidden_state)
        else:
            activation_surprise = 0.0

        # Compute baseline statistics
        baseline, std = self._compute_baseline_stats()
        zscore = self._compute_zscore(token_surprise, baseline, std)

        # Compute rank log for scale-invariant comparison
        from modelcypher.core.domain.geometry.numerical_stability import log_scalar
        rank_log = log_scalar(float(rank_surprise + 1), b)

        # Compute percentile in recent history (raw, no threshold)
        percentile = self._compute_percentile(token_surprise)

        # Get context
        context = tuple(self._token_history)

        # Update histories
        self._surprise_history.append(token_surprise)
        self._update_surprise_stats(token_surprise)
        self._token_history.append(actual_token_id)
        if hidden_state is not None:
            self._update_activation_stats(hidden_state)
        self._timestep += 1

        return SurpriseEvent(
            timestep=self._timestep - 1,
            token_id=actual_token_id,
            predicted_token_id=predicted_token_id,
            token_surprise=token_surprise,
            token_surprise_baseline=baseline,
            token_surprise_zscore=zscore,
            rank_surprise=rank_surprise,
            rank_log=rank_log,
            activation_surprise=activation_surprise,
            percentile=percentile,
            context_tokens=context,
        )

    def _flatten_logits(self, logits: Array) -> Array:
        """Flatten logits to 1D vocabulary vector."""
        if logits.ndim == 3:
            return logits[0, -1, :]
        elif logits.ndim == 2:
            return logits[0, :]
        return logits

    def _compute_token_surprise(self, logits: Array, token_id: int) -> float:
        """Compute cross-entropy surprise for the actual token.

        surprise = -log P(actual_token)
        """
        b = self._backend

        # Numerically stable softmax
        max_logit = b.max(logits)
        shifted = logits - max_logit
        exp_shifted = b.exp(shifted)
        log_sum_exp = b.log(b.sum(exp_shifted))

        # Log probability of actual token
        actual_logit = b.take(logits, b.array([token_id]), axis=0)
        log_prob = actual_logit - max_logit - log_sum_exp

        b.eval(log_prob)
        surprise = -float(b.to_scalar(log_prob))

        return surprise

    def _compute_rank_surprise(
        self, logits: Array, token_id: int
    ) -> tuple[int, int]:
        """Compute rank of actual token in predictions.

        Returns (predicted_token_id, rank_of_actual).
        """
        b = self._backend

        # Get sorted indices (descending by logit value)
        sorted_indices = b.argsort(logits)
        n = int(sorted_indices.shape[0])

        # Reverse to get descending order
        reverse_idx = b.arange(n - 1, -1, -1)
        sorted_indices = b.take(sorted_indices, reverse_idx, axis=0)

        b.eval(sorted_indices)

        # Top prediction
        top_idx = b.take(sorted_indices, b.array([0]), axis=0)
        b.eval(top_idx)
        predicted_token_id = int(b.to_scalar(top_idx))

        # Find rank of actual token
        sorted_list = b.tolist(sorted_indices)
        try:
            rank = sorted_list.index(token_id)
        except ValueError:
            rank = n  # Token not found (shouldn't happen)

        return predicted_token_id, rank

    def _compute_activation_surprise(self, hidden_state: Array) -> float:
        """Compute how surprising this activation is relative to running mean."""
        if self._activation_mean is None or self._activation_count < 2:
            return 0.0

        b = self._backend

        # Flatten hidden state
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        # Distance from mean (L2 normalized)
        diff = hidden_state - self._activation_mean
        dist_sq = b.sum(diff * diff)
        mean_sq = b.sum(self._activation_mean * self._activation_mean)
        b.eval(dist_sq, mean_sq)

        dist = float(b.to_scalar(dist_sq)) ** 0.5
        norm = float(b.to_scalar(mean_sq)) ** 0.5

        if norm == 0:
            return 0.0

        return dist / norm

    def _update_activation_stats(self, hidden_state: Array) -> None:
        """Update running activation statistics."""
        b = self._backend

        # Flatten
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        self._activation_count += 1
        if self._activation_mean is None:
            self._activation_mean = hidden_state
        else:
            delta = hidden_state - self._activation_mean
            self._activation_mean = self._activation_mean + delta / float(
                self._activation_count
            )
            b.eval(self._activation_mean)

    def _compute_baseline_stats(self) -> tuple[float, float]:
        """Compute baseline mean and std from surprise history.

        Returns:
            (mean, std) of recent surprises. Both 0.0 if insufficient history.
        """
        if self._surprise_count < 2:
            return 0.0, 0.0

        n = self._surprise_count
        mean = self._surprise_mean
        variance = self._surprise_m2 / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5

        return mean, std

    def _update_surprise_stats(self, surprise: float) -> None:
        """Update running mean/variance of surprise via Welford's algorithm."""
        self._surprise_count += 1
        if self._surprise_count == 1:
            self._surprise_mean = surprise
            self._surprise_m2 = 0.0
            return

        delta = surprise - self._surprise_mean
        self._surprise_mean += delta / self._surprise_count
        delta2 = surprise - self._surprise_mean
        self._surprise_m2 += delta * delta2

    def _compute_zscore(self, value: float, mean: float, std: float) -> float:
        """Compute z-score of a value given mean and std.

        Returns 0.0 if std is too small (insufficient variation).
        Uses dtype-derived epsilon for numerical stability.
        """
        from modelcypher.core.domain.geometry.numerical_stability import machine_epsilon

        # Get dtype-derived machine epsilon
        b = self._backend
        eps = float(machine_epsilon(b, b.array([1.0])))

        if std < eps:
            return 0.0
        return (value - mean) / std

    def _compute_percentile(self, surprise: float) -> float:
        """Compute percentile of a surprise value in recent history.

        Returns 0.5 if insufficient history (neutral).
        This is a raw measurement - caller decides what percentile threshold to use.
        """
        if len(self._surprise_history) < 2:
            return 0.5

        count_below = sum(1 for s in self._surprise_history if s < surprise)
        return count_below / len(self._surprise_history)

    def get_baseline_surprise(self) -> float:
        """Get current baseline surprise level."""
        if self._surprise_count == 0:
            return 0.0
        return self._surprise_mean

    def get_surprise_percentile(self, surprise: float) -> float:
        """Get percentile of a surprise score in recent history."""
        return self._compute_percentile(surprise)

    def reset(self) -> None:
        """Reset all state."""
        self._surprise_history.clear()
        self._surprise_count = 0
        self._surprise_mean = 0.0
        self._surprise_m2 = 0.0
        self._token_history.clear()
        self._activation_mean = None
        self._activation_count = 0
        self._timestep = 0

    @property
    def timestep(self) -> int:
        """Current timestep."""
        return self._timestep

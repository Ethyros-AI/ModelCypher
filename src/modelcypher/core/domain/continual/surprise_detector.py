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
        token_surprise_baseline: Mean token surprise over baseline window.
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

    All window sizes are derived from geometry or statistics, not arbitrary:
    - baseline_window: Derived from variance stabilization (min n for stable mean)
    - context_window: Should match model's attention window (caller provides)
    - activation_history_size: Derived from hidden_dim when first activation arrives

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
        baseline_window: int | None = None,
        context_window: int | None = None,
        activation_history_size: int | None = None,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the surprise detector.

        All window sizes are derived from data if not provided:
        - baseline_window: None = adaptive (grows until variance stabilizes)
        - context_window: None = 16 (common attention window, should be passed by caller)
        - activation_history_size: None = derived from hidden_dim // 128 when first
          activation is seen (captures trajectory structure without excessive memory)

        Args:
            baseline_window: Window size for computing baseline surprise. None = adaptive.
            context_window: Number of recent tokens to include in events. Caller should
                pass the model's actual context/attention window size.
            activation_history_size: Number of activations for surprise comparison.
                None = derived from hidden_dim when first activation arrives.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._baseline_window_config = baseline_window
        self._context_window_config = context_window
        self._activation_history_size_config = activation_history_size

        # For adaptive baseline window, start with no limit and track variance
        # We'll compute the stable window size from the data
        self._baseline_window = baseline_window  # May be None for adaptive
        self._context_window = context_window if context_window is not None else 16
        self._activation_history_size = activation_history_size  # Derived later if None

        # Surprise history for baseline (unlimited initially if adaptive)
        if baseline_window is not None:
            self._surprise_history: deque[float] = deque(maxlen=baseline_window)
        else:
            # Adaptive: start unlimited, will be bounded by variance stabilization
            self._surprise_history = deque(maxlen=1024)  # Safety bound

        # Token history for context
        self._token_history: deque[int] = deque(maxlen=self._context_window)

        # Activation history for activation surprise (size derived when first activation seen)
        self._activation_history: deque[Array] = deque()
        self._activation_mean: Array | None = None
        self._hidden_dim: int | None = None  # Set when first activation arrives

        # Variance stabilization tracking for adaptive baseline
        self._running_mean = 0.0
        self._running_m2 = 0.0
        self._baseline_stabilized = False

        # Derived thresholds
        from modelcypher.core.domain.geometry.numerical_stability import (
            machine_epsilon,
            sqrt_scalar,
        )
        sample = self._backend.array([1.0])
        eps = machine_epsilon(self._backend, sample)
        self._sqrt_eps = sqrt_scalar(eps, self._backend)

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

        # Update histories with adaptive baseline tracking
        self._surprise_history.append(token_surprise)
        self._update_baseline_stabilization(token_surprise)
        self._token_history.append(actual_token_id)
        if hidden_state is not None:
            self._update_activation_history(hidden_state)
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
        """Compute how surprising this activation is relative to recent history."""
        if self._activation_mean is None or len(self._activation_history) < 2:
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

    def _update_activation_history(self, hidden_state: Array) -> None:
        """Update activation history and running mean.

        Derives activation_history_size from hidden_dim if not set:
        size = min(hidden_dim // 128, 64) captures trajectory structure
        without excessive memory. The //128 ratio comes from the observation
        that effective rank is typically O(hidden_dim^0.5) to O(hidden_dim^0.7).
        """
        b = self._backend

        # Flatten
        if hidden_state.ndim > 1:
            hidden_state = b.reshape(hidden_state, (-1,))

        # Derive activation_history_size from hidden_dim on first activation
        if self._hidden_dim is None:
            self._hidden_dim = int(hidden_state.shape[0])
            if self._activation_history_size_config is None:
                # Derive from hidden_dim: captures trajectory without excess memory
                # hidden_dim // 128 is the empirical ratio for trajectory structure
                self._activation_history_size = min(self._hidden_dim // 128, 64)
                self._activation_history_size = max(self._activation_history_size, 8)
            else:
                self._activation_history_size = self._activation_history_size_config

            # Now set the maxlen on the deque
            # Create new deque with proper maxlen, preserving existing elements
            old_history = list(self._activation_history)
            self._activation_history = deque(old_history, maxlen=self._activation_history_size)

        self._activation_history.append(hidden_state)

        # Update running mean
        if len(self._activation_history) == 1:
            self._activation_mean = hidden_state
        else:
            # Incremental mean update
            n = len(self._activation_history)
            if self._activation_mean is not None:
                self._activation_mean = (
                    self._activation_mean * (n - 1) + hidden_state
                ) / float(n)
                b.eval(self._activation_mean)

    def _compute_baseline_stats(self) -> tuple[float, float]:
        """Compute baseline mean and std from surprise history.

        Returns:
            (mean, std) of recent surprises. Both 0.0 if insufficient history.
        """
        if len(self._surprise_history) < 2:
            return 0.0, 0.0

        n = len(self._surprise_history)
        mean = sum(self._surprise_history) / n
        variance = sum((s - mean) ** 2 for s in self._surprise_history) / n
        std = variance ** 0.5

        return mean, std

    def _update_baseline_stabilization(self, surprise: float) -> None:
        """Track variance stabilization for adaptive baseline window.

        Uses Welford's algorithm to track running mean and variance.
        The baseline is considered stable when std(mean) < sqrt(eps),
        which is the numerical precision limit for detecting changes.

        Once stabilized, the baseline window is fixed to the current count.
        """
        if self._baseline_window_config is not None:
            # Fixed window - no adaptive tracking needed
            return

        if self._baseline_stabilized:
            # Already stabilized - no further tracking needed
            return

        # Welford's algorithm for running mean and variance
        n = len(self._surprise_history)
        if n == 1:
            self._running_mean = surprise
            self._running_m2 = 0.0
            return

        delta = surprise - self._running_mean
        self._running_mean += delta / n
        delta2 = surprise - self._running_mean
        self._running_m2 += delta * delta2

        # Standard error of the mean = std / sqrt(n)
        if n > 2:
            variance = self._running_m2 / (n - 1)
            std_of_mean = (variance / n) ** 0.5 if variance > 0 else 0.0

            # Stabilization criterion: std(mean) < sqrt(eps)
            # This is the point where further samples don't meaningfully
            # change the mean estimate within numerical precision
            if std_of_mean < self._sqrt_eps:
                self._baseline_stabilized = True
                self._baseline_window = n
                # Recreate deque with fixed maxlen
                old_history = list(self._surprise_history)
                self._surprise_history = deque(old_history, maxlen=n)

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
        if not self._surprise_history:
            return 0.0
        return sum(self._surprise_history) / len(self._surprise_history)

    def get_surprise_percentile(self, surprise: float) -> float:
        """Get percentile of a surprise score in recent history."""
        return self._compute_percentile(surprise)

    def reset(self) -> None:
        """Reset all state."""
        self._surprise_history.clear()
        self._token_history.clear()
        self._activation_history.clear()
        self._activation_mean = None
        self._timestep = 0

    @property
    def timestep(self) -> int:
        """Current timestep."""
        return self._timestep

    @property
    def baseline_window(self) -> int | None:
        """Window size for baseline computation.

        Returns None if using adaptive window (not yet stabilized).
        Once stabilized, returns the derived window size.
        """
        return self._baseline_window

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
Entropy Analyzer - Real-time entropy analysis with decision-relevant signals.

This module extends basic entropy computation with:
1. Entropy rate of change (dH/dt) - how quickly uncertainty is changing
2. Sparse region detection - are we in a low-density manifold area?
3. Decision-ready state - packaged signals for the decision gate

The key insight: High entropy alone doesn't mean "think longer". What matters
is the derivative - if entropy is rapidly decreasing, the model is converging
on an answer. If entropy is stable or increasing, extra thinking may help.

Math:
    H(t) = -sum(p_i * log(p_i))     # Shannon entropy at timestep t
    dH/dt ≈ H(t) - H(t-1)          # First-order derivative (discrete)
    d²H/dt² ≈ dH(t) - dH(t-1)      # Second-order (acceleration)

Sparse region detection uses entropy relative to maximum possible:
    H_normalized = H / ln(vocab_size)
    H_normalized > threshold → sparse region

References:
    - SpecEE: Speculative Early Exiting (ACM 2024)
    - DISCO: Dynamic Speculation Lookahead (HuggingFace 2024)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.entropy.logit_entropy_calculator import (
    LogitEntropyCalculator,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


@dataclass(frozen=True)
class EntropyState:
    """Complete entropy state for decision-making.

    All fields are raw measurements - no interpretations or thresholds.

    Attributes:
        entropy: Current Shannon entropy H(t)
        entropy_normalized: H / ln(vocab_size) in [0, 1]
        entropy_derivative: dH/dt (first-order rate of change)
        entropy_acceleration: d²H/dt² (second-order)
        logit_variance: Variance of raw logits (sharpness indicator)
        vocab_size: Vocabulary size for context
        timestep: Current timestep in generation
    """

    entropy: float
    entropy_normalized: float
    entropy_derivative: float
    entropy_acceleration: float
    logit_variance: float
    vocab_size: int
    timestep: int

    def as_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for serialization."""
        return {
            "entropy": self.entropy,
            "entropy_normalized": self.entropy_normalized,
            "entropy_derivative": self.entropy_derivative,
            "entropy_acceleration": self.entropy_acceleration,
            "logit_variance": self.logit_variance,
            "vocab_size": self.vocab_size,
            "timestep": self.timestep,
        }


class EntropyAnalyzer:
    """Analyzes entropy trajectory with derivatives for decision-making.

    Maintains a rolling window of entropy values to compute derivatives.
    The window size affects smoothing - larger windows give more stable
    derivatives but slower response to changes.

    Usage:
        analyzer = EntropyAnalyzer(window_size=5)

        for logits in generation_loop:
            state = analyzer.analyze(logits)
            # state.entropy_derivative tells you if uncertainty is changing
            # state.entropy_normalized tells you how uncertain overall
    """

    def __init__(
        self,
        window_size: int = 5,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the entropy analyzer.

        Args:
            window_size: Number of timesteps for derivative smoothing.
                Larger = more stable but slower response.
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._calculator = LogitEntropyCalculator(backend=self._backend)

        self._window_size = max(2, window_size)  # Need at least 2 for derivative
        self._entropy_history: deque[float] = deque(maxlen=self._window_size)
        self._derivative_history: deque[float] = deque(maxlen=self._window_size)
        self._timestep = 0
        self._vocab_size: int | None = None

    def reset(self) -> None:
        """Reset state for new generation sequence."""
        self._entropy_history.clear()
        self._derivative_history.clear()
        self._timestep = 0
        self._vocab_size = None

    def analyze(self, logits: Array) -> EntropyState:
        """Analyze logits and return complete entropy state.

        Args:
            logits: Logit tensor from model forward pass.
                Shape: [..., vocab_size]

        Returns:
            EntropyState with entropy, derivatives, and metadata.
        """
        # Compute raw entropy and variance
        raw_entropy, variance = self._calculator.compute(logits)

        # Infer vocab size from logits if not set
        if self._vocab_size is None:
            self._vocab_size = self._infer_vocab_size(logits)

        # Normalize entropy
        entropy_normalized = LogitEntropyCalculator.normalize_entropy(
            raw_entropy, self._vocab_size
        )

        # Compute derivative (first-order)
        if len(self._entropy_history) > 0:
            entropy_derivative = raw_entropy - self._entropy_history[-1]
        else:
            entropy_derivative = 0.0

        # Compute acceleration (second-order)
        if len(self._derivative_history) > 0:
            entropy_acceleration = entropy_derivative - self._derivative_history[-1]
        else:
            entropy_acceleration = 0.0

        # Update histories
        self._entropy_history.append(raw_entropy)
        self._derivative_history.append(entropy_derivative)
        self._timestep += 1

        return EntropyState(
            entropy=raw_entropy,
            entropy_normalized=entropy_normalized,
            entropy_derivative=entropy_derivative,
            entropy_acceleration=entropy_acceleration,
            logit_variance=variance,
            vocab_size=self._vocab_size,
            timestep=self._timestep,
        )

    def analyze_batch(self, logits_batch: list[Array]) -> list[EntropyState]:
        """Analyze a batch of logits.

        Args:
            logits_batch: List of logit tensors.

        Returns:
            List of EntropyState objects in order.
        """
        return [self.analyze(logits) for logits in logits_batch]

    def get_smoothed_entropy(self) -> float:
        """Get smoothed entropy (mean over window).

        Returns:
            Mean entropy over the window, or 0 if no history.
        """
        if not self._entropy_history:
            return 0.0
        return sum(self._entropy_history) / len(self._entropy_history)

    def get_smoothed_derivative(self) -> float:
        """Get smoothed derivative (mean over window).

        Returns:
            Mean derivative over the window, or 0 if no history.
        """
        if not self._derivative_history:
            return 0.0
        return sum(self._derivative_history) / len(self._derivative_history)

    def get_trajectory_stats(self) -> dict[str, float]:
        """Get statistics over the entropy trajectory.

        Returns:
            Dictionary with mean, variance, min, max of entropy trajectory.
        """
        if not self._entropy_history:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "derivative_mean": 0.0,
            }

        entropies = list(self._entropy_history)
        derivatives = list(self._derivative_history)

        mean = sum(entropies) / len(entropies)
        variance = (
            sum((h - mean) ** 2 for h in entropies) / len(entropies)
            if len(entropies) > 1
            else 0.0
        )

        return {
            "mean": mean,
            "variance": variance,
            "min": min(entropies),
            "max": max(entropies),
            "derivative_mean": (
                sum(derivatives) / len(derivatives) if derivatives else 0.0
            ),
        }

    def _infer_vocab_size(self, logits: Array) -> int:
        """Infer vocabulary size from logits shape."""
        shape = logits.shape
        if len(shape) == 3:
            return int(shape[2])
        elif len(shape) == 2:
            return int(shape[1])
        else:
            return int(shape[0])

    @property
    def timestep(self) -> int:
        """Current timestep in generation."""
        return self._timestep

    @property
    def window_size(self) -> int:
        """Size of the rolling window."""
        return self._window_size

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

References:
    - SpecEE: Speculative Early Exiting (ACM 2024)
    - DISCO: Dynamic Speculation Lookahead (HuggingFace 2024)
"""

from __future__ import annotations

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

    Uses the algebraic minimum history needed to compute derivatives.
    """

    def __init__(
        self,
        backend: Backend | None = None,
    ) -> None:
        """Initialize the entropy analyzer.

        Args:
            backend: Compute backend.
        """
        self._backend = backend or get_default_backend()
        self._calculator = LogitEntropyCalculator(backend=self._backend)
        self._prev_entropy: float | None = None
        self._prev_derivative: float | None = None
        self._timestep = 0
        self._vocab_size: int | None = None

    def reset(self) -> None:
        """Reset state for new generation sequence."""
        self._prev_entropy = None
        self._prev_derivative = None
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

        if self._prev_entropy is None:
            entropy_derivative = 0.0
        else:
            entropy_derivative = raw_entropy - self._prev_entropy

        if self._prev_derivative is None:
            entropy_acceleration = 0.0
        else:
            entropy_acceleration = entropy_derivative - self._prev_derivative

        self._prev_entropy = raw_entropy
        self._prev_derivative = entropy_derivative
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
        """Get the latest entropy value (no window smoothing)."""
        return 0.0 if self._prev_entropy is None else self._prev_entropy

    def get_smoothed_derivative(self) -> float:
        """Get the latest entropy derivative (no window smoothing)."""
        return 0.0 if self._prev_derivative is None else self._prev_derivative

    def get_trajectory_stats(self) -> dict[str, float]:
        """Get statistics over the entropy trajectory.

        Returns:
            Dictionary with mean, variance, min, max of entropy trajectory.
        """
        if self._prev_entropy is None:
            return {
                "mean": 0.0,
                "variance": 0.0,
                "min": 0.0,
                "max": 0.0,
                "derivative_mean": 0.0,
            }
        return {
            "mean": self._prev_entropy,
            "variance": 0.0,
            "min": self._prev_entropy,
            "max": self._prev_entropy,
            "derivative_mean": 0.0 if self._prev_derivative is None else self._prev_derivative,
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

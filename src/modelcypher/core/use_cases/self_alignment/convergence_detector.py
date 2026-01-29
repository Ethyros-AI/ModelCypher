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

"""Convergence detection for geometric self-alignment.

Detects when the manifold has reached a stable state where:
1. Entropy variance < sqrt(eps) over a sliding window
2. Fundamental constant alignment is improving or stable
3. SVD signature is stable

The geometry tells us when we're done - no arbitrary epoch counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Backend

    from modelcypher.core.domain.geometry.manifold_entropy import ManifoldEntropyResult


@dataclass
class ConvergenceMetrics:
    """Metrics tracked for convergence detection."""

    # Entropy history
    entropy_values: List[float] = field(default_factory=list)

    # Alignment metric history (raw: complexity law r_squared)
    alignment_values: List[float] = field(default_factory=list)

    # SVD metric history (raw: mean_error)
    svd_quality_values: List[float] = field(default_factory=list)

    # Number of rounds without improvement
    rounds_without_improvement: int = 0

    # Best entropy seen
    best_entropy: float = float("inf")

    # Round at which best entropy was achieved
    best_entropy_round: int = 0


@dataclass
class ConvergenceResult:
    """Result of convergence check."""

    is_converged: bool
    reason: str
    metrics: ConvergenceMetrics

    # Detailed breakdown
    entropy_stable: bool = False
    alignment_improving: bool = False
    svd_stable: bool = False
    no_improvement_timeout: bool = False


class ConvergenceDetector:
    """Detect convergence of geometric self-alignment.

    Convergence is geometric, not epoch-based. We're done when:
    1. Entropy variance drops below sqrt(machine_epsilon)
    2. Alignment quality stops improving
    3. SVD signature stabilizes

    Usage:
        detector = ConvergenceDetector(window_size=20)

        for round in range(max_rounds):
            entropy_result = compute_entropy(...)
            detector.update(entropy_result, round)

            if detector.is_converged():
                break
    """

    def __init__(
        self,
        backend: "Backend | None" = None,
        window_size: int = 20,
        patience: int = 10,
    ) -> None:
        """Initialize convergence detector.

        Args:
            backend: Computational backend
            window_size: Number of rounds to consider for stability
            patience: Rounds without improvement before convergence
        """
        self._backend = backend or get_default_backend()
        self._window_size = window_size
        self._patience = patience
        self._metrics = ConvergenceMetrics()

        # Compute convergence threshold from machine epsilon
        ref = self._backend.array([1.0], dtype="float32")
        eps = machine_epsilon(self._backend, ref)
        self._sqrt_eps = sqrt_scalar(eps, self._backend)

    def update(
        self,
        entropy_result: "ManifoldEntropyResult",
        round_idx: int,
    ) -> None:
        """Update convergence metrics with new measurement.

        Args:
            entropy_result: Latest manifold entropy measurement
            round_idx: Current round number
        """
        # Track entropy
        self._metrics.entropy_values.append(entropy_result.total_entropy)

        # Track alignment metric (raw)
        alignment_value = 0.0
        if entropy_result.complexity_law is not None:
            alignment_value = entropy_result.complexity_law.r_squared
        self._metrics.alignment_values.append(alignment_value)

        # Track SVD metric (raw)
        svd_metric = 0.0
        if entropy_result.svd_signature is not None:
            svd_metric = entropy_result.svd_signature.mean_error
        self._metrics.svd_quality_values.append(svd_metric)

        # Check for improvement
        if entropy_result.total_entropy < self._metrics.best_entropy:
            self._metrics.best_entropy = entropy_result.total_entropy
            self._metrics.best_entropy_round = round_idx
            self._metrics.rounds_without_improvement = 0
        else:
            self._metrics.rounds_without_improvement += 1

    def check_convergence(self) -> ConvergenceResult:
        """Check if the optimization has converged.

        Returns:
            ConvergenceResult with detailed breakdown
        """
        n = len(self._metrics.entropy_values)

        if n < self._window_size:
            return ConvergenceResult(
                is_converged=False,
                reason=f"Not enough data: {n}/{self._window_size} rounds",
                metrics=self._metrics,
            )

        # Get recent window
        window = self._metrics.entropy_values[-self._window_size:]

        # Check entropy stability (variance < (sqrt_eps * scale)^2)
        mean_entropy = sum(window) / len(window)
        variance = sum((x - mean_entropy) ** 2 for x in window) / len(window)
        entropy_scale = max(1.0, abs(mean_entropy))
        entropy_threshold = (self._sqrt_eps * entropy_scale) ** 2
        entropy_stable = variance < entropy_threshold

        # Check alignment improvement
        alignment_window = self._metrics.alignment_values[-self._window_size:]
        first_half = alignment_window[:self._window_size // 2]
        second_half = alignment_window[self._window_size // 2:]
        first_mean = sum(first_half) / len(first_half) if first_half else 0
        second_mean = sum(second_half) / len(second_half) if second_half else 0
        alignment_scale = max(1.0, abs(first_mean), abs(second_mean))
        alignment_tol = self._sqrt_eps * alignment_scale
        alignment_improving = second_mean >= first_mean - alignment_tol

        # Check SVD signature stability
        svd_window = self._metrics.svd_quality_values[-self._window_size:]
        svd_mean = sum(svd_window) / len(svd_window) if svd_window else 0.0
        svd_variance = sum((x - svd_mean) ** 2 for x in svd_window) / len(svd_window)
        svd_scale = max(1.0, abs(svd_mean))
        svd_threshold = (self._sqrt_eps * svd_scale) ** 2
        svd_stable = svd_variance < svd_threshold

        # Check patience timeout
        no_improvement_timeout = (
            self._metrics.rounds_without_improvement >= self._patience
        )

        # Determine convergence
        is_converged = False
        reason = ""

        if entropy_stable and svd_stable:
            is_converged = True
            reason = "Entropy and SVD signature stabilized"
        elif no_improvement_timeout:
            is_converged = True
            reason = f"No improvement for {self._patience} rounds"
        elif entropy_stable and alignment_improving:
            is_converged = True
            reason = "Entropy stable with good alignment"

        return ConvergenceResult(
            is_converged=is_converged,
            reason=reason,
            metrics=self._metrics,
            entropy_stable=entropy_stable,
            alignment_improving=alignment_improving,
            svd_stable=svd_stable,
            no_improvement_timeout=no_improvement_timeout,
        )

    def is_converged(self) -> bool:
        """Quick check for convergence.

        Returns:
            True if converged, False otherwise
        """
        return self.check_convergence().is_converged

    def reset(self) -> None:
        """Reset convergence detector for a new run."""
        self._metrics = ConvergenceMetrics()

    @property
    def metrics(self) -> ConvergenceMetrics:
        """Get current convergence metrics."""
        return self._metrics

    @property
    def rounds_completed(self) -> int:
        """Number of rounds completed."""
        return len(self._metrics.entropy_values)

    @property
    def best_entropy(self) -> float:
        """Best entropy seen so far."""
        return self._metrics.best_entropy

    @property
    def current_entropy(self) -> float:
        """Most recent entropy value."""
        if self._metrics.entropy_values:
            return self._metrics.entropy_values[-1]
        return float("inf")


__all__ = [
    "ConvergenceDetector",
    "ConvergenceMetrics",
    "ConvergenceResult",
]

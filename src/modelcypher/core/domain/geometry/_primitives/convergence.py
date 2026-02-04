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

"""Unified convergence monitoring for iterative algorithms.

Provides dtype-derived convergence thresholds for algorithms like:
- Gromov-Wasserstein (Frank-Wolfe)
- Frechet mean (gradient descent)
- Generalized Procrustes
- Power iteration

All thresholds are derived from machine precision, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modelcypher.core.domain.geometry.numerical_stability import (
    log2_scalar,
    machine_epsilon,
    sqrt_scalar,
)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

__all__ = [
    "ConvergenceState",
    "ConvergenceMonitor",
]


@dataclass(frozen=True)
class ConvergenceState:
    """Immutable state for convergence monitoring.

    Attributes:
        iteration: Current iteration number (1-indexed).
        converged: Whether convergence criterion is met.
        abs_change: Absolute change from previous value.
        rel_change: Relative change from previous value.
        current_value: The current objective value.
    """

    iteration: int
    converged: bool
    abs_change: float
    rel_change: float
    current_value: float


class ConvergenceMonitor:
    """Unified convergence checking for iterative algorithms.

    Derives all thresholds from dtype precision:
    - abs_threshold = sqrt(machine_epsilon)
    - rel_threshold = machine_epsilon
    - min_iterations derived from problem size (log2)

    Usage:
        monitor = ConvergenceMonitor(backend, reference_array)
        for i in range(max_iterations):
            value = compute_objective()
            state = monitor.check(value)
            if state.converged:
                break
    """

    def __init__(
        self,
        backend: "Backend",
        reference_array: "Array",
        min_iterations: int | None = None,
        max_iterations: int | None = None,
    ):
        """Initialize with dtype-derived thresholds.

        Args:
            backend: The compute backend.
            reference_array: Array used to derive dtype (for epsilon).
            min_iterations: Minimum iterations before checking convergence.
                If None, derives from log2(max_dimension).
            max_iterations: Maximum iterations allowed.
                If None, derives as 10 * max_dimension.
        """
        self._backend = backend
        eps = machine_epsilon(backend, reference_array)

        # sqrt(eps) for absolute threshold (more lenient)
        self._abs_threshold = sqrt_scalar(eps, backend)
        # eps for relative threshold (tighter)
        self._rel_threshold = eps

        # Derive min iterations from problem size
        shape = backend.shape(reference_array)
        max_dim = max(int(d) for d in shape) if shape else 1
        max_dim = max(max_dim, 2)  # Ensure log2 is valid

        if min_iterations is not None:
            self._min_iterations = min_iterations
        else:
            # ceil(log2(max_dim)) ensures geometric convergence has time
            self._min_iterations = max(3, int(log2_scalar(float(max_dim), backend)) + 1)

        if max_iterations is not None:
            self._max_iterations = max_iterations
        else:
            # 10 * dimension is conservative for gradient-based methods
            self._max_iterations = 10 * max_dim

        self._prev_value: float | None = None
        self._iteration = 0

    @property
    def abs_threshold(self) -> float:
        """Get the absolute convergence threshold."""
        return self._abs_threshold

    @property
    def rel_threshold(self) -> float:
        """Get the relative convergence threshold."""
        return self._rel_threshold

    @property
    def min_iterations(self) -> int:
        """Get the minimum iterations before convergence check."""
        return self._min_iterations

    @property
    def max_iterations(self) -> int:
        """Get the maximum iterations allowed."""
        return self._max_iterations

    @property
    def iteration(self) -> int:
        """Get the current iteration count."""
        return self._iteration

    def check(self, current_value: float) -> ConvergenceState:
        """Check convergence and return state.

        Args:
            current_value: Current objective value.

        Returns:
            ConvergenceState with convergence status and diagnostics.
        """
        self._iteration += 1

        if self._prev_value is None:
            self._prev_value = current_value
            return ConvergenceState(
                iteration=self._iteration,
                converged=False,
                abs_change=float("inf"),
                rel_change=float("inf"),
                current_value=current_value,
            )

        abs_change = abs(current_value - self._prev_value)
        # Safe relative change (avoid division by zero)
        denom = max(abs(self._prev_value), self._abs_threshold)
        rel_change = abs_change / denom

        # Check convergence only after minimum iterations
        converged = False
        if self._iteration >= self._min_iterations:
            converged = (
                abs_change < self._abs_threshold or rel_change < self._rel_threshold
            )

        self._prev_value = current_value
        return ConvergenceState(
            iteration=self._iteration,
            converged=converged,
            abs_change=abs_change,
            rel_change=rel_change,
            current_value=current_value,
        )

    def should_stop(self, state: ConvergenceState) -> bool:
        """Check if iteration should stop (converged or max reached).

        Args:
            state: The current convergence state.

        Returns:
            True if should stop iterating.
        """
        return state.converged or state.iteration >= self._max_iterations

    def reset(self) -> None:
        """Reset the monitor for a new sequence."""
        self._prev_value = None
        self._iteration = 0

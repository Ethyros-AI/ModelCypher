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

"""Array validation and convergence monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .precision import machine_epsilon
from .scalars import log2_scalar, sqrt_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Array Validation
# =============================================================================


@dataclass(frozen=True)
class ArrayNumerics:
    """Results from array numeric validation.

    Attributes:
        nan_count: Number of NaN values.
        inf_count: Number of infinite values (positive or negative).
        nonfinite_count: Total non-finite values (NaN + Inf).
        total_elements: Total number of elements in array.
        is_healthy: True if array contains only finite values.
    """

    nan_count: int
    inf_count: int
    nonfinite_count: int
    total_elements: int

    @property
    def is_healthy(self) -> bool:
        """Check if array contains only finite values."""
        return self.nonfinite_count == 0

    @property
    def nan_fraction(self) -> float:
        """Fraction of elements that are NaN."""
        if self.total_elements == 0:
            return 0.0
        return self.nan_count / self.total_elements

    @property
    def inf_fraction(self) -> float:
        """Fraction of elements that are infinite."""
        if self.total_elements == 0:
            return 0.0
        return self.inf_count / self.total_elements


def validate_array_numerics(
    arr: "Array",
    backend: "Backend",
) -> ArrayNumerics:
    """Validate array numerics in a single pass.

    Efficiently checks for NaN, Inf, and non-finite values using
    vectorized operations. All computation stays on GPU.

    Args:
        arr: Array to validate.
        backend: Backend protocol implementation.

    Returns:
        ArrayNumerics with counts and health status.
    """
    is_nan = backend.isnan(arr)
    is_inf = backend.isinf(arr)
    is_finite = backend.isfinite(arr)
    backend.eval(is_nan, is_inf, is_finite)

    nan_sum = backend.sum(is_nan)
    inf_sum = backend.sum(is_inf)
    nonfinite_sum = backend.sum(~is_finite)
    backend.eval(nan_sum, inf_sum, nonfinite_sum)

    nan_count = int(float(backend.to_scalar(nan_sum)))
    inf_count = int(float(backend.to_scalar(inf_sum)))
    nonfinite_count = int(float(backend.to_scalar(nonfinite_sum)))

    shape = backend.shape(arr)
    total = 1
    for dim in shape:
        total *= int(dim)

    return ArrayNumerics(
        nan_count=nan_count,
        inf_count=inf_count,
        nonfinite_count=nonfinite_count,
        total_elements=total,
    )


def count_nan(arr: "Array", backend: "Backend") -> int:
    """Count NaN values in array."""
    is_nan = backend.isnan(arr)
    nan_sum = backend.sum(is_nan)
    backend.eval(nan_sum)
    return int(float(backend.to_scalar(nan_sum)))


def count_inf(arr: "Array", backend: "Backend") -> int:
    """Count infinite values in array."""
    is_inf = backend.isinf(arr)
    inf_sum = backend.sum(is_inf)
    backend.eval(inf_sum)
    return int(float(backend.to_scalar(inf_sum)))


def count_nonfinite(arr: "Array", backend: "Backend") -> int:
    """Count non-finite values (NaN or Inf) in array."""
    is_finite = backend.isfinite(arr)
    nonfinite_sum = backend.sum(~is_finite)
    backend.eval(nonfinite_sum)
    return int(float(backend.to_scalar(nonfinite_sum)))


# =============================================================================
# Convergence Monitoring
# =============================================================================


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
    """

    def __init__(
        self,
        backend: "Backend",
        reference_array: "Array",
        min_iterations: int | None = None,
        max_iterations: int | None = None,
    ):
        """Initialize with dtype-derived thresholds."""
        self._backend = backend
        eps = machine_epsilon(backend, reference_array)

        self._abs_threshold = sqrt_scalar(eps, backend)
        self._rel_threshold = eps

        shape = backend.shape(reference_array)
        max_dim = max(int(d) for d in shape) if shape else 1
        max_dim = max(max_dim, 2)

        if min_iterations is not None:
            self._min_iterations = min_iterations
        else:
            self._min_iterations = max(3, int(log2_scalar(float(max_dim), backend)) + 1)

        if max_iterations is not None:
            self._max_iterations = max_iterations
        else:
            self._max_iterations = 10 * max_dim

        self._prev_value: float | None = None
        self._iteration = 0

    @property
    def abs_threshold(self) -> float:
        return self._abs_threshold

    @property
    def rel_threshold(self) -> float:
        return self._rel_threshold

    @property
    def min_iterations(self) -> int:
        return self._min_iterations

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def iteration(self) -> int:
        return self._iteration

    def check(self, current_value: float) -> ConvergenceState:
        """Check convergence and return state."""
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
        denom = max(abs(self._prev_value), self._abs_threshold)
        rel_change = abs_change / denom

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
        """Check if iteration should stop."""
        return state.converged or state.iteration >= self._max_iterations

    def reset(self) -> None:
        """Reset the monitor for a new sequence."""
        self._prev_value = None
        self._iteration = 0


__all__ = [
    # Array validation
    "ArrayNumerics",
    "validate_array_numerics",
    "count_nan",
    "count_inf",
    "count_nonfinite",
    # Convergence monitoring
    "ConvergenceState",
    "ConvergenceMonitor",
]

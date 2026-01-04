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

"""Array validation utilities for numerical health checks.

Provides GPU-accelerated validation to detect NaN, Inf, and non-finite values
in arrays. Uses single-pass computation for efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

__all__ = [
    "ArrayNumerics",
    "validate_array_numerics",
    "count_nan",
    "count_inf",
    "count_nonfinite",
]


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

    This replaces multiple calls to count_nan(), count_inf(), etc.
    with a single efficient pass.

    Args:
        arr: Array to validate.
        backend: Backend protocol implementation.

    Returns:
        ArrayNumerics with counts and health status.
    """
    # Single pass: compute all masks at once
    is_nan = backend.isnan(arr)
    is_inf = backend.isinf(arr)
    is_finite = backend.isfinite(arr)

    # Batch eval for efficiency
    backend.eval(is_nan, is_inf, is_finite)

    # Compute counts
    nan_sum = backend.sum(is_nan)
    inf_sum = backend.sum(is_inf)
    nonfinite_sum = backend.sum(~is_finite)
    backend.eval(nan_sum, inf_sum, nonfinite_sum)

    nan_count = int(float(backend.to_scalar(nan_sum)))
    inf_count = int(float(backend.to_scalar(inf_sum)))
    nonfinite_count = int(float(backend.to_scalar(nonfinite_sum)))

    # Compute total elements
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
    """Count NaN values in array.

    Prefer validate_array_numerics() for multiple checks.

    Args:
        arr: Array to check.
        backend: Backend protocol implementation.

    Returns:
        Number of NaN values.
    """
    is_nan = backend.isnan(arr)
    nan_sum = backend.sum(is_nan)
    backend.eval(nan_sum)
    return int(float(backend.to_scalar(nan_sum)))


def count_inf(arr: "Array", backend: "Backend") -> int:
    """Count infinite values in array.

    Prefer validate_array_numerics() for multiple checks.

    Args:
        arr: Array to check.
        backend: Backend protocol implementation.

    Returns:
        Number of infinite values.
    """
    is_inf = backend.isinf(arr)
    inf_sum = backend.sum(is_inf)
    backend.eval(inf_sum)
    return int(float(backend.to_scalar(inf_sum)))


def count_nonfinite(arr: "Array", backend: "Backend") -> int:
    """Count non-finite values (NaN or Inf) in array.

    Prefer validate_array_numerics() for multiple checks.

    Args:
        arr: Array to check.
        backend: Backend protocol implementation.

    Returns:
        Number of non-finite values.
    """
    is_finite = backend.isfinite(arr)
    nonfinite_sum = backend.sum(~is_finite)
    backend.eval(nonfinite_sum)
    return int(float(backend.to_scalar(nonfinite_sum)))

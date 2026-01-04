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

"""Dtype-derived epsilon and threshold utilities.

All epsilons and tolerances are derived from tensor precision, not arbitrary constants.
Use these functions instead of hardcoded values like 1e-8 or 1e-10.

All operations stay on GPU via the Backend protocol. No NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

__all__ = [
    # Backend scalar helpers (use instead of math module)
    "sqrt_scalar",
    "is_finite",
    "is_inf",
    "is_nan",
    "log_scalar",
    "exp_scalar",
    "power_scalar",
    "ceil_scalar",
    "floor_scalar",
    "ulp_scalar",
    "lgamma_scalar",
    "acos_scalar",
    "cos_scalar",
    "sin_scalar",
    "atan2_scalar",
    "log2_scalar",
    "pi_value",
    "e_value",
    "inf_value",
    # Epsilon and threshold utilities
    "machine_epsilon",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    "infinity_threshold",
]


# =============================================================================
# Backend Scalar Helpers
# =============================================================================
# Use these instead of math.sqrt, math.isfinite, etc. to keep computation on GPU.


def sqrt_scalar(value: float, backend: "Backend") -> float:
    """Compute sqrt of scalar using backend with non-negativity guard.

    Use instead of math.sqrt(value). Guards against numerical noise
    that might produce slightly negative values (returns 0.0 for v < 0).
    """
    # Guard against negative values from numerical noise
    safe_value = max(0.0, value)
    arr = backend.array([safe_value])
    result = backend.sqrt(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def is_finite(value: float, backend: "Backend") -> bool:
    """Check if scalar is finite using backend.

    Use instead of math.isfinite(value).
    """
    arr = backend.array([value])
    result = backend.isfinite(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_inf(value: float, backend: "Backend") -> bool:
    """Check if scalar is infinite using backend.

    Use instead of math.isinf(value).
    """
    arr = backend.array([value])
    result = backend.isinf(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def is_nan(value: float, backend: "Backend") -> bool:
    """Check if scalar is NaN using backend.

    Use instead of math.isnan(value).
    """
    arr = backend.array([value])
    result = backend.isnan(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def log_scalar(value: float, backend: "Backend") -> float:
    """Compute natural log of scalar using backend.

    Use instead of math.log(value).
    """
    arr = backend.array([value])
    result = backend.log(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def exp_scalar(value: float, backend: "Backend") -> float:
    """Compute exp of scalar using backend.

    Use instead of math.exp(value).
    """
    arr = backend.array([value])
    result = backend.exp(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def power_scalar(value: float, exponent: float, backend: "Backend") -> float:
    """Compute value ** exponent using backend.

    Use instead of value ** exponent for GPU acceleration.
    """
    arr = backend.array([value])
    result = arr**exponent
    backend.eval(result)
    return float(backend.to_scalar(result))


def ceil_scalar(value: float, backend: "Backend") -> int:
    """Compute ceil of scalar using backend.

    Use instead of math.ceil(value).
    """
    arr = backend.array([value])
    result = backend.ceil(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def floor_scalar(value: float, backend: "Backend") -> int:
    """Compute floor of scalar using backend.

    Use instead of math.floor(value).
    """
    arr = backend.array([value])
    result = backend.floor(arr)
    backend.eval(result)
    return int(backend.to_scalar(result))


def ulp_scalar(value: float, backend: "Backend") -> float:
    """Compute unit in last place for scalar using backend.

    Use instead of math.ulp(value).
    For normalized floats, ulp(x) ~ eps * abs(x).
    """
    eps = backend.finfo(backend.array([value]).dtype).eps
    return eps * abs(value) if value != 0.0 else eps


def lgamma_scalar(value: float, backend: "Backend") -> float:
    """Compute log-gamma of scalar using backend.

    Use instead of math.lgamma(value).
    """
    arr = backend.array([value])
    result = backend.lgamma(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def acos_scalar(value: float, backend: "Backend") -> float:
    """Compute arc cosine of scalar using backend.

    Use instead of math.acos(value).
    """
    arr = backend.array([value])
    result = backend.arccos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def cos_scalar(value: float, backend: "Backend") -> float:
    """Compute cosine of scalar using backend.

    Use instead of math.cos(value).
    """
    arr = backend.array([value])
    result = backend.cos(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def sin_scalar(value: float, backend: "Backend") -> float:
    """Compute sine of scalar using backend.

    Use instead of math.sin(value).
    """
    arr = backend.array([value])
    result = backend.sin(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def atan2_scalar(y: float, x: float, backend: "Backend") -> float:
    """Compute atan2(y, x) using backend.

    Use instead of math.atan2(y, x).
    Returns angle in radians in [-pi, pi].
    """
    # atan2(y, x) = arctan(y/x) with quadrant handling
    y_arr = backend.array([y])
    x_arr = backend.array([x])

    # Compute arctan(y/x) with safe division
    eps = backend.finfo(backend.array([1.0]).dtype).eps
    x_safe = backend.where(
        backend.abs(x_arr) < eps,
        backend.sign(x_arr) * eps + eps,  # Avoid division by zero
        x_arr,
    )
    ratio = y_arr / x_safe
    base_angle = backend.arctan(ratio)
    backend.eval(base_angle)

    # Adjust for quadrant
    angle = float(backend.to_scalar(base_angle))
    pi = pi_value(backend)

    if x < 0:
        if y >= 0:
            angle += pi
        else:
            angle -= pi
    elif x == 0:
        if y > 0:
            angle = pi / 2
        elif y < 0:
            angle = -pi / 2
        else:
            angle = 0.0

    return angle


def log2_scalar(value: float, backend: "Backend") -> float:
    """Compute log base 2 of scalar using backend.

    Use instead of math.log2(value).
    """
    arr = backend.array([value])
    result = backend.log2(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def pi_value(backend: "Backend") -> float:
    """Get pi using backend.

    Use instead of math.pi.
    """
    # Compute pi = 4 * arctan(1)
    one = backend.array([1.0])
    result = 4.0 * backend.arctan(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def e_value(backend: "Backend") -> float:
    """Get Euler's number e using backend.

    Use instead of math.e.
    """
    # Compute e = exp(1)
    one = backend.array([1.0])
    result = backend.exp(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def inf_value(backend: "Backend") -> float:
    """Get positive infinity using backend.

    Use instead of math.inf.
    """
    return float("inf")


# =============================================================================
# Dtype-Derived Epsilon Utilities
# =============================================================================


def machine_epsilon(backend: "Backend", array: "Array") -> float:
    """Get machine epsilon for the array's dtype.

    This is the smallest value such that 1.0 + epsilon != 1.0.
    Use for general numerical stability in comparisons.
    """
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe division operations.

    Uses sqrt(eps) to provide numerical headroom.
    Use when dividing to prevent division by zero.
    """
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


def regularization_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for matrix regularization.

    Uses eps^0.75 to scale regularization above division safety
    while remaining tied to dtype precision.
    """
    eps = backend.finfo(array.dtype).eps
    return power_scalar(eps, 0.75, backend)


def condition_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for condition number checks.

    Returns 1/eps, the inverse of machine epsilon.
    Matrices with condition number above this are numerically singular.
    """
    return 1.0 / backend.finfo(array.dtype).eps


def svd_rank_threshold(backend: "Backend", array: "Array", max_dim: int) -> float:
    """Get threshold for determining numerical rank from SVD.

    Uses the standard formula: max_dim * eps * largest_singular_value.
    Singular values below this threshold are considered zero.

    Args:
        backend: The compute backend.
        array: The array being decomposed (for dtype).
        max_dim: Maximum dimension of the matrix.

    Returns:
        Threshold scaled by matrix size and precision.
    """
    eps = backend.finfo(array.dtype).eps
    return float(max_dim) * eps


def tiny_value(backend: "Backend", array: "Array") -> float:
    """Get the smallest positive usable number for the dtype.

    Use as a floor when values must remain positive.
    """
    return backend.finfo(array.dtype).tiny


def safe_log_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe logarithm operations.

    Uses tiny value to prevent log(0) while maintaining precision.
    """
    return backend.finfo(array.dtype).tiny


def infinity_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for detecting near-infinite values.

    Values at or above this threshold should be treated as infinite
    (e.g., disconnected nodes in a graph). Derived from machine epsilon
    to be numerically principled rather than arbitrary.

    Uses: max_representable * sqrt(eps)

    This provides a threshold that is ~4 orders of magnitude below max
    for float32, robustly detecting overflow while avoiding false positives.

    For float32: eps ~ 1.2e-7, sqrt(eps) ~ 3.5e-4, max ~ 3.4e38
    Threshold ~ 3.4e38 * 3.5e-4 ~ 1.2e35

    This is much more conservative than the previous (1 - sqrt(eps)) * max
    which was only 0.034% below max.
    """
    finfo = backend.finfo(array.dtype)
    eps = finfo.eps
    max_val = finfo.max
    # Use sqrt(eps) * max for a robust threshold
    return float(max_val) * sqrt_scalar(eps, backend)

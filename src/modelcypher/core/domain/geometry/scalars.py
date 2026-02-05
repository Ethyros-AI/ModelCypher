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

"""Backend scalar helpers for numerical operations.

Use these instead of the math module to ensure consistency with the active backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


def _scalar_unary(
    value: float,
    backend: "Backend",
    op: Callable[["Array"], "Array"],
) -> float:
    arr = backend.array([value])
    result = op(arr)
    backend.eval(result)
    return float(backend.to_scalar(result))


def _scalar_unary_int(
    value: float,
    backend: "Backend",
    op: Callable[["Array"], "Array"],
) -> int:
    return int(_scalar_unary(value, backend, op))


def _scalar_unary_bool(
    value: float,
    backend: "Backend",
    op: Callable[["Array"], "Array"],
) -> bool:
    arr = backend.array([value])
    result = op(arr)
    backend.eval(result)
    return bool(backend.to_scalar(result))


def sqrt_scalar(value: float, backend: "Backend") -> float:
    """Compute sqrt of scalar using backend with non-negativity guard."""
    safe_value = max(0.0, value)
    return _scalar_unary(safe_value, backend, backend.sqrt)


def is_finite(value: float, backend: "Backend") -> bool:
    """Check if scalar is finite using backend."""
    return _scalar_unary_bool(value, backend, backend.isfinite)


def is_inf(value: float, backend: "Backend") -> bool:
    """Check if scalar is infinite using backend."""
    return _scalar_unary_bool(value, backend, backend.isinf)


def is_nan(value: float, backend: "Backend") -> bool:
    """Check if scalar is NaN using backend."""
    return _scalar_unary_bool(value, backend, backend.isnan)


def all_finite(arr: "Array", backend: "Backend") -> bool:
    """Check if all elements in array are finite (not NaN or Inf).

    Args:
        arr: Array to check (any shape).
        backend: Backend for tensor operations.

    Returns:
        True if all elements are finite, False otherwise.
    """
    finite_mask = backend.isfinite(arr)
    all_ok = backend.all(finite_mask)
    backend.eval(all_ok)
    return bool(backend.to_scalar(all_ok))


def log_scalar(value: float, backend: "Backend") -> float:
    """Compute natural log of scalar using backend."""
    return _scalar_unary(value, backend, backend.log)


def exp_scalar(value: float, backend: "Backend") -> float:
    """Compute exp of scalar using backend."""
    return _scalar_unary(value, backend, backend.exp)


def power_scalar(value: float, exponent: float, backend: "Backend") -> float:
    """Compute value ** exponent using backend."""
    arr = backend.array([value])
    result = arr**exponent
    backend.eval(result)
    return float(backend.to_scalar(result))


def ceil_scalar(value: float, backend: "Backend") -> int:
    """Compute ceil of scalar using backend."""
    return _scalar_unary_int(value, backend, backend.ceil)


def floor_scalar(value: float, backend: "Backend") -> int:
    """Compute floor of scalar using backend."""
    return _scalar_unary_int(value, backend, backend.floor)


def ulp_scalar(value: float, backend: "Backend") -> float:
    """Compute unit in last place for scalar using backend."""
    eps = backend.finfo(backend.array([value]).dtype).eps
    return eps * abs(value) if value != 0.0 else eps


def lgamma_scalar(value: float, backend: "Backend") -> float:
    """Compute log-gamma of scalar using backend."""
    return _scalar_unary(value, backend, backend.lgamma)


def acos_scalar(value: float, backend: "Backend") -> float:
    """Compute arc cosine of scalar using backend."""
    return _scalar_unary(value, backend, backend.arccos)


def cos_scalar(value: float, backend: "Backend") -> float:
    """Compute cosine of scalar using backend."""
    return _scalar_unary(value, backend, backend.cos)


def sin_scalar(value: float, backend: "Backend") -> float:
    """Compute sine of scalar using backend."""
    return _scalar_unary(value, backend, backend.sin)


def atan2_scalar(y: float, x: float, backend: "Backend") -> float:
    """Compute atan2(y, x) using backend."""
    y_arr = backend.array([y])
    x_arr = backend.array([x])

    eps = backend.finfo().eps
    x_safe = backend.where(
        backend.abs(x_arr) < eps,
        backend.sign(x_arr) * eps + eps,
        x_arr,
    )
    ratio = y_arr / x_safe
    base_angle = backend.arctan(ratio)
    backend.eval(base_angle)

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
    """Compute log base 2 of scalar using backend."""
    return _scalar_unary(value, backend, backend.log2)


def pi_value(backend: "Backend") -> float:
    """Get pi using backend."""
    one = backend.array([1.0])
    result = 4.0 * backend.arctan(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def e_value(backend: "Backend") -> float:
    """Get Euler's number e using backend."""
    one = backend.array([1.0])
    result = backend.exp(one)
    backend.eval(result)
    return float(backend.to_scalar(result))


def inf_value(backend: "Backend") -> float:
    """Get positive infinity using backend."""
    return float("inf")


__all__ = [
    "sqrt_scalar",
    "is_finite",
    "is_inf",
    "is_nan",
    "all_finite",
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
]

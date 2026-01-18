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

"""Dynamic numerical stability utilities.

All epsilons and tolerances are derived from tensor precision, not arbitrary constants.
Use these functions instead of hardcoded values like 1e-8 or 1e-10.

PRECISION PHILOSOPHY:
    Model weights define the precision ceiling. We cannot extract more information
    than exists in the source data. Using float64 for bf16 weights just adds zeros.

    The compute precision is MODEL-DRIVEN:
    - Detect the native dtype of model weights
    - Use that precision (or float32 max) for all computations
    - Never use float64 (no neural network stores weights at this precision)

    This reduces memory usage ~2x and speeds up all matrix operations while
    preserving 100% of the information in the original weights.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend


# =============================================================================
# Dtype Precision Detection
# =============================================================================

# Effective mantissa bits for common dtypes (determines precision ceiling)
_DTYPE_PRECISION_BITS: dict[str, int] = {
    # Quantized types (low precision, should be promoted)
    "int4": 4,
    "uint4": 4,
    "int8": 8,
    "uint8": 8,
    # Half precision floats
    "float16": 10,  # 10 mantissa bits
    "bfloat16": 7,  # 7 mantissa bits (wider range, less precision)
    # Standard precision (this is our cap)
    "float32": 23,  # 23 mantissa bits
    # Double precision (never needed for neural networks)
    "float64": 52,  # 52 mantissa bits - OVERKILL, never use
}

# Minimum compute precision for quantized types
_MIN_COMPUTE_DTYPE = "float16"

# Maximum compute precision (float64 is never needed)
_MAX_COMPUTE_DTYPE = "float32"


def dtype_precision_bits(dtype: object) -> int:
    """Get effective precision bits for a dtype.

    Returns the number of mantissa bits, which determines how much
    information the dtype can actually store. Higher bits = more precision.

    Args:
        dtype: A dtype object (from numpy, mlx, jax, etc.)

    Returns:
        Number of effective precision bits.
    """
    name = _dtype_name(dtype).lower()

    # Check in order from most specific to least specific
    # (bfloat16 must be checked before float16 since float16 is a substring)
    if "bfloat16" in name:
        return 7  # 7 mantissa bits
    if "float64" in name:
        return 52  # 52 mantissa bits - OVERKILL
    if "float32" in name:
        return 23  # 23 mantissa bits
    if "float16" in name:
        return 10  # 10 mantissa bits
    if "int8" in name or "uint8" in name:
        return 8
    if "int4" in name or "uint4" in name:
        return 4

    # Unknown dtype - assume float32 level
    logger.warning("Unknown dtype %s, assuming float32 precision", name)
    return 23


def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    return name.replace("mlx.core.", "").replace("jax.numpy.", "")


def detect_model_dtype(weights: dict, backend: "Backend") -> object:
    """Detect the dominant dtype from model weights.

    Scans weight tensors and returns the most common floating-point dtype.
    This represents the model's native precision.

    Args:
        weights: Dictionary of weight tensors (name -> array).
        backend: Backend for tensor operations.

    Returns:
        The dominant dtype found in weights.
    """
    dtype_counts: dict[str, int] = {}

    for name, tensor in weights.items():
        if not hasattr(tensor, "dtype"):
            continue

        dtype_str = _dtype_name(tensor.dtype)

        # Skip non-float types (embeddings might be int)
        if "int" in dtype_str.lower() or "bool" in dtype_str.lower():
            continue

        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

    if not dtype_counts:
        # No float weights found, default to float32
        logger.warning("No float weights found, defaulting to float32")
        return backend.array([1.0], dtype="float32").dtype

    # Find most common dtype
    dominant_dtype_str = max(dtype_counts, key=dtype_counts.get)  # type: ignore[arg-type]

    logger.info(
        "MODEL DTYPE: detected %s (from %d tensors, breakdown: %s)",
        dominant_dtype_str,
        sum(dtype_counts.values()),
        dtype_counts,
    )

    # Convert string back to actual dtype
    try:
        return backend.array([1.0], dtype=dominant_dtype_str).dtype
    except Exception:
        # Fallback if dtype string isn't recognized
        return backend.array([1.0], dtype="float32").dtype


def compute_precision_for_merge(
    source_weights: dict,
    target_weights: dict,
    backend: "Backend",
) -> object:
    """Determine compute precision from source and target model weights.

    Rules:
    1. Detect native dtype of both models
    2. Use the HIGHER precision of the two (to preserve info from both)
    3. Promote bfloat16 to float32 to preserve range in compute
    4. Cap at float32 (float64 never adds value for neural network weights)
    5. For quantized (int4/int8), use float16 minimum

    Args:
        source_weights: Source model weight dict.
        target_weights: Target model weight dict.
        backend: Backend for tensor operations.

    Returns:
        The appropriate compute dtype for the merge.
    """
    source_dtype = detect_model_dtype(source_weights, backend)
    target_dtype = detect_model_dtype(target_weights, backend)

    source_bits = dtype_precision_bits(source_dtype)
    target_bits = dtype_precision_bits(target_dtype)
    source_name = _dtype_name(source_dtype).lower()
    target_name = _dtype_name(target_dtype).lower()

    # Use higher precision of the two
    if source_bits >= target_bits:
        chosen_dtype = source_dtype
        chosen_bits = source_bits
    else:
        chosen_dtype = target_dtype
        chosen_bits = target_bits

    # Promote bfloat16 to float32 to preserve range in compute
    if "bfloat16" in source_name or "bfloat16" in target_name:
        logger.info("PRECISION PROMOTION: bfloat16 detected -> float32 compute")
        chosen_dtype = backend.array([1.0], dtype="float32").dtype
        chosen_bits = dtype_precision_bits(chosen_dtype)

    # Cap at float32 (23 bits) - float64 is never needed
    if chosen_bits > 23:
        logger.info(
            "PRECISION CAP: %s (%d bits) -> float32 (23 bits)",
            _dtype_name(chosen_dtype),
            chosen_bits,
        )
        chosen_dtype = backend.array([1.0], dtype="float32").dtype

    # Ensure minimum precision for quantized types
    if chosen_bits < 10:  # Less than float16
        logger.info(
            "PRECISION FLOOR: %s (%d bits) -> float16 (10 bits)",
            _dtype_name(chosen_dtype),
            chosen_bits,
        )
        chosen_dtype = backend.array([1.0], dtype=_MIN_COMPUTE_DTYPE).dtype

    logger.info(
        "MERGE PRECISION: source=%s (%d bits), target=%s (%d bits) -> compute=%s",
        _dtype_name(source_dtype),
        source_bits,
        _dtype_name(target_dtype),
        target_bits,
        _dtype_name(chosen_dtype),
    )

    return chosen_dtype


# Thread-local storage for model-driven precision
_model_compute_dtype: object | None = None


def set_model_compute_dtype(dtype: object | None) -> None:
    """Set the model-driven compute dtype for the current merge operation.

    Call this at the start of a merge with the result from
    compute_precision_for_merge(). All subsequent precision_dtype()
    calls will respect this ceiling.

    Args:
        dtype: The compute dtype to use, or None to reset to default.
    """
    global _model_compute_dtype
    _model_compute_dtype = dtype
    if dtype is not None:
        logger.info("SET MODEL PRECISION: %s", _dtype_name(dtype))


def get_model_compute_dtype() -> object | None:
    """Get the current model-driven compute dtype, if set."""
    return _model_compute_dtype


def _default_float_dtype(backend: "Backend") -> object:
    """Return the compute dtype, respecting model-driven precision.

    Priority:
    1. Model-driven precision (if set via set_model_compute_dtype)
    2. float32 (the maximum precision we ever need)

    float64 is NEVER returned - it adds computational overhead
    without any precision benefit for neural network weights.
    """
    # Check for model-driven precision
    if _model_compute_dtype is not None:
        return _model_compute_dtype

    # Default to float32 (never float64)
    return backend.array([1.0], dtype="float32").dtype


def precision_dtype(
    backend: "Backend",
    reference: "Array | None" = None,
    model_dtype: object | None = None,
) -> object:
    """Select compute precision, respecting model precision ceiling.

    The returned dtype will be:
    - At least as precise as the reference array (if provided)
    - At most as precise as the model dtype (if provided or globally set)
    - Never more than float32 (float64 is overkill for neural networks)

    Args:
        backend: Backend for tensor operations.
        reference: Optional reference array whose precision should be preserved.
        model_dtype: Optional explicit model dtype cap. If not provided,
                     uses the globally set model compute dtype.

    Returns:
        Appropriate compute dtype.
    """
    # Determine precision ceiling
    ceiling = model_dtype or _model_compute_dtype
    if ceiling is None:
        # No model precision set - use float32 as max
        ceiling = backend.array([1.0], dtype="float32").dtype

    ceiling_bits = dtype_precision_bits(ceiling)

    # If no reference, just return ceiling
    if reference is None or not hasattr(reference, "dtype"):
        return ceiling

    ref_bits = dtype_precision_bits(reference.dtype)

    # Use higher of reference and ceiling, but cap at float32
    if ref_bits > ceiling_bits:
        # Reference is higher precision - use ceiling (model-driven cap)
        return ceiling
    else:
        # Reference is lower precision - promote to ceiling
        # (but only if ceiling is actually higher precision)
        try:
            ref_eps = backend.finfo(reference.dtype).eps
            ceiling_eps = backend.finfo(ceiling).eps
            if ceiling_eps < ref_eps:
                return ceiling
            return reference.dtype
        except Exception:
            return ceiling


def _promote_precision(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: object | None = None,
) -> "Array":
    """Promote low-precision or integer arrays to at least default float dtype."""
    if min_dtype is None:
        min_dtype = _default_float_dtype(backend)

    if not hasattr(array, "dtype"):
        return backend.array(array, dtype=min_dtype)

    dtype_name = _dtype_name(array.dtype)
    if (
        "float16" in dtype_name
        or "bfloat16" in dtype_name
        or "int" in dtype_name
        or "uint" in dtype_name
        or "bool" in dtype_name
    ):
        return backend.astype(array, min_dtype)

    try:
        current_eps = backend.finfo(array.dtype).eps
        min_eps = backend.finfo(min_dtype).eps
    except Exception:
        return backend.astype(array, min_dtype)

    if current_eps > min_eps:
        return backend.astype(array, min_dtype)

    return array

__all__ = [
    # Model-driven precision detection
    "dtype_precision_bits",
    "detect_model_dtype",
    "compute_precision_for_merge",
    "set_model_compute_dtype",
    "get_model_compute_dtype",
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
    "precision_dtype",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    "infinity_threshold",
    # Data-derived thresholds
    "find_magnitude_gap_threshold",
    # Statistical utilities
    "compute_median",
    "compute_median_nonzero",
    "compute_pearson_correlation",
    "compute_spearman_correlation",
    # Matrix decomposition
    "safe_inverse",
    "geodesic_svd",
    "geodesic_pinv",
    "power_iteration_eigh",
    # GPU-accelerated linear algebra
    "gpu_lstsq",
# Invariant alignment (linear CKA = 1.0 by construction)
    "invariant_alignment",
    # Geodesic invariant alignment (preserves manifold structure)
    "geodesic_invariant_alignment",
]


# =============================================================================
# Backend Scalar Helpers
# =============================================================================


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


# =============================================================================
# Epsilon and Threshold Utilities
# =============================================================================


def machine_epsilon(backend: "Backend", array: "Array") -> float:
    """Get machine epsilon for the array's dtype."""
    return backend.finfo(array.dtype).eps


def division_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe division operations. Uses sqrt(eps)."""
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


def regularization_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for matrix regularization. Uses eps^0.75."""
    eps = backend.finfo(array.dtype).eps
    return power_scalar(eps, 0.75, backend)


def condition_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for condition number checks. Returns 1/eps."""
    return 1.0 / backend.finfo(array.dtype).eps


def svd_rank_threshold(backend: "Backend", array: "Array", max_dim: int) -> float:
    """Get threshold for determining numerical rank from SVD."""
    eps = backend.finfo(array.dtype).eps
    return float(max_dim) * eps


def tiny_value(backend: "Backend", array: "Array") -> float:
    """Get the smallest positive usable number for the dtype."""
    return backend.finfo(array.dtype).tiny


def safe_log_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe logarithm operations."""
    return backend.finfo(array.dtype).tiny


def infinity_threshold(backend: "Backend", array: "Array") -> float:
    """Get threshold for detecting near-infinite values."""
    finfo = backend.finfo(array.dtype)
    eps = finfo.eps
    max_val = finfo.max
    return float(max_val) * sqrt_scalar(eps, backend)


# =============================================================================
# Data-Derived Thresholds
# =============================================================================


def find_magnitude_gap_threshold(
    sorted_values: list[float] | "Array",
    eps: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Find the natural break point in a sorted magnitude distribution."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    if hasattr(sorted_values, "shape"):
        values_arr = sorted_values  # type: ignore[assignment]
        n = int(values_arr.shape[0])
    else:
        if not sorted_values:
            return 0.0
        values_arr = backend.array(sorted_values)
        n = len(sorted_values)

    if n == 0:
        return 0.0

    if eps is None:
        abs_arr = backend.abs(values_arr)
        scale_arr = backend.maximum(backend.max(abs_arr), backend.array([1.0]))
        backend.eval(scale_arr)
        scale = float(backend.to_scalar(scale_arr))
        eps = ulp_scalar(scale, backend)

    if n == 1:
        # Single value - return it as the threshold
        first_val = backend.take(values_arr, backend.array([0]), axis=0)
        first_val = backend.squeeze(first_val)
        backend.eval(first_val)
        return float(backend.to_scalar(first_val))

    if n == 2:
        # Two values - check for a significant relative gap
        # If gap > 50%, return the smaller (threshold), else return larger (no outlier)
        first_val = backend.take(values_arr, backend.array([0]), axis=0)
        second_val = backend.take(values_arr, backend.array([1]), axis=0)
        first_val = backend.squeeze(first_val)
        second_val = backend.squeeze(second_val)
        backend.eval(first_val, second_val)
        v0 = float(backend.to_scalar(first_val))
        v1 = float(backend.to_scalar(second_val))
        if v0 > eps:
            rel_gap = (v1 - v0) / v0
            if rel_gap > 0.5:  # Significant gap - return smaller as threshold
                return v0
        return v1  # No significant gap - return larger (nothing will be flagged)

    idx = backend.arange(0, n - 1)
    next_idx = backend.arange(1, n)
    curr = backend.take(values_arr, idx, axis=0)
    next_vals = backend.take(values_arr, next_idx, axis=0)
    diffs = next_vals - curr
    eps_arr = backend.array([eps])
    valid = curr > eps_arr
    denom = backend.where(valid, curr, backend.ones_like(curr))
    rel_gaps = diffs / denom
    rel_gaps = backend.where(valid, rel_gaps, backend.zeros_like(rel_gaps))
    max_gap_arr = backend.max(rel_gaps)
    backend.eval(max_gap_arr)
    max_mask = rel_gaps == max_gap_arr
    indices = backend.arange(0, n - 1)
    inf_val = backend.full(indices.shape, float("inf"))
    masked_indices = backend.where(max_mask, indices, inf_val)
    gap_index_arr = backend.min(masked_indices)
    backend.eval(gap_index_arr)
    max_gap = float(backend.to_scalar(max_gap_arr))

    if max_gap <= 0.0:
        mid_idx = backend.array([n // 2])
        mid_val = backend.take(values_arr, mid_idx, axis=0)
        mid_val = backend.squeeze(mid_val)
        backend.eval(mid_val)
        return float(backend.to_scalar(mid_val))

    gap_index = int(backend.to_scalar(gap_index_arr))
    gap_val = backend.take(values_arr, backend.array([gap_index]), axis=0)
    gap_val = backend.squeeze(gap_val)
    backend.eval(gap_val)
    return float(backend.to_scalar(gap_val))


# =============================================================================
# Statistical Utilities
# =============================================================================


def _geodesic_norm_scalar(array: "Array", backend: "Backend") -> float:
    """Compute a geodesic norm scalar for any array shape."""
    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    vec = backend.reshape(array, (1, -1))
    zero = backend.zeros_like(vec)
    points = backend.concatenate([zero, vec], axis=0)
    rg = RiemannianGeometry(backend)
    point_count = int(backend.shape(points)[0])
    geo_result = rg.geodesic_distances(points, k_neighbors=point_count - 1)
    backend.eval(geo_result.distances)
    return max(0.0, float(backend.to_scalar(geo_result.distances[0, 1])))


def compute_median(array: "Array", backend: "Backend") -> float:
    """Compute median of array values using O(n) argpartition."""
    flat = backend.reshape(array, (-1,))
    backend.eval(flat)

    n = int(flat.shape[0])
    if n == 0:
        return 0.0
    if n == 1:
        return float(backend.to_scalar(flat))

    mid = n // 2

    if n % 2 == 1:
        partitioned = backend.argpartition(flat, mid)
        backend.eval(partitioned)
        prefix = backend.take(partitioned, backend.arange(mid + 1), axis=0)
        median_arr = backend.max(backend.take(flat, prefix, axis=0))
    else:
        low_part = backend.argpartition(flat, mid - 1)
        backend.eval(low_part)
        low_prefix = backend.take(low_part, backend.arange(mid), axis=0)
        low = backend.max(backend.take(flat, low_prefix, axis=0))

        high_part = backend.argpartition(flat, mid)
        backend.eval(high_part)
        high_prefix = backend.take(high_part, backend.arange(mid + 1), axis=0)
        high = backend.max(backend.take(flat, high_prefix, axis=0))

        median_arr = (low + high) * 0.5

    median_arr = backend.squeeze(median_arr)
    backend.eval(median_arr)
    return float(backend.to_scalar(median_arr))


def compute_median_nonzero(
    array: "Array",
    backend: "Backend",
    zero_threshold: float | None = None,
) -> float:
    """Compute median of non-zero values using O(n) argpartition."""
    flat = backend.reshape(array, (-1,))
    backend.eval(flat)

    n = int(flat.shape[0])
    if n == 0:
        return 0.0

    if zero_threshold is None:
        zero_threshold = division_epsilon(backend, flat)

    zero_mask = flat <= zero_threshold
    zero_count_arr = backend.sum(backend.astype(zero_mask, "int32"))
    backend.eval(zero_count_arr)
    zero_count = int(backend.to_scalar(zero_count_arr))

    non_zero_count = n - zero_count
    if non_zero_count <= 0:
        return 0.0

    median_idx = min(zero_count + (non_zero_count // 2), n - 1)

    partitioned = backend.argpartition(flat, median_idx)
    backend.eval(partitioned)

    prefix_indices = backend.take(partitioned, backend.arange(median_idx + 1), axis=0)
    prefix_vals = backend.take(flat, prefix_indices, axis=0)
    backend.eval(prefix_vals)

    median_arr = backend.max(prefix_vals)
    backend.eval(median_arr)

    return float(backend.to_scalar(median_arr))


def compute_pearson_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute geodesic correlation coefficient between two lists."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    error_value = default if default is not None else float("nan")

    if not lhs or len(lhs) != len(rhs):
        return error_value

    lhs_arr = backend.array(lhs)
    rhs_arr = backend.array(rhs)
    mean_l = backend.mean(lhs_arr)
    mean_r = backend.mean(rhs_arr)
    diff_l = lhs_arr - mean_l
    diff_r = rhs_arr - mean_r
    backend.eval(diff_l, diff_r)

    eps = division_epsilon(backend, lhs_arr)
    d0a = _geodesic_norm_scalar(diff_l, backend)
    d0b = _geodesic_norm_scalar(diff_r, backend)
    if d0a <= eps or d0b <= eps:
        return error_value

    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    diff_l_vec = backend.reshape(diff_l, (1, -1))
    diff_r_vec = backend.reshape(diff_r, (1, -1))
    points = backend.concatenate([diff_l_vec, diff_r_vec], axis=0)
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(points, k_neighbors=1)
    backend.eval(geo_result.distances)
    dab = float(backend.to_scalar(geo_result.distances[0, 1]))

    denom = 2.0 * d0a * d0b
    if denom <= eps:
        return error_value
    cos_val = (d0a * d0a + d0b * d0b - dab * dab) / denom
    corr = max(-1.0, min(1.0, cos_val))
    return corr if is_finite(corr, backend) else error_value


def compute_spearman_correlation(
    lhs: list[float],
    rhs: list[float],
    *,
    default: float | None = None,
    backend: "Backend | None" = None,
) -> float:
    """Compute Spearman rank correlation coefficient between two lists."""
    if backend is None:
        from modelcypher.core.domain._backend import get_default_backend

        backend = get_default_backend()

    error_value = default if default is not None else float("nan")

    if not lhs or len(lhs) != len(rhs) or len(lhs) < 2:
        return error_value

    lhs_arr = backend.array(lhs)
    rhs_arr = backend.array(rhs)

    lhs_rank = backend.argsort(backend.argsort(lhs_arr, axis=0), axis=0)
    rhs_rank = backend.argsort(backend.argsort(rhs_arr, axis=0), axis=0)
    rank_dtype = precision_dtype(backend, reference=lhs_rank)
    lhs_rank = backend.astype(lhs_rank, rank_dtype)
    rhs_rank = backend.astype(rhs_rank, rank_dtype)

    mean_l = backend.mean(lhs_rank)
    mean_r = backend.mean(rhs_rank)
    diff_l = lhs_rank - mean_l
    diff_r = rhs_rank - mean_r
    backend.eval(diff_l, diff_r)

    eps = division_epsilon(backend, lhs_rank)
    d0a = _geodesic_norm_scalar(diff_l, backend)
    d0b = _geodesic_norm_scalar(diff_r, backend)
    if d0a <= eps or d0b <= eps:
        return error_value

    from modelcypher.core.domain.geometry.riemannian_core import RiemannianGeometry

    diff_l_vec = backend.reshape(diff_l, (1, -1))
    diff_r_vec = backend.reshape(diff_r, (1, -1))
    points = backend.concatenate([diff_l_vec, diff_r_vec], axis=0)
    rg = RiemannianGeometry(backend)
    geo_result = rg.geodesic_distances(points, k_neighbors=1)
    backend.eval(geo_result.distances)
    dab = float(backend.to_scalar(geo_result.distances[0, 1]))

    denom = 2.0 * d0a * d0b
    if denom <= eps:
        return error_value
    cos_val = (d0a * d0a + d0b * d0b - dab * dab) / denom
    corr = max(-1.0, min(1.0, cos_val))
    return corr if is_finite(corr, backend) else error_value


# =============================================================================
# Matrix Decomposition
# =============================================================================


def power_iteration_eigh(
    backend: "Backend",
    matrix: "Array",
    k: int = 10,
    use_ritz: bool = True,
) -> tuple["Array", "Array"]:
    """Compute EXACT eigendecomposition using native backend operation."""
    b = backend
    matrix = _promote_precision(b.array(matrix), b)
    b.eval(matrix)
    dtype = matrix.dtype

    shape = matrix.shape
    n = int(shape[-1])
    batch_shape = shape[:-2]
    k = min(k, n)
    if k == 0:
        return (
            b.zeros(batch_shape + (0,), dtype=dtype),
            b.zeros(batch_shape + (n, 0), dtype=dtype),
        )

    eigenvalues_full, eigenvectors_full = b.eigh(matrix)
    b.eval(eigenvalues_full, eigenvectors_full)

    order = b.argsort(-eigenvalues_full, axis=-1)
    eigenvalues_sorted = b.take_along_axis(eigenvalues_full, order, axis=-1)
    order_exp = b.expand_dims(order, axis=-2)
    order_tiled = b.broadcast_to(order_exp, eigenvectors_full.shape)
    eigenvectors_sorted = b.take_along_axis(eigenvectors_full, order_tiled, axis=-1)
    b.eval(order, eigenvalues_sorted, eigenvectors_sorted)

    eigenvalues = eigenvalues_sorted[..., :k]
    eigenvectors = eigenvectors_sorted[..., :k]
    b.eval(eigenvalues, eigenvectors)

    return eigenvalues, eigenvectors


def geodesic_svd(
    backend: "Backend",
    array: "Array",
    k: int | None = None,
) -> tuple["Array", "Array", "Array"]:
    """Compute EXACT SVD using native backend operation."""
    b = backend
    A = _promote_precision(b.array(array), b)
    b.eval(A)
    dtype = A.dtype

    shape = A.shape
    if len(shape) < 2:
        raise ValueError("geodesic_svd requires at least 2D input")
    m = int(shape[-2])
    n = int(shape[-1])
    batch_shape = shape[:-2]
    max_rank = min(m, n)

    if m == 0 or n == 0:
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    # Defensive checks for LAPACK
    A_sum = b.sum(A)
    b.eval(A_sum)
    A_sum_val = float(b.to_scalar(A_sum))

    if A_sum_val != A_sum_val:  # NaN check
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    if abs(A_sum_val) == float("inf"):
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    A_norm_sq = b.sum(A * A)
    b.eval(A_norm_sq)
    A_norm_sq_val = float(b.to_scalar(A_norm_sq))
    tiny = tiny_value(b, A)
    zero_threshold = tiny * max(1.0, float(m * n))
    if A_norm_sq_val <= zero_threshold:
        U = b.zeros(batch_shape + (m, 0), dtype=dtype)
        S = b.zeros(batch_shape + (0,), dtype=dtype)
        Vt = b.zeros(batch_shape + (0, n), dtype=dtype)
        return U, S, Vt

    U_full, S_full, Vt_full = b.svd(A, compute_uv=True)
    b.eval(U_full, S_full, Vt_full)

    k = min(k or max_rank, max_rank)
    if k == 0:
        U = b.zeros((m, 0), dtype=dtype)
        S = b.zeros((0,), dtype=dtype)
        Vt = b.zeros((0, n), dtype=dtype)
        return U, S, Vt

    U = U_full[..., :k]
    S = S_full[..., :k]
    Vt = Vt_full[..., :k, :]
    b.eval(U, S, Vt)

    return U, S, Vt


def orthogonalize_alignment(
    alignment: "Array",
    backend: "Backend",
) -> tuple["Array", float]:
    """Extract orthogonal part of alignment transform via polar decomposition.

    Given an alignment transform F (which may include scaling), extract the
    orthogonal component U such that F = U @ P where U is orthogonal and P
    is positive semidefinite.

    This is the Lie algebra decomposition: the alignment lives on the manifold
    of linear maps, and we extract its rotation component (element of SO(n)/O(n))
    separate from its scaling component.

    For cross-dimensional alignment F [d_src, d_tgt], computes:
        U, S, Vt = SVD(F)
        U_orth = U @ Vt  (orthogonal part)
        scale_factor = mean(S)  (average scaling)

    The orthogonal part U_orth preserves norms: ||x @ U_orth|| = ||x||
    when U_orth is square. For non-square, it preserves as much as possible.

    Args:
        alignment: Alignment transform [d_src, d_tgt].
        backend: Compute backend.

    Returns:
        (U_orth, scale_factor) where U_orth is the orthogonal part and
        scale_factor is the average singular value (measure of scaling).
    """
    b = backend
    F = _promote_precision(b.array(alignment), b)
    b.eval(F)

    m, n = int(F.shape[0]), int(F.shape[1])
    k = min(m, n)

    # Compute SVD
    U, S, Vt = geodesic_svd(b, F, k=k)
    b.eval(U, S, Vt)

    # Extract orthogonal part via polar decomposition
    # U_orth = U @ Vt gives the closest orthogonal matrix in O(n)
    U_orth = b.matmul(U, Vt)
    b.eval(U_orth)

    # Enforce det=+1 to get SO(n) (proper rotation) not O(n) (may be reflection)
    # If det < 0, flip the sign of the last column of U before computing U @ Vt
    # This ensures we get a rotation, not a reflection
    if m == n:  # Only check determinant for square matrices
        det_val = b.det(U_orth)
        b.eval(det_val)
        if float(b.to_scalar(det_val)) < 0:
            # Flip last column of U to get det=+1
            # This is equivalent to: U[:, -1] *= -1; U_orth = U @ Vt
            # We can do this post-hoc by flipping the last column of U_orth
            # Since U_orth = U @ Vt, and we want U' @ Vt where U'[:,-1] = -U[:,-1]
            # U' @ Vt = U @ Vt - 2 * U[:,-1:] @ Vt[-1:,:]
            # Simpler: just flip the last column of U_orth
            U_fixed = b.concatenate([U[:, :-1], -U[:, -1:]], axis=1)
            b.eval(U_fixed)
            U_orth = b.matmul(U_fixed, Vt)
            b.eval(U_orth)

    # Compute scale factor as mean of singular values
    # This measures how much F scales on average
    eps = division_epsilon(b, S)
    n_singular = max(1, int(S.shape[0]))
    scale_factor = float(b.to_scalar(b.sum(S))) / n_singular

    return U_orth, max(scale_factor, float(eps))


def svd_auto_rank(
    singular_values: "Array",
    backend: "Backend",
    energy_threshold: float = 0.99,
) -> int:
    """Determine optimal SVD rank by cumulative energy.

    Finds the minimum k such that the top-k singular values capture at least
    energy_threshold fraction of total variance (Frobenius norm squared).

    Formula: k = min{ j : sum(S[:j]^2) / sum(S^2) >= energy_threshold }

    This is the principled way to determine low-rank truncation without
    arbitrary thresholds. Research shows task matrices are inherently low-rank:
    - ~3% of singular components capture 98.5% of task information
    - Remaining components are noise from training dynamics

    Parameters
    ----------
    singular_values : Array
        Singular values from SVD, sorted in descending order.
    backend : Backend
        Backend for tensor operations.
    energy_threshold : float
        Fraction of total energy to preserve. Default 0.99 captures nearly
        all task-specific information while filtering noise.

    Returns
    -------
    int
        Optimal rank k (number of singular values to keep).

    References
    ----------
    - Yu et al. (2025). "TSV-Merge: Task Singular Vectors for Multi-Task Model Merging"
    - Zhang et al. (2025). "STF: Superpose Task-specific Features for Multi-task Fine-tuned Models"
    """
    b = backend
    S = b.astype(
        b.array(singular_values),
        precision_dtype(b, reference=b.array(singular_values)),
    )
    b.eval(S)

    n = int(S.shape[0])
    if n == 0:
        return 0

    # Compute squared singular values (energy per component)
    S_sq = S * S
    b.eval(S_sq)

    # Total energy (Frobenius norm squared)
    total_energy = b.sum(S_sq)
    b.eval(total_energy)
    total_energy_val = float(b.to_scalar(total_energy))

    if total_energy_val <= 0:
        return 0

    # Cumulative sum of squared singular values
    cumsum = b.cumsum(S_sq)
    b.eval(cumsum)

    # Threshold: energy_threshold * total_energy
    threshold = energy_threshold * total_energy_val

    # Find first index where cumsum >= threshold
    mask = cumsum >= threshold
    mask_int = b.astype(mask, "int32")
    b.eval(mask_int)

    # argmax on mask gives first True index (or 0 if all False)
    first_idx = b.argmax(mask_int)
    b.eval(first_idx)
    k = int(b.to_scalar(first_idx)) + 1  # +1 to include that component

    # Clamp to valid range
    k = max(1, min(k, n))

    return k


def geodesic_pinv(backend: "Backend", array: "Array") -> "Array":
    """Compute exact Moore-Penrose pseudo-inverse with session cache reuse."""
    b = backend
    A = _promote_precision(b.array(array), b)
    b.eval(A)

    from modelcypher.core.domain.cache import ComputationCache

    cache = ComputationCache.shared()
    A_pinv = cache.get_or_compute_pinv(A, b)

    return A_pinv


def numerical_rank_truncated_lstsq(
    backend: "Backend",
    source: "Array",
    target: "Array",
) -> tuple["Array", int, int, int, float]:
    """Solve least squares with numerical-rank truncation.

    MATHEMATICAL FOUNDATION (not heuristic):
    ========================================
    Singular values below σ_max × sqrt(ε_machine) are indistinguishable from
    floating-point noise. For float32, ε ≈ 1e-7, so threshold is σ_max × ~3e-4.

    Truncating below this threshold removes numerical garbage, not meaningful
    signal. The alignment operates in k = min(rank_source, rank_target) dimensions
    where both models actually have signal.

    This is mathematically closed-form in the subspace where both models have
    signal - not a heuristic approximation.

    Parameters
    ----------
    backend : Backend
        Backend for tensor operations.
    source : Array
        Source activations [n_samples, d_source].
    target : Array
        Target activations [n_samples, d_target].

    Returns
    -------
    tuple containing:
        F : Array
            Transform matrix [d_source, d_target] mapping source to target space.
        source_rank : int
            Numerical rank of source activations.
        target_rank : int
            Numerical rank of target activations.
        alignment_rank : int
            Rank used for alignment: min(source_rank, target_rank).
        condition_number : float
            Condition number in the truncated space (should be < 1e5 by construction).
    """
    b = backend

    # Promote to highest precision available
    A = _promote_precision(b.array(source), b)
    B = _promote_precision(b.array(target), b)
    b.eval(A, B)

    n_samples, d_source = int(A.shape[0]), int(A.shape[1])
    _, d_target = int(B.shape[0]), int(B.shape[1])

    # Machine precision threshold: sqrt(ε_machine)
    # This is THE threshold below which singular values are noise
    eps = machine_epsilon(b, A)
    precision_thresh = sqrt_scalar(eps, b)

    # SVD of source: A = U @ diag(S) @ Vt
    U_s, S_s, Vt_s = geodesic_svd(b, A)
    b.eval(U_s, S_s, Vt_s)

    # Numerical rank of source: count(σ > σ_max × sqrt(ε))
    if int(S_s.shape[0]) > 0:
        max_s_source = float(b.to_scalar(S_s[0]))  # Singular values are sorted desc
        thresh_source = max_s_source * precision_thresh
        source_rank_mask = S_s > thresh_source
        source_rank_arr = b.sum(b.astype(source_rank_mask, "int32"))
        b.eval(source_rank_arr)
        source_rank = int(b.to_scalar(source_rank_arr))
    else:
        source_rank = 0
        max_s_source = 0.0

    # SVD of target: B = U @ diag(S) @ Vt
    U_t, S_t, Vt_t = geodesic_svd(b, B)
    b.eval(U_t, S_t, Vt_t)

    # Numerical rank of target
    if int(S_t.shape[0]) > 0:
        max_s_target = float(b.to_scalar(S_t[0]))
        thresh_target = max_s_target * precision_thresh
        target_rank_mask = S_t > thresh_target
        target_rank_arr = b.sum(b.astype(target_rank_mask, "int32"))
        b.eval(target_rank_arr)
        target_rank = int(b.to_scalar(target_rank_arr))
    else:
        target_rank = 0

    # Alignment rank: use source_rank for the pseudoinverse computation
    # The target rank is informative but doesn't constrain the solution
    # We truncate to source_rank to remove numerical noise, not to match target
    alignment_rank = source_rank
    alignment_rank = max(1, alignment_rank)  # At least 1 to avoid degenerate case

    logger.info(
        "NUMERICAL RANK: source_rank=%d/%d, target_rank=%d/%d, alignment_rank=%d",
        source_rank, d_source, target_rank, d_target, alignment_rank,
    )

    # Truncate source to top-k singular components
    # A_k = U_k @ diag(S_k) @ Vt_k where k = source_rank (numerical rank of source)
    k = alignment_rank
    U_k = U_s[:, :k]  # [n, k]
    S_k = S_s[:k]  # [k]
    Vt_k = Vt_s[:k, :]  # [k, d_source]
    b.eval(U_k, S_k, Vt_k)

    # Compute condition number in truncated space
    if k > 0 and int(S_k.shape[0]) > 0:
        max_s_k = float(b.to_scalar(S_k[0]))
        min_s_k = float(b.to_scalar(S_k[k - 1]))
        if min_s_k > 0:
            condition_number = max_s_k / min_s_k
        else:
            condition_number = float("inf")
    else:
        condition_number = float("inf")

    logger.info(
        "TRUNCATED CONDITION: κ=%.2e (should be < 1e5)",
        condition_number,
    )

    # Solve in truncated space:
    # We want F such that A @ F ≈ B
    # In truncated space: A_k = U_k @ diag(S_k) @ Vt_k
    # pinv(A_k) = V_k @ diag(1/S_k) @ U_k^T
    # F = pinv(A_k) @ B

    # Compute S_k_inv with safe division
    div_eps = division_epsilon(b, S_k)
    S_k_safe = b.maximum(S_k, b.full(S_k.shape, div_eps))
    S_k_inv = 1.0 / S_k_safe
    b.eval(S_k_inv)

    # pinv(A_k) @ B = V_k @ diag(1/S_k) @ U_k^T @ B
    # Step 1: U_k^T @ B  -> [k, d_target]
    UtB = b.matmul(b.transpose(U_k), B)
    b.eval(UtB)

    # Step 2: diag(1/S_k) @ (U_k^T @ B)  -> [k, d_target]
    # Reshape S_k_inv to [k, 1] for broadcasting
    S_k_inv_col = b.reshape(S_k_inv, (k, 1))
    scaled = S_k_inv_col * UtB
    b.eval(scaled)

    # Step 3: V_k @ (diag(1/S_k) @ U_k^T @ B)  -> [d_source, d_target]
    # V_k = Vt_k^T  -> [d_source, k]
    V_k = b.transpose(Vt_k)
    F = b.matmul(V_k, scaled)
    b.eval(F)

    return F, source_rank, target_rank, alignment_rank, condition_number


def safe_inverse(
    backend: "Backend",
    matrix: "Array",
    regularize: bool = True,
) -> tuple["Array", float]:
    """Compute matrix inverse with condition number check and optional regularization."""
    b = backend
    matrix = _promote_precision(b.array(matrix), b)
    b.eval(matrix)
    dtype = matrix.dtype

    n = int(matrix.shape[0])
    if n == 0:
        return matrix, float("inf")

    _, S, _ = geodesic_svd(b, matrix)
    b.eval(S)

    if int(S.shape[0]) == 0:
        return b.eye(n, dtype=dtype), float("inf")

    max_s_arr = b.max(S)
    b.eval(max_s_arr)
    max_s = float(b.to_scalar(max_s_arr))

    if max_s == 0:
        return b.eye(n, dtype=dtype), float("inf")

    eps = division_epsilon(b, matrix)
    pos_mask = S > eps
    pos_inf = float(b.finfo().max)
    min_candidates = b.where(pos_mask, S, b.full(S.shape, pos_inf))
    min_s_arr = b.min(min_candidates)
    b.eval(min_s_arr)
    min_s = float(b.to_scalar(min_s_arr))

    if min_s <= 0 or min_s >= pos_inf:
        min_s = eps

    cond = max_s / min_s

    cond_thresh = condition_threshold(b, matrix)

    if regularize and cond > 1.0:
        ramp = min(1.0, (cond - 1.0) / (cond_thresh - 1.0)) if cond_thresh > 1.0 else 1.0
        max_reg = regularization_epsilon(b, matrix)
        reg = max_reg * ramp

        if reg > 0:
            matrix = matrix + reg * b.eye(n, dtype=dtype)
            b.eval(matrix)

    inv_matrix = b.inv(matrix)
    b.eval(inv_matrix)

    return inv_matrix, cond


# =============================================================================
# GPU-Accelerated Linear Algebra
# =============================================================================


def newton_schulz_inverse(
    backend: "Backend",
    A: "Array",
    max_iter: int = 15,
    tol: float | None = None,
) -> "Array":
    """Pure matmul matrix inverse via Newton-Schulz iteration.

    Solves: X = A^{-1} using only matmuls (backend-only).

    Algorithm: X_{k+1} = X_k @ (2I - A @ X_k)
    Converges quadratically when ||I - X_0 @ A|| < 1.

    For SPD matrices (like Gram matrices), we use:
        X_0 = I / trace(A)  (scales by average eigenvalue)

    This is the GPU-friendly alternative to solve/cholesky which fall back to CPU.
    """
    b = backend

    A = _promote_precision(b.array(A), b)
    b.eval(A)
    dtype = A.dtype

    n = int(b.shape(A)[0])
    eps = machine_epsilon(b, A)
    if tol is None:
        tol = sqrt_scalar(eps, b) * float(n)  # Scale tolerance by dimension

    # Use Frobenius norm as upper bound on spectral radius
    # ||A||_2 ≤ ||A||_F, so scaling by 1/||A||_F ensures spectral radius ≤ 1
    A_norm = b.norm(A)
    b.eval(A_norm)
    A_norm_val = float(b.to_scalar(A_norm))

    if A_norm_val < eps:
        # Near-zero matrix
        return b.eye(n, dtype=dtype) / eps

    # Scale A to have spectral radius < 1 for convergence.
    # Use 1/||A||_F as conservative scaling.
    scale = 1.0 / A_norm_val
    A_scaled = A * scale

    # Initial guess for scaled problem: X_0 = A^T * scale (matches the spectral structure)
    # For SPD matrices, A^T = A, so X_0 = A * scale^2
    X = b.transpose(A) * (scale * scale)
    I = b.eye(n, dtype=dtype)
    b.eval(X, A_scaled)

    prev_err = float("inf")

    for i in range(max_iter):
        # Newton-Schulz: X' = X @ (2I - A_scaled @ X)
        AX = b.matmul(A_scaled, X)
        diff = 2.0 * I - AX
        X_new = b.matmul(X, diff)
        b.eval(X_new)

        # Check convergence: ||I - A_scaled @ X||_F
        err_mat = I - b.matmul(A_scaled, X_new)
        err = b.norm(err_mat)
        b.eval(err)
        err_val = float(b.to_scalar(err))

        if err_val <= tol:
            logger.debug(
                "Newton-Schulz: Converged in %d iters, ||I - A X||=%.2e",
                i + 1, err_val
            )
            # Scale back: inv(A) = scale * inv(A_scaled)
            return X_new * scale

        if err_val >= prev_err * 1.01:  # Allow small fluctuations
            # Diverging - return current best
            logger.debug(
                "Newton-Schulz: Stalled at iter %d, ||I - A X||=%.2e",
                i + 1, err_val
            )
            return X * scale

        X = X_new
        prev_err = err_val

    return X * scale


def gpu_lstsq(
    backend: "Backend",
    A: "Array",
    B: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Least squares via closed-form normal equations.

    Solves: minimize ||A @ X - B||² for X

    When n >= d (overdetermined), uses DIRECT closed-form:
        F = (A^T @ A + λI)^{-1} @ A^T @ B

    When n < d (underdetermined), uses DUAL closed-form:
        F = A^T @ (A @ A^T + λI)^{-1} @ B

    Both are O(min(n,d)³) via the backend's solve() which uses the most
    appropriate method (Cholesky for SPD matrices). Note: MLX's solve()
    may fall back to CPU for the linear system solve, but matmuls remain
    on GPU. Performance is still excellent due to small matrix dimensions.

    Raises ValueError if the matrix is singular (no fallback to iterative
    methods, as that would change mathematical semantics and violate the
    CKA=1.0 invariant for alignment).

    If stats is provided, populates:
    - iterations: 0 for direct solve
    - residual_norm: final residual norm
    - rhs_norm: right-hand-side norm
    - method: 'normal_equations' or 'normal_equations_dual'
    """
    b = backend

    A = _promote_precision(b.array(A), b)
    A_shape = b.shape(A)
    n, d = int(A_shape[0]), int(A_shape[1])

    B = _promote_precision(b.array(B), b, min_dtype=A.dtype)
    B_shape = b.shape(B)
    if len(B_shape) == 1:
        B = b.reshape(B, (-1, 1))
        squeeze_output = True
    else:
        squeeze_output = False

    b.eval(A, B)

    eps = machine_epsilon(b, A)
    sqrt_eps = sqrt_scalar(eps, b)

    # =========================================================================
    # DIRECT SOLVE via Normal Equations (when n >= d)
    # =========================================================================
    # The pseudoinverse for tall matrices has a closed form:
    #   pinv(A) = (A^T @ A)^{-1} @ A^T
    #
    # So: X = pinv(A) @ B = (A^T @ A)^{-1} @ A^T @ B
    #
    # Let G = A^T @ A (d × d) and H = A^T @ B (d × k)
    # Then X = solve(G, H) using Cholesky (G is positive semidefinite)
    #
    # This is O(d³) instead of O(max_iter × n × d), instant on GPU.
    # =========================================================================

    if n >= d:
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ: Direct solve [%d x %d] -> [%d x %d] (n >= d, using closed-form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A^T @ A (d × d) - this is the Gram matrix in feature space
        A_T = b.transpose(A)
        G = b.matmul(A_T, A)  # d × d
        b.eval(G)

        # Add Tikhonov regularization for numerical stability
        # λ = eps × max(diag(G)) ensures well-conditioning
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))
        reg_lambda = eps * max(max_diag_val, 1.0)

        G_reg = G + reg_lambda * b.eye(d)
        b.eval(G_reg)

        # Compute A^T @ B (d × k)
        H = b.matmul(A_T, B)  # d × k
        b.eval(H)

        # Solve G @ X = H directly
        # The backend's solve() uses the most appropriate method internally
        try:
            X = b.solve(G_reg, H)
            b.eval(X)

            elapsed = time.perf_counter() - start_time

            # Compute residual for logging
            res = b.matmul(A, X) - B
            res_norm = b.norm(res)
            B_norm = b.norm(B)
            b.eval(res_norm, B_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            B_norm_val = float(b.to_scalar(B_norm))

            logger.info(
                "NORMAL_EQ: Solved in %.3fs, residual=%.2e, ||B||=%.2e",
                elapsed, res_norm_val, B_norm_val
            )

            if stats is not None:
                stats["iterations"] = 0.0
                stats["residual_norm"] = res_norm_val
                stats["rhs_norm"] = B_norm_val
                stats["method"] = "normal_equations"

            if squeeze_output:
                X = b.reshape(X, (-1,))

            return X

        except Exception as e:
            # Direct solve failed, fall back to CGLS
            logger.warning(
                "NORMAL_EQ: Direct solve failed (%s), falling back to CGLS",
                str(e)
            )

    # =========================================================================
    # DIRECT SOLVE for Underdetermined Systems (when n < d)
    # =========================================================================
    # For underdetermined systems, use the dual normal equations:
    #   pinv(A) = A^T @ (A @ A^T)^{-1}
    #
    # So: X = pinv(A) @ B = A^T @ (A @ A^T)^{-1} @ B
    #
    # Let G = A @ A^T (n × n) - this is much smaller than d × d!
    # Then solve G @ Y = B, and X = A^T @ Y
    #
    # This is O(n³) instead of O(d³) or iterating.
    # =========================================================================

    elif n < d:  # Underdetermined case
        start_time = time.perf_counter()
        logger.info(
            "NORMAL_EQ_DUAL: Direct solve [%d x %d] -> [%d x %d] (n < d, using dual form)",
            n, d, d, int(b.shape(B)[1])
        )

        # Compute A @ A^T (n × n) - the Gram matrix in sample space
        A_T = b.transpose(A)
        G = b.matmul(A, A_T)  # n × n (much smaller than d × d!)
        b.eval(G)

        # Add minimal Tikhonov regularization for numerical conditioning only.
        #
        # IMPORTANT: Underdetermined (n < d) does NOT mean rank-deficient.
        # If A has full row rank n, then A @ A^T is n×n with full rank and
        # the pseudoinverse X = A^T @ (A @ A^T)^{-1} @ B gives ZERO residual.
        #
        # The regularization (A @ A^T + λI) introduces residual proportional to λ:
        #   A @ X = B - λ(A @ A^T + λI)^{-1} @ B
        #
        # So λ must be minimal - just enough to handle numerical conditioning,
        # not scaled by underdetermination ratio. That scaling introduces
        # artificial residual where none should exist.
        G_diag = b.diag(G)
        max_diag = b.max(b.abs(G_diag))
        b.eval(max_diag)
        max_diag_val = float(b.to_scalar(max_diag))

        # Minimal regularization: just eps * scale for numerical conditioning
        reg_lambda = eps * max(max_diag_val, 1.0)

        G_reg = G + reg_lambda * b.eye(n)
        b.eval(G_reg)

        try:
            # Solve G @ Y = B for Y (n × k)
            Y = b.solve(G_reg, B)
            b.eval(Y)

            # X = A^T @ Y (d × k)
            X = b.matmul(A_T, Y)
            b.eval(X)

            elapsed = time.perf_counter() - start_time

            # Compute residual for logging
            res = b.matmul(A, X) - B
            res_norm = b.norm(res)
            B_norm = b.norm(B)
            b.eval(res_norm, B_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            B_norm_val = float(b.to_scalar(B_norm))

            # Log residual for diagnostics
            # With minimal regularization, residual should be near-zero for full-rank A
            relative_residual = res_norm_val / max(B_norm_val, eps)
            logger.info(
                "NORMAL_EQ_DUAL: Solved in %.3fs, residual=%.2e (%.2f%% of ||B||=%.2e)",
                elapsed, res_norm_val, 100.0 * relative_residual, B_norm_val
            )

            if stats is not None:
                stats["iterations"] = 0.0
                stats["residual_norm"] = res_norm_val
                stats["rhs_norm"] = B_norm_val
                stats["method"] = "normal_equations_dual"

            if squeeze_output:
                X = b.reshape(X, (-1,))

            return X

        except Exception as e:
            logger.warning(
                "NORMAL_EQ_DUAL: Direct solve failed (%s), falling back to CGLS",
                str(e)
            )

    # =========================================================================
    # CGLS Fallback (only if direct solves failed or n == d exactly)
    # =========================================================================
    # CRITICAL: This is necessary for underdetermined systems because:
    #   - The dual normal equations give the MINIMUM-NORM solution
    #   - CGLS gives the MINIMUM-RESIDUAL solution
    #   - For underdetermined systems (n < d), these are DIFFERENT
    #   - Minimum-norm shrinks X toward zero → A @ X ≈ 0 → residual ≈ ||B||
    #   - Minimum-residual iterates until A @ X ≈ B (small residual)
    # For stitch computation, we need the solution that FITS the target weights.
    # =========================================================================

    # Precondition: scale columns to unit norm
    col_norms = b.norm(A, axis=0)
    col_norms = col_norms + eps
    inv_col_norms = 1.0 / col_norms
    row_scale = b.reshape(inv_col_norms, (int(d), 1))
    diag_reg = b.reshape(inv_col_norms * inv_col_norms * eps, (int(d), 1))
    A_T = b.transpose(A)
    b.eval(col_norms, inv_col_norms, row_scale, diag_reg, A_T)

    # Right-hand side: A_hat^T B
    rhs = b.matmul(A_T, B)
    rhs = rhs * row_scale
    b.eval(rhs)

    # Solve (A_hat^T A_hat + diag_reg) Y = rhs
    Y = b.zeros((int(d), int(b.shape(B)[1])), dtype=precision_dtype(b, reference=A))
    R = rhs
    P = rhs
    b.eval(Y, R, P)

    rhs_norm_sq = b.sum(rhs * rhs)
    b.eval(rhs_norm_sq)
    rhs_norm_sq_val = float(b.to_scalar(rhs_norm_sq))
    rhs_norm = sqrt_scalar(rhs_norm_sq_val, b)
    sqrt_d = sqrt_scalar(float(d), b)
    rhs_scale = rhs_norm / max(sqrt_d, 1.0)
    tol = sqrt_eps * max(rhs_scale, eps)

    B_norm_arr = b.norm(B)
    b.eval(B_norm_arr)
    B_norm_val = float(b.to_scalar(B_norm_arr))
    sqrt_n = sqrt_scalar(float(n), b)
    tol_primal = sqrt_eps * max(B_norm_val / max(sqrt_n, 1.0), eps)

    rnorm_sq_val = rhs_norm_sq_val
    rnorm_val = rhs_norm
    prev_rnorm = rnorm_val

    refresh_interval = max(1, int(min(n, d)))
    restart_budget = int(max(1, ceil_scalar(log2_scalar(1.0 / eps, b), b)))
    max_iter = max(1, int(d)) * restart_budget

    # Log CGLS start
    start_time = time.perf_counter()
    log_interval = max(500, max_iter // 20)  # Log ~20 times during run
    logger.info(
        "CGLS: Starting [%d x %d] -> [%d x %d], max_iter=%d, tol=%.2e",
        n, d, d, int(b.shape(B)[1]), max_iter, tol
    )

    iterations_used = 0
    for step in range(max_iter):
        # Apply normal equations with preconditioning + Tikhonov
        P_scaled = P * row_scale
        AP = b.matmul(A, P_scaled)
        ATAP = b.matmul(A_T, AP)
        ATAP = ATAP * row_scale
        ATAP = ATAP + diag_reg * P
        b.eval(ATAP)

        denom = b.sum(P * ATAP)
        b.eval(denom)
        denom_val = float(b.to_scalar(denom))
        if denom_val <= eps:
            denom_val = eps

        alpha = rnorm_sq_val / denom_val
        Y = Y + alpha * P
        R = R - alpha * ATAP
        b.eval(Y, R)

        old_rnorm_sq = rnorm_sq_val
        rnorm_sq = b.sum(R * R)
        b.eval(rnorm_sq)
        rnorm_sq_val = float(b.to_scalar(rnorm_sq))
        rnorm_val = sqrt_scalar(rnorm_sq_val, b)

        iterations_used = step + 1

        # Progress logging
        if iterations_used % log_interval == 0:
            elapsed = time.perf_counter() - start_time
            iters_per_sec = iterations_used / max(elapsed, 0.001)
            remaining = (max_iter - iterations_used) / max(iters_per_sec, 0.001)
            logger.info(
                "CGLS: iter %d/%d (%.1f%%), residual=%.2e, %.1f iter/s, ~%.0fs remaining",
                iterations_used, max_iter, 100.0 * iterations_used / max_iter,
                rnorm_val, iters_per_sec, remaining
            )

        if rnorm_val <= tol:
            X_tmp = b.matmul(A, Y * row_scale)
            res = X_tmp - B
            res_norm = b.norm(res)
            b.eval(res_norm)
            res_norm_val = float(b.to_scalar(res_norm))
            if res_norm_val <= tol_primal:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "CGLS: Converged at iter %d (%.2fs), residual=%.2e, primal=%.2e",
                    iterations_used, elapsed, rnorm_val, res_norm_val
                )
                break

        # Refresh residuals on drift or periodic cadence
        stagnation_threshold = sqrt_eps * max(prev_rnorm, rhs_norm, eps)
        if (
            not is_finite(rnorm_val, b)
            or rnorm_val > prev_rnorm * (1.0 + sqrt_eps)
            or rnorm_val >= prev_rnorm - stagnation_threshold
        ):
            X_tmp = b.matmul(A, Y * row_scale)
            R = b.matmul(A_T, X_tmp)
            R = rhs - R * row_scale - diag_reg * Y
            b.eval(R)
            rnorm_sq = b.sum(R * R)
            b.eval(rnorm_sq)
            rnorm_sq_val = float(b.to_scalar(rnorm_sq))
            rnorm_val = sqrt_scalar(rnorm_sq_val, b)
            P = R
            prev_rnorm = rnorm_val
            continue

        if (step + 1) % refresh_interval == 0:
            X_tmp = b.matmul(A, Y * row_scale)
            R = b.matmul(A_T, X_tmp)
            R = rhs - R * row_scale - diag_reg * Y
            b.eval(R)
            rnorm_sq = b.sum(R * R)
            b.eval(rnorm_sq)
            rnorm_sq_val = float(b.to_scalar(rnorm_sq))
            rnorm_val = sqrt_scalar(rnorm_sq_val, b)
            P = R
            prev_rnorm = rnorm_val
            continue

        beta = rnorm_sq_val / max(old_rnorm_sq, eps)
        P = R + beta * P
        b.eval(P)
        prev_rnorm = rnorm_val

    X = Y * row_scale
    b.eval(X)

    # Log completion
    total_elapsed = time.perf_counter() - start_time
    if iterations_used >= max_iter:
        logger.warning(
            "CGLS: Hit max_iter=%d (%.2fs), residual=%.2e (may not have converged)",
            max_iter, total_elapsed, rnorm_val
        )
    else:
        logger.info(
            "CGLS: Completed in %d iters (%.2fs), final residual=%.2e",
            iterations_used, total_elapsed, rnorm_val
        )

    if stats is not None:
        stats["iterations"] = float(iterations_used)
        stats["residual_norm"] = float(rnorm_val)
        stats["rhs_norm"] = float(rhs_norm)
        stats["method"] = "cgls"

    if squeeze_output:
        X = b.reshape(X, (-1,))

    return X


# =============================================================================
# Invariant Alignment: Linear CKA = 1.0 by Construction
# =============================================================================


def invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Compute the linear alignment transform F = pinv(source) @ target.

    Uses the Moore-Penrose pseudoinverse to solve the least-squares alignment and
    optionally truncates to the effective rank from the source Gram spectrum.

    References:
        - Penrose, R. (1955). "A generalized inverse for matrices."
          Proceedings of the Cambridge Philosophical Society.
    """
    b = backend

    source = _promote_precision(b.array(source), b)
    target = _promote_precision(b.array(target), b, min_dtype=source.dtype)
    b.eval(source, target)

    # Center both matrices (CKA uses centered Gram matrices)
    source_mean = b.mean(source, axis=0, keepdims=True)
    target_mean = b.mean(target, axis=0, keepdims=True)
    source_c = source - source_mean
    target_c = target - target_mean
    b.eval(source_c, target_c)

    # THE FORMULA: F = pinv(source) @ target (solved via GPU CGLS)
    F = gpu_lstsq(b, source_c, target_c, stats=stats)
    b.eval(F)

    # =========================================================================
    # GEOMETRY CHECK: Warn if underdetermined (n_samples < d_source)
    # =========================================================================
    # When n < d, the least squares solution has infinitely many solutions.
    # The minimum-norm solution (from gpu_lstsq) is mathematically valid but
    # may overfit to the specific probes used. Ensure sufficient probe coverage
    # by using a geometry-derived probe count (n >= d_source).
    #
    # We DON'T truncate F because:
    # 1. Truncation might discard source-unique knowledge that appears as small
    #    eigenvalues but is actually valuable for transfer
    # 2. With proper probe coverage (n > d), this isn't needed anyway
    # 3. The null-space projection stage handles transfer decisions
    n_samples = int(b.shape(source_c)[0])
    d_source = int(b.shape(source_c)[1])

    if n_samples < d_source:
        # Compute the under-determination ratio for informational logging
        ratio = d_source / n_samples
        logger.info(
            "UNDERDETERMINED ALIGNMENT: n=%d < d=%d (ratio=%.1fx). "
            "Using sqrt-scaled Tikhonov regularization for stability.",
            n_samples, d_source, ratio
        )
        if stats is not None:
            stats["underdetermined"] = 1.0
            stats["n_samples"] = float(n_samples)
            stats["d_source"] = float(d_source)
            stats["underdetermination_ratio"] = ratio

    return F


def geodesic_invariant_alignment(
    backend: "Backend",
    source: "Array",
    target: "Array",
    stats: dict[str, float] | None = None,
) -> "Array":
    """Find alignment F that preserves geodesic manifold structure.

    Computes geodesic cosine matrices, aligns them via Procrustes, and
    recovers a feature-space transform.

    Reference: Yu et al. 2025 "Relative Geodesic Representations" (NeurIPS).
    """
    from modelcypher.core.domain.geometry.riemannian_utils import geodesic_cosine_matrix

    b = backend

    source = _promote_precision(b.array(source), b)
    target = _promote_precision(b.array(target), b, min_dtype=source.dtype)
    b.eval(source, target)

    n_samples = int(b.shape(source)[0])
    d_source = int(b.shape(source)[1])
    d_target = int(b.shape(target)[1])

    if stats is not None:
        stats["geodesic_alignment"] = 1.0
        stats["n_samples"] = float(n_samples)
        stats["d_source"] = float(d_source)
        stats["d_target"] = float(d_target)

    # Step 1: Compute geodesic cosine matrices (relative representations)
    # Each row represents a point by its geodesic similarities to all others
    t0 = time.time()
    G_source = geodesic_cosine_matrix(source, b)
    G_target = geodesic_cosine_matrix(target, b)
    b.eval(G_source, G_target)
    t_geo = time.time() - t0

    if stats is not None:
        stats["geodesic_time_sec"] = t_geo

    logger.info(
        "Geodesic cosine matrices computed: source %s, target %s (%.2fs)",
        b.shape(G_source), b.shape(G_target), t_geo
    )

    # Step 2: Center both matrices for Procrustes
    G_source_mean = b.mean(G_source, axis=0, keepdims=True)
    G_target_mean = b.mean(G_target, axis=0, keepdims=True)
    G_source_c = G_source - G_source_mean
    G_target_c = G_target - G_target_mean
    b.eval(G_source_c, G_target_c)

    # Step 3: Procrustes alignment in relative space
    # Find R such that G_source @ R ≈ G_target
    # SVD of G_source.T @ G_target = U @ S @ V.T, then R = U @ V.T
    C = b.matmul(b.transpose(G_source_c), G_target_c)
    b.eval(C)

    U, S, Vt = b.svd(C)
    b.eval(U, S, Vt)

    R = b.matmul(U, Vt)
    b.eval(R)

    # Ensure proper rotation (det = +1)
    det_val = b.det(R)
    b.eval(det_val)
    det_scalar = float(b.to_scalar(det_val))
    if det_scalar < 0:
        # Flip sign of last column of U to get proper rotation
        n_cols = int(b.shape(U)[1])
        sign_arr = b.ones((n_cols,))
        idx = b.arange(n_cols)
        sign_arr = b.where(idx == (n_cols - 1), b.full(sign_arr.shape, -1.0), sign_arr)
        U = U * sign_arr
        R = b.matmul(U, Vt)
        b.eval(R)

    if stats is not None:
        # Measure alignment quality in relative space
        G_aligned = b.matmul(G_source_c, R)
        b.eval(G_aligned)
        diff = G_aligned - G_target_c
        frob_norm = b.sqrt(b.sum(diff * diff))
        target_norm = b.sqrt(b.sum(G_target_c * G_target_c))
        b.eval(frob_norm, target_norm)
        denom_eps = division_epsilon(b, target_norm)
        rel_error = float(b.to_scalar(frob_norm)) / max(float(b.to_scalar(target_norm)), denom_eps)
        stats["relative_space_alignment_error"] = rel_error
        logger.info("Relative space alignment error: %.6f", rel_error)

    # Step 4: Transfer through aligned relative space
    # transferred = G_source @ R @ pinv(G_target) @ target
    # Use SVD for stable pseudo-inverse of G_target

    G_aligned = b.matmul(G_source, R)
    b.eval(G_aligned)

    # Stable pseudo-inverse of G_target via SVD
    U_t, S_t, Vt_t = b.svd(G_target)
    b.eval(U_t, S_t, Vt_t)

    # =========================================================================
    # PRINCIPLED RANK DETERMINATION VIA GEODESIC INTRINSIC DIMENSION
    # =========================================================================
    # OLD HEURISTIC: threshold = eps * max_s * n_samples (arbitrary)
    #
    # NEW PRINCIPLED: Use intrinsic dimension of target manifold.
    # The effective rank of G_target (similarity matrix) equals the intrinsic
    # dimension of the underlying manifold - no magic numbers needed.
    #
    # Reference: Facco et al. (2017) TwoNN with geodesic distances
    # =========================================================================
    from modelcypher.core.domain.geometry.intrinsic_dimension import IntrinsicDimension

    id_estimator = IntrinsicDimension(b)
    id_result = id_estimator.compute(target)
    intrinsic_dim = id_result.intrinsic_dimension

    # Effective rank = intrinsic dimension (clamped to valid range)
    effective_rank = max(1, min(int(round(intrinsic_dim)), n_samples))

    logger.info(
        "Geodesic alignment: intrinsic_dim=%.2f, effective_rank=%d/%d",
        intrinsic_dim, effective_rank, n_samples
    )

    # Keep top effective_rank singular values, zero out the rest
    # Create mask: True for indices < effective_rank
    indices = b.arange(len(S_t))
    rank_mask = indices < effective_rank
    rank_mask_float = b.astype(rank_mask, S_t.dtype)

    # Safe inversion: replace zeros with ones before division, then mask out
    S_t_safe = b.where(S_t > machine_epsilon(b, S_t), S_t, b.ones_like(S_t))
    S_t_inv = rank_mask_float / S_t_safe
    b.eval(S_t_inv)

    # pinv(G_target) = V @ diag(S_inv) @ U.T
    G_target_pinv = b.matmul(b.transpose(Vt_t), S_t_inv[:, None] * b.transpose(U_t))
    b.eval(G_target_pinv)

    # transferred = G_aligned @ pinv(G_target) @ target
    temp = b.matmul(G_aligned, G_target_pinv)
    transferred = b.matmul(temp, target)
    b.eval(transferred)

    # Step 5: Recover feature-space transform F
    # F = lstsq(source, transferred) using gpu_lstsq for stability
    F = gpu_lstsq(b, source, transferred, stats=stats)
    b.eval(F)

    logger.info(
        "Geodesic alignment complete: F shape %s, d_source=%d, d_target=%d",
        b.shape(F), d_source, d_target
    )

    return F

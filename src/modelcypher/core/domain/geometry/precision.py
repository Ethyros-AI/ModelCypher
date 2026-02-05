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

"""Dtype precision detection and epsilon/threshold utilities.

Epsilons and tolerances derive from tensor precision. Compute precision never
exceeds the model's ceiling (float32 max); upcasting to float64 adds zeros, not
information.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .scalars import sqrt_scalar, ulp_scalar

if TYPE_CHECKING:
    from modelcypher.ports.backend import Array, Backend

logger = logging.getLogger(__name__)


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
        dtype: A dtype object from the active backend runtime.

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
    if "bool" in name:
        # Boolean arrays should be converted to float for numerical operations
        # Treat as needing float32 precision
        return 23

    # Unknown dtype - assume float32 level
    logger.warning("Unknown dtype %s, assuming float32 precision", name)
    return 23


def _dtype_name(dtype: object) -> str:
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None) or str(dtype)
    parts = name.split(".")
    return parts[-1] if parts else name


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
    """Select compute dtype from source/target model precision.

    Chooses the higher precision of the two, promotes bf16 to float32, caps at
    float32, and floors quantized weights to float16.

    Args:
        source_weights: Source model weight dict.
        target_weights: Target model weight dict.
        backend: Backend for tensor operations.

    Returns:
        Compute dtype for merge operations.
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

    # Boolean references should use float32 precision (cannot use finfo on bool)
    ref_dtype_name = _dtype_name(reference.dtype).lower()
    if "bool" in ref_dtype_name:
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


def _float_dtype_for(array: "Array | None", backend: "Backend") -> object:
    """Get float dtype from array or fall back to default."""
    if array is not None and hasattr(array, "dtype"):
        name = _dtype_name(array.dtype)
        if "float" in name:
            return array.dtype
    return _default_float_dtype(backend)


def _promote_precision(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: object | None = None,
    promote_ints: bool = True,
    promote_bools: bool = True,
) -> "Array":
    """Promote low-precision or integer arrays to at least default float dtype."""
    if min_dtype is None:
        min_dtype = _default_float_dtype(backend)

    if not hasattr(array, "dtype"):
        return backend.array(array, dtype=min_dtype)

    dtype_name = _dtype_name(array.dtype).lower()
    is_low_float = "float16" in dtype_name or "bfloat16" in dtype_name
    is_int = "int" in dtype_name or "uint" in dtype_name
    is_bool = "bool" in dtype_name

    if is_low_float or (promote_ints and is_int) or (promote_bools and is_bool):
        return backend.astype(array, min_dtype)

    try:
        current_eps = backend.finfo(array.dtype).eps
        min_eps = backend.finfo(min_dtype).eps
    except Exception:
        return backend.astype(array, min_dtype)

    if current_eps > min_eps:
        return backend.astype(array, min_dtype)

    return array


def _promote_precision_float32(
    array: "Array",
    backend: "Backend",
    *,
    min_dtype: object | None = "float32",
) -> "Array":
    """Promote to float32 without promoting ints/bools."""
    return _promote_precision(
        array,
        backend,
        min_dtype=min_dtype,
        promote_ints=False,
        promote_bools=False,
    )


def _mask_sum(
    mask: "Array",
    backend: "Backend",
    *,
    dtype_source: "Array | None" = None,
) -> "Array":
    """Sum a boolean mask with proper float dtype."""
    dtype = _float_dtype_for(dtype_source, backend)
    return backend.sum(backend.astype(mask, dtype))


# =============================================================================
# Epsilon and Threshold Utilities
# =============================================================================


def machine_epsilon(backend: "Backend", array: "Array") -> float:
    """Get machine epsilon for the array's dtype."""
    return backend.finfo(array.dtype).eps


def model_eps() -> float:
    """Get machine epsilon using default backend and precision.

    Convenience function for contexts where no backend/array is available.
    Uses the default backend with the current precision dtype.
    """
    from modelcypher.core.domain._backend import get_default_backend

    b = get_default_backend()
    return machine_epsilon(b, b.array([1.0], dtype=precision_dtype(b)))


def division_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for safe division operations. Uses sqrt(eps)."""
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


def regularization_epsilon(backend: "Backend", array: "Array") -> float:
    """Get epsilon for matrix regularization.

    Uses sqrt(eps), the standard numerical analysis threshold for relative
    precision. Values below sqrt(eps) × scale are indistinguishable from
    roundoff noise in relative terms.
    """
    eps = backend.finfo(array.dtype).eps
    return sqrt_scalar(eps, backend)


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
    """Return the value before the largest relative gap in a sorted magnitude list."""
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
        first_val = backend.take(values_arr, backend.array([0]), axis=0)
        second_val = backend.take(values_arr, backend.array([1]), axis=0)
        first_val = backend.squeeze(first_val)
        second_val = backend.squeeze(second_val)
        backend.eval(first_val, second_val)
        v0 = float(backend.to_scalar(first_val))
        v1 = float(backend.to_scalar(second_val))
        if v0 > eps:
            rel_gap = (v1 - v0) / v0
        else:
            rel_gap = 0.0
        if rel_gap <= eps:
            return v1
        return v0

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

    if max_gap <= eps:
        last_idx = backend.array([n - 1])
        last_val = backend.take(values_arr, last_idx, axis=0)
        last_val = backend.squeeze(last_val)
        backend.eval(last_val)
        return float(backend.to_scalar(last_val))

    gap_index = int(backend.to_scalar(gap_index_arr))
    gap_val = backend.take(values_arr, backend.array([gap_index]), axis=0)
    gap_val = backend.squeeze(gap_val)
    backend.eval(gap_val)
    return float(backend.to_scalar(gap_val))


__all__ = [
    # Model-driven precision detection
    "dtype_precision_bits",
    "detect_model_dtype",
    "compute_precision_for_merge",
    "set_model_compute_dtype",
    "get_model_compute_dtype",
    "precision_dtype",
    # Internal helpers (exported for other modules in this package)
    "_dtype_name",
    "_default_float_dtype",
    "_float_dtype_for",
    "_promote_precision",
    "_promote_precision_float32",
    "_mask_sum",
    # Epsilon and threshold utilities
    "machine_epsilon",
    "model_eps",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    "infinity_threshold",
    # Data-derived thresholds
    "find_magnitude_gap_threshold",
]

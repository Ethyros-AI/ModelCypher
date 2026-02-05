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

"""Numerical stability utilities with model-driven precision.

This module re-exports from focused submodules for backward compatibility.
For new code, import directly from the submodules:

- scalars: Backend scalar helpers (sqrt_scalar, is_finite, etc.)
- precision: Dtype detection, epsilon/threshold utilities
- statistics: Median, correlation functions
- decomposition: SVD, pseudoinverse, null space projector
- alignment: Invariant alignment, geodesic alignment
- spectral_init: Spectral-normalized weight initialization
- validation: Array validation, convergence monitoring
"""

from __future__ import annotations

# Re-export from scalars
from .scalars import (
    sqrt_scalar,
    is_finite,
    is_inf,
    is_nan,
    all_finite,
    log_scalar,
    exp_scalar,
    power_scalar,
    ceil_scalar,
    floor_scalar,
    ulp_scalar,
    lgamma_scalar,
    acos_scalar,
    cos_scalar,
    sin_scalar,
    atan2_scalar,
    log2_scalar,
    pi_value,
    e_value,
    inf_value,
)

# Re-export from precision
from .precision import (
    dtype_precision_bits,
    detect_model_dtype,
    compute_precision_for_merge,
    set_model_compute_dtype,
    get_model_compute_dtype,
    precision_dtype,
    _dtype_name,
    _default_float_dtype,
    _float_dtype_for,
    _promote_precision,
    _promote_precision_float32,
    _mask_sum,
    machine_epsilon,
    model_eps,
    division_epsilon,
    regularization_epsilon,
    condition_threshold,
    svd_rank_threshold,
    tiny_value,
    safe_log_epsilon,
    infinity_threshold,
    find_magnitude_gap_threshold,
)

# Re-export from statistics
from .statistics import (
    compute_median,
    compute_median_nonzero,
    compute_pearson_correlation,
    compute_spearman_correlation,
)

# Re-export from decomposition
from .decomposition import (
    power_iteration_eigh,
    geodesic_svd,
    orthogonalize_alignment,
    orthogonalize_alignment_full,
    svd_auto_rank,
    geodesic_pinv,
    null_space_projector,
    numerical_rank_truncated_lstsq,
    safe_inverse,
    gpu_lstsq,
)

# Re-export from alignment
from .alignment import (
    invariant_alignment,
    geodesic_invariant_alignment,
)

# Re-export from spectral_init
from .spectral_init import (
    spectral_normalized_init,
    spectral_normalized_lora_init,
)

# Re-export from validation
from .validation import (
    ArrayNumerics,
    validate_array_numerics,
    count_nan,
    count_inf,
    count_nonfinite,
    ConvergenceState,
    ConvergenceMonitor,
)


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
    # Epsilon and threshold utilities
    "machine_epsilon",
    "model_eps",
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
    "null_space_projector",
    "power_iteration_eigh",
    "numerical_rank_truncated_lstsq",
    "svd_auto_rank",
    # GPU-accelerated linear algebra
    "gpu_lstsq",
    # Orthogonalization (polar decomposition)
    "orthogonalize_alignment",
    "orthogonalize_alignment_full",
    # Invariant alignment (linear CKA = 1.0 by construction)
    "invariant_alignment",
    # Geodesic invariant alignment (preserves manifold structure)
    "geodesic_invariant_alignment",
    # Spectral-normalized initialization
    "spectral_normalized_init",
    "spectral_normalized_lora_init",
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

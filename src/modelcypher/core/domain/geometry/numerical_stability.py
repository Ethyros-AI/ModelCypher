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

"""Canonical numerical stability exports with model-driven precision.

This module is the geometry-wide import surface for numerical precision,
decomposition, alignment, and validation helpers. The focused submodules remain
the implementation units:

- scalars: Backend scalar helpers (sqrt_scalar, is_finite, etc.)
- precision: Dtype detection, epsilon/threshold utilities
- statistics: Median, correlation functions
- decomposition: SVD, pseudoinverse, null space projector
- alignment: Invariant alignment, geodesic alignment
- spectral_init: Spectral-normalized weight initialization
- validation: Array validation, convergence monitoring
"""

from __future__ import annotations

# Canonical exports from alignment
from .alignment import (
    geodesic_invariant_alignment,
    invariant_alignment,
)

# Canonical exports from decomposition
from .decomposition import (
    geodesic_pinv,
    geodesic_svd,
    gpu_lstsq,
    null_space_projector,
    numerical_rank_truncated_lstsq,
    orthogonalize_alignment,
    orthogonalize_alignment_full,
    power_iteration_eigh,
    safe_inverse,
    svd_auto_rank,
)

# Precision internals used across geometry submodules
from .precision import _float_dtype_for as _float_dtype_for  # noqa: F401
from .precision import _mask_sum as _mask_sum  # noqa: F401
from .precision import _promote_precision as _promote_precision  # noqa: F401
from .precision import (  # noqa: F401
    _promote_precision_float32 as _promote_precision_float32,
)

# Canonical public exports from precision
from .precision import (
    compute_precision_for_merge,
    condition_threshold,
    detect_model_dtype,
    division_epsilon,
    dtype_precision_bits,
    find_magnitude_gap_threshold,
    get_model_compute_dtype,
    infinity_threshold,
    machine_epsilon,
    model_eps,
    precision_dtype,
    regularization_epsilon,
    safe_log_epsilon,
    set_model_compute_dtype,
    svd_rank_threshold,
    tiny_value,
)

# Canonical exports from scalars
from .scalars import (
    acos_scalar,
    all_finite,
    atan2_scalar,
    ceil_scalar,
    cos_scalar,
    e_value,
    exp_scalar,
    floor_scalar,
    inf_value,
    is_finite,
    is_inf,
    is_nan,
    lgamma_scalar,
    log2_scalar,
    log_scalar,
    pi_value,
    power_scalar,
    sin_scalar,
    sqrt_scalar,
    ulp_scalar,
)

# Canonical exports from spectral_init
from .spectral_init import (
    spectral_normalized_init,
    spectral_normalized_lora_init,
)

# Canonical exports from statistics
from .statistics import (
    compute_median,
    compute_median_nonzero,
    compute_pearson_correlation,
    compute_spearman_correlation,
)

# Re-export from validation
from .validation import (
    ArrayNumerics,
    ConvergenceMonitor,
    ConvergenceState,
    count_inf,
    count_nan,
    count_nonfinite,
    validate_array_numerics,
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

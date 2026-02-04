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

"""Shared low-level primitives for geometry computations.

This module provides foundational utilities used across the geometry module:
- epsilon_utils: Dtype-derived epsilon and threshold utilities
- convergence: Unified convergence monitoring for iterative algorithms
- validation: Array validation utilities (NaN, Inf detection)

All operations stay on GPU via the Backend protocol. No NumPy.
"""

from modelcypher.core.domain.geometry.numerical_stability import (
    ceil_scalar,
    condition_threshold,
    division_epsilon,
    e_value,
    exp_scalar,
    floor_scalar,
    inf_value,
    infinity_threshold,
    is_finite,
    is_inf,
    is_nan,
    lgamma_scalar,
    log2_scalar,
    log_scalar,
    machine_epsilon,
    pi_value,
    power_scalar,
    regularization_epsilon,
    safe_log_epsilon,
    sqrt_scalar,
    svd_rank_threshold,
    tiny_value,
    ulp_scalar,
)
from modelcypher.core.domain.geometry._primitives.convergence import (
    ConvergenceMonitor,
    ConvergenceState,
)
from modelcypher.core.domain.geometry._primitives.validation import (
    count_inf,
    count_nan,
    count_nonfinite,
    validate_array_numerics,
)

__all__ = [
    # Scalar helpers
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
    "log2_scalar",
    "pi_value",
    "e_value",
    "inf_value",
    # Epsilon utilities
    "machine_epsilon",
    "division_epsilon",
    "regularization_epsilon",
    "condition_threshold",
    "svd_rank_threshold",
    "tiny_value",
    "safe_log_epsilon",
    "infinity_threshold",
    # Convergence
    "ConvergenceMonitor",
    "ConvergenceState",
    # Validation
    "validate_array_numerics",
    "count_nan",
    "count_inf",
    "count_nonfinite",
]

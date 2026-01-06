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

"""Tests for riemannian_validation.py - Numeric validation."""

from __future__ import annotations

import pytest
from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.riemannian_validation import (
    all_finite,
    count_finite,
    count_inf,
    count_nan,
    count_nonfinite,
    has_inf,
    has_nan,
    safe_arithmetic_mean,
    set_matrix_element,
    validate_array_numerics,
)


class TestRiemannianValidation:
    """Tests for numerical checks using backend."""

    def test_nan_detection(self):
        """Detect NaNs correctly."""
        backend = get_default_backend()
        arr = backend.array([1.0, float("nan"), 3.0])
        backend.eval(arr)
        
        assert count_nan(arr, backend) == 1
        assert has_nan(arr, backend) is True
        assert all_finite(arr, backend) is False

    def test_inf_detection(self):
        """Detect Infinity correctly."""
        backend = get_default_backend()
        arr = backend.array([1.0, float("inf"), float("-inf")])
        backend.eval(arr)
        
        assert count_inf(arr, backend) == 2
        assert has_inf(arr, backend) is True
        assert all_finite(arr, backend) is False

    def test_finite_counting(self):
        """Count finite values."""
        backend = get_default_backend()
        arr = backend.array([1.0, float("nan"), float("inf"), 4.0])
        backend.eval(arr)
        
        assert count_finite(arr, backend) == 2
        assert count_nonfinite(arr, backend) == 2

    def test_validate_array_numerics(self):
        """Validate numerics returns all counts."""
        backend = get_default_backend()
        arr = backend.array([1.0, float("nan"), float("inf"), 4.0])
        backend.eval(arr)
        
        nan_c, inf_c, nonfin_c = validate_array_numerics(arr, backend)
        assert nan_c == 1
        assert inf_c == 1
        assert nonfin_c == 2

    def test_safe_arithmetic_mean(self):
        """Compute mean safely."""
        assert safe_arithmetic_mean([1.0, 2.0, 3.0]) == 2.0
        assert safe_arithmetic_mean([]) == 0.0

    def test_set_matrix_element(self):
        """Set matrix element correctly."""
        backend = get_default_backend()
        mat = backend.zeros((2, 2))
        mat = set_matrix_element(backend, mat, 0, 1, 5.0)
        backend.eval(mat)
        
        val = float(backend.to_scalar(mat[0, 1]))
        assert abs(val - 5.0) < 1e-6

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

"""Extended tests for transplant helper functions.

Tests critical helper APIs:
- _promote_precision(): Dtype promotion for numerical stability
- _geodesic_pinv(): Pseudo-inverse computation with fallback
- _set_submatrix(): Submatrix insertion
- _compute_dimension_projection(): Orthogonal projection between dims
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    all_finite,
    machine_epsilon,
    regularization_epsilon,
)
from modelcypher.core.use_cases.merge.stages.transplant_helpers import (
    _compute_dimension_projection,
    _geodesic_pinv,
    _promote_precision,
    _set_submatrix,
)


@pytest.fixture
def backend():
    return get_default_backend()


class TestPromotePrecision:
    """Tests for _promote_precision()."""

    def test_float32_unchanged(self, backend):
        """float32 arrays should pass through unchanged."""
        arr = backend.random_normal((8, 8))
        arr = backend.astype(arr, "float32")
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        # Should be same object or same dtype
        assert "float32" in str(backend.dtype(result)).lower()

    def test_float16_promoted(self, backend):
        """float16 arrays should be promoted to float32."""
        arr = backend.random_normal((8, 8))
        arr = backend.astype(arr, "float16")
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        assert "float32" in str(backend.dtype(result)).lower()

    def test_bfloat16_promoted(self, backend):
        """bfloat16 arrays should be promoted to float32."""
        arr = backend.random_normal((8, 8))
        try:
            arr = backend.astype(arr, "bfloat16")
            backend.eval(arr)

            result = _promote_precision(arr, backend)

            assert "float32" in str(backend.dtype(result)).lower()
        except (ValueError, TypeError):
            # Some backends may not support bfloat16
            pytest.skip("bfloat16 not supported by backend")

    def test_preserves_values(self, backend):
        """Promotion should preserve numerical values (within precision)."""
        arr = backend.random_normal((8, 8))
        arr = backend.astype(arr, "float16")
        backend.eval(arr)

        result = _promote_precision(arr, backend)

        # Convert both to float32 for comparison
        arr_f32 = backend.astype(arr, "float32")
        diff = backend.mean(backend.abs(result - arr_f32))
        backend.eval(diff)

        tol = machine_epsilon(backend, arr)
        assert float(backend.to_scalar(diff)) <= tol  # Within float16 precision


class TestGeodesicPinv:
    """Tests for _geodesic_pinv()."""

    def test_square_matrix(self, backend):
        """Pseudo-inverse of square matrix should work."""
        A = backend.random_normal((8, 8))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        assert A_pinv is not None
        assert backend.shape(A_pinv) == (8, 8)
        assert all_finite(A_pinv, backend)

    def test_tall_matrix(self, backend):
        """Pseudo-inverse of tall matrix (n > m) should work."""
        A = backend.random_normal((16, 8))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        assert A_pinv is not None
        assert backend.shape(A_pinv) == (8, 16)
        assert all_finite(A_pinv, backend)

    def test_wide_matrix(self, backend):
        """Pseudo-inverse of wide matrix (n < m) should work."""
        A = backend.random_normal((8, 16))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        assert A_pinv is not None
        assert backend.shape(A_pinv) == (16, 8)
        assert all_finite(A_pinv, backend)

    def test_pinv_property_A_Apinv_A(self, backend):
        """A @ A_pinv @ A should approximately equal A."""
        A = backend.random_normal((8, 6))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        # A @ A+ @ A should equal A
        reconstructed = backend.matmul(backend.matmul(A, A_pinv), A)
        backend.eval(reconstructed)

        diff = backend.mean(backend.abs(reconstructed - A))
        backend.eval(diff)

        tol = regularization_epsilon(backend, A)
        assert float(backend.to_scalar(diff)) <= tol

    def test_pinv_property_Apinv_A_Apinv(self, backend):
        """A_pinv @ A @ A_pinv should approximately equal A_pinv."""
        A = backend.random_normal((6, 8))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        # A+ @ A @ A+ should equal A+
        reconstructed = backend.matmul(backend.matmul(A_pinv, A), A_pinv)
        backend.eval(reconstructed)

        diff = backend.mean(backend.abs(reconstructed - A_pinv))
        backend.eval(diff)

        tol = regularization_epsilon(backend, A_pinv)
        assert float(backend.to_scalar(diff)) <= tol


class TestSetSubmatrix:
    """Tests for _set_submatrix()."""

    def test_basic_insert(self, backend):
        """Basic submatrix insertion should work."""
        target = backend.zeros((4, 4))
        source = backend.ones((2, 2))
        backend.eval(target, source)

        result = _set_submatrix(backend, target, source, 1, 1)

        # Check that source was inserted at (1,1)
        center = result[1:3, 1:3]
        backend.eval(center)
        mean_center = backend.mean(center)
        backend.eval(mean_center)

        assert float(backend.to_scalar(mean_center)) == 1.0

    def test_corner_insert(self, backend):
        """Insert at top-left corner (0,0)."""
        target = backend.zeros((4, 4))
        source = backend.ones((2, 2))
        backend.eval(target, source)

        result = _set_submatrix(backend, target, source, 0, 0)

        # Check top-left is ones
        corner = result[0:2, 0:2]
        backend.eval(corner)
        mean_corner = backend.mean(corner)
        backend.eval(mean_corner)

        assert float(backend.to_scalar(mean_corner)) == 1.0

    def test_bottom_right_insert(self, backend):
        """Insert at bottom-right corner."""
        target = backend.zeros((4, 4))
        source = backend.ones((2, 2))
        backend.eval(target, source)

        result = _set_submatrix(backend, target, source, 2, 2)

        # Check bottom-right is ones
        corner = result[2:4, 2:4]
        backend.eval(corner)
        mean_corner = backend.mean(corner)
        backend.eval(mean_corner)

        assert float(backend.to_scalar(mean_corner)) == 1.0

    def test_empty_source_returns_target(self, backend):
        """Empty source should return target unchanged."""
        target = backend.ones((4, 4))
        source = backend.zeros((0, 2))  # Empty rows
        backend.eval(target, source)

        result = _set_submatrix(backend, target, source, 0, 0)

        # Should be unchanged
        diff = backend.mean(backend.abs(result - target))
        backend.eval(diff)

        assert float(backend.to_scalar(diff)) == 0.0

    def test_preserves_other_values(self, backend):
        """Insertion should preserve values outside the submatrix region."""
        target = backend.random_normal((4, 4))
        source = backend.zeros((2, 2))
        backend.eval(target, source)

        result = _set_submatrix(backend, target, source, 1, 1)

        # Top row should be unchanged
        diff_top = backend.mean(backend.abs(result[0, :] - target[0, :]))
        backend.eval(diff_top)

        tol = regularization_epsilon(backend, result)
        assert float(backend.to_scalar(diff_top)) <= tol


class TestComputeDimensionProjection:
    """Tests for _compute_dimension_projection()."""

    def test_same_dimension_is_identity(self, backend):
        """Same source/target dim should produce identity."""
        proj = _compute_dimension_projection(backend, 8, 8)

        expected = backend.eye(8)
        diff = backend.mean(backend.abs(proj - expected))
        backend.eval(diff)

        tol = regularization_epsilon(backend, proj)
        assert float(backend.to_scalar(diff)) <= tol

    def test_downproject(self, backend):
        """Larger src_dim should produce [I; 0] projection."""
        proj = _compute_dimension_projection(backend, 8, 4)

        assert backend.shape(proj) == (8, 4)

        # Top part should be identity
        top = proj[:4, :]
        expected_top = backend.eye(4)
        diff_top = backend.mean(backend.abs(top - expected_top))
        backend.eval(diff_top)

        # Bottom part should be zeros
        bottom = proj[4:, :]
        sum_bottom = backend.sum(backend.abs(bottom))
        backend.eval(sum_bottom)

        tol = regularization_epsilon(backend, proj)
        assert float(backend.to_scalar(diff_top)) <= tol
        assert float(backend.to_scalar(sum_bottom)) <= tol

    def test_upproject(self, backend):
        """Smaller src_dim should produce [I | 0] projection."""
        proj = _compute_dimension_projection(backend, 4, 8)

        assert backend.shape(proj) == (4, 8)

        # Left part should be identity
        left = proj[:, :4]
        expected_left = backend.eye(4)
        diff_left = backend.mean(backend.abs(left - expected_left))
        backend.eval(diff_left)

        # Right part should be zeros
        right = proj[:, 4:]
        sum_right = backend.sum(backend.abs(right))
        backend.eval(sum_right)

        tol = regularization_epsilon(backend, proj)
        assert float(backend.to_scalar(diff_left)) <= tol
        assert float(backend.to_scalar(sum_right)) <= tol


class TestTransplantHelpersMathematicalProperties:
    """Hypothesis-based tests for mathematical invariants."""

    @given(
        n=st.integers(min_value=4, max_value=16),
        m=st.integers(min_value=4, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_pinv_shape_transposed(self, n, m):
        """Pseudo-inverse shape should be (m, n) for (n, m) input."""
        backend = get_default_backend()
        A = backend.random_normal((n, m))
        backend.eval(A)

        A_pinv = _geodesic_pinv(backend, A)

        assert backend.shape(A_pinv) == (m, n)

    @given(
        src=st.integers(min_value=2, max_value=16),
        tgt=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_projection_shape_correct(self, src, tgt):
        """Projection shape should be (src, tgt)."""
        backend = get_default_backend()
        proj = _compute_dimension_projection(backend, src, tgt)

        assert backend.shape(proj) == (src, tgt)

    @given(
        src=st.integers(min_value=2, max_value=16),
        tgt=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=10, deadline=None)
    def test_projection_is_finite(self, src, tgt):
        """Projection should contain only finite values."""
        backend = get_default_backend()
        proj = _compute_dimension_projection(backend, src, tgt)

        assert all_finite(proj, backend)

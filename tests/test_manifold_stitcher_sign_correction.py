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

"""Tests for SVD sign correction in manifold_stitcher.py.

These tests verify that the _ensure_proper_rotation helper function correctly
converts reflections (det = -1) to proper rotations (det = +1) while preserving
orthogonality.

Mathematical background:
- SVD-based Procrustes gives R = U @ V^T
- R is orthogonal: R @ R^T = I
- det(R) = ±1: +1 means proper rotation, -1 means reflection
- To ensure det = +1, flip the sign of last column of U if det < 0
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.manifold_stitcher import _ensure_proper_rotation
from modelcypher.core.domain.geometry.numerical_stability import division_epsilon


def _det_scalar(backend, matrix) -> float:
    det_value = backend.det(matrix)
    backend.eval(det_value)
    return float(backend.to_scalar(det_value))


def _max_abs_diff(backend, left, right) -> float:
    diff = backend.abs(left - right)
    max_diff = backend.max(diff)
    backend.eval(max_diff)
    return float(backend.to_scalar(max_diff))


class TestEnsureProperRotation:
    """Tests for _ensure_proper_rotation helper function."""

    def test_identity_rotation_unchanged(self) -> None:
        """Identity rotation (det=+1) should be unchanged."""
        backend = get_default_backend()
        # U = I, Vt = I -> omega = I, det = 1
        n = 4
        u = backend.eye(n, dtype="float32")
        vt = backend.eye(n, dtype="float32")
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)
        eps = division_epsilon(backend, result)

        # Should be identity
        eye = backend.eye(n, dtype="float32")
        assert _max_abs_diff(backend, result, eye) <= eps
        # det should be +1
        det_scalar = _det_scalar(backend, result)
        assert det_scalar >= 1.0 - eps

    def test_reflection_fixed_to_rotation(self) -> None:
        """Reflection matrix (det=-1) should be fixed to proper rotation."""
        backend = get_default_backend()
        n = 4
        # Create a reflection by flipping first element on diagonal
        # Start with identity and flip sign of first diagonal element
        u = backend.diag(backend.array([-1.0, 1.0, 1.0, 1.0], dtype="float32"))
        vt = backend.eye(n, dtype="float32")

        omega = backend.matmul(u, vt)
        omega_det = _det_scalar(backend, omega)
        eps = division_epsilon(backend, omega)
        assert omega_det <= -eps, "Setup: omega should be a reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)

        # det should now be +1
        det_scalar = _det_scalar(backend, result)
        assert det_scalar >= 1.0 - eps, f"Expected det ≈ +1, got {det_scalar}"

        # Should still be orthogonal
        product = backend.matmul(result, backend.transpose(result))
        eye_arr = backend.eye(n, dtype="float32")
        assert _max_abs_diff(backend, product, eye_arr) <= eps

    def test_random_svd_reflection_fixed(self) -> None:
        """Random SVD that produces reflection should be fixed."""
        backend = get_default_backend()
        backend.random_seed(42)
        n = 8

        # Create random matrices and compute SVD
        # This can produce either rotation or reflection
        A = backend.random_normal((n, n))
        u, s, vt = backend.svd(A)
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)
        eps = division_epsilon(backend, result)

        det_scalar = _det_scalar(backend, result)

        # det should be +1 (or very close)
        assert det_scalar >= 1.0 - eps, f"Expected det ≈ +1, got {det_scalar}"

        # Should still be orthogonal
        product = backend.matmul(result, backend.transpose(result))
        eye_arr = backend.eye(n, dtype="float32")
        assert _max_abs_diff(backend, product, eye_arr) <= eps

    def test_orthogonality_preserved(self) -> None:
        """Sign correction should preserve orthogonality of the matrix."""
        backend = get_default_backend()
        n = 6
        # Create reflection by flipping first diagonal element
        diag_vals = backend.array([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype="float32")
        u3 = backend.diag(diag_vals)
        vt3 = backend.eye(n, dtype="float32")
        omega3 = backend.matmul(u3, vt3)

        omega3_det = _det_scalar(backend, omega3)
        eps = division_epsilon(backend, omega3)
        assert omega3_det <= -eps, "omega3 should be reflection"

        result = _ensure_proper_rotation(u3, vt3, omega3, backend)

        # Check orthogonality: R @ R^T = I
        product1 = backend.matmul(result, backend.transpose(result))
        eye_arr = backend.eye(n, dtype="float32")
        assert _max_abs_diff(backend, product1, eye_arr) <= eps
        # Check R^T @ R = I
        product2 = backend.matmul(backend.transpose(result), result)
        assert _max_abs_diff(backend, product2, eye_arr) <= eps

    def test_small_matrix(self) -> None:
        """Test with 2x2 matrix (minimum size for rotation)."""
        backend = get_default_backend()
        # 2D reflection matrix
        u = backend.array([[1, 0], [0, -1]], dtype="float32")
        vt = backend.eye(2, dtype="float32")
        omega = backend.matmul(u, vt)

        omega_det = _det_scalar(backend, omega)
        eps = division_epsilon(backend, omega)
        assert omega_det <= -eps

        result = _ensure_proper_rotation(u, vt, omega, backend)

        result_det = _det_scalar(backend, result)
        assert result_det >= 1.0 - eps
        product = backend.matmul(result, backend.transpose(result))
        eye_arr = backend.eye(2, dtype="float32")
        assert _max_abs_diff(backend, product, eye_arr) <= eps

    def test_large_matrix(self) -> None:
        """Test with large matrix (typical hidden dimension)."""
        backend = get_default_backend()
        backend.random_seed(123)
        n = 128  # Smaller than 4096 but still tests scaling

        # Create orthogonal reflection
        random_mat = backend.random_normal((n, n))
        q, _ = backend.qr(random_mat)

        # Ensure it's a reflection by multiplying first column by -1 if needed
        q_det = backend.det(q)
        backend.eval(q_det)
        if float(backend.to_scalar(q_det)) > 0:
            # Multiply first column by -1 using backend operations
            sign_vec = backend.ones((n,), dtype="float32")
            sign_vec = backend.concatenate([backend.array([-1.0], dtype="float32"), sign_vec[1:]])
            sign_mat = backend.diag(sign_vec)
            q = backend.matmul(q, sign_mat)

        # Decompose as if from SVD
        u = q
        vt = backend.eye(n, dtype="float32")
        omega = backend.matmul(u, vt)

        omega_det = _det_scalar(backend, omega)
        eps = division_epsilon(backend, omega)
        assert omega_det <= -eps, "Setup: should be reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)

        result_det = _det_scalar(backend, result)
        assert result_det >= 1.0 - eps
        product = backend.matmul(result, backend.transpose(result))
        eye_arr = backend.eye(n, dtype="float32")
        assert _max_abs_diff(backend, product, eye_arr) <= eps

    def test_already_proper_rotation_unchanged(self) -> None:
        """Proper rotation (det=+1) should pass through unchanged."""
        backend = get_default_backend()
        backend.random_seed(456)
        n = 8

        # Create proper rotation via QR
        random_mat = backend.random_normal((n, n))
        q, _ = backend.qr(random_mat)

        # Ensure it's a rotation not reflection by multiplying first column by -1 if needed
        q_det = backend.det(q)
        backend.eval(q_det)
        if float(backend.to_scalar(q_det)) < 0:
            # Multiply first column by -1 using backend operations
            sign_vec = backend.ones((n,), dtype="float32")
            sign_vec = backend.concatenate([backend.array([-1.0], dtype="float32"), sign_vec[1:]])
            sign_mat = backend.diag(sign_vec)
            q = backend.matmul(q, sign_mat)

        eps = division_epsilon(backend, q)
        final_det = _det_scalar(backend, q)
        assert final_det >= 1.0 - eps

        # Use as omega with identity U and Vt
        u = q
        vt = backend.eye(n, dtype="float32")
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)

        # Should be essentially unchanged
        assert _max_abs_diff(backend, result, omega) <= eps

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


class TestEnsureProperRotation:
    """Tests for _ensure_proper_rotation helper function."""

    def test_identity_rotation_unchanged(self) -> None:
        """Identity rotation (det=+1) should be unchanged."""
        backend = get_default_backend()
        # U = I, Vt = I -> omega = I, det = 1
        n = 4
        u = backend.eye(n, dtype=backend.float32)
        vt = backend.eye(n, dtype=backend.float32)
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # Should be identity
        eye_np = backend.to_numpy(backend.eye(n))
        assert backend.allclose(backend.array(result_np), backend.array(eye_np), atol=1e-6)
        # det should be +1
        det = backend.det(backend.array(result_np))
        assert float(backend.to_numpy(det)) > 0

    def test_reflection_fixed_to_rotation(self) -> None:
        """Reflection matrix (det=-1) should be fixed to proper rotation."""
        backend = get_default_backend()
        n = 4
        # Create a reflection by flipping one axis
        u = backend.eye(n, dtype=backend.float32)
        vt = backend.eye(n, dtype=backend.float32)
        u_np = backend.to_numpy(u)
        u_np[0, 0] = -1  # Makes det(U) = -1, so det(omega) = -1
        u = backend.array(u_np)

        omega = backend.matmul(u, vt)
        omega_det = backend.det(omega)
        assert float(backend.to_numpy(omega_det)) < 0, "Setup: omega should be a reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # det should now be +1
        det = backend.det(backend.array(result_np))
        det_scalar = float(backend.to_numpy(det))
        assert det_scalar > 0, f"Expected det > 0, got {det_scalar}"

        # Should still be orthogonal
        result_arr = backend.array(result_np)
        product = backend.matmul(result_arr, backend.transpose(result_arr))
        eye_arr = backend.eye(n)
        assert backend.allclose(product, eye_arr, atol=1e-6)

    def test_random_svd_reflection_fixed(self) -> None:
        """Random SVD that produces reflection should be fixed."""
        backend = get_default_backend()
        backend.random_seed(42)
        n = 8

        # Create random matrices and compute SVD
        # This can produce either rotation or reflection
        A = backend.random_randn((n, n))
        u, s, vt = backend.svd(A, full_matrices=True)
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        det_after = backend.det(backend.array(result_np))
        det_scalar = float(backend.to_numpy(det_after))

        # det should be +1 (or very close)
        assert det_scalar > 0.99, f"Expected det ≈ +1, got {det_scalar}"

        # Should still be orthogonal
        result_arr = backend.array(result_np)
        product = backend.matmul(result_arr, backend.transpose(result_arr))
        eye_arr = backend.eye(n)
        assert backend.allclose(product, eye_arr, atol=1e-5)

    def test_orthogonality_preserved(self) -> None:
        """Sign correction should preserve orthogonality of the matrix."""
        backend = get_default_backend()
        n = 6
        # Simpler: just flip one column of U only
        u3 = backend.eye(n, dtype=backend.float32)
        u3_np = backend.to_numpy(u3)
        u3_np[:, 0] *= -1
        u3 = backend.array(u3_np)
        vt3 = backend.eye(n, dtype=backend.float32)
        omega3 = backend.matmul(u3, vt3)

        omega3_det = backend.det(omega3)
        assert float(backend.to_numpy(omega3_det)) < 0, "omega3 should be reflection"

        result = _ensure_proper_rotation(u3, vt3, omega3, backend)
        result_np = backend.to_numpy(result)

        # Check orthogonality: R @ R^T = I
        result_arr = backend.array(result_np)
        product1 = backend.matmul(result_arr, backend.transpose(result_arr))
        eye_arr = backend.eye(n)
        assert backend.allclose(product1, eye_arr, atol=1e-6)
        # Check R^T @ R = I
        product2 = backend.matmul(backend.transpose(result_arr), result_arr)
        assert backend.allclose(product2, eye_arr, atol=1e-6)

    def test_small_matrix(self) -> None:
        """Test with 2x2 matrix (minimum size for rotation)."""
        backend = get_default_backend()
        # 2D reflection matrix
        u = backend.array([[1, 0], [0, -1]], dtype=backend.float32)
        vt = backend.eye(2, dtype=backend.float32)
        omega = backend.matmul(u, vt)

        omega_det = backend.det(omega)
        assert float(backend.to_numpy(omega_det)) < 0

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        result_det = backend.det(backend.array(result_np))
        assert float(backend.to_numpy(result_det)) > 0
        result_arr = backend.array(result_np)
        product = backend.matmul(result_arr, backend.transpose(result_arr))
        eye_arr = backend.eye(2)
        assert backend.allclose(product, eye_arr, atol=1e-6)

    def test_large_matrix(self) -> None:
        """Test with large matrix (typical hidden dimension)."""
        backend = get_default_backend()
        backend.random_seed(123)
        n = 128  # Smaller than 4096 but still tests scaling

        # Create orthogonal reflection
        random_mat = backend.random_randn((n, n))
        q, _ = backend.qr(random_mat)

        # Ensure it's a reflection
        q_det = backend.det(q)
        if float(backend.to_numpy(q_det)) > 0:
            q_np = backend.to_numpy(q)
            q_np[:, 0] *= -1
            q = backend.array(q_np)

        # Decompose as if from SVD
        u = q
        vt = backend.eye(n, dtype=backend.float32)
        omega = backend.matmul(u, vt)

        omega_det = backend.det(omega)
        assert float(backend.to_numpy(omega_det)) < 0, "Setup: should be reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        result_det = backend.det(backend.array(result_np))
        assert float(backend.to_numpy(result_det)) > 0.99
        result_arr = backend.array(result_np)
        product = backend.matmul(result_arr, backend.transpose(result_arr))
        eye_arr = backend.eye(n)
        assert backend.allclose(product, eye_arr, atol=1e-4)

    def test_already_proper_rotation_unchanged(self) -> None:
        """Proper rotation (det=+1) should pass through unchanged."""
        backend = get_default_backend()
        backend.random_seed(456)
        n = 8

        # Create proper rotation via QR
        random_mat = backend.random_randn((n, n))
        q, _ = backend.qr(random_mat)

        # Ensure it's a rotation not reflection
        q_det = backend.det(q)
        if float(backend.to_numpy(q_det)) < 0:
            q_np = backend.to_numpy(q)
            q_np[:, 0] *= -1
            q = backend.array(q_np)

        final_det = backend.det(q)
        assert float(backend.to_numpy(final_det)) > 0.99

        # Use as omega with identity U and Vt
        u = q
        vt = backend.eye(n, dtype=backend.float32)
        omega = backend.matmul(u, vt)

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # Should be essentially unchanged
        omega_np = backend.to_numpy(omega)
        assert backend.allclose(backend.array(result_np), backend.array(omega_np), atol=1e-5)

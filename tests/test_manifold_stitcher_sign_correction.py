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

import numpy as np
import pytest

from modelcypher.core.domain.geometry.manifold_stitcher import _ensure_proper_rotation


class MockBackend:
    """Minimal backend for testing sign correction."""

    def to_numpy(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def array(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b


class TestEnsureProperRotation:
    """Tests for _ensure_proper_rotation helper function."""

    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend()

    def test_identity_rotation_unchanged(self, backend: MockBackend) -> None:
        """Identity rotation (det=+1) should be unchanged."""
        # U = I, Vt = I -> omega = I, det = 1
        n = 4
        u = np.eye(n, dtype=np.float32)
        vt = np.eye(n, dtype=np.float32)
        omega = u @ vt

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # Should be identity
        np.testing.assert_allclose(result_np, np.eye(n), atol=1e-6)
        # det should be +1
        assert np.linalg.det(result_np) > 0

    def test_reflection_fixed_to_rotation(self, backend: MockBackend) -> None:
        """Reflection matrix (det=-1) should be fixed to proper rotation."""
        n = 4
        # Create a reflection by flipping one axis
        u = np.eye(n, dtype=np.float32)
        vt = np.eye(n, dtype=np.float32)
        u[0, 0] = -1  # Makes det(U) = -1, so det(omega) = -1

        omega = u @ vt
        assert np.linalg.det(omega) < 0, "Setup: omega should be a reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # det should now be +1
        det = np.linalg.det(result_np)
        assert det > 0, f"Expected det > 0, got {det}"

        # Should still be orthogonal
        np.testing.assert_allclose(
            result_np @ result_np.T, np.eye(n), atol=1e-6,
            err_msg="Result should be orthogonal"
        )

    def test_random_svd_reflection_fixed(self, backend: MockBackend) -> None:
        """Random SVD that produces reflection should be fixed."""
        np.random.seed(42)
        n = 8

        # Create random matrices and compute SVD
        # This can produce either rotation or reflection
        A = np.random.randn(n, n).astype(np.float32)
        u, s, vt = np.linalg.svd(A)
        omega = u @ vt

        det_before = np.linalg.det(omega)

        result = _ensure_proper_rotation(
            u.astype(np.float32),
            vt.astype(np.float32),
            omega.astype(np.float32),
            backend
        )
        result_np = backend.to_numpy(result)

        det_after = np.linalg.det(result_np)

        # det should be +1 (or very close)
        assert det_after > 0.99, f"Expected det ≈ +1, got {det_after}"

        # Should still be orthogonal
        np.testing.assert_allclose(
            result_np @ result_np.T, np.eye(n), atol=1e-5,
            err_msg="Result should be orthogonal"
        )

    def test_orthogonality_preserved(self, backend: MockBackend) -> None:
        """Sign correction should preserve orthogonality of the matrix."""
        n = 6
        # Create known orthogonal reflection
        u = np.eye(n, dtype=np.float32)
        u[:, -1] *= -1  # Flip last column to make det(U) = -1
        vt = np.eye(n, dtype=np.float32)
        vt[-1, :] *= -1  # Flip last row

        omega = u @ vt
        # This gives det(omega) = (-1) * (-1) = +1 actually
        # Let's make a true reflection
        u2 = np.eye(n, dtype=np.float32)
        u2[0, 0] = -1
        omega2 = u2 @ vt  # det = -1 * -1 = 1? No wait...

        # Simpler: just flip one column of U only
        u3 = np.eye(n, dtype=np.float32)
        u3[:, 0] *= -1
        vt3 = np.eye(n, dtype=np.float32)
        omega3 = u3 @ vt3

        assert np.linalg.det(omega3) < 0, "omega3 should be reflection"

        result = _ensure_proper_rotation(u3, vt3, omega3, backend)
        result_np = backend.to_numpy(result)

        # Check orthogonality: R @ R^T = I
        np.testing.assert_allclose(
            result_np @ result_np.T, np.eye(n), atol=1e-6
        )
        # Check R^T @ R = I
        np.testing.assert_allclose(
            result_np.T @ result_np, np.eye(n), atol=1e-6
        )

    def test_small_matrix(self, backend: MockBackend) -> None:
        """Test with 2x2 matrix (minimum size for rotation)."""
        # 2D reflection matrix
        u = np.array([[1, 0], [0, -1]], dtype=np.float32)
        vt = np.eye(2, dtype=np.float32)
        omega = u @ vt

        assert np.linalg.det(omega) < 0

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        assert np.linalg.det(result_np) > 0
        np.testing.assert_allclose(result_np @ result_np.T, np.eye(2), atol=1e-6)

    def test_large_matrix(self, backend: MockBackend) -> None:
        """Test with large matrix (typical hidden dimension)."""
        np.random.seed(123)
        n = 128  # Smaller than 4096 but still tests scaling

        # Create orthogonal reflection
        q, _ = np.linalg.qr(np.random.randn(n, n))
        q = q.astype(np.float32)

        # Ensure it's a reflection
        if np.linalg.det(q) > 0:
            q[:, 0] *= -1

        # Decompose as if from SVD
        u = q
        vt = np.eye(n, dtype=np.float32)
        omega = u @ vt

        assert np.linalg.det(omega) < 0, "Setup: should be reflection"

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        assert np.linalg.det(result_np) > 0.99
        np.testing.assert_allclose(
            result_np @ result_np.T, np.eye(n), atol=1e-4
        )

    def test_already_proper_rotation_unchanged(self, backend: MockBackend) -> None:
        """Proper rotation (det=+1) should pass through unchanged."""
        np.random.seed(456)
        n = 8

        # Create proper rotation via QR
        q, _ = np.linalg.qr(np.random.randn(n, n))
        q = q.astype(np.float32)

        # Ensure it's a rotation not reflection
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1

        assert np.linalg.det(q) > 0.99

        # Use as omega with identity U and Vt
        u = q
        vt = np.eye(n, dtype=np.float32)
        omega = u @ vt

        result = _ensure_proper_rotation(u, vt, omega, backend)
        result_np = backend.to_numpy(result)

        # Should be essentially unchanged
        np.testing.assert_allclose(result_np, omega, atol=1e-5)

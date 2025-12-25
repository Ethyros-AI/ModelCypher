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

"""
Multi-backend CKA tests.

Verifies that CKA computations produce consistent results across:
- NumPy (CPU reference)
- MLX (Metal GPU on Apple Silicon)
- JAX (GPU/TPU)

Mathematical properties tested:
- Self-similarity: CKA(X, X) = 1.0
- Symmetry: CKA(X, Y) = CKA(Y, X)
- Scale invariance: CKA(αX, Y) = CKA(X, Y)
- Rotation invariance: CKA(XR, Y) = CKA(X, Y) for orthogonal R
- Range: 0 ≤ CKA(X, Y) ≤ 1
- Cross-backend consistency: Same inputs → Same outputs (within tolerance)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelcypher.core.domain.geometry.cka import (
    compute_cka,
    compute_cka_backend,
)
from modelcypher.ports.backend import Backend

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_activations() -> tuple[np.ndarray, np.ndarray]:
    """Generate random activation matrices for testing."""
    np.random.seed(42)
    n_samples, dim_x, dim_y = 50, 128, 64
    x = np.random.randn(n_samples, dim_x).astype(np.float32)
    y = np.random.randn(n_samples, dim_y).astype(np.float32)
    return x, y


@pytest.fixture
def correlated_activations() -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated activation matrices (high CKA expected)."""
    np.random.seed(42)
    n_samples, dim = 50, 64
    base = np.random.randn(n_samples, dim).astype(np.float32)
    # Y is a noisy version of X
    noise = np.random.randn(n_samples, dim).astype(np.float32) * 0.1
    y = base + noise
    return base, y


# =============================================================================
# NumPy Reference Tests (compute_cka)
# =============================================================================


class TestCKANumpyReference:
    """Tests for the NumPy reference implementation."""

    def test_self_similarity_is_one(self, random_activations):
        """CKA(X, X) = 1.0 exactly."""
        x, _ = random_activations
        result = compute_cka(x, x)
        assert result.is_valid
        assert result.cka == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self, random_activations):
        """CKA(X, Y) = CKA(Y, X)."""
        x, y = random_activations
        result_xy = compute_cka(x, y)
        result_yx = compute_cka(y, x)
        assert result_xy.cka == pytest.approx(result_yx.cka, abs=1e-6)

    def test_range_bounds(self, random_activations):
        """CKA must be in [0, 1]."""
        x, y = random_activations
        result = compute_cka(x, y)
        assert 0.0 <= result.cka <= 1.0

    def test_scale_invariance(self, random_activations):
        """CKA(αX, Y) = CKA(X, Y) for any scalar α > 0."""
        x, y = random_activations
        result_base = compute_cka(x, y)

        # Scale X by various factors
        for scale in [0.1, 2.0, 100.0]:
            result_scaled = compute_cka(x * scale, y)
            assert result_scaled.cka == pytest.approx(result_base.cka, abs=1e-5), (
                f"Scale invariance failed for α={scale}"
            )

    def test_rotation_invariance(self, random_activations):
        """CKA(XR, Y) = CKA(X, Y) for orthogonal R."""
        x, y = random_activations
        result_base = compute_cka(x, y)

        # Create random orthogonal matrix via QR decomposition
        np.random.seed(123)
        random_matrix = np.random.randn(x.shape[1], x.shape[1]).astype(np.float32)
        q, _ = np.linalg.qr(random_matrix)

        # Rotate X
        x_rotated = x @ q
        result_rotated = compute_cka(x_rotated, y)

        assert result_rotated.cka == pytest.approx(result_base.cka, abs=1e-5), (
            "Rotation invariance failed"
        )

    def test_correlated_higher_than_random(self, correlated_activations, random_activations):
        """Correlated activations should have higher CKA than random."""
        x_corr, y_corr = correlated_activations
        x_rand, y_rand = random_activations

        cka_corr = compute_cka(x_corr, y_corr).cka
        cka_rand = compute_cka(x_rand, y_rand).cka

        assert cka_corr > cka_rand, (
            f"Correlated CKA ({cka_corr:.4f}) should be > random CKA ({cka_rand:.4f})"
        )

    def test_minimum_samples_required(self):
        """CKA requires at least 2 samples."""
        x = np.random.randn(1, 10).astype(np.float32)
        y = np.random.randn(1, 10).astype(np.float32)
        result = compute_cka(x, y)
        assert result.cka == 0.0
        assert result.sample_count == 1


# =============================================================================
# Multi-Backend Tests (compute_cka_backend)
# =============================================================================


class TestCKAMultiBackend:
    """Tests that run on all available backends (numpy, mlx, jax)."""

    def test_self_similarity_is_one(self, any_backend: Backend):
        """CKA(X, X) = 1.0 on all backends."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)

        # Convert to backend array
        x = any_backend.array(x_np)

        cka = compute_cka_backend(x, x, any_backend)
        assert cka == pytest.approx(1.0, abs=1e-5), (
            f"Self-similarity failed on {type(any_backend).__name__}"
        )

    def test_symmetry(self, any_backend: Backend):
        """CKA(X, Y) = CKA(Y, X) on all backends."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)
        y_np = np.random.randn(50, 32).astype(np.float32)

        x = any_backend.array(x_np)
        y = any_backend.array(y_np)

        cka_xy = compute_cka_backend(x, y, any_backend)
        cka_yx = compute_cka_backend(y, x, any_backend)

        assert cka_xy == pytest.approx(cka_yx, abs=1e-5), (
            f"Symmetry failed on {type(any_backend).__name__}"
        )

    def test_range_bounds(self, any_backend: Backend):
        """CKA must be in [0, 1] on all backends."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)
        y_np = np.random.randn(50, 32).astype(np.float32)

        x = any_backend.array(x_np)
        y = any_backend.array(y_np)

        cka = compute_cka_backend(x, y, any_backend)
        assert 0.0 <= cka <= 1.0, f"Range violation on {type(any_backend).__name__}: CKA={cka}"

    def test_scale_invariance(self, any_backend: Backend):
        """CKA(αX, Y) = CKA(X, Y) on all backends."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)
        y_np = np.random.randn(50, 32).astype(np.float32)

        x = any_backend.array(x_np)
        y = any_backend.array(y_np)

        cka_base = compute_cka_backend(x, y, any_backend)

        # Scale X by 10
        x_scaled = any_backend.array(x_np * 10.0)
        cka_scaled = compute_cka_backend(x_scaled, y, any_backend)

        assert cka_scaled == pytest.approx(cka_base, abs=1e-4), (
            f"Scale invariance failed on {type(any_backend).__name__}"
        )

    def test_orthogonal_representations_low_cka(self, any_backend: Backend):
        """Orthogonal representations should have CKA ≈ 0."""
        n_samples = 50

        # Create orthogonal representations
        # X = [1, 0, 0, ...] pattern, Y = [0, 1, 0, ...] pattern
        x_np = np.zeros((n_samples, 20), dtype=np.float32)
        y_np = np.zeros((n_samples, 20), dtype=np.float32)
        x_np[:, :10] = np.random.randn(n_samples, 10).astype(np.float32)
        y_np[:, 10:] = np.random.randn(n_samples, 10).astype(np.float32)

        x = any_backend.array(x_np)
        y = any_backend.array(y_np)

        cka = compute_cka_backend(x, y, any_backend)

        # Orthogonal subspaces should have low CKA
        assert cka < 0.3, f"Orthogonal representations have unexpectedly high CKA={cka:.4f}"


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestCKACrossBackendConsistency:
    """Tests that verify consistency across different backends."""

    def test_numpy_vs_backend_consistency(self, any_backend: Backend):
        """Backend result should match NumPy reference (within tolerance)."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)
        y_np = np.random.randn(50, 32).astype(np.float32)

        # NumPy reference
        result_np = compute_cka(x_np, y_np)
        cka_numpy = result_np.cka

        # Backend implementation
        x = any_backend.array(x_np)
        y = any_backend.array(y_np)
        cka_backend = compute_cka_backend(x, y, any_backend)

        # Allow slightly larger tolerance due to different numerical paths
        # NumPy uses centered HSIC, backend uses uncentered (mathematically equivalent for CKA)
        assert abs(cka_numpy - cka_backend) < 0.05, (
            f"NumPy ({cka_numpy:.6f}) vs {type(any_backend).__name__} ({cka_backend:.6f}) "
            f"differ by {abs(cka_numpy - cka_backend):.6f}"
        )

    def test_deterministic_across_calls(self, any_backend: Backend):
        """Same inputs should produce identical outputs."""
        np.random.seed(42)
        x_np = np.random.randn(50, 64).astype(np.float32)
        y_np = np.random.randn(50, 32).astype(np.float32)

        x = any_backend.array(x_np)
        y = any_backend.array(y_np)

        cka1 = compute_cka_backend(x, y, any_backend)
        cka2 = compute_cka_backend(x, y, any_backend)
        cka3 = compute_cka_backend(x, y, any_backend)

        assert cka1 == cka2 == cka3, f"Non-deterministic results: {cka1}, {cka2}, {cka3}"


# =============================================================================
# Accelerator-Only Tests
# =============================================================================


class TestCKAAccelerator:
    """Tests that specifically require GPU/accelerator backends."""

    def test_large_matrix_performance(self, accelerated_backend: Backend):
        """Large matrices should complete without memory issues."""
        np.random.seed(42)
        # Large activation matrices (realistic LLM hidden states)
        x_np = np.random.randn(512, 4096).astype(np.float32)
        y_np = np.random.randn(512, 4096).astype(np.float32)

        x = accelerated_backend.array(x_np)
        y = accelerated_backend.array(y_np)

        cka = compute_cka_backend(x, y, accelerated_backend)

        assert 0.0 <= cka <= 1.0
        assert math.isfinite(cka)

    def test_numerical_stability_extreme_values(self, accelerated_backend: Backend):
        """CKA should handle extreme activation magnitudes."""
        np.random.seed(42)

        # Very large activations
        x_large = np.random.randn(50, 64).astype(np.float32) * 1e6
        y_large = np.random.randn(50, 32).astype(np.float32) * 1e6

        x = accelerated_backend.array(x_large)
        y = accelerated_backend.array(y_large)

        cka = compute_cka_backend(x, y, accelerated_backend)

        assert math.isfinite(cka), f"CKA is not finite: {cka}"
        assert 0.0 <= cka <= 1.0

    def test_batch_consistency(self, accelerated_backend: Backend):
        """Multiple CKA computations should be consistent."""
        np.random.seed(42)

        results = []
        for _ in range(5):
            x_np = np.random.randn(50, 64).astype(np.float32)
            y_np = np.random.randn(50, 64).astype(np.float32)

            x = accelerated_backend.array(x_np)
            y = accelerated_backend.array(y_np)

            # Self-similarity should always be 1.0
            cka_self = compute_cka_backend(x, x, accelerated_backend)
            results.append(cka_self)

        # All self-similarities should be ~1.0
        for i, cka in enumerate(results):
            assert cka == pytest.approx(1.0, abs=1e-5), f"Batch {i} self-similarity = {cka}"

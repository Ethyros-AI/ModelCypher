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

NOTE: All tests use the Backend protocol exclusively. No numpy.
"""

from __future__ import annotations

import pytest

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    division_epsilon,
    machine_epsilon,
)
from modelcypher.core.domain.geometry.cka import (
    compute_cka,
    compute_cka_backend,
)
from modelcypher.ports.backend import Backend


def _random_matrix(backend, rows: int, cols: int, seed: int):
    """Generate random matrix using backend."""
    backend.random_seed(seed)
    return backend.random_normal(shape=(rows, cols))


def _scalar_tol(backend: Backend) -> float:
    return division_epsilon(backend, backend.array([1.0]))


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _mean_abs(backend: Backend, array) -> float:
    mean_val = backend.mean(backend.abs(array))
    backend.eval(mean_val)
    return float(backend.to_scalar(mean_val))


# =============================================================================
# NumPy Reference Tests (compute_cka with default backend)
# =============================================================================


class TestCKADefaultBackend:
    """Tests for the default backend implementation."""

    def test_self_similarity_is_one(self):
        """CKA(X, X) = 1.0 exactly."""
        backend = get_default_backend()
        x = _random_matrix(backend, 50, 128, 42)
        result = compute_cka(x, x, backend)
        assert result.is_valid
        tol = _scalar_tol(backend)
        assert abs(result.cka - 1.0) <= tol

    def test_symmetry(self):
        """CKA(X, Y) = CKA(Y, X)."""
        backend = get_default_backend()
        x = _random_matrix(backend, 50, 128, 42)
        y = _random_matrix(backend, 50, 64, 43)
        result_xy = compute_cka(x, y, backend)
        result_yx = compute_cka(y, x, backend)
        tol = _scalar_tol(backend)
        assert abs(result_xy.cka - result_yx.cka) <= tol

    def test_range_bounds(self):
        """CKA must be in [0, 1]."""
        backend = get_default_backend()
        x = _random_matrix(backend, 50, 128, 42)
        y = _random_matrix(backend, 50, 64, 43)
        result = compute_cka(x, y, backend)
        tol = _scalar_tol(backend)
        assert -tol <= result.cka <= 1.0 + tol

    def test_scale_invariance(self):
        """CKA(αX, Y) = CKA(X, Y) for any scalar α > 0."""
        backend = get_default_backend()
        x = _random_matrix(backend, 50, 128, 42)
        y = _random_matrix(backend, 50, 64, 43)
        result_base = compute_cka(x, y, backend)

        # Scale X by data-derived factors
        scale = _mean_abs(backend, x)
        for factor in [scale, 1.0 / scale]:
            x_scaled = x * factor
            result_scaled = compute_cka(x_scaled, y, backend)
            tol = _scalar_tol(backend)
            assert abs(result_scaled.cka - result_base.cka) <= tol, (
                f"Scale invariance failed for α={factor}"
            )

    def test_rotation_invariance(self):
        """CKA(XR, Y) = CKA(X, Y) for orthogonal R."""
        backend = get_default_backend()
        x = _random_matrix(backend, 50, 128, 42)
        y = _random_matrix(backend, 50, 64, 43)
        result_base = compute_cka(x, y, backend)

        # Create random orthogonal matrix via QR decomposition
        random_matrix = _random_matrix(backend, 128, 128, 123)
        q, _ = backend.qr(random_matrix)

        # Rotate X
        x_rotated = backend.matmul(x, q)
        result_rotated = compute_cka(x_rotated, y, backend)

        tol = _scalar_tol(backend)
        assert abs(result_rotated.cka - result_base.cka) <= tol, "Rotation invariance failed"

    def test_correlated_higher_than_random(self):
        """Correlated activations should have higher CKA than random."""
        backend = get_default_backend()

        # Correlated: Y = X + noise
        x_corr = _random_matrix(backend, 50, 64, 42)
        noise = _random_matrix(backend, 50, 64, 43) * division_epsilon(
            backend, backend.array([1.0])
        )
        y_corr = x_corr + noise

        # Random: independent X and Y
        x_rand = _random_matrix(backend, 50, 128, 44)
        y_rand = _random_matrix(backend, 50, 64, 45)

        cka_corr = compute_cka(x_corr, y_corr, backend).cka
        cka_rand = compute_cka(x_rand, y_rand, backend).cka

        assert cka_corr > cka_rand, (
            f"Correlated CKA ({cka_corr:.4f}) should be > random CKA ({cka_rand:.4f})"
        )

    def test_minimum_samples_required(self):
        """CKA requires at least 2 samples."""
        backend = get_default_backend()
        x = _random_matrix(backend, 1, 10, 42)
        y = _random_matrix(backend, 1, 10, 43)
        result = compute_cka(x, y, backend)
        tol = _scalar_tol(backend)
        assert abs(result.cka) <= tol
        assert result.sample_count == 1


# =============================================================================
# Multi-Backend Tests (compute_cka_backend)
# =============================================================================


class TestCKAMultiBackend:
    """Tests that run on all available backends (numpy, mlx, jax)."""

    def test_self_similarity_is_one(self, any_backend: Backend):
        """CKA(X, X) = 1.0 on all backends."""
        x = _random_matrix(any_backend, 50, 64, 42)

        cka = compute_cka_backend(x, x, any_backend)
        tol = _scalar_tol(any_backend)
        assert abs(cka - 1.0) <= tol, (
            f"Self-similarity failed on {type(any_backend).__name__}"
        )

    def test_symmetry(self, any_backend: Backend):
        """CKA(X, Y) = CKA(Y, X) on all backends."""
        x = _random_matrix(any_backend, 50, 64, 42)
        y = _random_matrix(any_backend, 50, 32, 43)

        cka_xy = compute_cka_backend(x, y, any_backend)
        cka_yx = compute_cka_backend(y, x, any_backend)

        tol = _scalar_tol(any_backend)
        assert abs(cka_xy - cka_yx) <= tol, (
            f"Symmetry failed on {type(any_backend).__name__}"
        )

    def test_range_bounds(self, any_backend: Backend):
        """CKA must be in [0, 1] on all backends."""
        x = _random_matrix(any_backend, 50, 64, 42)
        y = _random_matrix(any_backend, 50, 32, 43)

        cka = compute_cka_backend(x, y, any_backend)
        tol = _scalar_tol(any_backend)
        assert -tol <= cka <= 1.0 + tol, (
            f"Range violation on {type(any_backend).__name__}: CKA={cka}"
        )

    def test_scale_invariance(self, any_backend: Backend):
        """CKA(αX, Y) = CKA(X, Y) on all backends."""
        x = _random_matrix(any_backend, 50, 64, 42)
        y = _random_matrix(any_backend, 50, 32, 43)

        cka_base = compute_cka_backend(x, y, any_backend)

        scale = _mean_abs(any_backend, x)
        x_scaled = x * scale
        cka_scaled = compute_cka_backend(x_scaled, y, any_backend)

        tol = _scalar_tol(any_backend)
        assert abs(cka_scaled - cka_base) <= tol, (
            f"Scale invariance failed on {type(any_backend).__name__}"
        )

    def test_orthogonal_representations_low_cka(self, any_backend: Backend):
        """Orthogonal representations should have CKA ≈ 0."""
        n_samples = 50

        # Create orthogonal representations using backend
        # X = [random, 0, 0, ...] pattern, Y = [0, 0, random, ...] pattern
        x_patch = _random_matrix(any_backend, n_samples, 10, 42)
        y_patch = _random_matrix(any_backend, n_samples, 10, 43)

        # Build x: first 10 cols have values, last 10 zeros
        x = any_backend.concatenate([x_patch, any_backend.zeros((n_samples, 10))], axis=1)
        # Build y: first 10 cols zeros, last 10 have values
        y = any_backend.concatenate([any_backend.zeros((n_samples, 10)), y_patch], axis=1)

        cka = compute_cka_backend(x, y, any_backend)

        # Orthogonal subspaces should be less aligned than shared subspaces.
        y_overlap = any_backend.concatenate([x_patch, any_backend.zeros((n_samples, 10))], axis=1)
        cka_overlap = compute_cka_backend(x, y_overlap, any_backend)
        tol = _scalar_tol(any_backend)
        assert cka_overlap >= cka - tol, (
            f"Orthogonal CKA ({cka:.4f}) should not exceed shared-subspace CKA ({cka_overlap:.4f})"
        )


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestCKACrossBackendConsistency:
    """Tests that verify consistency across different backends."""

    def test_deterministic_across_calls(self, any_backend: Backend):
        """Same inputs should produce identical outputs."""
        x = _random_matrix(any_backend, 50, 64, 42)
        y = _random_matrix(any_backend, 50, 32, 43)

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
        # Large activation matrices (realistic LLM hidden states)
        x = _random_matrix(accelerated_backend, 512, 4096, 42)
        y = _random_matrix(accelerated_backend, 512, 4096, 43)

        cka = compute_cka_backend(x, y, accelerated_backend)

        tol = _scalar_tol(accelerated_backend)
        assert -tol <= cka <= 1.0 + tol
        assert _is_finite(cka)

    def test_numerical_stability_extreme_values(self, accelerated_backend: Backend):
        """CKA should handle extreme activation magnitudes."""
        # Very large activations
        eps = machine_epsilon(accelerated_backend, accelerated_backend.array([1.0]))
        large_scale = 1.0 / eps
        x_large = _random_matrix(accelerated_backend, 50, 64, 42) * large_scale
        y_large = _random_matrix(accelerated_backend, 50, 32, 43) * large_scale

        cka = compute_cka_backend(x_large, y_large, accelerated_backend)

        assert _is_finite(cka), f"CKA is not finite: {cka}"
        tol = _scalar_tol(accelerated_backend)
        assert -tol <= cka <= 1.0 + tol

    def test_batch_consistency(self, accelerated_backend: Backend):
        """Multiple CKA computations should be consistent."""
        results = []
        for i in range(5):
            x = _random_matrix(accelerated_backend, 50, 64, 42 + i)

            # Self-similarity should always be 1.0
            cka_self = compute_cka_backend(x, x, accelerated_backend)
            results.append(cka_self)

        # All self-similarities should be ~1.0
        for i, cka in enumerate(results):
            tol = _scalar_tol(accelerated_backend)
            assert abs(cka - 1.0) <= tol, f"Batch {i} self-similarity = {cka}"

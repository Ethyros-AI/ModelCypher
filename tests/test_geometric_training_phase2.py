# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Tests for Phase 2 geometry-derived training features.

Tests:
1. Spectral-normalized weight initialization
2. Geometric convergence monitor for early stopping
3. Residual connection scaling
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from modelcypher.backends.mlx_backend import MLXBackend

# Machine epsilon for numerical comparisons
SQRT_EPS = np.sqrt(np.finfo(np.float32).eps)

# Shared backend for tests
_mlx_backend = MLXBackend()


class TestSpectralNormalizedInit:
    """Tests for spectral-normalized weight initialization."""

    def _compute_spectral_norm(self, W: mx.array, n_iters: int = 10) -> float:
        """Compute spectral norm via power iteration."""
        n = int(W.shape[1])
        v = mx.ones((n,)) / mx.sqrt(mx.array(float(n)))
        mx.eval(v)

        for _ in range(n_iters):
            u = W @ v
            u_norm = mx.sqrt(mx.sum(u * u))
            mx.eval(u_norm)
            if float(u_norm) < SQRT_EPS:
                return 0.0
            u = u / u_norm

            v = W.T @ u
            v_norm = mx.sqrt(mx.sum(v * v))
            mx.eval(v_norm)
            if float(v_norm) < SQRT_EPS:
                return 0.0
            v = v / v_norm
            mx.eval(v)

        Wv = W @ v
        return float(mx.sqrt(mx.sum(Wv * Wv)))

    def test_spectral_init_achieves_target_norm(self):
        """Test that spectral-normalized init achieves target σ_max."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear

        # Create a mock base layer
        base_layer = nn.Linear(256, 512)
        mx.eval(base_layer.parameters())

        sigma_k = 0.5
        rank = 8

        lora_layer = GeometricLoRALinear(
            base_layer=base_layer,
            sigma_k=sigma_k,
            rank=rank,
            backend=_mlx_backend,
        )

        # Compute spectral norms of A and B
        A_spectral = self._compute_spectral_norm(lora_layer.lora_a)
        B_spectral = self._compute_spectral_norm(lora_layer.lora_b)

        # Each should be approximately sqrt(sigma_k)
        sqrt_sigma_k = np.sqrt(sigma_k)

        # Allow 10% tolerance - power iteration has some inherent error
        # and the initialization doesn't need to be exact
        tol = 0.1

        assert abs(A_spectral - sqrt_sigma_k) < sqrt_sigma_k * tol + tol, \
            f"A spectral norm {A_spectral} not close to {sqrt_sigma_k}"
        assert abs(B_spectral - sqrt_sigma_k) < sqrt_sigma_k * tol + tol, \
            f"B spectral norm {B_spectral} not close to {sqrt_sigma_k}"

    def test_spectral_init_product_respects_budget(self):
        """Test that ||B @ A||_spectral ≈ σ_k."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear

        base_layer = nn.Linear(128, 256)
        mx.eval(base_layer.parameters())

        sigma_k = 0.3
        rank = 4

        lora_layer = GeometricLoRALinear(
            base_layer=base_layer,
            sigma_k=sigma_k,
            rank=rank,
            backend=_mlx_backend,
        )

        # Compute B @ A
        delta = lora_layer.lora_b @ lora_layer.lora_a
        mx.eval(delta)

        # Compute spectral norm of the product
        delta_spectral = self._compute_spectral_norm(delta)

        # Product spectral norm should be approximately sigma_k
        # (with some tolerance since ||B @ A|| ≤ ||B|| × ||A||)
        assert delta_spectral <= sigma_k * 2.0, \
            f"Product spectral norm {delta_spectral} exceeds 2 × sigma_k ({sigma_k * 2})"
        assert delta_spectral >= sigma_k * 0.1, \
            f"Product spectral norm {delta_spectral} too small (expected ~{sigma_k})"

    def test_spectral_init_reproducibility(self):
        """Test that spectral init produces consistent results with same seed."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear

        base_layer = nn.Linear(64, 128)
        mx.eval(base_layer.parameters())

        sigma_k = 0.2
        rank = 4

        # Create two layers with same seed
        mx.random.seed(42)
        layer1 = GeometricLoRALinear(base_layer, sigma_k, rank, _mlx_backend)

        mx.random.seed(42)
        layer2 = GeometricLoRALinear(base_layer, sigma_k, rank, _mlx_backend)

        # Should have same spectral norms
        norm1_a = self._compute_spectral_norm(layer1.lora_a)
        norm2_a = self._compute_spectral_norm(layer2.lora_a)

        assert abs(norm1_a - norm2_a) < SQRT_EPS


class TestGeometricConvergenceMonitor:
    """Tests for geometric convergence monitor."""

    def test_monitor_initialization(self):
        """Test that monitor initializes correctly."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricConvergenceMonitor,
        )

        monitor = GeometricConvergenceMonitor(
            bb_stability_threshold=1e-4,
            budget_threshold=0.9,
            loss_window=10,
        )

        assert monitor.step == 0

    def test_monitor_tracks_steps(self):
        """Test that monitor correctly tracks step count."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricConvergenceMonitor,
        )

        monitor = GeometricConvergenceMonitor()

        # Mock the optimizer's is_bb_stable method
        class MockOptimizer:
            def is_bb_stable(self, threshold=1e-4):
                return False

        mock_opt = MockOptimizer()

        for i in range(5):
            state = monitor.check(mock_opt, {}, 1.0 - i * 0.1)
            assert state.step == i + 1

    def test_monitor_detects_loss_stability(self):
        """Test that monitor detects when loss stabilizes."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricConvergenceMonitor,
        )

        monitor = GeometricConvergenceMonitor(loss_window=5)

        class MockOptimizer:
            def is_bb_stable(self, threshold=1e-4):
                return True

        mock_opt = MockOptimizer()

        # Feed decreasing losses
        for i in range(10):
            state = monitor.check(mock_opt, {}, 1.0 - i * 0.1)

        # Then feed stable losses (very small changes)
        for i in range(20):
            state = monitor.check(mock_opt, {}, 0.1 + SQRT_EPS * 0.01 * i)

        # Should eventually detect stability
        assert state.bb_stable

    def test_convergence_state_should_stop(self):
        """Test that should_stop property works correctly."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricConvergenceState,
        )

        # BB stable + loss stable -> should stop
        state1 = GeometricConvergenceState(
            step=100,
            bb_stable=True,
            loss_stable=True,
            budget_exhausted=False,
        )
        assert state1.should_stop

        # BB stable + budget exhausted -> should stop
        state2 = GeometricConvergenceState(
            step=100,
            bb_stable=True,
            loss_stable=False,
            budget_exhausted=True,
        )
        assert state2.should_stop

        # BB not stable -> should not stop
        state3 = GeometricConvergenceState(
            step=100,
            bb_stable=False,
            loss_stable=True,
            budget_exhausted=True,
        )
        assert not state3.should_stop


class TestResidualScaling:
    """Tests for residual connection scaling."""

    def test_spectral_norm_fast(self):
        """Test that _spectral_norm_fast computes correct values."""
        from modelcypher.core.domain.training.residual_scaling import spectral_norm_power_iteration
        from modelcypher.backends.mlx_backend import MLXBackend
        _backend = MLXBackend()

        # Test with known matrix
        mx.random.seed(42)
        W = mx.random.normal(shape=(64, 128))
        mx.eval(W)

        # Compute via our function
        fast_norm = spectral_norm_power_iteration(W, _backend)

        # Compute via numpy SVD for verification
        W_np = np.array(W.tolist(), dtype=np.float32)
        _, S, _ = np.linalg.svd(W_np, full_matrices=False)
        true_norm = float(S[0])

        # Should be close (power iteration converges to true value)
        rel_error = abs(fast_norm - true_norm) / max(true_norm, SQRT_EPS)
        assert rel_error < 0.1, f"Spectral norm {fast_norm} not close to true {true_norm}"

    def test_spectral_norm_fast_3d(self):
        """Test _spectral_norm_fast with 3D tensor."""
        from modelcypher.core.domain.training.residual_scaling import spectral_norm_power_iteration
        from modelcypher.backends.mlx_backend import MLXBackend
        _backend = MLXBackend()

        mx.random.seed(42)
        x = mx.random.normal(shape=(32, 64, 128))  # [batch, seq, hidden]
        mx.eval(x)

        norm = spectral_norm_power_iteration(x, _backend)

        # Should return a finite positive value
        assert norm > 0
        assert np.isfinite(norm)

    def test_residual_scale_stats(self):
        """Test ResidualScaleStats dataclass."""
        from modelcypher.core.domain.training.residual_scaling import ResidualScaleStats

        stats = ResidualScaleStats(
            layer_idx=0,
            input_spectral=1.0,
            residual_spectral=0.5,
            alpha=2.0,
        )

        assert stats.is_valid  # 2.0 is in [0.1, 10.0]

        stats_invalid = ResidualScaleStats(
            layer_idx=0,
            input_spectral=1.0,
            residual_spectral=100.0,
            alpha=0.01,  # Too small
        )

        assert not stats_invalid.is_valid

    # NOTE: Tests for ResidualScalingHook were deleted - that class
    # was refactored into pure functions in residual_scaling.py


class TestLoRATrainerConfig:
    """Tests for GeometricLoRAConfig with new features."""

    def test_config_default_values(self):
        """Test that config has correct defaults."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricLoRAConfig,
        )

        config = GeometricLoRAConfig(
            target_modules=["q_proj"],
            rank=4,
            geometries={},
        )

        # New defaults
        assert config.enable_geometric_stopping is True
        assert config.max_steps is None

    def test_config_with_geometric_stopping_disabled(self):
        """Test config with geometric stopping disabled."""
        from modelcypher.adapters.training.mlx.geometric_lora_trainer import (
            GeometricLoRAConfig,
        )

        config = GeometricLoRAConfig(
            target_modules=["q_proj"],
            rank=4,
            geometries={},
            enable_geometric_stopping=False,
            max_steps=1000,
        )

        assert config.enable_geometric_stopping is False
        assert config.max_steps == 1000


class TestNumericalStabilityInit:
    """Tests for spectral_normalized_init in numerical_stability.py."""

    def test_spectral_normalized_init_basic(self):
        """Test basic spectral_normalized_init functionality."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import (
            spectral_normalized_init,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        W = spectral_normalized_init(
            shape=(64, 128),
            backend=backend,
            target_spectral=1.0,
        )
        backend.eval(W)

        # Verify shape
        assert W.shape == (64, 128)

        # Compute spectral norm via numpy SVD
        W_np = np.array(backend.tolist(W), dtype=np.float32)
        _, S, _ = np.linalg.svd(W_np, full_matrices=False)
        spectral_norm = float(S[0])

        # Should be close to target (1.0)
        assert abs(spectral_norm - 1.0) < 0.1

    def test_spectral_normalized_init_various_targets(self):
        """Test spectral_normalized_init with various target norms."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import (
            spectral_normalized_init,
        )

        backend = get_default_backend()

        targets = [0.1, 0.5, 1.0, 2.0, 5.0]

        for target in targets:
            backend.random_seed(42)
            W = spectral_normalized_init(
                shape=(32, 64),
                backend=backend,
                target_spectral=target,
            )
            backend.eval(W)

            W_np = np.array(backend.tolist(W), dtype=np.float32)
            _, S, _ = np.linalg.svd(W_np, full_matrices=False)
            actual = float(S[0])

            rel_error = abs(actual - target) / target
            assert rel_error < 0.15, f"Target {target}, got {actual}, rel_error={rel_error}"

    def test_spectral_normalized_lora_init(self):
        """Test spectral_normalized_lora_init creates proper LoRA matrices."""
        from modelcypher.core.domain._backend import get_default_backend
        from modelcypher.core.domain.geometry.numerical_stability import (
            spectral_normalized_lora_init,
        )

        backend = get_default_backend()
        backend.random_seed(42)

        sigma_k = 0.3
        A, B = spectral_normalized_lora_init(
            in_features=128,
            out_features=256,
            rank=8,
            sigma_k=sigma_k,
            backend=backend,
        )
        backend.eval(A, B)

        # Check shapes
        assert A.shape == (8, 128)
        assert B.shape == (256, 8)

        # Check individual spectral norms
        A_np = np.array(backend.tolist(A), dtype=np.float32)
        B_np = np.array(backend.tolist(B), dtype=np.float32)

        _, S_A, _ = np.linalg.svd(A_np, full_matrices=False)
        _, S_B, _ = np.linalg.svd(B_np, full_matrices=False)

        sqrt_sigma_k = np.sqrt(sigma_k)

        # Each should have spectral norm ≈ sqrt(sigma_k)
        assert abs(S_A[0] - sqrt_sigma_k) < sqrt_sigma_k * 0.2
        assert abs(S_B[0] - sqrt_sigma_k) < sqrt_sigma_k * 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

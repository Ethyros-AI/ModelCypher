# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.

"""Tests for geometry-derived optimizer."""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

from modelcypher.core.domain.training.geometric_optimizer import (
    GeometricOptimizer,
    LayerGeometricConfig,
    analyze_model_for_optimizer,
    _compute_geometric_epsilon,
    _compute_decay_scale,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_dim: int = 64, hidden_dim: int = 128, out_dim: int = 32):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x):
        x = self.layer1(x)
        x = nn.relu(x)
        x = self.layer2(x)
        return x


class TestGeometricEpsilon:
    """Tests for geometry-derived epsilon computation."""

    def test_epsilon_from_sigma_k(self):
        """Epsilon should be at least σ_k²."""
        sigma_max = 10.0
        sigma_k = 0.1
        eps = _compute_geometric_epsilon(sigma_max, sigma_k)
        assert eps >= sigma_k ** 2

    def test_epsilon_machine_floor(self):
        """Epsilon should respect machine precision floor."""
        sigma_max = 100.0
        sigma_k = 1e-10  # Very small σ_k
        eps = _compute_geometric_epsilon(sigma_max, sigma_k)
        # Should use machine floor instead of σ_k²
        sqrt_eps = np.sqrt(np.finfo(np.float32).eps)
        assert eps >= sqrt_eps * sigma_max ** 2

    def test_epsilon_well_conditioned(self):
        """Well-conditioned matrix uses σ_k² as epsilon."""
        sigma_max = 1.0
        sigma_k = 0.5  # Well conditioned (κ = 2)
        eps = _compute_geometric_epsilon(sigma_max, sigma_k)
        # σ_k² should dominate for well-conditioned matrices
        assert eps == pytest.approx(sigma_k ** 2, rel=1e-6)


class TestDecayScale:
    """Tests for condition-aware weight decay scaling."""

    def test_well_conditioned_full_decay(self):
        """Well-conditioned layers get full decay."""
        sigma_max = 1.0
        sigma_k = 1.0  # κ = 1 (perfectly conditioned)
        scale = _compute_decay_scale(sigma_max, sigma_k)
        assert scale == pytest.approx(1.0)

    def test_ill_conditioned_reduced_decay(self):
        """Ill-conditioned layers get reduced decay."""
        sigma_max = 100.0
        sigma_k = 1.0  # κ = 100
        scale = _compute_decay_scale(sigma_max, sigma_k)
        assert scale == pytest.approx(0.01)

    def test_decay_scale_range(self):
        """Decay scale should be in (0, 1]."""
        for kappa in [1, 10, 100, 1000]:
            sigma_max = float(kappa)
            sigma_k = 1.0
            scale = _compute_decay_scale(sigma_max, sigma_k)
            assert 0 < scale <= 1.0


class TestModelAnalysis:
    """Tests for model geometry analysis."""

    def test_analyze_simple_model(self):
        """Should analyze all 2D weight matrices."""
        model = SimpleModel()
        mx.eval(model.parameters())

        configs = analyze_model_for_optimizer(model)

        # Should have configs for layer1.weight and layer2.weight
        assert len(configs) >= 2
        assert any("layer1" in key for key in configs.keys())
        assert any("layer2" in key for key in configs.keys())

    def test_lr_scale_derivation(self):
        """LR scale should be max_σ / σ_max_i."""
        model = SimpleModel()
        mx.eval(model.parameters())

        configs = analyze_model_for_optimizer(model)

        # Find max sigma across all layers
        max_sigma = max(cfg.sigma_max for cfg in configs.values())

        # Verify lr_scale formula
        for cfg in configs.values():
            expected_scale = max_sigma / cfg.sigma_max
            assert cfg.lr_scale == pytest.approx(expected_scale, rel=1e-4)

    def test_lr_scale_minimum_is_one(self):
        """Layer with max σ should have lr_scale = 1.0."""
        model = SimpleModel()
        mx.eval(model.parameters())

        configs = analyze_model_for_optimizer(model)
        lr_scales = [cfg.lr_scale for cfg in configs.values()]

        # Minimum lr_scale should be 1.0 (for layer with max σ)
        assert min(lr_scales) == pytest.approx(1.0, rel=1e-4)


class TestGeometricOptimizer:
    """Tests for the optimizer itself."""

    def test_init_from_model(self):
        """Optimizer should initialize from model geometry."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=0.01)
        optimizer.init_from_model(model)

        assert optimizer.base_lr is not None
        assert optimizer.base_lr > 0
        assert len(optimizer.layer_configs) >= 2

    def test_base_lr_derivation(self):
        """Base LR should be 1 / max(σ_max)."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        # Compute expected base_lr
        max_sigma = max(cfg.sigma_max for cfg in optimizer.layer_configs.values())
        expected_lr = 1.0 / max_sigma

        assert optimizer.base_lr == pytest.approx(expected_lr, rel=1e-6)

    def test_update_reduces_loss(self):
        """Optimizer update should reduce loss on simple problem."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=0.0)
        optimizer.init_from_model(model)

        # Simple regression target
        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        initial_loss = float(loss_fn(model))

        # Take a few gradient steps
        for _ in range(10):
            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        final_loss = float(loss_fn(model))
        assert final_loss < initial_loss

    def test_state_serialization(self):
        """Optimizer state should be serializable for checkpointing."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer(base_decay=0.01)
        optimizer.init_from_model(model)

        state = optimizer.state

        assert state["type"] == "geometric"
        assert state["base_lr"] == optimizer.base_lr
        assert state["base_decay"] == 0.01
        assert "layer_configs" in state
        assert len(state["layer_configs"]) == len(optimizer.layer_configs)

    def test_state_restore(self):
        """Should restore optimizer state from checkpoint."""
        model = SimpleModel()
        mx.eval(model.parameters())

        # Initialize and get state
        opt1 = GeometricOptimizer(base_decay=0.01)
        opt1.init_from_model(model)
        state = opt1.state

        # Restore to new optimizer
        opt2 = GeometricOptimizer()
        opt2.load_state(state)

        assert opt2.base_lr == opt1.base_lr
        assert opt2.base_decay == opt1.base_decay
        assert len(opt2.layer_configs) == len(opt1.layer_configs)

    def test_learning_rate_property(self):
        """Learning rate property should work for engine compatibility."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        original_lr = optimizer.learning_rate

        # Setter should work (for warmup)
        optimizer.learning_rate = 0.001
        assert optimizer.learning_rate == 0.001

        # Restore
        optimizer.learning_rate = original_lr
        assert optimizer.learning_rate == original_lr

    def test_uninitialized_update_raises(self):
        """Update without init should raise."""
        model = SimpleModel()
        optimizer = GeometricOptimizer()

        with pytest.raises(RuntimeError, match="not initialized"):
            optimizer.update(model, {})

    def test_no_momentum_state(self):
        """Geometric optimizer should have no momentum state."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        state = optimizer.state

        # Should not have any momentum-related keys
        assert "m" not in state
        assert "v" not in state
        assert "momentum" not in str(state).lower()


class TestEffectiveLearningRate:
    """Tests for effective per-layer learning rate."""

    def test_effective_lr_is_inverse_sigma(self):
        """Effective LR should be 1/σ_max for each layer."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        for cfg in optimizer.layer_configs.values():
            effective_lr = optimizer.base_lr * cfg.lr_scale
            expected = 1.0 / cfg.sigma_max
            assert effective_lr == pytest.approx(expected, rel=1e-4)

    def test_larger_weights_smaller_steps(self):
        """Layers with larger σ_max should get smaller effective LR."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        configs = list(optimizer.layer_configs.values())
        if len(configs) >= 2:
            # Sort by sigma_max
            sorted_configs = sorted(configs, key=lambda c: c.sigma_max)

            # Smaller sigma_max should have larger lr_scale
            for i in range(len(sorted_configs) - 1):
                smaller = sorted_configs[i]
                larger = sorted_configs[i + 1]
                assert smaller.lr_scale >= larger.lr_scale


class TestBarzilaiBorwein:
    """Tests for Barzilai-Borwein adaptive learning rate."""

    def test_first_step_uses_spectral_lr(self):
        """First step should use spectral LR (1/σ_max) - no gradient history yet."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        # Get expected spectral LRs
        expected_lrs = {
            key: optimizer.base_lr * cfg.lr_scale
            for key, cfg in optimizer.layer_configs.items()
        }

        # Take one gradient step
        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        _, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Check that first step used spectral LRs
        for key, lr in optimizer._per_layer_lr.items():
            if key in expected_lrs:
                assert lr == pytest.approx(expected_lrs[key], rel=1e-6)

    def test_bb_adapts_after_first_step(self):
        """BB should adapt LR after first step based on gradient history."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        # Take two steps
        _, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        first_step_lrs = dict(optimizer._per_layer_lr)

        _, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        second_step_lrs = dict(optimizer._per_layer_lr)

        # Step count should have advanced
        assert optimizer._step_count == 2

        # Both steps should have recorded LRs
        assert len(first_step_lrs) > 0
        assert len(second_step_lrs) > 0

    def test_bb_respects_spectral_bounds(self):
        """BB LR should be bounded by [σ_k/σ_max, 1/σ_max]."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        # Take several steps to let BB adapt
        for _ in range(5):
            _, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

            # Check all LRs are within spectral bounds
            for key, lr in optimizer._per_layer_lr.items():
                config = optimizer.layer_configs.get(key)
                if config is not None:
                    min_lr = config.sigma_k / config.sigma_max
                    max_lr = 1.0 / config.sigma_max
                    assert lr >= min_lr - 1e-10, f"{key}: LR {lr} < min {min_lr}"
                    assert lr <= max_lr + 1e-10, f"{key}: LR {lr} > max {max_lr}"

    def test_get_lr_stats(self):
        """get_lr_stats should return statistics about per-layer LRs."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        # Before any updates, stats should be empty
        stats = optimizer.get_lr_stats()
        assert stats == {}

        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        # Take a step
        _, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # Now stats should have values
        stats = optimizer.get_lr_stats()
        assert "lr_mean" in stats
        assert "lr_min" in stats
        assert "lr_max" in stats
        assert "lr_std" in stats
        assert "step_count" in stats
        assert stats["step_count"] == 1
        assert stats["lr_min"] <= stats["lr_mean"] <= stats["lr_max"]

    def test_bb_reduces_loss(self):
        """BB adaptation should reduce loss."""
        model = SimpleModel(in_dim=16, hidden_dim=32, out_dim=8)
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        x = mx.random.normal(shape=(4, 16))
        y = mx.random.normal(shape=(4, 8))

        def loss_fn(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        initial_loss = float(loss_fn(model))

        # Take gradient steps
        for _ in range(20):
            _, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        final_loss = float(loss_fn(model))
        assert final_loss < initial_loss

    def test_state_includes_step_count(self):
        """Optimizer state should include step count for proper resumption."""
        model = SimpleModel()
        mx.eval(model.parameters())

        optimizer = GeometricOptimizer()
        optimizer.init_from_model(model)

        state = optimizer.state
        assert "step_count" in state

    def test_state_restore_preserves_step_count(self):
        """Restoring state should preserve step count."""
        model = SimpleModel()
        mx.eval(model.parameters())

        opt1 = GeometricOptimizer()
        opt1.init_from_model(model)
        opt1._step_count = 42  # Simulate some training
        state = opt1.state

        opt2 = GeometricOptimizer()
        opt2.load_state(state)

        assert opt2._step_count == 42

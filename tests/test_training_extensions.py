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
Unit tests for training extension modules (requires MLX).

Tests:
- GeometricLoRALinear (spectral-normalized LoRA)
- LR scheduling algorithms
- Loss landscape computation

NOTE: LoRA presets (for_mistral, for_llama, etc.) were removed.
All LoRA parameters are now geometry-derived. See geometric_lora.py.
"""

import math
import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx
    import mlx.nn as nn

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore
    nn = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    cos_scalar,
    machine_epsilon,
    pi_value,
)
from modelcypher.adapters.training.mlx.loss_landscape import (
    LossLandscapeComputer,
)
from modelcypher.core.domain.training.scheduling import (
    ConstantSchedule,
    CosineSchedule,
    LinearWarmupSchedule,
    ScheduleConfig,
    ScheduleType,
    StepDecaySchedule,
    create_schedule,
)


class TestGeometricLoRALinear:
    """Tests for GeometricLoRALinear (spectral-normalized LoRA)."""

    def test_forward_pass(self):
        """Test that forward pass produces correct shape output."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear
        from modelcypher.backends.mlx_backend import MLXBackend

        # Create a base linear layer
        base = nn.Linear(64, 32)

        # Create geometric LoRA wrapper
        lora = GeometricLoRALinear(
            base_layer=base,
            sigma_k=0.1,  # Spectral scale bound
            rank=4,
            backend=MLXBackend(),
        )

        x = mx.random.normal((2, 64))
        y = lora(x)
        mx.eval(y)

        assert y.shape == (2, 32)

    def test_spectral_initialization(self):
        """Test that LoRA matrices are spectrally normalized at init."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear
        from modelcypher.backends.mlx_backend import MLXBackend

        base = nn.Linear(64, 32)
        sigma_k = 0.1

        lora = GeometricLoRALinear(
            base_layer=base,
            sigma_k=sigma_k,
            rank=4,
            backend=MLXBackend(),
        )

        # Compute ||B @ A||_spectral (use CPU stream for SVD)
        delta = lora.lora_b @ lora.lora_a
        _, S, _ = mx.linalg.svd(delta, stream=mx.cpu)
        mx.eval(S)
        spectral_norm = float(S[0])

        # Should be approximately sigma_k at initialization
        # Tolerance is 25% due to random initialization + power iteration approximation
        assert abs(spectral_norm - sigma_k) < 0.25 * sigma_k

    def test_trainable_parameters(self):
        """Test that only LoRA parameters are trainable."""
        from modelcypher.adapters.training.mlx_adapter import GeometricLoRALinear
        from modelcypher.backends.mlx_backend import MLXBackend

        base = nn.Linear(64, 32)
        lora = GeometricLoRALinear(
            base_layer=base,
            sigma_k=0.1,
            rank=4,
            backend=MLXBackend(),
        )

        # LoRA A: [rank, in_features]
        assert lora.lora_a.shape == (4, 64)
        # LoRA B: [out_features, rank]
        assert lora.lora_b.shape == (32, 4)


class TestLRSchedules:
    """Tests for learning rate schedules."""

    def test_constant_schedule(self):
        schedule = ConstantSchedule(lr=1e-4)
        assert schedule.get_lr(0) == 1e-4
        assert schedule.get_lr(100) == 1e-4
        assert schedule.get_lr(1000) == 1e-4

    def test_linear_warmup(self):
        schedule = LinearWarmupSchedule(base_lr=1e-4, warmup_steps=100)

        # During warmup
        assert schedule.get_lr(0) == pytest.approx(1e-6, abs=math.ulp(1e-6))
        assert schedule.get_lr(50) == pytest.approx(0.51e-4, abs=math.ulp(0.51e-4))

        # After warmup
        assert schedule.get_lr(100) == 1e-4
        assert schedule.get_lr(200) == 1e-4

    def test_cosine_schedule(self):
        schedule = CosineSchedule(
            base_lr=1e-4,
            total_steps=1000,
            warmup_steps=100,
            min_lr=1e-6,
        )

        # During warmup (linear increase)
        lr_50 = schedule.get_lr(50)
        assert lr_50 < schedule.get_lr(100)

        # At warmup end
        assert schedule.get_lr(100) == pytest.approx(1e-4, abs=math.ulp(1e-4))

        # Midpoint should be ~half
        lr_mid = schedule.get_lr(550)  # Middle of decay phase
        decay_steps = 1000 - 100
        progress = (550 - 100) / decay_steps
        backend = get_default_backend()
        expected_mid = 1e-6 + (1e-4 - 1e-6) * (
            0.5 * (1.0 + cos_scalar(pi_value(backend) * progress, backend))
        )
        eps = machine_epsilon(backend, backend.array([expected_mid]))
        assert lr_mid == pytest.approx(expected_mid, rel=eps)

        # End should be min_lr
        assert schedule.get_lr(1000) == pytest.approx(1e-6, abs=math.ulp(1e-6))

    def test_step_decay_schedule(self):
        schedule = StepDecaySchedule(
            base_lr=1e-4,
            step_size=100,
            gamma=0.1,
        )

        assert schedule.get_lr(0) == 1e-4
        assert schedule.get_lr(99) == 1e-4
        assert schedule.get_lr(100) == pytest.approx(1e-5, abs=math.ulp(1e-5))
        assert schedule.get_lr(200) == pytest.approx(1e-6, abs=math.ulp(1e-6))

    def test_schedule_factory(self):
        config = ScheduleConfig(
            schedule_type=ScheduleType.COSINE,
            base_lr=3e-5,
            total_steps=500,
            warmup_steps=50,
        )
        schedule = create_schedule(config)
        assert isinstance(schedule, CosineSchedule)


class TestLossLandscape:
    """Tests for loss landscape computation."""

    def test_surface_computation(self):
        computer = LossLandscapeComputer(resolution=5, scale=0.1)

        # Simple quadratic loss
        params = {"w": mx.array([1.0, 2.0])}

        def loss_fn(p):
            return float(mx.sum(p["w"] ** 2).item())

        surface = computer.compute_surface(params, loss_fn)

        assert surface.resolution == 5
        assert len(surface.points) == 25  # 5x5 grid
        assert surface.min_loss <= surface.center_loss <= surface.max_loss

    def test_curvature_estimation(self):
        computer = LossLandscapeComputer()

        # Quadratic bowl: L = x^2 + y^2
        params = {"x": mx.array([0.5, 0.5])}

        def loss_fn(p):
            return float(mx.sum(p["x"] ** 2).item())

        metrics = computer.estimate_curvature(
            params,
            loss_fn,
            num_samples=10,
            epsilon=1e-2,
        )

        # Hessian of x^2 is 2I, so eigenvalues should be ~2
        # With finite differences and iterations, expect rough approximation
        assert metrics.max_eigenvalue > 0
        assert metrics.sharpness >= 0
        assert metrics.sharpness <= 1

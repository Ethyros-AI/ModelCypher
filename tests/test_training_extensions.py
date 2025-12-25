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
Unit tests for training extension parity modules (requires MLX).

Tests:
- LoRA configuration and target resolution
- LR scheduling algorithms
- Loss landscape computation
"""

import pytest

# Attempt MLX import - skip module entirely if unavailable
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore

# Skip all tests in this module if MLX unavailable
pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available (requires Apple Silicon)")
from modelcypher.core.domain.training.lora_mlx import (
    LoRAConfig,
    LoRALinear,
)
from modelcypher.core.domain.training.loss_landscape_mlx import (
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


class TestLoRAConfig:
    """Tests for LoRA configuration."""

    def test_default_config(self):
        config = LoRAConfig.default()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

    def test_scale_calculation(self):
        config = LoRAConfig(rank=8, alpha=16.0)
        assert config.scale == 2.0

        config = LoRAConfig(rank=16, alpha=32.0)
        assert config.scale == 2.0

    def test_presets(self):
        mistral = LoRAConfig.for_mistral()
        assert mistral.rank == 16
        assert "o_proj" in mistral.target_modules

        llama = LoRAConfig.for_llama()
        assert llama.rank == 8

        qwen = LoRAConfig.for_qwen()
        assert "gate_proj" in qwen.target_modules


class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_forward_pass(self):
        lora = LoRALinear(
            in_features=64,
            out_features=32,
            rank=4,
            alpha=8.0,
        )

        x = mx.random.normal((2, 64))
        y = lora(x)
        mx.eval(y)

        assert y.shape == (2, 32)

    def test_trainable_parameters(self):
        lora = LoRALinear(
            in_features=64,
            out_features=32,
            rank=4,
        )

        params = lora.trainable_parameters()
        assert "lora_a" in params
        assert "lora_b" in params
        assert params["lora_a"].shape == (4, 64)
        assert params["lora_b"].shape == (32, 4)


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
        assert schedule.get_lr(0) == pytest.approx(1e-6, rel=0.01)
        assert schedule.get_lr(50) == pytest.approx(0.51e-4, rel=0.01)

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
        assert schedule.get_lr(100) == pytest.approx(1e-4, rel=0.01)

        # Midpoint should be ~half
        lr_mid = schedule.get_lr(550)  # Middle of decay phase
        assert lr_mid < 1e-4
        assert lr_mid > 1e-6

        # End should be min_lr
        assert schedule.get_lr(1000) == pytest.approx(1e-6, rel=0.01)

    def test_step_decay_schedule(self):
        schedule = StepDecaySchedule(
            base_lr=1e-4,
            step_size=100,
            gamma=0.1,
        )

        assert schedule.get_lr(0) == 1e-4
        assert schedule.get_lr(99) == 1e-4
        assert schedule.get_lr(100) == pytest.approx(1e-5, rel=0.01)
        assert schedule.get_lr(200) == pytest.approx(1e-6, rel=0.01)

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

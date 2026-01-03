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

"""Tests for learning rate scheduling."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelcypher.core.domain._backend import get_default_backend
from modelcypher.core.domain.geometry.numerical_stability import (
    cos_scalar,
    division_epsilon,
    pi_value,
)
from modelcypher.core.domain.training.scheduling import (
    ConstantSchedule,
    CosineSchedule,
    IdleSchedulerConfig,
    IdleTrainingScheduler,
    IdleTrainingState,
    LinearDecaySchedule,
    LinearWarmupSchedule,
    ScheduleConfig,
    ScheduleType,
    StepDecaySchedule,
    create_schedule,
)


def _eps(value: float) -> float:
    b = get_default_backend()
    arr = b.array([value])
    b.eval(arr)
    return division_epsilon(b, arr) * max(1.0, abs(value))


def _cosine_expected(
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float,
    step: int,
) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    decay_steps = total_steps - warmup_steps
    decay_step = step - warmup_steps
    progress = decay_step / decay_steps
    b = get_default_backend()
    cosine_decay = 0.5 * (1.0 + cos_scalar(pi_value(b) * progress, b))
    return min_lr + (base_lr - min_lr) * cosine_decay


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_constant_value(self):
        assert ScheduleType.CONSTANT.value == "constant"

    def test_linear_value(self):
        assert ScheduleType.LINEAR.value == "linear"

    def test_cosine_value(self):
        assert ScheduleType.COSINE.value == "cosine"

    def test_warmup_cosine_value(self):
        assert ScheduleType.WARMUP_COSINE.value == "warmup_cosine"

    def test_step_value(self):
        assert ScheduleType.STEP.value == "step"


class TestConstantSchedule:
    """Tests for ConstantSchedule."""

    def test_constant_returns_same_value(self):
        schedule = ConstantSchedule(lr=3e-5)
        for step in [0, 1, 10, 100, 1000]:
            assert schedule.get_lr(step) == 3e-5

    def test_base_lr_property(self):
        schedule = ConstantSchedule(lr=1e-4)
        assert schedule.base_lr == 1e-4

    @given(st.floats(min_value=1e-7, max_value=1e-2), st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20)
    def test_constant_always_returns_base_lr(self, lr: float, step: int):
        """Property: constant schedule always returns base_lr."""
        schedule = ConstantSchedule(lr=lr)
        assert schedule.get_lr(step) == lr


class TestLinearWarmupSchedule:
    """Tests for LinearWarmupSchedule."""

    def test_warmup_starts_at_fraction(self):
        schedule = LinearWarmupSchedule(base_lr=1e-4, warmup_steps=100)
        # Step 0 should give 1/100 of base_lr
        expected = 1e-4 / 100
        assert abs(schedule.get_lr(0) - expected) <= _eps(expected)

    def test_warmup_ends_at_base_lr(self):
        schedule = LinearWarmupSchedule(base_lr=1e-4, warmup_steps=100)
        # At warmup_steps-1, should be close to base_lr
        expected = 1e-4
        assert abs(schedule.get_lr(99) - expected) <= _eps(expected)

    def test_after_warmup_is_constant(self):
        schedule = LinearWarmupSchedule(base_lr=1e-4, warmup_steps=100)
        # After warmup, should stay at base_lr
        assert schedule.get_lr(100) == 1e-4
        assert schedule.get_lr(500) == 1e-4

    def test_base_lr_property(self):
        schedule = LinearWarmupSchedule(base_lr=5e-5, warmup_steps=50)
        assert schedule.base_lr == 5e-5

    def test_zero_warmup_steps_treated_as_one(self):
        schedule = LinearWarmupSchedule(base_lr=1e-4, warmup_steps=0)
        # Should clamp to 1 warmup step
        assert schedule.warmup_steps == 1

    @given(
        st.floats(min_value=1e-6, max_value=1e-3),
        st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=20)
    def test_warmup_monotonically_increasing(self, base_lr: float, warmup_steps: int):
        """Property: warmup phase should be monotonically increasing."""
        schedule = LinearWarmupSchedule(base_lr=base_lr, warmup_steps=warmup_steps)
        prev_lr = 0.0
        for step in range(warmup_steps):
            lr = schedule.get_lr(step)
            assert lr >= prev_lr, f"LR should increase during warmup at step {step}"
            prev_lr = lr


class TestLinearDecaySchedule:
    """Tests for LinearDecaySchedule."""

    def test_warmup_phase(self):
        schedule = LinearDecaySchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=100, min_lr=1e-6
        )
        # During warmup, should increase
        assert schedule.get_lr(0) < schedule.get_lr(50) < schedule.get_lr(99)

    def test_decay_after_warmup(self):
        schedule = LinearDecaySchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=100, min_lr=1e-6
        )
        # After warmup, should decay
        assert schedule.get_lr(100) > schedule.get_lr(500) > schedule.get_lr(999)

    def test_reaches_min_lr_at_total_steps(self):
        schedule = LinearDecaySchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=1e-6
        )
        expected = 1e-6 + (1e-4 - 1e-6) * (1.0 - (999 / 1000))
        assert abs(schedule.get_lr(999) - expected) <= _eps(expected)

    def test_after_total_steps_returns_min_lr(self):
        schedule = LinearDecaySchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=1e-6
        )
        assert schedule.get_lr(1000) == 1e-6
        assert schedule.get_lr(2000) == 1e-6

    def test_no_warmup_starts_at_base(self):
        schedule = LinearDecaySchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=0.0
        )
        # No warmup should start at base_lr
        assert schedule.get_lr(0) == 1e-4

    @given(
        st.floats(min_value=1e-6, max_value=1e-3),
        st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=20)
    def test_decay_monotonically_decreasing(self, base_lr: float, total_steps: int):
        """Property: decay phase should be monotonically decreasing."""
        schedule = LinearDecaySchedule(
            base_lr=base_lr, total_steps=total_steps, warmup_steps=0, min_lr=0.0
        )
        prev_lr = base_lr
        for step in range(0, total_steps, max(1, total_steps // 20)):
            lr = schedule.get_lr(step)
            assert lr <= prev_lr, f"LR should decrease during decay at step {step}"
            prev_lr = lr


class TestCosineSchedule:
    """Tests for CosineSchedule."""

    def test_starts_at_base_lr_no_warmup(self):
        schedule = CosineSchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=0.0
        )
        assert schedule.get_lr(0) == 1e-4

    def test_cosine_shape(self):
        schedule = CosineSchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=0.0
        )
        for step in (0, 500, 999):
            expected = _cosine_expected(1e-4, 1000, 0, 0.0, step)
            lr = schedule.get_lr(step)
            assert abs(lr - expected) <= _eps(expected)

    def test_warmup_then_cosine(self):
        schedule = CosineSchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=100, min_lr=0.0
        )
        # During warmup, increases
        assert schedule.get_lr(0) < schedule.get_lr(50)
        # At warmup end, at base_lr
        expected = _cosine_expected(1e-4, 1000, 100, 0.0, 99)
        assert abs(schedule.get_lr(99) - expected) <= _eps(expected)
        # After warmup, cosine decay begins
        assert schedule.get_lr(100) > schedule.get_lr(500)

    def test_after_total_steps_returns_min_lr(self):
        schedule = CosineSchedule(
            base_lr=1e-4, total_steps=1000, warmup_steps=0, min_lr=1e-6
        )
        assert schedule.get_lr(1000) == 1e-6
        assert schedule.get_lr(2000) == 1e-6

    @given(
        st.floats(min_value=1e-6, max_value=1e-3),
        st.integers(min_value=100, max_value=5000),
        st.floats(min_value=0.0, max_value=1e-6),
    )
    @settings(max_examples=20)
    def test_cosine_bounded(self, base_lr: float, total_steps: int, min_lr: float):
        """Property: cosine LR should be bounded between min_lr and base_lr."""
        schedule = CosineSchedule(
            base_lr=base_lr, total_steps=total_steps, warmup_steps=0, min_lr=min_lr
        )
        for step in range(0, total_steps, max(1, total_steps // 20)):
            lr = schedule.get_lr(step)
            eps = _eps(base_lr)
            assert min_lr - eps <= lr <= base_lr + eps, (
                f"LR out of bounds at step {step}"
            )


class TestStepDecaySchedule:
    """Tests for StepDecaySchedule."""

    def test_constant_within_step_size(self):
        schedule = StepDecaySchedule(base_lr=1e-4, step_size=100, gamma=0.1)
        # Within first step_size, should be constant
        assert schedule.get_lr(0) == 1e-4
        assert schedule.get_lr(50) == 1e-4
        assert schedule.get_lr(99) == 1e-4

    def test_decay_at_step_size(self):
        schedule = StepDecaySchedule(base_lr=1e-4, step_size=100, gamma=0.1)
        # At step 100, should decay by gamma
        expected = 1e-4 * 0.1
        assert abs(schedule.get_lr(100) - expected) <= _eps(expected)

    def test_multiple_decays(self):
        schedule = StepDecaySchedule(base_lr=1e-4, step_size=100, gamma=0.5)
        expected_0 = 1e-4
        expected_1 = 0.5e-4
        expected_2 = 0.25e-4
        assert abs(schedule.get_lr(0) - expected_0) <= _eps(expected_0)
        assert abs(schedule.get_lr(100) - expected_1) <= _eps(expected_1)
        assert abs(schedule.get_lr(200) - expected_2) <= _eps(expected_2)

    def test_respects_min_lr(self):
        schedule = StepDecaySchedule(base_lr=1e-4, step_size=10, gamma=0.1, min_lr=1e-8)
        # After many decays, should clamp to min_lr
        assert schedule.get_lr(1000) >= 1e-8

    def test_base_lr_property(self):
        schedule = StepDecaySchedule(base_lr=3e-5, step_size=100, gamma=0.1)
        assert schedule.base_lr == 3e-5

    @given(
        st.floats(min_value=1e-6, max_value=1e-3),
        st.integers(min_value=10, max_value=500),
        st.floats(min_value=0.1, max_value=0.9),
    )
    @settings(max_examples=20)
    def test_step_decay_decreases(self, base_lr: float, step_size: int, gamma: float):
        """Property: step decay should decrease at step boundaries."""
        schedule = StepDecaySchedule(base_lr=base_lr, step_size=step_size, gamma=gamma)
        # LR at step 0 should be > LR at step_size
        assert schedule.get_lr(0) > schedule.get_lr(step_size)


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_default_values(self):
        config = ScheduleConfig()
        assert config.schedule_type == ScheduleType.COSINE
        assert config.base_lr == 3e-5
        assert config.total_steps == 1000
        assert config.warmup_steps == 0
        assert config.min_lr == 0.0

    def test_custom_values(self):
        config = ScheduleConfig(
            schedule_type=ScheduleType.LINEAR,
            base_lr=1e-4,
            total_steps=5000,
            warmup_steps=500,
            min_lr=1e-6,
        )
        assert config.schedule_type == ScheduleType.LINEAR
        assert config.base_lr == 1e-4


class TestCreateSchedule:
    """Tests for create_schedule factory."""

    def test_create_constant(self):
        config = ScheduleConfig(schedule_type=ScheduleType.CONSTANT, base_lr=1e-4)
        schedule = create_schedule(config)
        assert isinstance(schedule, ConstantSchedule)
        assert schedule.base_lr == 1e-4

    def test_create_linear(self):
        config = ScheduleConfig(schedule_type=ScheduleType.LINEAR, total_steps=1000)
        schedule = create_schedule(config)
        assert isinstance(schedule, LinearDecaySchedule)

    def test_create_cosine(self):
        config = ScheduleConfig(schedule_type=ScheduleType.COSINE, total_steps=1000)
        schedule = create_schedule(config)
        assert isinstance(schedule, CosineSchedule)

    def test_create_warmup_cosine_is_cosine(self):
        config = ScheduleConfig(schedule_type=ScheduleType.WARMUP_COSINE)
        schedule = create_schedule(config)
        assert isinstance(schedule, CosineSchedule)

    def test_create_step(self):
        config = ScheduleConfig(
            schedule_type=ScheduleType.STEP, step_size=100, gamma=0.5
        )
        schedule = create_schedule(config)
        assert isinstance(schedule, StepDecaySchedule)


class TestIdleSchedulerConfig:
    """Tests for IdleSchedulerConfig."""

    def test_default_values(self):
        config = IdleSchedulerConfig()
        assert config.enabled is True
        assert config.min_idle_seconds == 60.0
        assert config.max_batch_steps == 10
        assert config.pause_on_activity is True

    def test_custom_values(self):
        config = IdleSchedulerConfig(
            enabled=False,
            min_idle_seconds=120.0,
            max_batch_steps=5,
            pause_on_activity=False,
        )
        assert config.enabled is False
        assert config.min_idle_seconds == 120.0


class TestIdleTrainingState:
    """Tests for IdleTrainingState enum."""

    def test_all_states(self):
        assert IdleTrainingState.IDLE.value == "idle"
        assert IdleTrainingState.WAITING.value == "waiting"
        assert IdleTrainingState.TRAINING.value == "training"
        assert IdleTrainingState.PAUSED.value == "paused"
        assert IdleTrainingState.DISABLED.value == "disabled"


class TestIdleTrainingScheduler:
    """Tests for IdleTrainingScheduler."""

    def test_enabled_starts_in_idle(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(enabled=True))
        assert scheduler.state == IdleTrainingState.IDLE

    def test_disabled_starts_in_disabled(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(enabled=False))
        assert scheduler.state == IdleTrainingState.DISABLED

    def test_default_config_if_none(self):
        scheduler = IdleTrainingScheduler(None)
        assert scheduler.config.enabled is True

    def test_enable_changes_state(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(enabled=False))
        assert scheduler.state == IdleTrainingState.DISABLED
        scheduler.enable()
        assert scheduler.state == IdleTrainingState.IDLE

    def test_disable_changes_state(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(enabled=True))
        assert scheduler.state == IdleTrainingState.IDLE
        scheduler.disable()
        assert scheduler.state == IdleTrainingState.DISABLED

    def test_should_train_false_when_disabled(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(enabled=False))
        assert scheduler.should_train() is False

    def test_on_step_complete_increments_counter(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(max_batch_steps=3))
        scheduler._state = IdleTrainingState.TRAINING
        scheduler.on_step_complete()
        assert scheduler._accumulated_steps == 1
        scheduler.on_step_complete()
        assert scheduler._accumulated_steps == 2

    def test_max_batch_steps_transitions_to_waiting(self):
        scheduler = IdleTrainingScheduler(IdleSchedulerConfig(max_batch_steps=2))
        scheduler._state = IdleTrainingState.TRAINING
        scheduler.on_step_complete()  # 1
        assert scheduler.state == IdleTrainingState.TRAINING
        scheduler.on_step_complete()  # 2 - triggers transition
        assert scheduler.state == IdleTrainingState.WAITING
        assert scheduler._accumulated_steps == 0

    def test_on_activity_pauses_training(self):
        scheduler = IdleTrainingScheduler(
            IdleSchedulerConfig(pause_on_activity=True)
        )
        scheduler._state = IdleTrainingState.TRAINING
        scheduler.on_activity()
        assert scheduler.state == IdleTrainingState.PAUSED

    def test_on_activity_no_pause_if_disabled(self):
        scheduler = IdleTrainingScheduler(
            IdleSchedulerConfig(pause_on_activity=False)
        )
        scheduler._state = IdleTrainingState.TRAINING
        scheduler.on_activity()
        assert scheduler.state == IdleTrainingState.TRAINING

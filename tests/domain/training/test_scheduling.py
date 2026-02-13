# Copyright (C) 2025 EthyrosAI LLC / Jason Kempf
#
# This file is part of ModelCypher.
#
# ModelCypher is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import math
import time

import pytest

from modelcypher.core.domain.training.scheduling import (
    ConstantSchedule,
    CoreIdleTrainingScheduler,
    CosineSchedule,
    IdleSchedulerConfig,
    IdleTrainingState,
    LinearDecaySchedule,
    LinearWarmupSchedule,
    ScheduleConfig,
    ScheduleType,
    StepDecaySchedule,
    create_schedule,
)


class TestConstantSchedule:
    def test_constant_lr(self):
        sched = ConstantSchedule(lr=0.01)
        assert sched.get_lr(0) == 0.01
        assert sched.get_lr(100) == 0.01
        assert sched.base_lr == 0.01


class TestLinearWarmupSchedule:
    def test_step_zero(self):
        sched = LinearWarmupSchedule(base_lr=0.1, warmup_steps=10)
        # step=0: 0.1 * (0+1)/10 = 0.01
        assert sched.get_lr(0) == pytest.approx(0.01)

    def test_at_warmup(self):
        sched = LinearWarmupSchedule(base_lr=0.1, warmup_steps=10)
        # step=10 >= warmup → base_lr
        assert sched.get_lr(10) == pytest.approx(0.1)

    def test_after_warmup(self):
        sched = LinearWarmupSchedule(base_lr=0.1, warmup_steps=10)
        assert sched.get_lr(20) == pytest.approx(0.1)


class TestLinearDecaySchedule:
    def test_warmup_then_decay(self):
        sched = LinearDecaySchedule(base_lr=0.1, total_steps=100, warmup_steps=10)
        # During warmup
        assert sched.get_lr(0) < sched.get_lr(5) < sched.get_lr(9)
        # After warmup, should decay
        assert sched.get_lr(10) > sched.get_lr(50) > sched.get_lr(90)

    def test_at_total_steps(self):
        sched = LinearDecaySchedule(base_lr=0.1, total_steps=100, min_lr=0.001)
        assert sched.get_lr(100) == pytest.approx(0.001)


class TestCosineSchedule:
    def test_endpoints(self, any_backend):
        sched = CosineSchedule(base_lr=0.1, total_steps=100, min_lr=0.0)
        assert sched.get_lr(0) == pytest.approx(0.1, abs=1e-6)
        assert sched.get_lr(100) == pytest.approx(0.0, abs=1e-6)

    def test_midpoint(self, any_backend):
        sched = CosineSchedule(base_lr=0.1, total_steps=100, min_lr=0.0)
        mid_lr = sched.get_lr(50)
        # cos(pi * 0.5) = 0 → lr = 0.5 * 0.1 * (1 + 0) = 0.05
        assert mid_lr == pytest.approx(0.05, abs=1e-4)

    def test_monotonically_decreasing(self, any_backend):
        sched = CosineSchedule(base_lr=0.1, total_steps=100, min_lr=0.0)
        lrs = [sched.get_lr(s) for s in range(0, 101, 10)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1]


class TestStepDecaySchedule:
    def test_before_step_size(self):
        sched = StepDecaySchedule(base_lr=0.1, step_size=10, gamma=0.5)
        assert sched.get_lr(0) == pytest.approx(0.1)
        assert sched.get_lr(9) == pytest.approx(0.1)

    def test_after_step_size(self):
        sched = StepDecaySchedule(base_lr=0.1, step_size=10, gamma=0.5)
        assert sched.get_lr(10) == pytest.approx(0.05)

    def test_min_lr_clamp(self):
        sched = StepDecaySchedule(base_lr=0.1, step_size=1, gamma=0.1, min_lr=0.001)
        # After many steps, gamma^n gets very small
        assert sched.get_lr(100) == pytest.approx(0.001)


class TestCreateScheduleFactory:
    def test_constant_produces_flat_lr(self):
        """Factory CONSTANT schedule returns same LR at all steps."""
        cfg = ScheduleConfig(schedule_type=ScheduleType.CONSTANT, base_lr=0.01)
        sched = create_schedule(cfg)
        assert sched.get_lr(0) == sched.get_lr(500) == 0.01

    def test_linear_decays_over_time(self):
        """Factory LINEAR schedule LR decreases from base to min."""
        cfg = ScheduleConfig(
            schedule_type=ScheduleType.LINEAR, base_lr=0.01,
            total_steps=100, min_lr=0.001,
        )
        sched = create_schedule(cfg)
        assert sched.get_lr(0) > sched.get_lr(50) > sched.get_lr(100)
        assert sched.get_lr(100) == pytest.approx(0.001)

    def test_step_decays_at_boundary(self):
        """Factory STEP schedule multiplies by gamma at each step_size."""
        cfg = ScheduleConfig(
            schedule_type=ScheduleType.STEP, base_lr=0.1,
            step_size=10, gamma=0.5,
        )
        sched = create_schedule(cfg)
        assert sched.get_lr(9) == pytest.approx(0.1)
        assert sched.get_lr(10) == pytest.approx(0.05)

    def test_unknown_type_raises(self):
        """Factory raises ValueError for unknown schedule type."""
        cfg = ScheduleConfig(schedule_type=ScheduleType.CONSTANT)
        # Manually set to something invalid
        cfg.schedule_type = "bogus"
        with pytest.raises(ValueError, match="Unknown schedule type"):
            create_schedule(cfg)


class TestIdleTrainingScheduler:
    def test_enabled_scheduler_can_train_after_idle(self):
        """Enabled scheduler → can eventually train after idle period."""
        config = IdleSchedulerConfig(min_idle_seconds=0.0)
        scheduler = CoreIdleTrainingScheduler(config)
        scheduler._last_activity_time = time.time() - 1.0
        assert scheduler.should_train() is True

    def test_disabled_scheduler_never_trains(self):
        """Disabled scheduler → should_train always False regardless of idle time."""
        config = IdleSchedulerConfig(enabled=False, min_idle_seconds=0.0)
        scheduler = CoreIdleTrainingScheduler(config)
        scheduler._last_activity_time = 0.0  # infinitely idle
        assert scheduler.should_train() is False

    def test_on_activity_pauses_training(self):
        scheduler = CoreIdleTrainingScheduler()
        scheduler._state = IdleTrainingState.TRAINING
        scheduler.on_activity()
        assert scheduler.state == IdleTrainingState.PAUSED

    def test_should_train_when_idle(self):
        config = IdleSchedulerConfig(min_idle_seconds=0.0)
        scheduler = CoreIdleTrainingScheduler(config)
        # Set last activity to past
        scheduler._last_activity_time = time.time() - 1.0
        assert scheduler.should_train() is True
        assert scheduler.state == IdleTrainingState.TRAINING

    def test_should_not_train_when_disabled(self):
        config = IdleSchedulerConfig(enabled=False)
        scheduler = CoreIdleTrainingScheduler(config)
        assert scheduler.should_train() is False

    def test_step_complete_resets_after_max_batch(self):
        config = IdleSchedulerConfig(max_batch_steps=3)
        scheduler = CoreIdleTrainingScheduler(config)
        scheduler._state = IdleTrainingState.TRAINING

        scheduler.on_step_complete()
        scheduler.on_step_complete()
        assert scheduler.state == IdleTrainingState.TRAINING

        scheduler.on_step_complete()  # 3rd step → resets
        assert scheduler.state == IdleTrainingState.WAITING

    def test_enable_disable(self):
        scheduler = CoreIdleTrainingScheduler()
        scheduler.disable()
        assert scheduler.state == IdleTrainingState.DISABLED

        scheduler.enable()
        assert scheduler.state == IdleTrainingState.IDLE

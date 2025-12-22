"""
Learning Rate Scheduling for Training.

Ported from the reference Swift implementation.

Supported schedules:
- Constant
- Linear warmup + decay
- Cosine annealing
- Warmup + cosine
- Step decay

Usage:
    schedule = CosineSchedule(base_lr=3e-5, total_steps=1000, warmup_steps=100)
    for step in range(1000):
        lr = schedule.get_lr(step)
        optimizer.learning_rate = lr
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScheduleType(str, Enum):
    """Learning rate schedule types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"
    STEP = "step"


class LRSchedule(ABC):
    """Abstract base for learning rate schedules."""

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step."""
        pass

    @property
    @abstractmethod
    def base_lr(self) -> float:
        """Base learning rate."""
        pass


class ConstantSchedule(LRSchedule):
    """Constant learning rate."""

    def __init__(self, lr: float):
        self._lr = lr

    def get_lr(self, step: int) -> float:
        return self._lr

    @property
    def base_lr(self) -> float:
        return self._lr


class LinearWarmupSchedule(LRSchedule):
    """Linear warmup from 0 to base_lr, then constant."""

    def __init__(self, base_lr: float, warmup_steps: int):
        self._base_lr = base_lr
        self.warmup_steps = max(warmup_steps, 1)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self._base_lr * (step + 1) / self.warmup_steps
        return self._base_lr

    @property
    def base_lr(self) -> float:
        return self._base_lr


class LinearDecaySchedule(LRSchedule):
    """Linear warmup then linear decay to min_lr."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        self._base_lr = base_lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = min(warmup_steps, total_steps - 1)
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self._base_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        decay_ratio = 1.0 - (decay_step / decay_steps)
        return self.min_lr + (self._base_lr - self.min_lr) * decay_ratio

    @property
    def base_lr(self) -> float:
        return self._base_lr


class CosineSchedule(LRSchedule):
    """Cosine annealing with optional warmup."""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ):
        self._base_lr = base_lr
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = min(warmup_steps, total_steps - 1)
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self._base_lr * (step + 1) / self.warmup_steps

        if step >= self.total_steps:
            return self.min_lr

        decay_steps = self.total_steps - self.warmup_steps
        decay_step = step - self.warmup_steps
        progress = decay_step / decay_steps

        # Cosine annealing: lr = min + 0.5 * (max - min) * (1 + cos(π * progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self._base_lr - self.min_lr) * cosine_decay

    @property
    def base_lr(self) -> float:
        return self._base_lr


class StepDecaySchedule(LRSchedule):
    """Step decay: multiply by gamma every step_size steps."""

    def __init__(
        self,
        base_lr: float,
        step_size: int,
        gamma: float = 0.1,
        min_lr: float = 0.0,
    ):
        self._base_lr = base_lr
        self.step_size = max(step_size, 1)
        self.gamma = gamma
        self.min_lr = min_lr

    def get_lr(self, step: int) -> float:
        num_decays = step // self.step_size
        lr = self._base_lr * (self.gamma ** num_decays)
        return max(lr, self.min_lr)

    @property
    def base_lr(self) -> float:
        return self._base_lr


# =============================================================================
# Factory
# =============================================================================

@dataclass
class ScheduleConfig:
    """Configuration for creating a schedule."""
    schedule_type: ScheduleType = ScheduleType.COSINE
    base_lr: float = 3e-5
    total_steps: int = 1000
    warmup_steps: int = 0
    min_lr: float = 0.0
    step_size: int = 100  # For step decay
    gamma: float = 0.1    # For step decay


def create_schedule(config: ScheduleConfig) -> LRSchedule:
    """Create a learning rate schedule from configuration."""
    if config.schedule_type == ScheduleType.CONSTANT:
        return ConstantSchedule(config.base_lr)

    elif config.schedule_type == ScheduleType.LINEAR:
        return LinearDecaySchedule(
            base_lr=config.base_lr,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

    elif config.schedule_type == ScheduleType.COSINE:
        return CosineSchedule(
            base_lr=config.base_lr,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

    elif config.schedule_type == ScheduleType.WARMUP_COSINE:
        # Alias for cosine with warmup
        return CosineSchedule(
            base_lr=config.base_lr,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_lr,
        )

    elif config.schedule_type == ScheduleType.STEP:
        return StepDecaySchedule(
            base_lr=config.base_lr,
            step_size=config.step_size,
            gamma=config.gamma,
            min_lr=config.min_lr,
        )

    else:
        raise ValueError(f"Unknown schedule type: {config.schedule_type}")


# =============================================================================
# Idle Training Scheduler (Background Training)
# =============================================================================

@dataclass
class IdleSchedulerConfig:
    """Configuration for idle-time background training."""
    enabled: bool = True
    min_idle_seconds: float = 60.0  # Wait before starting
    max_batch_steps: int = 10       # Steps per idle session
    pause_on_activity: bool = True  # Pause when user active


class IdleTrainingState(str, Enum):
    """State of idle training scheduler."""
    IDLE = "idle"
    WAITING = "waiting"
    TRAINING = "training"
    PAUSED = "paused"
    DISABLED = "disabled"


class IdleTrainingScheduler:
    """
    Schedules training during system idle time.

    Monitors system activity and runs training batches
    when the system has been idle for a threshold period.
    """

    def __init__(self, config: Optional[IdleSchedulerConfig] = None):
        self.config = config or IdleSchedulerConfig()
        self._state = IdleTrainingState.IDLE if self.config.enabled else IdleTrainingState.DISABLED
        self._last_activity_time = 0.0
        self._accumulated_steps = 0

    @property
    def state(self) -> IdleTrainingState:
        return self._state

    def on_activity(self):
        """Called when system activity is detected."""
        import time
        self._last_activity_time = time.time()

        if self._state == IdleTrainingState.TRAINING and self.config.pause_on_activity:
            self._state = IdleTrainingState.PAUSED

    def should_train(self) -> bool:
        """Check if we should start a training batch."""
        if not self.config.enabled:
            return False

        import time
        idle_time = time.time() - self._last_activity_time

        if idle_time >= self.config.min_idle_seconds:
            self._state = IdleTrainingState.TRAINING
            return True

        return False

    def on_step_complete(self):
        """Called after each training step."""
        self._accumulated_steps += 1

        if self._accumulated_steps >= self.config.max_batch_steps:
            self._state = IdleTrainingState.WAITING
            self._accumulated_steps = 0

    def enable(self):
        """Enable idle training."""
        self.config.enabled = True
        self._state = IdleTrainingState.IDLE

    def disable(self):
        """Disable idle training."""
        self.config.enabled = False
        self._state = IdleTrainingState.DISABLED

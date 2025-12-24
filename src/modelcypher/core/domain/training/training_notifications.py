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

"""Training notification system.

Provides event-based notifications for training progress updates.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
from uuid import uuid4


class TrainingEventKind(str, Enum):
    """Kind of training event."""

    PROGRESS = "progress"
    """Progress update during training."""

    STARTED = "started"
    """Training job started."""

    COMPLETED = "completed"
    """Training job completed successfully."""

    FAILED = "failed"
    """Training job failed."""

    CANCELLED = "cancelled"
    """Training job cancelled."""

    CHECKPOINT_SAVED = "checkpoint_saved"
    """Checkpoint was saved."""

    MEMORY_WARNING = "memory_warning"
    """Memory pressure warning."""


@dataclass
class TrainingProgress:
    """Training progress information."""

    job_id: str
    """Training job identifier."""

    step: int
    """Current training step."""

    total_steps: int
    """Total steps in training job."""

    loss: float | None = None
    """Current loss value."""

    learning_rate: float | None = None
    """Current learning rate."""

    tokens_per_second: float | None = None
    """Current throughput."""

    memory_usage_gb: float | None = None
    """Current memory usage in GB."""

    @property
    def progress_fraction(self) -> float:
        """Progress as fraction (0.0-1.0)."""
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, self.step / self.total_steps)

    @property
    def progress_percent(self) -> int:
        """Progress as percentage (0-100)."""
        return int(self.progress_fraction * 100)


@dataclass
class TrainingEvent:
    """Training event payload."""

    kind: TrainingEventKind
    """Kind of event."""

    job_id: str
    """Training job identifier."""

    progress: TrainingProgress | None = None
    """Progress information (for PROGRESS events)."""

    message: str | None = None
    """Optional message."""

    error: str | None = None
    """Error message (for FAILED events)."""

    checkpoint_step: int | None = None
    """Checkpoint step (for CHECKPOINT_SAVED events)."""


# Type for event handlers
TrainingEventHandler = Callable[[TrainingEvent], None]
AsyncTrainingEventHandler = Callable[[TrainingEvent], Any]


class TrainingEventBus:
    """Event bus for training notifications.

    Provides both synchronous and async event handling.
    """

    def __init__(self):
        """Initialize event bus."""
        self._handlers: dict[str, TrainingEventHandler] = {}
        self._async_handlers: dict[str, AsyncTrainingEventHandler] = {}
        self._queues: dict[str, asyncio.Queue[TrainingEvent]] = {}

    def subscribe(self, handler: TrainingEventHandler) -> str:
        """Subscribe to training events.

        Args:
            handler: Callback function for events.

        Returns:
            Subscription ID for unsubscribing.
        """
        sub_id = str(uuid4())
        self._handlers[sub_id] = handler
        return sub_id

    def subscribe_async(self, handler: AsyncTrainingEventHandler) -> str:
        """Subscribe to training events with async handler.

        Args:
            handler: Async callback function for events.

        Returns:
            Subscription ID for unsubscribing.
        """
        sub_id = str(uuid4())
        self._async_handlers[sub_id] = handler
        return sub_id

    def create_event_queue(self) -> tuple[str, asyncio.Queue[TrainingEvent]]:
        """Create an async queue for receiving events.

        Returns:
            Tuple of (queue_id, queue).
        """
        queue_id = str(uuid4())
        queue: asyncio.Queue[TrainingEvent] = asyncio.Queue()
        self._queues[queue_id] = queue
        return queue_id, queue

    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from training events.

        Args:
            subscription_id: ID returned from subscribe.
        """
        self._handlers.pop(subscription_id, None)
        self._async_handlers.pop(subscription_id, None)
        self._queues.pop(subscription_id, None)

    def emit(self, event: TrainingEvent) -> None:
        """Emit a training event to all subscribers.

        Args:
            event: Event to emit.
        """
        # Call sync handlers
        for handler in list(self._handlers.values()):
            try:
                handler(event)
            except Exception:
                pass  # Don't let handler errors stop other handlers

        # Queue for async consumers
        for queue in list(self._queues.values()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if full

    async def emit_async(self, event: TrainingEvent) -> None:
        """Emit a training event asynchronously.

        Args:
            event: Event to emit.
        """
        # Call sync handlers
        self.emit(event)

        # Call async handlers
        for handler in list(self._async_handlers.values()):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Don't let handler errors stop other handlers

    def emit_progress(
        self,
        job_id: str,
        step: int,
        total_steps: int,
        loss: float | None = None,
        learning_rate: float | None = None,
        tokens_per_second: float | None = None,
        memory_usage_gb: float | None = None,
    ) -> None:
        """Convenience method to emit a progress event.

        Args:
            job_id: Training job identifier.
            step: Current training step.
            total_steps: Total steps in training job.
            loss: Current loss value.
            learning_rate: Current learning rate.
            tokens_per_second: Current throughput.
            memory_usage_gb: Current memory usage in GB.
        """
        progress = TrainingProgress(
            job_id=job_id,
            step=step,
            total_steps=total_steps,
            loss=loss,
            learning_rate=learning_rate,
            tokens_per_second=tokens_per_second,
            memory_usage_gb=memory_usage_gb,
        )
        event = TrainingEvent(
            kind=TrainingEventKind.PROGRESS,
            job_id=job_id,
            progress=progress,
        )
        self.emit(event)


# Global event bus instance
_default_event_bus: TrainingEventBus | None = None


def get_training_event_bus() -> TrainingEventBus:
    """Get the default training event bus.

    Returns:
        Default event bus instance.
    """
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = TrainingEventBus()
    return _default_event_bus


def reset_training_event_bus() -> None:
    """Reset the default training event bus (for testing)."""
    global _default_event_bus
    _default_event_bus = None

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
Training Resource Guard for Exclusive GPU Access.

Enhanced port from the reference Swift implementation.

Features:
- Exclusive GPU access serialization
- Async activity subscribers
- Stale session watchdog
- Training and inference session management
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ResourceIntensiveOperation(str, Enum):
    """Operations requiring GPU resources."""

    RAG_INDEXING = "RAG Indexing"
    RAG_QUERY = "RAG Query"
    MODEL_INFERENCE = "Model Inference"

    @property
    def unavailable_message(self) -> str:
        return f"{self.value} Unavailable: Training is currently using the GPU."


class InferenceOwner(str, Enum):
    """Owners of inference resources."""

    USER = "user"
    COMPARISON_SESSION = "comparison_session"


@dataclass
class TrainingSessionInfo:
    """Information about an active training session."""

    job_id: str
    start_time: float
    duration: float

    @property
    def formatted_duration(self) -> str:
        minutes = int(self.duration) // 60
        seconds = int(self.duration) % 60
        return f"{minutes}:{seconds:02d}"


class TrainingReleaseReason(str, Enum):
    """Reason for training session release."""

    NORMAL = "normal"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    CANCELLED = "cancelled"


@dataclass
class TrainingActivityState:
    """Broadcast payload for training activity changes."""

    is_training: bool
    active_job_id: str | None = None
    termination_reason: TrainingReleaseReason | None = None


@dataclass
class WorkloadActivityState:
    """Broadcast payload for GPU workload activity (training OR inference)."""

    is_active: bool
    training_job_id: str | None = None
    inference_owner: str | None = None


class ResourceError(Exception):
    """Resource access error."""

    pass


class TrainingInProgressError(ResourceError):
    """Training is in progress."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"Training job {job_id} is already running.")


class InferenceInProgressError(ResourceError):
    """Inference is in progress."""

    def __init__(self, owner: str):
        self.owner = owner
        super().__init__(f"Inference in progress by: {owner}")


# =============================================================================
# Async Activity Stream
# =============================================================================


class ActivitySubscriber:
    """
    Async subscriber for activity state changes.

    Usage:
        async for state in subscriber:
            print(f"Training active: {state.is_training}")
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active = True
        self.id = str(uuid.uuid4())

    def emit(self, state: Any):
        """Emit state to subscriber (non-blocking)."""
        if self._active:
            try:
                self._queue.put_nowait(state)
            except asyncio.QueueFull:
                pass  # Drop if full

    def close(self):
        """Close the subscriber."""
        self._active = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._active:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            if not self._active:
                raise StopAsyncIteration
            return await self.__anext__()


# =============================================================================
# Training Resource Guard
# =============================================================================


class TrainingResourceGuard:
    """
    Singleton enforcing exclusive GPU access across training, RAG, and inference.

    Replicates Swift's Actor isolation using asyncio.Lock with:
    - Async activity subscribers
    - Stale session watchdog
    - Training/inference session management
    """

    _instance: "TrainingResourceGuard" | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._lock = asyncio.Lock()
        self._active_training_job_id: str | None = None
        self._training_start_time: float | None = None

        # Inference owners
        self._active_inference_owners: set[str] = set()
        self._max_concurrent_inference_owners = 2

        # Subscribers
        self._training_activity_subscribers: dict[str, ActivitySubscriber] = {}
        self._workload_activity_subscribers: dict[str, ActivitySubscriber] = {}

        # Watchdog
        self._max_training_duration = 24 * 3600  # 24 hours
        self._watchdog_poll_interval = 5 * 60  # 5 minutes
        self._watchdog_task: asyncio.Task | None = None

        self._initialized = True

    @classmethod
    def shared(cls) -> "TrainingResourceGuard":
        """Get the shared singleton instance."""
        return cls()

    @classmethod
    def reset_for_testing(cls):
        """Reset singleton for testing."""
        if cls._instance:
            cls._instance._initialized = False
            cls._instance = None

    # =========================================================================
    # Training Control
    # =========================================================================

    @property
    def is_training_active(self) -> bool:
        return self._active_training_job_id is not None

    async def get_current_training_session(self) -> TrainingSessionInfo | None:
        async with self._lock:
            if not self._active_training_job_id or not self._training_start_time:
                return None

            duration = time.time() - self._training_start_time
            return TrainingSessionInfo(
                job_id=self._active_training_job_id,
                start_time=self._training_start_time,
                duration=duration,
            )

    async def begin_training(self, job_id: str):
        """Begin a training session - locks resources."""
        async with self._lock:
            if self._active_inference_owners:
                owner = next(iter(self._active_inference_owners))
                raise InferenceInProgressError(owner)

            if self._active_training_job_id:
                raise TrainingInProgressError(self._active_training_job_id)

            self._active_training_job_id = job_id
            self._training_start_time = time.time()
            self._broadcast_training_activity()
            self._broadcast_workload_activity()
            self._start_watchdog()

    async def end_training(
        self, job_id: str, reason: TrainingReleaseReason = TrainingReleaseReason.NORMAL
    ):
        """End a training session - releases resources."""
        async with self._lock:
            if self._active_training_job_id == job_id:
                self._active_training_job_id = None
                self._training_start_time = None
                self._stop_watchdog()
                self._broadcast_training_activity(reason)
                self._broadcast_workload_activity()

    @asynccontextmanager
    async def training_session(self, job_id: str):
        """Context manager for safe training session resource usage."""
        await self.begin_training(job_id)
        try:
            yield
        except asyncio.CancelledError:
            await self.end_training(job_id, TrainingReleaseReason.CANCELLED)
            raise
        except Exception:
            await self.end_training(job_id)
            raise
        else:
            await self.end_training(job_id)

    # =========================================================================
    # Inference Control
    # =========================================================================

    async def begin_inference(self, owner: str):
        """Begin an inference session."""
        async with self._lock:
            if self._active_training_job_id:
                raise ResourceError(f"Training job {self._active_training_job_id} is active.")

            if owner in self._active_inference_owners:
                return

            if len(self._active_inference_owners) >= self._max_concurrent_inference_owners:
                raise ResourceError("Maximum concurrent inference sessions reached.")

            self._active_inference_owners.add(owner)
            self._broadcast_workload_activity()

    async def end_inference(self, owner: str):
        """End an inference session."""
        async with self._lock:
            if owner in self._active_inference_owners:
                self._active_inference_owners.remove(owner)
                self._broadcast_workload_activity()

    @asynccontextmanager
    async def inference_session(self, owner: str):
        """Context manager for inference session."""
        await self.begin_inference(owner)
        try:
            yield
        finally:
            await self.end_inference(owner)

    # =========================================================================
    # Resource Access Control
    # =========================================================================

    async def request_resource_access(self, operation: ResourceIntensiveOperation):
        """Request permission for GPU-intensive operation."""
        async with self._lock:
            if self._active_training_job_id:
                raise ResourceError(operation.unavailable_message)

    async def can_perform_operation(self, operation: ResourceIntensiveOperation) -> bool:
        """Check if an operation is allowed."""
        async with self._lock:
            if operation == ResourceIntensiveOperation.MODEL_INFERENCE:
                return (
                    self._active_training_job_id is None
                    and len(self._active_inference_owners) < self._max_concurrent_inference_owners
                )
            return self._active_training_job_id is None

    async def is_workload_active(self) -> bool:
        """Check if any GPU-intensive workload is active."""
        async with self._lock:
            return (
                self._active_training_job_id is not None or len(self._active_inference_owners) > 0
            )

    # =========================================================================
    # Activity Subscribers
    # =========================================================================

    def training_activity_updates(self) -> ActivitySubscriber:
        """
        Subscribe to training activity changes.

        Returns an async iterator that yields TrainingActivityState.
        """
        subscriber = ActivitySubscriber()
        self._training_activity_subscribers[subscriber.id] = subscriber

        # Emit current state immediately
        state = TrainingActivityState(
            is_training=self._active_training_job_id is not None,
            active_job_id=self._active_training_job_id,
        )
        subscriber.emit(state)

        return subscriber

    def workload_activity_updates(self) -> ActivitySubscriber:
        """
        Subscribe to workload activity changes.

        Returns an async iterator that yields WorkloadActivityState.
        """
        subscriber = ActivitySubscriber()
        self._workload_activity_subscribers[subscriber.id] = subscriber

        # Emit current state immediately
        state = WorkloadActivityState(
            is_active=(
                self._active_training_job_id is not None or len(self._active_inference_owners) > 0
            ),
            training_job_id=self._active_training_job_id,
            inference_owner=next(iter(self._active_inference_owners), None),
        )
        subscriber.emit(state)

        return subscriber

    def unsubscribe(self, subscriber: ActivitySubscriber):
        """Remove a subscriber."""
        subscriber.close()
        self._training_activity_subscribers.pop(subscriber.id, None)
        self._workload_activity_subscribers.pop(subscriber.id, None)

    def _broadcast_training_activity(self, reason: TrainingReleaseReason | None = None):
        """Broadcast training activity change to all subscribers."""
        state = TrainingActivityState(
            is_training=self._active_training_job_id is not None,
            active_job_id=self._active_training_job_id,
            termination_reason=reason,
        )
        for subscriber in list(self._training_activity_subscribers.values()):
            subscriber.emit(state)

    def _broadcast_workload_activity(self):
        """Broadcast workload activity change to all subscribers."""
        state = WorkloadActivityState(
            is_active=(
                self._active_training_job_id is not None or len(self._active_inference_owners) > 0
            ),
            training_job_id=self._active_training_job_id,
            inference_owner=next(iter(self._active_inference_owners), None),
        )
        for subscriber in list(self._workload_activity_subscribers.values()):
            subscriber.emit(state)

    # =========================================================================
    # Watchdog
    # =========================================================================

    def _start_watchdog(self):
        """Start the stale session watchdog."""
        self._stop_watchdog()
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def _stop_watchdog(self):
        """Stop the watchdog task."""
        if self._watchdog_task:
            self._watchdog_task.cancel()
            self._watchdog_task = None

    async def _watchdog_loop(self):
        """Watchdog loop that checks for stale sessions."""
        while True:
            try:
                await asyncio.sleep(self._watchdog_poll_interval)
            except asyncio.CancelledError:
                break

            await self._check_and_release_stale_session()

            # Check if still active
            if self._active_training_job_id is None:
                break

    async def _check_and_release_stale_session(self):
        """Check for and release stale training sessions."""
        async with self._lock:
            if not self._active_training_job_id or not self._training_start_time:
                return

            elapsed = time.time() - self._training_start_time
            if elapsed > self._max_training_duration:
                job_id = self._active_training_job_id
                self._active_training_job_id = None
                self._training_start_time = None
                self._stop_watchdog()
                self._broadcast_training_activity(TrainingReleaseReason.WATCHDOG_TIMEOUT)
                self._broadcast_workload_activity()
                logger.warning(
                    "Watchdog: Released stale training session %s after %.0fs", job_id, elapsed
                )

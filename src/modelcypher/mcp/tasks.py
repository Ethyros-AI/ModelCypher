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
MCP Tasks Framework for Async Operations.

Implements MCP 2025-11-25 Tasks spec for long-running operations:
- Training jobs
- Model merges  
- Batch inference
- Large dataset operations

Features:
- Task submission with immediate ID return
- Status polling with progress updates
- Task cancellation
- Result retrieval
- Automatic cleanup of completed tasks

Usage:
    from modelcypher.mcp.tasks import TaskManager, TaskStatus

    # Create task
    task_id = task_manager.create("train", config)
    
    # Check status
    status = task_manager.get_status(task_id)
    
    # Cancel if needed
    task_manager.cancel(task_id)
    
    # Get result when complete
    result = task_manager.get_result(task_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of async tasks."""
    TRAINING = "training"
    MERGE = "merge"
    INFERENCE_BATCH = "inference_batch"
    DATASET_PROCESS = "dataset_process"
    EVALUATION = "evaluation"


@dataclass
class TaskProgress:
    """Progress update for a running task."""
    current_step: int
    total_steps: int
    message: str
    percentage: float = field(init=False)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.percentage = (self.current_step / max(self.total_steps, 1)) * 100


@dataclass
class Task:
    """Represents an async task."""
    id: str
    type: TaskType
    status: TaskStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: TaskProgress | None = None
    result: Any | None = None
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float | None:
        """Duration of task execution."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "createdAt": self.created_at.isoformat(),
            "startedAt": self.started_at.isoformat() if self.started_at else None,
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
            "durationSeconds": self.duration_seconds,
            "progress": {
                "currentStep": self.progress.current_step,
                "totalSteps": self.progress.total_steps,
                "percentage": self.progress.percentage,
                "message": self.progress.message,
                "metrics": self.progress.metrics,
            } if self.progress else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class TaskManager:
    """
    Manages async tasks for MCP server.
    
    Thread-safe task management with automatic cleanup.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        task_ttl_seconds: int = 3600,  # 1 hour
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._task_ttl = task_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._running = True
        self._cancellation_events: dict[str, threading.Event] = {}
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def create(
        self,
        task_type: TaskType,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new pending task."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = Task(
            id=task_id,
            type=task_type,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            config=config,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._tasks[task_id] = task
            self._cancellation_events[task_id] = threading.Event()
        
        logger.info(f"Created task {task_id} of type {task_type.value}")
        return task_id
    
    def submit(
        self,
        task_id: str,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> None:
        """Submit a task for execution."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status != TaskStatus.PENDING:
                raise ValueError(f"Task {task_id} is not pending")
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
        
        # Submit to thread pool
        future = self._executor.submit(
            self._execute_task,
            task_id,
            func,
            args,
            kwargs,
        )
        
        logger.info(f"Submitted task {task_id} for execution")
    
    def _execute_task(
        self,
        task_id: str,
        func: Callable[..., Any],
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Execute a task in the thread pool."""
        try:
            # Add progress callback and cancellation check to kwargs
            kwargs["_progress_callback"] = lambda p: self.update_progress(task_id, p)
            kwargs["_should_cancel"] = lambda: self.is_cancelled(task_id)
            
            result = func(*args, **kwargs)
            
            with self._lock:
                task = self._tasks.get(task_id)
                if task and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result
                    logger.info(f"Task {task_id} completed successfully")
                    
        except Exception as e:
            with self._lock:
                task = self._tasks.get(task_id)
                if task:
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    task.error = str(e)
                    logger.error(f"Task {task_id} failed: {e}")
    
    def update_progress(self, task_id: str, progress: TaskProgress) -> None:
        """Update task progress."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.RUNNING:
                task.progress = progress
    
    def is_cancelled(self, task_id: str) -> bool:
        """Check if task should be cancelled."""
        event = self._cancellation_events.get(task_id)
        return event.is_set() if event else False
    
    def cancel(self, task_id: str) -> bool:
        """Request task cancellation."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False
            
            # Set cancellation event
            event = self._cancellation_events.get(task_id)
            if event:
                event.set()
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            logger.info(f"Task {task_id} cancelled")
            return True
    
    def get_status(self, task_id: str) -> Task | None:
        """Get task status."""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_result(self, task_id: str) -> Any | None:
        """Get task result if completed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            return None
    
    def list_tasks(
        self,
        task_type: TaskType | None = None,
        status: TaskStatus | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """List tasks with optional filtering."""
        with self._lock:
            tasks = list(self._tasks.values())
        
        if task_type:
            tasks = [t for t in tasks if t.type == task_type]
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
    
    def delete_task(self, task_id: str) -> bool:
        """Delete a completed/failed/cancelled task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return False  # Can't delete active tasks
            
            del self._tasks[task_id]
            self._cancellation_events.pop(task_id, None)
            logger.info(f"Deleted task {task_id}")
            return True
    
    def _cleanup_loop(self) -> None:
        """Periodically clean up old completed tasks."""
        while self._running:
            time.sleep(self._cleanup_interval)
            self._cleanup_old_tasks()
    
    def _cleanup_old_tasks(self) -> None:
        """Remove tasks older than TTL."""
        now = datetime.now()
        to_delete = []
        
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    if task.completed_at:
                        age = (now - task.completed_at).total_seconds()
                        if age > self._task_ttl:
                            to_delete.append(task_id)
            
            for task_id in to_delete:
                del self._tasks[task_id]
                self._cancellation_events.pop(task_id, None)
        
        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} old tasks")
    
    def shutdown(self) -> None:
        """Shutdown the task manager."""
        self._running = False
        self._executor.shutdown(wait=False)


# Global task manager instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


__all__ = [
    "TaskManager",
    "TaskStatus",
    "TaskType",
    "TaskProgress",
    "Task",
    "get_task_manager",
]

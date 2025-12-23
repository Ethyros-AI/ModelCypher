"""
MCP Task Management Tools.

Provides MCP tools for managing async tasks:
- mc_task_list: List all tasks
- mc_task_status: Get task status
- mc_task_cancel: Cancel a running task
- mc_task_result: Get task result
- mc_task_delete: Delete completed task
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from modelcypher.mcp.tasks import (
    TaskManager,
    TaskStatus,
    TaskType,
    get_task_manager,
)


READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

DESTRUCTIVE_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
    "openWorldHint": False,
}


def register_task_tools(mcp: FastMCP, tool_set: set[str]) -> None:
    """Register task management MCP tools."""
    
    if "mc_task_list" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_task_list(
            taskType: str | None = None,
            status: str | None = None,
            limit: int = 50,
        ) -> Dict[str, Any]:
            """
            List async tasks with optional filtering.
            
            Args:
                taskType: Filter by type (training, merge, inference_batch, dataset_process, evaluation)
                status: Filter by status (pending, running, completed, failed, cancelled)
                limit: Maximum number of tasks to return (default 50)
            
            Returns:
                List of tasks with status information
            """
            manager = get_task_manager()
            
            # Parse filters
            type_filter = TaskType(taskType) if taskType else None
            status_filter = TaskStatus(status) if status else None
            
            tasks = manager.list_tasks(
                task_type=type_filter,
                status=status_filter,
                limit=limit,
            )
            
            # Count by status
            running = sum(1 for t in tasks if t.status == TaskStatus.RUNNING)
            pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
            completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
            
            return {
                "_schema": "mc_task_list_response",
                "tasks": [t.to_dict() for t in tasks],
                "count": len(tasks),
                "summary": {
                    "running": running,
                    "pending": pending,
                    "completed": completed,
                    "failed": failed,
                },
                "nextActions": [
                    "mc_task_status taskId=<id> for details",
                    "mc_task_cancel taskId=<id> to cancel running task",
                ],
            }
    
    if "mc_task_status" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_task_status(taskId: str) -> Dict[str, Any]:
            """
            Get detailed status of an async task.
            
            Args:
                taskId: The task ID to check
            
            Returns:
                Task status with progress information
            """
            manager = get_task_manager()
            task = manager.get_status(taskId)
            
            if task is None:
                return {
                    "_schema": "error",
                    "error": f"Task not found: {taskId}",
                    "nextActions": ["mc_task_list to see available tasks"],
                }
            
            # Build next actions based on status
            next_actions = []
            if task.status == TaskStatus.RUNNING:
                next_actions = [
                    f"mc_task_status taskId={taskId} to refresh",
                    f"mc_task_cancel taskId={taskId} to stop",
                ]
            elif task.status == TaskStatus.PENDING:
                next_actions = [
                    f"mc_task_status taskId={taskId} to check if started",
                    f"mc_task_cancel taskId={taskId} to cancel",
                ]
            elif task.status == TaskStatus.COMPLETED:
                next_actions = [
                    f"mc_task_result taskId={taskId} for full result",
                    f"mc_task_delete taskId={taskId} to clean up",
                ]
            elif task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                next_actions = [
                    f"mc_task_delete taskId={taskId} to clean up",
                    "mc_task_list for other tasks",
                ]
            
            return {
                "_schema": "mc_task_status_response",
                **task.to_dict(),
                "nextActions": next_actions,
            }
    
    if "mc_task_cancel" in tool_set:
        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_task_cancel(taskId: str) -> Dict[str, Any]:
            """
            Cancel a running or pending task.
            
            Args:
                taskId: The task ID to cancel
            
            Returns:
                Cancellation result
            """
            manager = get_task_manager()
            
            task = manager.get_status(taskId)
            if task is None:
                return {
                    "_schema": "error",
                    "error": f"Task not found: {taskId}",
                }
            
            if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return {
                    "_schema": "error",
                    "error": f"Cannot cancel task with status: {task.status.value}",
                    "nextActions": ["mc_task_list for active tasks"],
                }
            
            success = manager.cancel(taskId)
            
            return {
                "_schema": "mc_task_cancel_response",
                "taskId": taskId,
                "cancelled": success,
                "previousStatus": task.status.value,
                "nextActions": [
                    "mc_task_list for other tasks",
                    f"mc_task_delete taskId={taskId} to clean up",
                ],
            }
    
    if "mc_task_result" in tool_set:
        @mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
        def mc_task_result(taskId: str) -> Dict[str, Any]:
            """
            Get the result of a completed task.
            
            Args:
                taskId: The task ID to get result for
            
            Returns:
                Task result or error
            """
            manager = get_task_manager()
            task = manager.get_status(taskId)
            
            if task is None:
                return {
                    "_schema": "error",
                    "error": f"Task not found: {taskId}",
                }
            
            if task.status != TaskStatus.COMPLETED:
                return {
                    "_schema": "error",
                    "error": f"Task not completed, status: {task.status.value}",
                    "nextActions": [f"mc_task_status taskId={taskId} to check progress"],
                }
            
            return {
                "_schema": "mc_task_result_response",
                "taskId": taskId,
                "taskType": task.type.value,
                "durationSeconds": task.duration_seconds,
                "result": task.result,
                "nextActions": [
                    f"mc_task_delete taskId={taskId} to clean up",
                ],
            }
    
    if "mc_task_delete" in tool_set:
        @mcp.tool(annotations=DESTRUCTIVE_ANNOTATIONS)
        def mc_task_delete(taskId: str) -> Dict[str, Any]:
            """
            Delete a completed, failed, or cancelled task.
            
            Args:
                taskId: The task ID to delete
            
            Returns:
                Deletion result
            """
            manager = get_task_manager()
            
            task = manager.get_status(taskId)
            if task is None:
                return {
                    "_schema": "error",
                    "error": f"Task not found: {taskId}",
                }
            
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                return {
                    "_schema": "error",
                    "error": f"Cannot delete active task with status: {task.status.value}",
                    "nextActions": [f"mc_task_cancel taskId={taskId} first"],
                }
            
            success = manager.delete_task(taskId)
            
            return {
                "_schema": "mc_task_delete_response",
                "taskId": taskId,
                "deleted": success,
                "nextActions": ["mc_task_list for remaining tasks"],
            }


__all__ = ["register_task_tools"]

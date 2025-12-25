"""Task coordination (December 2025).

Consolidates task-related functionality from:
- task_coordinator.py (task lifecycle)
- task_decorators.py (task decorators)

This module re-exports all task-related APIs for unified access.

Usage:
    from app.coordination.core.tasks import (
        TaskCoordinator,
        CoordinatedTask,
        TaskInfo,
    )
"""

from __future__ import annotations

# Re-export from task_coordinator
from app.coordination.task_coordinator import (
    TaskCoordinator,
    CoordinatedTask,
    TaskInfo,
    TaskHeartbeatMonitor,
    CoordinatorState,
    BackpressureLevel,
    QueueType,
    ResourceType,
    RateLimiter,
)

# Re-export from task_decorators
from app.coordination.task_decorators import (
    coordinate_task,
    coordinate_async_task,
    TaskContext,
    get_current_task_context,
)

__all__ = [
    # From task_coordinator
    "TaskCoordinator",
    "CoordinatedTask",
    "TaskInfo",
    "TaskHeartbeatMonitor",
    "CoordinatorState",
    "BackpressureLevel",
    "QueueType",
    "ResourceType",
    "RateLimiter",
    # From task_decorators
    "coordinate_task",
    "coordinate_async_task",
    "TaskContext",
    "get_current_task_context",
]

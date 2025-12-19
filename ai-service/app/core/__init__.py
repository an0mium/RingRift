"""Core shared infrastructure for the RingRift AI service.

This package provides standardized utilities used across all scripts:
- logging_config: Unified logging setup
- error_handler: Retry decorators, error recovery, emergency halt
- shutdown: Graceful shutdown coordination (December 2025)
"""

from app.core.logging_config import setup_logging, get_logger
from app.core.error_handler import (
    retry,
    retry_async,
    with_emergency_halt_check,
    RingRiftError,
    RetryableError,
    FatalError,
)
from app.core.shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    on_shutdown,
    request_shutdown,
    is_shutting_down,
    shutdown_scope,
)
from app.core.tasks import (
    background_task,
    TaskManager,
    get_task_manager,
    TaskInfo,
    TaskState,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Error handling
    "retry",
    "retry_async",
    "with_emergency_halt_check",
    "RingRiftError",
    "RetryableError",
    "FatalError",
    # Shutdown (December 2025)
    "ShutdownManager",
    "get_shutdown_manager",
    "on_shutdown",
    "request_shutdown",
    "is_shutting_down",
    "shutdown_scope",
    # Background tasks (December 2025)
    "background_task",
    "TaskManager",
    "get_task_manager",
    "TaskInfo",
    "TaskState",
]

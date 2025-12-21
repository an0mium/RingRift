"""Core shared infrastructure for the RingRift AI service.

This package provides standardized utilities used across all scripts:
- logging_config: Unified logging setup
- error_handler: Retry decorators, error recovery, emergency halt
- shutdown: Graceful shutdown coordination (December 2025)
- singleton_mixin: Thread-safe singleton patterns (December 2025)
- marshalling: Unified serialization patterns (December 2025)
"""

from app.core.error_handler import (
    FatalError,
    RetryableError,
    RingRiftError,
    retry,
    retry_async,
    with_emergency_halt_check,
)
from app.core.logging_config import get_logger, setup_logging
from app.core.shutdown import (
    ShutdownManager,
    get_shutdown_manager,
    is_shutting_down,
    on_shutdown,
    request_shutdown,
    shutdown_scope,
)
from app.core.singleton_mixin import (
    SingletonMeta,
    SingletonMixin,
    ThreadSafeSingletonMixin,
    singleton,
)
from app.core.tasks import (
    TaskInfo,
    TaskManager,
    TaskState,
    background_task,
    get_task_manager,
)

# Marshalling/Serialization (December 2025)
from app.core.marshalling import (
    Codec,
    Serializable,
    SerializationError,
    deserialize,
    from_json,
    register_codec,
    serialize,
    to_json,
)

__all__ = [
    # Marshalling/Serialization (December 2025)
    "Codec",
    "Serializable",
    "SerializationError",
    "deserialize",
    "from_json",
    "register_codec",
    "serialize",
    "to_json",
    # Error handling
    "FatalError",
    "RetryableError",
    "RingRiftError",
    "retry",
    "retry_async",
    "with_emergency_halt_check",
    # Shutdown (December 2025)
    "ShutdownManager",
    "get_shutdown_manager",
    "is_shutting_down",
    "on_shutdown",
    "request_shutdown",
    "shutdown_scope",
    # Singleton patterns (December 2025)
    "SingletonMeta",
    "SingletonMixin",
    "ThreadSafeSingletonMixin",
    "singleton",
    # Background tasks (December 2025)
    "TaskInfo",
    "TaskManager",
    "TaskState",
    "background_task",
    "get_task_manager",
    # Logging
    "get_logger",
    "setup_logging",
]

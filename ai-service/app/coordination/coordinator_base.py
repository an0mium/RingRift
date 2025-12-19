"""Base classes and protocols for coordinators.

This module provides a unified base for all coordinator/manager classes in the
system, consolidating common patterns:
- SQLite persistence with thread-local connections
- State management (running, status)
- Dependency injection via setters
- Async locking for concurrency control
- Stats/metrics reporting
- Lifecycle management (initialize, start, stop, shutdown)

Usage:
    from app.coordination.coordinator_base import (
        CoordinatorBase,
        CoordinatorProtocol,
        SQLitePersistenceMixin,
    )

    class MyCoordinator(CoordinatorBase, SQLitePersistenceMixin):
        def __init__(self, db_path: Path):
            super().__init__()
            self.init_db(db_path)

        def _get_schema(self) -> str:
            return '''CREATE TABLE IF NOT EXISTS my_table (...)'''

        async def get_stats(self) -> Dict[str, Any]:
            return {"status": self.status, "count": self._count}

See: docs/CONSOLIDATION_ROADMAP.md for consolidation context.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class CoordinatorStatus(str, Enum):
    """Common status values for coordinators."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class CoordinatorStats:
    """Base statistics for coordinators.

    Subclasses can extend this with additional fields.
    """
    status: CoordinatorStatus = CoordinatorStatus.INITIALIZING
    uptime_seconds: float = 0.0
    operations_count: int = 0
    errors_count: int = 0
    last_operation_time: float = 0.0
    last_error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "operations_count": self.operations_count,
            "errors_count": self.errors_count,
            "last_operation_time": self.last_operation_time,
            "last_error": self.last_error,
            **self.extra,
        }


@runtime_checkable
class CoordinatorProtocol(Protocol):
    """Protocol for coordinator classes.

    This defines the common interface that all coordinators should implement.
    Using Protocol allows structural subtyping - classes that have these methods
    are considered coordinators even without explicit inheritance.
    """

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        ...

    @property
    def is_running(self) -> bool:
        """Whether the coordinator is actively running."""
        ...

    async def initialize(self) -> None:
        """Initialize the coordinator (async setup)."""
        ...

    async def start(self) -> None:
        """Start the coordinator's main operation."""
        ...

    async def stop(self) -> None:
        """Stop the coordinator gracefully."""
        ...

    async def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        ...


class CoordinatorBase(ABC):
    """Base class for coordinators with common functionality.

    Provides:
    - Status and running state management
    - Async lock for concurrency control
    - Lifecycle management (initialize, start, stop, shutdown)
    - Stats tracking
    - Dependency injection pattern via setters

    Subclasses should implement:
    - async def _do_start(self): Main startup logic
    - async def _do_stop(self): Main shutdown logic
    - async def get_stats(self) -> Dict[str, Any]: Custom stats
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the coordinator base.

        Args:
            name: Optional name for logging. Defaults to class name.
        """
        self._name = name or self.__class__.__name__
        self._status = CoordinatorStatus.INITIALIZING
        self._running = False
        self._start_time: float = 0.0
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = CoordinatorStats()
        self._operations_count = 0
        self._errors_count = 0
        self._last_operation_time: float = 0.0
        self._last_error: Optional[str] = None

        # Dependency injection slots
        self._dependencies: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Coordinator name for logging."""
        return self._name

    @property
    def status(self) -> CoordinatorStatus:
        """Current status of the coordinator."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Whether the coordinator is actively running."""
        return self._running

    @property
    def uptime_seconds(self) -> float:
        """Seconds since start (0 if not running)."""
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    def set_dependency(self, name: str, value: Any) -> None:
        """Set a named dependency.

        This is the preferred pattern for dependency injection,
        allowing dependencies to be set after construction.

        Args:
            name: Dependency name
            value: Dependency value
        """
        self._dependencies[name] = value
        logger.debug(f"[{self._name}] Set dependency: {name}")

    def get_dependency(self, name: str, default: Any = None) -> Any:
        """Get a named dependency.

        Args:
            name: Dependency name
            default: Default value if not set

        Returns:
            The dependency value or default
        """
        return self._dependencies.get(name, default)

    def has_dependency(self, name: str) -> bool:
        """Check if a dependency is set."""
        return name in self._dependencies

    async def initialize(self) -> None:
        """Initialize the coordinator.

        Override _do_initialize for custom initialization.
        """
        async with self._lock:
            if self._status != CoordinatorStatus.INITIALIZING:
                return

            try:
                await self._do_initialize()
                self._status = CoordinatorStatus.READY
                logger.info(f"[{self._name}] Initialized")
            except Exception as e:
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Initialization failed: {e}")
                raise

    async def start(self) -> None:
        """Start the coordinator.

        Override _do_start for custom start logic.
        """
        async with self._lock:
            if self._running:
                return

            if self._status == CoordinatorStatus.INITIALIZING:
                await self._do_initialize()

            try:
                self._running = True
                self._start_time = time.time()
                self._status = CoordinatorStatus.RUNNING
                await self._do_start()
                logger.info(f"[{self._name}] Started")
            except Exception as e:
                self._running = False
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Start failed: {e}")
                raise

    async def stop(self) -> None:
        """Stop the coordinator gracefully.

        Override _do_stop for custom stop logic.
        """
        async with self._lock:
            if not self._running:
                return

            try:
                self._status = CoordinatorStatus.DRAINING
                await self._do_stop()
                self._running = False
                self._status = CoordinatorStatus.STOPPED
                logger.info(f"[{self._name}] Stopped")
            except Exception as e:
                self._status = CoordinatorStatus.ERROR
                self._last_error = str(e)
                logger.error(f"[{self._name}] Stop failed: {e}")
                raise

    async def shutdown(self) -> None:
        """Shutdown and cleanup all resources.

        Override _do_shutdown for custom cleanup.
        """
        await self.stop()
        await self._do_shutdown()
        logger.info(f"[{self._name}] Shutdown complete")

    async def pause(self) -> None:
        """Pause the coordinator temporarily."""
        if self._running and self._status == CoordinatorStatus.RUNNING:
            self._status = CoordinatorStatus.PAUSED
            logger.info(f"[{self._name}] Paused")

    async def resume(self) -> None:
        """Resume from paused state."""
        if self._running and self._status == CoordinatorStatus.PAUSED:
            self._status = CoordinatorStatus.RUNNING
            logger.info(f"[{self._name}] Resumed")

    def record_operation(self) -> None:
        """Record a successful operation."""
        self._operations_count += 1
        self._last_operation_time = time.time()

    def record_error(self, error: Optional[Exception] = None) -> None:
        """Record an error."""
        self._errors_count += 1
        if error:
            self._last_error = str(error)

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics.

        Subclasses should call super().get_stats() and extend.

        Returns:
            Dictionary of statistics
        """
        return {
            "name": self._name,
            "status": self._status.value,
            "is_running": self._running,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "operations_count": self._operations_count,
            "errors_count": self._errors_count,
            "last_operation_time": self._last_operation_time,
            "last_error": self._last_error,
        }

    # Hooks for subclasses - override these
    async def _do_initialize(self) -> None:
        """Custom initialization logic. Override in subclass."""
        pass

    async def _do_start(self) -> None:
        """Custom start logic. Override in subclass."""
        pass

    async def _do_stop(self) -> None:
        """Custom stop logic. Override in subclass."""
        pass

    async def _do_shutdown(self) -> None:
        """Custom shutdown/cleanup logic. Override in subclass."""
        pass


class SQLitePersistenceMixin:
    """Mixin for SQLite-based persistence.

    Provides:
    - Thread-local database connections
    - WAL mode for concurrent access
    - Schema initialization

    Usage:
        class MyManager(CoordinatorBase, SQLitePersistenceMixin):
            def __init__(self, db_path: Path):
                super().__init__()
                self.init_db(db_path)

            def _get_schema(self) -> str:
                return '''CREATE TABLE IF NOT EXISTS my_table (...)'''
    """

    _db_path: Optional[Path] = None
    _db_local: Optional[threading.local] = None

    def init_db(self, db_path: Path) -> None:
        """Initialize database with schema.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path
        self._db_local = threading.local()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        conn = self._get_connection()
        schema = self._get_schema()
        if schema:
            conn.executescript(schema)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        Returns:
            SQLite connection configured for WAL mode
        """
        if self._db_local is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")

        if not hasattr(self._db_local, "conn") or self._db_local.conn is None:
            self._db_local.conn = sqlite3.connect(str(self._db_path), timeout=30.0)
            self._db_local.conn.row_factory = sqlite3.Row
            self._db_local.conn.execute('PRAGMA journal_mode=WAL')
            self._db_local.conn.execute('PRAGMA busy_timeout=10000')
            self._db_local.conn.execute('PRAGMA synchronous=NORMAL')
        return self._db_local.conn

    def _close_connection(self) -> None:
        """Close the thread-local database connection."""
        if self._db_local and hasattr(self._db_local, "conn") and self._db_local.conn:
            self._db_local.conn.close()
            self._db_local.conn = None

    def _get_schema(self) -> str:
        """Get database schema SQL.

        Override in subclass to define tables and indexes.

        Returns:
            SQL CREATE statements
        """
        return ""


class SingletonMixin:
    """Mixin for singleton pattern.

    Ensures only one instance per class exists.

    Usage:
        class MyManager(CoordinatorBase, SingletonMixin):
            @classmethod
            def get_instance(cls) -> 'MyManager':
                return cls._get_or_create_instance()
    """

    _instances: Dict[type, Any] = {}
    _instance_lock = threading.Lock()

    @classmethod
    def _get_or_create_instance(cls, *args, **kwargs) -> Any:
        """Get or create the singleton instance.

        Args:
            *args, **kwargs: Constructor arguments (only used on first call)

        Returns:
            The singleton instance
        """
        if cls not in cls._instances:
            with cls._instance_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls(*args, **kwargs)
        return cls._instances[cls]

    @classmethod
    def _clear_instance(cls) -> None:
        """Clear the singleton instance (mainly for testing)."""
        with cls._instance_lock:
            cls._instances.pop(cls, None)


class CallbackMixin:
    """Mixin for callback-based extensibility.

    Provides a pattern for registering and invoking callbacks.

    Usage:
        class MyCoordinator(CoordinatorBase, CallbackMixin):
            async def on_event(self, event_type: str, data: dict):
                await self.invoke_callbacks(event_type, data)

        # Register callbacks
        coordinator.register_callback("job_complete", my_handler)
    """

    _callbacks: Dict[str, List[Callable]]

    def __init_callbacks__(self) -> None:
        """Initialize callbacks dictionary. Call in __init__."""
        if not hasattr(self, "_callbacks"):
            self._callbacks = {}

    def register_callback(
        self,
        event_type: str,
        callback: Callable,
    ) -> None:
        """Register a callback for an event type.

        Args:
            event_type: Type of event to handle
            callback: Callback function (sync or async)
        """
        if not hasattr(self, "_callbacks"):
            self._callbacks = {}
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def unregister_callback(
        self,
        event_type: str,
        callback: Callable,
    ) -> None:
        """Unregister a callback.

        Args:
            event_type: Type of event
            callback: Callback to remove
        """
        if hasattr(self, "_callbacks") and event_type in self._callbacks:
            try:
                self._callbacks[event_type].remove(callback)
            except ValueError:
                pass

    async def invoke_callbacks(
        self,
        event_type: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Invoke all callbacks for an event type.

        Args:
            event_type: Type of event
            *args, **kwargs: Arguments to pass to callbacks

        Returns:
            List of callback return values
        """
        if not hasattr(self, "_callbacks"):
            return []

        results = []
        for callback in self._callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Callback error for {event_type}: {e}")
                results.append(None)
        return results


# Convenience function to check if something is a coordinator
def is_coordinator(obj: Any) -> bool:
    """Check if an object implements the CoordinatorProtocol.

    Args:
        obj: Object to check

    Returns:
        True if obj is a coordinator
    """
    return isinstance(obj, CoordinatorProtocol)

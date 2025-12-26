"""Base Orchestrator Class for RingRift Coordination.

Captures common patterns across all orchestrators:
- Singleton management
- Event subscription and handling
- Status reporting
- Callback registration
- State management

Usage:
    from app.coordination.base_orchestrator import BaseOrchestrator

    class MyOrchestrator(BaseOrchestrator):
        def __init__(self):
            super().__init__(name="my_orchestrator")
            self._my_state = {}

        async def _on_my_event(self, event) -> None:
            payload = event.payload
            # Handle event

        def get_status(self) -> dict:
            base_status = super().get_status()
            base_status.update({
                "my_custom_status": self._my_state
            })
            return base_status

    # Use singleton
    orchestrator = MyOrchestrator.get_instance()
    await orchestrator.subscribe_to_events()

Created: December 26, 2025
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorStatus:
    """Base status information for all orchestrators."""

    name: str
    subscribed: bool = False
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: str | None = None


class BaseOrchestrator(ABC):
    """Abstract base class for all orchestrators in RingRift.

    Provides:
    - Singleton management with thread-safe lazy initialization
    - Event subscription and handler infrastructure
    - Status reporting and health checks
    - Callback registration mechanism
    - Error tracking and logging
    - State management with activity timestamps

    Subclasses should:
    1. Call super().__init__(name=...) in __init__
    2. Implement async event handlers as _on_<event_name>
    3. Override get_status() to add custom status fields
    4. Use self._record_error() for error tracking
    """

    # Singleton storage per class (class-level dictionary)
    _instances: ClassVar[dict[str, BaseOrchestrator]] = {}
    _singleton_lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(self, name: str):
        """Initialize base orchestrator.

        Args:
            name: Unique orchestrator name (used for logging/identification)
        """
        self._name = name
        self._subscribed = False
        self._status = OrchestratorStatus(name=name)
        self._callbacks: dict[str, list[Callable]] = {}
        self._error_log: list[dict[str, Any]] = []
        self._max_error_log = 100

    @property
    def name(self) -> str:
        """Get orchestrator name."""
        return self._name

    @property
    def is_subscribed(self) -> bool:
        """Check if orchestrator is subscribed to events."""
        return self._subscribed

    # =========================================================================
    # Singleton Management
    # =========================================================================

    @classmethod
    def get_instance(cls) -> BaseOrchestrator:
        """Get or create singleton instance (thread-safe).

        Subclasses can override this to customize singleton behavior.

        Returns:
            Singleton instance of this orchestrator class
        """
        instance_key = cls.__name__
        with cls._singleton_lock:
            if instance_key not in cls._instances:
                cls._instances[instance_key] = cls()
            return cls._instances[instance_key]

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing).

        WARNING: Only use in tests. This breaks the singleton contract.
        """
        instance_key = cls.__name__
        with cls._singleton_lock:
            if instance_key in cls._instances:
                instance = cls._instances[instance_key]
                if hasattr(instance, "shutdown"):
                    try:
                        import asyncio

                        if asyncio.iscoroutinefunction(instance.shutdown):
                            # Can't await here, just remove
                            pass
                        else:
                            instance.shutdown()
                    except (AttributeError, RuntimeError, TypeError, OSError):
                        # Ignore errors during cleanup shutdown
                        pass
                del cls._instances[instance_key]

    # =========================================================================
    # Event Subscription (Template Method Pattern)
    # =========================================================================

    async def subscribe_to_events(self) -> bool:
        """Subscribe to all relevant events.

        Base implementation handles common setup.
        Subclasses can override to customize or call super().

        Returns:
            True if subscription successful
        """
        if self._subscribed:
            return True

        self._subscribed = True
        self._status.last_activity = time.time()
        logger.debug(f"[{self._name}] Subscribed to events")
        return True

    async def unsubscribe_from_events(self) -> bool:
        """Unsubscribe from all events.

        Returns:
            True if unsubscription successful
        """
        self._subscribed = False
        logger.debug(f"[{self._name}] Unsubscribed from events")
        return True

    # =========================================================================
    # Event Handler Infrastructure
    # =========================================================================

    async def handle_event(self, event_name: str, event: Any) -> None:
        """Route event to appropriate handler.

        Looks for _on_<event_name> method on subclass and calls it.

        Args:
            event_name: Name of the event
            event: Event object/payload
        """
        handler_name = f"_on_{event_name}"
        if hasattr(self, handler_name):
            try:
                handler = getattr(self, handler_name)
                await handler(event)
                self._status.last_activity = time.time()
            except Exception as e:
                self._record_error(f"Event handler error for {event_name}: {e}")
                raise
        else:
            logger.debug(f"[{self._name}] No handler for event: {event_name}")

    # =========================================================================
    # Callback Management
    # =========================================================================

    def register_callback(
        self,
        event_name: str,
        callback: Callable[[Any], None],
    ) -> None:
        """Register a callback for an event.

        Args:
            event_name: Event to listen for
            callback: Function to call when event occurs
        """
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        self._callbacks[event_name].append(callback)

    def unregister_callback(
        self,
        event_name: str,
        callback: Callable[[Any], None],
    ) -> bool:
        """Unregister a callback.

        Args:
            event_name: Event to stop listening for
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if event_name in self._callbacks:
            try:
                self._callbacks[event_name].remove(callback)
                return True
            except ValueError:
                return False
        return False

    async def _invoke_callbacks(self, event_name: str, data: Any) -> None:
        """Invoke all callbacks for an event.

        Errors in callbacks are logged but don't propagate.

        Args:
            event_name: Event that occurred
            data: Data to pass to callbacks
        """
        if event_name not in self._callbacks:
            return

        for callback in self._callbacks[event_name]:
            try:
                if hasattr(callback, "__await__"):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self._record_error(f"Callback error for {event_name}: {e}")

    # =========================================================================
    # Status and Health Reporting
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status for monitoring.

        Returns:
            Dict with status information. Subclasses should extend this.
        """
        return {
            "name": self._name,
            "subscribed": self._subscribed,
            "created_at": self._status.created_at,
            "last_activity": self._status.last_activity,
            "uptime_seconds": time.time() - self._status.created_at,
            "error_count": self._status.error_count,
            "last_error": self._status.last_error,
            "is_healthy": self.is_healthy(),
        }

    def is_healthy(self) -> bool:
        """Check if orchestrator is healthy.

        Base implementation checks error count.
        Subclasses can override for custom health logic.

        Returns:
            True if orchestrator is considered healthy
        """
        # Allow some errors but flag if too many recent
        recent_errors = sum(
            1
            for e in self._error_log[-10:]
            if time.time() - e.get("timestamp", 0) < 300
        )
        return recent_errors < 5

    # =========================================================================
    # Error Tracking
    # =========================================================================

    def _record_error(self, message: str, error_type: str = "general") -> None:
        """Record an error in the error log.

        Args:
            message: Error message
            error_type: Category of error (for filtering)
        """
        self._status.error_count += 1
        self._status.last_error = message

        error_entry = {
            "timestamp": time.time(),
            "message": message,
            "type": error_type,
        }
        self._error_log.append(error_entry)

        # Keep error log bounded
        if len(self._error_log) > self._max_error_log:
            self._error_log = self._error_log[-self._max_error_log :]

        logger.error(f"[{self._name}] {error_type}: {message}")

    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error entries
        """
        return self._error_log[-limit:]

    def clear_error_log(self) -> int:
        """Clear error log.

        Returns:
            Number of errors cleared
        """
        count = len(self._error_log)
        self._error_log.clear()
        return count

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> bool:
        """Start the orchestrator.

        Base implementation subscribes to events.
        Subclasses can override to add custom startup logic.

        Returns:
            True if started successfully
        """
        logger.info(f"[{self._name}] Starting...")
        return await self.subscribe_to_events()

    async def stop(self) -> bool:
        """Stop the orchestrator.

        Base implementation unsubscribes from events.
        Subclasses can override to add custom shutdown logic.

        Returns:
            True if stopped successfully
        """
        logger.info(f"[{self._name}] Stopping...")
        return await self.unsubscribe_from_events()

    async def shutdown(self) -> None:
        """Graceful shutdown.

        Subclasses can override to add cleanup logic.
        """
        await self.stop()

    # =========================================================================
    # Debug/Monitoring Utilities
    # =========================================================================

    def get_state_summary(self) -> dict[str, Any]:
        """Get summary of internal state (for debugging).

        Subclasses should override to expose relevant state.

        Returns:
            Dict with state information
        """
        return {
            "name": self._name,
            "subscribed": self._subscribed,
            "callback_count": sum(len(cbs) for cbs in self._callbacks.values()),
            "error_count": len(self._error_log),
        }

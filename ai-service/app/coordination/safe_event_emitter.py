"""Unified safe event emission mixin.

Jan 4, 2026 - Sprint 17.9: Now delegates to event_emission_helpers.py for
consolidated implementation with optional logging support.

Consolidates 6 duplicate `_safe_emit_event()` implementations across:
- availability/node_monitor.py
- availability/recovery_engine.py
- availability/capacity_planner.py
- availability/provisioner.py
- scripts/p2p/managers/state_manager.py
- scripts/p2p/p2p_mixin_base.py

Usage:
    from app.coordination.safe_event_emitter import SafeEventEmitterMixin

    class MyCoordinator(SafeEventEmitterMixin):
        _event_source = "MyCoordinator"

        def do_something(self):
            self._safe_emit_event("MY_EVENT", {"key": "value"})

        # With logging (new in Sprint 17.9):
        def do_something_with_logging(self):
            self._safe_emit_event(
                "MY_EVENT",
                {"key": "value"},
                log_before="Starting operation",
                log_after="Event emitted",
            )

    class MyAsyncDaemon(SafeEventEmitterMixin):
        _event_source = "MyAsyncDaemon"

        async def do_something_async(self):
            await self._safe_emit_event_async("MY_EVENT", {"key": "value"})
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class SafeEventEmitterMixin:
    """Mixin providing unified safe event emission for coordinators and daemons.

    Provides both sync and async emission methods that:
    - Never raise exceptions (log failures instead)
    - Return bool indicating success/failure
    - Use lazy imports to avoid circular dependencies
    - Track event source for debugging

    Example - Sync coordinator:
        >>> class MyCoordinator(SafeEventEmitterMixin):
        ...     _event_source = "MyCoordinator"
        ...
        ...     def process_data(self, config_key: str):
        ...         # Do processing...
        ...         success = self._safe_emit_event(
        ...             "DATA_PROCESSED",
        ...             {"config_key": config_key, "status": "complete"},
        ...         )
        ...         if not success:
        ...             logger.warning("Event emission failed, continuing anyway")

    Example - Async daemon:
        >>> class MyAsyncDaemon(SafeEventEmitterMixin):
        ...     _event_source = "MyAsyncDaemon"
        ...
        ...     async def run_cycle(self):
        ...         result = await self.do_work()
        ...         await self._safe_emit_event_async(
        ...             "CYCLE_COMPLETED",
        ...             {"result": result, "timestamp": time.time()},
        ...         )

    Example - Check emission success:
        >>> if self._safe_emit_event("CRITICAL_EVENT", payload):
        ...     logger.info("Event emitted successfully")
        ... else:
        ...     logger.warning("Event bus unavailable, using fallback")
        ...     self._queue_for_retry(payload)

    Attributes:
        _event_source: Class-level identifier for the event source.
                      Override in subclasses to set custom source name.
    """

    _event_source: ClassVar[str] = "unknown"

    def _safe_emit_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        log_before: str | None = None,
        log_after: str | None = None,
    ) -> bool:
        """Safely emit an event via the event router.

        Wraps event emission in try-catch to prevent event failures
        from crashing the caller. Delegates to event_emission_helpers.

        Args:
            event_type: Event type string to emit (e.g., "TRAINING_COMPLETED")
            payload: Optional event payload dict
            log_before: Optional message to log before emission
            log_after: Optional message to log after successful emission

        Returns:
            True if event was scheduled successfully, False otherwise

        Example:
            self._safe_emit_event(
                "HOST_OFFLINE",
                {"node_id": peer_id, "reason": "timeout"},
            )

            # With logging:
            self._safe_emit_event(
                "SYNC_COMPLETED",
                {"files": 42},
                log_before="Finishing sync",
                log_after="Sync event emitted",
            )
        """
        # Delegate to consolidated implementation
        from app.coordination.event_emission_helpers import (
            safe_emit_event as _consolidated_emit,
        )

        return _consolidated_emit(
            event_type,
            payload,
            log_before=log_before,
            log_after=log_after,
            context=self._event_source,
            source=self._event_source,
        )

    async def _safe_emit_event_async(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        log_before: str | None = None,
        log_after: str | None = None,
    ) -> bool:
        """Async version of safe event emission.

        For use in async contexts where blocking on the event bus
        could cause issues. Delegates to event_emission_helpers.

        Args:
            event_type: Event type string to emit
            payload: Optional event payload dict
            log_before: Optional message to log before emission
            log_after: Optional message to log after successful emission

        Returns:
            True if event was emitted successfully, False otherwise
        """
        from app.coordination.event_emission_helpers import (
            safe_emit_event_async as _consolidated_emit_async,
        )

        return await _consolidated_emit_async(
            event_type,
            payload,
            log_before=log_before,
            log_after=log_after,
            context=self._event_source,
            source=self._event_source,
        )


# Module-level helper for non-class contexts
# Delegates to event_emission_helpers for unified implementation
def safe_emit_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    source: str = "module",
    *,
    log_before: str | None = None,
    log_after: str | None = None,
    context: str | None = None,
) -> bool:
    """Module-level safe event emission.

    For use in module-level functions or contexts without a class.
    Delegates to event_emission_helpers.safe_emit_event for unified implementation.

    Args:
        event_type: Event type string to emit
        payload: Optional event payload dict
        source: Source identifier for the event
        log_before: Optional message to log before emission
        log_after: Optional message to log after successful emission
        context: Context string for error messages (default: source)

    Returns:
        True if event was scheduled successfully, False otherwise

    Example:
        from app.coordination.safe_event_emitter import safe_emit_event

        def my_function():
            safe_emit_event("MY_EVENT", {"key": "value"}, source="my_module")

        # With logging:
        safe_emit_event(
            "SYNC_COMPLETED",
            {"files": 42},
            source="sync_module",
            log_before="Starting sync completion",
            log_after="Sync complete event emitted",
        )
    """
    # Import here to avoid circular dependencies
    from app.coordination.event_emission_helpers import (
        safe_emit_event as _consolidated_emit,
    )

    return _consolidated_emit(
        event_type,
        payload,
        log_before=log_before,
        log_after=log_after,
        context=context or source,
        source=source,
    )


__all__ = [
    "SafeEventEmitterMixin",
    "safe_emit_event",
]

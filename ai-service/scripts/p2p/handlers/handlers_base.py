"""Base utilities for P2P HTTP handler mixins.

Consolidates common patterns across all handler files:
- Event bridge with graceful fallback
- Exception handling decorators
- Authorization decorators
- Status tracking mixins
- Response formatting utilities

Dec 2025: Extracted from 14 handler files to eliminate ~400 LOC duplication.

Usage:
    from scripts.p2p.handlers.handlers_base import (
        safe_handler,
        leader_only,
        EventBridgeManager,
        HandlerStatusMixin,
    )

    class MyHandlersMixin(HandlerStatusMixin):
        @safe_handler("my_endpoint")
        @leader_only
        async def handle_my_endpoint(self, request):
            data = await request.json()
            return {"result": "ok"}
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from aiohttp import web

if TYPE_CHECKING:
    from aiohttp.web import Request, Response

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Event Bridge Manager
# =============================================================================


class EventBridgeManager:
    """Safe event emission with graceful fallback.

    Handles ImportError gracefully and provides fire-and-forget semantics
    for event emission. Thread-safe for concurrent handler use.

    Usage:
        events = EventBridgeManager()

        # Emit event (fire-and-forget)
        await events.emit("LEADER_CHANGED", {"old": "a", "new": "b"})

        # Check availability
        if events.available:
            print("Events will be emitted")
    """

    def __init__(self) -> None:
        self._bridge = None
        self._available: bool | None = None  # Lazy-loaded
        self._emit_functions: dict[str, Callable] = {}

    @property
    def available(self) -> bool:
        """Check if event bridge is available."""
        if self._available is None:
            self._ensure_loaded()
        return self._available or False

    def _ensure_loaded(self) -> None:
        """Lazy load event bridge module."""
        if self._available is not None:
            return

        try:
            from scripts.p2p import p2p_event_bridge
            self._bridge = p2p_event_bridge
            self._available = True

            # Cache commonly used emit functions
            for name in dir(p2p_event_bridge):
                if name.startswith("emit_"):
                    self._emit_functions[name] = getattr(p2p_event_bridge, name)

            logger.debug("[EventBridgeManager] Event bridge loaded successfully")
        except ImportError as e:
            self._available = False
            logger.debug(f"[EventBridgeManager] Event bridge not available: {e}")

    async def emit(self, event_name: str, payload: dict[str, Any]) -> bool:
        """Emit an event via the bridge.

        Args:
            event_name: Event function name without 'emit_' prefix
                       (e.g., "p2p_leader_changed" calls emit_p2p_leader_changed)
            payload: Event payload dict

        Returns:
            True if event was emitted, False if bridge unavailable
        """
        self._ensure_loaded()

        if not self._available:
            return False

        func_name = f"emit_{event_name}"
        emit_func = self._emit_functions.get(func_name)

        if emit_func is None:
            logger.debug(f"[EventBridgeManager] Unknown event function: {func_name}")
            return False

        try:
            result = emit_func(**payload)
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as e:
            logger.debug(f"[EventBridgeManager] Event emission failed: {e}")
            return False

    def get_emit_func(self, func_name: str) -> Callable | None:
        """Get a specific emit function for direct use.

        Args:
            func_name: Full function name (e.g., "emit_p2p_leader_changed")

        Returns:
            The emit function or None if unavailable
        """
        self._ensure_loaded()
        return self._emit_functions.get(func_name)


# Global event bridge manager instance
_event_bridge = EventBridgeManager()


def get_event_bridge() -> EventBridgeManager:
    """Get the global event bridge manager."""
    return _event_bridge


# =============================================================================
# Exception Handling Decorator
# =============================================================================


def safe_handler(handler_name: str) -> Callable[[F], F]:
    """Decorator for consistent exception handling in HTTP handlers.

    Wraps handler with try-except that:
    - Returns 400 for KeyError/ValueError (client errors)
    - Returns 504 for TimeoutError
    - Returns 500 for other exceptions
    - Logs errors with handler-specific prefix

    Args:
        handler_name: Short name for logging (e.g., "election_start")

    Usage:
        @safe_handler("election_start")
        async def handle_election_start(self, request):
            data = await request.json()
            return {"status": "ok"}
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(self, request: Request) -> Response:
            try:
                result = await func(self, request)

                # If handler returns dict, wrap in json_response
                if isinstance(result, dict):
                    return web.json_response(result)
                return result

            except KeyError as e:
                logger.warning(f"[{handler_name}] Missing field: {e}")
                return web.json_response(
                    {"error": f"Missing required field: {e}", "handler": handler_name},
                    status=400
                )
            except ValueError as e:
                logger.warning(f"[{handler_name}] Invalid value: {e}")
                return web.json_response(
                    {"error": f"Invalid value: {e}", "handler": handler_name},
                    status=400
                )
            except asyncio.TimeoutError:
                logger.error(f"[{handler_name}] Operation timed out")
                return web.json_response(
                    {"error": "Operation timed out", "handler": handler_name},
                    status=504
                )
            except web.HTTPException:
                # Re-raise HTTP exceptions (already formatted)
                raise
            except Exception as e:
                logger.error(f"[{handler_name}] Unexpected error: {e}", exc_info=True)
                return web.json_response(
                    {"error": "Internal server error", "handler": handler_name},
                    status=500
                )

        return wrapper  # type: ignore
    return decorator


# =============================================================================
# Authorization Decorators
# =============================================================================


def leader_only(func: F) -> F:
    """Decorator to restrict handler to leader node only.

    Returns 403 Forbidden if the current node is not the leader.
    Expects the mixin class to have a `role` attribute.

    Usage:
        @leader_only
        async def handle_dispatch_job(self, request):
            # Only leader can dispatch jobs
            ...
    """
    @functools.wraps(func)
    async def wrapper(self, request: Request) -> Response:
        # Import NodeRole lazily to avoid circular imports
        try:
            from scripts.p2p.types import NodeRole
        except ImportError:
            from enum import Enum
            class NodeRole(str, Enum):
                LEADER = "leader"

        role = getattr(self, "role", None)
        if role != NodeRole.LEADER:
            node_id = getattr(self, "node_id", "unknown")
            logger.debug(f"[leader_only] Rejected: {node_id} is {role}, not LEADER")
            return web.json_response(
                {"error": "Only leader can perform this action", "role": str(role)},
                status=403
            )
        return await func(self, request)

    return wrapper  # type: ignore


def voter_only(func: F) -> F:
    """Decorator to restrict handler to voter nodes only.

    Returns 403 Forbidden if the current node is not a voter.
    Expects the mixin class to have `node_id` and `voter_node_ids` attributes.

    Usage:
        @voter_only
        async def handle_vote_request(self, request):
            # Only voters can participate in elections
            ...
    """
    @functools.wraps(func)
    async def wrapper(self, request: Request) -> Response:
        node_id = getattr(self, "node_id", None)
        voter_ids = getattr(self, "voter_node_ids", [])

        if node_id not in voter_ids:
            logger.debug(f"[voter_only] Rejected: {node_id} is not a voter")
            return web.json_response(
                {"error": "Only voter nodes can perform this action"},
                status=403
            )
        return await func(self, request)

    return wrapper  # type: ignore


# =============================================================================
# Status Tracking Mixin
# =============================================================================


class HandlerStatusMixin:
    """Mixin providing common status tracking for handler classes.

    Tracks:
    - Request counts per endpoint
    - Error counts and last error
    - Handler health metrics

    Usage:
        class MyHandlersMixin(HandlerStatusMixin):
            def __init__(self):
                super().__init__()
                self._init_handler_status("MyHandlers")
    """

    def _init_handler_status(self, handler_name: str) -> None:
        """Initialize status tracking for this handler."""
        self._handler_name = handler_name
        self._handler_request_counts: dict[str, int] = {}
        self._handler_error_counts: dict[str, int] = {}
        self._handler_last_error: str = ""
        self._handler_start_time: float = time.time()

    def _record_request(self, endpoint: str) -> None:
        """Record a request to an endpoint."""
        if not hasattr(self, "_handler_request_counts"):
            return
        self._handler_request_counts[endpoint] = (
            self._handler_request_counts.get(endpoint, 0) + 1
        )

    def _record_error(self, endpoint: str, error: str) -> None:
        """Record an error on an endpoint."""
        if not hasattr(self, "_handler_error_counts"):
            return
        self._handler_error_counts[endpoint] = (
            self._handler_error_counts.get(endpoint, 0) + 1
        )
        self._handler_last_error = f"{endpoint}: {error}"

    def get_handler_status(self) -> dict[str, Any]:
        """Get status summary for this handler."""
        if not hasattr(self, "_handler_name"):
            return {"error": "Handler status not initialized"}

        total_requests = sum(self._handler_request_counts.values())
        total_errors = sum(self._handler_error_counts.values())
        uptime = time.time() - self._handler_start_time

        return {
            "handler": self._handler_name,
            "uptime_seconds": uptime,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "success_rate": (
                (total_requests - total_errors) / total_requests
                if total_requests > 0
                else 1.0
            ),
            "requests_by_endpoint": dict(self._handler_request_counts),
            "errors_by_endpoint": dict(self._handler_error_counts),
            "last_error": self._handler_last_error,
        }


# =============================================================================
# Response Formatting Utilities
# =============================================================================


def success_response(
    data: dict[str, Any] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Format a success response.

    Args:
        data: Response data
        message: Optional success message

    Returns:
        Formatted response dict
    """
    response: dict[str, Any] = {
        "success": True,
        "timestamp": time.time(),
    }
    if message:
        response["message"] = message
    if data:
        response.update(data)
    return response


def error_response(
    error: str,
    details: dict[str, Any] | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    """Format an error response.

    Args:
        error: Error message
        details: Additional error details
        code: Error code for programmatic handling

    Returns:
        Formatted response dict
    """
    response: dict[str, Any] = {
        "success": False,
        "error": error,
        "timestamp": time.time(),
    }
    if code:
        response["code"] = code
    if details:
        response["details"] = details
    return response


# =============================================================================
# Request Parsing Utilities
# =============================================================================


async def parse_json_request(
    request: Request,
    required_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Parse JSON request body with optional field validation.

    Args:
        request: aiohttp Request object
        required_fields: List of required field names

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If required fields are missing
        web.HTTPBadRequest: If JSON parsing fails
    """
    try:
        data = await request.json()
    except Exception as e:
        raise web.HTTPBadRequest(text=f"Invalid JSON: {e}")

    if required_fields:
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    return data


def validate_node_id(node_id: str | None, peers: dict) -> str:
    """Validate that a node ID exists in the peer list.

    Args:
        node_id: Node ID to validate
        peers: Dict of known peers

    Returns:
        The validated node_id

    Raises:
        ValueError: If node_id is None or not in peers
    """
    if not node_id:
        raise ValueError("node_id is required")

    if node_id not in peers:
        raise ValueError(f"Unknown peer: {node_id}")

    return node_id


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Event bridge
    "EventBridgeManager",
    "get_event_bridge",
    # Decorators
    "safe_handler",
    "leader_only",
    "voter_only",
    # Mixins
    "HandlerStatusMixin",
    # Response utilities
    "success_response",
    "error_response",
    # Request utilities
    "parse_json_request",
    "validate_node_id",
]

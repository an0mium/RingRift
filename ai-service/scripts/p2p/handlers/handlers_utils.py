"""Shared utilities for P2P HTTP handlers.

Contains helper functions used across multiple handler mixins:
- Peer validation and lookup
- Timestamp formatting
- Retry logic for distributed operations
- Metrics collection helpers

Dec 2025: Extracted from handler files to reduce duplication.

Usage:
    from scripts.p2p.handlers.handlers_utils import (
        get_peer_info,
        format_timestamp,
        RetryStrategy,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Peer Utilities
# =============================================================================


def get_peer_info(
    peers: dict[str, Any],
    peer_id: str,
    peers_lock: Any = None,
) -> dict[str, Any] | None:
    """Get peer info with optional lock handling.

    Args:
        peers: Dict mapping peer_id to NodeInfo
        peer_id: ID of peer to look up
        peers_lock: Optional lock for thread-safe access

    Returns:
        Peer info dict or None if not found
    """
    try:
        if peers_lock:
            with peers_lock:
                peer = peers.get(peer_id)
        else:
            peer = peers.get(peer_id)

        if peer is None:
            return None

        # Convert to dict if it has to_dict method
        if hasattr(peer, "to_dict"):
            return peer.to_dict()
        elif hasattr(peer, "__dict__"):
            return dict(peer.__dict__)
        else:
            return {"peer_id": peer_id, "info": str(peer)}

    except Exception as e:
        logger.debug(f"[get_peer_info] Error getting peer {peer_id}: {e}")
        return None


def get_alive_peers(
    peers: dict[str, Any],
    peers_lock: Any = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get list of alive peer IDs.

    Args:
        peers: Dict mapping peer_id to NodeInfo
        peers_lock: Optional lock for thread-safe access
        exclude: Peer IDs to exclude from result

    Returns:
        List of alive peer IDs
    """
    exclude_set = set(exclude or [])
    result = []

    try:
        if peers_lock:
            with peers_lock:
                items = list(peers.items())
        else:
            items = list(peers.items())

        for peer_id, peer in items:
            if peer_id in exclude_set:
                continue
            # Check if peer is alive (has is_alive method or attribute)
            if hasattr(peer, "is_alive"):
                is_alive = peer.is_alive() if callable(peer.is_alive) else peer.is_alive
            else:
                is_alive = True  # Assume alive if no method

            if is_alive:
                result.append(peer_id)

    except Exception as e:
        logger.debug(f"[get_alive_peers] Error: {e}")

    return result


# =============================================================================
# Timestamp Utilities
# =============================================================================


def format_timestamp(ts: float | None = None) -> str:
    """Format timestamp for response.

    Args:
        ts: Unix timestamp (defaults to current time)

    Returns:
        ISO-formatted timestamp string
    """
    if ts is None:
        ts = time.time()

    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.isoformat()


def time_since(ts: float) -> float:
    """Get seconds elapsed since timestamp.

    Args:
        ts: Unix timestamp

    Returns:
        Seconds since ts
    """
    return time.time() - ts


def is_expired(ts: float, ttl_seconds: float) -> bool:
    """Check if timestamp has expired.

    Args:
        ts: Unix timestamp of creation/update
        ttl_seconds: Time-to-live in seconds

    Returns:
        True if ts + ttl_seconds < now
    """
    return time.time() > ts + ttl_seconds


# =============================================================================
# Retry Strategy
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry operations."""

    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random jitter factor (0-1)


@dataclass
class RetryResult:
    """Result of a retry operation."""

    success: bool
    attempts: int
    last_error: str | None = None
    result: Any = None
    total_time: float = 0.0


class RetryStrategy:
    """Retry strategy with exponential backoff.

    Provides retry logic for distributed operations with:
    - Exponential backoff
    - Maximum retry limit
    - Optional jitter
    - Detailed result tracking

    Usage:
        retry = RetryStrategy(max_retries=3, base_delay=1.0)

        result = await retry.execute(async_operation, arg1, arg2)
        if result.success:
            print(f"Succeeded after {result.attempts} attempts")
        else:
            print(f"Failed: {result.last_error}")
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        exponential_base: float = 2.0,
    ):
        self.config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
        )

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter > 0:
            import random
            jitter = random.uniform(-self.config.jitter, self.config.jitter)
            delay *= (1 + jitter)

        return max(0.01, delay)  # Minimum 10ms delay

    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> RetryResult:
        """Execute function with retry.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            RetryResult with success status and details
        """
        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result

                return RetryResult(
                    success=True,
                    attempts=attempt + 1,
                    result=result,
                    total_time=time.time() - start_time,
                )

            except Exception as e:
                last_error = str(e)
                logger.debug(
                    f"[RetryStrategy] Attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"failed: {e}"
                )

                if attempt < self.config.max_retries:
                    delay = self._get_delay(attempt)
                    await asyncio.sleep(delay)

        return RetryResult(
            success=False,
            attempts=self.config.max_retries + 1,
            last_error=last_error,
            total_time=time.time() - start_time,
        )


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class HandlerMetrics:
    """Metrics for a single handler endpoint."""

    endpoint: str
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    last_request_time: float = 0.0
    last_error: str = ""

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def error_rate(self) -> float:
        """Error rate (0.0 - 1.0)."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class MetricsCollector:
    """Collect and report handler metrics.

    Usage:
        collector = MetricsCollector()

        with collector.track("my_endpoint"):
            # Handle request
            pass

        # Get metrics
        metrics = collector.get_all_metrics()
    """

    def __init__(self):
        self._metrics: dict[str, HandlerMetrics] = {}

    def _get_or_create(self, endpoint: str) -> HandlerMetrics:
        """Get or create metrics for endpoint."""
        if endpoint not in self._metrics:
            self._metrics[endpoint] = HandlerMetrics(endpoint=endpoint)
        return self._metrics[endpoint]

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a request.

        Args:
            endpoint: Handler endpoint name
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
        """
        metrics = self._get_or_create(endpoint)
        metrics.request_count += 1
        metrics.total_latency_ms += latency_ms
        metrics.last_request_time = time.time()

        if error:
            metrics.error_count += 1
            metrics.last_error = error

    def track(self, endpoint: str):
        """Context manager for tracking request metrics.

        Usage:
            with collector.track("my_endpoint") as tracker:
                # Handle request
                if error:
                    tracker.set_error("Something went wrong")
        """
        return _MetricsTracker(self, endpoint)

    def get_metrics(self, endpoint: str) -> dict[str, Any] | None:
        """Get metrics for a specific endpoint."""
        if endpoint not in self._metrics:
            return None

        m = self._metrics[endpoint]
        return {
            "endpoint": m.endpoint,
            "request_count": m.request_count,
            "error_count": m.error_count,
            "avg_latency_ms": m.avg_latency_ms,
            "error_rate": m.error_rate,
            "last_request_time": m.last_request_time,
            "last_error": m.last_error,
        }

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all endpoints."""
        return {
            endpoint: self.get_metrics(endpoint)
            for endpoint in self._metrics
            if self.get_metrics(endpoint) is not None
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()


class _MetricsTracker:
    """Context manager for tracking a single request."""

    def __init__(self, collector: MetricsCollector, endpoint: str):
        self.collector = collector
        self.endpoint = endpoint
        self.start_time = 0.0
        self.error: str | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self.start_time) * 1000

        if exc_val is not None:
            self.error = str(exc_val)

        self.collector.record_request(
            self.endpoint,
            latency_ms,
            self.error,
        )
        return False  # Don't suppress exceptions

    def set_error(self, error: str) -> None:
        """Set error for this request."""
        self.error = error


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Peer utilities
    "get_peer_info",
    "get_alive_peers",
    # Timestamp utilities
    "format_timestamp",
    "time_since",
    "is_expired",
    # Retry
    "RetryConfig",
    "RetryResult",
    "RetryStrategy",
    # Metrics
    "HandlerMetrics",
    "MetricsCollector",
]

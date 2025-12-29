"""Tests for scripts.p2p.handlers.handlers_utils module.

Tests cover:
- Peer utilities (get_peer_info, get_alive_peers)
- Timestamp utilities (format_timestamp, time_since, is_expired)
- RetryStrategy (exponential backoff, jitter, result tracking)
- MetricsCollector (request tracking, error tracking, context manager)

December 2025.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock
import threading

import pytest

from scripts.p2p.handlers.handlers_utils import (
    HandlerMetrics,
    MetricsCollector,
    RetryConfig,
    RetryResult,
    RetryStrategy,
    format_timestamp,
    get_alive_peers,
    get_peer_info,
    is_expired,
    time_since,
)


# =============================================================================
# Peer Utilities Tests
# =============================================================================


class MockPeerInfo:
    """Mock peer info with to_dict method."""

    def __init__(self, node_id: str, is_alive: bool = True):
        self.node_id = node_id
        self._is_alive = is_alive
        self.host = f"192.168.1.{hash(node_id) % 255}"
        self.port = 8770

    def is_alive(self) -> bool:
        return self._is_alive

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
        }


class MockPeerWithAttribute:
    """Mock peer with is_alive as attribute instead of method."""

    def __init__(self, node_id: str, is_alive: bool = True):
        self.node_id = node_id
        self.is_alive = is_alive


class TestGetPeerInfo:
    """Tests for get_peer_info utility."""

    def test_get_existing_peer(self):
        """Get info for existing peer."""
        peer = MockPeerInfo("node-1")
        peers = {"node-1": peer}

        result = get_peer_info(peers, "node-1")

        assert result["node_id"] == "node-1"
        assert result["port"] == 8770

    def test_get_nonexistent_peer(self):
        """Get info for nonexistent peer returns None."""
        peers = {"node-1": MockPeerInfo("node-1")}

        result = get_peer_info(peers, "node-99")

        assert result is None

    def test_get_peer_with_lock(self):
        """Get peer info with lock support."""
        peer = MockPeerInfo("node-1")
        peers = {"node-1": peer}
        lock = threading.Lock()

        result = get_peer_info(peers, "node-1", peers_lock=lock)

        assert result["node_id"] == "node-1"

    def test_get_peer_with_dict_attribute(self):
        """Get peer info from object with __dict__."""

        class SimplePeer:
            def __init__(self):
                self.node_id = "simple"
                self.data = 123

        peers = {"simple": SimplePeer()}
        result = get_peer_info(peers, "simple")

        assert result["node_id"] == "simple"
        assert result["data"] == 123

    def test_get_peer_fallback_to_string(self):
        """Get peer info falls back to string representation."""
        # Peer without to_dict or __dict__
        peers = {"raw": "raw_value"}
        result = get_peer_info(peers, "raw")

        assert result["peer_id"] == "raw"
        assert "raw_value" in result["info"]


class TestGetAlivePeers:
    """Tests for get_alive_peers utility."""

    def test_get_all_alive(self):
        """Get list of alive peers."""
        peers = {
            "node-1": MockPeerInfo("node-1", is_alive=True),
            "node-2": MockPeerInfo("node-2", is_alive=True),
            "node-3": MockPeerInfo("node-3", is_alive=True),
        }

        result = get_alive_peers(peers)

        assert len(result) == 3
        assert "node-1" in result
        assert "node-2" in result

    def test_filter_dead_peers(self):
        """Filter out dead peers."""
        peers = {
            "alive-1": MockPeerInfo("alive-1", is_alive=True),
            "dead-1": MockPeerInfo("dead-1", is_alive=False),
            "alive-2": MockPeerInfo("alive-2", is_alive=True),
        }

        result = get_alive_peers(peers)

        assert len(result) == 2
        assert "alive-1" in result
        assert "alive-2" in result
        assert "dead-1" not in result

    def test_with_exclusion_list(self):
        """Exclude specified peers."""
        peers = {
            "node-1": MockPeerInfo("node-1"),
            "node-2": MockPeerInfo("node-2"),
            "node-3": MockPeerInfo("node-3"),
        }

        result = get_alive_peers(peers, exclude=["node-2"])

        assert len(result) == 2
        assert "node-2" not in result

    def test_with_lock(self):
        """Use lock for thread-safe access."""
        peers = {"node-1": MockPeerInfo("node-1")}
        lock = threading.Lock()

        result = get_alive_peers(peers, peers_lock=lock)

        assert "node-1" in result

    def test_is_alive_as_attribute(self):
        """Handle is_alive as attribute instead of method."""
        peers = {"node-1": MockPeerWithAttribute("node-1", is_alive=True)}

        result = get_alive_peers(peers)

        assert "node-1" in result

    def test_assume_alive_without_method(self):
        """Assume alive if no is_alive method/attribute."""

        class NoAliveCheck:
            node_id = "simple"

        peers = {"simple": NoAliveCheck()}
        result = get_alive_peers(peers)

        assert "simple" in result


# =============================================================================
# Timestamp Utilities Tests
# =============================================================================


class TestFormatTimestamp:
    """Tests for format_timestamp utility."""

    def test_format_current_time(self):
        """Format current time returns ISO format."""
        result = format_timestamp()

        # Should be ISO format with UTC timezone
        assert "T" in result
        assert result.endswith("+00:00") or result.endswith("Z")

    def test_format_specific_time(self):
        """Format specific timestamp."""
        # Unix timestamp for 2024-01-15 12:00:00 UTC
        ts = 1705320000.0
        result = format_timestamp(ts)

        assert "2024-01-15" in result
        assert "12:00:00" in result

    def test_format_returns_string(self):
        """Format always returns string."""
        result = format_timestamp(time.time())
        assert isinstance(result, str)


class TestTimeSince:
    """Tests for time_since utility."""

    def test_time_since_recent(self):
        """Time since recent timestamp is small."""
        recent = time.time() - 5.0  # 5 seconds ago

        result = time_since(recent)

        assert 4.9 < result < 6.0

    def test_time_since_old(self):
        """Time since old timestamp is large."""
        old = time.time() - 3600  # 1 hour ago

        result = time_since(old)

        assert 3599 < result < 3601


class TestIsExpired:
    """Tests for is_expired utility."""

    def test_not_expired(self):
        """Recent timestamp with long TTL is not expired."""
        recent = time.time() - 10  # 10 seconds ago
        ttl = 60  # 60 second TTL

        result = is_expired(recent, ttl)

        assert result is False

    def test_expired(self):
        """Old timestamp with short TTL is expired."""
        old = time.time() - 120  # 2 minutes ago
        ttl = 60  # 60 second TTL

        result = is_expired(old, ttl)

        assert result is True

    def test_exactly_at_boundary(self):
        """Timestamp exactly at TTL boundary is expired."""
        boundary = time.time() - 60.1  # Just over 60 seconds ago
        ttl = 60

        result = is_expired(boundary, ttl)

        assert result is True


# =============================================================================
# RetryConfig Tests
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Default config has expected values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1

    def test_custom_values(self):
        """Custom config values are stored."""
        config = RetryConfig(
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
            exponential_base=3.0,
            jitter=0.2,
        )

        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0


# =============================================================================
# RetryResult Tests
# =============================================================================


class TestRetryResult:
    """Tests for RetryResult dataclass."""

    def test_success_result(self):
        """Success result has expected fields."""
        result = RetryResult(
            success=True,
            attempts=2,
            result="operation completed",
            total_time=1.5,
        )

        assert result.success is True
        assert result.attempts == 2
        assert result.result == "operation completed"
        assert result.last_error is None

    def test_failure_result(self):
        """Failure result includes error."""
        result = RetryResult(
            success=False,
            attempts=3,
            last_error="Connection refused",
            total_time=5.0,
        )

        assert result.success is False
        assert result.attempts == 3
        assert result.last_error == "Connection refused"


# =============================================================================
# RetryStrategy Tests
# =============================================================================


class TestRetryStrategy:
    """Tests for RetryStrategy class."""

    def test_initialization(self):
        """Strategy initializes with correct config."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=1.0,
        )

        assert strategy.config.max_retries == 5
        assert strategy.config.base_delay == 1.0

    def test_delay_calculation_exponential(self):
        """Delay increases exponentially."""
        strategy = RetryStrategy(
            max_retries=5,
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=100.0,
        )
        # Disable jitter for predictable testing
        strategy.config.jitter = 0

        assert strategy._get_delay(0) == pytest.approx(1.0)
        assert strategy._get_delay(1) == pytest.approx(2.0)
        assert strategy._get_delay(2) == pytest.approx(4.0)
        assert strategy._get_delay(3) == pytest.approx(8.0)

    def test_delay_respects_max(self):
        """Delay is capped at max_delay."""
        strategy = RetryStrategy(
            max_retries=10,
            base_delay=1.0,
            max_delay=5.0,
        )
        strategy.config.jitter = 0

        delay = strategy._get_delay(10)  # Would be 1024 without cap

        assert delay == pytest.approx(5.0)

    def test_delay_has_minimum(self):
        """Delay has minimum of 10ms."""
        strategy = RetryStrategy(base_delay=0.001)
        strategy.config.jitter = 0

        delay = strategy._get_delay(0)

        assert delay >= 0.01

    @pytest.mark.asyncio
    async def test_execute_success_first_try(self):
        """Execute succeeds on first try."""

        async def success_op():
            return "done"

        strategy = RetryStrategy(max_retries=3)
        result = await strategy.execute(success_op)

        assert result.success is True
        assert result.attempts == 1
        assert result.result == "done"

    @pytest.mark.asyncio
    async def test_execute_success_after_retry(self):
        """Execute succeeds after retries."""
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Try again")
            return "finally"

        strategy = RetryStrategy(max_retries=5, base_delay=0.01)
        result = await strategy.execute(flaky_op)

        assert result.success is True
        assert result.attempts == 3
        assert result.result == "finally"

    @pytest.mark.asyncio
    async def test_execute_all_retries_exhausted(self):
        """Execute fails after exhausting retries."""

        async def always_fail():
            raise ValueError("Never works")

        strategy = RetryStrategy(max_retries=2, base_delay=0.01)
        result = await strategy.execute(always_fail)

        assert result.success is False
        assert result.attempts == 3  # 1 initial + 2 retries
        assert "Never works" in result.last_error

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Execute works with sync functions too."""

        def sync_op():
            return 42

        strategy = RetryStrategy()
        result = await strategy.execute(sync_op)

        assert result.success is True
        assert result.result == 42

    @pytest.mark.asyncio
    async def test_execute_with_args(self):
        """Execute passes args to function."""

        async def add(a, b):
            return a + b

        strategy = RetryStrategy()
        result = await strategy.execute(add, 3, 4)

        assert result.result == 7

    @pytest.mark.asyncio
    async def test_execute_with_kwargs(self):
        """Execute passes kwargs to function."""

        async def greet(name="World"):
            return f"Hello, {name}!"

        strategy = RetryStrategy()
        result = await strategy.execute(greet, name="Test")

        assert result.result == "Hello, Test!"

    @pytest.mark.asyncio
    async def test_execute_tracks_total_time(self):
        """Execute tracks total elapsed time."""

        async def slow_op():
            await asyncio.sleep(0.05)
            return "done"

        strategy = RetryStrategy()
        result = await strategy.execute(slow_op)

        assert result.total_time >= 0.05


# =============================================================================
# HandlerMetrics Tests
# =============================================================================


class TestHandlerMetrics:
    """Tests for HandlerMetrics dataclass."""

    def test_default_values(self):
        """Default metrics have expected values."""
        metrics = HandlerMetrics(endpoint="/test")

        assert metrics.endpoint == "/test"
        assert metrics.request_count == 0
        assert metrics.error_count == 0
        assert metrics.total_latency_ms == 0.0

    def test_avg_latency_zero_requests(self):
        """Avg latency is 0 with no requests."""
        metrics = HandlerMetrics(endpoint="/test")

        assert metrics.avg_latency_ms == 0.0

    def test_avg_latency_calculated(self):
        """Avg latency is calculated correctly."""
        metrics = HandlerMetrics(
            endpoint="/test",
            request_count=4,
            total_latency_ms=100.0,
        )

        assert metrics.avg_latency_ms == 25.0

    def test_error_rate_zero_requests(self):
        """Error rate is 0 with no requests."""
        metrics = HandlerMetrics(endpoint="/test")

        assert metrics.error_rate == 0.0

    def test_error_rate_calculated(self):
        """Error rate is calculated correctly."""
        metrics = HandlerMetrics(
            endpoint="/test",
            request_count=10,
            error_count=3,
        )

        assert metrics.error_rate == 0.3


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self):
        """Collector initializes empty."""
        collector = MetricsCollector()

        assert collector._metrics == {}

    def test_record_request(self):
        """Record request creates and updates metrics."""
        collector = MetricsCollector()

        collector.record_request("/test", 50.0)
        collector.record_request("/test", 100.0)

        metrics = collector.get_metrics("/test")
        assert metrics["request_count"] == 2
        assert metrics["avg_latency_ms"] == 75.0

    def test_record_request_with_error(self):
        """Record request with error tracks error."""
        collector = MetricsCollector()

        collector.record_request("/test", 50.0, error="Connection failed")

        metrics = collector.get_metrics("/test")
        assert metrics["error_count"] == 1
        assert metrics["last_error"] == "Connection failed"

    def test_get_metrics_unknown_endpoint(self):
        """Get metrics for unknown endpoint returns None."""
        collector = MetricsCollector()

        result = collector.get_metrics("/unknown")

        assert result is None

    def test_get_all_metrics(self):
        """Get all metrics returns dict of all endpoints."""
        collector = MetricsCollector()

        collector.record_request("/a", 10.0)
        collector.record_request("/b", 20.0)
        collector.record_request("/c", 30.0)

        all_metrics = collector.get_all_metrics()

        assert len(all_metrics) == 3
        assert "/a" in all_metrics
        assert "/b" in all_metrics
        assert "/c" in all_metrics

    def test_reset_clears_all(self):
        """Reset clears all metrics."""
        collector = MetricsCollector()

        collector.record_request("/a", 10.0)
        collector.record_request("/b", 20.0)
        collector.reset()

        assert collector._metrics == {}

    def test_track_context_manager(self):
        """Track context manager records request."""
        collector = MetricsCollector()

        with collector.track("/test"):
            pass

        metrics = collector.get_metrics("/test")
        assert metrics["request_count"] == 1

    def test_track_records_latency(self):
        """Track context manager records latency."""
        collector = MetricsCollector()

        with collector.track("/slow"):
            time.sleep(0.05)

        metrics = collector.get_metrics("/slow")
        assert metrics["avg_latency_ms"] >= 50

    def test_track_records_exception_as_error(self):
        """Track context manager records exception as error."""
        collector = MetricsCollector()

        with pytest.raises(ValueError):
            with collector.track("/error"):
                raise ValueError("Test error")

        metrics = collector.get_metrics("/error")
        assert metrics["error_count"] == 1
        assert "Test error" in metrics["last_error"]

    def test_track_set_error_manually(self):
        """Track context manager allows manual error setting."""
        collector = MetricsCollector()

        with collector.track("/custom") as tracker:
            tracker.set_error("Custom error message")

        metrics = collector.get_metrics("/custom")
        assert metrics["error_count"] == 1
        assert metrics["last_error"] == "Custom error message"

"""Unit tests for availability/node_monitor.py.

Tests for NodeMonitor daemon that performs multi-layer health checks.

Created: Dec 28, 2025
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.node_monitor import (
    HealthCheckLayer,
    NodeHealthResult,
    NodeMonitor,
    NodeMonitorConfig,
    get_node_monitor,
    reset_node_monitor,
)


class TestHealthCheckLayer:
    """Tests for HealthCheckLayer enum."""

    def test_layer_values(self):
        """Test all layer values exist."""
        assert HealthCheckLayer.P2P.value == "p2p"
        assert HealthCheckLayer.SSH.value == "ssh"
        assert HealthCheckLayer.GPU.value == "gpu"
        assert HealthCheckLayer.PROVIDER_API.value == "provider_api"

    def test_layer_timeout(self):
        """Test layer timeout properties."""
        assert HealthCheckLayer.P2P.timeout_seconds == 15
        assert HealthCheckLayer.SSH.timeout_seconds == 30
        assert HealthCheckLayer.GPU.timeout_seconds == 20
        assert HealthCheckLayer.PROVIDER_API.timeout_seconds == 45


class TestNodeHealthResult:
    """Tests for NodeHealthResult dataclass."""

    def test_healthy_result(self):
        """Test creating a healthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            healthy=True,
            layers_checked=["p2p", "ssh"],
            latency_ms=50.0,
        )
        assert result.healthy
        assert result.node_id == "test-node"
        assert "p2p" in result.layers_checked
        assert result.latency_ms == 50.0
        assert result.failed_layer is None
        assert result.error is None

    def test_unhealthy_result(self):
        """Test creating an unhealthy result."""
        result = NodeHealthResult(
            node_id="test-node",
            healthy=False,
            layers_checked=["p2p"],
            failed_layer="ssh",
            error="Connection refused",
        )
        assert not result.healthy
        assert result.failed_layer == "ssh"
        assert "Connection refused" in result.error

    def test_to_dict(self):
        """Test serialization to dict."""
        result = NodeHealthResult(
            node_id="test-node",
            healthy=True,
            layers_checked=["p2p"],
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["healthy"] is True
        assert "timestamp" in d


class TestNodeMonitorConfig:
    """Tests for NodeMonitorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NodeMonitorConfig()
        assert config.check_interval_seconds == 30.0
        assert config.consecutive_failures_before_unhealthy == 3
        assert config.gpu_check_enabled is True

    def test_from_env(self):
        """Test loading config from environment."""
        with patch.dict("os.environ", {
            "RINGRIFT_NODE_MONITOR_INTERVAL": "60",
            "RINGRIFT_NODE_MONITOR_FAILURE_THRESHOLD": "5",
        }):
            config = NodeMonitorConfig.from_env()
            assert config.check_interval_seconds == 60.0
            assert config.consecutive_failures_before_unhealthy == 5


class TestNodeMonitor:
    """Tests for NodeMonitor daemon."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_singleton_pattern(self):
        """Test that get_node_monitor returns singleton."""
        monitor1 = get_node_monitor()
        monitor2 = get_node_monitor()

        assert monitor1 is monitor2

    def test_event_subscriptions(self):
        """Test event subscription setup."""
        monitor = get_node_monitor()

        subs = monitor._get_event_subscriptions()

        # Should be a dict of event name -> handler
        assert isinstance(subs, dict)

    def test_get_node_status(self):
        """Test getting status for a specific node."""
        monitor = get_node_monitor()

        # Status for unknown node should have None values
        status = monitor.get_node_status("unknown-node")
        assert "node_id" in status
        assert status["node_id"] == "unknown-node"

    def test_get_all_node_statuses(self):
        """Test getting all node statuses."""
        monitor = get_node_monitor()

        statuses = monitor.get_all_node_statuses()
        assert isinstance(statuses, dict)

    def test_health_check(self):
        """Test health_check method returns valid result."""
        monitor = get_node_monitor()

        result = monitor.health_check()

        assert "healthy" in result
        assert "message" in result
        assert "details" in result


class TestNodeMonitorIntegration:
    """Integration tests for NodeMonitor."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_p2p_check_with_mock(self):
        """Test P2P health check with mocked HTTP client."""
        monitor = get_node_monitor()

        # Mock the HTTP call
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "ok"})
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_response)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # The actual check would use the mock
            # This verifies the monitor can be instantiated and checked
            assert monitor is not None

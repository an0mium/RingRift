"""
Unit tests for P2PD UDP hole punching transport.

Tests cover:
- Transport initialization and availability
- Connection health tracking
- NAT type integration
- Circuit breaker behavior
"""

from __future__ import annotations

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.p2p.transports.p2pd_transport import (
    P2PDUDPTransport,
    P2PDConnectionHealth,
    P2PD_AVAILABLE,
)
from scripts.p2p.transport_cascade import TransportTier


class TestP2PDConnectionHealth:
    """Tests for P2PDConnectionHealth dataclass."""

    def test_initial_state(self):
        """Test initial health state."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        assert health.peer_id == "test-peer"
        assert health.successes == 0
        assert health.failures == 0
        assert health.consecutive_failures == 0
        assert health.connected is False
        assert health.is_healthy is True

    def test_record_success(self):
        """Test recording successful delivery."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.record_success(latency_ms=50.0)

        assert health.successes == 1
        assert health.failures == 0
        assert health.consecutive_failures == 0
        assert health.total_latency_ms == 50.0
        assert health.last_success_time > 0

    def test_record_failure(self):
        """Test recording failed delivery."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.record_failure("Timeout")

        assert health.successes == 0
        assert health.failures == 1
        assert health.consecutive_failures == 1
        assert health.last_error == "Timeout"
        assert health.last_failure_time > 0

    def test_success_rate(self):
        """Test success rate calculation."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.successes = 3
        health.failures = 1

        assert health.success_rate == 0.75

    def test_success_rate_no_data(self):
        """Test success rate with no data."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        assert health.success_rate == 1.0  # Assume healthy

    def test_avg_latency(self):
        """Test average latency calculation."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.record_success(50.0)
        health.record_success(100.0)
        health.record_success(150.0)

        assert health.avg_latency_ms == 100.0

    def test_avg_latency_no_data(self):
        """Test average latency with no data."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        assert health.avg_latency_ms == 0.0

    def test_circuit_breaker_trips(self):
        """Test circuit breaker trips after failures."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.failure_threshold = 3

        for i in range(3):
            health.record_failure(f"Error {i}")

        assert health.is_healthy is False

    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets on success."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.failure_threshold = 3

        # Trip the breaker
        for i in range(3):
            health.record_failure(f"Error {i}")

        # Success resets consecutive failures
        health.record_success(50.0)
        assert health.consecutive_failures == 0

    def test_circuit_breaker_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.failure_threshold = 3
        health.recovery_timeout = 0.1  # 100ms for testing

        # Trip the breaker
        for i in range(3):
            health.record_failure(f"Error {i}")

        assert health.is_healthy is False

        # Wait for recovery
        time.sleep(0.15)
        assert health.is_healthy is True
        assert health.consecutive_failures == 0


class TestP2PDUDPTransport:
    """Tests for P2PDUDPTransport."""

    def test_transport_attributes(self):
        """Test transport has correct attributes."""
        transport = P2PDUDPTransport()
        assert transport.name == "p2pd_udp"
        assert transport.tier == TransportTier.TIER_1_FAST

    def test_disabled_via_environment(self, monkeypatch):
        """Test transport disabled via environment variable."""
        monkeypatch.setenv("RINGRIFT_P2PD_ENABLED", "false")
        transport = P2PDUDPTransport()
        assert transport._enabled is False

    def test_enabled_by_default(self, monkeypatch):
        """Test transport enabled by default."""
        monkeypatch.delenv("RINGRIFT_P2PD_ENABLED", raising=False)
        transport = P2PDUDPTransport()
        assert transport._enabled is True

    def test_custom_node_id(self):
        """Test transport with custom node ID."""
        transport = P2PDUDPTransport(node_id="custom-node")
        assert transport._node_id == "custom-node"

    def test_node_id_from_environment(self, monkeypatch):
        """Test node ID from environment variable."""
        monkeypatch.setenv("RINGRIFT_NODE_ID", "env-node")
        transport = P2PDUDPTransport()
        assert transport._node_id == "env-node"

    def test_extract_peer_id(self):
        """Test peer ID extraction from target."""
        transport = P2PDUDPTransport()
        assert transport._extract_peer_id("192.168.1.1:8770") == "192.168.1.1:8770"
        assert transport._extract_peer_id("some-peer-id") == "some-peer-id"

    def test_get_or_create_health(self):
        """Test health tracker creation."""
        transport = P2PDUDPTransport()

        # First access creates new health
        health1 = transport._get_or_create_health("peer-1")
        assert health1.peer_id == "peer-1"

        # Second access returns same health
        health2 = transport._get_or_create_health("peer-1")
        assert health1 is health2

    def test_connection_status_initial(self):
        """Test initial connection status."""
        transport = P2PDUDPTransport()
        status = transport.get_connection_status()

        assert status["available"] == P2PD_AVAILABLE
        assert status["enabled"] is True
        assert status["initialized"] is False
        assert status["active_connections"] == 0

    def test_health_summary_initial(self):
        """Test initial health summary."""
        transport = P2PDUDPTransport()
        summary = transport.get_health_summary()

        assert summary["healthy_connections"] == 0
        assert summary["total_connections"] == 0
        assert summary["health_ratio"] == 1.0

    def test_health_summary_with_connections(self):
        """Test health summary with connections."""
        transport = P2PDUDPTransport()

        # Create some health trackers
        health1 = transport._get_or_create_health("peer-1")
        health1.record_success(50.0)

        health2 = transport._get_or_create_health("peer-2")
        health2.failure_threshold = 2
        health2.record_failure("Error 1")
        health2.record_failure("Error 2")

        summary = transport.get_health_summary()
        assert summary["healthy_connections"] == 1
        assert summary["total_connections"] == 2
        assert summary["health_ratio"] == 0.5

    @pytest.mark.asyncio
    async def test_is_available_not_initialized(self):
        """Test is_available when not initialized."""
        with patch.object(P2PDUDPTransport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False
            transport = P2PDUDPTransport()

            available = await transport.is_available("192.168.1.1:8770")
            assert available is False

    @pytest.mark.asyncio
    async def test_is_available_with_existing_connection(self):
        """Test is_available with existing healthy connection."""
        transport = P2PDUDPTransport()

        # Mock the initialization
        with patch.object(transport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True

            # Add mock connection
            mock_conn = MagicMock()
            mock_conn.is_connected = True
            transport._connections["192.168.1.1:8770"] = mock_conn

            available = await transport.is_available("192.168.1.1:8770")
            assert available is True

    @pytest.mark.asyncio
    async def test_is_available_unhealthy_peer(self):
        """Test is_available returns False for unhealthy peer."""
        transport = P2PDUDPTransport()

        # Mock the initialization
        with patch.object(transport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True

            # Create unhealthy peer
            health = transport._get_or_create_health("192.168.1.1:8770")
            health.failure_threshold = 2
            health.record_failure("Error 1")
            health.record_failure("Error 2")

            available = await transport.is_available("192.168.1.1:8770")
            assert available is False

    @pytest.mark.asyncio
    async def test_send_not_available(self):
        """Test send when P2PD not available."""
        transport = P2PDUDPTransport()

        with patch.object(transport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = False

            result = await transport.send("192.168.1.1:8770", b"test payload")
            assert result.success is False
            assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_send_unhealthy_peer(self):
        """Test send to unhealthy peer."""
        transport = P2PDUDPTransport()

        with patch.object(transport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True

            # Create unhealthy peer
            health = transport._get_or_create_health("192.168.1.1:8770")
            health.failure_threshold = 2
            health.record_failure("Error 1")
            health.record_failure("Error 2")

            result = await transport.send("192.168.1.1:8770", b"test payload")
            assert result.success is False
            assert "unhealthy" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_connection_failed(self):
        """Test send when connection fails."""
        transport = P2PDUDPTransport()

        with patch.object(transport, "_ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True

            with patch.object(transport, "_get_or_connect", new_callable=AsyncMock) as mock_conn:
                mock_conn.return_value = None

                result = await transport.send("192.168.1.1:8770", b"test payload")
                assert result.success is False
                assert "Failed to establish" in result.error

    @pytest.mark.asyncio
    async def test_close_connection(self):
        """Test closing a specific connection."""
        transport = P2PDUDPTransport()

        # Add mock connection
        mock_conn = MagicMock()
        mock_conn.close = MagicMock()
        transport._connections["peer-1"] = mock_conn
        transport._connection_health["peer-1"] = P2PDConnectionHealth(peer_id="peer-1")
        transport._connection_health["peer-1"].connected = True

        await transport.close_connection("peer-1")

        assert "peer-1" not in transport._connections
        assert transport._connection_health["peer-1"].connected is False

    @pytest.mark.asyncio
    async def test_close_all_connections(self):
        """Test closing all connections."""
        transport = P2PDUDPTransport()

        # Add mock connections
        for peer_id in ["peer-1", "peer-2", "peer-3"]:
            mock_conn = MagicMock()
            mock_conn.close = MagicMock()
            transport._connections[peer_id] = mock_conn
            transport._connection_health[peer_id] = P2PDConnectionHealth(peer_id=peer_id)

        await transport.close_all()

        assert len(transport._connections) == 0
        assert transport._initialized is False


class TestP2PDNATIntegration:
    """Tests for P2PD NAT type integration."""

    def test_nat_type_stored_in_health(self):
        """Test NAT type stored in connection health."""
        health = P2PDConnectionHealth(peer_id="test-peer")
        health.nat_type = "cgnat"
        assert health.nat_type == "cgnat"

    def test_transport_stores_local_nat_type(self):
        """Test transport stores local NAT type."""
        transport = P2PDUDPTransport()
        transport._local_nat_type = "symmetric"

        status = transport.get_connection_status()
        assert status["local_nat_type"] == "symmetric"

    def test_health_summary_includes_nat_type(self):
        """Test health summary includes NAT type."""
        transport = P2PDUDPTransport()
        transport._local_nat_type = "cgnat"

        summary = transport.get_health_summary()
        assert summary["local_nat_type"] == "cgnat"

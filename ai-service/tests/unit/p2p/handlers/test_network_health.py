"""Tests for NetworkHealthMixin.

December 30, 2025: Tests for the P2P network health endpoint that
cross-verifies P2P mesh connectivity against Tailscale status.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import web
from aiohttp.test_utils import TestClient, AioHTTPTestCase

from scripts.p2p.handlers.network_health import (
    NetworkHealthMixin,
    setup_network_health_routes,
)
from scripts.p2p.handlers.base import BaseP2PHandler


class MockPeerInfo:
    """Mock PeerInfo for testing."""

    def __init__(self, node_id: str, alive: bool = True):
        self.node_id = node_id
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class MockOrchestrator(NetworkHealthMixin):
    """Mock orchestrator with required attributes for NetworkHealthMixin."""

    def __init__(self):
        self.node_id = "test-node"
        self.auth_token = None
        self.peers = {}
        self._tailscale_discovery_loop = None

    def _load_distributed_hosts(self) -> dict:
        """Return mock host configuration."""
        return {
            "hosts": {
                "node-a": {
                    "tailscale_ip": "100.100.1.1",
                    "p2p_enabled": True,
                    "p2p_port": 8770,
                },
                "node-b": {
                    "tailscale_ip": "100.100.1.2",
                    "p2p_enabled": True,
                    "p2p_port": 8770,
                },
                "node-c": {
                    "tailscale_ip": "100.100.1.3",
                    "p2p_enabled": True,
                    "p2p_port": 8770,
                },
            }
        }

    async def _get_tailscale_status(self) -> dict[str, bool]:
        """Return mock Tailscale status."""
        return {
            "100.100.1.1": True,
            "100.100.1.2": True,
            "100.100.1.3": False,  # Offline
        }

    async def _reconnect_discovered_peer(
        self, node_id: str, host: str, port: int
    ) -> bool:
        """Mock reconnection - always succeed."""
        return True


class TestNetworkHealthMixin:
    """Unit tests for NetworkHealthMixin."""

    @pytest.fixture
    def orchestrator(self):
        """Create mock orchestrator."""
        return MockOrchestrator()

    @pytest.fixture
    def app(self, orchestrator):
        """Create aiohttp app with network health routes."""
        app = web.Application()
        setup_network_health_routes(app, orchestrator)
        return app

    @pytest.mark.asyncio
    async def test_handle_network_health_all_connected(self, orchestrator):
        """Test /network/health when all online peers are connected."""
        # Setup: 2 Tailscale online, both in P2P
        orchestrator.peers = {
            "node-a": MockPeerInfo("node-a", alive=True),
            "node-b": MockPeerInfo("node-b", alive=True),
        }

        # Create mock request
        request = MagicMock()

        # Call handler
        response = await orchestrator.handle_network_health(request)

        # Parse response
        import json
        data = json.loads(response.body)

        # Verify
        assert data["tailscale_online"] == 2
        assert data["p2p_connected"] == 2
        assert data["missing_from_p2p"] == 0
        assert data["missing_peers"] == []
        assert data["health_score"] == 1.0
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_handle_network_health_missing_peers(self, orchestrator):
        """Test /network/health when some online peers are missing from P2P."""
        # Setup: 2 Tailscale online, only 1 in P2P
        orchestrator.peers = {
            "node-a": MockPeerInfo("node-a", alive=True),
        }

        # Create mock request
        request = MagicMock()

        # Call handler
        response = await orchestrator.handle_network_health(request)

        # Parse response
        import json
        data = json.loads(response.body)

        # Verify
        assert data["tailscale_online"] == 2
        assert data["p2p_connected"] == 1
        assert data["missing_from_p2p"] == 1
        assert "node-b" in data["missing_peers"]
        assert data["health_score"] == 0.5
        assert data["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_handle_network_reconnect(self, orchestrator):
        """Test /network/reconnect attempts to reconnect missing peers."""
        # Setup: 2 Tailscale online, only 1 in P2P
        orchestrator.peers = {
            "node-a": MockPeerInfo("node-a", alive=True),
        }

        # Create mock request with no body
        request = MagicMock()
        request.json = AsyncMock(side_effect=Exception("No body"))

        # Call handler
        response = await orchestrator.handle_network_reconnect(request)

        # Parse response
        import json
        data = json.loads(response.body)

        # Verify reconnection was attempted
        assert data["attempted"] == 1
        assert "node-b" in data["attempted_peers"]
        assert "node-b" in data["reconnected"]
        assert data["failed"] == []
        assert data["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_handle_network_status(self, orchestrator):
        """Test /network/status returns detailed peer info."""
        # Setup
        orchestrator.peers = {
            "node-a": MockPeerInfo("node-a", alive=True),
            "node-b": MockPeerInfo("node-b", alive=False),
        }

        # Create mock request
        request = MagicMock()

        # Call handler
        response = await orchestrator.handle_network_status(request)

        # Parse response
        import json
        data = json.loads(response.body)

        # Verify summary
        assert data["summary"]["total_configured"] == 3
        assert data["summary"]["tailscale_online"] == 2
        assert data["summary"]["p2p_alive"] == 1
        assert data["summary"]["missing_from_p2p"] == 1

        # Verify peer details
        peers = {p["node_id"]: p for p in data["peers"]}
        assert peers["node-a"]["tailscale_online"] is True
        assert peers["node-a"]["p2p_status"] == "alive"
        assert peers["node-b"]["tailscale_online"] is True
        assert peers["node-b"]["p2p_status"] == "dead"
        assert peers["node-c"]["tailscale_online"] is False


class TestSetupNetworkHealthRoutes:
    """Tests for route registration."""

    def test_routes_registered(self):
        """Test that all routes are registered."""
        app = web.Application()
        handler = MockOrchestrator()

        setup_network_health_routes(app, handler)

        # Check routes exist
        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/network/health" in routes
        assert "/network/reconnect" in routes
        assert "/network/status" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

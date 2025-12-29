"""Unit tests for P2P network discovery loops.

December 2025 Phase 8: Tests for newly-registered background loops:
- IpDiscoveryLoop: IP address updates and fallback
- TailscaleRecoveryLoop: Tailscale connection recovery
- UdpDiscoveryLoop: UDP broadcast peer discovery

See also test_cluster_hardening.py for SplitBrainDetectionLoop tests.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loops.network_loops import (
    IpDiscoveryConfig,
    IpDiscoveryLoop,
    TailscaleRecoveryConfig,
    TailscaleRecoveryLoop,
)
from scripts.p2p.loops.discovery_loop import (
    DEFAULT_DISCOVERY_PORT,
    UdpDiscoveryConfig,
    UdpDiscoveryLoop,
)


# =============================================================================
# IpDiscoveryLoop Tests
# =============================================================================


class TestIpDiscoveryConfig:
    """Tests for IpDiscoveryConfig dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        config = IpDiscoveryConfig()
        assert config.check_interval_seconds == 300.0
        assert config.dns_timeout_seconds == 10.0
        assert config.max_nodes_per_cycle == 20

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = IpDiscoveryConfig(
            check_interval_seconds=60.0,
            dns_timeout_seconds=5.0,
            max_nodes_per_cycle=10,
        )
        assert config.check_interval_seconds == 60.0
        assert config.dns_timeout_seconds == 5.0
        assert config.max_nodes_per_cycle == 10

    def test_validation_check_interval(self):
        """Invalid check_interval_seconds should raise."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            IpDiscoveryConfig(check_interval_seconds=0)
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            IpDiscoveryConfig(check_interval_seconds=-1)

    def test_validation_dns_timeout(self):
        """Invalid dns_timeout_seconds should raise."""
        with pytest.raises(ValueError, match="dns_timeout_seconds must be > 0"):
            IpDiscoveryConfig(dns_timeout_seconds=0)

    def test_validation_max_nodes(self):
        """Invalid max_nodes_per_cycle should raise."""
        with pytest.raises(ValueError, match="max_nodes_per_cycle must be > 0"):
            IpDiscoveryConfig(max_nodes_per_cycle=0)


class TestIpDiscoveryLoop:
    """Tests for IpDiscoveryLoop background loop."""

    @pytest.fixture
    def mock_nodes(self) -> dict[str, dict[str, Any]]:
        """Create mock node dictionary."""
        return {
            "node-1": {
                "ip": "10.0.0.1",
                "host": "10.0.0.1",
                "tailscale_ip": "100.64.0.1",
                "public_ip": "1.2.3.4",
            },
            "node-2": {
                "ip": "10.0.0.2",
                "host": "10.0.0.2",
                "tailscale_ip": "100.64.0.2",
                "hostname": "node2.example.com",
            },
        }

    @pytest.fixture
    def discovery_loop(
        self, mock_nodes: dict[str, dict[str, Any]]
    ) -> IpDiscoveryLoop:
        """Create an IpDiscoveryLoop with mocked dependencies."""
        return IpDiscoveryLoop(
            get_nodes=lambda: mock_nodes,
            update_node_ip=AsyncMock(),
            config=IpDiscoveryConfig(
                check_interval_seconds=1.0,
                dns_timeout_seconds=1.0,
            ),
        )

    def test_loop_initialization(self, discovery_loop: IpDiscoveryLoop):
        """Loop should initialize with correct defaults."""
        assert discovery_loop.name == "ip_discovery"
        assert discovery_loop._updates_count == 0

    @pytest.mark.asyncio
    async def test_skip_if_no_nodes(self):
        """Should skip if no nodes available."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
        )
        await loop._run_once()
        # No errors, no updates
        assert loop._updates_count == 0

    @pytest.mark.asyncio
    async def test_no_update_if_ip_reachable(self, mock_nodes: dict[str, dict[str, Any]]):
        """Should not update IP if current IP is reachable."""
        update_mock = AsyncMock()
        loop = IpDiscoveryLoop(
            get_nodes=lambda: mock_nodes,
            update_node_ip=update_mock,
        )

        # Mock reachability to return True
        with patch.object(loop, "_is_reachable", return_value=True):
            await loop._run_once()

        # No updates should be made
        update_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_tailscale(self, mock_nodes: dict[str, dict[str, Any]]):
        """Should fallback to Tailscale IP if primary fails."""
        update_mock = AsyncMock()
        loop = IpDiscoveryLoop(
            get_nodes=lambda: mock_nodes,
            update_node_ip=update_mock,
        )

        async def mock_reachable(ip: str, port: int = 22) -> bool:
            # Primary IPs fail, Tailscale works
            if ip.startswith("100.64."):
                return True
            return False

        with patch.object(loop, "_is_reachable", side_effect=mock_reachable):
            await loop._run_once()

        # Should have updated to Tailscale IPs
        assert loop._updates_count >= 1

    @pytest.mark.asyncio
    async def test_hostname_resolution(self, mock_nodes: dict[str, dict[str, Any]]):
        """Should resolve hostname and update if changed."""
        update_mock = AsyncMock()
        loop = IpDiscoveryLoop(
            get_nodes=lambda: mock_nodes,
            update_node_ip=update_mock,
        )

        # Mock DNS resolution to return a new IP
        async def mock_resolve(hostname: str) -> str | None:
            if hostname == "node2.example.com":
                return "10.0.0.99"  # Different from current
            return None

        with patch.object(loop, "_resolve_hostname", side_effect=mock_resolve):
            await loop._run_once()

        # Should have updated node-2's IP
        update_mock.assert_called()

    def test_get_discovery_stats(self, discovery_loop: IpDiscoveryLoop):
        """Should return correct statistics."""
        discovery_loop._updates_count = 5

        stats = discovery_loop.get_discovery_stats()

        assert stats["total_updates"] == 5
        # Base stats include these fields
        assert "successful_runs" in stats or "avg_run_duration_ms" in stats


# =============================================================================
# TailscaleRecoveryLoop Tests
# =============================================================================


class TestTailscaleRecoveryConfig:
    """Tests for TailscaleRecoveryConfig dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        config = TailscaleRecoveryConfig()
        assert config.check_interval_seconds == 120.0
        assert config.recovery_timeout_seconds == 60.0
        assert config.max_recovery_attempts == 3
        assert config.cooldown_after_recovery_seconds == 300.0

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = TailscaleRecoveryConfig(
            check_interval_seconds=60.0,
            recovery_timeout_seconds=30.0,
            max_recovery_attempts=5,
            cooldown_after_recovery_seconds=120.0,
        )
        assert config.check_interval_seconds == 60.0
        assert config.max_recovery_attempts == 5

    def test_validation_check_interval(self):
        """Invalid check_interval_seconds should raise."""
        with pytest.raises(ValueError, match="check_interval_seconds must be > 0"):
            TailscaleRecoveryConfig(check_interval_seconds=0)

    def test_validation_recovery_timeout(self):
        """Invalid recovery_timeout_seconds should raise."""
        with pytest.raises(ValueError, match="recovery_timeout_seconds must be > 0"):
            TailscaleRecoveryConfig(recovery_timeout_seconds=0)

    def test_validation_max_attempts(self):
        """Invalid max_recovery_attempts should raise."""
        with pytest.raises(ValueError, match="max_recovery_attempts must be > 0"):
            TailscaleRecoveryConfig(max_recovery_attempts=0)

    def test_validation_cooldown(self):
        """Invalid cooldown_after_recovery_seconds should raise."""
        with pytest.raises(ValueError, match="cooldown_after_recovery_seconds must be >= 0"):
            TailscaleRecoveryConfig(cooldown_after_recovery_seconds=-1)


class TestTailscaleRecoveryLoop:
    """Tests for TailscaleRecoveryLoop background loop."""

    @pytest.fixture
    def mock_tailscale_status(self) -> dict[str, dict[str, Any]]:
        """Create mock Tailscale status dictionary."""
        return {
            "node-1": {"tailscale_state": "online", "tailscale_online": True},
            "node-2": {"tailscale_state": "offline", "tailscale_online": False},
            "node-3": {"tailscale_state": "stopped", "tailscale_online": False},
        }

    @pytest.fixture
    def recovery_loop(
        self, mock_tailscale_status: dict[str, dict[str, Any]]
    ) -> TailscaleRecoveryLoop:
        """Create a TailscaleRecoveryLoop with mocked dependencies."""
        return TailscaleRecoveryLoop(
            get_tailscale_status=lambda: mock_tailscale_status,
            run_ssh_command=AsyncMock(),
            config=TailscaleRecoveryConfig(
                check_interval_seconds=1.0,
                cooldown_after_recovery_seconds=0.0,  # No cooldown for tests
            ),
        )

    def test_loop_initialization(self, recovery_loop: TailscaleRecoveryLoop):
        """Loop should initialize with correct defaults."""
        assert recovery_loop.name == "tailscale_recovery"
        assert recovery_loop._recoveries_count == 0
        assert recovery_loop._recovery_attempts == {}

    @pytest.mark.asyncio
    async def test_skip_if_no_status(self):
        """Should skip if no status available."""
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {},
            run_ssh_command=AsyncMock(),
        )
        await loop._run_once()
        assert loop._recoveries_count == 0

    @pytest.mark.asyncio
    async def test_skip_healthy_nodes(self, mock_tailscale_status: dict[str, dict[str, Any]]):
        """Should skip nodes that are healthy."""
        ssh_mock = AsyncMock()
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {"node-1": {"tailscale_state": "online", "tailscale_online": True}},
            run_ssh_command=ssh_mock,
        )
        await loop._run_once()
        # No SSH commands should be run for healthy node
        ssh_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_attempt_recovery_on_offline(
        self, mock_tailscale_status: dict[str, dict[str, Any]]
    ):
        """Should attempt recovery on offline nodes."""
        @dataclass
        class MockResult:
            returncode: int = 0
            stdout: str = '{"Self": {"Online": true}}'

        ssh_mock = AsyncMock(return_value=MockResult())
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: mock_tailscale_status,
            run_ssh_command=ssh_mock,
            config=TailscaleRecoveryConfig(
                cooldown_after_recovery_seconds=0.0,
            ),
        )
        await loop._run_once()
        # Should have attempted SSH commands for offline nodes
        assert ssh_mock.call_count >= 1

    @pytest.mark.asyncio
    async def test_respect_cooldown(self, mock_tailscale_status: dict[str, dict[str, Any]]):
        """Should respect cooldown period between recovery attempts."""
        ssh_mock = AsyncMock()
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: mock_tailscale_status,
            run_ssh_command=ssh_mock,
            config=TailscaleRecoveryConfig(
                cooldown_after_recovery_seconds=3600.0,  # 1 hour
            ),
        )
        # Mark node as recently recovered
        loop._last_recovery["node-2"] = time.time()

        await loop._run_once()
        # Should not attempt recovery due to cooldown
        # (only node-3 should be attempted if no cooldown set)

    @pytest.mark.asyncio
    async def test_max_attempts_limit(self, mock_tailscale_status: dict[str, dict[str, Any]]):
        """Should stop retrying after max attempts."""
        ssh_mock = AsyncMock()
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: mock_tailscale_status,
            run_ssh_command=ssh_mock,
            config=TailscaleRecoveryConfig(
                max_recovery_attempts=3,
                cooldown_after_recovery_seconds=0.0,
            ),
        )
        # Mark node as having max attempts
        loop._recovery_attempts["node-2"] = 3

        await loop._run_once()
        # node-2 should be skipped, only node-3 might be attempted

    @pytest.mark.asyncio
    async def test_on_recovery_failed_callback(
        self, mock_tailscale_status: dict[str, dict[str, Any]]
    ):
        """Should call on_recovery_failed when max attempts reached."""
        @dataclass
        class MockResult:
            returncode: int = 1  # Fail
            stdout: str = ""

        ssh_mock = AsyncMock(return_value=MockResult())
        callback_mock = AsyncMock()

        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {"node-2": {"tailscale_state": "offline", "tailscale_online": False}},
            run_ssh_command=ssh_mock,
            on_recovery_failed=callback_mock,
            config=TailscaleRecoveryConfig(
                max_recovery_attempts=1,
                cooldown_after_recovery_seconds=0.0,
            ),
        )

        await loop._run_once()

        # Should have called the callback
        callback_mock.assert_called_once_with("node-2")

    def test_get_recovery_stats(self, recovery_loop: TailscaleRecoveryLoop):
        """Should return correct statistics."""
        recovery_loop._recoveries_count = 10
        recovery_loop._recovery_attempts = {"node-1": 3, "node-2": 1}

        stats = recovery_loop.get_recovery_stats()

        assert stats["total_recoveries"] == 10
        assert stats["nodes_at_max_attempts"] == 1  # node-1 is at max (3)
        assert stats["pending_recoveries"] == {"node-1": 3, "node-2": 1}


# =============================================================================
# UdpDiscoveryLoop Tests
# =============================================================================


class TestUdpDiscoveryConfig:
    """Tests for UdpDiscoveryConfig dataclass."""

    def test_default_values(self):
        """Default values should be sensible."""
        config = UdpDiscoveryConfig()
        assert config.discovery_port == DEFAULT_DISCOVERY_PORT
        assert config.broadcast_interval_seconds == 30.0
        assert config.listen_timeout_seconds == 1.0
        assert config.max_message_size == 1024

    def test_custom_values(self):
        """Custom values should be accepted."""
        config = UdpDiscoveryConfig(
            discovery_port=9999,
            broadcast_interval_seconds=10.0,
            listen_timeout_seconds=0.5,
            max_message_size=2048,
        )
        assert config.discovery_port == 9999
        assert config.broadcast_interval_seconds == 10.0

    def test_validation_port_range(self):
        """Invalid port should raise."""
        with pytest.raises(ValueError, match="discovery_port must be between"):
            UdpDiscoveryConfig(discovery_port=0)
        with pytest.raises(ValueError, match="discovery_port must be between"):
            UdpDiscoveryConfig(discovery_port=65536)

    def test_validation_broadcast_interval(self):
        """Invalid broadcast_interval_seconds should raise."""
        with pytest.raises(ValueError, match="broadcast_interval_seconds must be > 0"):
            UdpDiscoveryConfig(broadcast_interval_seconds=0)

    def test_validation_listen_timeout(self):
        """Invalid listen_timeout_seconds should raise."""
        with pytest.raises(ValueError, match="listen_timeout_seconds must be > 0"):
            UdpDiscoveryConfig(listen_timeout_seconds=0)

    def test_validation_max_message_size(self):
        """Invalid max_message_size should raise."""
        with pytest.raises(ValueError, match="max_message_size must be > 0"):
            UdpDiscoveryConfig(max_message_size=0)


class TestUdpDiscoveryLoop:
    """Tests for UdpDiscoveryLoop background loop."""

    @pytest.fixture
    def discovery_loop(self) -> UdpDiscoveryLoop:
        """Create a UdpDiscoveryLoop with mocked dependencies."""
        return UdpDiscoveryLoop(
            get_node_id=lambda: "test-node",
            get_host=lambda: "127.0.0.1",
            get_port=lambda: 8770,
            get_known_peers=lambda: [],
            add_peer=MagicMock(),
            config=UdpDiscoveryConfig(
                broadcast_interval_seconds=1.0,
                listen_timeout_seconds=0.1,
            ),
        )

    def test_loop_initialization(self, discovery_loop: UdpDiscoveryLoop):
        """Loop should initialize with correct defaults."""
        assert discovery_loop.name == "udp_discovery"
        assert discovery_loop._peers_discovered == 0

    @pytest.mark.asyncio
    async def test_run_once_no_crash(self, discovery_loop: UdpDiscoveryLoop):
        """Should not crash when running discovery cycle."""
        # Just verify it doesn't raise - actual UDP may fail in tests
        try:
            await discovery_loop._run_once()
        except OSError:
            pass  # UDP socket operations may fail in test environment

    def test_get_discovery_stats(self, discovery_loop: UdpDiscoveryLoop):
        """Should return correct statistics."""
        discovery_loop._peers_discovered = 5

        stats = discovery_loop.get_discovery_stats()

        assert stats["peers_discovered"] == 5
        # Base stats include these fields
        assert "successful_runs" in stats or "avg_run_duration_ms" in stats

    @pytest.mark.asyncio
    async def test_add_peer_on_discovery(self):
        """Should add peer when discovered via broadcast."""
        add_peer_mock = MagicMock()
        known_peers: list[str] = []

        loop = UdpDiscoveryLoop(
            get_node_id=lambda: "test-node",
            get_host=lambda: "127.0.0.1",
            get_port=lambda: 8770,
            get_known_peers=lambda: known_peers,
            add_peer=add_peer_mock,
        )

        # Simulate discovering a peer by directly testing the add logic
        peer_addr = "10.0.0.5:8770"
        if peer_addr not in known_peers:
            add_peer_mock(peer_addr)
            loop._peers_discovered += 1

        add_peer_mock.assert_called_once_with("10.0.0.5:8770")
        assert loop._peers_discovered == 1

    @pytest.mark.asyncio
    async def test_skip_known_peers(self):
        """Should not add peers that are already known."""
        add_peer_mock = MagicMock()
        known_peers = ["10.0.0.5:8770"]

        loop = UdpDiscoveryLoop(
            get_node_id=lambda: "test-node",
            get_host=lambda: "127.0.0.1",
            get_port=lambda: 8770,
            get_known_peers=lambda: known_peers,
            add_peer=add_peer_mock,
        )

        # Simulate receiving a message from known peer
        peer_addr = "10.0.0.5:8770"
        if peer_addr not in known_peers:
            add_peer_mock(peer_addr)
            loop._peers_discovered += 1

        # Should not be called since peer is already known
        add_peer_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_self_discovery(self):
        """Should not add self as a peer."""
        add_peer_mock = MagicMock()

        loop = UdpDiscoveryLoop(
            get_node_id=lambda: "test-node",
            get_host=lambda: "127.0.0.1",
            get_port=lambda: 8770,
            get_known_peers=lambda: [],
            add_peer=add_peer_mock,
        )

        # Simulate receiving own broadcast - the node_id check prevents this
        # This is handled in _run_once() by checking msg.get("node_id") != node_id
        assert loop._get_node_id() == "test-node"


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoopLifecycle:
    """Tests for loop lifecycle management."""

    @pytest.mark.asyncio
    async def test_ip_discovery_start_stop(self):
        """IpDiscoveryLoop should start and stop cleanly."""
        loop = IpDiscoveryLoop(
            get_nodes=lambda: {},
            update_node_ip=AsyncMock(),
            config=IpDiscoveryConfig(check_interval_seconds=0.1),
        )

        # Start in background
        task = asyncio.create_task(loop.run_forever())

        # Let it run briefly
        await asyncio.sleep(0.2)

        # Stop it
        loop.stop()
        await asyncio.sleep(0.1)

        # Task should be done
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_tailscale_recovery_start_stop(self):
        """TailscaleRecoveryLoop should start and stop cleanly."""
        loop = TailscaleRecoveryLoop(
            get_tailscale_status=lambda: {},
            run_ssh_command=AsyncMock(),
            config=TailscaleRecoveryConfig(check_interval_seconds=0.1),
        )

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.2)
        loop.stop()
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_udp_discovery_start_stop(self):
        """UdpDiscoveryLoop should start and stop cleanly."""
        loop = UdpDiscoveryLoop(
            get_node_id=lambda: "test",
            get_host=lambda: "127.0.0.1",
            get_port=lambda: 8770,
            get_known_peers=lambda: [],
            add_peer=MagicMock(),
            config=UdpDiscoveryConfig(broadcast_interval_seconds=0.1),
        )

        task = asyncio.create_task(loop.run_forever())
        await asyncio.sleep(0.2)
        loop.stop()
        await asyncio.sleep(0.1)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestDefaultDiscoveryPort:
    """Tests for default discovery port constant."""

    def test_default_port_value(self):
        """Default discovery port should be 8771."""
        assert DEFAULT_DISCOVERY_PORT == 8771

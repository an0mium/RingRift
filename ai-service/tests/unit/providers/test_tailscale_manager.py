"""Tests for Tailscale manager."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import json

from app.providers.tailscale_manager import (
    TailscaleManager,
    TailscalePeer,
    TailscaleStatus,
)


class TestTailscalePeer:
    """Tests for TailscalePeer dataclass."""

    def test_create_peer(self):
        """Can create a peer."""
        peer = TailscalePeer(
            hostname="lambda-h100",
            tailscale_ip="100.1.2.3",
            online=True,
            os="linux",
            relay="direct",
        )
        assert peer.hostname == "lambda-h100"
        assert peer.online is True

    def test_peer_defaults(self):
        """Peer has sensible defaults."""
        peer = TailscalePeer(
            hostname="test",
            tailscale_ip="100.1.2.3",
        )
        assert peer.online is False
        assert peer.os is None
        assert peer.relay is None


class TestTailscaleStatus:
    """Tests for TailscaleStatus dataclass."""

    def test_create_status(self):
        """Can create status."""
        status = TailscaleStatus(
            self_hostname="local",
            self_ip="100.1.2.3",
            peers=[],
            online=True,
        )
        assert status.online is True
        assert len(status.peers) == 0

    def test_status_with_peers(self):
        """Status with peers."""
        peers = [
            TailscalePeer(hostname="node1", tailscale_ip="100.1.2.4", online=True),
            TailscalePeer(hostname="node2", tailscale_ip="100.1.2.5", online=False),
        ]
        status = TailscaleStatus(
            self_hostname="local",
            self_ip="100.1.2.3",
            peers=peers,
            online=True,
        )
        assert len(status.peers) == 2
        assert status.peers[0].online is True


class TestTailscaleManager:
    """Tests for TailscaleManager class."""

    def test_init(self):
        """Can initialize manager."""
        manager = TailscaleManager()
        assert manager is not None

    @pytest.mark.asyncio
    async def test_get_local_status_not_installed(self):
        """Returns None if tailscale not installed."""
        manager = TailscaleManager()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.side_effect = FileNotFoundError()
            status = await manager.get_local_status()
            assert status is None

    @pytest.mark.asyncio
    async def test_get_local_status_parses_json(self):
        """Parses tailscale status JSON."""
        manager = TailscaleManager()

        mock_output = json.dumps({
            "Self": {
                "HostName": "local-machine",
                "TailscaleIPs": ["100.1.2.3"],
            },
            "Peer": {
                "abc123": {
                    "HostName": "remote-node",
                    "TailscaleIPs": ["100.1.2.4"],
                    "Online": True,
                    "OS": "linux",
                    "Relay": "direct",
                },
            },
        })

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            status = await manager.get_local_status()

            assert status is not None
            assert status.self_hostname == "local-machine"
            assert status.self_ip == "100.1.2.3"
            assert len(status.peers) == 1
            assert status.peers[0].hostname == "remote-node"
            assert status.peers[0].online is True

    @pytest.mark.asyncio
    async def test_ping_peer(self):
        """Can ping a peer."""
        manager = TailscaleManager()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"pong from 100.1.2.4", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            result = await manager.ping_peer("100.1.2.4")
            assert result is True

    @pytest.mark.asyncio
    async def test_ping_peer_fails(self):
        """Ping returns False on failure."""
        manager = TailscaleManager()

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"timeout"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            result = await manager.ping_peer("100.1.2.4")
            assert result is False

    @pytest.mark.asyncio
    async def test_check_health(self):
        """Health check for tailscale peer uses ping."""
        manager = TailscaleManager()

        mock_peer = TailscalePeer(
            hostname="remote-node",
            tailscale_ip="100.1.2.4",
            online=True,
        )

        with patch.object(manager, "get_peer_status", new_callable=AsyncMock) as mock_status:
            with patch.object(manager, "ping_peer", new_callable=AsyncMock) as mock_ping:
                mock_status.return_value = mock_peer
                mock_ping.return_value = True

                result = await manager.check_health("100.1.2.4")
                assert result.healthy is True

    @pytest.mark.asyncio
    async def test_check_health_disconnected(self):
        """Health check fails when peer unreachable."""
        manager = TailscaleManager()

        mock_peer = TailscalePeer(
            hostname="remote-node",
            tailscale_ip="100.1.2.4",
            online=True,
        )

        with patch.object(manager, "get_peer_status", new_callable=AsyncMock) as mock_status:
            with patch.object(manager, "ping_peer", new_callable=AsyncMock) as mock_ping:
                mock_status.return_value = mock_peer
                mock_ping.return_value = False

                result = await manager.check_health("100.1.2.4")
                assert result.healthy is False


class TestTailscaleRemoteOperations:
    """Tests for remote Tailscale operations via SSH."""

    @pytest.mark.asyncio
    async def test_get_remote_status(self):
        """Can get status from remote host via SSH."""
        manager = TailscaleManager()

        mock_output = json.dumps({
            "Self": {"HostName": "remote", "TailscaleIPs": ["100.1.2.4"]},
            "Peer": {},
        })

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(mock_output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            status = await manager.get_remote_status("100.1.2.3")

            assert status is not None
            assert status.self_hostname == "remote"

    @pytest.mark.asyncio
    async def test_restart_remote_tailscale(self):
        """Can restart tailscale on remote host via SSH."""
        manager = TailscaleManager()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            result = await manager.restart_remote_tailscale("100.1.2.3")

            assert result is True

    @pytest.mark.asyncio
    async def test_force_reauth_remote(self):
        """Can force re-auth on remote host via SSH."""
        manager = TailscaleManager()

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock:
            mock.return_value = mock_proc
            result = await manager.force_reauth_remote("100.1.2.3", auth_key="tskey-auth-xxx")

            assert result is True

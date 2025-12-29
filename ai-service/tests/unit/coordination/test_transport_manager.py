"""Tests for TransportManager unified data transfer layer.

This module tests:
- Transport enum - all transport mechanism values
- TransportResult dataclass - creation, to_dict()
- TransportConfig dataclass - defaults, SSH settings, size thresholds
- TRANSPORT_CHAINS dict - all scenarios
- TransportManager class:
  - __init__ with default and custom config
  - select_transport_chain() for different file sizes
  - _is_circuit_open() / _record_transport_failure() / _record_transport_success()
  - transfer_file() with mocked subprocess
  - _execute_transfer() routing to specific transports
  - _transfer_rsync()
  - _transfer_scp()
  - _transfer_p2p()
  - _transfer_http()
  - _transfer_s3()
  - _transfer_base64()
  - _verify_checksum()
  - health_check()
  - get_stats()
  - get_transport_manager() singleton
  - reset_transport_manager()

December 2025 - RingRift AI Service
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.transport_manager import (
    TRANSPORT_CHAINS,
    Transport,
    TransportConfig,
    TransportManager,
    TransportResult,
    get_transport_manager,
    reset_transport_manager,
)


# =============================================================================
# Test Transport Enum
# =============================================================================


class TestTransportEnum:
    """Tests for Transport enum values."""

    def test_p2p_gossip_value(self):
        """Transport.P2P_GOSSIP should have correct value."""
        assert Transport.P2P_GOSSIP.value == "p2p"

    def test_http_fetch_value(self):
        """Transport.HTTP_FETCH should have correct value."""
        assert Transport.HTTP_FETCH.value == "http"

    def test_rsync_value(self):
        """Transport.RSYNC should have correct value."""
        assert Transport.RSYNC.value == "rsync"

    def test_s3_value(self):
        """Transport.S3 should have correct value."""
        assert Transport.S3.value == "s3"

    def test_scp_value(self):
        """Transport.SCP should have correct value."""
        assert Transport.SCP.value == "scp"

    def test_base64_ssh_value(self):
        """Transport.BASE64_SSH should have correct value."""
        assert Transport.BASE64_SSH.value == "base64"

    def test_all_transports_count(self):
        """Transport enum should have 6 members."""
        assert len(Transport) == 6

    def test_all_transports_iterable(self):
        """All Transport members should be iterable."""
        transports = list(Transport)
        assert Transport.P2P_GOSSIP in transports
        assert Transport.RSYNC in transports
        assert Transport.S3 in transports


# =============================================================================
# Test TransportResult Dataclass
# =============================================================================


class TestTransportResult:
    """Tests for TransportResult dataclass."""

    def test_success_result_creation(self):
        """TransportResult should be created for success."""
        result = TransportResult(
            success=True,
            transport_used=Transport.RSYNC,
            bytes_transferred=1024,
            duration_seconds=2.5,
            checksum="abc123",
        )
        assert result.success is True
        assert result.transport_used == Transport.RSYNC
        assert result.bytes_transferred == 1024
        assert result.duration_seconds == 2.5
        assert result.error is None
        assert result.checksum == "abc123"

    def test_failure_result_creation(self):
        """TransportResult should be created for failure."""
        result = TransportResult(
            success=False,
            transport_used=Transport.SCP,
            error="Connection refused",
        )
        assert result.success is False
        assert result.transport_used == Transport.SCP
        assert result.error == "Connection refused"
        assert result.bytes_transferred == 0

    def test_to_dict_all_fields(self):
        """to_dict should return all fields correctly."""
        result = TransportResult(
            success=True,
            transport_used=Transport.P2P_GOSSIP,
            bytes_transferred=2048,
            duration_seconds=1.5,
            retries=2,
            checksum="sha256hash",
        )
        d = result.to_dict()

        assert d["success"] is True
        assert d["transport"] == "p2p"
        assert d["bytes"] == 2048
        assert d["duration"] == 1.5
        assert d["retries"] == 2
        assert d["checksum"] == "sha256hash"
        assert d["error"] is None

    def test_to_dict_failure(self):
        """to_dict should include error on failure."""
        result = TransportResult(
            success=False,
            transport_used=Transport.HTTP_FETCH,
            error="Timeout exceeded",
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["transport"] == "http"
        assert d["error"] == "Timeout exceeded"

    def test_default_values(self):
        """TransportResult should have sensible defaults."""
        result = TransportResult(
            success=True,
            transport_used=Transport.RSYNC,
        )
        assert result.bytes_transferred == 0
        assert result.duration_seconds == 0.0
        assert result.retries == 0
        assert result.checksum == ""


# =============================================================================
# Test TransportConfig Dataclass
# =============================================================================


class TestTransportConfig:
    """Tests for TransportConfig dataclass."""

    def test_default_timeouts(self):
        """TransportConfig should have sensible default timeouts."""
        config = TransportConfig()
        assert config.connect_timeout == 10.0
        assert config.transfer_timeout == 600.0
        assert config.small_file_timeout == 60.0

    def test_default_size_thresholds(self):
        """TransportConfig should have correct size thresholds."""
        config = TransportConfig()
        assert config.large_file_threshold_bytes == 100 * 1024 * 1024  # 100MB
        assert config.small_file_threshold_bytes == 1 * 1024 * 1024  # 1MB

    def test_default_retry_settings(self):
        """TransportConfig should have correct retry settings."""
        config = TransportConfig()
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 5.0
        assert config.retry_backoff_multiplier == 2.0

    def test_default_bandwidth_limits(self):
        """TransportConfig should have correct bandwidth limits."""
        config = TransportConfig()
        assert config.default_bandwidth_limit == 0  # Unlimited
        assert config.rsync_bandwidth_limit == 50 * 1024 * 1024  # 50 MB/s

    def test_default_ssh_settings(self):
        """TransportConfig should have correct SSH settings."""
        config = TransportConfig()
        assert config.ssh_key_path == "~/.ssh/id_cluster"
        assert config.ssh_connect_timeout == 10
        assert len(config.ssh_options) > 0
        assert "-o" in config.ssh_options
        assert "StrictHostKeyChecking=no" in config.ssh_options

    def test_default_p2p_settings(self):
        """TransportConfig should have correct P2P settings."""
        config = TransportConfig()
        assert config.p2p_port == 8770
        assert config.http_data_port == 8780

    def test_default_s3_settings(self):
        """TransportConfig should have correct S3 settings."""
        config = TransportConfig()
        assert config.s3_bucket == "ringrift-models-20251214"

    def test_custom_config(self):
        """TransportConfig should accept custom values."""
        config = TransportConfig(
            connect_timeout=20.0,
            max_retries=5,
            p2p_port=9999,
        )
        assert config.connect_timeout == 20.0
        assert config.max_retries == 5
        assert config.p2p_port == 9999


# =============================================================================
# Test TRANSPORT_CHAINS Dict
# =============================================================================


class TestTransportChains:
    """Tests for TRANSPORT_CHAINS configuration."""

    def test_small_file_chain(self):
        """small_file chain should prioritize P2P and HTTP."""
        chain = TRANSPORT_CHAINS["small_file"]
        assert chain[0] == Transport.P2P_GOSSIP
        assert chain[1] == Transport.HTTP_FETCH
        assert Transport.BASE64_SSH in chain  # Fallback

    def test_large_file_chain(self):
        """large_file chain should prioritize rsync."""
        chain = TRANSPORT_CHAINS["large_file"]
        assert chain[0] == Transport.RSYNC
        assert Transport.SCP in chain
        assert Transport.BASE64_SSH in chain  # Fallback

    def test_s3_backup_chain(self):
        """s3_backup chain should use S3 first."""
        chain = TRANSPORT_CHAINS["s3_backup"]
        assert chain[0] == Transport.S3
        assert Transport.RSYNC in chain

    def test_ephemeral_urgent_chain(self):
        """ephemeral_urgent chain should skip P2P."""
        chain = TRANSPORT_CHAINS["ephemeral_urgent"]
        assert Transport.P2P_GOSSIP not in chain
        assert chain[0] == Transport.RSYNC

    def test_model_distribution_chain(self):
        """model_distribution chain should include P2P and rsync."""
        chain = TRANSPORT_CHAINS["model_distribution"]
        assert Transport.P2P_GOSSIP in chain
        assert Transport.RSYNC in chain

    def test_default_chain(self):
        """default chain should have comprehensive fallbacks."""
        chain = TRANSPORT_CHAINS["default"]
        assert chain[0] == Transport.RSYNC
        assert Transport.BASE64_SSH in chain  # Last resort

    def test_all_chains_exist(self):
        """All expected chains should be defined."""
        expected_chains = [
            "small_file",
            "large_file",
            "s3_backup",
            "ephemeral_urgent",
            "model_distribution",
            "default",
        ]
        for chain_name in expected_chains:
            assert chain_name in TRANSPORT_CHAINS


# =============================================================================
# Test TransportManager Initialization
# =============================================================================


class TestTransportManagerInit:
    """Tests for TransportManager initialization."""

    def test_default_initialization(self):
        """TransportManager should initialize with defaults."""
        manager = TransportManager()
        assert manager.config is not None
        assert isinstance(manager.config, TransportConfig)

    def test_custom_config(self):
        """TransportManager should accept custom config."""
        config = TransportConfig(max_retries=10)
        manager = TransportManager(config=config)
        assert manager.config.max_retries == 10

    def test_stats_initialized(self):
        """TransportManager should initialize stats tracking."""
        manager = TransportManager()
        stats = manager.get_stats()
        assert stats["total_transfers"] == 0
        assert stats["successful_transfers"] == 0
        assert stats["failed_transfers"] == 0
        assert stats["total_bytes"] == 0

    def test_by_transport_stats(self):
        """Stats should track per-transport counts."""
        manager = TransportManager()
        stats = manager.get_stats()
        assert "by_transport" in stats
        for transport in Transport:
            assert transport.value in stats["by_transport"]

    def test_circuit_breakers_initialized(self):
        """TransportManager should initialize circuit breakers."""
        manager = TransportManager()
        for transport in Transport:
            assert transport in manager._circuit_breakers
            assert manager._circuit_breakers[transport]["failures"] == 0


# =============================================================================
# Test Transport Chain Selection
# =============================================================================


class TestSelectTransportChain:
    """Tests for select_transport_chain method."""

    def test_explicit_scenario(self):
        """Explicit scenario should override size-based selection."""
        manager = TransportManager()
        chain = manager.select_transport_chain(
            size_bytes=1000,  # Small
            scenario="large_file",  # But explicit large_file scenario
        )
        assert chain == TRANSPORT_CHAINS["large_file"]

    def test_ephemeral_node(self):
        """Ephemeral nodes should use ephemeral_urgent chain."""
        manager = TransportManager()
        chain = manager.select_transport_chain(
            size_bytes=50_000_000,  # Medium size
            is_ephemeral=True,
        )
        assert chain == TRANSPORT_CHAINS["ephemeral_urgent"]

    def test_large_file_threshold(self):
        """Large files should use large_file chain."""
        manager = TransportManager()
        large_size = 200 * 1024 * 1024  # 200MB
        chain = manager.select_transport_chain(size_bytes=large_size)
        assert chain == TRANSPORT_CHAINS["large_file"]

    def test_small_file_threshold(self):
        """Small files should use small_file chain."""
        manager = TransportManager()
        small_size = 500 * 1024  # 500KB
        chain = manager.select_transport_chain(size_bytes=small_size)
        assert chain == TRANSPORT_CHAINS["small_file"]

    def test_medium_file_default(self):
        """Medium files should use default chain."""
        manager = TransportManager()
        medium_size = 50 * 1024 * 1024  # 50MB
        chain = manager.select_transport_chain(size_bytes=medium_size)
        assert chain == TRANSPORT_CHAINS["default"]

    def test_unknown_scenario_falls_back(self):
        """Unknown scenario should fall back to size-based selection."""
        manager = TransportManager()
        chain = manager.select_transport_chain(
            size_bytes=500 * 1024,  # Small
            scenario="nonexistent_scenario",
        )
        assert chain == TRANSPORT_CHAINS["small_file"]


# =============================================================================
# Test Circuit Breaker Logic
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker logic."""

    def test_circuit_initially_closed(self):
        """Circuit should be closed initially."""
        manager = TransportManager()
        assert manager._is_circuit_open(Transport.RSYNC) is False

    def test_record_failure_increments_count(self):
        """Recording failure should increment failure count."""
        manager = TransportManager()
        manager._record_transport_failure(Transport.SCP)
        assert manager._circuit_breakers[Transport.SCP]["failures"] == 1

    def test_circuit_opens_after_threshold(self):
        """Circuit should open after threshold failures."""
        manager = TransportManager()
        for _ in range(manager._circuit_breaker_threshold):
            manager._record_transport_failure(Transport.HTTP_FETCH)
        assert manager._is_circuit_open(Transport.HTTP_FETCH) is True

    def test_success_resets_failures(self):
        """Recording success should reset failure count."""
        manager = TransportManager()
        manager._record_transport_failure(Transport.P2P_GOSSIP)
        manager._record_transport_failure(Transport.P2P_GOSSIP)
        assert manager._circuit_breakers[Transport.P2P_GOSSIP]["failures"] == 2

        manager._record_transport_success(Transport.P2P_GOSSIP)
        assert manager._circuit_breakers[Transport.P2P_GOSSIP]["failures"] == 0

    def test_circuit_closes_after_timeout(self):
        """Circuit should close after reset time expires."""
        manager = TransportManager()
        manager._circuit_breaker_reset_time = 0.1  # 100ms for test

        # Open the circuit
        for _ in range(manager._circuit_breaker_threshold):
            manager._record_transport_failure(Transport.S3)
        assert manager._is_circuit_open(Transport.S3) is True

        # Wait for reset
        time.sleep(0.15)
        assert manager._is_circuit_open(Transport.S3) is False


# =============================================================================
# Test Transfer Methods - Rsync
# =============================================================================


class TestTransferRsync:
    """Tests for _transfer_rsync method."""

    @pytest.mark.asyncio
    async def test_rsync_success(self):
        """Rsync transfer should succeed with return code 0."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"sent 1,024 bytes", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_get_rsync_spec", side_effect=lambda n, p: f"{n}:{p}"):
                result = await manager._transfer_rsync(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is True
        assert result.transport_used == Transport.RSYNC
        assert result.bytes_transferred == 1024

    @pytest.mark.asyncio
    async def test_rsync_failure(self):
        """Rsync transfer should fail with non-zero return code."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Permission denied")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_get_rsync_spec", side_effect=lambda n, p: f"{n}:{p}"):
                result = await manager._transfer_rsync(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_rsync_timeout(self):
        """Rsync transfer should handle timeout."""
        manager = TransportManager()
        manager.config.transfer_timeout = 0.01  # Very short timeout

        async def slow_communicate():
            await asyncio.sleep(1)
            return (b"", b"")

        mock_process = AsyncMock()
        mock_process.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_get_rsync_spec", side_effect=lambda n, p: f"{n}:{p}"):
                result = await manager._transfer_rsync(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "timeout" in result.error.lower()


# =============================================================================
# Test Transfer Methods - SCP
# =============================================================================


class TestTransferScp:
    """Tests for _transfer_scp method."""

    @pytest.mark.asyncio
    async def test_scp_success(self):
        """SCP transfer should succeed with return code 0."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_get_scp_spec", side_effect=lambda n, p: f"{n}:{p}"):
                result = await manager._transfer_scp(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is True
        assert result.transport_used == Transport.SCP

    @pytest.mark.asyncio
    async def test_scp_failure(self):
        """SCP transfer should fail with non-zero return code."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Host key verification failed")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            with patch.object(manager, "_get_scp_spec", side_effect=lambda n, p: f"{n}:{p}"):
                result = await manager._transfer_scp(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "Host key" in result.error


# =============================================================================
# Test Transfer Methods - HTTP
# =============================================================================


class TestTransferHttp:
    """Tests for _transfer_http method."""

    @pytest.mark.asyncio
    async def test_http_success(self):
        """HTTP transfer should succeed when wget succeeds."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch.object(manager, "_get_node_ip", return_value="10.0.0.1"):
            with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.2"):
                with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                    result = await manager._transfer_http(
                        "node-1", "node-2", "/data/file.db", "/data/file.db"
                    )

        assert result.success is True
        assert result.transport_used == Transport.HTTP_FETCH

    @pytest.mark.asyncio
    async def test_http_unknown_source_node(self):
        """HTTP transfer should fail for unknown source node."""
        manager = TransportManager()

        with patch.object(manager, "_get_node_ip", return_value=None):
            result = await manager._transfer_http(
                "unknown-node", "node-2", "/data/file.db", "/data/file.db"
            )

        assert result.success is False
        assert "Unknown source node" in result.error

    @pytest.mark.asyncio
    async def test_http_unknown_target_node(self):
        """HTTP transfer should fail for unknown target node."""
        manager = TransportManager()

        with patch.object(manager, "_get_node_ip", return_value="10.0.0.1"):
            with patch.object(manager, "_get_ssh_target", return_value=None):
                result = await manager._transfer_http(
                    "node-1", "unknown-node", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "Unknown target node" in result.error


# =============================================================================
# Test Transfer Methods - P2P
# =============================================================================


class TestTransferP2p:
    """Tests for _transfer_p2p method."""

    @pytest.mark.asyncio
    async def test_p2p_success(self):
        """P2P transfer should succeed with successful response."""
        manager = TransportManager()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"success": True, "bytes": 512}
        )

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(manager, "_get_node_ip", return_value="10.0.0.1"):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await manager._transfer_p2p(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is True
        assert result.transport_used == Transport.P2P_GOSSIP
        assert result.bytes_transferred == 512

    @pytest.mark.asyncio
    async def test_p2p_unknown_node(self):
        """P2P transfer should fail for unknown node."""
        manager = TransportManager()

        with patch.object(manager, "_get_node_ip", return_value=None):
            result = await manager._transfer_p2p(
                "unknown-node", "node-2", "/data/file.db", "/data/file.db"
            )

        assert result.success is False
        assert "Unknown node IP" in result.error

    @pytest.mark.asyncio
    async def test_p2p_http_error(self):
        """P2P transfer should handle HTTP errors."""
        manager = TransportManager()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(manager, "_get_node_ip", return_value="10.0.0.1"):
            with patch("aiohttp.ClientSession", return_value=mock_session):
                result = await manager._transfer_p2p(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "500" in result.error


# =============================================================================
# Test Transfer Methods - S3
# =============================================================================


class TestTransferS3:
    """Tests for _transfer_s3 method."""

    @pytest.mark.asyncio
    async def test_s3_success(self):
        """S3 transfer should succeed with both upload and download."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await manager._transfer_s3(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is True
        assert result.transport_used == Transport.S3

    @pytest.mark.asyncio
    async def test_s3_upload_failure(self):
        """S3 transfer should fail on upload failure."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Upload failed: access denied")
        )

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await manager._transfer_s3(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "upload failed" in result.error.lower()


# =============================================================================
# Test Transfer Methods - Base64
# =============================================================================


class TestTransferBase64:
    """Tests for _transfer_base64 method."""

    @pytest.mark.asyncio
    async def test_base64_success(self):
        """Base64 transfer should succeed with shell command."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await manager._transfer_base64(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is True
        assert result.transport_used == Transport.BASE64_SSH

    @pytest.mark.asyncio
    async def test_base64_failure(self):
        """Base64 transfer should fail with non-zero return code."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"base64: invalid input")
        )

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_shell", return_value=mock_process):
                result = await manager._transfer_base64(
                    "node-1", "node-2", "/data/file.db", "/data/file.db"
                )

        assert result.success is False
        assert "invalid input" in result.error

    @pytest.mark.asyncio
    async def test_base64_unknown_node(self):
        """Base64 transfer should fail for unknown node."""
        manager = TransportManager()

        with patch.object(manager, "_get_ssh_target", return_value=None):
            result = await manager._transfer_base64(
                "unknown-node", "node-2", "/data/file.db", "/data/file.db"
            )

        assert result.success is False
        assert "Unknown node" in result.error


# =============================================================================
# Test transfer_file Integration
# =============================================================================


class TestTransferFile:
    """Tests for transfer_file method."""

    @pytest.mark.asyncio
    async def test_transfer_file_success(self):
        """transfer_file should succeed with first transport."""
        manager = TransportManager()

        success_result = TransportResult(
            success=True,
            transport_used=Transport.RSYNC,
            bytes_transferred=1024,
        )

        with patch.object(manager, "_execute_transfer", return_value=success_result):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                with patch.object(manager, "_verify_checksum", return_value=True):
                    result = await manager.transfer_file(
                        source_node="node-1",
                        target_node="node-2",
                        source_path="/data/file.db",
                        target_path="/data/file.db",
                        size_bytes=50_000_000,
                    )

        assert result.success is True
        assert manager._stats["successful_transfers"] == 1
        assert manager._stats["total_bytes"] == 1024

    @pytest.mark.asyncio
    async def test_transfer_file_fallback(self):
        """transfer_file should fallback to next transport on failure."""
        manager = TransportManager()
        manager.config.max_retries = 1  # Single retry per transport

        failure_result = TransportResult(
            success=False,
            transport_used=Transport.RSYNC,
            error="Connection refused",
        )
        success_result = TransportResult(
            success=True,
            transport_used=Transport.SCP,
            bytes_transferred=1024,
        )

        call_count = [0]

        async def mock_execute(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return failure_result
            return success_result

        with patch.object(manager, "_execute_transfer", side_effect=mock_execute):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                with patch.object(manager, "_verify_checksum", return_value=True):
                    result = await manager.transfer_file(
                        source_node="node-1",
                        target_node="node-2",
                        source_path="/data/file.db",
                        target_path="/data/file.db",
                        size_bytes=200_000_000,  # Large file
                    )

        assert result.success is True
        assert result.transport_used == Transport.SCP

    @pytest.mark.asyncio
    async def test_transfer_file_all_fail(self):
        """transfer_file should fail when all transports fail."""
        manager = TransportManager()
        manager.config.max_retries = 1

        failure_result = TransportResult(
            success=False,
            transport_used=Transport.RSYNC,
            error="Failed",
        )

        with patch.object(manager, "_execute_transfer", return_value=failure_result):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                result = await manager.transfer_file(
                    source_node="node-1",
                    target_node="node-2",
                    source_path="/data/file.db",
                    target_path="/data/file.db",
                    size_bytes=200_000_000,
                )

        assert result.success is False
        assert manager._stats["failed_transfers"] == 1

    @pytest.mark.asyncio
    async def test_transfer_file_updates_stats(self):
        """transfer_file should update transfer stats."""
        manager = TransportManager()

        success_result = TransportResult(
            success=True,
            transport_used=Transport.P2P_GOSSIP,
            bytes_transferred=2048,
        )

        with patch.object(manager, "_execute_transfer", return_value=success_result):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                with patch.object(manager, "_verify_checksum", return_value=True):
                    await manager.transfer_file(
                        source_node="node-1",
                        target_node="node-2",
                        source_path="/data/file.db",
                        target_path="/data/file.db",
                        size_bytes=500_000,  # Small file
                    )

        stats = manager.get_stats()
        assert stats["total_transfers"] == 1
        assert stats["by_transport"]["p2p"] == 1


# =============================================================================
# Test Verify Checksum
# =============================================================================


class TestVerifyChecksum:
    """Tests for _verify_checksum method."""

    @pytest.mark.asyncio
    async def test_verify_checksum_success(self):
        """_verify_checksum should return True for matching checksum."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"abc123def456  /data/file.db\n", b"")
        )

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await manager._verify_checksum(
                    "node-1", "/data/file.db", "abc123def456"
                )

        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checksum_mismatch(self):
        """_verify_checksum should return False for mismatched checksum."""
        manager = TransportManager()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(b"different_hash  /data/file.db\n", b"")
        )

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            with patch("asyncio.create_subprocess_exec", return_value=mock_process):
                result = await manager._verify_checksum(
                    "node-1", "/data/file.db", "expected_hash"
                )

        assert result is False

    @pytest.mark.asyncio
    async def test_verify_checksum_unknown_node(self):
        """_verify_checksum should return False for unknown node."""
        manager = TransportManager()

        with patch.object(manager, "_get_ssh_target", return_value=None):
            result = await manager._verify_checksum(
                "unknown-node", "/data/file.db", "hash"
            )

        assert result is False


# =============================================================================
# Test Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_healthy(self):
        """health_check should return healthy when no issues."""
        manager = TransportManager()
        result = manager.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_open_circuits(self):
        """health_check should report open circuit breakers."""
        manager = TransportManager()

        # Open multiple circuit breakers
        for transport in [Transport.RSYNC, Transport.SCP, Transport.HTTP_FETCH,
                         Transport.P2P_GOSSIP, Transport.S3]:
            for _ in range(manager._circuit_breaker_threshold):
                manager._record_transport_failure(transport)

        result = manager.health_check()
        assert result.healthy is False
        assert "circuit" in result.message.lower()

    def test_health_check_high_failure_rate(self):
        """health_check should report high failure rate."""
        manager = TransportManager()
        manager._stats["total_transfers"] = 100
        manager._stats["failed_transfers"] = 60

        result = manager.health_check()
        assert result.healthy is False
        assert "failure rate" in result.message.lower()

    def test_health_check_details(self):
        """health_check should include detailed stats."""
        manager = TransportManager()
        manager._stats["total_transfers"] = 10
        manager._stats["failed_transfers"] = 1

        result = manager.health_check()
        assert result.details is not None
        assert "total_transfers" in result.details
        assert "failure_rate" in result.details


# =============================================================================
# Test Get Stats
# =============================================================================


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats_initial(self):
        """get_stats should return initial zero values."""
        manager = TransportManager()
        stats = manager.get_stats()

        assert stats["total_transfers"] == 0
        assert stats["successful_transfers"] == 0
        assert stats["failed_transfers"] == 0
        assert stats["total_bytes"] == 0

    def test_get_stats_circuit_breakers(self):
        """get_stats should include circuit breaker status."""
        manager = TransportManager()

        # Record some failures
        manager._record_transport_failure(Transport.RSYNC)
        manager._record_transport_failure(Transport.RSYNC)

        stats = manager.get_stats()
        assert "circuit_breakers" in stats
        assert stats["circuit_breakers"]["rsync"]["failures"] == 2


# =============================================================================
# Test Singleton Pattern
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_transport_manager_returns_same_instance(self):
        """get_transport_manager should return same instance."""
        reset_transport_manager()

        manager1 = get_transport_manager()
        manager2 = get_transport_manager()

        assert manager1 is manager2

    def test_reset_transport_manager(self):
        """reset_transport_manager should clear the singleton."""
        reset_transport_manager()

        manager1 = get_transport_manager()
        reset_transport_manager()
        manager2 = get_transport_manager()

        assert manager1 is not manager2

    def test_config_only_used_on_first_call(self):
        """Config should only be used on first get_transport_manager call."""
        reset_transport_manager()

        config1 = TransportConfig(max_retries=10)
        manager1 = get_transport_manager(config=config1)
        assert manager1.config.max_retries == 10

        config2 = TransportConfig(max_retries=20)
        manager2 = get_transport_manager(config=config2)
        # Should still use config1
        assert manager2.config.max_retries == 10


# =============================================================================
# Test Helper Methods
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_is_ephemeral_node_vast(self):
        """_is_ephemeral_node should detect Vast.ai nodes."""
        manager = TransportManager()
        assert manager._is_ephemeral_node("vast-12345") is True

    def test_is_ephemeral_node_spot(self):
        """_is_ephemeral_node should detect spot instances."""
        manager = TransportManager()
        assert manager._is_ephemeral_node("spot-instance-1") is True

    def test_is_ephemeral_node_preempt(self):
        """_is_ephemeral_node should detect preemptible instances."""
        manager = TransportManager()
        assert manager._is_ephemeral_node("preempt-worker") is True

    def test_is_ephemeral_node_regular(self):
        """_is_ephemeral_node should not match regular nodes."""
        manager = TransportManager()
        assert manager._is_ephemeral_node("nebius-h100") is False
        assert manager._is_ephemeral_node("runpod-a100") is False

    def test_get_rsync_spec_local(self):
        """_get_rsync_spec should handle local paths."""
        manager = TransportManager()
        spec = manager._get_rsync_spec("local", "/data/file.db")
        assert spec == "/data/file.db"

    def test_get_rsync_spec_remote(self):
        """_get_rsync_spec should format remote paths."""
        manager = TransportManager()

        with patch.object(manager, "_get_ssh_target", return_value="ubuntu@10.0.0.1"):
            spec = manager._get_rsync_spec("remote-node", "/data/file.db")

        assert spec == "ubuntu@10.0.0.1:/data/file.db"

    def test_parse_rsync_bytes_sent(self):
        """_parse_rsync_bytes should parse 'sent X bytes' format."""
        manager = TransportManager()
        output = "sent 1,024 bytes  received 35 bytes"
        assert manager._parse_rsync_bytes(output) == 1024

    def test_parse_rsync_bytes_total_size(self):
        """_parse_rsync_bytes should parse 'total size is X' format."""
        manager = TransportManager()
        output = "total size is 2,048  speedup is 1.00"
        assert manager._parse_rsync_bytes(output) == 2048

    def test_parse_rsync_bytes_no_match(self):
        """_parse_rsync_bytes should return 0 for unmatched output."""
        manager = TransportManager()
        output = "some unrelated output"
        assert manager._parse_rsync_bytes(output) == 0


# =============================================================================
# Test Execute Transfer Routing
# =============================================================================


class TestExecuteTransfer:
    """Tests for _execute_transfer routing."""

    @pytest.mark.asyncio
    async def test_execute_transfer_rsync(self):
        """_execute_transfer should route to rsync."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_rsync", return_value=TransportResult(
            success=True, transport_used=Transport.RSYNC
        )) as mock_rsync:
            await manager._execute_transfer(
                Transport.RSYNC, "n1", "n2", "/p1", "/p2", 0
            )
            mock_rsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transfer_scp(self):
        """_execute_transfer should route to scp."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_scp", return_value=TransportResult(
            success=True, transport_used=Transport.SCP
        )) as mock_scp:
            await manager._execute_transfer(
                Transport.SCP, "n1", "n2", "/p1", "/p2", 0
            )
            mock_scp.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transfer_http(self):
        """_execute_transfer should route to http."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_http", return_value=TransportResult(
            success=True, transport_used=Transport.HTTP_FETCH
        )) as mock_http:
            await manager._execute_transfer(
                Transport.HTTP_FETCH, "n1", "n2", "/p1", "/p2", 0
            )
            mock_http.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transfer_p2p(self):
        """_execute_transfer should route to p2p."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_p2p", return_value=TransportResult(
            success=True, transport_used=Transport.P2P_GOSSIP
        )) as mock_p2p:
            await manager._execute_transfer(
                Transport.P2P_GOSSIP, "n1", "n2", "/p1", "/p2", 0
            )
            mock_p2p.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transfer_s3(self):
        """_execute_transfer should route to s3."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_s3", return_value=TransportResult(
            success=True, transport_used=Transport.S3
        )) as mock_s3:
            await manager._execute_transfer(
                Transport.S3, "n1", "n2", "/p1", "/p2", 0
            )
            mock_s3.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_transfer_base64(self):
        """_execute_transfer should route to base64."""
        manager = TransportManager()

        with patch.object(manager, "_transfer_base64", return_value=TransportResult(
            success=True, transport_used=Transport.BASE64_SSH
        )) as mock_base64:
            await manager._execute_transfer(
                Transport.BASE64_SSH, "n1", "n2", "/p1", "/p2", 0
            )
            mock_base64.assert_called_once()


# =============================================================================
# Test Timeout Handling
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling in transfers."""

    @pytest.mark.asyncio
    async def test_transfer_file_timeout_exception(self):
        """transfer_file should handle TimeoutError."""
        manager = TransportManager()
        manager.config.max_retries = 1

        async def raise_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        with patch.object(manager, "_execute_transfer", side_effect=raise_timeout):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                result = await manager.transfer_file(
                    source_node="node-1",
                    target_node="node-2",
                    source_path="/data/file.db",
                    target_path="/data/file.db",
                )

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_transfer_file_generic_exception(self):
        """transfer_file should handle generic exceptions."""
        manager = TransportManager()
        manager.config.max_retries = 1

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Unexpected error")

        with patch.object(manager, "_execute_transfer", side_effect=raise_error):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                result = await manager.transfer_file(
                    source_node="node-1",
                    target_node="node-2",
                    source_path="/data/file.db",
                    target_path="/data/file.db",
                )

        assert result.success is False
        assert "Unexpected error" in result.error


# =============================================================================
# Test Checksum Verification
# =============================================================================


class TestChecksumVerification:
    """Tests for checksum verification during transfer."""

    @pytest.mark.asyncio
    async def test_transfer_file_checksum_mismatch(self):
        """transfer_file should fail on checksum mismatch."""
        manager = TransportManager()
        manager.config.max_retries = 1

        success_result = TransportResult(
            success=True,
            transport_used=Transport.RSYNC,
            checksum="expected_hash",
        )

        with patch.object(manager, "_execute_transfer", return_value=success_result):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                with patch.object(manager, "_verify_checksum", return_value=False):
                    result = await manager.transfer_file(
                        source_node="node-1",
                        target_node="node-2",
                        source_path="/data/file.db",
                        target_path="/data/file.db",
                        verify_checksum=True,
                    )

        # Should fail due to checksum mismatch
        assert result.success is False
        assert "Checksum mismatch" in result.error

    @pytest.mark.asyncio
    async def test_transfer_file_skip_checksum(self):
        """transfer_file should skip checksum when verify_checksum=False."""
        manager = TransportManager()

        success_result = TransportResult(
            success=True,
            transport_used=Transport.RSYNC,
            checksum="some_hash",
        )

        with patch.object(manager, "_execute_transfer", return_value=success_result):
            with patch.object(manager, "_is_ephemeral_node", return_value=False):
                with patch.object(manager, "_verify_checksum") as mock_verify:
                    result = await manager.transfer_file(
                        source_node="node-1",
                        target_node="node-2",
                        source_path="/data/file.db",
                        target_path="/data/file.db",
                        verify_checksum=False,
                    )

        assert result.success is True
        mock_verify.assert_not_called()

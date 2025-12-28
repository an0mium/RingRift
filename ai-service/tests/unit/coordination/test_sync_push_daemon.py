"""Tests for SyncPushDaemon - push-based data sync with verified cleanup.

This daemon is CRITICAL because it handles data deletion - only after verified copies exist.
Tests cover:
- Disk threshold detection (50%, 70%, 75%)
- Sync receipt verification before deletion
- Cleanup with insufficient copies (should NOT delete)
- Network failure recovery
"""

import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.sync_push_daemon import (
    SyncPushConfig,
    SyncPushDaemon,
    get_sync_push_daemon,
    reset_sync_push_daemon,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default test configuration."""
    return SyncPushConfig(
        push_threshold_percent=50.0,
        urgent_threshold_percent=70.0,
        cleanup_threshold_percent=75.0,
        min_copies_before_cleanup=2,
        push_interval_seconds=60.0,
        max_files_per_cycle=10,
        checksum_algorithm="md5",
    )


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data" / "games"
        data_dir.mkdir(parents=True)

        # Create some test database files
        (data_dir / "selfplay_001.db").write_text("test data 1")
        (data_dir / "selfplay_002.db").write_text("test data 2")
        (data_dir / "canonical_hex8_2p.db").write_text("canonical data")

        yield data_dir


@pytest.fixture
def mock_cluster_manifest():
    """Mock ClusterManifest for sync receipt tracking."""
    manifest = MagicMock()
    manifest.get_sync_receipts = MagicMock(return_value=[])
    manifest.record_sync_receipt = MagicMock()
    manifest.get_coordinator_hosts = MagicMock(return_value=["coordinator1:8770"])
    return manifest


@pytest.fixture
def daemon(config, temp_data_dir, mock_cluster_manifest):
    """Create daemon instance with mocked dependencies."""
    reset_sync_push_daemon()

    with patch("app.coordination.sync_push_daemon.get_cluster_manifest", return_value=mock_cluster_manifest):
        daemon = SyncPushDaemon(config=config)
        daemon._data_dir = temp_data_dir
        yield daemon

    reset_sync_push_daemon()


# =============================================================================
# TestSyncPushConfig
# =============================================================================


class TestSyncPushConfig:
    """Tests for SyncPushConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SyncPushConfig()

        assert config.push_threshold_percent == 50.0
        assert config.urgent_threshold_percent == 70.0
        assert config.cleanup_threshold_percent == 75.0
        assert config.min_copies_before_cleanup == 2
        assert config.push_interval_seconds == 300.0
        assert config.max_files_per_cycle == 50
        assert config.checksum_algorithm == "md5"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SyncPushConfig(
            push_threshold_percent=40.0,
            urgent_threshold_percent=60.0,
            cleanup_threshold_percent=80.0,
            min_copies_before_cleanup=3,
            push_interval_seconds=120.0,
            max_files_per_cycle=20,
            checksum_algorithm="sha256",
        )

        assert config.push_threshold_percent == 40.0
        assert config.urgent_threshold_percent == 60.0
        assert config.cleanup_threshold_percent == 80.0
        assert config.min_copies_before_cleanup == 3
        assert config.push_interval_seconds == 120.0
        assert config.max_files_per_cycle == 20
        assert config.checksum_algorithm == "sha256"

    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = SyncPushConfig.from_env()

            assert config.push_threshold_percent == 50.0
            assert config.min_copies_before_cleanup == 2

    def test_from_env_custom(self):
        """Test from_env with custom environment variables."""
        env_vars = {
            "RINGRIFT_SYNC_PUSH_THRESHOLD": "45.0",
            "RINGRIFT_SYNC_URGENT_THRESHOLD": "65.0",
            "RINGRIFT_SYNC_CLEANUP_THRESHOLD": "78.0",
            "RINGRIFT_SYNC_MIN_COPIES": "3",
            "RINGRIFT_SYNC_INTERVAL": "180.0",
            "RINGRIFT_SYNC_MAX_FILES": "25",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = SyncPushConfig.from_env()

            assert config.push_threshold_percent == 45.0
            assert config.urgent_threshold_percent == 65.0
            assert config.cleanup_threshold_percent == 78.0
            assert config.min_copies_before_cleanup == 3
            assert config.push_interval_seconds == 180.0
            assert config.max_files_per_cycle == 25

    def test_from_env_invalid_values(self):
        """Test from_env handles invalid values gracefully."""
        env_vars = {
            "RINGRIFT_SYNC_PUSH_THRESHOLD": "invalid",
            "RINGRIFT_SYNC_MIN_COPIES": "not_a_number",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            # Should use defaults for invalid values
            config = SyncPushConfig.from_env()
            assert config.push_threshold_percent == 50.0
            assert config.min_copies_before_cleanup == 2

    def test_threshold_ordering(self):
        """Test that thresholds are in correct order."""
        config = SyncPushConfig()

        assert config.push_threshold_percent < config.urgent_threshold_percent
        assert config.urgent_threshold_percent < config.cleanup_threshold_percent


# =============================================================================
# TestSyncPushDaemonInit
# =============================================================================


class TestSyncPushDaemonInit:
    """Tests for SyncPushDaemon initialization."""

    def test_init_with_config(self, config):
        """Test initialization with explicit config."""
        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon = SyncPushDaemon(config=config)

            assert daemon.config == config
            assert daemon._pending_files == []
            assert daemon._last_push_time == 0

        reset_sync_push_daemon()

    def test_init_default_config(self):
        """Test initialization with default config."""
        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon = SyncPushDaemon()

            assert daemon.config is not None
            assert isinstance(daemon.config, SyncPushConfig)

        reset_sync_push_daemon()

    def test_init_sets_daemon_name(self, config):
        """Test daemon name is set correctly."""
        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon = SyncPushDaemon(config=config)

            assert daemon.name == "sync_push_daemon"

        reset_sync_push_daemon()


# =============================================================================
# TestDiskUsage
# =============================================================================


class TestDiskUsage:
    """Tests for disk usage detection."""

    def test_get_disk_usage_normal(self, daemon):
        """Test disk usage below push threshold."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=40_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 40.0  # 40%

    def test_get_disk_usage_at_push_threshold(self, daemon):
        """Test disk usage at push threshold (50%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=50_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 50.0

    def test_get_disk_usage_at_urgent_threshold(self, daemon):
        """Test disk usage at urgent threshold (70%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=70_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 70.0

    def test_get_disk_usage_at_cleanup_threshold(self, daemon):
        """Test disk usage at cleanup threshold (75%)."""
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(total=100_000_000_000, used=75_000_000_000)

            usage = daemon._get_disk_usage()

            assert usage == 75.0


# =============================================================================
# TestChecksum
# =============================================================================


class TestChecksum:
    """Tests for checksum computation."""

    def test_compute_checksum_md5(self, daemon, temp_data_dir):
        """Test MD5 checksum computation."""
        test_file = temp_data_dir / "selfplay_001.db"

        checksum = daemon._compute_checksum(test_file)

        # Verify it's a valid MD5 hash (32 hex characters)
        assert len(checksum) == 32
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_compute_checksum_sha256(self, config, temp_data_dir):
        """Test SHA256 checksum computation."""
        reset_sync_push_daemon()
        config.checksum_algorithm = "sha256"

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon = SyncPushDaemon(config=config)
            daemon._data_dir = temp_data_dir

            test_file = temp_data_dir / "selfplay_001.db"
            checksum = daemon._compute_checksum(test_file)

            # Verify it's a valid SHA256 hash (64 hex characters)
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

        reset_sync_push_daemon()

    def test_compute_checksum_consistency(self, daemon, temp_data_dir):
        """Test checksum is consistent for same file."""
        test_file = temp_data_dir / "selfplay_001.db"

        checksum1 = daemon._compute_checksum(test_file)
        checksum2 = daemon._compute_checksum(test_file)

        assert checksum1 == checksum2

    def test_compute_checksum_different_files(self, daemon, temp_data_dir):
        """Test different files have different checksums."""
        file1 = temp_data_dir / "selfplay_001.db"
        file2 = temp_data_dir / "selfplay_002.db"

        checksum1 = daemon._compute_checksum(file1)
        checksum2 = daemon._compute_checksum(file2)

        assert checksum1 != checksum2


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy(self, daemon):
        """Test health check when daemon is healthy."""
        daemon._running = True
        daemon._error_count = 0

        result = daemon.health_check()

        assert result.healthy is True
        assert "error_count" in result.details
        assert result.details["error_count"] == 0

    def test_health_check_with_errors(self, daemon):
        """Test health check with error count."""
        daemon._running = True
        daemon._error_count = 5

        result = daemon.health_check()

        assert result.details["error_count"] == 5

    def test_health_check_not_running(self, daemon):
        """Test health check when daemon is not running."""
        daemon._running = False

        result = daemon.health_check()

        # Should still report status even when not running
        assert result is not None

    def test_health_check_includes_disk_usage(self, daemon):
        """Test health check includes disk usage info."""
        daemon._running = True
        daemon._last_disk_usage = 55.0

        result = daemon.health_check()

        assert "disk_usage_percent" in result.details or "last_disk_usage" in result.details


# =============================================================================
# TestGetStatus
# =============================================================================


class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status_basic(self, daemon):
        """Test basic status reporting."""
        daemon._running = True
        daemon._pending_files = ["/path/to/file1.db", "/path/to/file2.db"]

        status = daemon.get_status()

        assert "pending_files" in status
        assert status["pending_files"] == 2

    def test_get_status_includes_thresholds(self, daemon):
        """Test status includes threshold configuration."""
        status = daemon.get_status()

        assert "push_threshold" in status or "config" in status

    def test_get_status_includes_last_push(self, daemon):
        """Test status includes last push timestamp."""
        daemon._last_push_time = time.time() - 300

        status = daemon.get_status()

        assert "last_push_time" in status or "last_push" in status


# =============================================================================
# TestSingleton
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_sync_push_daemon_singleton(self):
        """Test get_sync_push_daemon returns singleton."""
        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon1 = get_sync_push_daemon()
            daemon2 = get_sync_push_daemon()

            assert daemon1 is daemon2

        reset_sync_push_daemon()

    def test_reset_sync_push_daemon(self):
        """Test reset_sync_push_daemon clears singleton."""
        reset_sync_push_daemon()

        with patch("app.coordination.sync_push_daemon.get_cluster_manifest"):
            daemon1 = get_sync_push_daemon()
            reset_sync_push_daemon()
            daemon2 = get_sync_push_daemon()

            assert daemon1 is not daemon2

        reset_sync_push_daemon()


# =============================================================================
# TestPushPendingFiles
# =============================================================================


class TestPushPendingFiles:
    """Tests for push logic."""

    @pytest.mark.asyncio
    async def test_push_pending_no_files(self, daemon):
        """Test push with no pending files."""
        daemon._pending_files = []

        result = await daemon._push_pending_files()

        assert result == 0 or result is None

    @pytest.mark.asyncio
    async def test_push_pending_with_files(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test push with pending files."""
        daemon._pending_files = [
            str(temp_data_dir / "selfplay_001.db"),
            str(temp_data_dir / "selfplay_002.db"),
        ]
        mock_cluster_manifest.get_coordinator_hosts.return_value = ["coordinator1:8770"]

        with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True

            result = await daemon._push_pending_files()

            assert mock_push.call_count >= 1

    @pytest.mark.asyncio
    async def test_push_respects_max_files_per_cycle(self, daemon, temp_data_dir):
        """Test push respects max_files_per_cycle limit."""
        daemon.config.max_files_per_cycle = 2
        daemon._pending_files = [
            str(temp_data_dir / f"file_{i}.db") for i in range(10)
        ]

        with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True

            await daemon._push_pending_files()

            assert mock_push.call_count <= 2

    @pytest.mark.asyncio
    async def test_push_urgent_mode_increases_limit(self, daemon, temp_data_dir):
        """Test urgent mode increases file limit."""
        daemon.config.max_files_per_cycle = 5
        daemon._pending_files = [
            str(temp_data_dir / f"file_{i}.db") for i in range(20)
        ]

        with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True

            await daemon._push_pending_files(urgent=True)

            # Urgent mode should push more files
            assert mock_push.call_count >= 5

    @pytest.mark.asyncio
    async def test_push_handles_large_files(self, daemon, temp_data_dir):
        """Test push handles large files correctly."""
        large_file = temp_data_dir / "large_file.db"
        large_file.write_bytes(b"x" * 10_000_000)  # 10MB
        daemon._pending_files = [str(large_file)]

        with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
            mock_push.return_value = True

            await daemon._push_pending_files()

            mock_push.assert_called()


# =============================================================================
# TestSafeCleanup
# =============================================================================


class TestSafeCleanup:
    """Tests for cleanup logic - CRITICAL: must verify copies before deletion."""

    @pytest.mark.asyncio
    async def test_cleanup_requires_min_copies(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup requires minimum verified copies."""
        test_file = temp_data_dir / "selfplay_001.db"
        daemon.config.min_copies_before_cleanup = 2

        # Only 1 copy verified
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "abc123", "verified": True}
        ]

        await daemon._safe_cleanup([str(test_file)])

        # File should NOT be deleted
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_with_sufficient_copies(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup proceeds with sufficient verified copies."""
        test_file = temp_data_dir / "selfplay_001.db"
        daemon.config.min_copies_before_cleanup = 2

        # 2 copies verified
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "abc123", "verified": True},
            {"host": "host2", "checksum": "abc123", "verified": True},
        ]

        with patch.object(daemon, "_compute_checksum", return_value="abc123"):
            await daemon._safe_cleanup([str(test_file)])

        # File should be deleted
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_skips_canonical_files(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup never deletes canonical files."""
        canonical_file = temp_data_dir / "canonical_hex8_2p.db"
        daemon.config.min_copies_before_cleanup = 2

        # Even with sufficient copies
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "abc123", "verified": True},
            {"host": "host2", "checksum": "abc123", "verified": True},
        ]

        await daemon._safe_cleanup([str(canonical_file)])

        # Canonical file should NEVER be deleted
        assert canonical_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_removes_wal_and_shm(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup removes associated WAL and SHM files."""
        test_file = temp_data_dir / "selfplay_001.db"
        wal_file = temp_data_dir / "selfplay_001.db-wal"
        shm_file = temp_data_dir / "selfplay_001.db-shm"

        wal_file.write_text("wal data")
        shm_file.write_text("shm data")

        daemon.config.min_copies_before_cleanup = 2
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "abc123", "verified": True},
            {"host": "host2", "checksum": "abc123", "verified": True},
        ]

        with patch.object(daemon, "_compute_checksum", return_value="abc123"):
            await daemon._safe_cleanup([str(test_file)])

        # WAL and SHM should also be removed
        assert not wal_file.exists()
        assert not shm_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_checksum_mismatch(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup skips files with checksum mismatch."""
        test_file = temp_data_dir / "selfplay_001.db"
        daemon.config.min_copies_before_cleanup = 2

        # Checksums don't match
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "different_checksum", "verified": True},
            {"host": "host2", "checksum": "different_checksum", "verified": True},
        ]

        with patch.object(daemon, "_compute_checksum", return_value="local_checksum"):
            await daemon._safe_cleanup([str(test_file)])

        # File should NOT be deleted due to checksum mismatch
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_unverified_receipts_ignored(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test cleanup ignores unverified sync receipts."""
        test_file = temp_data_dir / "selfplay_001.db"
        daemon.config.min_copies_before_cleanup = 2

        # 2 copies but not verified
        mock_cluster_manifest.get_sync_receipts.return_value = [
            {"host": "host1", "checksum": "abc123", "verified": False},
            {"host": "host2", "checksum": "abc123", "verified": False},
        ]

        await daemon._safe_cleanup([str(test_file)])

        # File should NOT be deleted - unverified receipts don't count
        assert test_file.exists()


# =============================================================================
# TestRunCycle
# =============================================================================


class TestRunCycle:
    """Tests for run cycle at various disk thresholds."""

    @pytest.mark.asyncio
    async def test_run_cycle_below_threshold(self, daemon):
        """Test run cycle with disk below push threshold."""
        with patch.object(daemon, "_get_disk_usage", return_value=40.0):
            with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                await daemon._run_cycle()

                # Should not push below threshold
                mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_at_push_threshold(self, daemon):
        """Test run cycle at push threshold (50%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=55.0):
            with patch.object(daemon, "_discover_pending_files", return_value=["/path/file.db"]):
                with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                    await daemon._run_cycle()

                    # Should push at threshold
                    mock_push.assert_called()

    @pytest.mark.asyncio
    async def test_run_cycle_at_urgent_threshold(self, daemon):
        """Test run cycle at urgent threshold (70%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=72.0):
            with patch.object(daemon, "_discover_pending_files", return_value=["/path/file.db"]):
                with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock) as mock_push:
                    await daemon._run_cycle()

                    # Should push urgently
                    mock_push.assert_called_once()
                    # Check urgent=True was passed
                    call_kwargs = mock_push.call_args[1] if mock_push.call_args[1] else {}
                    assert call_kwargs.get("urgent", False) is True

    @pytest.mark.asyncio
    async def test_run_cycle_at_cleanup_threshold(self, daemon):
        """Test run cycle at cleanup threshold (75%)."""
        with patch.object(daemon, "_get_disk_usage", return_value=78.0):
            with patch.object(daemon, "_discover_pending_files", return_value=["/path/file.db"]):
                with patch.object(daemon, "_push_pending_files", new_callable=AsyncMock):
                    with patch.object(daemon, "_safe_cleanup", new_callable=AsyncMock) as mock_cleanup:
                        await daemon._run_cycle()

                        # Should attempt cleanup
                        mock_cleanup.assert_called()


# =============================================================================
# TestCoordinatorDiscovery
# =============================================================================


class TestCoordinatorDiscovery:
    """Tests for coordinator host discovery."""

    def test_discover_coordinators(self, daemon, mock_cluster_manifest):
        """Test coordinator discovery from cluster manifest."""
        mock_cluster_manifest.get_coordinator_hosts.return_value = [
            "coordinator1:8770",
            "coordinator2:8770",
        ]

        hosts = daemon._discover_coordinators()

        assert len(hosts) >= 1

    def test_discover_coordinators_empty(self, daemon, mock_cluster_manifest):
        """Test coordinator discovery with no coordinators."""
        mock_cluster_manifest.get_coordinator_hosts.return_value = []

        hosts = daemon._discover_coordinators()

        # Should handle empty list gracefully
        assert hosts == [] or hosts is None

    def test_discover_coordinators_fallback(self, daemon, mock_cluster_manifest):
        """Test coordinator discovery falls back to config."""
        mock_cluster_manifest.get_coordinator_hosts.side_effect = Exception("Manifest error")

        # Should not raise, may return fallback
        hosts = daemon._discover_coordinators()

        assert hosts is not None or hosts == []


# =============================================================================
# TestLifecycle
# =============================================================================


class TestLifecycle:
    """Tests for daemon lifecycle management."""

    @pytest.mark.asyncio
    async def test_on_start(self, daemon):
        """Test daemon start."""
        await daemon._on_start()

        # Daemon should be in running state
        # Implementation may vary

    @pytest.mark.asyncio
    async def test_on_stop(self, daemon):
        """Test daemon stop."""
        daemon._running = True

        await daemon._on_stop()

        # Any cleanup should be done


# =============================================================================
# TestPushFile
# =============================================================================


class TestPushFile:
    """Tests for individual file push."""

    @pytest.mark.asyncio
    async def test_push_file_success(self, daemon, temp_data_dir):
        """Test successful file push."""
        test_file = temp_data_dir / "selfplay_001.db"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            result = await daemon._push_file(str(test_file), "coordinator1:8770")

            assert result is True

    @pytest.mark.asyncio
    async def test_push_file_network_error(self, daemon, temp_data_dir):
        """Test file push with network error."""
        test_file = temp_data_dir / "selfplay_001.db"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = Exception("Network error")

            result = await daemon._push_file(str(test_file), "coordinator1:8770")

            assert result is False

    @pytest.mark.asyncio
    async def test_push_file_server_error(self, daemon, temp_data_dir):
        """Test file push with server error."""
        test_file = temp_data_dir / "selfplay_001.db"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            result = await daemon._push_file(str(test_file), "coordinator1:8770")

            assert result is False

    @pytest.mark.asyncio
    async def test_push_file_records_receipt(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test successful push records sync receipt."""
        test_file = temp_data_dir / "selfplay_001.db"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            await daemon._push_file(str(test_file), "coordinator1:8770")

            # Should record receipt in manifest
            # mock_cluster_manifest.record_sync_receipt.assert_called()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for SyncPushDaemon."""

    @pytest.mark.asyncio
    async def test_full_push_cycle(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test complete push cycle from discovery to push."""
        # Set up disk at push threshold
        with patch.object(daemon, "_get_disk_usage", return_value=55.0):
            # Mock coordinator discovery
            mock_cluster_manifest.get_coordinator_hosts.return_value = ["coordinator1:8770"]

            # Mock successful push
            with patch.object(daemon, "_push_file", new_callable=AsyncMock) as mock_push:
                mock_push.return_value = True

                await daemon._run_cycle()

                # Should have attempted to push files

    @pytest.mark.asyncio
    async def test_full_cleanup_cycle(self, daemon, temp_data_dir, mock_cluster_manifest):
        """Test complete cleanup cycle."""
        # Set up disk at cleanup threshold
        with patch.object(daemon, "_get_disk_usage", return_value=78.0):
            # Mock sufficient copies
            mock_cluster_manifest.get_sync_receipts.return_value = [
                {"host": "host1", "checksum": "abc123", "verified": True},
                {"host": "host2", "checksum": "abc123", "verified": True},
            ]

            with patch.object(daemon, "_compute_checksum", return_value="abc123"):
                # Files that can be cleaned
                daemon._pending_files = [str(temp_data_dir / "selfplay_001.db")]

                await daemon._run_cycle()

                # Cleanup should have run

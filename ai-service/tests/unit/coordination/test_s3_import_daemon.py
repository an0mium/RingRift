"""Tests for S3ImportDaemon.

Tests the S3 import daemon for cluster recovery and bootstrap operations.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from app.coordination.s3_import_daemon import (
    S3ImportConfig,
    S3ImportStats,
    S3FileInfo,
    S3ImportDaemon,
    get_s3_import_daemon,
    reset_s3_import_daemon,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    S3ImportDaemon._instance = None
    yield
    S3ImportDaemon._instance = None


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials via environment variables."""
    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test-key-id",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        },
    ):
        yield


@pytest.fixture
def mock_safe_emit():
    """Mock safe_emit_event for event emission tests."""
    with patch(
        "app.coordination.s3_import_daemon.safe_emit_event"
    ) as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def tmp_base_path(tmp_path):
    """Create temporary base path with required directories."""
    games_dir = tmp_path / "data" / "games"
    training_dir = tmp_path / "data" / "training"
    models_dir = tmp_path / "models"

    games_dir.mkdir(parents=True)
    training_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    return tmp_path


# ============================================================================
# S3ImportConfig Tests
# ============================================================================


class TestS3ImportConfig:
    """Tests for S3ImportConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = S3ImportConfig()

        assert config.bucket == "ringrift-models-20251214"
        assert config.region == "us-east-1"
        assert config.enabled is True
        assert config.import_on_startup is False
        assert config.check_interval == 3600
        assert config.download_timeout == 600

    def test_custom_values(self):
        """Test custom configuration values."""
        config = S3ImportConfig(
            bucket="custom-bucket",
            region="eu-west-1",
            enabled=False,
            import_on_startup=True,
            check_interval=1800,
        )

        assert config.bucket == "custom-bucket"
        assert config.region == "eu-west-1"
        assert config.enabled is False
        assert config.import_on_startup is True
        assert config.check_interval == 1800

    def test_env_var_override(self):
        """Test environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "RINGRIFT_S3_BUCKET": "env-bucket",
                "RINGRIFT_S3_REGION": "ap-northeast-1",
                "RINGRIFT_S3_IMPORT_ENABLED": "false",
                "RINGRIFT_S3_IMPORT_ON_STARTUP": "true",
            },
        ):
            config = S3ImportConfig()

            assert config.bucket == "env-bucket"
            assert config.region == "ap-northeast-1"
            assert config.enabled is False
            assert config.import_on_startup is True

    def test_local_directory_defaults(self):
        """Test local directory path defaults."""
        config = S3ImportConfig()

        assert config.local_games_dir == "data/games"
        assert config.local_training_dir == "data/training"
        assert config.local_models_dir == "models"

    def test_s3_prefix_defaults(self):
        """Test S3 prefix defaults."""
        config = S3ImportConfig()

        assert config.s3_games_prefix == "consolidated/games"
        assert config.s3_training_prefix == "consolidated/training"
        assert config.s3_models_prefix == "models"


# ============================================================================
# S3ImportStats Tests
# ============================================================================


class TestS3ImportStats:
    """Tests for S3ImportStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = S3ImportStats()

        assert stats.total_files_imported == 0
        assert stats.total_bytes_imported == 0
        assert stats.last_import_time == 0.0
        assert stats.import_errors == 0
        assert stats.last_error is None
        assert stats.configs_imported == []

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = S3ImportStats(
            total_files_imported=10,
            total_bytes_imported=1024 * 1024 * 100,
            import_errors=2,
            last_error="Test error",
            configs_imported=["hex8_2p", "square8_4p"],
        )

        assert stats.total_files_imported == 10
        assert stats.total_bytes_imported == 104857600
        assert stats.import_errors == 2
        assert stats.last_error == "Test error"
        assert stats.configs_imported == ["hex8_2p", "square8_4p"]


# ============================================================================
# S3FileInfo Tests
# ============================================================================


class TestS3FileInfo:
    """Tests for S3FileInfo dataclass."""

    def test_basic_creation(self):
        """Test basic file info creation."""
        info = S3FileInfo(
            key="consolidated/games/canonical_hex8_2p.db",
            size=1048576,
            last_modified="2025-01-01T12:00:00Z",
            etag="abc123def456",
        )

        assert info.key == "consolidated/games/canonical_hex8_2p.db"
        assert info.size == 1048576
        assert info.last_modified == "2025-01-01T12:00:00Z"
        assert info.etag == "abc123def456"


# ============================================================================
# S3ImportDaemon Basic Tests
# ============================================================================


class TestS3ImportDaemonBasic:
    """Basic tests for S3ImportDaemon."""

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        daemon1 = S3ImportDaemon.get_instance()
        daemon2 = S3ImportDaemon.get_instance()

        assert daemon1 is daemon2

    def test_singleton_reset(self):
        """Test singleton reset."""
        daemon1 = S3ImportDaemon.get_instance()
        S3ImportDaemon.reset_instance()
        daemon2 = S3ImportDaemon.get_instance()

        assert daemon1 is not daemon2

    def test_initialization_default(self):
        """Test initialization with default config."""
        daemon = S3ImportDaemon()

        assert daemon.config is not None
        assert daemon.config.bucket == "ringrift-models-20251214"
        assert daemon._s3_available is True
        assert len(daemon._s3_inventory) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = S3ImportConfig(bucket="test-bucket", region="us-west-2")
        daemon = S3ImportDaemon(config=config)

        assert daemon.config.bucket == "test-bucket"
        assert daemon.config.region == "us-west-2"

    def test_module_functions(self):
        """Test module-level helper functions."""
        daemon1 = get_s3_import_daemon()
        daemon2 = get_s3_import_daemon()

        assert daemon1 is daemon2

        reset_s3_import_daemon()
        daemon3 = get_s3_import_daemon()

        assert daemon1 is not daemon3


# ============================================================================
# AWS Credentials Tests
# ============================================================================


class TestAWSCredentials:
    """Tests for AWS credentials checking."""

    def test_credentials_from_env_vars(self, mock_aws_credentials):
        """Test credentials detection from environment variables."""
        daemon = S3ImportDaemon()

        assert daemon._check_aws_credentials() is True

    def test_credentials_missing(self):
        """Test when credentials are missing."""
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            # Also mock the home directory check
            with patch.object(Path, "exists", return_value=False):
                daemon = S3ImportDaemon()

                assert daemon._check_aws_credentials() is False

    def test_credentials_from_aws_config(self):
        """Test credentials detection from AWS config file."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "exists", return_value=True):
                daemon = S3ImportDaemon()

                assert daemon._check_aws_credentials() is True


# ============================================================================
# S3 File Listing Tests
# ============================================================================


class TestListS3Files:
    """Tests for S3 file listing."""

    @pytest.mark.asyncio
    async def test_list_files_no_credentials(self):
        """Test listing files when credentials are missing."""
        daemon = S3ImportDaemon()

        with patch.object(daemon, "_check_aws_credentials", return_value=False):
            files = await daemon._list_s3_files("test-prefix")

            assert files == []

    @pytest.mark.asyncio
    async def test_list_files_success(self, mock_aws_credentials):
        """Test successful file listing."""
        daemon = S3ImportDaemon()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "Contents": [
                {
                    "Key": "consolidated/games/canonical_hex8_2p.db",
                    "Size": 1048576,
                    "LastModified": "2025-01-01T12:00:00Z",
                    "ETag": '"abc123"',
                },
                {
                    "Key": "consolidated/games/canonical_hex8_4p.db",
                    "Size": 2097152,
                    "LastModified": "2025-01-02T12:00:00Z",
                    "ETag": '"def456"',
                },
            ]
        })

        with patch("asyncio.to_thread", return_value=mock_result):
            files = await daemon._list_s3_files("consolidated/games")

            assert len(files) == 2
            assert files[0].key == "consolidated/games/canonical_hex8_2p.db"
            assert files[0].size == 1048576
            assert files[0].etag == "abc123"
            assert files[1].key == "consolidated/games/canonical_hex8_4p.db"

    @pytest.mark.asyncio
    async def test_list_files_error(self, mock_aws_credentials):
        """Test file listing with AWS CLI error."""
        daemon = S3ImportDaemon()

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Access denied"

        with patch("asyncio.to_thread", return_value=mock_result):
            files = await daemon._list_s3_files("test-prefix")

            assert files == []

    @pytest.mark.asyncio
    async def test_list_files_empty(self, mock_aws_credentials):
        """Test listing when no files match."""
        daemon = S3ImportDaemon()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({})

        with patch("asyncio.to_thread", return_value=mock_result):
            files = await daemon._list_s3_files("empty-prefix")

            assert files == []


# ============================================================================
# File Download Tests
# ============================================================================


class TestDownloadFile:
    """Tests for file downloading."""

    @pytest.mark.asyncio
    async def test_download_success(self, mock_aws_credentials, tmp_base_path):
        """Test successful file download."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        local_path = tmp_base_path / "data" / "games" / "test.db"

        mock_result = MagicMock()
        mock_result.returncode = 0

        # Create the file to simulate successful download
        with patch("asyncio.to_thread", return_value=mock_result):
            # Mock the validation
            with patch.object(daemon, "_validate_import") as mock_validate:
                mock_validate.return_value = MagicMock(valid=True)

                # Create file to satisfy stat()
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(b"test data")

                result = await daemon._download_file("test/key", local_path)

                assert result is True
                assert daemon._import_stats.total_files_imported == 1

    @pytest.mark.asyncio
    async def test_download_failure(self, mock_aws_credentials, tmp_base_path):
        """Test failed file download."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        local_path = tmp_base_path / "data" / "games" / "test.db"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Download failed"

        with patch("asyncio.to_thread", return_value=mock_result):
            result = await daemon._download_file("test/key", local_path)

            assert result is False
            assert daemon._import_stats.import_errors == 1

    @pytest.mark.asyncio
    async def test_download_validation_failure(self, mock_aws_credentials, tmp_base_path):
        """Test download with validation failure."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        local_path = tmp_base_path / "data" / "games" / "test.db"

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("asyncio.to_thread", return_value=mock_result):
            with patch.object(daemon, "_validate_import") as mock_validate:
                mock_validate.return_value = MagicMock(valid=False, error="Corrupt file")

                # Create file to satisfy stat()
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(b"corrupt data")

                result = await daemon._download_file("test/key", local_path)

                assert result is False
                assert daemon._import_stats.import_errors == 1

    @pytest.mark.asyncio
    async def test_download_timeout(self, mock_aws_credentials, tmp_base_path):
        """Test download timeout handling."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        local_path = tmp_base_path / "data" / "games" / "test.db"

        import subprocess
        with patch("asyncio.to_thread", side_effect=subprocess.TimeoutExpired(cmd="aws", timeout=600)):
            result = await daemon._download_file("test/key", local_path)

            assert result is False
            assert daemon._import_stats.import_errors == 1


# ============================================================================
# Inventory Refresh Tests
# ============================================================================


class TestRefreshInventory:
    """Tests for inventory refresh."""

    @pytest.mark.asyncio
    async def test_refresh_no_credentials(self):
        """Test inventory refresh without credentials."""
        daemon = S3ImportDaemon()

        with patch.object(daemon, "_check_aws_credentials", return_value=False):
            await daemon.refresh_inventory()

            assert daemon._s3_available is False
            assert len(daemon._s3_inventory) == 0

    @pytest.mark.asyncio
    async def test_refresh_with_files(self, mock_aws_credentials):
        """Test inventory refresh with files found."""
        daemon = S3ImportDaemon()

        async def mock_list_files(prefix):
            if "games" in prefix:
                return [S3FileInfo("games/db1.db", 100, "2025-01-01", "abc")]
            elif "training" in prefix:
                return [S3FileInfo("training/npz1.npz", 200, "2025-01-01", "def")]
            elif "models" in prefix:
                return [S3FileInfo("models/model1.pth", 300, "2025-01-01", "ghi")]
            return []

        with patch.object(daemon, "_list_s3_files", side_effect=mock_list_files):
            await daemon.refresh_inventory()

            assert daemon._s3_available is True
            assert len(daemon._s3_inventory) == 3

    @pytest.mark.asyncio
    async def test_refresh_empty(self, mock_aws_credentials):
        """Test inventory refresh with no files."""
        daemon = S3ImportDaemon()

        with patch.object(daemon, "_list_s3_files", return_value=[]):
            await daemon.refresh_inventory()

            assert daemon._s3_available is False
            assert len(daemon._s3_inventory) == 0


# ============================================================================
# Import Operations Tests
# ============================================================================


class TestImportFromS3:
    """Tests for import_from_s3 method."""

    @pytest.mark.asyncio
    async def test_import_disabled(self):
        """Test import when daemon is disabled."""
        config = S3ImportConfig(enabled=False)
        daemon = S3ImportDaemon(config=config)

        result = await daemon.import_from_s3()

        assert result["success"] is False
        assert "disabled" in result["error"]

    @pytest.mark.asyncio
    async def test_import_no_credentials(self):
        """Test import without credentials."""
        daemon = S3ImportDaemon()

        with patch.object(daemon, "_check_aws_credentials", return_value=False):
            result = await daemon.import_from_s3()

            assert result["success"] is False
            assert "credentials" in result["error"]

    @pytest.mark.asyncio
    async def test_import_specific_config(self, mock_aws_credentials, tmp_base_path, mock_safe_emit):
        """Test import with config filter."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        daemon._s3_inventory = {
            "consolidated/games/canonical_hex8_2p.db": S3FileInfo(
                "consolidated/games/canonical_hex8_2p.db", 1000, "2025-01-01", "abc"
            ),
            "consolidated/games/canonical_square8_2p.db": S3FileInfo(
                "consolidated/games/canonical_square8_2p.db", 2000, "2025-01-01", "def"
            ),
        }

        async def mock_download(s3_key, local_path):
            local_path.write_bytes(b"test")
            return True

        with patch.object(daemon, "_download_file", side_effect=mock_download):
            result = await daemon.import_from_s3(config_key="hex8_2p")

            assert result["success"] is True
            assert result["files_imported"] == 1

    @pytest.mark.asyncio
    async def test_import_skip_existing(self, mock_aws_credentials, tmp_base_path):
        """Test that existing files are skipped without force."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        # Create existing file
        existing = tmp_base_path / "data" / "games" / "canonical_hex8_2p.db"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"existing data")

        daemon._s3_inventory = {
            "consolidated/games/canonical_hex8_2p.db": S3FileInfo(
                "consolidated/games/canonical_hex8_2p.db", 1000, "2025-01-01", "abc"
            ),
        }

        result = await daemon.import_from_s3(data_type="games", force=False)

        assert result["files_imported"] == 0  # Skipped existing

    @pytest.mark.asyncio
    async def test_import_force_redownload(self, mock_aws_credentials, tmp_base_path, mock_safe_emit):
        """Test force redownload of existing files."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        # Create existing file
        existing = tmp_base_path / "data" / "games" / "canonical_hex8_2p.db"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"existing data")

        daemon._s3_inventory = {
            "consolidated/games/canonical_hex8_2p.db": S3FileInfo(
                "consolidated/games/canonical_hex8_2p.db", 1000, "2025-01-01", "abc"
            ),
        }

        async def mock_download(s3_key, local_path):
            local_path.write_bytes(b"new data")
            return True

        with patch.object(daemon, "_download_file", side_effect=mock_download):
            result = await daemon.import_from_s3(data_type="games", force=True)

            assert result["files_imported"] == 1


class TestImportForRecovery:
    """Tests for full recovery import."""

    @pytest.mark.asyncio
    async def test_recovery_import(self, mock_aws_credentials, tmp_base_path, mock_safe_emit):
        """Test full recovery import."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        async def mock_refresh():
            daemon._s3_inventory = {
                "consolidated/games/db1.db": S3FileInfo("consolidated/games/db1.db", 100, "2025-01-01", "abc"),
            }

        async def mock_import(**kwargs):
            return {"success": True, "files_imported": 1, "bytes_imported": 100, "errors": []}

        with patch.object(daemon, "refresh_inventory", side_effect=mock_refresh):
            with patch.object(daemon, "import_from_s3", side_effect=mock_import):
                result = await daemon.import_for_recovery()

                assert result["success"] is True
                assert result["files_imported"] == 1


class TestImportMissingConfigs:
    """Tests for missing config import."""

    @pytest.mark.asyncio
    async def test_import_missing_configs(self, mock_aws_credentials, tmp_base_path):
        """Test importing missing configurations."""
        daemon = S3ImportDaemon()
        daemon._base_path = tmp_base_path

        # Create local games directory with one config
        games_dir = tmp_base_path / "data" / "games"
        games_dir.mkdir(parents=True, exist_ok=True)
        (games_dir / "canonical_hex8_2p.db").write_bytes(b"local")

        async def mock_refresh():
            daemon._s3_inventory = {
                "consolidated/games/canonical_hex8_2p.db": S3FileInfo(
                    "consolidated/games/canonical_hex8_2p.db", 100, "2025-01-01", "abc"
                ),
                "consolidated/games/canonical_hex8_4p.db": S3FileInfo(
                    "consolidated/games/canonical_hex8_4p.db", 200, "2025-01-01", "def"
                ),
            }

        async def mock_import(config_key=None, **kwargs):
            return {"success": True, "files_imported": 1 if config_key else 0, "errors": []}

        with patch.object(daemon, "refresh_inventory", side_effect=mock_refresh):
            with patch.object(daemon, "import_from_s3", side_effect=mock_import):
                result = await daemon.import_missing_configs()

                assert "hex8_4p" in result["missing_configs"]
                assert "hex8_2p" not in result["missing_configs"]


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Tests for event emission."""

    def test_emit_import_complete(self, mock_safe_emit):
        """Test import complete event emission."""
        daemon = S3ImportDaemon()

        results = {"files_imported": 5, "bytes_imported": 1024 * 1024}

        daemon._emit_import_complete(results)

        mock_safe_emit.assert_called_once()
        call_args = mock_safe_emit.call_args
        assert call_args[0][0] == "data_sync_completed"
        assert call_args[0][1]["sync_type"] == "s3_import"
        assert call_args[0][1]["files_imported"] == 5


# ============================================================================
# Event Subscription Tests
# ============================================================================


class TestEventSubscriptions:
    """Tests for event subscriptions."""

    def test_get_event_subscriptions(self):
        """Test event subscription registration."""
        daemon = S3ImportDaemon()

        subs = daemon._get_event_subscriptions()

        assert "CLUSTER_RECOVERY_REQUESTED" in subs
        assert "NODE_BOOTSTRAP_REQUESTED" in subs

    @pytest.mark.asyncio
    async def test_on_recovery_requested(self, mock_aws_credentials):
        """Test recovery request handler."""
        daemon = S3ImportDaemon()

        async def mock_recovery():
            return {"success": True, "files_imported": 5}

        with patch.object(daemon, "import_for_recovery", side_effect=mock_recovery):
            await daemon._on_recovery_requested({})

    @pytest.mark.asyncio
    async def test_on_bootstrap_requested(self, mock_aws_credentials):
        """Test bootstrap request handler."""
        daemon = S3ImportDaemon()

        async def mock_import():
            return {"success": True, "imported": ["hex8_2p"]}

        with patch.object(daemon, "import_missing_configs", side_effect=mock_import):
            await daemon._on_bootstrap_requested({})


# ============================================================================
# Run Cycle Tests
# ============================================================================


class TestRunCycle:
    """Tests for daemon run cycle."""

    @pytest.mark.asyncio
    async def test_run_cycle_disabled(self):
        """Test run cycle when daemon is disabled."""
        config = S3ImportConfig(enabled=False)
        daemon = S3ImportDaemon(config=config)

        with patch.object(daemon, "refresh_inventory") as mock_refresh:
            await daemon._run_cycle()

            mock_refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_cycle_enabled(self, mock_aws_credentials):
        """Test run cycle when daemon is enabled."""
        daemon = S3ImportDaemon()

        with patch.object(daemon, "refresh_inventory") as mock_refresh:
            await daemon._run_cycle()

            mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_startup_import(self, mock_aws_credentials):
        """Test run cycle with startup import enabled."""
        config = S3ImportConfig(import_on_startup=True)
        daemon = S3ImportDaemon(config=config)
        daemon._stats.cycles_completed = 0

        with patch.object(daemon, "refresh_inventory"):
            with patch.object(daemon, "import_missing_configs") as mock_import:
                await daemon._run_cycle()

                mock_import.assert_called_once()


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check."""

    def test_health_check_not_running(self):
        """Test health check when not running."""
        daemon = S3ImportDaemon()
        daemon._running = False

        result = daemon.health_check()

        assert result.healthy is False
        assert "not running" in result.message.lower()

    def test_health_check_no_credentials(self, mock_aws_credentials):
        """Test health check without AWS credentials."""
        daemon = S3ImportDaemon()
        daemon._running = True

        with patch.object(daemon, "_check_aws_credentials", return_value=False):
            result = daemon.health_check()

            assert result.healthy is True
            assert "credentials not configured" in result.message.lower()
            assert result.details["s3_available"] is False

    def test_health_check_healthy(self, mock_aws_credentials):
        """Test health check when healthy."""
        daemon = S3ImportDaemon()
        daemon._running = True
        daemon._s3_inventory = {"key1": MagicMock(), "key2": MagicMock()}
        daemon._stats.cycles_completed = 5
        daemon._import_stats.total_files_imported = 10

        result = daemon.health_check()

        assert result.healthy is True
        assert "2 files" in result.message
        assert result.details["inventory_size"] == 2
        assert result.details["total_files_imported"] == 10


# ============================================================================
# Statistics Tests
# ============================================================================


class TestGetStats:
    """Tests for statistics retrieval."""

    def test_get_stats(self):
        """Test statistics retrieval."""
        daemon = S3ImportDaemon()
        daemon._import_stats.total_files_imported = 15
        daemon._import_stats.total_bytes_imported = 100 * 1024 * 1024
        daemon._import_stats.import_errors = 3
        daemon._import_stats.configs_imported = ["hex8_2p", "hex8_4p"]
        daemon._s3_inventory = {"k1": MagicMock(), "k2": MagicMock(), "k3": MagicMock()}

        stats = daemon.get_stats()

        assert stats["total_files_imported"] == 15
        assert stats["total_mb_imported"] == 100.0
        assert stats["import_errors"] == 3
        assert stats["inventory_size"] == 3
        assert "hex8_2p" in stats["configs_imported"]


class TestGetInventory:
    """Tests for inventory retrieval."""

    def test_get_inventory(self):
        """Test inventory retrieval."""
        daemon = S3ImportDaemon()
        daemon._s3_inventory = {
            "key1": S3FileInfo("key1", 1000, "2025-01-01", "abc123"),
            "key2": S3FileInfo("key2", 2000, "2025-01-02", "def456"),
        }

        inventory = daemon.get_inventory()

        assert len(inventory) == 2
        assert inventory["key1"]["size"] == 1000
        assert inventory["key1"]["etag"] == "abc123"
        assert inventory["key2"]["size"] == 2000

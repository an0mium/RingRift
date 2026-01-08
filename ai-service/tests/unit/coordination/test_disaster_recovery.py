"""Tests for DisasterRecoveryManager.

Tests the disaster recovery capabilities:
1. Restore from S3 bucket
2. Restore from OWC external drive
3. Backup completeness verification
4. Selective restoration by config
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.disaster_recovery import (
    BackupVerificationResult,
    DisasterRecoveryManager,
    RecoveryConfig,
    RestoredFile,
    RestoreResult,
    RestoreStatus,
    get_disaster_recovery_manager,
    reset_disaster_recovery_manager,
)
from app.distributed.cluster_manifest import DataSource


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_disaster_recovery_manager()
    yield
    reset_disaster_recovery_manager()


@pytest.fixture
def mock_manifest():
    """Mock cluster manifest."""
    with patch("app.coordination.disaster_recovery.get_cluster_manifest") as mock:
        manifest = MagicMock()
        manifest.find_across_all_sources.return_value = {}
        manifest.find_external_storage_for_config.return_value = []
        mock.return_value = manifest
        yield manifest


@pytest.fixture
def tmp_restore_dir(tmp_path):
    """Create a temporary restore directory."""
    restore_dir = tmp_path / "restored"
    restore_dir.mkdir()
    return restore_dir


# =============================================================================
# Test RestoreStatus Enum
# =============================================================================


class TestRestoreStatus:
    """Tests for RestoreStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert RestoreStatus.SUCCESS.value == "success"
        assert RestoreStatus.PARTIAL.value == "partial"
        assert RestoreStatus.FAILED.value == "failed"
        assert RestoreStatus.NO_DATA.value == "no_data"

    def test_status_is_string_enum(self):
        """Test status is string-based."""
        assert isinstance(RestoreStatus.SUCCESS, str)
        assert RestoreStatus.SUCCESS == "success"


# =============================================================================
# Test RestoredFile Dataclass
# =============================================================================


class TestRestoredFile:
    """Tests for RestoredFile dataclass."""

    def test_basic_creation(self):
        """Test basic RestoredFile creation."""
        file = RestoredFile(
            config_key="hex8_2p",
            source_path="s3://bucket/db.db",
            local_path=Path("/data/db.db"),
            game_count=1000,
            file_size_mb=50.5,
            restore_time_seconds=10.5,
            success=True,
        )
        assert file.config_key == "hex8_2p"
        assert file.source_path == "s3://bucket/db.db"
        assert file.local_path == Path("/data/db.db")
        assert file.game_count == 1000
        assert file.file_size_mb == 50.5
        assert file.restore_time_seconds == 10.5
        assert file.success is True
        assert file.error is None

    def test_creation_with_error(self):
        """Test RestoredFile with error."""
        file = RestoredFile(
            config_key="hex8_2p",
            source_path="s3://bucket/db.db",
            local_path=Path("/data/db.db"),
            game_count=0,
            file_size_mb=0,
            restore_time_seconds=5.0,
            success=False,
            error="Download failed",
        )
        assert file.success is False
        assert file.error == "Download failed"


# =============================================================================
# Test RestoreResult Dataclass
# =============================================================================


class TestRestoreResult:
    """Tests for RestoreResult dataclass."""

    def test_basic_creation(self):
        """Test basic RestoreResult creation."""
        result = RestoreResult(
            status=RestoreStatus.SUCCESS,
            source=DataSource.S3,
            target_dir=Path("/data/restored"),
        )
        assert result.status == RestoreStatus.SUCCESS
        assert result.source == DataSource.S3
        assert result.target_dir == Path("/data/restored")
        assert result.files_restored == []
        assert result.files_failed == []
        assert result.total_games_restored == 0
        assert result.total_size_mb == 0.0
        assert result.total_time_seconds == 0.0
        assert result.error is None

    def test_success_property_true(self):
        """Test success property returns True for SUCCESS."""
        result = RestoreResult(
            status=RestoreStatus.SUCCESS,
            source=DataSource.S3,
            target_dir=Path("/data"),
        )
        assert result.success is True

    def test_success_property_partial(self):
        """Test success property returns True for PARTIAL."""
        result = RestoreResult(
            status=RestoreStatus.PARTIAL,
            source=DataSource.S3,
            target_dir=Path("/data"),
        )
        assert result.success is True

    def test_success_property_failed(self):
        """Test success property returns False for FAILED."""
        result = RestoreResult(
            status=RestoreStatus.FAILED,
            source=DataSource.S3,
            target_dir=Path("/data"),
        )
        assert result.success is False

    def test_success_property_no_data(self):
        """Test success property returns False for NO_DATA."""
        result = RestoreResult(
            status=RestoreStatus.NO_DATA,
            source=DataSource.S3,
            target_dir=Path("/data"),
        )
        assert result.success is False


# =============================================================================
# Test BackupVerificationResult Dataclass
# =============================================================================


class TestBackupVerificationResult:
    """Tests for BackupVerificationResult dataclass."""

    def test_basic_creation(self):
        """Test basic BackupVerificationResult creation."""
        result = BackupVerificationResult(
            timestamp=1234567890.0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p", "square8_2p"],
            owc_configs=["hex8_2p", "square8_2p"],
            s3_only_configs=[],
            owc_only_configs=[],
            both_configs=["hex8_2p", "square8_2p"],
            missing_configs=[],
            s3_total_games=10000,
            owc_total_games=10000,
            recommendation="All good",
        )
        assert result.timestamp == 1234567890.0
        assert result.s3_verified is True
        assert len(result.s3_configs) == 2
        assert result.s3_total_games == 10000

    def test_fully_backed_up_true(self):
        """Test fully_backed_up when no missing configs."""
        result = BackupVerificationResult(
            timestamp=0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p"],
            owc_configs=["hex8_2p"],
            s3_only_configs=[],
            owc_only_configs=[],
            both_configs=["hex8_2p"],
            missing_configs=[],
            s3_total_games=100,
            owc_total_games=100,
            recommendation="Good",
        )
        assert result.fully_backed_up is True

    def test_fully_backed_up_false(self):
        """Test fully_backed_up when configs are missing."""
        result = BackupVerificationResult(
            timestamp=0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p"],
            owc_configs=["hex8_2p"],
            s3_only_configs=[],
            owc_only_configs=[],
            both_configs=["hex8_2p"],
            missing_configs=["square8_2p"],
            s3_total_games=100,
            owc_total_games=100,
            recommendation="Bad",
        )
        assert result.fully_backed_up is False

    def test_redundant_true(self):
        """Test redundant when all configs in both."""
        result = BackupVerificationResult(
            timestamp=0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p"],
            owc_configs=["hex8_2p"],
            s3_only_configs=[],
            owc_only_configs=[],
            both_configs=["hex8_2p"],
            missing_configs=[],
            s3_total_games=100,
            owc_total_games=100,
            recommendation="Good",
        )
        assert result.redundant is True

    def test_redundant_false_s3_only(self):
        """Test redundant when some configs only in S3."""
        result = BackupVerificationResult(
            timestamp=0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p", "square8_2p"],
            owc_configs=["hex8_2p"],
            s3_only_configs=["square8_2p"],
            owc_only_configs=[],
            both_configs=["hex8_2p"],
            missing_configs=[],
            s3_total_games=200,
            owc_total_games=100,
            recommendation="Partial",
        )
        assert result.redundant is False

    def test_redundant_false_owc_only(self):
        """Test redundant when some configs only in OWC."""
        result = BackupVerificationResult(
            timestamp=0,
            s3_verified=True,
            owc_verified=True,
            s3_configs=["hex8_2p"],
            owc_configs=["hex8_2p", "square8_2p"],
            s3_only_configs=[],
            owc_only_configs=["square8_2p"],
            both_configs=["hex8_2p"],
            missing_configs=[],
            s3_total_games=100,
            owc_total_games=200,
            recommendation="Partial",
        )
        assert result.redundant is False


# =============================================================================
# Test RecoveryConfig Dataclass
# =============================================================================


class TestRecoveryConfig:
    """Tests for RecoveryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RecoveryConfig()
        assert config.s3_bucket == "ringrift-models-20251214"
        assert config.s3_region == "us-east-1"
        assert config.s3_prefix == "databases/"
        assert config.owc_host == "mac-studio"
        assert config.owc_base_path == "/Volumes/RingRift-Data"
        assert config.owc_db_subpath == "databases"
        assert config.download_timeout == 600.0
        assert config.verify_timeout == 300.0
        assert config.target_dir == Path("data/games/restored")

    def test_custom_config(self):
        """Test custom configuration."""
        config = RecoveryConfig(
            s3_bucket="my-bucket",
            owc_host="other-host",
            download_timeout=120.0,
            target_dir=Path("/custom/path"),
        )
        assert config.s3_bucket == "my-bucket"
        assert config.owc_host == "other-host"
        assert config.download_timeout == 120.0
        assert config.target_dir == Path("/custom/path")


# =============================================================================
# Test DisasterRecoveryManager Basic
# =============================================================================


class TestDisasterRecoveryManagerBasic:
    """Basic tests for DisasterRecoveryManager."""

    def test_initialization_default(self, mock_manifest):
        """Test default initialization."""
        manager = DisasterRecoveryManager()
        assert manager.config is not None
        assert manager.config.s3_bucket == "ringrift-models-20251214"

    def test_initialization_with_config(self, mock_manifest):
        """Test initialization with custom config."""
        config = RecoveryConfig(s3_bucket="custom-bucket")
        manager = DisasterRecoveryManager(config=config)
        assert manager.config.s3_bucket == "custom-bucket"


# =============================================================================
# Test Restore from S3
# =============================================================================


class TestRestoreFromS3:
    """Tests for restore_from_s3 method."""

    @pytest.mark.asyncio
    async def test_restore_no_data(self, mock_manifest, tmp_restore_dir):
        """Test restore when no S3 data exists."""
        mock_manifest.find_across_all_sources.return_value = {}
        mock_manifest.find_external_storage_for_config.return_value = []

        manager = DisasterRecoveryManager()
        result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.NO_DATA
        assert result.source == DataSource.S3
        assert result.error == "No S3 backup data found"

    @pytest.mark.asyncio
    async def test_restore_success(self, mock_manifest, tmp_restore_dir):
        """Test successful S3 restore."""
        # Create a mock file to be "downloaded"
        test_file = tmp_restore_dir / "hex8_2p_s3_restored.db"

        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "s3_key": "databases/hex8_2p.db",
                "s3_bucket": "ringrift-models-20251214",
                "game_count": 1000,
            }
        ]

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            # Simulate successful download
            async def create_file(*args, **kwargs):
                test_file.write_bytes(b"mock database content")
                return True

            mock_download.side_effect = create_file

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.SUCCESS
        assert result.source == DataSource.S3
        assert len(result.files_restored) == 1
        assert result.files_restored[0].config_key == "hex8_2p"
        assert result.files_restored[0].success is True

    @pytest.mark.asyncio
    async def test_restore_partial(self, mock_manifest, tmp_restore_dir):
        """Test partial S3 restore (some files fail)."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "s3_key": "databases/hex8_2p.db",
                "game_count": 1000,
            },
            {
                "source": "s3",
                "config_key": "square8_2p",
                "s3_key": "databases/square8_2p.db",
                "game_count": 2000,
            },
        ]

        test_file = tmp_restore_dir / "hex8_2p_s3_restored.db"
        call_count = [0]

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            async def partial_download(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    test_file.write_bytes(b"mock content")
                    return True
                return False  # Second call fails

            mock_download.side_effect = partial_download

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.PARTIAL
        assert len(result.files_restored) == 1
        assert len(result.files_failed) == 1

    @pytest.mark.asyncio
    async def test_restore_failed(self, mock_manifest, tmp_restore_dir):
        """Test failed S3 restore (all files fail)."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "s3_key": "databases/hex8_2p.db",
                "game_count": 1000,
            },
        ]

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            mock_download.return_value = False

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.FAILED
        assert len(result.files_restored) == 0
        assert len(result.files_failed) == 1

    @pytest.mark.asyncio
    async def test_restore_with_config_filter(self, mock_manifest, tmp_restore_dir):
        """Test S3 restore with specific config keys."""
        mock_manifest.find_across_all_sources.return_value = {
            DataSource.S3: [
                {
                    "config_key": "hex8_2p",
                    "s3_key": "databases/hex8_2p.db",
                    "game_count": 1000,
                }
            ]
        }

        test_file = tmp_restore_dir / "hex8_2p_s3_restored.db"

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            async def create_file(*args, **kwargs):
                test_file.write_bytes(b"mock content")
                return True

            mock_download.side_effect = create_file

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(
                target_dir=tmp_restore_dir,
                config_keys=["hex8_2p"],
            )

        assert result.status == RestoreStatus.SUCCESS
        mock_manifest.find_across_all_sources.assert_called_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_restore_exception_handling(self, mock_manifest, tmp_restore_dir):
        """Test S3 restore handles exceptions gracefully."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "s3_key": "databases/hex8_2p.db",
                "game_count": 1000,
            },
        ]

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            mock_download.side_effect = Exception("Network error")

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.FAILED
        assert len(result.files_failed) == 1
        assert "Network error" in result.files_failed[0].error


# =============================================================================
# Test Restore from OWC
# =============================================================================


class TestRestoreFromOWC:
    """Tests for restore_from_owc method."""

    @pytest.mark.asyncio
    async def test_restore_no_data(self, mock_manifest, tmp_restore_dir):
        """Test restore when no OWC data exists."""
        mock_manifest.find_across_all_sources.return_value = {}
        mock_manifest.find_external_storage_for_config.return_value = []

        manager = DisasterRecoveryManager()
        result = await manager.restore_from_owc(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.NO_DATA
        assert result.source == DataSource.OWC
        assert result.error == "No OWC backup data found"

    @pytest.mark.asyncio
    async def test_restore_success(self, mock_manifest, tmp_restore_dir):
        """Test successful OWC restore."""
        test_file = tmp_restore_dir / "hex8_2p_owc_restored.db"

        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "owc",
                "config_key": "hex8_2p",
                "owc_path": "/Volumes/RingRift-Data/hex8_2p.db",
                "owc_host": "mac-studio",
                "game_count": 1000,
            }
        ]

        with patch.object(
            DisasterRecoveryManager, "_download_from_owc"
        ) as mock_download:
            async def create_file(*args, **kwargs):
                test_file.write_bytes(b"mock database content")
                return True

            mock_download.side_effect = create_file

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_owc(target_dir=tmp_restore_dir)

        assert result.status == RestoreStatus.SUCCESS
        assert result.source == DataSource.OWC
        assert len(result.files_restored) == 1
        assert result.files_restored[0].config_key == "hex8_2p"

    @pytest.mark.asyncio
    async def test_restore_with_custom_host(self, mock_manifest, tmp_restore_dir):
        """Test OWC restore with custom host."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "owc",
                "config_key": "hex8_2p",
                "owc_path": "/Volumes/Data/hex8_2p.db",
                "game_count": 1000,
            }
        ]

        test_file = tmp_restore_dir / "hex8_2p_owc_restored.db"

        with patch.object(
            DisasterRecoveryManager, "_download_from_owc"
        ) as mock_download:
            async def create_file(*args, owc_host=None, **kwargs):
                test_file.write_bytes(b"mock content")
                return True

            mock_download.side_effect = create_file

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_owc(
                target_dir=tmp_restore_dir,
                owc_host="custom-host",
            )

        assert result.status == RestoreStatus.SUCCESS


# =============================================================================
# Test Backup Verification
# =============================================================================


class TestVerifyBackupCompleteness:
    """Tests for verify_backup_completeness method."""

    @pytest.mark.asyncio
    async def test_verify_both_complete(self, mock_manifest):
        """Test verification when both S3 and OWC are complete."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
            {
                "source": "owc",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
        ]

        manager = DisasterRecoveryManager()
        result = await manager.verify_backup_completeness()

        assert result.s3_verified is True
        assert result.owc_verified is True
        assert "hex8_2p" in result.both_configs
        assert len(result.s3_only_configs) == 0
        assert len(result.owc_only_configs) == 0
        assert "No action needed" in result.recommendation

    @pytest.mark.asyncio
    async def test_verify_s3_only(self, mock_manifest):
        """Test verification when config only in S3."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
            {
                "source": "s3",
                "config_key": "square8_2p",
                "game_count": 2000,
            },
            {
                "source": "owc",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
        ]

        manager = DisasterRecoveryManager()
        result = await manager.verify_backup_completeness()

        assert "square8_2p" in result.s3_only_configs
        assert "hex8_2p" in result.both_configs
        assert "WARNING" in result.recommendation

    @pytest.mark.asyncio
    async def test_verify_owc_only(self, mock_manifest):
        """Test verification when config only in OWC."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
            {
                "source": "owc",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
            {
                "source": "owc",
                "config_key": "square8_2p",
                "game_count": 2000,
            },
        ]

        manager = DisasterRecoveryManager()
        result = await manager.verify_backup_completeness()

        assert "square8_2p" in result.owc_only_configs
        assert "hex8_2p" in result.both_configs

    @pytest.mark.asyncio
    async def test_verify_missing_configs(self, mock_manifest):
        """Test verification when configs are missing."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "game_count": 1000,
            },
        ]
        mock_manifest.get_all_configs.return_value = ["hex8_2p", "square8_2p"]

        manager = DisasterRecoveryManager()
        result = await manager.verify_backup_completeness()

        assert "square8_2p" in result.missing_configs
        assert "CRITICAL" in result.recommendation

    @pytest.mark.asyncio
    async def test_verify_no_backups(self, mock_manifest):
        """Test verification when no backups exist."""
        mock_manifest.find_external_storage_for_config.return_value = []

        manager = DisasterRecoveryManager()
        result = await manager.verify_backup_completeness()

        assert result.s3_verified is False
        assert result.owc_verified is False


# =============================================================================
# Test Download from S3
# =============================================================================


class TestDownloadFromS3:
    """Tests for _download_from_s3 method."""

    @pytest.mark.asyncio
    async def test_download_success(self, mock_manifest, tmp_restore_dir):
        """Test successful S3 download."""
        local_path = tmp_restore_dir / "test.db"

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            manager = DisasterRecoveryManager()
            result = await manager._download_from_s3(
                s3_bucket="test-bucket",
                s3_key="databases/test.db",
                local_path=local_path,
            )

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "aws" in call_args
        assert "s3" in call_args
        assert "cp" in call_args

    @pytest.mark.asyncio
    async def test_download_failure(self, mock_manifest, tmp_restore_dir):
        """Test failed S3 download."""
        local_path = tmp_restore_dir / "test.db"

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            manager = DisasterRecoveryManager()
            result = await manager._download_from_s3(
                s3_bucket="test-bucket",
                s3_key="databases/test.db",
                local_path=local_path,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_download_timeout(self, mock_manifest, tmp_restore_dir):
        """Test S3 download timeout."""
        local_path = tmp_restore_dir / "test.db"

        config = RecoveryConfig(download_timeout=0.1)  # Very short timeout

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="aws", timeout=0.1)

            manager = DisasterRecoveryManager(config=config)
            result = await manager._download_from_s3(
                s3_bucket="test-bucket",
                s3_key="databases/test.db",
                local_path=local_path,
            )

        assert result is False


# =============================================================================
# Test Download from OWC
# =============================================================================


class TestDownloadFromOWC:
    """Tests for _download_from_owc method."""

    @pytest.mark.asyncio
    async def test_download_success(self, mock_manifest, tmp_restore_dir):
        """Test successful OWC download."""
        local_path = tmp_restore_dir / "test.db"

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            manager = DisasterRecoveryManager()
            result = await manager._download_from_owc(
                owc_host="mac-studio",
                owc_path="/Volumes/Data/test.db",
                local_path=local_path,
            )

        assert result is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "rsync" in call_args
        assert "-az" in call_args

    @pytest.mark.asyncio
    async def test_download_failure(self, mock_manifest, tmp_restore_dir):
        """Test failed OWC download."""
        local_path = tmp_restore_dir / "test.db"

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            manager = DisasterRecoveryManager()
            result = await manager._download_from_owc(
                owc_host="mac-studio",
                owc_path="/Volumes/Data/test.db",
                local_path=local_path,
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_download_timeout(self, mock_manifest, tmp_restore_dir):
        """Test OWC download timeout."""
        local_path = tmp_restore_dir / "test.db"

        config = RecoveryConfig(download_timeout=0.1)

        with patch("app.coordination.disaster_recovery.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="rsync", timeout=0.1)

            manager = DisasterRecoveryManager(config=config)
            result = await manager._download_from_owc(
                owc_host="mac-studio",
                owc_path="/Volumes/Data/test.db",
                local_path=local_path,
            )

        assert result is False


# =============================================================================
# Test Get Locations
# =============================================================================


class TestGetLocations:
    """Tests for _get_s3_locations and _get_owc_locations methods."""

    @pytest.mark.asyncio
    async def test_get_s3_locations_all(self, mock_manifest):
        """Test getting all S3 locations."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {"source": "s3", "config_key": "hex8_2p"},
            {"source": "owc", "config_key": "hex8_2p"},
            {"source": "s3", "config_key": "square8_2p"},
        ]

        manager = DisasterRecoveryManager()
        locations = await manager._get_s3_locations()

        assert len(locations) == 2
        assert all(loc["source"] == "s3" for loc in locations)

    @pytest.mark.asyncio
    async def test_get_s3_locations_filtered(self, mock_manifest):
        """Test getting S3 locations for specific configs."""
        mock_manifest.find_across_all_sources.return_value = {
            DataSource.S3: [{"config_key": "hex8_2p"}],
            DataSource.OWC: [],
        }

        manager = DisasterRecoveryManager()
        locations = await manager._get_s3_locations(config_keys=["hex8_2p"])

        assert len(locations) == 1
        mock_manifest.find_across_all_sources.assert_called_with("hex8_2p")

    @pytest.mark.asyncio
    async def test_get_owc_locations_all(self, mock_manifest):
        """Test getting all OWC locations."""
        mock_manifest.find_external_storage_for_config.return_value = [
            {"source": "owc", "config_key": "hex8_2p"},
            {"source": "s3", "config_key": "hex8_2p"},
        ]

        manager = DisasterRecoveryManager()
        locations = await manager._get_owc_locations()

        assert len(locations) == 1
        assert locations[0]["source"] == "owc"

    @pytest.mark.asyncio
    async def test_get_owc_locations_filtered(self, mock_manifest):
        """Test getting OWC locations for specific configs."""
        mock_manifest.find_across_all_sources.return_value = {
            DataSource.S3: [],
            DataSource.OWC: [{"config_key": "hex8_2p"}],
        }

        manager = DisasterRecoveryManager()
        locations = await manager._get_owc_locations(config_keys=["hex8_2p"])

        assert len(locations) == 1


# =============================================================================
# Test Module Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_get_disaster_recovery_manager(self, mock_manifest):
        """Test singleton accessor."""
        manager1 = get_disaster_recovery_manager()
        manager2 = get_disaster_recovery_manager()
        assert manager1 is manager2

    def test_reset_disaster_recovery_manager(self, mock_manifest):
        """Test singleton reset."""
        manager1 = get_disaster_recovery_manager()
        reset_disaster_recovery_manager()
        manager2 = get_disaster_recovery_manager()
        assert manager1 is not manager2


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration-style tests for disaster recovery scenarios."""

    @pytest.mark.asyncio
    async def test_full_restore_workflow(self, mock_manifest, tmp_restore_dir):
        """Test complete restore workflow."""
        # Setup: S3 has backup
        mock_manifest.find_external_storage_for_config.return_value = [
            {
                "source": "s3",
                "config_key": "hex8_2p",
                "s3_key": "databases/hex8_2p.db",
                "s3_bucket": "test-bucket",
                "game_count": 5000,
            },
        ]

        test_file = tmp_restore_dir / "hex8_2p_s3_restored.db"

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            async def create_large_file(*args, **kwargs):
                test_file.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MB
                return True

            mock_download.side_effect = create_large_file

            manager = DisasterRecoveryManager()
            result = await manager.restore_from_s3(target_dir=tmp_restore_dir)

        assert result.success is True
        assert result.total_games_restored == 5000
        assert result.total_size_mb >= 10.0
        assert result.total_time_seconds > 0

    @pytest.mark.asyncio
    async def test_verify_and_restore_workflow(self, mock_manifest, tmp_restore_dir):
        """Test verify then restore workflow."""
        # Setup: Some data missing from OWC
        mock_manifest.find_external_storage_for_config.return_value = [
            {"source": "s3", "config_key": "hex8_2p", "game_count": 1000},
            {"source": "s3", "config_key": "square8_2p", "game_count": 2000},
            {"source": "owc", "config_key": "hex8_2p", "game_count": 1000},
            # square8_2p missing from OWC
        ]

        manager = DisasterRecoveryManager()

        # Step 1: Verify backups
        verify_result = await manager.verify_backup_completeness()
        assert "square8_2p" in verify_result.s3_only_configs
        assert verify_result.redundant is False

        # Step 2: Restore missing from S3
        test_file = tmp_restore_dir / "square8_2p_s3_restored.db"
        mock_manifest.find_across_all_sources.return_value = {
            DataSource.S3: [
                {
                    "config_key": "square8_2p",
                    "s3_key": "databases/square8_2p.db",
                    "game_count": 2000,
                }
            ]
        }

        with patch.object(
            DisasterRecoveryManager, "_download_from_s3"
        ) as mock_download:
            async def create_file(*args, **kwargs):
                test_file.write_bytes(b"mock content")
                return True

            mock_download.side_effect = create_file

            restore_result = await manager.restore_from_s3(
                target_dir=tmp_restore_dir,
                config_keys=["square8_2p"],
            )

        assert restore_result.success is True

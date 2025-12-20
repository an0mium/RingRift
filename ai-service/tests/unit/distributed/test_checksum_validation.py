"""Tests for data sync checksum validation."""

import asyncio
import hashlib
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestChecksumValidation:
    """Tests for checksum validation in data sync."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary valid SQLite database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, data TEXT)")
        cursor.execute("INSERT INTO games (data) VALUES ('test')")
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def corrupt_db(self, tmp_path):
        """Create a corrupted database file."""
        db_path = tmp_path / "corrupt.db"
        # Write invalid data that looks like SQLite header but is corrupted
        db_path.write_bytes(b"SQLite format 3\x00" + b"\xff" * 100)
        return db_path

    def test_compute_file_checksum(self, temp_db):
        """Should compute correct SHA256 checksum."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        config = SyncConfig()
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config

        checksum = service._compute_file_checksum(temp_db)

        # Verify it's a valid hex SHA256 (64 characters)
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

        # Verify it matches manual computation
        expected = hashlib.sha256(temp_db.read_bytes()).hexdigest()
        assert checksum == expected

    def test_checksum_changes_with_content(self, tmp_path):
        """Checksum should change when file content changes."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        config = SyncConfig()
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config

        # Create first file
        file1 = tmp_path / "file1.db"
        file1.write_bytes(b"content 1")
        checksum1 = service._compute_file_checksum(file1)

        # Create second file with different content
        file2 = tmp_path / "file2.db"
        file2.write_bytes(b"content 2")
        checksum2 = service._compute_file_checksum(file2)

        assert checksum1 != checksum2

    @pytest.mark.asyncio
    async def test_validate_synced_files_valid(self, tmp_path):
        """Should validate valid database files successfully."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        # Create valid database
        db_path = tmp_path / "valid.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        # Create service with mock manifest
        config = SyncConfig(checksum_validation=True)
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config
        service.manifest = MagicMock()

        result = await service._validate_synced_files(tmp_path, "test-host")

        assert result["valid"] is True
        assert result["files_validated"] == 1
        assert len(result["errors"]) == 0
        assert "valid.db" in result["checksums"]

    @pytest.mark.asyncio
    async def test_validate_synced_files_corrupt(self, corrupt_db):
        """Should detect corrupted database files."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        config = SyncConfig(checksum_validation=True)
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config
        service.manifest = MagicMock()

        result = await service._validate_synced_files(corrupt_db.parent, "test-host")

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        # Should have database error or integrity failure
        assert any("error" in err.lower() or "integrity" in err.lower() for err in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_synced_files_empty_dir(self, tmp_path):
        """Should handle empty directory gracefully."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        config = SyncConfig(checksum_validation=True)
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config
        service.manifest = MagicMock()

        result = await service._validate_synced_files(tmp_path, "test-host")

        assert result["valid"] is True
        assert result["files_validated"] == 0
        assert len(result["checksums"]) == 0

    @pytest.mark.asyncio
    async def test_validate_synced_files_stores_checksums(self, tmp_path):
        """Should store checksums in manifest if available."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        # Create valid database
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER)")
        conn.commit()
        conn.close()

        config = SyncConfig(checksum_validation=True)
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config
        mock_manifest = MagicMock()
        mock_manifest.record_checksums = MagicMock()
        service.manifest = mock_manifest

        await service._validate_synced_files(tmp_path, "test-host")

        # Verify checksums were recorded
        mock_manifest.record_checksums.assert_called_once()
        call_args = mock_manifest.record_checksums.call_args
        assert call_args[0][0] == "test-host"
        assert "test.db" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_validate_synced_files_multiple_dbs(self, tmp_path):
        """Should validate multiple database files."""
        from app.distributed.unified_data_sync import SyncConfig, UnifiedDataSyncService

        # Create multiple valid databases
        for i in range(3):
            db_path = tmp_path / f"db_{i}.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER)")
            conn.commit()
            conn.close()

        config = SyncConfig(checksum_validation=True)
        service = UnifiedDataSyncService.__new__(UnifiedDataSyncService)
        service.config = config
        service.manifest = MagicMock()

        result = await service._validate_synced_files(tmp_path, "test-host")

        assert result["valid"] is True
        assert result["files_validated"] == 3
        assert len(result["checksums"]) == 3


class TestSyncConfigChecksum:
    """Tests for checksum_validation config option."""

    def test_checksum_validation_default(self):
        """Should have checksum validation enabled by default."""
        from app.distributed.unified_data_sync import SyncConfig

        config = SyncConfig()
        assert config.checksum_validation is True

    def test_checksum_validation_disabled(self):
        """Should allow disabling checksum validation."""
        from app.distributed.unified_data_sync import SyncConfig

        config = SyncConfig(checksum_validation=False)
        assert config.checksum_validation is False

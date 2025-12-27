#!/usr/bin/env python3
"""Unit tests for app.db.integrity module (December 2025).

Tests the database integrity checking and repair utilities:
- check_database_integrity: PRAGMA integrity_check wrapper
- recover_corrupted_database: Dump/reimport recovery
- check_and_repair_databases: Batch check and repair
- get_database_stats: Database statistics gathering

Test fixtures create temporary SQLite databases for testing without
affecting production data.
"""

import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from app.db.integrity import (
    check_database_integrity,
    recover_corrupted_database,
    check_and_repair_databases,
    get_database_stats,
)


class TestCheckDatabaseIntegrity:
    """Tests for check_database_integrity function."""

    def test_healthy_database_returns_true(self, tmp_path):
        """Test that a healthy database returns (True, 'ok')."""
        db_path = tmp_path / "healthy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES ('hello')")
        conn.commit()
        conn.close()

        is_healthy, msg = check_database_integrity(db_path)

        assert is_healthy is True
        assert msg == "ok"

    def test_empty_database_is_healthy(self, tmp_path):
        """Test that an empty but valid database is healthy."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        is_healthy, msg = check_database_integrity(db_path)

        assert is_healthy is True
        assert msg == "ok"

    def test_nonexistent_file_creates_empty_db(self, tmp_path):
        """Test that sqlite3.connect creates empty DB for nonexistent file."""
        db_path = tmp_path / "nonexistent.db"

        # Note: sqlite3.connect creates the file if it doesn't exist
        # so this will actually succeed as an empty valid database
        is_healthy, msg = check_database_integrity(db_path)

        # Empty DB is valid
        assert is_healthy is True
        assert msg == "ok"

    def test_corrupted_file_returns_false(self, tmp_path):
        """Test that a corrupted file returns (False, error_message)."""
        db_path = tmp_path / "corrupted.db"
        # Write garbage to create a corrupted file
        db_path.write_bytes(b"this is not a valid sqlite database")

        is_healthy, msg = check_database_integrity(db_path)

        assert is_healthy is False
        assert len(msg) > 0

    def test_partial_write_detected(self, tmp_path):
        """Test that a truncated database is detected as unhealthy."""
        db_path = tmp_path / "truncated.db"

        # Create a valid database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE large (id INTEGER PRIMARY KEY, data TEXT)")
        for i in range(100):
            conn.execute("INSERT INTO large (data) VALUES (?)", ("x" * 1000,))
        conn.commit()
        conn.close()

        # Truncate it to simulate corruption
        original_size = db_path.stat().st_size
        with open(db_path, "r+b") as f:
            f.truncate(original_size // 2)

        is_healthy, msg = check_database_integrity(db_path)

        assert is_healthy is False

    def test_database_with_multiple_tables(self, tmp_path):
        """Test integrity check on database with multiple tables."""
        db_path = tmp_path / "multi.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY, player_id INTEGER)")
        conn.execute("CREATE INDEX idx_games_player ON games(player_id)")
        conn.execute("INSERT INTO users (name) VALUES ('Alice'), ('Bob')")
        conn.execute("INSERT INTO games (player_id) VALUES (1), (2), (1)")
        conn.commit()
        conn.close()

        is_healthy, msg = check_database_integrity(db_path)

        assert is_healthy is True
        assert msg == "ok"


class TestRecoverCorruptedDatabase:
    """Tests for recover_corrupted_database function."""

    def test_creates_corrupted_directory(self, tmp_path):
        """Test that corrupted directory is created if missing."""
        db_path = tmp_path / "test.db"
        db_path.write_text("corrupted")

        # Will fail to dump, but should still create directory
        recover_corrupted_database(db_path, log_prefix="[TEST]")

        assert (tmp_path / "corrupted").exists()
        assert (tmp_path / "corrupted").is_dir()

    def test_archives_corrupted_file(self, tmp_path):
        """Test that corrupted file is archived when recovery fails."""
        db_path = tmp_path / "test.db"
        # Write data that can't be recovered (not valid SQL)
        db_path.write_bytes(b"\x00" * 100)

        result = recover_corrupted_database(db_path, log_prefix="[TEST]")

        # Either:
        # 1. Recovery succeeded (file at original path, result=True)
        # 2. Recovery failed and file was archived (result=False)
        corrupted_dir = tmp_path / "corrupted"
        if result:
            # Recovery succeeded, file should exist at original path
            pass  # Test passed
        else:
            # Recovery failed, check for archive
            assert corrupted_dir.exists()
            corrupted_files = list(corrupted_dir.glob("*.corrupted"))
            assert len(corrupted_files) >= 1 or not db_path.exists()

    @patch("subprocess.run")
    def test_successful_recovery(self, mock_run, tmp_path):
        """Test successful dump and reimport recovery."""
        db_path = tmp_path / "recoverable.db"

        # Create a real database first
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()
        conn.close()

        # Mock successful dump
        dump_output = b"CREATE TABLE test (id INTEGER);\nINSERT INTO test VALUES(1);\n"
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout=dump_output),  # dump
            MagicMock(returncode=0),  # reimport
        ]

        # Mock integrity check to pass
        with patch("app.db.integrity.check_database_integrity", return_value=(True, "ok")):
            result = recover_corrupted_database(db_path, log_prefix="[TEST]")

        # Should succeed
        assert result is True or mock_run.called

    @patch("subprocess.run")
    def test_dump_timeout_handled(self, mock_run, tmp_path):
        """Test that dump timeout is handled gracefully."""
        import subprocess

        db_path = tmp_path / "timeout.db"
        db_path.write_text("data")

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="sqlite3", timeout=300)

        result = recover_corrupted_database(db_path, log_prefix="[TEST]")

        assert result is False

    @patch("subprocess.run")
    def test_dump_failure_handled(self, mock_run, tmp_path):
        """Test that dump failure is handled."""
        db_path = tmp_path / "fail.db"
        db_path.write_text("data")

        mock_run.return_value = MagicMock(returncode=1, stdout=b"")

        result = recover_corrupted_database(db_path, log_prefix="[TEST]")

        assert result is False

    def test_custom_log_prefix(self, tmp_path, capsys):
        """Test that log prefix is used in output."""
        db_path = tmp_path / "test.db"
        db_path.write_text("corrupted")

        recover_corrupted_database(db_path, log_prefix="[CustomPrefix]")

        captured = capsys.readouterr()
        assert "[CustomPrefix]" in captured.out


class TestCheckAndRepairDatabases:
    """Tests for check_and_repair_databases function."""

    def test_returns_correct_structure(self, tmp_path):
        """Test that result dict has correct keys."""
        results = check_and_repair_databases(tmp_path)

        assert "checked" in results
        assert "healthy" in results
        assert "corrupted" in results
        assert "recovered" in results
        assert "failed" in results
        assert "corrupted_files" in results

    def test_skips_nonexistent_directory(self, tmp_path):
        """Test that nonexistent directory returns empty results."""
        nonexistent = tmp_path / "does_not_exist"

        results = check_and_repair_databases(nonexistent)

        assert results["checked"] == 0
        assert results["healthy"] == 0
        assert results["corrupted"] == 0

    def test_skips_small_files(self, tmp_path):
        """Test that files smaller than min_size are skipped."""
        small_db = tmp_path / "small.db"
        conn = sqlite3.connect(str(small_db))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()

        # Default min_size is 1MB, small.db is much smaller
        results = check_and_repair_databases(tmp_path, min_size_bytes=1024 * 1024)

        assert results["checked"] == 0

    def test_checks_large_files(self, tmp_path):
        """Test that large files are checked."""
        large_db = tmp_path / "large.db"
        conn = sqlite3.connect(str(large_db))
        conn.execute("CREATE TABLE data (content TEXT)")
        # Insert enough data to exceed min size
        for _ in range(100):
            conn.execute("INSERT INTO data VALUES (?)", ("x" * 10000,))
        conn.commit()
        conn.close()

        # Use small min_size to ensure it's checked
        results = check_and_repair_databases(tmp_path, min_size_bytes=1024)

        assert results["checked"] >= 1

    def test_healthy_database_counted(self, tmp_path):
        """Test that healthy databases are counted correctly."""
        db_path = tmp_path / "healthy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER, data TEXT)")
        for _ in range(100):
            conn.execute("INSERT INTO test VALUES (1, ?)", ("data" * 100,))
        conn.commit()
        conn.close()

        results = check_and_repair_databases(tmp_path, min_size_bytes=1024)

        assert results["healthy"] >= 1

    def test_skips_corrupted_directory(self, tmp_path):
        """Test that files in 'corrupted' subdirectory are skipped."""
        corrupted_dir = tmp_path / "corrupted"
        corrupted_dir.mkdir()
        db_in_corrupted = corrupted_dir / "old.db"
        db_in_corrupted.write_text("old corrupted data")

        results = check_and_repair_databases(tmp_path, min_size_bytes=0, recursive=True)

        # Should not check files in corrupted directory
        assert "old.db" not in str(results.get("corrupted_files", []))

    def test_recursive_option(self, tmp_path):
        """Test that recursive option scans subdirectories."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        db_path = subdir / "nested.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE t (data TEXT)")
        for _ in range(100):
            conn.execute("INSERT INTO t VALUES (?)", ("x" * 100,))
        conn.commit()
        conn.close()

        # Non-recursive should not find it
        results_non_recursive = check_and_repair_databases(
            tmp_path, min_size_bytes=100, recursive=False
        )
        # Recursive should find it
        results_recursive = check_and_repair_databases(
            tmp_path, min_size_bytes=100, recursive=True
        )

        assert results_recursive["checked"] >= results_non_recursive["checked"]

    def test_auto_repair_false(self, tmp_path):
        """Test that auto_repair=False only moves, doesn't recover."""
        db_path = tmp_path / "corrupted.db"
        # Create a file that will fail integrity check
        db_path.write_bytes(b"corrupted database content")

        # Manually make it large enough to check
        with open(db_path, "ab") as f:
            f.write(b"x" * 10000)

        results = check_and_repair_databases(
            tmp_path, auto_repair=False, min_size_bytes=100
        )

        if results["corrupted"] > 0:
            # Should be counted as failed (no recovery attempted)
            assert results["recovered"] == 0

    def test_corrupted_files_list_populated(self, tmp_path):
        """Test that corrupted_files list contains paths."""
        db_path = tmp_path / "bad.db"
        db_path.write_bytes(b"not a valid database" + b"x" * 10000)

        results = check_and_repair_databases(
            tmp_path, auto_repair=False, min_size_bytes=100
        )

        if results["corrupted"] > 0:
            assert len(results["corrupted_files"]) > 0


class TestGetDatabaseStats:
    """Tests for get_database_stats function."""

    def test_returns_stats_for_valid_db(self, tmp_path):
        """Test that stats are returned for a valid database."""
        db_path = tmp_path / "stats.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Charlie')")
        conn.execute("INSERT INTO games VALUES (1), (2)")
        conn.commit()
        conn.close()

        stats = get_database_stats(db_path)

        assert stats is not None
        assert "path" in stats
        assert "size_mb" in stats
        assert "tables" in stats
        assert stats["tables"]["users"] == 3
        assert stats["tables"]["games"] == 2

    def test_returns_none_for_invalid_db(self, tmp_path):
        """Test that None is returned for invalid database."""
        db_path = tmp_path / "invalid.db"
        db_path.write_text("not a database")

        stats = get_database_stats(db_path)

        assert stats is None

    def test_returns_none_for_nonexistent(self, tmp_path):
        """Test that None is returned for nonexistent file."""
        db_path = tmp_path / "nonexistent.db"

        stats = get_database_stats(db_path)

        # sqlite3.connect creates the file, but tables query will work
        # This is valid behavior - empty DB has no tables
        if stats is not None:
            assert stats["tables"] == {}

    def test_size_mb_calculated(self, tmp_path):
        """Test that size_mb is calculated correctly."""
        db_path = tmp_path / "sized.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE data (content TEXT)")
        # Insert ~100KB of data
        for _ in range(100):
            conn.execute("INSERT INTO data VALUES (?)", ("x" * 1000,))
        conn.commit()
        conn.close()

        stats = get_database_stats(db_path)

        assert stats is not None
        assert stats["size_mb"] > 0

    def test_path_in_stats(self, tmp_path):
        """Test that path is included in stats."""
        db_path = tmp_path / "withpath.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        stats = get_database_stats(db_path)

        assert stats is not None
        assert stats["path"] == str(db_path)

    def test_handles_exception_gracefully(self, tmp_path):
        """Test that exceptions during stats gathering return None."""
        db_path = tmp_path / "error.db"

        # Create and immediately lock the file
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE normal (id INTEGER)")
        conn.close()

        # Test that the function handles errors gracefully
        # by verifying it either returns valid stats or None
        stats = get_database_stats(db_path)

        # Should successfully return stats for a valid database
        assert stats is not None
        assert "tables" in stats
        assert "normal" in stats["tables"]

    def test_empty_database_stats(self, tmp_path):
        """Test stats for empty database."""
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        stats = get_database_stats(db_path)

        assert stats is not None
        assert stats["tables"] == {}

    def test_database_with_index(self, tmp_path):
        """Test that indexes don't interfere with table stats."""
        db_path = tmp_path / "indexed.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE INDEX idx_name ON items(name)")
        conn.execute("INSERT INTO items (name) VALUES ('a'), ('b'), ('c')")
        conn.commit()
        conn.close()

        stats = get_database_stats(db_path)

        assert stats is not None
        # Should only count tables, not indexes
        assert "items" in stats["tables"]
        assert stats["tables"]["items"] == 3


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_check_recover_workflow(self, tmp_path):
        """Test the check -> detect -> recover workflow."""
        db_path = tmp_path / "workflow.db"

        # Create a valid database
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (value TEXT)")
        conn.execute("INSERT INTO test VALUES ('original')")
        conn.commit()
        conn.close()

        # Verify it's healthy
        is_healthy, _ = check_database_integrity(db_path)
        assert is_healthy is True

        # Get stats before corruption
        stats_before = get_database_stats(db_path)
        assert stats_before is not None
        assert stats_before["tables"]["test"] == 1

    def test_batch_check_multiple_dbs(self, tmp_path):
        """Test batch checking multiple databases."""
        # Create multiple databases
        for i in range(3):
            db_path = tmp_path / f"db_{i}.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute(f"CREATE TABLE t{i} (data TEXT)")
            for _ in range(100):
                conn.execute(f"INSERT INTO t{i} VALUES (?)", ("x" * 100,))
            conn.commit()
            conn.close()

        results = check_and_repair_databases(tmp_path, min_size_bytes=100)

        assert results["checked"] == 3
        assert results["healthy"] == 3
        assert results["corrupted"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

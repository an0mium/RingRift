"""Tests for async SQLite utilities.

Tests the async_sqlite_* functions in app/utils/async_utils.py.
"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.utils.async_utils import (
    SqliteError,
    SqliteResult,
    async_sqlite_execute,
    async_sqlite_executemany,
    async_sqlite_fetchall,
    async_sqlite_fetchone,
    async_sqlite_scalar,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create test table
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("""
            CREATE TABLE test_games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                board_type TEXT NOT NULL,
                num_players INTEGER NOT NULL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


class TestSqliteResult:
    """Tests for SqliteResult dataclass."""

    def test_first_with_rows(self):
        """Test first property with rows."""
        result = SqliteResult(rows=[("hex8", 2), ("square8", 4)], rowcount=2, lastrowid=None)
        assert result.first == ("hex8", 2)

    def test_first_empty(self):
        """Test first property with no rows."""
        result = SqliteResult(rows=[], rowcount=0, lastrowid=None)
        assert result.first is None

    def test_scalar_with_value(self):
        """Test scalar property with value."""
        result = SqliteResult(rows=[(42,)], rowcount=1, lastrowid=None)
        assert result.scalar == 42

    def test_scalar_empty(self):
        """Test scalar property with no rows."""
        result = SqliteResult(rows=[], rowcount=0, lastrowid=None)
        assert result.scalar is None

    def test_scalar_empty_tuple(self):
        """Test scalar property with empty tuple."""
        result = SqliteResult(rows=[()], rowcount=1, lastrowid=None)
        assert result.scalar is None


class TestAsyncSqliteExecute:
    """Tests for async_sqlite_execute function."""

    @pytest.mark.asyncio
    async def test_insert(self, temp_db):
        """Test INSERT query."""
        result = await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )
        assert result.rowcount == 1
        assert result.lastrowid is not None
        assert result.lastrowid > 0

    @pytest.mark.asyncio
    async def test_select(self, temp_db):
        """Test SELECT query."""
        # Insert test data
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )

        # Select it back
        result = await async_sqlite_execute(
            temp_db,
            "SELECT board_type, num_players FROM test_games WHERE board_type = ?",
            ("hex8",),
        )
        assert len(result.rows) == 1
        assert result.rows[0] == ("hex8", 2)

    @pytest.mark.asyncio
    async def test_update(self, temp_db):
        """Test UPDATE query."""
        # Insert test data
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )

        # Update it
        result = await async_sqlite_execute(
            temp_db,
            "UPDATE test_games SET status = ? WHERE board_type = ?",
            ("completed", "hex8"),
        )
        assert result.rowcount == 1

    @pytest.mark.asyncio
    async def test_delete(self, temp_db):
        """Test DELETE query."""
        # Insert test data
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )

        # Delete it
        result = await async_sqlite_execute(
            temp_db,
            "DELETE FROM test_games WHERE board_type = ?",
            ("hex8",),
        )
        assert result.rowcount == 1

    @pytest.mark.asyncio
    async def test_with_row_factory(self, temp_db):
        """Test with sqlite3.Row factory."""
        # Insert test data
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )

        # Select with Row factory
        result = await async_sqlite_execute(
            temp_db,
            "SELECT * FROM test_games LIMIT 1",
            row_factory=sqlite3.Row,
        )
        assert len(result.rows) == 1
        row = result.rows[0]
        assert row["board_type"] == "hex8"
        assert row["num_players"] == 2

    @pytest.mark.asyncio
    async def test_with_named_params(self, temp_db):
        """Test with named parameters."""
        result = await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (:board, :players)",
            {"board": "square8", "players": 4},
        )
        assert result.rowcount == 1

    @pytest.mark.asyncio
    async def test_no_params(self, temp_db):
        """Test query without parameters."""
        result = await async_sqlite_execute(
            temp_db,
            "SELECT COUNT(*) FROM test_games",
        )
        assert result.scalar == 0

    @pytest.mark.asyncio
    async def test_invalid_query(self, temp_db):
        """Test with invalid SQL."""
        with pytest.raises(SqliteError):
            await async_sqlite_execute(
                temp_db,
                "SELECT * FROM nonexistent_table",
            )


class TestAsyncSqliteExecutemany:
    """Tests for async_sqlite_executemany function."""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, temp_db):
        """Test bulk INSERT."""
        result = await async_sqlite_executemany(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            [("hex8", 2), ("square8", 4), ("hexagonal", 3)],
        )
        assert result.rowcount == 3

    @pytest.mark.asyncio
    async def test_empty_params(self, temp_db):
        """Test with empty params sequence."""
        result = await async_sqlite_executemany(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            [],
        )
        assert result.rowcount == 0


class TestAsyncSqliteFetchone:
    """Tests for async_sqlite_fetchone function."""

    @pytest.mark.asyncio
    async def test_with_result(self, temp_db):
        """Test fetchone with result."""
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            ("hex8", 2),
        )

        row = await async_sqlite_fetchone(
            temp_db,
            "SELECT board_type, num_players FROM test_games LIMIT 1",
        )
        assert row == ("hex8", 2)

    @pytest.mark.asyncio
    async def test_no_result(self, temp_db):
        """Test fetchone with no result."""
        row = await async_sqlite_fetchone(
            temp_db,
            "SELECT * FROM test_games WHERE board_type = ?",
            ("nonexistent",),
        )
        assert row is None


class TestAsyncSqliteFetchall:
    """Tests for async_sqlite_fetchall function."""

    @pytest.mark.asyncio
    async def test_with_results(self, temp_db):
        """Test fetchall with results."""
        await async_sqlite_executemany(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            [("hex8", 2), ("square8", 4)],
        )

        rows = await async_sqlite_fetchall(
            temp_db,
            "SELECT board_type, num_players FROM test_games ORDER BY board_type",
        )
        assert len(rows) == 2
        assert rows[0] == ("hex8", 2)
        assert rows[1] == ("square8", 4)

    @pytest.mark.asyncio
    async def test_empty_result(self, temp_db):
        """Test fetchall with no results."""
        rows = await async_sqlite_fetchall(
            temp_db,
            "SELECT * FROM test_games",
        )
        assert rows == []


class TestAsyncSqliteScalar:
    """Tests for async_sqlite_scalar function."""

    @pytest.mark.asyncio
    async def test_count(self, temp_db):
        """Test scalar with COUNT."""
        await async_sqlite_executemany(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            [("hex8", 2), ("square8", 4), ("hexagonal", 3)],
        )

        count = await async_sqlite_scalar(
            temp_db,
            "SELECT COUNT(*) FROM test_games",
        )
        assert count == 3

    @pytest.mark.asyncio
    async def test_with_default(self, temp_db):
        """Test scalar with default value."""
        value = await async_sqlite_scalar(
            temp_db,
            "SELECT board_type FROM test_games WHERE id = ?",
            (999,),
            default="unknown",
        )
        assert value == "unknown"

    @pytest.mark.asyncio
    async def test_null_value(self, temp_db):
        """Test scalar with NULL in database."""
        await async_sqlite_execute(
            temp_db,
            "INSERT INTO test_games (board_type, num_players, status) VALUES (?, ?, NULL)",
            ("hex8", 2),
        )

        # This returns the first column which is id, not NULL status
        value = await async_sqlite_scalar(
            temp_db,
            "SELECT status FROM test_games LIMIT 1",
            default="default_status",
        )
        # NULL should trigger default
        assert value == "default_status"


class TestConcurrency:
    """Tests for concurrent SQLite access."""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, temp_db):
        """Test concurrent read operations."""
        # Insert test data
        await async_sqlite_executemany(
            temp_db,
            "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
            [("hex8", 2), ("square8", 4), ("hexagonal", 3)],
        )

        # Run multiple concurrent reads
        tasks = [
            async_sqlite_scalar(temp_db, "SELECT COUNT(*) FROM test_games")
            for _ in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # All should return the same count
        assert all(r == 3 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_db):
        """Test concurrent write operations."""
        # Run multiple concurrent inserts
        tasks = [
            async_sqlite_execute(
                temp_db,
                "INSERT INTO test_games (board_type, num_players) VALUES (?, ?)",
                (f"board_{i}", i),
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.rowcount == 1 for r in results)

        # Verify total count
        count = await async_sqlite_scalar(
            temp_db,
            "SELECT COUNT(*) FROM test_games",
        )
        assert count == 10

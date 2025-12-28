"""Tests for unified_manifest.py - Unified Data Manifest for sync tracking.

Tests cover:
- DataManifest initialization and schema migration
- Game deduplication (by ID and content hash)
- Quality metadata tracking and scoring
- Priority sync queue management
- Host sync state persistence
- Sync history logging
- Dead letter queue for failed syncs
- Manifest statistics and cleanup

This module tests the DataManifest class which consolidates functionality
from multiple legacy manifest implementations.
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.distributed.unified_manifest import (
    DataManifest,
    DeadLetterEntry,
    GameQualityMetadata,
    HostSyncState,
    ManifestStats,
    PriorityQueueEntry,
    SyncHistoryEntry,
    create_manifest,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_manifest.db"


@pytest.fixture
def manifest(temp_db_path: Path) -> DataManifest:
    """Create a DataManifest with a temporary database."""
    m = DataManifest(db_path=temp_db_path)
    yield m
    m.close()


@pytest.fixture
def populated_manifest(manifest: DataManifest) -> DataManifest:
    """Create a manifest with some sample data."""
    # Add some games
    manifest.mark_games_synced(
        game_ids=["game-001", "game-002", "game-003"],
        source_host="node-a",
        source_db="/data/selfplay.db",
        board_type="hex8",
        num_players=2,
    )
    manifest.mark_games_synced(
        game_ids=["game-004", "game-005"],
        source_host="node-b",
        source_db="/data/selfplay.db",
        board_type="square8",
        num_players=4,
    )
    return manifest


# =============================================================================
# DataManifest Initialization Tests
# =============================================================================


class TestDataManifestInit:
    """Tests for DataManifest initialization and schema."""

    def test_init_creates_database(self, temp_db_path: Path) -> None:
        """Test that initialization creates the database file."""
        manifest = DataManifest(db_path=temp_db_path)
        assert temp_db_path.exists()
        manifest.close()

    def test_init_creates_tables(self, temp_db_path: Path) -> None:
        """Test that initialization creates all required tables."""
        manifest = DataManifest(db_path=temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "synced_games" in tables
        assert "sync_priority_queue" in tables
        assert "host_states" in tables
        assert "sync_history" in tables
        assert "dead_letter_queue" in tables
        assert "manifest_metadata" in tables

        manifest.close()

    def test_init_creates_indexes(self, temp_db_path: Path) -> None:
        """Test that initialization creates performance indexes."""
        manifest = DataManifest(db_path=temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "idx_synced_games_host" in indexes
        assert "idx_synced_games_time" in indexes
        assert "idx_synced_games_quality" in indexes

        manifest.close()

    def test_schema_version_stored(self, temp_db_path: Path) -> None:
        """Test that schema version is stored in metadata."""
        manifest = DataManifest(db_path=temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM manifest_metadata WHERE key='schema_version'"
        )
        version = cursor.fetchone()
        conn.close()

        assert version is not None
        assert version[0] == DataManifest.SCHEMA_VERSION

        manifest.close()

    def test_init_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that initialization creates parent directories if needed."""
        nested_path = tmp_path / "subdir" / "nested" / "manifest.db"
        manifest = DataManifest(db_path=nested_path)
        assert nested_path.exists()
        manifest.close()

    def test_factory_function(self, tmp_path: Path) -> None:
        """Test create_manifest factory function."""
        manifest = create_manifest(tmp_path)
        assert manifest.db_path == tmp_path / "data_manifest.db"
        manifest.close()


# =============================================================================
# Game Deduplication Tests
# =============================================================================


class TestGameDeduplication:
    """Tests for game ID and content hash deduplication."""

    def test_is_game_synced_returns_false_for_new_game(
        self, manifest: DataManifest
    ) -> None:
        """Test is_game_synced returns False for unsynced game."""
        assert manifest.is_game_synced("nonexistent-game") is False

    def test_is_game_synced_returns_true_after_sync(
        self, manifest: DataManifest
    ) -> None:
        """Test is_game_synced returns True after marking synced."""
        manifest.mark_games_synced(
            ["game-001"], "node-a", "/data/db.db"
        )
        assert manifest.is_game_synced("game-001") is True

    def test_mark_games_synced_returns_count(
        self, manifest: DataManifest
    ) -> None:
        """Test mark_games_synced returns number of new games marked."""
        count = manifest.mark_games_synced(
            ["game-001", "game-002", "game-003"],
            "node-a",
            "/data/db.db",
        )
        assert count == 3

    def test_mark_games_synced_ignores_duplicates(
        self, manifest: DataManifest
    ) -> None:
        """Test mark_games_synced skips already-synced games."""
        manifest.mark_games_synced(["game-001"], "node-a", "/data/db.db")
        # Try to mark same game again
        count = manifest.mark_games_synced(["game-001"], "node-b", "/other.db")
        assert count == 0

    def test_mark_games_synced_with_content_hash(
        self, manifest: DataManifest
    ) -> None:
        """Test mark_games_synced stores content hashes."""
        manifest.mark_games_synced(
            game_ids=["game-001"],
            source_host="node-a",
            source_db="/data/db.db",
            content_hashes=["abc123hash"],
        )
        assert manifest.is_content_synced("abc123hash") is True

    def test_is_content_synced_false_for_unknown_hash(
        self, manifest: DataManifest
    ) -> None:
        """Test is_content_synced returns False for unknown hash."""
        assert manifest.is_content_synced("unknown-hash") is False

    def test_get_unsynced_game_ids(
        self, manifest: DataManifest
    ) -> None:
        """Test filtering to only unsynced game IDs."""
        manifest.mark_games_synced(
            ["game-001", "game-002"], "node-a", "/data/db.db"
        )

        unsynced = manifest.get_unsynced_game_ids(
            ["game-001", "game-002", "game-003", "game-004"]
        )

        assert set(unsynced) == {"game-003", "game-004"}

    def test_get_unsynced_game_ids_empty_input(
        self, manifest: DataManifest
    ) -> None:
        """Test get_unsynced_game_ids with empty input."""
        result = manifest.get_unsynced_game_ids([])
        assert result == []

    def test_mark_games_synced_empty_list(
        self, manifest: DataManifest
    ) -> None:
        """Test mark_games_synced with empty list."""
        count = manifest.mark_games_synced([], "node-a", "/data/db.db")
        assert count == 0


# =============================================================================
# Quality Metadata Tests
# =============================================================================


class TestGameQualityMetadata:
    """Tests for quality score computation and tracking."""

    def test_compute_quality_score_default_weights(self) -> None:
        """Test quality score computation with default weights."""
        score = GameQualityMetadata.compute_quality_score(
            avg_elo=1800,
            game_length=100,
            is_decisive=True,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be good quality

    def test_compute_quality_score_low_elo(self) -> None:
        """Test quality score with low Elo is lower."""
        high_elo_score = GameQualityMetadata.compute_quality_score(
            avg_elo=2200, game_length=100, is_decisive=True
        )
        low_elo_score = GameQualityMetadata.compute_quality_score(
            avg_elo=1200, game_length=100, is_decisive=True
        )
        assert high_elo_score > low_elo_score

    def test_compute_quality_score_decisive_vs_draw(self) -> None:
        """Test decisive games score higher than draws."""
        decisive_score = GameQualityMetadata.compute_quality_score(
            avg_elo=1500, game_length=100, is_decisive=True
        )
        draw_score = GameQualityMetadata.compute_quality_score(
            avg_elo=1500, game_length=100, is_decisive=False
        )
        assert decisive_score > draw_score

    def test_compute_quality_score_short_game(self) -> None:
        """Test short games score lower."""
        long_game_score = GameQualityMetadata.compute_quality_score(
            avg_elo=1500, game_length=150, is_decisive=True
        )
        short_game_score = GameQualityMetadata.compute_quality_score(
            avg_elo=1500, game_length=5, is_decisive=True
        )
        assert long_game_score > short_game_score

    def test_compute_quality_score_clamped_to_range(self) -> None:
        """Test quality score is always in [0, 1]."""
        # Extreme values
        score = GameQualityMetadata.compute_quality_score(
            avg_elo=5000, game_length=1000, is_decisive=True
        )
        assert score <= 1.0

        score = GameQualityMetadata.compute_quality_score(
            avg_elo=0, game_length=0, is_decisive=False
        )
        assert score >= 0.0

    def test_mark_games_synced_with_quality(
        self, manifest: DataManifest
    ) -> None:
        """Test marking games with quality metadata."""
        games = [
            GameQualityMetadata(
                game_id="game-q1",
                avg_player_elo=1800,
                game_length=100,
                is_decisive=True,
                quality_score=0.85,
            ),
            GameQualityMetadata(
                game_id="game-q2",
                avg_player_elo=1600,
                game_length=50,
                is_decisive=False,
                quality_score=0.5,
            ),
        ]

        count = manifest.mark_games_synced_with_quality(
            games, "node-a", "/data/db.db", "hex8", 2
        )

        assert count == 2
        assert manifest.is_game_synced("game-q1") is True
        assert manifest.is_game_synced("game-q2") is True

    def test_update_game_quality(self, manifest: DataManifest) -> None:
        """Test updating quality metadata for existing game."""
        manifest.mark_games_synced(["game-001"], "node-a", "/data/db.db")

        quality = GameQualityMetadata(
            game_id="game-001",
            avg_player_elo=1900,
            game_length=120,
            quality_score=0.9,
        )
        result = manifest.update_game_quality("game-001", quality)
        assert result is True

    def test_get_high_quality_games(self, manifest: DataManifest) -> None:
        """Test retrieving high quality games."""
        games = [
            GameQualityMetadata(
                game_id=f"game-{i}",
                avg_player_elo=1500 + i * 100,
                game_length=50 + i * 10,
                quality_score=0.3 + i * 0.15,
            )
            for i in range(5)
        ]
        manifest.mark_games_synced_with_quality(
            games, "node-a", "/data/db.db", "hex8", 2
        )

        high_quality = manifest.get_high_quality_games(
            min_quality_score=0.6, limit=10
        )
        assert len(high_quality) >= 2
        # Should be sorted by quality score descending
        if len(high_quality) >= 2:
            assert high_quality[0].quality_score >= high_quality[1].quality_score

    def test_get_quality_distribution(self, manifest: DataManifest) -> None:
        """Test getting quality score distribution stats."""
        games = [
            GameQualityMetadata(
                game_id=f"game-{i}",
                avg_player_elo=1600,
                game_length=100,
                is_decisive=i % 2 == 0,
                quality_score=0.5 + i * 0.1,
            )
            for i in range(5)
        ]
        manifest.mark_games_synced_with_quality(
            games, "node-a", "/data/db.db", "hex8", 2
        )

        dist = manifest.get_quality_distribution(board_type="hex8", num_players=2)
        assert dist["total_games"] == 5
        assert dist["avg_quality_score"] > 0
        assert dist["decisive_count"] >= 0


# =============================================================================
# Priority Sync Queue Tests
# =============================================================================


class TestPrioritySyncQueue:
    """Tests for priority-based sync queue management."""

    def test_add_to_priority_queue(self, manifest: DataManifest) -> None:
        """Test adding game to priority queue."""
        entry_id = manifest.add_to_priority_queue(
            game_id="game-p1",
            source_host="node-a",
            source_db="/data/db.db",
            priority_score=0.9,
            avg_player_elo=1800,
            game_length=100,
            is_decisive=True,
        )
        assert entry_id > 0

    def test_get_priority_queue_batch(self, manifest: DataManifest) -> None:
        """Test retrieving batch from priority queue."""
        # Add entries with different priorities
        for i in range(5):
            manifest.add_to_priority_queue(
                game_id=f"game-p{i}",
                source_host="node-a",
                source_db="/data/db.db",
                priority_score=0.5 + i * 0.1,
            )

        batch = manifest.get_priority_queue_batch(limit=3)
        assert len(batch) == 3
        # Should be sorted by priority descending
        assert batch[0].priority_score >= batch[1].priority_score
        assert batch[1].priority_score >= batch[2].priority_score

    def test_get_priority_queue_batch_with_host_filter(
        self, manifest: DataManifest
    ) -> None:
        """Test filtering priority queue by source host."""
        manifest.add_to_priority_queue(
            "game-1", "node-a", "/data/db.db", priority_score=0.8
        )
        manifest.add_to_priority_queue(
            "game-2", "node-b", "/data/db.db", priority_score=0.9
        )

        batch = manifest.get_priority_queue_batch(limit=10, source_host="node-a")
        assert len(batch) == 1
        assert batch[0].source_host == "node-a"

    def test_mark_queue_entries_synced(self, manifest: DataManifest) -> None:
        """Test marking queue entries as synced."""
        entry_id = manifest.add_to_priority_queue(
            "game-1", "node-a", "/data/db.db", priority_score=0.8
        )

        count = manifest.mark_queue_entries_synced([entry_id])
        assert count == 1

        # Should no longer appear in unsynced batch
        batch = manifest.get_priority_queue_batch(limit=10)
        game_ids = {e.game_id for e in batch}
        assert "game-1" not in game_ids

    def test_get_priority_queue_stats(self, manifest: DataManifest) -> None:
        """Test getting priority queue statistics."""
        manifest.add_to_priority_queue(
            "game-1", "node-a", "/data/db.db", priority_score=0.8
        )
        manifest.add_to_priority_queue(
            "game-2", "node-a", "/data/db.db", priority_score=0.9
        )

        stats = manifest.get_priority_queue_stats()
        assert stats["total_entries"] == 2
        assert stats["pending"] == 2
        assert stats["synced"] == 0
        assert "node-a" in stats["by_host"]

    def test_cleanup_old_queue_entries(self, manifest: DataManifest) -> None:
        """Test cleaning up old synced queue entries."""
        entry_id = manifest.add_to_priority_queue(
            "game-old", "node-a", "/data/db.db", priority_score=0.5
        )
        manifest.mark_queue_entries_synced([entry_id])

        # Directly update synced_at to old timestamp
        conn = sqlite3.connect(manifest.db_path)
        old_time = time.time() - 10 * 86400  # 10 days ago
        conn.execute(
            "UPDATE sync_priority_queue SET synced_at = ? WHERE id = ?",
            (old_time, entry_id),
        )
        conn.commit()
        conn.close()

        removed = manifest.cleanup_old_queue_entries(days=7)
        assert removed == 1


# =============================================================================
# Host Sync State Tests
# =============================================================================


class TestHostSyncState:
    """Tests for host sync state persistence."""

    def test_host_sync_state_dataclass(self) -> None:
        """Test HostSyncState dataclass creation."""
        state = HostSyncState(
            name="node-a",
            last_sync_time=1000.0,
            total_games_synced=500,
            is_ephemeral=True,
        )
        assert state.name == "node-a"
        assert state.total_games_synced == 500
        assert state.is_ephemeral is True

    def test_host_sync_state_to_dict(self) -> None:
        """Test HostSyncState serialization to dict."""
        state = HostSyncState(
            name="node-a",
            last_sync_time=1234.5,
            consecutive_failures=2,
        )
        data = state.to_dict()
        assert data["name"] == "node-a"
        assert data["last_sync_time"] == 1234.5
        assert data["consecutive_failures"] == 2

    def test_host_sync_state_from_dict(self) -> None:
        """Test HostSyncState deserialization from dict."""
        data = {
            "name": "node-b",
            "total_games_synced": 1000,
            "is_ephemeral": True,
            "storage_type": "ephemeral",
        }
        state = HostSyncState.from_dict(data)
        assert state.name == "node-b"
        assert state.total_games_synced == 1000
        assert state.is_ephemeral is True

    def test_save_host_state(self, manifest: DataManifest) -> None:
        """Test saving host sync state."""
        state = HostSyncState(
            name="node-a",
            last_sync_time=time.time(),
            total_games_synced=100,
        )
        manifest.save_host_state(state)

        # Verify saved
        loaded = manifest.load_host_state("node-a")
        assert loaded is not None
        assert loaded.name == "node-a"
        assert loaded.total_games_synced == 100

    def test_load_host_state_not_found(self, manifest: DataManifest) -> None:
        """Test loading non-existent host state."""
        loaded = manifest.load_host_state("nonexistent-host")
        assert loaded is None

    def test_load_all_host_states(self, manifest: DataManifest) -> None:
        """Test loading all host states."""
        manifest.save_host_state(HostSyncState(name="node-a"))
        manifest.save_host_state(HostSyncState(name="node-b"))
        manifest.save_host_state(HostSyncState(name="node-c"))

        states = manifest.load_all_host_states()
        names = {s.name for s in states}
        assert names == {"node-a", "node-b", "node-c"}

    def test_get_ephemeral_hosts(self, manifest: DataManifest) -> None:
        """Test getting list of ephemeral hosts."""
        manifest.save_host_state(
            HostSyncState(name="node-a", is_ephemeral=True)
        )
        manifest.save_host_state(
            HostSyncState(name="node-b", is_ephemeral=False)
        )
        manifest.save_host_state(
            HostSyncState(name="node-c", is_ephemeral=True)
        )

        ephemeral = manifest.get_ephemeral_hosts()
        assert set(ephemeral) == {"node-a", "node-c"}

    def test_update_host_state_overwrites(self, manifest: DataManifest) -> None:
        """Test saving host state overwrites existing."""
        manifest.save_host_state(
            HostSyncState(name="node-a", total_games_synced=100)
        )
        manifest.save_host_state(
            HostSyncState(name="node-a", total_games_synced=200)
        )

        loaded = manifest.load_host_state("node-a")
        assert loaded is not None
        assert loaded.total_games_synced == 200


# =============================================================================
# Sync History Tests
# =============================================================================


class TestSyncHistory:
    """Tests for sync history logging."""

    def test_log_sync_success(self, manifest: DataManifest) -> None:
        """Test logging successful sync."""
        log_id = manifest.log_sync(
            host_name="node-a",
            games_synced=50,
            duration_seconds=10.5,
            success=True,
            sync_method="rsync",
        )
        assert log_id > 0

    def test_log_sync_failure(self, manifest: DataManifest) -> None:
        """Test logging failed sync."""
        log_id = manifest.log_sync(
            host_name="node-a",
            games_synced=0,
            duration_seconds=5.0,
            success=False,
            error_message="Connection refused",
        )
        assert log_id > 0

    def test_record_sync_alias(self, manifest: DataManifest) -> None:
        """Test record_sync alias for backward compatibility."""
        log_id = manifest.record_sync(
            host_name="node-a",
            games_synced=25,
            duration_seconds=3.0,
            success=True,
        )
        assert log_id > 0

    def test_get_recent_syncs(self, manifest: DataManifest) -> None:
        """Test retrieving recent sync history."""
        manifest.log_sync("node-a", 10, 1.0, True)
        manifest.log_sync("node-b", 20, 2.0, True)
        manifest.log_sync("node-a", 5, 0.5, False, error_message="Error")

        recent = manifest.get_recent_syncs(hours=24)
        assert len(recent) == 3

    def test_get_recent_syncs_by_host(self, manifest: DataManifest) -> None:
        """Test filtering sync history by host."""
        manifest.log_sync("node-a", 10, 1.0, True)
        manifest.log_sync("node-b", 20, 2.0, True)
        manifest.log_sync("node-a", 5, 0.5, True)

        node_a_syncs = manifest.get_recent_syncs(hours=24, host_name="node-a")
        assert len(node_a_syncs) == 2
        assert all(s.host_name == "node-a" for s in node_a_syncs)

    def test_sync_history_entry_fields(self, manifest: DataManifest) -> None:
        """Test SyncHistoryEntry fields."""
        manifest.log_sync(
            host_name="node-a",
            games_synced=100,
            duration_seconds=15.5,
            success=True,
            sync_method="rsync",
        )

        recent = manifest.get_recent_syncs(hours=24)
        entry = recent[0]
        assert entry.host_name == "node-a"
        assert entry.games_synced == 100
        assert entry.duration_seconds == 15.5
        assert entry.success is True
        assert entry.sync_method == "rsync"


# =============================================================================
# Dead Letter Queue Tests
# =============================================================================


class TestDeadLetterQueue:
    """Tests for dead letter queue (failed sync tracking)."""

    def test_add_to_dead_letter(self, manifest: DataManifest) -> None:
        """Test adding failed sync to dead letter queue."""
        entry_id = manifest.add_to_dead_letter(
            game_id="game-fail-1",
            source_host="node-a",
            source_db="/data/db.db",
            error_message="Parse error",
            error_type="data_error",
        )
        assert entry_id > 0

    def test_get_dead_letter_entries(self, manifest: DataManifest) -> None:
        """Test retrieving dead letter entries."""
        manifest.add_to_dead_letter(
            "game-1", "node-a", "/data/db.db", "Error 1", "type_a"
        )
        manifest.add_to_dead_letter(
            "game-2", "node-b", "/data/db.db", "Error 2", "type_b"
        )

        entries = manifest.get_dead_letter_entries(limit=10)
        assert len(entries) == 2

    def test_dead_letter_excludes_resolved(self, manifest: DataManifest) -> None:
        """Test dead letter entries exclude resolved by default."""
        id1 = manifest.add_to_dead_letter(
            "game-1", "node-a", "/data/db.db", "Error", "error"
        )
        manifest.add_to_dead_letter(
            "game-2", "node-a", "/data/db.db", "Error", "error"
        )

        # Resolve first entry
        manifest.mark_dead_letter_resolved([id1])

        entries = manifest.get_dead_letter_entries(limit=10)
        assert len(entries) == 1
        assert entries[0].game_id == "game-2"

    def test_get_dead_letter_include_resolved(
        self, manifest: DataManifest
    ) -> None:
        """Test getting dead letter entries including resolved."""
        id1 = manifest.add_to_dead_letter(
            "game-1", "node-a", "/data/db.db", "Error", "error"
        )
        manifest.add_to_dead_letter(
            "game-2", "node-a", "/data/db.db", "Error", "error"
        )
        manifest.mark_dead_letter_resolved([id1])

        entries = manifest.get_dead_letter_entries(
            limit=10, include_resolved=True
        )
        assert len(entries) == 2

    def test_increment_dead_letter_retry(self, manifest: DataManifest) -> None:
        """Test incrementing retry count."""
        entry_id = manifest.add_to_dead_letter(
            "game-1", "node-a", "/data/db.db", "Error", "error"
        )

        manifest.increment_dead_letter_retry(entry_id)
        manifest.increment_dead_letter_retry(entry_id)

        entries = manifest.get_dead_letter_entries(limit=10)
        assert entries[0].retry_count == 2
        assert entries[0].last_retry_at is not None

    def test_cleanup_old_dead_letters(self, manifest: DataManifest) -> None:
        """Test cleaning up old resolved dead letters."""
        entry_id = manifest.add_to_dead_letter(
            "game-old", "node-a", "/data/db.db", "Error", "error"
        )
        manifest.mark_dead_letter_resolved([entry_id])

        # Set old timestamp
        conn = sqlite3.connect(manifest.db_path)
        old_time = time.time() - 10 * 86400
        conn.execute(
            "UPDATE dead_letter_queue SET added_at = ? WHERE id = ?",
            (old_time, entry_id),
        )
        conn.commit()
        conn.close()

        removed = manifest.cleanup_old_dead_letters(days=7)
        assert removed == 1


# =============================================================================
# Statistics Tests
# =============================================================================


class TestManifestStatistics:
    """Tests for manifest statistics."""

    def test_get_synced_count(self, manifest: DataManifest) -> None:
        """Test getting total synced game count."""
        manifest.mark_games_synced(
            ["game-1", "game-2", "game-3"], "node-a", "/data/db.db"
        )
        count = manifest.get_synced_count()
        assert count == 3

    def test_get_synced_count_by_host(
        self, populated_manifest: DataManifest
    ) -> None:
        """Test getting synced count broken down by host."""
        counts = populated_manifest.get_synced_count_by_host()
        assert counts["node-a"] == 3
        assert counts["node-b"] == 2

    def test_get_synced_count_by_config(
        self, populated_manifest: DataManifest
    ) -> None:
        """Test getting synced count by board configuration."""
        counts = populated_manifest.get_synced_count_by_config()
        assert counts["hex8_2p"] == 3
        assert counts["square8_4p"] == 2

    def test_get_stats(self, populated_manifest: DataManifest) -> None:
        """Test getting full manifest statistics."""
        stats = populated_manifest.get_stats()

        assert stats.total_games == 5
        assert len(stats.games_by_host) == 2
        assert len(stats.games_by_board_type) == 2

    def test_manifest_stats_dataclass(self) -> None:
        """Test ManifestStats dataclass fields."""
        stats = ManifestStats(
            total_games=100,
            games_by_host={"node-a": 60, "node-b": 40},
            games_by_board_type={"hex8_2p": 50, "square8_4p": 50},
            recent_sync_count=25,
            dead_letter_count=3,
        )
        assert stats.total_games == 100
        assert stats.dead_letter_count == 3

    def test_cleanup_old_history(self, manifest: DataManifest) -> None:
        """Test cleaning up old sync history."""
        # Add old sync entries
        manifest.log_sync("node-a", 10, 1.0, True)

        # Set old timestamp
        conn = sqlite3.connect(manifest.db_path)
        old_time = time.time() - 45 * 86400  # 45 days ago
        conn.execute(
            "UPDATE sync_history SET sync_time = ?",
            (old_time,),
        )
        conn.commit()
        conn.close()

        removed = manifest.cleanup_old_history(days=30)
        assert removed == 1


# =============================================================================
# Connection Management Tests
# =============================================================================


class TestConnectionManagement:
    """Tests for database connection management."""

    def test_connection_persistence(self, manifest: DataManifest) -> None:
        """Test that connection is reused."""
        # Multiple operations should use same connection
        manifest.is_game_synced("game-1")
        manifest.is_game_synced("game-2")
        manifest.get_synced_count()

        # Connection should be established
        assert manifest._conn is not None

    def test_close_releases_connection(self, temp_db_path: Path) -> None:
        """Test close() releases the connection."""
        manifest = DataManifest(db_path=temp_db_path)
        manifest.is_game_synced("test")  # Force connection
        assert manifest._conn is not None

        manifest.close()
        assert manifest._conn is None

    def test_connection_thread_safety(self, manifest: DataManifest) -> None:
        """Test concurrent access from multiple threads."""
        results = []
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(20):
                    manifest.mark_games_synced(
                        [f"game-t{thread_id}-{i}"],
                        f"node-{thread_id}",
                        "/data/db.db",
                    )
                results.append(thread_id)
            except Exception as e:
                errors.append(str(e))

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert len(errors) == 0
        assert manifest.get_synced_count() == 100  # 5 threads * 20 games


# =============================================================================
# Data Classes Tests
# =============================================================================


class TestDataClasses:
    """Tests for data class structures."""

    def test_sync_history_entry_fields(self) -> None:
        """Test SyncHistoryEntry dataclass."""
        entry = SyncHistoryEntry(
            id=1,
            host_name="node-a",
            sync_time=1234567890.0,
            games_synced=100,
            duration_seconds=15.5,
            success=True,
            sync_method="rsync",
            error_message="",
        )
        assert entry.id == 1
        assert entry.success is True

    def test_dead_letter_entry_fields(self) -> None:
        """Test DeadLetterEntry dataclass."""
        entry = DeadLetterEntry(
            id=1,
            game_id="game-001",
            source_host="node-a",
            source_db="/data/db.db",
            error_message="Parse error",
            error_type="data_error",
            added_at=time.time(),
            retry_count=3,
            resolved=False,
        )
        assert entry.game_id == "game-001"
        assert entry.retry_count == 3
        assert entry.resolved is False

    def test_priority_queue_entry_fields(self) -> None:
        """Test PriorityQueueEntry dataclass."""
        entry = PriorityQueueEntry(
            id=1,
            game_id="game-001",
            source_host="node-a",
            source_db="/data/db.db",
            priority_score=0.85,
            avg_player_elo=1800,
            game_length=100,
            is_decisive=True,
            queued_at=time.time(),
        )
        assert entry.priority_score == 0.85
        assert entry.is_decisive is True


# =============================================================================
# Schema Migration Tests
# =============================================================================


class TestSchemaMigration:
    """Tests for schema migration and backward compatibility."""

    def test_ensure_schema_adds_missing_columns(
        self, temp_db_path: Path
    ) -> None:
        """Test that schema migration adds missing columns."""
        # Create database with minimal schema (missing new columns)
        conn = sqlite3.connect(temp_db_path)
        conn.execute("""
            CREATE TABLE synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE manifest_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Create manifest - should migrate schema
        manifest = DataManifest(db_path=temp_db_path)

        # Check that new columns exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(synced_games)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        assert "board_type" in columns
        assert "quality_score" in columns
        assert "avg_player_elo" in columns

        manifest.close()

    def test_schema_version_updated_on_migration(
        self, temp_db_path: Path
    ) -> None:
        """Test that schema version is updated after migration."""
        # Create old schema
        conn = sqlite3.connect(temp_db_path)
        conn.executescript("""
            CREATE TABLE synced_games (
                game_id TEXT PRIMARY KEY,
                source_host TEXT NOT NULL,
                source_db TEXT NOT NULL,
                synced_at REAL NOT NULL
            );
            CREATE TABLE manifest_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            INSERT INTO manifest_metadata VALUES ('schema_version', '1.0', 0);
        """)
        conn.commit()
        conn.close()

        manifest = DataManifest(db_path=temp_db_path)

        # Check version is updated
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM manifest_metadata WHERE key='schema_version'"
        )
        version = cursor.fetchone()[0]
        conn.close()

        assert version == DataManifest.SCHEMA_VERSION
        manifest.close()

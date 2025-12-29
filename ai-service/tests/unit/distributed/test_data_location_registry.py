"""Tests for app.distributed.data_location_registry module.

This module tests the DataLocationRegistry class which handles registration
and lookup for games, models, NPZ files, and checkpoints in the cluster manifest.
"""

import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_manifest.db"


@pytest.fixture
def connection_factory(temp_db_path):
    """Create a connection factory for testing."""
    # Initialize the database with required tables
    conn = sqlite3.connect(str(temp_db_path))
    cursor = conn.cursor()

    # Create game_locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS game_locations (
            game_id TEXT,
            node_id TEXT,
            db_path TEXT,
            board_type TEXT,
            num_players INTEGER,
            engine_mode TEXT,
            registered_at REAL,
            last_seen REAL,
            is_consolidated INTEGER DEFAULT 0,
            consolidated_at REAL,
            canonical_db TEXT,
            PRIMARY KEY (game_id, node_id)
        )
    """)

    # Create model_locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_locations (
            model_path TEXT,
            node_id TEXT,
            board_type TEXT,
            num_players INTEGER,
            model_version TEXT,
            file_size INTEGER,
            registered_at REAL,
            last_seen REAL,
            PRIMARY KEY (model_path, node_id)
        )
    """)

    # Create npz_locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS npz_locations (
            npz_path TEXT,
            node_id TEXT,
            board_type TEXT,
            num_players INTEGER,
            sample_count INTEGER,
            file_size INTEGER,
            registered_at REAL,
            last_seen REAL,
            PRIMARY KEY (npz_path, node_id)
        )
    """)

    # Create checkpoint_locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_locations (
            checkpoint_path TEXT,
            node_id TEXT,
            config_key TEXT,
            board_type TEXT,
            num_players INTEGER,
            epoch INTEGER,
            step INTEGER,
            loss REAL,
            file_size INTEGER,
            is_best INTEGER DEFAULT 0,
            registered_at REAL,
            last_seen REAL,
            PRIMARY KEY (checkpoint_path, node_id)
        )
    """)

    # Create database_locations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS database_locations (
            db_path TEXT,
            node_id TEXT,
            board_type TEXT,
            num_players INTEGER,
            config_key TEXT,
            game_count INTEGER,
            file_size INTEGER,
            engine_mode TEXT,
            registered_at REAL,
            last_seen REAL,
            PRIMARY KEY (db_path, node_id)
        )
    """)

    # Create node_capacity table for GPU node counting
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS node_capacity (
            node_id TEXT PRIMARY KEY,
            gpu_type TEXT,
            gpu_memory INTEGER,
            last_seen REAL
        )
    """)

    conn.commit()
    conn.close()

    @contextmanager
    def factory() -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(str(temp_db_path))
        try:
            yield connection
        finally:
            connection.close()

    return factory


@pytest.fixture
def registry(temp_db_path, connection_factory):
    """Create a DataLocationRegistry instance for testing."""
    from app.distributed.data_location_registry import DataLocationRegistry
    return DataLocationRegistry(
        db_path=temp_db_path,
        connection_factory=connection_factory,
        node_id="test-node",
    )


# =============================================================================
# Test DataLocationRegistry Initialization
# =============================================================================


class TestDataLocationRegistryInit:
    """Tests for DataLocationRegistry initialization."""

    def test_init_with_required_params(self, temp_db_path, connection_factory):
        """Test initialization with required parameters."""
        from app.distributed.data_location_registry import DataLocationRegistry

        registry = DataLocationRegistry(
            db_path=temp_db_path,
            connection_factory=connection_factory,
            node_id="my-node",
        )

        assert registry.db_path == temp_db_path
        assert registry.node_id == "my-node"

    def test_init_stores_connection_factory(self, temp_db_path, connection_factory):
        """Test that connection factory is stored."""
        from app.distributed.data_location_registry import DataLocationRegistry

        registry = DataLocationRegistry(
            db_path=temp_db_path,
            connection_factory=connection_factory,
            node_id="test-node",
        )

        # The factory should be callable
        assert callable(registry._connection)


# =============================================================================
# Test Game Location Registry
# =============================================================================


class TestGameLocationRegistry:
    """Tests for game location registration and lookup."""

    def test_register_game(self, registry):
        """Test registering a single game."""
        registry.register_game(
            game_id="game-001",
            node_id="node-1",
            db_path="/data/games/selfplay.db",
            board_type="hex8",
            num_players=2,
            engine_mode="gumbel-mcts",
        )

        # Verify the game can be found
        locations = registry.find_game("game-001")
        assert len(locations) == 1
        assert locations[0].game_id == "game-001"
        assert locations[0].node_id == "node-1"
        assert locations[0].board_type == "hex8"
        assert locations[0].num_players == 2

    def test_register_game_updates_last_seen(self, registry):
        """Test that registering the same game updates last_seen."""
        registry.register_game(
            game_id="game-002",
            node_id="node-1",
            db_path="/data/games/selfplay.db",
        )

        time.sleep(0.01)  # Small delay to ensure different timestamp

        registry.register_game(
            game_id="game-002",
            node_id="node-1",
            db_path="/data/games/selfplay.db",
        )

        locations = registry.find_game("game-002")
        assert len(locations) == 1
        # last_seen should be updated
        assert locations[0].last_seen > locations[0].registered_at

    def test_register_games_batch(self, registry):
        """Test batch registration of games."""
        games = [
            ("game-101", "node-1", "/data/games/db1.db"),
            ("game-102", "node-2", "/data/games/db2.db"),
            ("game-103", "node-1", "/data/games/db1.db"),
        ]

        registered = registry.register_games_batch(
            games=games,
            board_type="square8",
            num_players=4,
        )

        assert registered == 3

        # Verify each game
        for game_id, node_id, db_path in games:
            locations = registry.find_game(game_id)
            assert len(locations) == 1
            assert locations[0].board_type == "square8"

    def test_register_games_batch_empty(self, registry):
        """Test batch registration with empty list."""
        registered = registry.register_games_batch(games=[])
        assert registered == 0

    def test_find_game_not_found(self, registry):
        """Test finding a game that doesn't exist."""
        locations = registry.find_game("nonexistent-game")
        assert len(locations) == 0

    def test_get_game_replication_count(self, registry):
        """Test counting game replication across nodes."""
        # Register same game on multiple nodes
        registry.register_game("game-200", "node-1", "/data/db1.db")
        registry.register_game("game-200", "node-2", "/data/db1.db")
        registry.register_game("game-200", "node-3", "/data/db1.db")

        count = registry.get_game_replication_count("game-200")
        assert count == 3

    def test_get_under_replicated_games(self, registry):
        """Test finding under-replicated games."""
        # Register games with varying replication
        registry.register_game("game-300", "node-1", "/data/db.db", "hex8", 2)
        registry.register_game("game-301", "node-1", "/data/db.db", "hex8", 2)
        registry.register_game("game-301", "node-2", "/data/db.db", "hex8", 2)
        registry.register_game("game-302", "node-1", "/data/db.db", "hex8", 2)
        registry.register_game("game-302", "node-2", "/data/db.db", "hex8", 2)
        registry.register_game("game-302", "node-3", "/data/db.db", "hex8", 2)

        # Find games with fewer than 2 copies
        under_replicated = registry.get_under_replicated_games(
            min_copies=2,
            board_type="hex8",
            num_players=2,
        )

        assert len(under_replicated) == 1
        assert under_replicated[0][0] == "game-300"
        assert under_replicated[0][1] == 1

    def test_mark_games_consolidated(self, registry):
        """Test marking games as consolidated."""
        # Register games
        registry.register_game("game-400", "node-1", "/data/db.db", "hex8", 2)
        registry.register_game("game-401", "node-1", "/data/db.db", "hex8", 2)

        # Mark as consolidated
        marked = registry.mark_games_consolidated(
            game_ids=["game-400", "game-401"],
            canonical_db="/data/canonical_hex8_2p.db",
            board_type="hex8",
            num_players=2,
        )

        assert marked == 2

    def test_get_unconsolidated_games(self, registry):
        """Test getting unconsolidated games."""
        # Register games
        registry.register_game("game-500", "node-1", "/data/db.db", "hex8", 2)
        registry.register_game("game-501", "node-1", "/data/db.db", "hex8", 2)

        # Mark one as consolidated
        registry.mark_games_consolidated(
            game_ids=["game-500"],
            canonical_db="/data/canonical.db",
        )

        # Get unconsolidated
        unconsolidated = registry.get_unconsolidated_games("hex8", 2)

        assert len(unconsolidated) == 1
        assert unconsolidated[0].game_id == "game-501"

    def test_get_game_locations(self, registry):
        """Test getting all game locations grouped by game_id."""
        registry.register_game("game-600", "node-1", "/db1.db", "hex8", 2)
        registry.register_game("game-600", "node-2", "/db2.db", "hex8", 2)
        registry.register_game("game-601", "node-1", "/db1.db", "hex8", 2)

        locations = registry.get_game_locations()

        assert "game-600" in locations
        assert "game-601" in locations
        assert len(locations["game-600"]["locations"]) == 2
        assert "node-1" in locations["game-600"]["locations"]
        assert "node-2" in locations["game-600"]["locations"]


# =============================================================================
# Test Model Location Registry
# =============================================================================


class TestModelLocationRegistry:
    """Tests for model location registration and lookup."""

    def test_register_model(self, registry):
        """Test registering a model."""
        registry.register_model(
            model_path="models/canonical_hex8_2p.pth",
            node_id="node-1",
            board_type="hex8",
            num_players=2,
            model_version="v2.1",
            file_size=40_000_000,
        )

        locations = registry.find_model("models/canonical_hex8_2p.pth")
        assert len(locations) == 1
        assert locations[0].model_path == "models/canonical_hex8_2p.pth"
        assert locations[0].model_version == "v2.1"
        assert locations[0].file_size == 40_000_000

    def test_find_model_not_found(self, registry):
        """Test finding a model that doesn't exist."""
        locations = registry.find_model("nonexistent.pth")
        assert len(locations) == 0

    def test_find_models_for_config(self, registry):
        """Test finding models for a specific config."""
        registry.register_model("model1.pth", "node-1", "hex8", 2, "v1")
        registry.register_model("model2.pth", "node-1", "hex8", 2, "v2")
        registry.register_model("model3.pth", "node-1", "square8", 4, "v1")

        models = registry.find_models_for_config("hex8", 2)

        assert len(models) == 2
        paths = [m.model_path for m in models]
        assert "model1.pth" in paths
        assert "model2.pth" in paths

    def test_get_model_availability_score_no_model(self, registry):
        """Test availability score for nonexistent model."""
        score = registry.get_model_availability_score("nonexistent.pth")
        assert score == 0.0

    def test_get_model_availability_score_with_models(self, registry, temp_db_path, connection_factory):
        """Test availability score calculation."""
        # Add some node capacity data (GPU nodes)
        with connection_factory() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO node_capacity (node_id, gpu_type) VALUES ('node-1', 'RTX4090')")
            cursor.execute("INSERT INTO node_capacity (node_id, gpu_type) VALUES ('node-2', 'A100')")
            conn.commit()

        # Register model on one node
        registry.register_model("model.pth", "node-1", "hex8", 2)

        score = registry.get_model_availability_score("model.pth")
        assert score == 0.5  # 1 node out of 2 GPU nodes

    def test_sync_model_locations_from_peers(self, registry):
        """Test syncing model locations from peer data."""
        now = time.time()
        peer_locations = [
            {
                "model_path": "peer_model1.pth",
                "node_id": "peer-1",
                "board_type": "hex8",
                "num_players": 2,
                "model_version": "v1",
                "file_size": 1000,
                "registered_at": now,
                "last_seen": now,
            },
            {
                "model_path": "peer_model2.pth",
                "node_id": "peer-2",
                "board_type": "square8",
                "num_players": 4,
                "model_version": "v2",
                "file_size": 2000,
                "registered_at": now,
                "last_seen": now,
            },
        ]

        count = registry.sync_model_locations_from_peers(peer_locations)
        assert count == 2

        # Verify synced
        locations = registry.find_model("peer_model1.pth")
        assert len(locations) == 1

    def test_sync_model_locations_skips_old(self, registry):
        """Test that old peer locations are skipped."""
        old_time = time.time() - 7200  # 2 hours ago
        peer_locations = [
            {
                "model_path": "old_model.pth",
                "node_id": "peer-1",
                "last_seen": old_time,
            },
        ]

        count = registry.sync_model_locations_from_peers(
            peer_locations,
            max_age_seconds=3600.0,  # 1 hour
        )
        assert count == 0

    def test_get_all_model_locations(self, registry):
        """Test getting all model locations as dicts."""
        registry.register_model("model1.pth", "node-1", "hex8", 2)
        registry.register_model("model2.pth", "node-2", "square8", 4)

        all_locations = registry.get_all_model_locations()

        assert len(all_locations) == 2
        assert all(isinstance(loc, dict) for loc in all_locations)
        paths = [loc["model_path"] for loc in all_locations]
        assert "model1.pth" in paths
        assert "model2.pth" in paths


# =============================================================================
# Test NPZ Location Registry
# =============================================================================


class TestNPZLocationRegistry:
    """Tests for NPZ file location registration and lookup."""

    def test_register_npz(self, registry):
        """Test registering an NPZ file."""
        registry.register_npz(
            npz_path="data/training/hex8_2p.npz",
            node_id="node-1",
            board_type="hex8",
            num_players=2,
            sample_count=100000,
            file_size=500_000_000,
        )

        locations = registry.find_npz_for_config("hex8", 2)
        assert len(locations) == 1
        assert locations[0].npz_path == "data/training/hex8_2p.npz"
        assert locations[0].sample_count == 100000

    def test_find_npz_for_config_empty(self, registry):
        """Test finding NPZ for config with no results."""
        locations = registry.find_npz_for_config("nonexistent", 99)
        assert len(locations) == 0

    def test_find_npz_ordered_by_sample_count(self, registry):
        """Test that NPZ files are ordered by sample count descending."""
        registry.register_npz("small.npz", "node-1", "hex8", 2, sample_count=1000)
        registry.register_npz("medium.npz", "node-1", "hex8", 2, sample_count=50000)
        registry.register_npz("large.npz", "node-1", "hex8", 2, sample_count=100000)

        locations = registry.find_npz_for_config("hex8", 2)

        assert len(locations) == 3
        assert locations[0].npz_path == "large.npz"
        assert locations[1].npz_path == "medium.npz"
        assert locations[2].npz_path == "small.npz"


# =============================================================================
# Test Checkpoint Location Registry
# =============================================================================


class TestCheckpointLocationRegistry:
    """Tests for checkpoint location registration and lookup."""

    def test_register_checkpoint(self, registry):
        """Test registering a checkpoint."""
        registry.register_checkpoint(
            checkpoint_path="checkpoints/hex8_2p_epoch_10.pth",
            node_id="node-1",
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            epoch=10,
            step=5000,
            loss=0.125,
            file_size=50_000_000,
            is_best=True,
        )

        locations = registry.find_checkpoint("checkpoints/hex8_2p_epoch_10.pth")
        assert len(locations) == 1
        assert locations[0].epoch == 10
        assert locations[0].is_best is True
        assert locations[0].loss == 0.125

    def test_register_checkpoint_auto_config_key(self, registry):
        """Test that config_key is auto-generated from board_type and num_players."""
        registry.register_checkpoint(
            checkpoint_path="ckpt.pth",
            node_id="node-1",
            board_type="square8",
            num_players=4,
            epoch=5,
        )

        locations = registry.find_checkpoints_for_config("square8_4p")
        assert len(locations) == 1

    def test_find_checkpoints_for_config(self, registry):
        """Test finding checkpoints for a config."""
        registry.register_checkpoint("ckpt1.pth", "node-1", "hex8_2p", epoch=1)
        registry.register_checkpoint("ckpt2.pth", "node-1", "hex8_2p", epoch=5)
        registry.register_checkpoint("ckpt3.pth", "node-1", "hex8_2p", epoch=10, is_best=True)
        registry.register_checkpoint("other.pth", "node-1", "square8_4p", epoch=1)

        all_ckpts = registry.find_checkpoints_for_config("hex8_2p")
        assert len(all_ckpts) == 3

        # Verify ordered by epoch descending
        assert all_ckpts[0].epoch == 10
        assert all_ckpts[1].epoch == 5
        assert all_ckpts[2].epoch == 1

    def test_find_checkpoints_only_best(self, registry):
        """Test finding only best checkpoints."""
        registry.register_checkpoint("ckpt1.pth", "node-1", "hex8_2p", epoch=1)
        registry.register_checkpoint("ckpt2.pth", "node-1", "hex8_2p", epoch=5, is_best=True)
        registry.register_checkpoint("ckpt3.pth", "node-1", "hex8_2p", epoch=10)

        best_only = registry.find_checkpoints_for_config("hex8_2p", only_best=True)
        assert len(best_only) == 1
        assert best_only[0].checkpoint_path == "ckpt2.pth"

    def test_get_latest_checkpoint_for_config(self, registry):
        """Test getting the latest checkpoint."""
        registry.register_checkpoint("ckpt1.pth", "node-1", "hex8_2p", epoch=1)
        registry.register_checkpoint("ckpt2.pth", "node-1", "hex8_2p", epoch=10)

        latest = registry.get_latest_checkpoint_for_config("hex8_2p", prefer_best=False)

        assert latest is not None
        assert latest.checkpoint_path == "ckpt2.pth"

    def test_get_latest_checkpoint_prefers_best(self, registry):
        """Test that latest checkpoint prefers best when prefer_best=True."""
        registry.register_checkpoint("ckpt1.pth", "node-1", "hex8_2p", epoch=5, is_best=True)
        registry.register_checkpoint("ckpt2.pth", "node-1", "hex8_2p", epoch=10)

        latest = registry.get_latest_checkpoint_for_config("hex8_2p", prefer_best=True)

        assert latest is not None
        assert latest.checkpoint_path == "ckpt1.pth"
        assert latest.is_best is True

    def test_get_latest_checkpoint_not_found(self, registry):
        """Test getting latest checkpoint when none exist."""
        latest = registry.get_latest_checkpoint_for_config("nonexistent")
        assert latest is None

    def test_mark_checkpoint_as_best(self, registry):
        """Test marking a checkpoint as best."""
        registry.register_checkpoint("ckpt1.pth", "node-1", "hex8_2p", epoch=1, is_best=True)
        registry.register_checkpoint("ckpt2.pth", "node-1", "hex8_2p", epoch=5)

        # Mark ckpt2 as best
        registry.mark_checkpoint_as_best("hex8_2p", "ckpt2.pth")

        # Verify ckpt1 is no longer best
        ckpt1 = registry.find_checkpoint("ckpt1.pth")[0]
        assert ckpt1.is_best is False

        # Verify ckpt2 is now best
        ckpt2 = registry.find_checkpoint("ckpt2.pth")[0]
        assert ckpt2.is_best is True


# =============================================================================
# Test Database Location Registry
# =============================================================================


class TestDatabaseLocationRegistry:
    """Tests for database file location registration and lookup."""

    def test_register_database(self, registry):
        """Test registering a database file."""
        registry.register_database(
            db_path="/data/games/canonical_hex8_2p.db",
            node_id="node-1",
            board_type="hex8",
            num_players=2,
            config_key="hex8_2p",
            game_count=5000,
            file_size=100_000_000,
            engine_mode="gumbel-mcts",
        )

        dbs = registry.find_databases_for_config("hex8_2p")
        assert len(dbs) == 1
        assert dbs[0]["db_path"] == "/data/games/canonical_hex8_2p.db"
        assert dbs[0]["game_count"] == 5000

    def test_register_database_auto_config_key(self, registry):
        """Test that config_key is auto-generated."""
        registry.register_database(
            db_path="/data/db.db",
            node_id="node-1",
            board_type="square8",
            num_players=4,
        )

        dbs = registry.find_databases_for_config("square8_4p")
        assert len(dbs) == 1

    def test_update_database_game_count(self, registry):
        """Test updating database game count."""
        registry.register_database(
            db_path="/data/db.db",
            node_id="node-1",
            config_key="hex8_2p",
            game_count=100,
        )

        registry.update_database_game_count(
            db_path="/data/db.db",
            node_id="node-1",
            game_count=500,
            file_size=200_000_000,
        )

        dbs = registry.find_databases_for_config("hex8_2p")
        assert dbs[0]["game_count"] == 500
        assert dbs[0]["file_size"] == 200_000_000

    def test_find_databases_for_config_by_board_type(self, registry):
        """Test finding databases by board_type and num_players."""
        registry.register_database("/db1.db", "node-1", board_type="hex8", num_players=2)
        registry.register_database("/db2.db", "node-1", board_type="hex8", num_players=4)
        registry.register_database("/db3.db", "node-1", board_type="square8", num_players=2)

        dbs = registry.find_databases_for_config(board_type="hex8", num_players=2)
        assert len(dbs) == 1
        assert dbs[0]["db_path"] == "/db1.db"

    def test_get_all_database_locations(self, registry):
        """Test getting all database locations."""
        registry.register_database("/db1.db", "node-1", config_key="hex8_2p")
        registry.register_database("/db2.db", "node-2", config_key="square8_4p")

        all_dbs = registry.get_all_database_locations()
        assert len(all_dbs) == 2

    def test_find_databases_ordered_by_game_count(self, registry):
        """Test that databases are ordered by game count descending."""
        registry.register_database("/small.db", "node-1", config_key="hex8_2p", game_count=100)
        registry.register_database("/large.db", "node-1", config_key="hex8_2p", game_count=10000)
        registry.register_database("/medium.db", "node-1", config_key="hex8_2p", game_count=1000)

        dbs = registry.find_databases_for_config("hex8_2p")

        assert dbs[0]["db_path"] == "/large.db"
        assert dbs[1]["db_path"] == "/medium.db"
        assert dbs[2]["db_path"] == "/small.db"


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_exports_data_location_registry(self):
        """Test that DataLocationRegistry is exported."""
        from app.distributed.data_location_registry import DataLocationRegistry
        assert DataLocationRegistry is not None

    def test_all_exports(self):
        """Test __all__ exports."""
        from app.distributed import data_location_registry
        assert "DataLocationRegistry" in data_location_registry.__all__

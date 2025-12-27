"""Tests for app.distributed.registries module.

Tests the decomposed ClusterManifest components:
- BaseRegistry
- GameLocationRegistry
- ModelRegistry
- NPZRegistry
- CheckpointRegistry
- NodeInventoryManager
- ReplicationManager

December 2025 - ClusterManifest decomposition.
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import pytest

from app.distributed.registries import (
    BaseRegistry,
    CheckpointRegistry,
    GameLocationRegistry,
    ModelRegistry,
    NodeInventoryManager,
    NPZRegistry,
    ReplicationManager,
)
from app.distributed.registries.game_registry import GameLocation
from app.distributed.registries.model_registry import ModelLocation
from app.distributed.registries.npz_registry import NPZLocation
from app.distributed.registries.checkpoint_registry import CheckpointLocation
from app.distributed.registries.node_inventory import NodeCapacity, NodeInventory
from app.distributed.registries.replication import (
    DataType,
    NodeRole,
    NodeSyncPolicy,
    SyncCandidateNode,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Create the schema
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS game_locations (
            game_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            db_path TEXT NOT NULL,
            board_type TEXT,
            num_players INTEGER,
            engine_mode TEXT,
            registered_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY (game_id, node_id)
        );

        CREATE TABLE IF NOT EXISTS model_locations (
            model_path TEXT NOT NULL,
            node_id TEXT NOT NULL,
            board_type TEXT,
            num_players INTEGER,
            model_version TEXT,
            file_size INTEGER DEFAULT 0,
            registered_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY (model_path, node_id)
        );

        CREATE TABLE IF NOT EXISTS npz_locations (
            npz_path TEXT NOT NULL,
            node_id TEXT NOT NULL,
            board_type TEXT,
            num_players INTEGER,
            sample_count INTEGER DEFAULT 0,
            file_size INTEGER DEFAULT 0,
            registered_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY (npz_path, node_id)
        );

        CREATE TABLE IF NOT EXISTS checkpoint_locations (
            checkpoint_path TEXT NOT NULL,
            node_id TEXT NOT NULL,
            config_key TEXT,
            board_type TEXT,
            num_players INTEGER,
            epoch INTEGER DEFAULT 0,
            step INTEGER DEFAULT 0,
            loss REAL DEFAULT 0.0,
            file_size INTEGER DEFAULT 0,
            is_best INTEGER DEFAULT 0,
            registered_at REAL NOT NULL,
            last_seen REAL NOT NULL,
            PRIMARY KEY (checkpoint_path, node_id)
        );

        CREATE TABLE IF NOT EXISTS node_capacity (
            node_id TEXT PRIMARY KEY,
            total_bytes INTEGER DEFAULT 0,
            used_bytes INTEGER DEFAULT 0,
            free_bytes INTEGER DEFAULT 0,
            usage_percent REAL DEFAULT 0.0,
            last_updated REAL NOT NULL
        );
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def game_registry(temp_db):
    """Create a GameLocationRegistry with temp database."""
    registry = GameLocationRegistry(db_path=temp_db)
    yield registry
    registry.close()


@pytest.fixture
def model_registry(temp_db):
    """Create a ModelRegistry with temp database."""
    registry = ModelRegistry(db_path=temp_db)
    yield registry
    registry.close()


@pytest.fixture
def npz_registry(temp_db):
    """Create an NPZRegistry with temp database."""
    registry = NPZRegistry(db_path=temp_db)
    yield registry
    registry.close()


@pytest.fixture
def checkpoint_registry(temp_db):
    """Create a CheckpointRegistry with temp database."""
    registry = CheckpointRegistry(db_path=temp_db)
    yield registry
    registry.close()


@pytest.fixture
def node_inventory(temp_db):
    """Create a NodeInventoryManager with temp database."""
    manager = NodeInventoryManager(db_path=temp_db)
    yield manager
    manager.close()


# =============================================================================
# BaseRegistry Tests
# =============================================================================


class TestBaseRegistry:
    """Tests for BaseRegistry."""

    def test_create_with_db_path(self, temp_db):
        """Should create registry with database path."""
        registry = BaseRegistry(db_path=temp_db)
        assert registry._db_path == temp_db
        registry.close()

    def test_external_connection(self, temp_db):
        """Should use external connection."""
        conn = sqlite3.connect(temp_db)
        registry = BaseRegistry(external_connection=conn)

        # Should use external connection
        with registry._connection() as c:
            assert c is conn

        registry.close()
        conn.close()

    def test_thread_safety(self, temp_db):
        """Should handle concurrent access."""
        registry = BaseRegistry(db_path=temp_db)
        results = []

        def worker():
            with registry._connection() as conn:
                time.sleep(0.01)
                results.append(True)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        registry.close()


# =============================================================================
# GameLocationRegistry Tests
# =============================================================================


class TestGameLocationRegistry:
    """Tests for GameLocationRegistry."""

    def test_register_game(self, game_registry):
        """Should register a game location."""
        game_registry.register_game(
            game_id="game-123",
            node_id="node-a",
            db_path="/data/games/selfplay.db",
            board_type="hex8",
            num_players=2,
        )

        locations = game_registry.find_game("game-123")
        assert len(locations) == 1
        assert locations[0].game_id == "game-123"
        assert locations[0].node_id == "node-a"
        assert locations[0].board_type == "hex8"

    def test_register_games_batch(self, game_registry):
        """Should register multiple games efficiently."""
        games = [
            ("game-1", "node-a", "/data/db.db"),
            ("game-2", "node-a", "/data/db.db"),
            ("game-3", "node-b", "/data/db.db"),
        ]

        count = game_registry.register_games_batch(
            games, board_type="square8", num_players=4
        )

        assert count == 3
        assert game_registry.get_total_unique_games() == 3

    def test_get_replication_count(self, game_registry):
        """Should count replications."""
        game_registry.register_game("game-1", "node-a", "/db.db")
        game_registry.register_game("game-1", "node-b", "/db.db")
        game_registry.register_game("game-1", "node-c", "/db.db")

        assert game_registry.get_replication_count("game-1") == 3
        assert game_registry.get_replication_count("game-missing") == 0

    def test_get_under_replicated_games(self, game_registry):
        """Should find under-replicated games."""
        game_registry.register_game("game-1", "node-a", "/db.db")  # 1 copy
        game_registry.register_game("game-2", "node-a", "/db.db")
        game_registry.register_game("game-2", "node-b", "/db.db")  # 2 copies

        under_replicated = game_registry.get_under_replicated_games(min_copies=2)
        assert len(under_replicated) == 1
        assert under_replicated[0][0] == "game-1"

    def test_count_by_config(self, game_registry):
        """Should count games by configuration."""
        game_registry.register_game("g1", "n1", "/db.db", "hex8", 2)
        game_registry.register_game("g2", "n1", "/db.db", "hex8", 2)
        game_registry.register_game("g3", "n1", "/db.db", "square8", 4)

        by_config = game_registry.count_games_by_config()
        assert by_config.get("hex8_2p") == 2
        assert by_config.get("square8_4p") == 1

    def test_export_import(self, game_registry):
        """Should export and import for P2P."""
        game_registry.register_game("g1", "node-a", "/db.db", "hex8", 2)

        exported = game_registry.export_for_node("node-a")
        assert len(exported) == 1

        # Import to different node
        imported = game_registry.import_from_remote("node-b", exported)
        assert imported == 1

        # Should now be on both nodes
        assert game_registry.get_replication_count("g1") == 2


# =============================================================================
# ModelRegistry Tests
# =============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self, model_registry):
        """Should register a model location."""
        model_registry.register_model(
            model_path="models/canonical_hex8_2p.pth",
            node_id="node-a",
            board_type="hex8",
            num_players=2,
            file_size=1024000,
        )

        locations = model_registry.find_model("models/canonical_hex8_2p.pth")
        assert len(locations) == 1
        assert locations[0].file_size == 1024000

    def test_find_models_for_config(self, model_registry):
        """Should find models by configuration."""
        model_registry.register_model("m1.pth", "n1", "hex8", 2)
        model_registry.register_model("m2.pth", "n1", "hex8", 2)
        model_registry.register_model("m3.pth", "n1", "square8", 4)

        hex8_models = model_registry.find_models_for_config("hex8", 2)
        assert len(hex8_models) == 2

    def test_total_size_by_node(self, model_registry):
        """Should calculate total size by node."""
        model_registry.register_model("m1.pth", "n1", file_size=1000)
        model_registry.register_model("m2.pth", "n1", file_size=2000)
        model_registry.register_model("m3.pth", "n2", file_size=3000)

        assert model_registry.get_total_size_by_node("n1") == 3000
        assert model_registry.get_total_size_by_node("n2") == 3000


# =============================================================================
# NPZRegistry Tests
# =============================================================================


class TestNPZRegistry:
    """Tests for NPZRegistry."""

    def test_register_npz(self, npz_registry):
        """Should register an NPZ file location."""
        npz_registry.register_npz(
            npz_path="data/training/hex8_2p.npz",
            node_id="node-a",
            board_type="hex8",
            num_players=2,
            sample_count=100000,
            file_size=50000000,
        )

        locations = npz_registry.find_npz("data/training/hex8_2p.npz")
        assert len(locations) == 1
        assert locations[0].sample_count == 100000

    def test_find_for_config(self, npz_registry):
        """Should find NPZ files by configuration."""
        npz_registry.register_npz("npz1.npz", "n1", "hex8", 2, sample_count=1000)
        npz_registry.register_npz("npz2.npz", "n1", "hex8", 2, sample_count=2000)

        files = npz_registry.find_npz_for_config("hex8", 2)
        assert len(files) == 2
        # Should be sorted by sample_count descending
        assert files[0].sample_count == 2000


# =============================================================================
# CheckpointRegistry Tests
# =============================================================================


class TestCheckpointRegistry:
    """Tests for CheckpointRegistry."""

    def test_register_checkpoint(self, checkpoint_registry):
        """Should register a checkpoint."""
        checkpoint_registry.register_checkpoint(
            checkpoint_path="checkpoints/hex8_2p/epoch_50.pth",
            node_id="node-a",
            config_key="hex8_2p",
            board_type="hex8",
            num_players=2,
            epoch=50,
            loss=0.5,
        )

        locations = checkpoint_registry.find_checkpoint(
            "checkpoints/hex8_2p/epoch_50.pth"
        )
        assert len(locations) == 1
        assert locations[0].epoch == 50

    def test_find_checkpoints_for_config(self, checkpoint_registry):
        """Should find checkpoints by config."""
        checkpoint_registry.register_checkpoint(
            "ckpt1.pth", "n1", "hex8_2p", epoch=10
        )
        checkpoint_registry.register_checkpoint(
            "ckpt2.pth", "n1", "hex8_2p", epoch=20
        )

        checkpoints = checkpoint_registry.find_checkpoints_for_config("hex8_2p")
        assert len(checkpoints) == 2
        # Should be sorted by epoch descending
        assert checkpoints[0].epoch == 20

    def test_mark_as_best(self, checkpoint_registry):
        """Should mark checkpoint as best."""
        checkpoint_registry.register_checkpoint("ckpt1.pth", "n1", "hex8_2p")
        checkpoint_registry.register_checkpoint("ckpt2.pth", "n1", "hex8_2p")

        checkpoint_registry.mark_as_best("hex8_2p", "ckpt2.pth")

        best = checkpoint_registry.find_checkpoints_for_config("hex8_2p", only_best=True)
        assert len(best) == 1
        assert best[0].checkpoint_path == "ckpt2.pth"

    def test_get_latest(self, checkpoint_registry):
        """Should get latest checkpoint."""
        checkpoint_registry.register_checkpoint(
            "ckpt1.pth", "n1", "hex8_2p", epoch=10
        )
        checkpoint_registry.register_checkpoint(
            "ckpt2.pth", "n1", "hex8_2p", epoch=20
        )

        latest = checkpoint_registry.get_latest_checkpoint("hex8_2p")
        assert latest is not None
        assert latest.epoch == 20


# =============================================================================
# NodeInventoryManager Tests
# =============================================================================


class TestNodeInventoryManager:
    """Tests for NodeInventoryManager."""

    def test_update_capacity(self, node_inventory):
        """Should update node capacity."""
        node_inventory.update_node_capacity(
            node_id="node-a",
            total_bytes=1000000000,
            used_bytes=300000000,
            free_bytes=700000000,
        )

        capacity = node_inventory.get_node_capacity("node-a")
        assert capacity is not None
        assert capacity.usage_percent == pytest.approx(30.0, rel=0.1)
        assert capacity.can_receive_sync is True

    def test_get_nodes_with_free_space(self, node_inventory):
        """Should find nodes with free space."""
        node_inventory.update_node_capacity("n1", 1000, 300, 700)  # 30% used
        node_inventory.update_node_capacity("n2", 1000, 800, 200)  # 80% used

        nodes = node_inventory.get_nodes_with_free_space(min_free_percent=30)
        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"

    def test_capacity_stats(self, node_inventory):
        """Should aggregate capacity stats."""
        node_inventory.update_node_capacity("n1", 1000, 300, 700)
        node_inventory.update_node_capacity("n2", 1000, 500, 500)

        stats = node_inventory.get_cluster_capacity_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_bytes"] == 2000
        assert stats["avg_usage_percent"] == pytest.approx(40.0, rel=0.1)


# =============================================================================
# ReplicationManager Tests
# =============================================================================


class TestReplicationManager:
    """Tests for ReplicationManager."""

    def test_default_policy(self, temp_db):
        """Should return default policy for unknown node."""
        manager = ReplicationManager(db_path=temp_db)
        policy = manager.get_sync_policy("unknown-node")

        assert policy.node_id == "unknown-node"
        assert policy.receive_games is True
        assert policy.receive_models is True
        manager.close()

    def test_can_receive_data(self, temp_db):
        """Should check data type permissions."""
        manager = ReplicationManager(db_path=temp_db)

        assert manager.can_receive_data("node-a", DataType.GAME) is True
        assert manager.can_receive_data("node-a", DataType.MODEL) is True
        manager.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Integration tests for registry components."""

    def test_shared_connection(self, temp_db):
        """Should share connection across registries."""
        conn = sqlite3.connect(temp_db)
        lock = threading.RLock()

        game_reg = GameLocationRegistry(external_connection=conn, external_lock=lock)
        model_reg = ModelRegistry(external_connection=conn, external_lock=lock)

        # Both should use same connection
        game_reg.register_game("g1", "n1", "/db.db")
        model_reg.register_model("m1.pth", "n1")

        assert game_reg.get_total_unique_games() == 1
        assert model_reg.get_total_unique_models() == 1

        game_reg.close()
        model_reg.close()
        conn.close()

    def test_node_inventory_with_registries(self, temp_db):
        """Should get inventory using registries."""
        conn = sqlite3.connect(temp_db)
        lock = threading.RLock()

        game_reg = GameLocationRegistry(external_connection=conn, external_lock=lock)
        model_reg = ModelRegistry(external_connection=conn, external_lock=lock)
        npz_reg = NPZRegistry(external_connection=conn, external_lock=lock)
        inventory = NodeInventoryManager(external_connection=conn, external_lock=lock)

        # Add data
        game_reg.register_game("g1", "n1", "/db.db")
        model_reg.register_model("m1.pth", "n1", file_size=1000)
        npz_reg.register_npz("npz1.npz", "n1", sample_count=100)

        # Get inventory
        inv = inventory.get_node_inventory(
            "n1",
            game_registry=game_reg,
            model_registry=model_reg,
            npz_registry=npz_reg,
        )

        assert inv.game_count == 1
        assert inv.model_count == 1
        assert inv.npz_count == 1
        assert inv.total_models_size == 1000

        game_reg.close()
        model_reg.close()
        npz_reg.close()
        inventory.close()
        conn.close()

"""Tests for ClusterManifest - Central Registry for Data Locations.

Tests cover:
- Game/model/NPZ location registration and lookup
- Node capacity tracking
- Replication target selection
- Sync policy enforcement
- Disk cleanup logic
- Manifest import/export for P2P gossip
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.cluster_manifest import (
    CleanupCandidate,
    ClusterManifest,
    DataType,
    DiskCleanupPolicy,
    GameLocation,
    ModelLocation,
    NodeCapacity,
    NodeInventory,
    NodeSyncPolicy,
    NPZLocation,
    REPLICATION_TARGET_COUNT,
    SyncTarget,
    get_cluster_manifest,
    reset_cluster_manifest,
)


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_manifest.db"


@pytest.fixture
def manifest(temp_db_path):
    """Create a ClusterManifest with a temporary database."""
    # Reset singleton
    reset_cluster_manifest()
    manifest = ClusterManifest(db_path=temp_db_path)
    yield manifest
    manifest.close()
    reset_cluster_manifest()


class TestClusterManifestInit:
    """Tests for ClusterManifest initialization."""

    def test_init_creates_database(self, temp_db_path):
        """Test that initialization creates the database file."""
        manifest = ClusterManifest(db_path=temp_db_path)
        assert temp_db_path.exists()
        manifest.close()

    def test_init_creates_tables(self, temp_db_path):
        """Test that initialization creates required tables."""
        manifest = ClusterManifest(db_path=temp_db_path)

        # Check tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "game_locations" in tables
        assert "model_locations" in tables
        assert "npz_locations" in tables
        assert "node_capacity" in tables
        assert "manifest_metadata" in tables

        manifest.close()

    def test_singleton_accessor(self, temp_db_path):
        """Test singleton accessor returns same instance."""
        reset_cluster_manifest()

        with patch(
            "app.distributed.cluster_manifest._cluster_manifest", None
        ):
            m1 = get_cluster_manifest()
            m2 = get_cluster_manifest()
            assert m1 is m2
            m1.close()

        reset_cluster_manifest()


class TestGameLocationRegistry:
    """Tests for game location registration and lookup."""

    def test_register_game(self, manifest):
        """Test registering a single game."""
        manifest.register_game(
            game_id="game-001",
            node_id="node-a",
            db_path="/data/games/selfplay.db",
            board_type="hex8",
            num_players=2,
            engine_mode="gumbel-mcts",
        )

        locations = manifest.find_game("game-001")
        assert len(locations) == 1
        assert locations[0].game_id == "game-001"
        assert locations[0].node_id == "node-a"
        assert locations[0].db_path == "/data/games/selfplay.db"
        assert locations[0].board_type == "hex8"
        assert locations[0].num_players == 2

    def test_register_game_multiple_locations(self, manifest):
        """Test game replicated across multiple nodes."""
        manifest.register_game("game-001", "node-a", "/data/selfplay.db")
        manifest.register_game("game-001", "node-b", "/data/selfplay.db")
        manifest.register_game("game-001", "node-c", "/data/selfplay.db")

        locations = manifest.find_game("game-001")
        assert len(locations) == 3
        node_ids = {loc.node_id for loc in locations}
        assert node_ids == {"node-a", "node-b", "node-c"}

    def test_register_games_batch(self, manifest):
        """Test batch registration of games."""
        games = [
            ("game-001", "node-a", "/data/db1.db"),
            ("game-002", "node-a", "/data/db1.db"),
            ("game-003", "node-b", "/data/db2.db"),
        ]

        count = manifest.register_games_batch(
            games, board_type="square8", num_players=4
        )

        assert count == 3
        assert len(manifest.find_game("game-001")) == 1
        assert len(manifest.find_game("game-002")) == 1
        assert len(manifest.find_game("game-003")) == 1

    def test_get_game_replication_count(self, manifest):
        """Test counting replication of a game."""
        manifest.register_game("game-001", "node-a", "/data/db.db")
        manifest.register_game("game-001", "node-b", "/data/db.db")

        count = manifest.get_game_replication_count("game-001")
        assert count == 2

    def test_get_under_replicated_games(self, manifest):
        """Test finding under-replicated games."""
        # Game with 1 copy (under-replicated)
        manifest.register_game("game-001", "node-a", "/data/db.db")

        # Game with 2 copies (fully replicated)
        manifest.register_game("game-002", "node-a", "/data/db.db")
        manifest.register_game("game-002", "node-b", "/data/db.db")

        under_rep = manifest.get_under_replicated_games(min_copies=2)
        game_ids = [g[0] for g in under_rep]

        assert "game-001" in game_ids
        assert "game-002" not in game_ids

    def test_find_nonexistent_game(self, manifest):
        """Test finding a game that doesn't exist."""
        locations = manifest.find_game("nonexistent")
        assert locations == []


class TestModelLocationRegistry:
    """Tests for model location registration and lookup."""

    def test_register_model(self, manifest):
        """Test registering a model."""
        manifest.register_model(
            model_path="models/canonical_hex8_2p.pth",
            node_id="node-a",
            board_type="hex8",
            num_players=2,
            model_version="v2",
            file_size=125_000_000,
        )

        locations = manifest.find_model("models/canonical_hex8_2p.pth")
        assert len(locations) == 1
        assert locations[0].model_path == "models/canonical_hex8_2p.pth"
        assert locations[0].node_id == "node-a"
        assert locations[0].file_size == 125_000_000

    def test_find_models_for_config(self, manifest):
        """Test finding models by board configuration."""
        manifest.register_model(
            "models/hex8_2p_v1.pth", "node-a", "hex8", 2
        )
        manifest.register_model(
            "models/hex8_2p_v2.pth", "node-b", "hex8", 2
        )
        manifest.register_model(
            "models/square8_2p.pth", "node-a", "square8", 2
        )

        hex8_models = manifest.find_models_for_config("hex8", 2)
        assert len(hex8_models) == 2

        square8_models = manifest.find_models_for_config("square8", 2)
        assert len(square8_models) == 1


class TestNPZLocationRegistry:
    """Tests for NPZ file location registration and lookup."""

    def test_register_npz(self, manifest):
        """Test registering an NPZ file."""
        manifest.register_npz(
            npz_path="data/training/hex8_2p.npz",
            node_id="node-a",
            board_type="hex8",
            num_players=2,
            sample_count=1_000_000,
            file_size=500_000_000,
        )

        locations = manifest.find_npz_for_config("hex8", 2)
        assert len(locations) == 1
        assert locations[0].sample_count == 1_000_000

    def test_find_npz_sorted_by_samples(self, manifest):
        """Test NPZ files are sorted by sample count."""
        manifest.register_npz(
            "data/hex8_2p_small.npz", "node-a", "hex8", 2,
            sample_count=100_000
        )
        manifest.register_npz(
            "data/hex8_2p_large.npz", "node-b", "hex8", 2,
            sample_count=1_000_000
        )

        locations = manifest.find_npz_for_config("hex8", 2)
        assert len(locations) == 2
        # Should be sorted with larger sample count first
        assert locations[0].sample_count > locations[1].sample_count


class TestNodeCapacity:
    """Tests for node capacity tracking."""

    def test_update_node_capacity(self, manifest):
        """Test updating node capacity."""
        manifest.update_node_capacity(
            node_id="node-a",
            total_bytes=1_000_000_000_000,  # 1TB
            used_bytes=500_000_000_000,      # 500GB
            free_bytes=500_000_000_000,      # 500GB
        )

        capacity = manifest.get_node_capacity("node-a")
        assert capacity is not None
        assert capacity.total_bytes == 1_000_000_000_000
        assert capacity.usage_percent == 50.0
        assert capacity.can_receive_sync is True

    def test_node_capacity_threshold(self, manifest):
        """Test capacity threshold for sync eligibility."""
        # Node at 75% usage (above threshold)
        manifest.update_node_capacity(
            node_id="node-full",
            total_bytes=1_000_000_000_000,
            used_bytes=750_000_000_000,
            free_bytes=250_000_000_000,
        )

        capacity = manifest.get_node_capacity("node-full")
        assert capacity.usage_percent == 75.0
        assert capacity.can_receive_sync is False

    def test_get_node_inventory(self, manifest):
        """Test getting full node inventory."""
        # Register various data
        manifest.register_game("game-001", "node-a", "/data/db.db")
        manifest.register_game("game-002", "node-a", "/data/db.db")
        manifest.register_model(
            "models/model.pth", "node-a", file_size=100_000_000
        )
        manifest.register_npz(
            "data/training.npz", "node-a", file_size=500_000_000
        )
        manifest.update_node_capacity(
            "node-a", 1_000_000_000_000, 500_000_000_000, 500_000_000_000
        )

        inventory = manifest.get_node_inventory("node-a")
        assert inventory.game_count == 2
        assert inventory.model_count == 1
        assert inventory.npz_count == 1
        assert inventory.capacity is not None


class TestSyncTargetSelection:
    """Tests for sync target selection logic."""

    def test_can_receive_data_game(self, manifest):
        """Test checking if node can receive game data."""
        # Mock sync policy
        manifest._exclusion_rules["node-a"] = NodeSyncPolicy(
            node_id="node-a",
            receive_games=True,
        )
        manifest._exclusion_rules["node-b"] = NodeSyncPolicy(
            node_id="node-b",
            receive_games=False,
            exclusion_reason="coordinator",
        )

        assert manifest.can_receive_data("node-a", DataType.GAME) is True
        assert manifest.can_receive_data("node-b", DataType.GAME) is False

    def test_get_replication_targets(self, manifest):
        """Test getting replication targets for a game."""
        # Register game on one node
        manifest.register_game("game-001", "node-a", "/data/db.db")

        # Add capacity for potential targets
        manifest.update_node_capacity(
            "node-b", 1_000_000_000_000, 300_000_000_000, 700_000_000_000
        )
        manifest.update_node_capacity(
            "node-c", 1_000_000_000_000, 400_000_000_000, 600_000_000_000
        )

        # Set up sync policies
        manifest._exclusion_rules["node-b"] = NodeSyncPolicy(
            node_id="node-b", receive_games=True
        )
        manifest._exclusion_rules["node-c"] = NodeSyncPolicy(
            node_id="node-c", receive_games=True
        )

        targets = manifest.get_replication_targets("game-001", min_copies=2)
        # Should return 1 target (need 2 copies, have 1)
        assert len(targets) == 1
        assert targets[0].node_id in {"node-b", "node-c"}


class TestManifestPropagation:
    """Tests for manifest export/import (P2P gossip)."""

    def test_export_local_state(self, manifest):
        """Test exporting local manifest state."""
        manifest.register_game("game-001", manifest.node_id, "/data/db.db")
        manifest.register_model(
            "models/model.pth", manifest.node_id, file_size=100_000
        )
        manifest.update_local_capacity()

        state = manifest.export_local_state()

        assert state["node_id"] == manifest.node_id
        assert len(state["games"]) == 1
        assert len(state["models"]) == 1
        assert "timestamp" in state

    def test_import_remote_state(self, manifest):
        """Test importing state from remote node."""
        remote_state = {
            "node_id": "remote-node",
            "timestamp": time.time(),
            "games": [
                {
                    "game_id": "game-remote-001",
                    "db_path": "/data/remote.db",
                    "board_type": "hex8",
                    "num_players": 2,
                    "last_seen": time.time(),
                }
            ],
            "models": [
                {
                    "model_path": "models/remote.pth",
                    "board_type": "hex8",
                    "num_players": 2,
                    "file_size": 100_000,
                    "last_seen": time.time(),
                }
            ],
            "npz_files": [],
            "capacity": {
                "total_bytes": 1_000_000_000_000,
                "used_bytes": 500_000_000_000,
                "free_bytes": 500_000_000_000,
            },
        }

        imported = manifest.import_remote_state(remote_state)

        assert imported == 2  # 1 game + 1 model
        locations = manifest.find_game("game-remote-001")
        assert len(locations) == 1
        assert locations[0].node_id == "remote-node"


class TestClusterStats:
    """Tests for cluster statistics."""

    def test_get_cluster_stats(self, manifest):
        """Test getting cluster-wide statistics."""
        # Register data on multiple nodes
        manifest.register_game("game-001", "node-a", "/data/db.db", "hex8", 2)
        manifest.register_game("game-001", "node-b", "/data/db.db", "hex8", 2)
        manifest.register_game("game-002", "node-a", "/data/db.db", "hex8", 2)
        manifest.register_model("models/m1.pth", "node-a")
        manifest.register_npz("data/t1.npz", "node-a")

        stats = manifest.get_cluster_stats()

        assert stats["total_games"] == 2  # 2 unique games
        assert stats["total_models"] == 1
        assert stats["total_npz_files"] == 1
        assert "hex8_2p" in stats["games_by_config"]


class TestDiskCleanup:
    """Tests for disk cleanup functionality."""

    def test_cleanup_candidate_priority(self):
        """Test cleanup priority calculation."""
        # Old, low quality, well-replicated = highest priority
        candidate1 = CleanupCandidate(
            path=Path("/data/old.db"),
            data_type=DataType.GAME,
            size_bytes=1_000_000,
            age_days=60,
            quality_score=0.2,
            replication_count=3,
        )

        # New, high quality, under-replicated = lower priority
        candidate2 = CleanupCandidate(
            path=Path("/data/new.db"),
            data_type=DataType.GAME,
            size_bytes=1_000_000,
            age_days=1,
            quality_score=0.9,
            replication_count=1,
        )

        assert candidate1.cleanup_priority > candidate2.cleanup_priority

    def test_canonical_never_deleted(self):
        """Test that canonical databases are never deleted."""
        candidate = CleanupCandidate(
            path=Path("/data/canonical.db"),
            data_type=DataType.GAME,
            size_bytes=1_000_000,
            age_days=365,
            quality_score=0.1,
            replication_count=5,
            is_canonical=True,
        )

        assert candidate.cleanup_priority == -1000.0

    def test_check_disk_cleanup_needed(self, manifest):
        """Test checking if cleanup is needed."""
        policy = DiskCleanupPolicy(trigger_usage_percent=70)

        # Mock local capacity at 60%
        with patch.object(manifest, "update_local_capacity") as mock:
            mock.return_value = NodeCapacity(
                node_id=manifest.node_id,
                total_bytes=1_000_000_000_000,
                used_bytes=600_000_000_000,
                free_bytes=400_000_000_000,
                usage_percent=60.0,
            )
            assert manifest.check_disk_cleanup_needed(policy) is False

        # Mock at 75%
        with patch.object(manifest, "update_local_capacity") as mock:
            mock.return_value = NodeCapacity(
                node_id=manifest.node_id,
                total_bytes=1_000_000_000_000,
                used_bytes=750_000_000_000,
                free_bytes=250_000_000_000,
                usage_percent=75.0,
            )
            assert manifest.check_disk_cleanup_needed(policy) is True


class TestDataClasses:
    """Tests for data classes."""

    def test_game_location_fields(self):
        """Test GameLocation dataclass."""
        loc = GameLocation(
            game_id="game-001",
            node_id="node-a",
            db_path="/data/db.db",
            board_type="hex8",
            num_players=2,
        )
        assert loc.game_id == "game-001"
        assert loc.registered_at == 0.0  # default

    def test_node_capacity_properties(self):
        """Test NodeCapacity computed properties."""
        cap = NodeCapacity(
            node_id="node-a",
            total_bytes=1000,
            used_bytes=650,
            free_bytes=350,
            usage_percent=65.0,
        )
        assert cap.can_receive_sync is True
        assert cap.free_percent == 35.0

        cap2 = NodeCapacity(
            node_id="node-b",
            usage_percent=75.0,
        )
        assert cap2.can_receive_sync is False

    def test_sync_target_fields(self):
        """Test SyncTarget dataclass."""
        target = SyncTarget(
            node_id="node-a",
            priority=80,
            reason="training node, 50% free",
        )
        assert target.priority == 80
        assert "training" in target.reason


class TestThreadSafety:
    """Tests for thread-safe database access."""

    def test_concurrent_registration(self, manifest):
        """Test concurrent game registration."""
        import threading

        results = []

        def register_games(start_id):
            for i in range(100):
                manifest.register_game(
                    f"game-{start_id}-{i}",
                    "node-a",
                    "/data/db.db"
                )
            results.append(True)

        threads = [
            threading.Thread(target=register_games, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5  # All threads completed

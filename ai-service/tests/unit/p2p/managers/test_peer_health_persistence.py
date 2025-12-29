"""Unit tests for peer health persistence in StateManager.

December 28, 2025: Phase 7 - Peer health state persistence tests.
Tests the PeerHealthState dataclass and StateManager persistence methods.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPeerHealthStateImport:
    """Test that PeerHealthState can be imported."""

    def test_import_peer_health_state(self):
        """Test basic import."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        assert PeerHealthState is not None


class TestPeerHealthStateDataclass:
    """Test PeerHealthState dataclass."""

    def test_create_with_minimal_args(self):
        """Test creating with minimal arguments."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(node_id="test-node", state="alive")
        assert health.node_id == "test-node"
        assert health.state == "alive"
        assert health.failure_count == 0
        assert health.gossip_failure_count == 0

    def test_create_with_all_args(self):
        """Test creating with all arguments."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(
            node_id="test-node",
            state="suspect",
            failure_count=5,
            gossip_failure_count=3,
            last_seen=1000.0,
            last_failure=900.0,
            circuit_state="open",
            circuit_opened_at=850.0,
            updated_at=1100.0,
        )
        assert health.node_id == "test-node"
        assert health.state == "suspect"
        assert health.failure_count == 5
        assert health.gossip_failure_count == 3
        assert health.last_seen == 1000.0
        assert health.last_failure == 900.0
        assert health.circuit_state == "open"
        assert health.circuit_opened_at == 850.0
        assert health.updated_at == 1100.0

    def test_to_dict(self):
        """Test to_dict serialization."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(
            node_id="test-node",
            state="dead",
            failure_count=10,
        )
        d = health.to_dict()

        assert d["node_id"] == "test-node"
        assert d["state"] == "dead"
        assert d["failure_count"] == 10
        assert "updated_at" in d

    def test_from_dict(self):
        """Test from_dict deserialization."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        d = {
            "node_id": "restored-node",
            "state": "retired",
            "failure_count": 7,
            "gossip_failure_count": 2,
            "circuit_state": "half_open",
        }
        health = PeerHealthState.from_dict(d)

        assert health.node_id == "restored-node"
        assert health.state == "retired"
        assert health.failure_count == 7
        assert health.gossip_failure_count == 2
        assert health.circuit_state == "half_open"

    def test_from_dict_with_defaults(self):
        """Test from_dict with missing fields uses defaults."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        d = {"node_id": "minimal-node"}
        health = PeerHealthState.from_dict(d)

        assert health.node_id == "minimal-node"
        assert health.state == "alive"
        assert health.failure_count == 0
        assert health.circuit_state == "closed"

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        original = PeerHealthState(
            node_id="roundtrip-node",
            state="suspect",
            failure_count=5,
            gossip_failure_count=3,
            last_seen=time.time(),
            circuit_state="open",
        )
        d = original.to_dict()
        restored = PeerHealthState.from_dict(d)

        assert restored.node_id == original.node_id
        assert restored.state == original.state
        assert restored.failure_count == original.failure_count
        assert restored.gossip_failure_count == original.gossip_failure_count
        assert restored.circuit_state == original.circuit_state


class TestStateManagerPeerHealthSchema:
    """Test StateManager creates peer_health_history table."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield Path(f.name)

    def test_init_creates_peer_health_table(self, temp_db):
        """Test init_database creates peer_health_history table."""
        from scripts.p2p.managers.state_manager import StateManager

        manager = StateManager(temp_db)
        manager.init_database()

        # Check table exists
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='peer_health_history'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_init_creates_peer_health_index(self, temp_db):
        """Test init_database creates index on peer_health_history."""
        from scripts.p2p.managers.state_manager import StateManager

        manager = StateManager(temp_db)
        manager.init_database()

        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_peer_health_state'"
        )
        assert cursor.fetchone() is not None
        conn.close()


class TestStateManagerSavePeerHealth:
    """Test StateManager.save_peer_health."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_save_single_peer_health(self, manager):
        """Test saving a single peer health record."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(
            node_id="test-peer",
            state="alive",
            failure_count=2,
        )
        result = manager.save_peer_health(health)
        assert result is True

    def test_save_updates_existing(self, manager):
        """Test saving updates existing record."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health1 = PeerHealthState(node_id="test-peer", state="alive", failure_count=1)
        health2 = PeerHealthState(node_id="test-peer", state="suspect", failure_count=5)

        manager.save_peer_health(health1)
        manager.save_peer_health(health2)

        loaded = manager.load_peer_health("test-peer")
        assert loaded.state == "suspect"
        assert loaded.failure_count == 5


class TestStateManagerSavePeerHealthBatch:
    """Test StateManager.save_peer_health_batch."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_save_batch_empty(self, manager):
        """Test saving empty batch."""
        result = manager.save_peer_health_batch([])
        assert result == 0

    def test_save_batch_multiple(self, manager):
        """Test saving multiple records in batch."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        healths = [
            PeerHealthState(node_id=f"peer-{i}", state="alive")
            for i in range(5)
        ]
        saved = manager.save_peer_health_batch(healths)
        assert saved == 5

    def test_save_batch_transaction(self, manager):
        """Test batch save is atomic."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        # Save initial state
        health1 = PeerHealthState(node_id="peer-1", state="alive", failure_count=0)
        manager.save_peer_health(health1)

        # Update in batch
        healths = [
            PeerHealthState(node_id="peer-1", state="suspect", failure_count=5),
            PeerHealthState(node_id="peer-2", state="alive", failure_count=0),
        ]
        manager.save_peer_health_batch(healths)

        # Verify both saved
        loaded1 = manager.load_peer_health("peer-1")
        loaded2 = manager.load_peer_health("peer-2")
        assert loaded1.state == "suspect"
        assert loaded2 is not None


class TestStateManagerLoadPeerHealth:
    """Test StateManager.load_peer_health."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_load_nonexistent_peer(self, manager):
        """Test loading nonexistent peer returns None."""
        result = manager.load_peer_health("nonexistent")
        assert result is None

    def test_load_existing_peer(self, manager):
        """Test loading existing peer."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(
            node_id="existing-peer",
            state="suspect",
            failure_count=3,
            gossip_failure_count=2,
        )
        manager.save_peer_health(health)

        loaded = manager.load_peer_health("existing-peer")
        assert loaded is not None
        assert loaded.node_id == "existing-peer"
        assert loaded.state == "suspect"
        assert loaded.failure_count == 3
        assert loaded.gossip_failure_count == 2


class TestStateManagerLoadAllPeerHealth:
    """Test StateManager.load_all_peer_health."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_load_all_empty(self, manager):
        """Test load_all when empty."""
        result = manager.load_all_peer_health()
        assert result == {}

    def test_load_all_with_data(self, manager):
        """Test load_all with data."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        healths = [
            PeerHealthState(node_id=f"peer-{i}", state="alive")
            for i in range(3)
        ]
        manager.save_peer_health_batch(healths)

        result = manager.load_all_peer_health()
        assert len(result) == 3
        assert "peer-0" in result
        assert "peer-1" in result
        assert "peer-2" in result

    def test_load_all_filters_by_age(self, manager):
        """Test load_all filters by max_age_seconds."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        # Save with explicit old timestamp
        with manager._db_connection() as conn:
            cursor = conn.cursor()
            old_time = time.time() - 7200  # 2 hours ago
            cursor.execute(
                """
                INSERT INTO peer_health_history
                (node_id, state, failure_count, gossip_failure_count,
                 last_seen, last_failure, circuit_state, circuit_opened_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("old-peer", "alive", 0, 0, 0, 0, "closed", 0, old_time),
            )
            conn.commit()

        # Save recent peer
        health = PeerHealthState(node_id="new-peer", state="alive")
        manager.save_peer_health(health)

        # Load with 1 hour max age
        result = manager.load_all_peer_health(max_age_seconds=3600)
        assert len(result) == 1
        assert "new-peer" in result
        assert "old-peer" not in result


class TestStateManagerClearStalePeerHealth:
    """Test StateManager.clear_stale_peer_health."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_clear_stale_empty(self, manager):
        """Test clear_stale when empty."""
        result = manager.clear_stale_peer_health()
        assert result == 0

    def test_clear_stale_removes_old(self, manager):
        """Test clear_stale removes old records."""
        # Insert old record directly
        with manager._db_connection() as conn:
            cursor = conn.cursor()
            old_time = time.time() - 100000  # Very old
            cursor.execute(
                """
                INSERT INTO peer_health_history
                (node_id, state, updated_at)
                VALUES (?, ?, ?)
                """,
                ("old-peer", "alive", old_time),
            )
            conn.commit()

        # Clear with default (24h)
        cleared = manager.clear_stale_peer_health(max_age_seconds=86400)
        assert cleared == 1

    def test_clear_stale_preserves_recent(self, manager):
        """Test clear_stale preserves recent records."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        health = PeerHealthState(node_id="recent-peer", state="alive")
        manager.save_peer_health(health)

        cleared = manager.clear_stale_peer_health(max_age_seconds=86400)
        assert cleared == 0

        # Verify still exists
        loaded = manager.load_peer_health("recent-peer")
        assert loaded is not None


class TestStateManagerGetPeerHealthSummary:
    """Test StateManager.get_peer_health_summary."""

    @pytest.fixture
    def manager(self):
        """Create a StateManager with temp database."""
        from scripts.p2p.managers.state_manager import StateManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        manager = StateManager(db_path)
        manager.init_database()
        yield manager

    def test_summary_empty(self, manager):
        """Test summary when empty."""
        summary = manager.get_peer_health_summary()
        assert summary["total_records"] == 0
        assert summary["state_counts"] == {}
        assert summary["circuit_state_counts"] == {}

    def test_summary_with_data(self, manager):
        """Test summary with data."""
        from scripts.p2p.managers.state_manager import PeerHealthState

        healths = [
            PeerHealthState(node_id="peer-1", state="alive", circuit_state="closed"),
            PeerHealthState(node_id="peer-2", state="alive", circuit_state="closed"),
            PeerHealthState(node_id="peer-3", state="suspect", circuit_state="open"),
            PeerHealthState(node_id="peer-4", state="dead", circuit_state="open"),
        ]
        manager.save_peer_health_batch(healths)

        summary = manager.get_peer_health_summary()
        assert summary["total_records"] == 4
        assert summary["state_counts"]["alive"] == 2
        assert summary["state_counts"]["suspect"] == 1
        assert summary["state_counts"]["dead"] == 1
        assert summary["circuit_state_counts"]["closed"] == 2
        assert summary["circuit_state_counts"]["open"] == 2

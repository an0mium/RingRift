"""Unit tests for peer_manager module.

Tests the PeerManagerMixin for peer discovery and reputation tracking.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


class MockOrchestrator:
    """Mock orchestrator implementing PeerManagerMixin requirements."""

    def __init__(self, db_path: Path, node_id: str = "test-node"):
        self.db_path = db_path
        self.node_id = node_id
        self.bootstrap_seeds: list[str] = ["192.168.1.100:8770", "192.168.1.101:8770"]
        self.verbose = False
        self.peers: dict = {}
        self._init_db()

    def _init_db(self) -> None:
        """Initialize test database with peer_cache table."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS peer_cache (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                tailscale_ip TEXT DEFAULT '',
                reputation_score REAL DEFAULT 0.5,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_seen REAL,
                is_bootstrap_seed INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()


# Import after mock is defined to avoid circular issues
from scripts.p2p.peer_manager import PeerManagerMixin


class TestablePeerManager(PeerManagerMixin, MockOrchestrator):
    """Testable class combining mixin with mock."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, db_path: Path, node_id: str = "test-node"):
        MockOrchestrator.__init__(self, db_path, node_id)


# =============================================================================
# PeerManagerMixin Tests
# =============================================================================


class TestPeerManagerMixin:
    """Test PeerManagerMixin functionality."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create test manager with temp database."""
        db_path = tmp_path / "test.db"
        return TestablePeerManager(db_path)

    def test_save_peer_to_cache(self, manager):
        """Should save peer to cache."""
        manager._save_peer_to_cache(
            node_id="peer-1",
            host="192.168.1.50",
            port=8770,
        )

        # Verify in database
        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT host, port FROM peer_cache WHERE node_id = 'peer-1'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "192.168.1.50"
        assert row[1] == 8770

    def test_save_peer_with_tailscale(self, manager):
        """Should save peer with Tailscale IP."""
        manager._save_peer_to_cache(
            node_id="peer-2",
            host="192.168.1.51",
            port=8770,
            tailscale_ip="100.100.100.1",
        )

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT tailscale_ip FROM peer_cache WHERE node_id = 'peer-2'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "100.100.100.1"

    def test_skip_self_node(self, manager):
        """Should not save self node to cache."""
        manager._save_peer_to_cache(
            node_id="test-node",  # Same as manager.node_id
            host="192.168.1.52",
            port=8770,
        )

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM peer_cache WHERE node_id = 'test-node'")
        row = cursor.fetchone()
        conn.close()

        assert row is None

    def test_update_peer_reputation_success(self, manager):
        """Should increase reputation on success."""
        # Save initial peer
        manager._save_peer_to_cache("peer-3", "192.168.1.53", 8770)

        # Record success
        manager._update_peer_reputation("peer-3", success=True)

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT reputation_score, success_count FROM peer_cache WHERE node_id = 'peer-3'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] > 0.5  # Reputation increased
        assert row[1] == 1  # Success count

    def test_update_peer_reputation_failure(self, manager):
        """Should decrease reputation on failure."""
        # Save initial peer
        manager._save_peer_to_cache("peer-4", "192.168.1.54", 8770)

        # Record failure
        manager._update_peer_reputation("peer-4", success=False)

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT reputation_score, failure_count FROM peer_cache WHERE node_id = 'peer-4'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] < 0.5  # Reputation decreased
        assert row[1] == 1  # Failure count

    def test_get_bootstrap_peers_by_reputation(self, manager):
        """Should return peers ordered by reputation."""
        # Add peers with different reputations
        manager._save_peer_to_cache("peer-low", "192.168.1.60", 8770)
        manager._save_peer_to_cache("peer-high", "192.168.1.61", 8770)

        # Update reputations
        manager._update_peer_reputation("peer-low", success=False)
        manager._update_peer_reputation("peer-low", success=False)
        manager._update_peer_reputation("peer-high", success=True)
        manager._update_peer_reputation("peer-high", success=True)

        peers = manager._get_bootstrap_peers_by_reputation(limit=5)

        assert len(peers) == 2
        # High reputation peer should come first
        assert "192.168.1.61" in peers[0]

    def test_get_bootstrap_peers_prefers_tailscale(self, manager):
        """Should prefer Tailscale IPs in results."""
        manager._save_peer_to_cache(
            node_id="peer-ts",
            host="192.168.1.62",
            port=8770,
            tailscale_ip="100.100.100.2",
        )

        peers = manager._get_bootstrap_peers_by_reputation(limit=5)

        assert len(peers) == 1
        assert "100.100.100.2" in peers[0]

    def test_get_peer_health_score_default(self, manager):
        """Should return default health score for unknown peer."""
        score = manager._get_peer_health_score("unknown-peer")
        assert 0.0 <= score <= 1.0
        assert score == 0.5  # Default neutral score

    def test_record_p2p_sync_result(self, manager):
        """Should record sync results."""
        manager._record_p2p_sync_result("sync-peer", success=True)
        manager._record_p2p_sync_result("sync-peer", success=True)
        manager._record_p2p_sync_result("sync-peer", success=False)

        # Check results are recorded
        assert hasattr(manager, "_p2p_sync_results")
        assert "sync-peer" in manager._p2p_sync_results
        assert len(manager._p2p_sync_results["sync-peer"]) == 3

    def test_record_p2p_sync_updates_reputation(self, manager):
        """Should update reputation when recording sync."""
        manager._save_peer_to_cache("sync-peer-2", "192.168.1.70", 8770)
        manager._record_p2p_sync_result("sync-peer-2", success=True)

        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT success_count FROM peer_cache WHERE node_id = 'sync-peer-2'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] >= 1

    def test_get_cached_peer_count(self, manager):
        """Should return correct peer count."""
        assert manager._get_cached_peer_count() == 0

        manager._save_peer_to_cache("peer-a", "192.168.1.80", 8770)
        manager._save_peer_to_cache("peer-b", "192.168.1.81", 8770)

        assert manager._get_cached_peer_count() == 2

    def test_clear_peer_cache(self, manager):
        """Should clear non-seed peers from cache."""
        # Add regular peer
        manager._save_peer_to_cache("peer-regular", "192.168.1.90", 8770)
        # Add seed peer
        manager._save_peer_to_cache("peer-seed", "192.168.1.100", 8770)

        assert manager._get_cached_peer_count() == 2

        deleted = manager._clear_peer_cache()

        # Regular peer cleared, seed kept
        assert deleted >= 1
        # Seed peer (192.168.1.100:8770) should be kept
        peers = manager._get_bootstrap_peers_by_reputation(limit=10)
        assert any("192.168.1.100" in p for p in peers)

    def test_prune_stale_peers(self, manager):
        """Should prune peers not seen recently."""
        # Add peer with old last_seen
        conn = sqlite3.connect(str(manager.db_path))
        cursor = conn.cursor()
        old_time = time.time() - 100000  # Very old
        cursor.execute(
            """
            INSERT INTO peer_cache (node_id, host, port, last_seen, is_bootstrap_seed)
            VALUES ('old-peer', '192.168.1.99', 8770, ?, 0)
        """,
            (old_time,),
        )
        conn.commit()
        conn.close()

        # Add fresh peer
        manager._save_peer_to_cache("fresh-peer", "192.168.1.98", 8770)

        # Prune with 1 day max age
        pruned = manager._prune_stale_peers(max_age_seconds=86400)

        assert pruned == 1
        assert manager._get_cached_peer_count() == 1


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_peer_manager_default(self):
        """Should return None by default."""
        # Clear any existing
        from scripts.p2p import peer_manager as pm
        from scripts.p2p.peer_manager import get_peer_manager

        pm._peer_manager = None

        assert get_peer_manager() is None

    def test_set_peer_manager(self, tmp_path):
        """Should set and retrieve peer manager."""
        from scripts.p2p.peer_manager import get_peer_manager, set_peer_manager

        db_path = tmp_path / "test.db"
        manager = TestablePeerManager(db_path)
        set_peer_manager(manager)

        assert get_peer_manager() is manager

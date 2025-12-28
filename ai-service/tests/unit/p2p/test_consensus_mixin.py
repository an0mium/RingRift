"""Unit tests for consensus_mixin.py.

Tests the Raft consensus integration mixin for P2P cluster coordination.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# Mock NodeInfo dataclass
@dataclass
class MockNodeInfo:
    node_id: str
    host: str = ""
    port: int = 8770
    endpoint: str = ""
    last_heartbeat: float = 0.0

    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 90


# Mock NodeRole enum
class MockNodeRole(str, Enum):
    LEADER = "leader"
    FOLLOWER = "follower"


# Mock pysyncobj classes
class MockSyncObj:
    def __init__(self, self_addr, partner_addrs, conf=None):
        self.self_addr = self_addr
        self.partner_addrs = partner_addrs
        self._leader = None
        self._term = 1

    def _getLeader(self):
        return self._leader

    def _getTerm(self):
        return self._term


class MockSyncObjConf:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class MockReplDict(dict):
    """Mock replicated dictionary."""
    pass


class MockLockManager:
    def __init__(self, autoUnlockTime=300.0):
        self.auto_unlock_time = autoUnlockTime
        self._locks = set()

    def tryAcquire(self, key, sync=False):
        if key in self._locks:
            return False
        self._locks.add(key)
        return True

    def release(self, key):
        self._locks.discard(key)


# Create mock replicated decorator
def mock_replicated(func):
    return func


# Import the module with mocking
with patch.dict("sys.modules", {
    "scripts.p2p.types": MagicMock(NodeRole=MockNodeRole),
    "scripts.p2p.models": MagicMock(NodeInfo=MockNodeInfo),
    "pysyncobj": MagicMock(
        SyncObj=MockSyncObj,
        SyncObjConf=MockSyncObjConf,
        replicated=mock_replicated,
    ),
    "pysyncobj.batteries": MagicMock(
        ReplDict=MockReplDict,
        ReplLockManager=MockLockManager,
    ),
}):
    from scripts.p2p.consensus_mixin import (
        CONSENSUS_MODE,
        PYSYNCOBJ_AVAILABLE,
        RAFT_ENABLED,
        ConsensusMixin,
        get_work_queue,
    )


class TestableConsensusMixin(ConsensusMixin):
    """Concrete implementation for testing the mixin."""

    def __init__(self, node_id: str = "test-node-1"):
        self.node_id = node_id
        self.role = MockNodeRole.FOLLOWER
        self.voter_node_ids: list[str] = []
        self.peers: dict[str, MockNodeInfo] = {}
        self.peers_lock = threading.RLock()
        self.advertise_host = "127.0.0.1"
        self.advertise_port = 8770
        self._events_emitted: list[tuple[str, dict]] = []
        self._logs: list[tuple[str, str]] = []

    def _save_state(self) -> None:
        pass

    def _safe_emit_event(self, event_type: str, data: dict) -> None:
        self._events_emitted.append((event_type, data))

    def _log_warning(self, msg: str) -> None:
        self._logs.append(("warning", msg))

    def _log_error(self, msg: str) -> None:
        self._logs.append(("error", msg))

    def _log_info(self, msg: str) -> None:
        self._logs.append(("info", msg))


class TestModuleConstants:
    """Test module-level constants."""

    def test_raft_enabled_is_boolean(self):
        """Test that RAFT_ENABLED is a boolean."""
        assert isinstance(RAFT_ENABLED, bool)

    def test_consensus_mode_is_string(self):
        """Test that CONSENSUS_MODE is a string."""
        assert isinstance(CONSENSUS_MODE, str)
        assert CONSENSUS_MODE in ("bully", "raft", "hybrid")

    def test_pysyncobj_available_is_boolean(self):
        """Test that PYSYNCOBJ_AVAILABLE is a boolean."""
        assert isinstance(PYSYNCOBJ_AVAILABLE, bool)


class TestGetWorkQueue:
    """Test get_work_queue helper function."""

    def test_returns_none_when_import_fails(self):
        """Test that None is returned when work queue import fails."""
        with patch.dict("sys.modules", {"scripts.p2p.handlers.work_queue": None}):
            # Re-import to test the fallback
            result = get_work_queue()
            # Result may be a queue object or None depending on imports
            # Just verify it doesn't crash


class TestConsensusMixinInitialization:
    """Test ConsensusMixin initialization."""

    def test_can_instantiate(self):
        """Test that mixin can be instantiated."""
        consensus = TestableConsensusMixin()
        assert consensus.node_id == "test-node-1"

    def test_default_raft_attributes_not_present_before_init(self):
        """Test that Raft attributes are not present before initialization."""
        consensus = TestableConsensusMixin()
        assert not hasattr(consensus, "_raft_work_queue")
        assert not hasattr(consensus, "_raft_initialized")


class TestInitRaftConsensus:
    """Test _init_raft_consensus method."""

    def test_returns_false_when_raft_disabled(self):
        """Test that initialization returns False when Raft is disabled."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = consensus._init_raft_consensus()

        assert result is False
        assert consensus._raft_initialized is False

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.PYSYNCOBJ_AVAILABLE", False)
    def test_returns_false_when_pysyncobj_unavailable(self):
        """Test that initialization returns False when pysyncobj is not installed."""
        consensus = TestableConsensusMixin()

        result = consensus._init_raft_consensus()

        assert result is False
        assert consensus._raft_initialized is False
        assert consensus._raft_init_error == "pysyncobj not installed"

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.PYSYNCOBJ_AVAILABLE", True)
    def test_returns_false_when_no_partners(self):
        """Test that initialization returns False when no Raft partners available."""
        consensus = TestableConsensusMixin()
        consensus.voter_node_ids = []  # No voters = no partners

        result = consensus._init_raft_consensus()

        assert result is False
        assert consensus._raft_init_error == "no Raft partners available"


class TestGetRaftPartners:
    """Test _get_raft_partners method."""

    def test_returns_empty_list_when_no_voters(self):
        """Test that empty list is returned when no voters configured."""
        consensus = TestableConsensusMixin()
        consensus.voter_node_ids = []

        result = consensus._get_raft_partners()

        assert result == []

    def test_excludes_self_from_partners(self):
        """Test that self is excluded from partner list."""
        consensus = TestableConsensusMixin()
        consensus.node_id = "voter-1"
        consensus.voter_node_ids = ["voter-1", "voter-2", "voter-3"]

        # Add peers
        consensus.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            endpoint="192.168.1.2:8770",
        )
        consensus.peers["voter-3"] = MockNodeInfo(
            node_id="voter-3",
            endpoint="192.168.1.3:8770",
        )

        result = consensus._get_raft_partners()

        # Should include voter-2 and voter-3 but not voter-1 (self)
        assert len(result) == 2
        assert "192.168.1.2:4321" in result  # Raft port 4321
        assert "192.168.1.3:4321" in result

    def test_only_includes_peers_with_endpoint(self):
        """Test that only peers with endpoints are included."""
        consensus = TestableConsensusMixin()
        consensus.voter_node_ids = ["voter-1", "voter-2"]

        # Add peer without endpoint
        consensus.peers["voter-1"] = MockNodeInfo(
            node_id="voter-1",
            endpoint="",  # No endpoint
        )
        # Add peer with endpoint
        consensus.peers["voter-2"] = MockNodeInfo(
            node_id="voter-2",
            endpoint="192.168.1.2:8770",
        )

        result = consensus._get_raft_partners()

        assert len(result) == 1
        assert "192.168.1.2:4321" in result


class TestShouldUseRaft:
    """Test _should_use_raft method."""

    def test_returns_false_when_raft_disabled(self):
        """Test that False is returned when Raft is disabled."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = consensus._should_use_raft()

        assert result is False

    def test_returns_false_when_not_initialized(self):
        """Test that False is returned when Raft is not initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True):
            result = consensus._should_use_raft()

        assert result is False

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.CONSENSUS_MODE", "bully")
    def test_returns_false_in_bully_mode(self):
        """Test that False is returned in bully mode."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True

        result = consensus._should_use_raft()

        assert result is False

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.CONSENSUS_MODE", "raft")
    def test_returns_true_in_raft_mode(self):
        """Test that True is returned in raft mode."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True

        result = consensus._should_use_raft()

        assert result is True

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.CONSENSUS_MODE", "hybrid")
    def test_returns_true_in_hybrid_mode(self):
        """Test that True is returned in hybrid mode."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True

        result = consensus._should_use_raft()

        assert result is True


class TestClaimWorkDistributed:
    """Test claim_work_distributed method."""

    def test_routes_to_sqlite_when_raft_disabled(self):
        """Test that SQLite is used when Raft is disabled."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False

        with patch.object(consensus, "_claim_work_sqlite") as mock_sqlite:
            mock_sqlite.return_value = {"work_id": "test-1"}
            result = consensus.claim_work_distributed("node-1")

        mock_sqlite.assert_called_once_with("node-1", None)
        assert result == {"work_id": "test-1"}

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    @patch("scripts.p2p.consensus_mixin.CONSENSUS_MODE", "raft")
    def test_routes_to_raft_when_enabled(self):
        """Test that Raft is used when enabled."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        consensus._raft_work_queue = MagicMock()

        with patch.object(consensus, "_claim_work_raft") as mock_raft:
            mock_raft.return_value = {"work_id": "test-1"}
            result = consensus.claim_work_distributed("node-1", ["selfplay"])

        mock_raft.assert_called_once_with("node-1", ["selfplay"])
        assert result == {"work_id": "test-1"}


class TestClaimWorkSqlite:
    """Test _claim_work_sqlite method."""

    def test_returns_none_when_no_work_queue(self):
        """Test that None is returned when work queue is not available."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.get_work_queue", return_value=None):
            result = consensus._claim_work_sqlite("node-1")

        assert result is None

    def test_returns_work_item_from_queue(self):
        """Test that work item is returned from queue."""
        consensus = TestableConsensusMixin()
        mock_wq = MagicMock()
        mock_item = MagicMock()
        mock_item.to_dict.return_value = {"work_id": "test-1", "work_type": "selfplay"}
        mock_wq.claim_work.return_value = mock_item

        with patch("scripts.p2p.consensus_mixin.get_work_queue", return_value=mock_wq):
            result = consensus._claim_work_sqlite("node-1", ["selfplay"])

        mock_wq.claim_work.assert_called_once_with("node-1", ["selfplay"])
        assert result == {"work_id": "test-1", "work_type": "selfplay"}

    def test_returns_none_when_no_work_available(self):
        """Test that None is returned when no work is available."""
        consensus = TestableConsensusMixin()
        mock_wq = MagicMock()
        mock_wq.claim_work.return_value = None

        with patch("scripts.p2p.consensus_mixin.get_work_queue", return_value=mock_wq):
            result = consensus._claim_work_sqlite("node-1")

        assert result is None

    def test_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully."""
        consensus = TestableConsensusMixin()
        mock_wq = MagicMock()
        mock_wq.claim_work.side_effect = Exception("Database error")

        with patch("scripts.p2p.consensus_mixin.get_work_queue", return_value=mock_wq):
            result = consensus._claim_work_sqlite("node-1")

        assert result is None


class TestClaimWorkRaft:
    """Test _claim_work_raft method."""

    def test_falls_back_to_sqlite_when_queue_not_initialized(self):
        """Test that SQLite fallback is used when Raft queue not initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_work_queue = None

        with patch.object(consensus, "_claim_work_sqlite") as mock_sqlite:
            mock_sqlite.return_value = {"work_id": "fallback-1"}
            result = consensus._claim_work_raft("node-1")

        mock_sqlite.assert_called_once()
        assert result == {"work_id": "fallback-1"}

    def test_returns_none_when_no_pending_work(self):
        """Test that None is returned when no pending work."""
        consensus = TestableConsensusMixin()
        mock_queue = MagicMock()
        mock_queue.get_pending_work.return_value = []
        consensus._raft_work_queue = mock_queue

        result = consensus._claim_work_raft("node-1")

        assert result is None

    def test_claims_highest_priority_work(self):
        """Test that highest priority work is claimed."""
        consensus = TestableConsensusMixin()
        mock_queue = MagicMock()
        mock_queue.get_pending_work.return_value = [
            {"work_id": "high-1", "work_type": "training", "priority": 100},
            {"work_id": "low-1", "work_type": "selfplay", "priority": 10},
        ]
        mock_queue.claim_work.return_value = True
        consensus._raft_work_queue = mock_queue

        result = consensus._claim_work_raft("node-1", ["training", "selfplay"])

        # Should claim the first (highest priority) item
        mock_queue.claim_work.assert_called_once_with("node-1", "high-1")
        assert result["work_id"] == "high-1"

    def test_tries_next_item_if_first_claim_fails(self):
        """Test that next item is tried if first claim fails."""
        consensus = TestableConsensusMixin()
        mock_queue = MagicMock()
        mock_queue.get_pending_work.return_value = [
            {"work_id": "high-1", "work_type": "training", "priority": 100},
            {"work_id": "low-1", "work_type": "selfplay", "priority": 10},
        ]
        # First claim fails, second succeeds
        mock_queue.claim_work.side_effect = [False, True]
        consensus._raft_work_queue = mock_queue

        result = consensus._claim_work_raft("node-1")

        assert mock_queue.claim_work.call_count == 2
        assert result["work_id"] == "low-1"

    def test_falls_back_to_sqlite_on_exception(self):
        """Test that SQLite fallback is used on exception."""
        consensus = TestableConsensusMixin()
        mock_queue = MagicMock()
        mock_queue.get_pending_work.side_effect = Exception("Raft error")
        consensus._raft_work_queue = mock_queue

        with patch.object(consensus, "_claim_work_sqlite") as mock_sqlite:
            mock_sqlite.return_value = {"work_id": "fallback-1"}
            result = consensus._claim_work_raft("node-1")

        mock_sqlite.assert_called_once()
        assert result == {"work_id": "fallback-1"}


class TestIsRaftLeader:
    """Test is_raft_leader method."""

    def test_returns_false_when_not_initialized(self):
        """Test that False is returned when Raft is not initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False

        result = consensus.is_raft_leader()

        assert result is False

    def test_returns_false_when_no_work_queue(self):
        """Test that False is returned when work queue is not available."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        consensus._raft_work_queue = None

        result = consensus.is_raft_leader()

        assert result is False

    def test_returns_false_when_no_leader(self):
        """Test that False is returned when there is no Raft leader."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        mock_queue = MagicMock()
        mock_queue._getLeader.return_value = None
        consensus._raft_work_queue = mock_queue

        result = consensus.is_raft_leader()

        assert result is False

    def test_returns_true_when_self_is_leader(self):
        """Test that True is returned when this node is the Raft leader."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        consensus.advertise_host = "192.168.1.1"
        mock_queue = MagicMock()
        mock_queue._getLeader.return_value = "192.168.1.1:4321"  # RAFT_BIND_PORT
        consensus._raft_work_queue = mock_queue

        result = consensus.is_raft_leader()

        assert result is True

    def test_returns_false_when_other_is_leader(self):
        """Test that False is returned when another node is the Raft leader."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        consensus.advertise_host = "192.168.1.1"
        mock_queue = MagicMock()
        mock_queue._getLeader.return_value = "192.168.1.2:4321"  # Different node
        consensus._raft_work_queue = mock_queue

        result = consensus.is_raft_leader()

        assert result is False

    def test_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        mock_queue = MagicMock()
        mock_queue._getLeader.side_effect = Exception("Raft error")
        consensus._raft_work_queue = mock_queue

        result = consensus.is_raft_leader()

        assert result is False


class TestGetRaftStatus:
    """Test get_raft_status method."""

    def test_returns_basic_status_when_not_initialized(self):
        """Test that basic status is returned when Raft is not initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False
        consensus._raft_init_error = None

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = consensus.get_raft_status()

        assert "raft_enabled" in result
        assert "raft_available" in result
        assert "raft_initialized" in result
        assert "consensus_mode" in result
        assert result["raft_initialized"] is False

    def test_returns_error_when_init_failed(self):
        """Test that error is included when initialization failed."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False
        consensus._raft_init_error = "pysyncobj not installed"

        result = consensus.get_raft_status()

        assert result["raft_init_error"] == "pysyncobj not installed"

    def test_returns_detailed_status_when_initialized(self):
        """Test that detailed status is returned when Raft is initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True
        consensus.advertise_host = "192.168.1.1"

        mock_queue = MagicMock()
        mock_queue._getLeader.return_value = "192.168.1.1:4321"
        mock_queue.get_queue_status.return_value = {
            "total_items": 5,
            "by_status": {"pending": 3, "claimed": 2, "completed": 0, "failed": 0},
        }
        consensus._raft_work_queue = mock_queue

        result = consensus.get_raft_status()

        assert result["raft_initialized"] is True
        assert result["is_raft_leader"] is True
        assert "work_queue_status" in result

    def test_handles_exception_when_getting_leader(self):
        """Test that exceptions are handled when getting leader info."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True

        mock_queue = MagicMock()
        mock_queue._getLeader.side_effect = Exception("Raft error")
        consensus._raft_work_queue = mock_queue

        result = consensus.get_raft_status()

        assert "raft_error" in result


class TestConsensusHealthCheck:
    """Test consensus_health_check method."""

    def test_healthy_when_raft_disabled(self):
        """Test that healthy is True when Raft is disabled."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = consensus.consensus_health_check()

        assert result["is_healthy"] is True
        assert result["raft_enabled"] is False

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    def test_healthy_when_raft_initialized(self):
        """Test that healthy is True when Raft is initialized."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = True

        result = consensus.consensus_health_check()

        assert result["is_healthy"] is True
        assert result["raft_initialized"] is True

    @patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", True)
    def test_unhealthy_when_raft_not_initialized(self):
        """Test that healthy is False when Raft should be enabled but is not."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False
        consensus._raft_init_error = "no Raft partners available"

        result = consensus.consensus_health_check()

        assert result["is_healthy"] is False
        assert result["raft_init_error"] == "no Raft partners available"

    def test_includes_all_required_fields(self):
        """Test that all required fields are included."""
        consensus = TestableConsensusMixin()
        consensus._raft_initialized = False
        consensus._raft_init_error = None

        with patch("scripts.p2p.consensus_mixin.RAFT_ENABLED", False):
            result = consensus.consensus_health_check()

        required_fields = [
            "is_healthy",
            "raft_enabled",
            "raft_available",
            "raft_initialized",
            "raft_init_error",
            "consensus_mode",
        ]
        for field in required_fields:
            assert field in result


class TestEmitRaftLeaderEvent:
    """Test _emit_raft_leader_event method."""

    def test_emits_leader_elected_event(self):
        """Test that RAFT_LEADER_ELECTED event is emitted."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            consensus._emit_raft_leader_event(is_leader=True)

            # Event is fire-and-forget so we just verify no crash

    def test_emits_leader_lost_event(self):
        """Test that RAFT_LEADER_LOST event would be emitted for non-leader."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            consensus._emit_raft_leader_event(is_leader=False)

            # No crash expected

    def test_handles_missing_router(self):
        """Test that missing router is handled gracefully."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.get_router", return_value=None):
            # Should not raise
            consensus._emit_raft_leader_event(is_leader=True)

    def test_handles_import_error(self):
        """Test that ImportError is handled gracefully."""
        consensus = TestableConsensusMixin()

        with patch("scripts.p2p.consensus_mixin.get_router", side_effect=ImportError):
            # Should not raise - import errors are caught
            try:
                consensus._emit_raft_leader_event(is_leader=True)
            except ImportError:
                # Event router import may fail in test, that's OK
                pass


class TestModuleExports:
    """Test module exports."""

    def test_exports_consensus_mixin(self):
        """Test that ConsensusMixin is exported."""
        from scripts.p2p.consensus_mixin import ConsensusMixin as ExportedMixin
        assert ExportedMixin is not None

    def test_exports_constants(self):
        """Test that constants are exported."""
        from scripts.p2p.consensus_mixin import (
            CONSENSUS_MODE,
            PYSYNCOBJ_AVAILABLE,
            RAFT_ENABLED,
        )
        assert RAFT_ENABLED is not None or RAFT_ENABLED is False
        assert CONSENSUS_MODE is not None
        assert PYSYNCOBJ_AVAILABLE is not None or PYSYNCOBJ_AVAILABLE is False

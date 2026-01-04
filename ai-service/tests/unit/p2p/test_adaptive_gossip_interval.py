"""Tests for adaptive gossip interval functionality.

Sprint 13 (Jan 3, 2026): Tests for dynamic gossip interval adjustment based on
partition status. The interval adapts to cluster health:
- Partition (isolated/minority): 5s for fast recovery
- Recovery (healthy but recently partitioned): 10s during stabilization
- Stable (consistently healthy): 30s for normal operation
"""

import pytest
import time
from unittest.mock import patch, MagicMock, PropertyMock


class MockPeer:
    """Mock peer for testing."""

    def __init__(self, node_id: str, alive: bool = True, retired: bool = False, long_dead: bool = False):
        self.node_id = node_id
        self._alive = alive
        self.retired = retired
        # Partition detection excludes peers dead for >300s ("long dead")
        # For testing partition scenarios, use recently dead peers (100s ago)
        # unless explicitly marked as long_dead
        if alive:
            self.last_heartbeat = time.time()
        elif long_dead:
            self.last_heartbeat = time.time() - 400  # >300s = long dead
        else:
            self.last_heartbeat = time.time() - 100  # <300s = recently dead

    def is_alive(self) -> bool:
        return self._alive


class TestAdaptiveGossipInterval:
    """Tests for adaptive gossip interval methods."""

    @pytest.fixture
    def mixin_class(self):
        """Create a class that includes GossipProtocolMixin for testing."""
        from scripts.p2p.gossip_protocol import GossipProtocolMixin
        import threading

        # GossipProtocolMixin already inherits from P2PMixinBase, so we only
        # need to inherit from it (not both)
        class TestMixin(GossipProtocolMixin):
            MIXIN_TYPE = "test_mixin"

            def __init__(self):
                self.peers = {}
                self.peers_lock = threading.RLock()
                self._init_gossip_protocol()

        return TestMixin

    @pytest.fixture
    def mixin(self, mixin_class):
        """Create an instance of the test mixin."""
        return mixin_class()

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_adaptive_interval_constants_exist(self, mixin):
        """Test that adaptive interval constants are defined."""
        assert hasattr(mixin, "GOSSIP_INTERVAL_PARTITION")
        assert hasattr(mixin, "GOSSIP_INTERVAL_RECOVERY")
        assert hasattr(mixin, "GOSSIP_INTERVAL_STABLE")
        assert hasattr(mixin, "GOSSIP_STABILITY_THRESHOLD")

    def test_adaptive_interval_default_values(self, mixin):
        """Test default values for adaptive intervals."""
        assert mixin.GOSSIP_INTERVAL_PARTITION == 5.0
        assert mixin.GOSSIP_INTERVAL_RECOVERY == 10.0
        assert mixin.GOSSIP_INTERVAL_STABLE == 30.0
        assert mixin.GOSSIP_STABILITY_THRESHOLD == 5

    def test_adaptive_state_initialized(self, mixin):
        """Test that adaptive state variables are initialized."""
        assert hasattr(mixin, "_gossip_consecutive_healthy")
        assert hasattr(mixin, "_gossip_last_partition_status")
        assert hasattr(mixin, "_gossip_adaptive_interval")

    def test_adaptive_interval_starts_at_recovery(self, mixin):
        """Test that interval starts at recovery level."""
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_RECOVERY

    # =========================================================================
    # Partition Detection Integration Tests
    # =========================================================================

    def test_isolated_uses_partition_interval(self, mixin):
        """Test that isolated status uses fast partition interval."""
        # No peers = isolated
        mixin.peers = {}

        interval = mixin._update_adaptive_gossip_interval()

        assert interval == mixin.GOSSIP_INTERVAL_PARTITION
        assert mixin._gossip_consecutive_healthy == 0

    def test_minority_uses_partition_interval(self, mixin):
        """Test that minority status uses fast partition interval."""
        # 3 alive, 7 dead = 30% = minority (between 20% and 50%)
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
            "peer3": MockPeer("peer3", alive=True),
            "peer4": MockPeer("peer4", alive=False),
            "peer5": MockPeer("peer5", alive=False),
            "peer6": MockPeer("peer6", alive=False),
            "peer7": MockPeer("peer7", alive=False),
            "peer8": MockPeer("peer8", alive=False),
            "peer9": MockPeer("peer9", alive=False),
            "peer10": MockPeer("peer10", alive=False),
        }

        interval = mixin._update_adaptive_gossip_interval()

        assert interval == mixin.GOSSIP_INTERVAL_PARTITION
        assert mixin._gossip_consecutive_healthy == 0

    def test_healthy_uses_recovery_interval_initially(self, mixin):
        """Test that first healthy check uses recovery interval."""
        # All alive = healthy
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
            "peer3": MockPeer("peer3", alive=True),
        }

        interval = mixin._update_adaptive_gossip_interval()

        assert interval == mixin.GOSSIP_INTERVAL_RECOVERY
        assert mixin._gossip_consecutive_healthy == 1

    def test_healthy_increments_consecutive_count(self, mixin):
        """Test that consecutive healthy checks are tracked."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # Multiple healthy checks
        for i in range(3):
            mixin._update_adaptive_gossip_interval()
            assert mixin._gossip_consecutive_healthy == i + 1

    def test_stable_after_threshold_reached(self, mixin):
        """Test that stable interval is used after threshold consecutive healthy checks."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # Run until threshold is reached
        for _ in range(mixin.GOSSIP_STABILITY_THRESHOLD):
            interval = mixin._update_adaptive_gossip_interval()

        assert interval == mixin.GOSSIP_INTERVAL_STABLE
        assert mixin._gossip_consecutive_healthy >= mixin.GOSSIP_STABILITY_THRESHOLD

    def test_partition_resets_consecutive_count(self, mixin):
        """Test that partition detection resets the consecutive healthy count."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # Build up some healthy checks
        for _ in range(3):
            mixin._update_adaptive_gossip_interval()

        assert mixin._gossip_consecutive_healthy == 3

        # Now simulate partition
        mixin.peers = {}
        mixin._update_adaptive_gossip_interval()

        assert mixin._gossip_consecutive_healthy == 0
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_PARTITION

    # =========================================================================
    # Status Tracking Tests
    # =========================================================================

    def test_status_transitions_tracked(self, mixin):
        """Test that partition status transitions are tracked."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_last_partition_status == "healthy"

        # Simulate partition
        mixin.peers = {}
        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_last_partition_status == "isolated"

    def test_get_adaptive_gossip_interval_readonly(self, mixin):
        """Test that get_adaptive_gossip_interval doesn't modify state."""
        mixin._gossip_adaptive_interval = 15.0
        mixin._gossip_consecutive_healthy = 3

        interval = mixin.get_adaptive_gossip_interval()

        assert interval == 15.0
        assert mixin._gossip_consecutive_healthy == 3  # Unchanged

    def test_get_adaptive_gossip_status(self, mixin):
        """Test the status endpoint returns correct information."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }
        mixin._gossip_consecutive_healthy = 3
        mixin._gossip_adaptive_interval = mixin.GOSSIP_INTERVAL_RECOVERY

        status = mixin.get_adaptive_gossip_status()

        assert status["current_interval"] == mixin.GOSSIP_INTERVAL_RECOVERY
        assert status["partition_status"] == "healthy"
        assert status["consecutive_healthy"] == 3
        assert status["stability_threshold"] == mixin.GOSSIP_STABILITY_THRESHOLD
        assert status["is_stable"] is False  # 3 < 5

    def test_is_stable_in_status(self, mixin):
        """Test that is_stable flag is correct in status."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # Not stable yet
        mixin._gossip_consecutive_healthy = 4
        status = mixin.get_adaptive_gossip_status()
        assert status["is_stable"] is False

        # Now stable
        mixin._gossip_consecutive_healthy = 5
        status = mixin.get_adaptive_gossip_status()
        assert status["is_stable"] is True

    # =========================================================================
    # Recovery Scenario Tests
    # =========================================================================

    def test_recovery_sequence_from_partition(self, mixin):
        """Test the full recovery sequence from partition to stable."""
        # Start with partition
        mixin.peers = {}
        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_PARTITION

        # Network recovers
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # First healthy check - recovery interval
        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_RECOVERY

        # Continue until stable
        for _ in range(mixin.GOSSIP_STABILITY_THRESHOLD - 1):
            mixin._update_adaptive_gossip_interval()

        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_STABLE

    def test_brief_partition_resets_stability(self, mixin):
        """Test that a brief partition resets stability counter."""
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }

        # Build stability
        for _ in range(mixin.GOSSIP_STABILITY_THRESHOLD):
            mixin._update_adaptive_gossip_interval()

        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_STABLE

        # Brief partition
        mixin.peers = {}
        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_PARTITION
        assert mixin._gossip_consecutive_healthy == 0

        # Recover
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
        }
        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_RECOVERY
        assert mixin._gossip_consecutive_healthy == 1

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_retired_peers_excluded(self, mixin):
        """Test that retired peers don't affect partition detection."""
        # 2 alive, 3 retired (should be healthy, not minority)
        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
            "peer2": MockPeer("peer2", alive=True),
            "peer3": MockPeer("peer3", alive=False, retired=True),
            "peer4": MockPeer("peer4", alive=False, retired=True),
            "peer5": MockPeer("peer5", alive=False, retired=True),
        }

        mixin._update_adaptive_gossip_interval()
        assert mixin._gossip_last_partition_status == "healthy"

    def test_interval_doesnt_decrease_when_already_at_minimum(self, mixin):
        """Test that interval stays at partition level when already partitioned."""
        mixin.peers = {}
        mixin._gossip_adaptive_interval = mixin.GOSSIP_INTERVAL_PARTITION

        mixin._update_adaptive_gossip_interval()

        assert mixin._gossip_adaptive_interval == mixin.GOSSIP_INTERVAL_PARTITION

    def test_fallback_when_state_not_initialized(self, mixin):
        """Test fallback behavior when state attributes missing."""
        # Remove state attributes
        if hasattr(mixin, "_gossip_last_partition_status"):
            delattr(mixin, "_gossip_last_partition_status")
        if hasattr(mixin, "_gossip_consecutive_healthy"):
            delattr(mixin, "_gossip_consecutive_healthy")

        mixin.peers = {
            "peer1": MockPeer("peer1", alive=True),
        }

        # Should not raise, should initialize state
        interval = mixin._update_adaptive_gossip_interval()
        assert interval is not None


class TestAdaptiveGossipEnvironmentOverrides:
    """Tests for environment variable overrides of adaptive gossip intervals."""

    def test_partition_interval_from_env(self):
        """Test that partition interval can be overridden via environment."""
        import os

        with patch.dict(os.environ, {"RINGRIFT_GOSSIP_INTERVAL_PARTITION": "3.0"}):
            # Need to reload the module to pick up env var
            from scripts.p2p import gossip_protocol
            import importlib

            importlib.reload(gossip_protocol)

            assert gossip_protocol.GossipProtocolMixin.GOSSIP_INTERVAL_PARTITION == 3.0

            # Reset
            with patch.dict(os.environ, {"RINGRIFT_GOSSIP_INTERVAL_PARTITION": "5.0"}):
                importlib.reload(gossip_protocol)

    def test_stability_threshold_from_env(self):
        """Test that stability threshold can be overridden via environment."""
        import os

        with patch.dict(os.environ, {"RINGRIFT_GOSSIP_STABILITY_THRESHOLD": "10"}):
            from scripts.p2p import gossip_protocol
            import importlib

            importlib.reload(gossip_protocol)

            assert gossip_protocol.GossipProtocolMixin.GOSSIP_STABILITY_THRESHOLD == 10

            # Reset
            with patch.dict(os.environ, {"RINGRIFT_GOSSIP_STABILITY_THRESHOLD": "5"}):
                importlib.reload(gossip_protocol)

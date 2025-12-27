"""Tests for time-since-sync priority weighting in SyncRouter.

December 2025: Added as part of sync router priority improvements.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.coordination.sync_router import SyncRouter, NodeSyncCapability
from app.distributed.cluster_manifest import DataType


class TestNodeSyncCapabilityLastSyncTime:
    """Tests for last_sync_time field in NodeSyncCapability."""

    def test_default_last_sync_time_is_zero(self):
        """Test that last_sync_time defaults to 0.0."""
        cap = NodeSyncCapability(node_id="test-node")
        assert cap.last_sync_time == 0.0

    def test_last_sync_time_can_be_set(self):
        """Test that last_sync_time can be set explicitly."""
        cap = NodeSyncCapability(node_id="test-node", last_sync_time=1234567890.0)
        assert cap.last_sync_time == 1234567890.0


class TestSyncRouterTimeWeight:
    """Tests for time-based priority weighting in _compute_target_priority."""

    @pytest.fixture
    def router_with_mock_config(self):
        """Create a SyncRouter with mocked configuration."""
        with patch.object(SyncRouter, '_load_config'):
            with patch.object(SyncRouter, '_load_sync_timestamps'):
                router = SyncRouter()
                router._node_capabilities = {}
                return router

    def test_never_synced_node_gets_max_weight(self, router_with_mock_config):
        """Nodes that never synced (last_sync_time=0) should get +30 priority."""
        router = router_with_mock_config
        cap = NodeSyncCapability(node_id="new-node", last_sync_time=0.0)

        priority = router._compute_target_priority(cap, DataType.GAME)

        # Base priority (50) + never synced bonus (30) = 80
        assert priority >= 80

    def test_recently_synced_node_gets_low_weight(self, router_with_mock_config):
        """Nodes synced recently should get minimal time boost."""
        router = router_with_mock_config
        # Synced 1 minute ago
        cap = NodeSyncCapability(node_id="recent-node", last_sync_time=time.time() - 60)

        priority = router._compute_target_priority(cap, DataType.GAME)

        # 1 minute = 60 seconds, time_weight = (60/3600) * 30 = 0.5
        # Base (50) + disk bonus (10 for <50% usage) + time (~0) = ~60
        assert priority < 65  # Should be close to base priority with disk bonus

    def test_30_minutes_node_gets_medium_weight(self, router_with_mock_config):
        """Nodes synced 30 minutes ago should get ~15 point boost."""
        router = router_with_mock_config
        # Synced 30 minutes ago
        cap = NodeSyncCapability(node_id="medium-node", last_sync_time=time.time() - 1800)

        priority = router._compute_target_priority(cap, DataType.GAME)

        # 30 min = 1800 seconds, time_weight = (1800/3600) * 30 = 15
        # Base (50) + disk bonus (10) + time (15) = 75
        assert 70 <= priority <= 80

    def test_stale_sync_node_gets_high_weight(self, router_with_mock_config):
        """Nodes synced 1+ hour ago should get ~30 points."""
        router = router_with_mock_config
        # Synced 1 hour ago
        cap = NodeSyncCapability(node_id="stale-node", last_sync_time=time.time() - 3600)

        priority = router._compute_target_priority(cap, DataType.GAME)

        # 1 hour = 3600 seconds, time_weight = (3600/3600) * 30 = 30
        # So priority should be base (50) + 30 = 80
        assert priority >= 75

    def test_time_weight_caps_at_30(self, router_with_mock_config):
        """Time weight should cap at 30 even for very stale nodes."""
        router = router_with_mock_config
        # Synced 24 hours ago
        cap = NodeSyncCapability(node_id="ancient-node", last_sync_time=time.time() - 86400)

        priority = router._compute_target_priority(cap, DataType.GAME)

        # Even with 24 hours, time_weight should cap at 30
        # Base (50) + disk bonus (10) + time (30 capped) = 90
        assert priority <= 95  # Should not exceed base + disk + max_time_weight

    def test_time_weight_combines_with_other_factors(self, router_with_mock_config):
        """Time weight should stack with training node and priority node bonuses."""
        router = router_with_mock_config
        cap = NodeSyncCapability(
            node_id="important-node",
            is_training_node=True,  # +30
            is_priority_node=True,  # +20
            last_sync_time=0.0,  # +30 (never synced)
        )

        priority = router._compute_target_priority(cap, DataType.GAME)

        # Base (50) + training (30) + priority (20) + time (30) = 130
        assert priority >= 120


class TestSyncRouterTimestampPersistence:
    """Tests for sync timestamp persistence."""

    @pytest.fixture
    def temp_state_file(self, tmp_path):
        """Create a temporary state file path."""
        return tmp_path / "test_sync_timestamps.json"

    def test_save_sync_timestamps(self, temp_state_file):
        """Test that _save_sync_timestamps writes correct JSON."""
        with patch.object(SyncRouter, '_load_config'):
            with patch.object(SyncRouter, '_load_sync_timestamps'):
                router = SyncRouter()
                router._SYNC_STATE_FILE = temp_state_file
                router._node_capabilities = {
                    "node-1": NodeSyncCapability(node_id="node-1", last_sync_time=1000.0),
                    "node-2": NodeSyncCapability(node_id="node-2", last_sync_time=2000.0),
                    "node-3": NodeSyncCapability(node_id="node-3", last_sync_time=0.0),  # Should be excluded
                }

                router._save_sync_timestamps()

                assert temp_state_file.exists()
                with open(temp_state_file) as f:
                    data = json.load(f)

                assert data["node-1"] == 1000.0
                assert data["node-2"] == 2000.0
                assert "node-3" not in data  # Zero timestamps excluded

    def test_load_sync_timestamps(self, temp_state_file):
        """Test that _load_sync_timestamps correctly loads state."""
        # Create state file
        temp_state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_file, "w") as f:
            json.dump({"node-1": 1000.0, "node-2": 2000.0}, f)

        with patch.object(SyncRouter, '_load_config'):
            router = SyncRouter.__new__(SyncRouter)
            router._SYNC_STATE_FILE = temp_state_file
            router._node_capabilities = {
                "node-1": NodeSyncCapability(node_id="node-1"),
                "node-2": NodeSyncCapability(node_id="node-2"),
                "node-3": NodeSyncCapability(node_id="node-3"),
            }

            router._load_sync_timestamps()

            assert router._node_capabilities["node-1"].last_sync_time == 1000.0
            assert router._node_capabilities["node-2"].last_sync_time == 2000.0
            assert router._node_capabilities["node-3"].last_sync_time == 0.0  # Unknown node unchanged

    def test_load_sync_timestamps_handles_missing_file(self, temp_state_file):
        """Test that missing state file is handled gracefully."""
        with patch.object(SyncRouter, '_load_config'):
            router = SyncRouter.__new__(SyncRouter)
            router._SYNC_STATE_FILE = temp_state_file  # Does not exist
            router._node_capabilities = {
                "node-1": NodeSyncCapability(node_id="node-1"),
            }

            # Should not raise
            router._load_sync_timestamps()

            # Timestamp should remain at default
            assert router._node_capabilities["node-1"].last_sync_time == 0.0

    def test_record_sync_success_updates_timestamp(self, temp_state_file):
        """Test that record_sync_success updates timestamp and persists."""
        with patch.object(SyncRouter, '_load_config'):
            with patch.object(SyncRouter, '_load_sync_timestamps'):
                router = SyncRouter()
                router._SYNC_STATE_FILE = temp_state_file
                router._node_capabilities = {
                    "node-1": NodeSyncCapability(node_id="node-1", last_sync_time=0.0),
                }

                before = time.time()
                router.record_sync_success("node-1")
                after = time.time()

                # Check timestamp was updated
                assert before <= router._node_capabilities["node-1"].last_sync_time <= after

                # Check file was written
                assert temp_state_file.exists()

    def test_record_sync_success_ignores_unknown_node(self, temp_state_file):
        """Test that record_sync_success ignores unknown nodes."""
        with patch.object(SyncRouter, '_load_config'):
            with patch.object(SyncRouter, '_load_sync_timestamps'):
                router = SyncRouter()
                router._SYNC_STATE_FILE = temp_state_file
                router._node_capabilities = {}

                # Should not raise
                router.record_sync_success("unknown-node")

                # File should not be created for unknown node
                assert not temp_state_file.exists()

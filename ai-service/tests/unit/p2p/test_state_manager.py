"""Unit tests for StateManager (P2P orchestrator state persistence).

December 29, 2025: Initial test coverage for P2P managers.
"""

import time
from unittest import TestCase
from scripts.p2p.managers.state_manager import PeerHealthState


class TestPeerHealthState(TestCase):
    """Tests for PeerHealthState dataclass."""

    def test_default_values(self):
        """Test that PeerHealthState has sensible defaults."""
        state = PeerHealthState(node_id="node-1", state="alive")
        self.assertEqual(state.node_id, "node-1")
        self.assertEqual(state.state, "alive")
        self.assertEqual(state.failure_count, 0)
        self.assertEqual(state.gossip_failure_count, 0)
        self.assertEqual(state.circuit_state, "closed")

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = PeerHealthState(
            node_id="node-1",
            state="suspect",
            failure_count=2,
            last_seen=1000.0,
        )
        d = state.to_dict()
        self.assertEqual(d["node_id"], "node-1")
        self.assertEqual(d["state"], "suspect")
        self.assertEqual(d["failure_count"], 2)
        self.assertEqual(d["last_seen"], 1000.0)

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "node_id": "node-2",
            "state": "dead",
            "failure_count": 5,
            "gossip_failure_count": 3,
            "last_seen": 500.0,
            "last_failure": 600.0,
            "circuit_state": "open",
            "circuit_opened_at": 550.0,
            "updated_at": 700.0,
        }
        state = PeerHealthState.from_dict(d)
        self.assertEqual(state.node_id, "node-2")
        self.assertEqual(state.state, "dead")
        self.assertEqual(state.failure_count, 5)
        self.assertEqual(state.gossip_failure_count, 3)
        self.assertEqual(state.circuit_state, "open")

    def test_from_dict_defaults(self):
        """Test from_dict handles missing keys gracefully."""
        state = PeerHealthState.from_dict({"node_id": "node-3"})
        self.assertEqual(state.node_id, "node-3")
        self.assertEqual(state.state, "alive")  # Default
        self.assertEqual(state.failure_count, 0)  # Default
        self.assertEqual(state.circuit_state, "closed")  # Default

    def test_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original = PeerHealthState(
            node_id="roundtrip-node",
            state="suspect",
            failure_count=3,
            gossip_failure_count=1,
            last_seen=1234.5,
            last_failure=1230.0,
            circuit_state="half_open",
            circuit_opened_at=1200.0,
            updated_at=1235.0,
        )
        restored = PeerHealthState.from_dict(original.to_dict())
        self.assertEqual(restored.node_id, original.node_id)
        self.assertEqual(restored.state, original.state)
        self.assertEqual(restored.failure_count, original.failure_count)
        self.assertEqual(restored.gossip_failure_count, original.gossip_failure_count)
        self.assertEqual(restored.last_seen, original.last_seen)
        self.assertEqual(restored.circuit_state, original.circuit_state)

    def test_updated_at_default_factory(self):
        """Test that updated_at uses current time as default."""
        before = time.time()
        state = PeerHealthState(node_id="timing-test", state="alive")
        after = time.time()
        self.assertGreaterEqual(state.updated_at, before)
        self.assertLessEqual(state.updated_at, after)


if __name__ == "__main__":
    import unittest
    unittest.main()

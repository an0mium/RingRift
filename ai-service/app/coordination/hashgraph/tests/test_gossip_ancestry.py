"""Tests for gossip ancestry tracking."""

from unittest import TestCase

from app.coordination.hashgraph.gossip_ancestry import (
    ValidationStatus,
    ValidationResult,
    AncestryEvent,
    GossipAncestryConfig,
    GossipAncestryTracker,
    add_ancestry_to_payload,
    validate_ancestry,
    has_ancestry_fields,
    get_gossip_ancestry_tracker,
    reset_gossip_ancestry_tracker,
)


class TestAncestryEvent(TestCase):
    """Tests for AncestryEvent dataclass."""

    def test_create_event(self) -> None:
        """Should create ancestry event."""
        event = AncestryEvent(
            event_hash="abc123",
            sender="node-1",
            self_parent="parent123",
            other_parent="other123",
            gossip_round=5,
            timestamp=1000.0,
            payload_hash="payload123",
        )
        self.assertEqual(event.event_hash, "abc123")
        self.assertEqual(event.sender, "node-1")
        self.assertEqual(event.gossip_round, 5)


class TestGossipAncestryConfig(TestCase):
    """Tests for GossipAncestryConfig."""

    def test_defaults(self) -> None:
        """Should have reasonable defaults."""
        config = GossipAncestryConfig()
        self.assertEqual(config.max_events_per_sender, 1000)
        self.assertEqual(config.max_total_events, 10000)
        self.assertFalse(config.require_ancestry)

    def test_custom_config(self) -> None:
        """Should accept custom values."""
        config = GossipAncestryConfig(
            max_events_per_sender=100,
            require_ancestry=True,
        )
        self.assertEqual(config.max_events_per_sender, 100)
        self.assertTrue(config.require_ancestry)


class TestGossipAncestryTracker(TestCase):
    """Tests for GossipAncestryTracker."""

    def setUp(self) -> None:
        """Create fresh tracker for each test."""
        self.tracker = GossipAncestryTracker(node_id="test-node")

    def test_add_ancestry_first_message(self) -> None:
        """First message should have no self_parent."""
        payload = {"sender": "test-node", "data": "test"}
        enhanced = self.tracker.add_ancestry(payload)

        self.assertIn("event_hash", enhanced)
        self.assertIn("gossip_round", enhanced)
        self.assertIsNone(enhanced["self_parent_hash"])
        self.assertIsNone(enhanced["other_parent_hash"])
        self.assertEqual(enhanced["gossip_round"], 1)

    def test_add_ancestry_second_message(self) -> None:
        """Second message should have self_parent."""
        payload1 = {"sender": "test-node", "data": "first"}
        enhanced1 = self.tracker.add_ancestry(payload1)
        first_hash = enhanced1["event_hash"]

        payload2 = {"sender": "test-node", "data": "second"}
        enhanced2 = self.tracker.add_ancestry(payload2)

        self.assertEqual(enhanced2["self_parent_hash"], first_hash)
        self.assertEqual(enhanced2["gossip_round"], 2)

    def test_add_ancestry_preserves_original(self) -> None:
        """Should not modify original payload."""
        payload = {"sender": "test-node", "data": "test"}
        enhanced = self.tracker.add_ancestry(payload)

        self.assertNotIn("event_hash", payload)
        self.assertIn("event_hash", enhanced)

    def test_validate_valid_message(self) -> None:
        """Should accept valid ancestry."""
        # Create a message with ancestry
        payload = {"sender": "other-node", "data": "test"}
        other_tracker = GossipAncestryTracker(node_id="other-node")
        enhanced = other_tracker.add_ancestry(payload)

        # Validate with our tracker
        result = self.tracker.validate_incoming(enhanced)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, ValidationStatus.VALID)

    def test_validate_no_ancestry_allowed(self) -> None:
        """Should accept messages without ancestry by default."""
        payload = {"sender": "old-node", "data": "legacy"}
        result = self.tracker.validate_incoming(payload)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, ValidationStatus.NO_ANCESTRY)
        self.assertFalse(result.has_ancestry)

    def test_validate_no_ancestry_required(self) -> None:
        """Should reject messages without ancestry when required."""
        config = GossipAncestryConfig(require_ancestry=True)
        strict_tracker = GossipAncestryTracker("strict", config)

        payload = {"sender": "old-node", "data": "legacy"}
        result = strict_tracker.validate_incoming(payload)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, ValidationStatus.NO_ANCESTRY)

    def test_detect_equivocation(self) -> None:
        """Should detect when sender forks their chain."""
        # Simulate receiving two events from same sender with same self_parent
        # This represents a Byzantine node creating a fork

        # First event from fork-node with no self_parent (genesis)
        event1 = {
            "sender": "fork-node",
            "data": "first_event",
            "event_hash": "hash_event_1",
            "self_parent_hash": None,
            "other_parent_hash": None,
            "gossip_round": 1,
        }
        self.tracker.validate_incoming(event1)

        # Second event from same sender with same self_parent (None)
        # This is a FORK - two genesis events from same sender
        event2 = {
            "sender": "fork-node",
            "data": "forked_event",
            "event_hash": "hash_event_2",
            "self_parent_hash": None,  # Same self_parent = FORK
            "other_parent_hash": None,
            "gossip_round": 2,
        }
        self.tracker.validate_incoming(event2)

        # Should detect equivocation
        forks = self.tracker.detect_equivocation("fork-node")
        self.assertEqual(len(forks), 1)
        # The fork pair should contain both event hashes
        fork_pair = forks[0]
        self.assertIn("hash_event_1", fork_pair)
        self.assertIn("hash_event_2", fork_pair)

    def test_get_sender_events(self) -> None:
        """Should track events per sender."""
        other_tracker = GossipAncestryTracker(node_id="sender")
        for i in range(3):
            payload = {"sender": "sender", "data": f"msg-{i}"}
            enhanced = other_tracker.add_ancestry(payload)
            self.tracker.validate_incoming(enhanced)

        events = self.tracker.get_sender_events("sender")
        self.assertEqual(len(events), 3)

    def test_get_latest_event(self) -> None:
        """Should return latest event per sender."""
        other_tracker = GossipAncestryTracker(node_id="sender")
        last_hash = None

        for i in range(3):
            payload = {"sender": "sender", "data": f"msg-{i}"}
            enhanced = other_tracker.add_ancestry(payload)
            last_hash = enhanced["event_hash"]
            self.tracker.validate_incoming(enhanced)

        latest = self.tracker.get_latest_event("sender")
        self.assertEqual(latest, last_hash)

    def test_get_event(self) -> None:
        """Should retrieve tracked event by hash."""
        other_tracker = GossipAncestryTracker(node_id="sender")
        payload = {"sender": "sender", "data": "test"}
        enhanced = other_tracker.add_ancestry(payload)
        event_hash = enhanced["event_hash"]

        self.tracker.validate_incoming(enhanced)
        event = self.tracker.get_event(event_hash)

        self.assertIsNotNone(event)
        self.assertEqual(event.sender, "sender")
        self.assertEqual(event.event_hash, event_hash)

    def test_get_stats(self) -> None:
        """Should return statistics."""
        other_tracker = GossipAncestryTracker(node_id="sender")
        for i in range(3):
            payload = {"sender": "sender", "data": f"msg-{i}"}
            enhanced = other_tracker.add_ancestry(payload)
            self.tracker.validate_incoming(enhanced)

        stats = self.tracker.get_stats()
        self.assertEqual(stats["total_events"], 3)
        self.assertEqual(stats["senders_tracked"], 1)
        self.assertEqual(stats["node_id"], "test-node")

    def test_clear(self) -> None:
        """Should clear all tracked events."""
        other_tracker = GossipAncestryTracker(node_id="sender")
        payload = {"sender": "sender", "data": "test"}
        enhanced = other_tracker.add_ancestry(payload)
        self.tracker.validate_incoming(enhanced)

        self.tracker.clear()

        stats = self.tracker.get_stats()
        self.assertEqual(stats["total_events"], 0)
        self.assertEqual(stats["senders_tracked"], 0)

    def test_event_limit_per_sender(self) -> None:
        """Should limit events per sender."""
        config = GossipAncestryConfig(max_events_per_sender=5)
        tracker = GossipAncestryTracker("test", config)

        other_tracker = GossipAncestryTracker(node_id="sender")
        for i in range(10):
            payload = {"sender": "sender", "data": f"msg-{i}"}
            enhanced = other_tracker.add_ancestry(payload)
            tracker.validate_incoming(enhanced)

        events = tracker.get_sender_events("sender")
        self.assertEqual(len(events), 5)  # Oldest removed

    def test_other_parent_tracking(self) -> None:
        """Should track other_parent from received messages."""
        # Send message from another node
        other_tracker = GossipAncestryTracker(node_id="other")
        other_payload = {"sender": "other", "data": "from_other"}
        other_enhanced = other_tracker.add_ancestry(other_payload)
        other_hash = other_enhanced["event_hash"]

        # Receive it
        self.tracker.validate_incoming(other_enhanced)

        # Our next message should have other_parent
        my_payload = {"sender": "test-node", "data": "my_message"}
        my_enhanced = self.tracker.add_ancestry(my_payload)

        self.assertEqual(my_enhanced["other_parent_hash"], other_hash)


class TestConvenienceFunctions(TestCase):
    """Tests for convenience functions."""

    def setUp(self) -> None:
        """Create tracker for tests."""
        self.tracker = GossipAncestryTracker(node_id="test")

    def test_add_ancestry_to_payload(self) -> None:
        """Should add ancestry via function."""
        payload = {"sender": "test", "data": "test"}
        enhanced = add_ancestry_to_payload(payload, self.tracker)
        self.assertIn("event_hash", enhanced)

    def test_validate_ancestry(self) -> None:
        """Should validate via function."""
        payload = {"sender": "test", "data": "test"}
        enhanced = add_ancestry_to_payload(payload, self.tracker)
        result = validate_ancestry(enhanced, self.tracker)
        self.assertTrue(result.is_valid)

    def test_has_ancestry_fields_true(self) -> None:
        """Should detect ancestry fields."""
        payload = {
            "event_hash": "abc",
            "self_parent_hash": None,
            "other_parent_hash": None,
            "gossip_round": 1,
        }
        self.assertTrue(has_ancestry_fields(payload))

    def test_has_ancestry_fields_false(self) -> None:
        """Should detect missing ancestry fields."""
        payload = {"sender": "test", "data": "no ancestry"}
        self.assertFalse(has_ancestry_fields(payload))


class TestSingletonTracker(TestCase):
    """Tests for singleton tracker."""

    def setUp(self) -> None:
        """Reset singleton before each test."""
        reset_gossip_ancestry_tracker()

    def tearDown(self) -> None:
        """Reset singleton after each test."""
        reset_gossip_ancestry_tracker()

    def test_get_tracker_requires_node_id(self) -> None:
        """Should require node_id on first call."""
        with self.assertRaises(ValueError):
            get_gossip_ancestry_tracker()

    def test_get_tracker_creates_singleton(self) -> None:
        """Should create and return singleton."""
        tracker1 = get_gossip_ancestry_tracker("node-1")
        tracker2 = get_gossip_ancestry_tracker()
        self.assertIs(tracker1, tracker2)

    def test_reset_clears_singleton(self) -> None:
        """Reset should clear singleton."""
        get_gossip_ancestry_tracker("node-1")
        reset_gossip_ancestry_tracker()

        with self.assertRaises(ValueError):
            get_gossip_ancestry_tracker()


class TestValidationResult(TestCase):
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Should create valid result."""
        result = ValidationResult(
            is_valid=True,
            status=ValidationStatus.VALID,
        )
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_ancestry)

    def test_invalid_result(self) -> None:
        """Should create invalid result with error."""
        result = ValidationResult(
            is_valid=False,
            status=ValidationStatus.EQUIVOCATION,
            error="Fork detected",
            detected_fork="abc123",
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(result.detected_fork, "abc123")


if __name__ == "__main__":
    import unittest
    unittest.main()

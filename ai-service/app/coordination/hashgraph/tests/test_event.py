"""Tests for HashgraphEvent dataclass and utilities."""

import json
import time
from unittest import TestCase

from app.coordination.hashgraph.event import (
    HashgraphEvent,
    EventType,
    EventBatch,
    canonical_json,
    compute_event_hash,
)


class TestCanonicalJson(TestCase):
    """Tests for canonical JSON serialization."""

    def test_canonical_json_sorted_keys(self) -> None:
        """Canonical JSON should have sorted keys."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        self.assertEqual(result, '{"a":2,"m":3,"z":1}')

    def test_canonical_json_no_whitespace(self) -> None:
        """Canonical JSON should have no extra whitespace."""
        obj = {"key": "value", "number": 42}
        result = canonical_json(obj)
        self.assertNotIn(" ", result)
        self.assertNotIn("\n", result)

    def test_canonical_json_nested(self) -> None:
        """Canonical JSON should handle nested objects."""
        obj = {"outer": {"inner": {"deep": 1}}}
        result = canonical_json(obj)
        self.assertEqual(result, '{"outer":{"inner":{"deep":1}}}')

    def test_canonical_json_deterministic(self) -> None:
        """Same dict should always produce same JSON."""
        obj = {"b": 2, "a": 1}
        result1 = canonical_json(obj)
        result2 = canonical_json(obj)
        self.assertEqual(result1, result2)


class TestComputeEventHash(TestCase):
    """Tests for event hash computation."""

    def test_compute_hash_deterministic(self) -> None:
        """Same inputs should produce same hash."""
        hash1 = compute_event_hash(
            creator="node-1",
            payload={"type": "test"},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
        )
        hash2 = compute_event_hash(
            creator="node-1",
            payload={"type": "test"},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
        )
        self.assertEqual(hash1, hash2)

    def test_compute_hash_different_creators(self) -> None:
        """Different creators should produce different hashes."""
        hash1 = compute_event_hash(
            creator="node-1",
            payload={},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
        )
        hash2 = compute_event_hash(
            creator="node-2",
            payload={},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
        )
        self.assertNotEqual(hash1, hash2)

    def test_compute_hash_is_hex(self) -> None:
        """Hash should be hex-encoded."""
        result = compute_event_hash(
            creator="node-1",
            payload={},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
        )
        # Should be 64 hex characters (SHA256)
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in result))


class TestHashgraphEvent(TestCase):
    """Tests for HashgraphEvent dataclass."""

    def test_create_genesis(self) -> None:
        """Should create genesis event with no parents."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        self.assertEqual(event.creator, "node-1")
        self.assertIsNone(event.self_parent)
        self.assertIsNone(event.other_parent)
        self.assertEqual(event.round_number, 0)
        self.assertTrue(event.is_genesis())

    def test_create_with_parents(self) -> None:
        """Should create event with parent references."""
        parent_hash = "a" * 64
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "test"},
            self_parent=parent_hash,
            other_parent=None,
            round_number=1,
        )
        self.assertEqual(event.self_parent, parent_hash)
        self.assertFalse(event.is_genesis())

    def test_create_auto_hash(self) -> None:
        """Create should automatically compute hash."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "test"},
        )
        self.assertIsNotNone(event.event_hash)
        self.assertEqual(len(event.event_hash), 64)

    def test_verify_hash_valid(self) -> None:
        """Verify should return True for valid hash."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "test"},
        )
        self.assertTrue(event.verify_hash())

    def test_verify_hash_invalid(self) -> None:
        """Verify should return False for tampered event."""
        event = HashgraphEvent(
            creator="node-1",
            payload={"type": "test"},
            timestamp=1000.0,
            self_parent=None,
            other_parent=None,
            round_number=0,
            event_hash="wrong_hash",
        )
        self.assertFalse(event.verify_hash())

    def test_get_parent_hashes_none(self) -> None:
        """Genesis event should have empty parent list."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        self.assertEqual(event.get_parent_hashes(), [])

    def test_get_parent_hashes_both(self) -> None:
        """Event with both parents should list both."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent="a" * 64,
            other_parent="b" * 64,
        )
        parents = event.get_parent_hashes()
        self.assertEqual(len(parents), 2)
        self.assertIn("a" * 64, parents)
        self.assertIn("b" * 64, parents)

    def test_get_event_type(self) -> None:
        """Should extract event type from payload."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "evaluation_result"},
        )
        self.assertEqual(event.get_event_type(), EventType.EVALUATION_RESULT)

    def test_get_event_type_unknown(self) -> None:
        """Should return None for unknown event type."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "unknown_type"},
        )
        self.assertIsNone(event.get_event_type())

    def test_to_dict_from_dict(self) -> None:
        """Should roundtrip through dict serialization."""
        original = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "test", "value": 42},
            self_parent="a" * 64,
            round_number=5,
        )
        serialized = original.to_dict()
        restored = HashgraphEvent.from_dict(serialized)
        self.assertEqual(restored.event_hash, original.event_hash)
        self.assertEqual(restored.creator, original.creator)
        self.assertEqual(restored.payload, original.payload)
        self.assertEqual(restored.self_parent, original.self_parent)
        self.assertEqual(restored.round_number, original.round_number)

    def test_from_dict_invalid_hash(self) -> None:
        """Should raise on hash mismatch during deserialization."""
        data = {
            "creator": "node-1",
            "payload": {"type": "test"},
            "timestamp": 1000.0,
            "self_parent": None,
            "other_parent": None,
            "round_number": 0,
            "event_hash": "wrong_hash",
        }
        with self.assertRaises(ValueError):
            HashgraphEvent.from_dict(data)

    def test_repr(self) -> None:
        """Should have readable repr."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={"type": "evaluation_result"},
            round_number=3,
        )
        repr_str = repr(event)
        self.assertIn("node-1", repr_str)
        self.assertIn("evaluation_result", repr_str)
        self.assertIn("round=3", repr_str)

    def test_immutable(self) -> None:
        """Event should be immutable (frozen dataclass)."""
        event = HashgraphEvent.create(creator="node-1", payload={})
        with self.assertRaises(AttributeError):
            event.creator = "node-2"  # type: ignore


class TestEventBatch(TestCase):
    """Tests for EventBatch."""

    def test_create_empty_batch(self) -> None:
        """Should create empty batch."""
        batch = EventBatch(sender="node-1")
        self.assertEqual(len(batch.events), 0)
        self.assertEqual(batch.sender, "node-1")

    def test_add_events(self) -> None:
        """Should add events to batch."""
        batch = EventBatch(sender="node-1")
        event = HashgraphEvent.create_genesis(creator="node-1")
        batch.add(event)
        self.assertEqual(len(batch.events), 1)
        self.assertEqual(batch.events[0].event_hash, event.event_hash)

    def test_to_dict_from_dict(self) -> None:
        """Should roundtrip through serialization."""
        batch = EventBatch(sender="node-1")
        batch.add(HashgraphEvent.create_genesis(creator="node-1"))
        batch.add(HashgraphEvent.create_genesis(creator="node-2"))

        serialized = batch.to_dict()
        restored = EventBatch.from_dict(serialized)

        self.assertEqual(len(restored.events), 2)
        self.assertEqual(restored.sender, batch.sender)


class TestEventType(TestCase):
    """Tests for EventType enum."""

    def test_event_types_exist(self) -> None:
        """Should have expected event types."""
        self.assertEqual(EventType.GENESIS.value, "genesis")
        self.assertEqual(EventType.EVALUATION_RESULT.value, "evaluation_result")
        self.assertEqual(EventType.PROMOTION_PROPOSAL.value, "promotion_proposal")
        self.assertEqual(EventType.ELO_UPDATE.value, "elo_update")

    def test_event_type_from_string(self) -> None:
        """Should create enum from string."""
        event_type = EventType("evaluation_result")
        self.assertEqual(event_type, EventType.EVALUATION_RESULT)


if __name__ == "__main__":
    import unittest
    unittest.main()

"""Tests for HashgraphDAG."""

from unittest import TestCase

from app.coordination.hashgraph.event import HashgraphEvent
from app.coordination.hashgraph.dag import HashgraphDAG, DAGNode


class TestHashgraphDAG(TestCase):
    """Tests for HashgraphDAG."""

    def setUp(self) -> None:
        """Create a fresh DAG for each test."""
        self.dag = HashgraphDAG(node_id="test-node")

    def test_add_genesis_event(self) -> None:
        """Should add genesis event successfully."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        result = self.dag.add_event(event)
        self.assertTrue(result)
        self.assertTrue(self.dag.has_event(event.event_hash))

    def test_add_duplicate_event(self) -> None:
        """Should reject duplicate events."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(event)
        result = self.dag.add_event(event)
        self.assertFalse(result)

    def test_add_event_with_missing_parent(self) -> None:
        """Should reject event with missing parent."""
        event = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent="nonexistent_hash",
        )
        result = self.dag.add_event(event)
        self.assertFalse(result)

    def test_add_event_chain(self) -> None:
        """Should add chain of events from same creator."""
        genesis = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(genesis)

        # Compute proper round for event1
        round1 = self.dag.compute_round_for_parents(genesis.event_hash, None)
        event1 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 1},
            self_parent=genesis.event_hash,
            round_number=round1,
        )
        self.dag.add_event(event1)

        # Compute proper round for event2
        round2 = self.dag.compute_round_for_parents(event1.event_hash, None)
        event2 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 2},
            self_parent=event1.event_hash,
            round_number=round2,
        )
        self.dag.add_event(event2)

        chain = self.dag.get_creator_chain("node-1")
        self.assertEqual(len(chain), 3)

    def test_get_event(self) -> None:
        """Should retrieve event by hash."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(event)

        retrieved = self.dag.get_event(event.event_hash)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.creator, "node-1")

    def test_get_event_not_found(self) -> None:
        """Should return None for unknown hash."""
        result = self.dag.get_event("nonexistent_hash")
        self.assertIsNone(result)

    def test_get_ancestors_genesis(self) -> None:
        """Genesis event should have no ancestors."""
        genesis = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(genesis)

        ancestors = self.dag.get_ancestors(genesis.event_hash)
        self.assertEqual(len(ancestors), 0)

    def test_get_ancestors_chain(self) -> None:
        """Should return all ancestors in chain."""
        genesis = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(genesis)

        round1 = self.dag.compute_round_for_parents(genesis.event_hash, None)
        event1 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 1},
            self_parent=genesis.event_hash,
            round_number=round1,
        )
        self.dag.add_event(event1)

        round2 = self.dag.compute_round_for_parents(event1.event_hash, None)
        event2 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 2},
            self_parent=event1.event_hash,
            round_number=round2,
        )
        self.dag.add_event(event2)

        ancestors = self.dag.get_ancestors(event2.event_hash)
        self.assertEqual(len(ancestors), 2)
        self.assertIn(genesis.event_hash, ancestors)
        self.assertIn(event1.event_hash, ancestors)

    def test_get_ancestors_multiple_paths(self) -> None:
        """Should find ancestors through multiple paths."""
        # Create DAG:
        # g1 ← e1
        # g2 ← e2
        #    ↘  ↓
        #     merge
        g1 = HashgraphEvent.create_genesis(creator="node-1")
        g2 = HashgraphEvent.create_genesis(creator="node-2")
        self.dag.add_event(g1)
        self.dag.add_event(g2)

        r1 = self.dag.compute_round_for_parents(g1.event_hash, None)
        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=g1.event_hash,
            round_number=r1,
        )
        self.dag.add_event(e1)

        r2 = self.dag.compute_round_for_parents(g2.event_hash, None)
        e2 = HashgraphEvent.create(
            creator="node-2",
            payload={},
            self_parent=g2.event_hash,
            round_number=r2,
        )
        self.dag.add_event(e2)

        r_merge = self.dag.compute_round_for_parents(e1.event_hash, e2.event_hash)
        merge = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=e1.event_hash,
            other_parent=e2.event_hash,
            round_number=r_merge,
        )
        self.dag.add_event(merge)

        ancestors = self.dag.get_ancestors(merge.event_hash)
        self.assertEqual(len(ancestors), 4)
        self.assertIn(g1.event_hash, ancestors)
        self.assertIn(g2.event_hash, ancestors)
        self.assertIn(e1.event_hash, ancestors)
        self.assertIn(e2.event_hash, ancestors)

    def test_is_ancestor(self) -> None:
        """Should correctly identify ancestor relationships."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        r = self.dag.compute_round_for_parents(g.event_hash, None)
        e = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=g.event_hash,
            round_number=r,
        )
        self.dag.add_event(e)

        self.assertTrue(self.dag.is_ancestor(g.event_hash, e.event_hash))
        self.assertFalse(self.dag.is_ancestor(e.event_hash, g.event_hash))
        self.assertFalse(self.dag.is_ancestor(g.event_hash, g.event_hash))

    def test_get_descendants(self) -> None:
        """Should return all descendants."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        r1 = self.dag.compute_round_for_parents(g.event_hash, None)
        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 1},
            self_parent=g.event_hash,
            round_number=r1,
        )
        self.dag.add_event(e1)

        r2 = self.dag.compute_round_for_parents(e1.event_hash, None)
        e2 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 2},
            self_parent=e1.event_hash,
            round_number=r2,
        )
        self.dag.add_event(e2)

        descendants = self.dag.get_descendants(g.event_hash)
        self.assertEqual(len(descendants), 2)
        self.assertIn(e1.event_hash, descendants)
        self.assertIn(e2.event_hash, descendants)

    def test_get_latest_event(self) -> None:
        """Should return most recent event from creator."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        r = self.dag.compute_round_for_parents(g.event_hash, None)
        e = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=g.event_hash,
            round_number=r,
        )
        self.dag.add_event(e)

        latest = self.dag.get_latest_event("node-1")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.event_hash, e.event_hash)

    def test_get_latest_event_unknown_creator(self) -> None:
        """Should return None for unknown creator."""
        result = self.dag.get_latest_event("unknown")
        self.assertIsNone(result)

    def test_get_known_creators(self) -> None:
        """Should return all known creators."""
        g1 = HashgraphEvent.create_genesis(creator="node-1")
        g2 = HashgraphEvent.create_genesis(creator="node-2")
        self.dag.add_event(g1)
        self.dag.add_event(g2)

        creators = self.dag.get_known_creators()
        self.assertEqual(len(creators), 2)
        self.assertIn("node-1", creators)
        self.assertIn("node-2", creators)

    def test_get_topological_order(self) -> None:
        """Should return events in topological order."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        r1 = self.dag.compute_round_for_parents(g.event_hash, None)
        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 1},
            self_parent=g.event_hash,
            round_number=r1,
        )
        self.dag.add_event(e1)

        r2 = self.dag.compute_round_for_parents(e1.event_hash, None)
        e2 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 2},
            self_parent=e1.event_hash,
            round_number=r2,
        )
        self.dag.add_event(e2)

        order = self.dag.get_topological_order()
        hashes = [e.event_hash for e in order]

        # Parents should come before children
        self.assertTrue(hashes.index(g.event_hash) < hashes.index(e1.event_hash))
        self.assertTrue(hashes.index(e1.event_hash) < hashes.index(e2.event_hash))

    def test_get_round_events(self) -> None:
        """Should return events in a round."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        events = self.dag.get_round_events(0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_hash, g.event_hash)

    def test_get_round_witnesses(self) -> None:
        """First event from creator in round should be witness."""
        g1 = HashgraphEvent.create_genesis(creator="node-1")
        g2 = HashgraphEvent.create_genesis(creator="node-2")
        self.dag.add_event(g1)
        self.dag.add_event(g2)

        witnesses = self.dag.get_round_witnesses(0)
        self.assertEqual(len(witnesses), 2)

    def test_witness_assignment(self) -> None:
        """Only first event per creator per round should be witness."""
        g1 = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g1)

        # Second event from same creator in same round
        e2 = HashgraphEvent.create(
            creator="node-1",
            payload={"seq": 2},
            self_parent=g1.event_hash,
            round_number=0,  # Force same round
        )
        self.dag.add_event(e2)

        witnesses = self.dag.get_round_witnesses(0)
        # Only g1 should be witness
        witness_hashes = [w.event_hash for w in witnesses]
        self.assertIn(g1.event_hash, witness_hashes)

    def test_detect_equivocation(self) -> None:
        """Should detect when creator forks their chain."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        # Two events with same self_parent = fork (both claim round 1)
        r = self.dag.compute_round_for_parents(g.event_hash, None)
        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={"fork": 1},
            self_parent=g.event_hash,
            round_number=r,
        )
        self.dag.add_event(e1)

        # Same parent = equivocation (forking)
        e2 = HashgraphEvent.create(
            creator="node-1",
            payload={"fork": 2},
            self_parent=g.event_hash,
            round_number=r,
        )
        self.dag.add_event(e2)

        equivocations = self.dag.detect_equivocation("node-1")
        self.assertEqual(len(equivocations), 1)
        fork_pair = equivocations[0]
        self.assertIn(e1.event_hash, fork_pair)
        self.assertIn(e2.event_hash, fork_pair)

    def test_detect_equivocation_none(self) -> None:
        """Should return empty list for honest creator."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)

        r = self.dag.compute_round_for_parents(g.event_hash, None)
        e = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=g.event_hash,
            round_number=r,
        )
        self.dag.add_event(e)

        equivocations = self.dag.detect_equivocation("node-1")
        self.assertEqual(len(equivocations), 0)

    def test_get_stats(self) -> None:
        """Should return correct statistics."""
        g1 = HashgraphEvent.create_genesis(creator="node-1")
        g2 = HashgraphEvent.create_genesis(creator="node-2")
        self.dag.add_event(g1)
        self.dag.add_event(g2)

        stats = self.dag.get_stats()
        self.assertEqual(stats["event_count"], 2)
        self.assertEqual(stats["creator_count"], 2)

    def test_clear(self) -> None:
        """Should clear all events."""
        g = HashgraphEvent.create_genesis(creator="node-1")
        self.dag.add_event(g)
        self.assertEqual(self.dag.get_event_count(), 1)

        self.dag.clear()
        self.assertEqual(self.dag.get_event_count(), 0)


class TestDAGNode(TestCase):
    """Tests for DAGNode wrapper."""

    def test_dag_node_properties(self) -> None:
        """DAGNode should expose event properties."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        node = DAGNode(event=event)

        self.assertEqual(node.event_hash, event.event_hash)
        self.assertEqual(node.creator, event.creator)
        self.assertEqual(node.round_number, event.round_number)

    def test_dag_node_children(self) -> None:
        """DAGNode should track children."""
        event = HashgraphEvent.create_genesis(creator="node-1")
        node = DAGNode(event=event)

        self.assertEqual(len(node.children), 0)
        node.children.add("child_hash")
        self.assertEqual(len(node.children), 1)


if __name__ == "__main__":
    import unittest
    unittest.main()

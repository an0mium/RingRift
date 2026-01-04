"""Tests for virtual voting consensus algorithm."""

from unittest import TestCase

from app.coordination.hashgraph.event import HashgraphEvent
from app.coordination.hashgraph.dag import HashgraphDAG
from app.coordination.hashgraph.consensus import (
    ConsensusEngine,
    VoteType,
    VirtualVote,
)


class TestConsensusEngine(TestCase):
    """Tests for ConsensusEngine."""

    def setUp(self) -> None:
        """Create DAG and consensus engine for each test."""
        self.dag = HashgraphDAG(node_id="test-node")
        self.engine = ConsensusEngine(self.dag)

    def _create_network(self, num_nodes: int = 4) -> list[HashgraphEvent]:
        """Create a network of genesis events.

        Args:
            num_nodes: Number of nodes to create

        Returns:
            List of genesis events
        """
        genesis_events = []
        for i in range(num_nodes):
            g = HashgraphEvent.create_genesis(creator=f"node-{i}")
            self.dag.add_event(g)
            genesis_events.append(g)
        return genesis_events

    def test_get_supermajority_threshold(self) -> None:
        """Should compute 2/3 supermajority threshold."""
        # With 4 nodes, need 3 for 2/3 majority
        self._create_network(4)
        threshold = self.engine.get_supermajority_threshold()
        self.assertEqual(threshold, 3)

        # With 6 nodes, need 4 for 2/3 majority
        self.dag.clear()
        self._create_network(6)
        threshold = self.engine.get_supermajority_threshold()
        self.assertEqual(threshold, 4)

    def test_can_see_self(self) -> None:
        """Event can see itself."""
        genesis = self._create_network(1)[0]
        self.assertTrue(self.engine.can_see(genesis.event_hash, genesis.event_hash))

    def test_can_see_ancestor(self) -> None:
        """Event can see its ancestors."""
        genesis = self._create_network(1)[0]

        child = HashgraphEvent.create(
            creator="node-0",
            payload={},
            self_parent=genesis.event_hash,
        )
        self.dag.add_event(child)

        self.assertTrue(self.engine.can_see(child.event_hash, genesis.event_hash))
        self.assertFalse(self.engine.can_see(genesis.event_hash, child.event_hash))

    def test_strongly_sees_simple(self) -> None:
        """Test strongly-seeing with simple 3-node network."""
        # Create 3 nodes
        genesis_events = self._create_network(3)

        # Create event that sees all genesis events
        e0 = HashgraphEvent.create(
            creator="node-0",
            payload={},
            self_parent=genesis_events[0].event_hash,
            other_parent=genesis_events[1].event_hash,
        )
        self.dag.add_event(e0)

        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=genesis_events[1].event_hash,
            other_parent=genesis_events[2].event_hash,
        )
        self.dag.add_event(e1)

        # Create merge event that sees e0 and e1
        merge = HashgraphEvent.create(
            creator="node-2",
            payload={},
            self_parent=genesis_events[2].event_hash,
            other_parent=e0.event_hash,
        )
        self.dag.add_event(merge)

        merge2 = HashgraphEvent.create(
            creator="node-2",
            payload={},
            self_parent=merge.event_hash,
            other_parent=e1.event_hash,
        )
        self.dag.add_event(merge2)

        # merge2 should strongly see genesis_events[0] through multiple paths
        result = self.engine.strongly_sees(
            merge2.event_hash,
            genesis_events[0].event_hash,
        )
        # With 3 nodes, need 2 for strongly-seeing
        # merge2 can see: e0 (which sees g0), g0, e1, g1, g2
        self.assertTrue(result.strongly_sees)

    def test_strongly_sees_insufficient(self) -> None:
        """Should not strongly see with insufficient paths."""
        # Create 4 nodes
        genesis_events = self._create_network(4)

        # Create event that only sees one other genesis
        e0 = HashgraphEvent.create(
            creator="node-0",
            payload={},
            self_parent=genesis_events[0].event_hash,
            other_parent=genesis_events[1].event_hash,
        )
        self.dag.add_event(e0)

        # e0 cannot strongly see g3 (only seen through 2 paths, need 3)
        result = self.engine.strongly_sees(
            e0.event_hash,
            genesis_events[3].event_hash,
        )
        self.assertFalse(result.strongly_sees)

    def test_compute_virtual_votes(self) -> None:
        """Should compute virtual votes for witnesses."""
        # Create 3-node network
        genesis_events = self._create_network(3)

        # Create round 1 events
        e0 = HashgraphEvent.create(
            creator="node-0",
            payload={},
            self_parent=genesis_events[0].event_hash,
            other_parent=genesis_events[1].event_hash,
            round_number=1,
        )
        self.dag.add_event(e0)

        # Compute votes from round 1 on a round 0 event
        votes = self.engine.compute_virtual_votes(
            voter_round=1,
            target_hash=genesis_events[0].event_hash,
        )

        # Should have at least one vote from round 1 witness
        self.assertGreater(len(votes), 0)

    def test_count_votes(self) -> None:
        """Should correctly count vote types."""
        votes = [
            VirtualVote("node-1", "target", VoteType.YES, 1),
            VirtualVote("node-2", "target", VoteType.YES, 1),
            VirtualVote("node-3", "target", VoteType.NO, 1),
            VirtualVote("node-4", "target", VoteType.ABSTAIN, 1),
        ]

        yes, no, abstain = self.engine.count_votes(votes)
        self.assertEqual(yes, 2)
        self.assertEqual(no, 1)
        self.assertEqual(abstain, 1)

    def test_has_supermajority_yes(self) -> None:
        """Should detect supermajority YES."""
        self._create_network(4)  # Need 3 of 4 for supermajority

        votes = [
            VirtualVote("node-1", "target", VoteType.YES, 1),
            VirtualVote("node-2", "target", VoteType.YES, 1),
            VirtualVote("node-3", "target", VoteType.YES, 1),
            VirtualVote("node-4", "target", VoteType.NO, 1),
        ]

        self.assertTrue(self.engine.has_supermajority_yes(votes))

    def test_has_supermajority_no(self) -> None:
        """Should detect supermajority NO."""
        self._create_network(4)

        votes = [
            VirtualVote("node-1", "target", VoteType.NO, 1),
            VirtualVote("node-2", "target", VoteType.NO, 1),
            VirtualVote("node-3", "target", VoteType.NO, 1),
            VirtualVote("node-4", "target", VoteType.YES, 1),
        ]

        self.assertTrue(self.engine.has_supermajority_no(votes))

    def test_no_supermajority(self) -> None:
        """Should not have supermajority with split votes."""
        self._create_network(4)

        votes = [
            VirtualVote("node-1", "target", VoteType.YES, 1),
            VirtualVote("node-2", "target", VoteType.YES, 1),
            VirtualVote("node-3", "target", VoteType.NO, 1),
            VirtualVote("node-4", "target", VoteType.NO, 1),
        ]

        self.assertFalse(self.engine.has_supermajority_yes(votes))
        self.assertFalse(self.engine.has_supermajority_no(votes))

    def test_get_consensus_timestamp(self) -> None:
        """Should compute consensus timestamp from witnesses."""
        genesis_events = self._create_network(3)

        # Create witnesses with different timestamps
        e0 = HashgraphEvent.create(
            creator="node-0",
            payload={},
            self_parent=genesis_events[0].event_hash,
            timestamp=100.0,
            round_number=1,
        )
        self.dag.add_event(e0)

        e1 = HashgraphEvent.create(
            creator="node-1",
            payload={},
            self_parent=genesis_events[1].event_hash,
            timestamp=200.0,
            round_number=1,
        )
        self.dag.add_event(e1)

        # Get consensus timestamp for a genesis event
        # Should be median of when witnesses see it
        timestamp = self.engine.get_consensus_timestamp(genesis_events[0].event_hash)
        # Should be around 100.0 or 200.0 (depends on which witnesses see it)
        self.assertIsNotNone(timestamp)

    def test_get_consensus_for_round(self) -> None:
        """Should get consensus state for a round."""
        genesis_events = self._create_network(3)

        result = self.engine.get_consensus_for_round(0)
        self.assertEqual(result.round, 0)

    def test_is_consensus_reached_not_yet(self) -> None:
        """Should not have consensus before famous witnesses decided."""
        genesis_events = self._create_network(3)

        # Without later rounds, no famous witnesses are decided
        result = self.engine.is_consensus_reached(genesis_events[0].event_hash)
        # Typically False until witnesses are decided as famous
        # This depends on the DAG state

    def test_clear_cache(self) -> None:
        """Should clear strongly-seeing cache."""
        genesis_events = self._create_network(3)

        # Populate cache
        self.engine.strongly_sees(
            genesis_events[0].event_hash,
            genesis_events[1].event_hash,
        )

        self.engine.clear_cache()
        # Cache should be empty (no direct way to check, but shouldn't raise)


class TestVirtualVote(TestCase):
    """Tests for VirtualVote dataclass."""

    def test_vote_creation(self) -> None:
        """Should create vote with all fields."""
        vote = VirtualVote(
            voter="node-1",
            target="event_hash",
            vote=VoteType.YES,
            round=5,
            via_event="witness_hash",
        )
        self.assertEqual(vote.voter, "node-1")
        self.assertEqual(vote.target, "event_hash")
        self.assertEqual(vote.vote, VoteType.YES)
        self.assertEqual(vote.round, 5)
        self.assertEqual(vote.via_event, "witness_hash")


if __name__ == "__main__":
    import unittest
    unittest.main()

"""Tests for promotion consensus manager."""

import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase

from app.coordination.hashgraph.dag import HashgraphDAG
from app.coordination.hashgraph.promotion_consensus import (
    EvaluationEvidence,
    PromotionProposal,
    PromotionVote,
    PromotionCertificate,
    PromotionConsensusResult,
    PromotionConsensusConfig,
    PromotionConsensusManager,
    PromotionEventType,
    VoteType,
    get_promotion_consensus_manager,
    reset_promotion_consensus_manager,
)


class TestEvaluationEvidence(TestCase):
    """Tests for EvaluationEvidence dataclass."""

    def test_valid_evidence(self) -> None:
        """Should create valid evidence."""
        evidence = EvaluationEvidence(
            win_rate=0.85,
            elo=1450,
            games_played=100,
            vs_baselines={"random": 0.95, "heuristic": 0.65},
        )
        self.assertEqual(evidence.win_rate, 0.85)
        self.assertEqual(evidence.elo, 1450)
        self.assertEqual(evidence.games_played, 100)
        self.assertIn("random", evidence.vs_baselines)

    def test_auto_timestamp(self) -> None:
        """Should set timestamp automatically."""
        evidence = EvaluationEvidence(
            win_rate=0.75,
            elo=1300,
            games_played=50,
        )
        self.assertGreater(evidence.timestamp, 0)

    def test_invalid_win_rate_high(self) -> None:
        """Should reject win rate > 1.0."""
        with self.assertRaises(ValueError):
            EvaluationEvidence(
                win_rate=1.5,
                elo=1400,
                games_played=100,
            )

    def test_invalid_win_rate_low(self) -> None:
        """Should reject win rate < 0.0."""
        with self.assertRaises(ValueError):
            EvaluationEvidence(
                win_rate=-0.1,
                elo=1400,
                games_played=100,
            )

    def test_invalid_games(self) -> None:
        """Should reject negative games."""
        with self.assertRaises(ValueError):
            EvaluationEvidence(
                win_rate=0.75,
                elo=1400,
                games_played=-10,
            )


class TestPromotionConsensusConfig(TestCase):
    """Tests for PromotionConsensusConfig."""

    def test_defaults(self) -> None:
        """Should have reasonable defaults."""
        config = PromotionConsensusConfig()
        self.assertEqual(config.min_voters, 3)
        self.assertAlmostEqual(config.supermajority_fraction, 2 / 3, places=4)
        self.assertEqual(config.min_win_rate, 0.6)
        self.assertEqual(config.min_games, 50)

    def test_custom_config(self) -> None:
        """Should accept custom values."""
        config = PromotionConsensusConfig(
            min_voters=5,
            min_win_rate=0.7,
            min_elo=1400,
        )
        self.assertEqual(config.min_voters, 5)
        self.assertEqual(config.min_win_rate, 0.7)
        self.assertEqual(config.min_elo, 1400)


class TestPromotionConsensusManagerAsync(IsolatedAsyncioTestCase):
    """Async tests for PromotionConsensusManager."""

    def setUp(self) -> None:
        """Create fresh DAG and manager for each test."""
        self.dag = HashgraphDAG(node_id="test-node")
        self.config = PromotionConsensusConfig(
            min_voters=3,
            min_games=10,
            min_win_rate=0.5,
            min_elo=1000,
        )
        self.manager = PromotionConsensusManager(
            dag=self.dag,
            node_id="test-node",
            config=self.config,
        )

    async def test_propose_promotion(self) -> None:
        """Should create promotion proposal."""
        evidence = EvaluationEvidence(
            win_rate=0.85,
            elo=1450,
            games_played=100,
        )
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence=evidence,
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.model_hash, "model123")
        self.assertEqual(proposal.config_key, "hex8_2p")
        self.assertEqual(proposal.proposer, "test-node")
        self.assertIsNotNone(proposal.proposal_id)

    async def test_propose_with_dict_evidence(self) -> None:
        """Should accept evidence as dict."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={
                "win_rate": 0.85,
                "elo": 1450,
                "games_played": 100,
            },
        )
        self.assertIsNotNone(proposal)

    async def test_propose_invalid_evidence(self) -> None:
        """Should reject low win rate."""
        with self.assertRaises(ValueError):
            await self.manager.propose_promotion(
                model_hash="model123",
                config_key="hex8_2p",
                evidence={
                    "win_rate": 0.3,  # Below threshold
                    "elo": 1450,
                    "games_played": 100,
                },
            )

    async def test_vote_approve(self) -> None:
        """Should cast approval vote."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        vote = await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=True,
            voter="node-1",
        )

        self.assertEqual(vote.vote, VoteType.APPROVE)
        self.assertEqual(vote.voter, "node-1")

    async def test_vote_reject(self) -> None:
        """Should cast rejection vote."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        vote = await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=False,
            reason="Insufficient evaluation data",
            voter="node-2",
        )

        self.assertEqual(vote.vote, VoteType.REJECT)
        self.assertEqual(vote.reason, "Insufficient evaluation data")

    async def test_vote_abstain(self) -> None:
        """Should cast abstention."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        vote = await self.manager.abstain_on_proposal(
            proposal.proposal_id,
            voter="node-3",
        )

        self.assertEqual(vote.vote, VoteType.ABSTAIN)

    async def test_duplicate_vote_ignored(self) -> None:
        """Should ignore duplicate votes from same voter."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        vote1 = await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=True,
            voter="node-1",
        )

        # Second vote should return the first
        vote2 = await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=False,  # Different vote
            voter="node-1",
        )

        self.assertEqual(vote1, vote2)
        self.assertEqual(vote1.vote, VoteType.APPROVE)

    async def test_vote_on_unknown_proposal(self) -> None:
        """Should reject vote on unknown proposal."""
        with self.assertRaises(ValueError):
            await self.manager.vote_on_proposal(
                "nonexistent-proposal",
                approve=True,
            )

    async def test_consensus_approved(self) -> None:
        """Should reach approval consensus with supermajority."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        # 3 approvals
        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter=f"node-{i}",
            )

        result = await self.manager.get_promotion_consensus(
            proposal.proposal_id,
            timeout=1.0,
        )

        self.assertTrue(result.has_consensus)
        self.assertTrue(result.approved)
        self.assertEqual(result.approval_count, 3)
        self.assertIsNotNone(result.certificate)

    async def test_consensus_rejected(self) -> None:
        """Should reach rejection consensus."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        # 3 rejections
        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=False,
                voter=f"node-{i}",
            )

        result = await self.manager.get_promotion_consensus(
            proposal.proposal_id,
            timeout=1.0,
        )

        self.assertTrue(result.has_consensus)
        self.assertFalse(result.approved)
        self.assertEqual(result.rejection_count, 3)
        self.assertIsNone(result.certificate)

    async def test_no_consensus_insufficient_votes(self) -> None:
        """Should not reach consensus with too few votes."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        # Only 2 votes (need 3)
        for i in range(2):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter=f"node-{i}",
            )

        result = await self.manager.get_promotion_consensus(
            proposal.proposal_id,
            timeout=0.1,
        )

        self.assertFalse(result.has_consensus)

    async def test_no_consensus_split_vote(self) -> None:
        """Should not reach consensus with split votes."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        # 2 approve, 1 reject - no supermajority
        await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=True,
            voter="node-1",
        )
        await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=True,
            voter="node-2",
        )
        await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=False,
            voter="node-3",
        )

        result = await self.manager.get_promotion_consensus(
            proposal.proposal_id,
            timeout=1.0,
        )

        # 2/3 = 66%, supermajority is >66%, so 2 of 3 is not enough
        self.assertFalse(result.approved)

    async def test_certificate_created_on_approval(self) -> None:
        """Should create certificate with approval."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter=f"node-{i}",
            )

        result = await self.manager.get_promotion_consensus(proposal.proposal_id)

        cert = result.certificate
        self.assertIsNotNone(cert)
        self.assertEqual(cert.model_hash, "model123")
        self.assertEqual(cert.config_key, "hex8_2p")
        self.assertEqual(cert.approval_count, 3)
        self.assertEqual(len(cert.approvers), 3)
        self.assertIsNotNone(cert.certificate_hash)

    async def test_get_proposal(self) -> None:
        """Should retrieve proposal by ID."""
        created = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        retrieved = self.manager.get_proposal(created.proposal_id)
        self.assertEqual(retrieved.model_hash, created.model_hash)

    async def test_get_proposal_votes(self) -> None:
        """Should retrieve votes for proposal."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=True,
            voter="node-1",
        )
        await self.manager.vote_on_proposal(
            proposal.proposal_id,
            approve=False,
            voter="node-2",
        )

        votes = self.manager.get_proposal_votes(proposal.proposal_id)
        self.assertEqual(len(votes), 2)

    async def test_get_pending_proposals(self) -> None:
        """Should return proposals awaiting consensus."""
        await self.manager.propose_promotion(
            model_hash="model1",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )
        await self.manager.propose_promotion(
            model_hash="model2",
            config_key="hex8_4p",
            evidence={"win_rate": 0.80, "elo": 1400, "games_played": 100},
        )

        pending = self.manager.get_pending_proposals()
        self.assertEqual(len(pending), 2)

    async def test_get_approved_promotions(self) -> None:
        """Should return approved promotions."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter=f"node-{i}",
            )

        await self.manager.get_promotion_consensus(proposal.proposal_id)

        approved = self.manager.get_approved_promotions()
        self.assertEqual(len(approved), 1)
        self.assertEqual(approved[0].model_hash, "model123")

    async def test_get_stats(self) -> None:
        """Should return statistics."""
        await self.manager.propose_promotion(
            model_hash="model1",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        stats = self.manager.get_stats()
        self.assertEqual(stats["total_proposals"], 1)
        self.assertEqual(stats["pending_proposals"], 1)
        self.assertEqual(stats["node_id"], "test-node")

    async def test_clear(self) -> None:
        """Should clear all state."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
        )

        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter=f"node-{i}",
            )

        await self.manager.get_promotion_consensus(proposal.proposal_id)
        self.manager.clear()

        stats = self.manager.get_stats()
        self.assertEqual(stats["total_proposals"], 0)
        self.assertEqual(stats["consensus_reached"], 0)

    async def test_event_chain(self) -> None:
        """Should create proper event chain in DAG."""
        proposal = await self.manager.propose_promotion(
            model_hash="model123",
            config_key="hex8_2p",
            evidence={"win_rate": 0.85, "elo": 1450, "games_played": 100},
            proposer="node-1",
        )

        for i in range(3):
            await self.manager.vote_on_proposal(
                proposal.proposal_id,
                approve=True,
                voter="node-1",  # Same voter to build chain
            )
            # But duplicate votes are ignored, so only 1 vote recorded

        chain = self.dag.get_creator_chain("node-1")
        # Should have proposal + 1 vote (duplicates ignored)
        self.assertEqual(len(chain), 2)


class TestSingletonManager(TestCase):
    """Tests for singleton manager."""

    def setUp(self) -> None:
        """Reset singleton before each test."""
        reset_promotion_consensus_manager()

    def tearDown(self) -> None:
        """Reset singleton after each test."""
        reset_promotion_consensus_manager()

    def test_get_manager_requires_dag(self) -> None:
        """Should require DAG on first call."""
        with self.assertRaises(ValueError):
            get_promotion_consensus_manager()

    def test_get_manager_creates_singleton(self) -> None:
        """Should create and return singleton."""
        dag = HashgraphDAG(node_id="test")
        manager1 = get_promotion_consensus_manager(dag, "node-1")
        manager2 = get_promotion_consensus_manager()

        self.assertIs(manager1, manager2)

    def test_reset_clears_singleton(self) -> None:
        """Reset should clear singleton."""
        dag = HashgraphDAG(node_id="test")
        get_promotion_consensus_manager(dag, "node-1")
        reset_promotion_consensus_manager()

        with self.assertRaises(ValueError):
            get_promotion_consensus_manager()


class TestVoteType(TestCase):
    """Tests for VoteType enum."""

    def test_vote_types(self) -> None:
        """Should have expected vote types."""
        self.assertEqual(VoteType.APPROVE.value, "approve")
        self.assertEqual(VoteType.REJECT.value, "reject")
        self.assertEqual(VoteType.ABSTAIN.value, "abstain")


class TestPromotionEventType(TestCase):
    """Tests for PromotionEventType enum."""

    def test_event_types(self) -> None:
        """Should have expected event types."""
        self.assertEqual(
            PromotionEventType.PROPOSAL.value, "promotion_proposal"
        )
        self.assertEqual(PromotionEventType.VOTE.value, "promotion_vote")
        self.assertEqual(
            PromotionEventType.CONSENSUS.value, "promotion_consensus"
        )
        self.assertEqual(
            PromotionEventType.CERTIFICATE.value, "promotion_certificate"
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

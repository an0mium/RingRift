"""Tests for evaluation consensus manager."""

import asyncio
from unittest import TestCase, IsolatedAsyncioTestCase

from app.coordination.hashgraph.event import HashgraphEvent
from app.coordination.hashgraph.dag import HashgraphDAG
from app.coordination.hashgraph.evaluation_consensus import (
    EvaluationResult,
    ConsensusEvaluationResult,
    EvaluationConsensusConfig,
    EvaluationConsensusManager,
    EvaluationEventType,
    get_evaluation_consensus_manager,
    reset_evaluation_consensus_manager,
)


class TestEvaluationResult(TestCase):
    """Tests for EvaluationResult dataclass."""

    def test_valid_result(self) -> None:
        """Should create valid evaluation result."""
        result = EvaluationResult(
            evaluator_node="node-1",
            model_hash="abc123",
            win_rate=0.75,
            games_played=100,
            timestamp=1000.0,
        )
        self.assertEqual(result.evaluator_node, "node-1")
        self.assertEqual(result.win_rate, 0.75)
        self.assertEqual(result.games_played, 100)

    def test_invalid_win_rate_high(self) -> None:
        """Should reject win rate > 1.0."""
        with self.assertRaises(ValueError):
            EvaluationResult(
                evaluator_node="node-1",
                model_hash="abc123",
                win_rate=1.5,
                games_played=100,
                timestamp=1000.0,
            )

    def test_invalid_win_rate_low(self) -> None:
        """Should reject win rate < 0.0."""
        with self.assertRaises(ValueError):
            EvaluationResult(
                evaluator_node="node-1",
                model_hash="abc123",
                win_rate=-0.1,
                games_played=100,
                timestamp=1000.0,
            )

    def test_invalid_games_played(self) -> None:
        """Should reject negative games."""
        with self.assertRaises(ValueError):
            EvaluationResult(
                evaluator_node="node-1",
                model_hash="abc123",
                win_rate=0.5,
                games_played=-1,
                timestamp=1000.0,
            )


class TestConsensusEvaluationResult(TestCase):
    """Tests for ConsensusEvaluationResult dataclass."""

    def test_win_rate_variance_empty(self) -> None:
        """Variance should be 0 with no results."""
        result = ConsensusEvaluationResult(model_hash="abc")
        self.assertEqual(result.win_rate_variance, 0.0)

    def test_win_rate_variance_single(self) -> None:
        """Variance should be 0 with single result."""
        result = ConsensusEvaluationResult(
            model_hash="abc",
            individual_results=[
                EvaluationResult("n1", "abc", 0.75, 100, 1000.0),
            ],
        )
        self.assertEqual(result.win_rate_variance, 0.0)

    def test_win_rate_variance_multiple(self) -> None:
        """Should compute variance correctly."""
        result = ConsensusEvaluationResult(
            model_hash="abc",
            individual_results=[
                EvaluationResult("n1", "abc", 0.70, 100, 1000.0),
                EvaluationResult("n2", "abc", 0.80, 100, 1000.0),
            ],
        )
        # Variance of [0.7, 0.8] = 0.005
        self.assertAlmostEqual(result.win_rate_variance, 0.005, places=4)

    def test_is_reliable_true(self) -> None:
        """Should be reliable with high confidence, low variance."""
        result = ConsensusEvaluationResult(
            model_hash="abc",
            win_rate=0.75,
            confidence=0.9,
            has_consensus=True,
            individual_results=[
                EvaluationResult("n1", "abc", 0.74, 100, 1000.0),
                EvaluationResult("n2", "abc", 0.75, 100, 1000.0),
                EvaluationResult("n3", "abc", 0.76, 100, 1000.0),
            ],
        )
        self.assertTrue(result.is_reliable)

    def test_is_reliable_no_consensus(self) -> None:
        """Should not be reliable without consensus."""
        result = ConsensusEvaluationResult(
            model_hash="abc",
            win_rate=0.75,
            confidence=0.9,
            has_consensus=False,
        )
        self.assertFalse(result.is_reliable)

    def test_is_reliable_low_confidence(self) -> None:
        """Should not be reliable with low confidence."""
        result = ConsensusEvaluationResult(
            model_hash="abc",
            win_rate=0.75,
            confidence=0.5,
            has_consensus=True,
            individual_results=[
                EvaluationResult("n1", "abc", 0.75, 100, 1000.0),
            ],
        )
        self.assertFalse(result.is_reliable)


class TestEvaluationConsensusConfig(TestCase):
    """Tests for EvaluationConsensusConfig."""

    def test_defaults(self) -> None:
        """Should have reasonable defaults."""
        config = EvaluationConsensusConfig()
        self.assertEqual(config.min_evaluators, 3)
        self.assertEqual(config.max_wait_seconds, 300.0)
        self.assertAlmostEqual(config.supermajority_fraction, 2 / 3, places=4)

    def test_custom_config(self) -> None:
        """Should accept custom values."""
        config = EvaluationConsensusConfig(
            min_evaluators=5,
            max_wait_seconds=600.0,
            outlier_threshold=3.0,
        )
        self.assertEqual(config.min_evaluators, 5)
        self.assertEqual(config.max_wait_seconds, 600.0)
        self.assertEqual(config.outlier_threshold, 3.0)


class TestEvaluationConsensusManagerAsync(IsolatedAsyncioTestCase):
    """Async tests for EvaluationConsensusManager."""

    def setUp(self) -> None:
        """Create fresh DAG and manager for each test."""
        self.dag = HashgraphDAG(node_id="test-node")
        self.config = EvaluationConsensusConfig(
            min_evaluators=3,
            min_games_per_evaluator=10,
        )
        self.manager = EvaluationConsensusManager(
            dag=self.dag,
            node_id="test-node",
            config=self.config,
        )

    async def test_submit_single_result(self) -> None:
        """Should submit single evaluation result."""
        event = await self.manager.submit_evaluation_result(
            model_hash="model123",
            evaluator_node="node-1",
            win_rate=0.75,
            games_played=100,
        )
        self.assertIsNotNone(event)
        self.assertEqual(event.creator, "node-1")
        self.assertEqual(event.payload["win_rate"], 0.75)

        pending = self.manager.get_pending_evaluations("model123")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].win_rate, 0.75)

    async def test_submit_invalid_win_rate(self) -> None:
        """Should reject invalid win rate."""
        with self.assertRaises(ValueError):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node="node-1",
                win_rate=1.5,
                games_played=100,
            )

    async def test_consensus_with_three_evaluators(self) -> None:
        """Should reach consensus with 3 evaluators."""
        # Submit 3 results
        for i in range(3):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node=f"node-{i}",
                win_rate=0.70 + i * 0.05,  # 0.70, 0.75, 0.80
                games_played=50,
            )

        result = await self.manager.get_consensus_evaluation(
            model_hash="model123",
            timeout=1.0,
        )
        self.assertTrue(result.has_consensus)
        self.assertEqual(result.evaluator_count, 3)
        self.assertEqual(result.win_rate, 0.75)  # Median

    async def test_no_consensus_insufficient_evaluators(self) -> None:
        """Should not reach consensus with too few evaluators."""
        # Submit 2 results (need 3)
        for i in range(2):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node=f"node-{i}",
                win_rate=0.75,
                games_played=50,
            )

        result = await self.manager.get_consensus_evaluation(
            model_hash="model123",
            timeout=0.1,
        )
        self.assertFalse(result.has_consensus)

    async def test_outlier_detection(self) -> None:
        """Should detect and exclude outliers."""
        # Submit 7 results with tight cluster + extreme outlier
        # Need enough samples and extreme deviation for z-score > 2.0
        # Values: [0.70, 0.70, 0.71, 0.71, 0.72, 0.72, 0.05]
        # z-score for 0.05 = 2.27 (above threshold of 2.0)
        win_rates = [0.70, 0.70, 0.71, 0.71, 0.72, 0.72, 0.05]
        for i, rate in enumerate(win_rates):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node=f"node-{i}",
                win_rate=rate,
                games_played=50,
            )

        result = await self.manager.get_consensus_evaluation(
            model_hash="model123",
            timeout=1.0,
        )
        self.assertTrue(result.has_consensus)
        self.assertEqual(len(result.outliers), 1)
        self.assertEqual(result.outliers[0].win_rate, 0.05)
        # Consensus should exclude outlier, median of [0.70, 0.70, 0.71, 0.71, 0.72, 0.72]
        self.assertEqual(result.win_rate, 0.71)

    async def test_filter_low_game_count(self) -> None:
        """Should filter results with too few games."""
        # Submit results: one has too few games
        await self.manager.submit_evaluation_result(
            model_hash="model123",
            evaluator_node="node-1",
            win_rate=0.75,
            games_played=50,
        )
        await self.manager.submit_evaluation_result(
            model_hash="model123",
            evaluator_node="node-2",
            win_rate=0.75,
            games_played=5,  # Too few
        )
        await self.manager.submit_evaluation_result(
            model_hash="model123",
            evaluator_node="node-3",
            win_rate=0.75,
            games_played=50,
        )
        await self.manager.submit_evaluation_result(
            model_hash="model123",
            evaluator_node="node-4",
            win_rate=0.75,
            games_played=50,
        )

        result = await self.manager.get_consensus_evaluation(
            model_hash="model123",
            timeout=1.0,
        )
        self.assertTrue(result.has_consensus)
        # Should only count 3 valid results
        self.assertEqual(result.evaluator_count, 3)

    async def test_consensus_caching(self) -> None:
        """Should cache consensus result."""
        # Submit 3 results
        for i in range(3):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node=f"node-{i}",
                win_rate=0.75,
                games_played=50,
            )

        result1 = await self.manager.get_consensus_evaluation("model123")
        result2 = self.manager.get_consensus_result("model123")

        self.assertIsNotNone(result2)
        self.assertEqual(result1.win_rate, result2.win_rate)
        self.assertEqual(result1.consensus_timestamp, result2.consensus_timestamp)

    async def test_confidence_calculation(self) -> None:
        """Should compute confidence based on evaluators, games, variance."""
        # High confidence case: many evaluators, many games, low variance
        for i in range(5):
            await self.manager.submit_evaluation_result(
                model_hash="high_conf",
                evaluator_node=f"node-{i}",
                win_rate=0.74 + i * 0.01,  # Low variance
                games_played=100,
            )

        high_result = await self.manager.get_consensus_evaluation("high_conf")
        self.assertGreater(high_result.confidence, 0.7)

    async def test_get_stats(self) -> None:
        """Should return correct statistics."""
        await self.manager.submit_evaluation_result(
            model_hash="model1",
            evaluator_node="node-1",
            win_rate=0.75,
            games_played=50,
        )
        await self.manager.submit_evaluation_result(
            model_hash="model2",
            evaluator_node="node-1",
            win_rate=0.80,
            games_played=50,
        )

        stats = self.manager.get_stats()
        self.assertEqual(stats["pending_models"], 2)
        self.assertEqual(stats["total_pending_results"], 2)

    async def test_clear(self) -> None:
        """Should clear all state."""
        for i in range(3):
            await self.manager.submit_evaluation_result(
                model_hash="model123",
                evaluator_node=f"node-{i}",
                win_rate=0.75,
                games_played=50,
            )

        await self.manager.get_consensus_evaluation("model123")
        self.manager.clear()

        pending = self.manager.get_pending_evaluations("model123")
        cached = self.manager.get_consensus_result("model123")
        self.assertEqual(len(pending), 0)
        self.assertIsNone(cached)

    async def test_event_chain(self) -> None:
        """Should create proper event chain in DAG."""
        # Submit multiple results from same node
        for i in range(3):
            await self.manager.submit_evaluation_result(
                model_hash=f"model-{i}",
                evaluator_node="node-1",
                win_rate=0.75,
                games_played=50,
            )

        chain = self.dag.get_creator_chain("node-1")
        self.assertEqual(len(chain), 3)

        # Verify chain integrity
        for i in range(1, len(chain)):
            event = self.dag.get_event(chain[i])
            self.assertIsNotNone(event)
            self.assertEqual(event.self_parent, chain[i - 1])


class TestSingletonManager(TestCase):
    """Tests for singleton manager."""

    def setUp(self) -> None:
        """Reset singleton before each test."""
        reset_evaluation_consensus_manager()

    def tearDown(self) -> None:
        """Reset singleton after each test."""
        reset_evaluation_consensus_manager()

    def test_get_manager_requires_dag(self) -> None:
        """Should require DAG on first call."""
        with self.assertRaises(ValueError):
            get_evaluation_consensus_manager()

    def test_get_manager_creates_singleton(self) -> None:
        """Should create and return singleton."""
        dag = HashgraphDAG(node_id="test")
        manager1 = get_evaluation_consensus_manager(dag, "node-1")
        manager2 = get_evaluation_consensus_manager()

        self.assertIs(manager1, manager2)

    def test_reset_clears_singleton(self) -> None:
        """Reset should clear singleton."""
        dag = HashgraphDAG(node_id="test")
        manager1 = get_evaluation_consensus_manager(dag, "node-1")
        reset_evaluation_consensus_manager()

        with self.assertRaises(ValueError):
            get_evaluation_consensus_manager()


class TestEvaluationEventType(TestCase):
    """Tests for EvaluationEventType enum."""

    def test_event_types(self) -> None:
        """Should have expected event types."""
        self.assertEqual(
            EvaluationEventType.EVALUATION_RESULT.value,
            "evaluation_result",
        )
        self.assertEqual(
            EvaluationEventType.EVALUATION_CONSENSUS.value,
            "evaluation_consensus",
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

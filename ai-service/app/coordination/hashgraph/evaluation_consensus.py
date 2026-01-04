"""Hashgraph-inspired consensus for model evaluation results.

Uses virtual voting to aggregate evaluation results from multiple nodes,
providing Byzantine fault tolerance for Elo ratings.

Key benefits:
1. Survives faulty evaluation nodes (GPU errors, timeouts)
2. Prevents single-node manipulation of Elo
3. Enables parallel evaluation across cluster
4. Provides audit trail of evaluation decisions

Usage:
    manager = EvaluationConsensusManager(dag)

    # Submit local evaluation result
    await manager.submit_evaluation_result(
        model_hash="abc123",
        evaluator_node="node-1",
        win_rate=0.75,
        games_played=100,
    )

    # Wait for consensus
    result = await manager.get_consensus_evaluation(
        model_hash="abc123",
        min_evaluators=3,
    )

    if result.has_consensus:
        print(f"Consensus win rate: {result.win_rate}")
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.coordination.hashgraph.event import HashgraphEvent, EventType
from app.coordination.hashgraph.dag import HashgraphDAG
from app.coordination.hashgraph.consensus import ConsensusEngine

logger = logging.getLogger(__name__)


class EvaluationEventType(str, Enum):
    """Types of evaluation-related events."""

    EVALUATION_RESULT = "evaluation_result"
    EVALUATION_CONSENSUS = "evaluation_consensus"
    EVALUATION_DISPUTE = "evaluation_dispute"


@dataclass
class EvaluationResult:
    """Single evaluation result from one node.

    Attributes:
        evaluator_node: Node that performed the evaluation
        model_hash: Hash of the evaluated model
        win_rate: Win rate from evaluation (0.0 to 1.0)
        games_played: Number of games in evaluation
        timestamp: When evaluation completed
        event_hash: Hash of the event containing this result
    """

    evaluator_node: str
    model_hash: str
    win_rate: float
    games_played: int
    timestamp: float
    event_hash: str = ""

    def __post_init__(self) -> None:
        """Validate win rate."""
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"Win rate must be 0.0-1.0, got {self.win_rate}")
        if self.games_played < 0:
            raise ValueError(f"Games played must be non-negative, got {self.games_played}")


@dataclass
class ConsensusEvaluationResult:
    """Consensus result for a model evaluation.

    Attributes:
        model_hash: Hash of the evaluated model
        win_rate: Consensus win rate (median of submitted results)
        confidence: Confidence level (0.0 to 1.0)
        games_total: Total games across all evaluators
        evaluator_count: Number of evaluators that submitted
        has_consensus: Whether consensus was reached
        individual_results: All individual results used
        consensus_timestamp: When consensus was reached
        outliers: Results flagged as outliers
    """

    model_hash: str
    win_rate: float = 0.0
    confidence: float = 0.0
    games_total: int = 0
    evaluator_count: int = 0
    has_consensus: bool = False
    individual_results: list[EvaluationResult] = field(default_factory=list)
    consensus_timestamp: float = 0.0
    outliers: list[EvaluationResult] = field(default_factory=list)

    @property
    def win_rate_variance(self) -> float:
        """Variance in win rates across evaluators."""
        if len(self.individual_results) < 2:
            return 0.0
        rates = [r.win_rate for r in self.individual_results]
        return statistics.variance(rates)

    @property
    def is_reliable(self) -> bool:
        """Check if result is reliable (high confidence, low variance)."""
        return (
            self.has_consensus
            and self.confidence >= 0.75
            and self.win_rate_variance < 0.05  # 5% variance threshold
        )


@dataclass
class EvaluationConsensusConfig:
    """Configuration for evaluation consensus.

    Attributes:
        min_evaluators: Minimum evaluators for consensus (default: 3)
        max_wait_seconds: Max time to wait for consensus (default: 300)
        outlier_threshold: Z-score threshold for outlier detection (default: 2.0)
        min_games_per_evaluator: Minimum games each evaluator must play (default: 10)
        supermajority_fraction: Fraction for supermajority (default: 2/3)
    """

    min_evaluators: int = 3
    max_wait_seconds: float = 300.0
    outlier_threshold: float = 2.0
    min_games_per_evaluator: int = 10
    supermajority_fraction: float = 2 / 3


class EvaluationConsensusManager:
    """Hashgraph-inspired consensus for model evaluation results.

    Uses virtual voting to aggregate evaluation results from multiple nodes.
    Provides Byzantine fault tolerance for Elo ratings.

    Attributes:
        dag: The hashgraph DAG for event storage
        engine: The consensus engine for virtual voting
        config: Configuration for consensus parameters
        node_id: Local node identifier
    """

    def __init__(
        self,
        dag: HashgraphDAG,
        node_id: str = "",
        config: EvaluationConsensusConfig | None = None,
    ):
        """Initialize evaluation consensus manager.

        Args:
            dag: Hashgraph DAG for event storage
            node_id: Local node identifier
            config: Optional configuration
        """
        self.dag = dag
        self.engine = ConsensusEngine(dag)
        self.config = config or EvaluationConsensusConfig()
        self.node_id = node_id or dag.node_id

        # Track pending evaluations
        self._pending: dict[str, list[EvaluationResult]] = {}
        self._consensus_reached: dict[str, ConsensusEvaluationResult] = {}
        self._consensus_waiters: dict[str, list[asyncio.Event]] = {}
        self._lock = asyncio.Lock()

    async def submit_evaluation_result(
        self,
        model_hash: str,
        evaluator_node: str,
        win_rate: float,
        games_played: int,
        timestamp: float | None = None,
    ) -> HashgraphEvent:
        """Submit local evaluation result to consensus.

        Creates a hashgraph event with the evaluation result and adds
        it to the DAG.

        Args:
            model_hash: Hash of the evaluated model
            evaluator_node: Node that performed evaluation
            win_rate: Win rate from evaluation (0.0 to 1.0)
            games_played: Number of games played
            timestamp: Optional timestamp (uses current time if not provided)

        Returns:
            The created hashgraph event

        Raises:
            ValueError: If win_rate is not in [0.0, 1.0]
        """
        if not 0.0 <= win_rate <= 1.0:
            raise ValueError(f"Win rate must be 0.0-1.0, got {win_rate}")

        timestamp = timestamp or time.time()

        # Get latest event from this node for self_parent
        latest = self.dag.get_latest_event(evaluator_node)
        self_parent = latest.event_hash if latest else None

        # Compute round for the new event
        round_number = self.dag.compute_round_for_parents(self_parent, None)

        # Create evaluation result event
        event = HashgraphEvent.create(
            creator=evaluator_node,
            payload={
                "type": EvaluationEventType.EVALUATION_RESULT.value,
                "model_hash": model_hash,
                "win_rate": win_rate,
                "games_played": games_played,
            },
            self_parent=self_parent,
            round_number=round_number,
            timestamp=timestamp,
        )

        # Add to DAG
        if not self.dag.add_event(event):
            logger.warning(
                f"[{self.node_id}] Failed to add evaluation event: {event.event_hash[:8]}"
            )
            # Return event anyway for tracking
            return event

        # Track result
        result = EvaluationResult(
            evaluator_node=evaluator_node,
            model_hash=model_hash,
            win_rate=win_rate,
            games_played=games_played,
            timestamp=timestamp,
            event_hash=event.event_hash,
        )

        async with self._lock:
            if model_hash not in self._pending:
                self._pending[model_hash] = []
            self._pending[model_hash].append(result)

            # Check if we have enough for consensus
            await self._check_consensus(model_hash)

        return event

    async def get_consensus_evaluation(
        self,
        model_hash: str,
        min_evaluators: int | None = None,
        timeout: float | None = None,
    ) -> ConsensusEvaluationResult:
        """Get Byzantine-tolerant consensus on evaluation result.

        Waits for sufficient evaluators and computes consensus.

        Args:
            model_hash: Hash of the model to get consensus for
            min_evaluators: Override minimum evaluators (uses config default)
            timeout: Override timeout (uses config default)

        Returns:
            Consensus evaluation result

        Note:
            Returns a result with has_consensus=False if timeout or
            insufficient evaluators.
        """
        min_evaluators = min_evaluators or self.config.min_evaluators
        timeout = timeout or self.config.max_wait_seconds

        # Check if already have consensus
        async with self._lock:
            if model_hash in self._consensus_reached:
                return self._consensus_reached[model_hash]

            # Check if we have enough results now
            if model_hash in self._pending:
                if len(self._pending[model_hash]) >= min_evaluators:
                    return await self._compute_consensus(model_hash)

            # Set up waiter
            if model_hash not in self._consensus_waiters:
                self._consensus_waiters[model_hash] = []
            waiter = asyncio.Event()
            self._consensus_waiters[model_hash].append(waiter)

        # Wait for consensus
        try:
            await asyncio.wait_for(waiter.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.debug(f"[{self.node_id}] Consensus timeout for {model_hash[:8]}")
            pass

        # Return whatever we have
        async with self._lock:
            if model_hash in self._consensus_reached:
                return self._consensus_reached[model_hash]

            # Compute partial consensus
            return await self._compute_consensus(model_hash, require_consensus=False)

    async def _check_consensus(self, model_hash: str) -> None:
        """Check if consensus can be reached and notify waiters.

        Must be called with _lock held.
        """
        if model_hash not in self._pending:
            return

        results = self._pending[model_hash]
        if len(results) < self.config.min_evaluators:
            return

        # Compute consensus
        consensus = await self._compute_consensus(model_hash)

        if consensus.has_consensus:
            # Notify waiters
            if model_hash in self._consensus_waiters:
                for waiter in self._consensus_waiters[model_hash]:
                    waiter.set()
                del self._consensus_waiters[model_hash]

    async def _compute_consensus(
        self,
        model_hash: str,
        require_consensus: bool = True,
    ) -> ConsensusEvaluationResult:
        """Compute consensus from collected results.

        Uses median for robustness against outliers.

        Args:
            model_hash: Model to compute consensus for
            require_consensus: Whether to require supermajority

        Returns:
            Consensus result
        """
        if model_hash not in self._pending:
            return ConsensusEvaluationResult(
                model_hash=model_hash,
                has_consensus=False,
            )

        results = self._pending[model_hash]

        # Filter out results with too few games
        valid_results = [
            r for r in results if r.games_played >= self.config.min_games_per_evaluator
        ]

        if not valid_results:
            return ConsensusEvaluationResult(
                model_hash=model_hash,
                has_consensus=False,
                evaluator_count=len(results),
            )

        # Detect outliers using z-score
        win_rates = [r.win_rate for r in valid_results]
        outliers: list[EvaluationResult] = []
        filtered_results = valid_results

        if len(win_rates) >= 3:
            mean = statistics.mean(win_rates)
            stdev = statistics.stdev(win_rates)

            if stdev > 0:
                filtered_results = []
                for r in valid_results:
                    z_score = abs(r.win_rate - mean) / stdev
                    if z_score > self.config.outlier_threshold:
                        outliers.append(r)
                    else:
                        filtered_results.append(r)

        # Use filtered results if we have enough, otherwise use all
        if len(filtered_results) >= self.config.min_evaluators:
            final_results = filtered_results
        else:
            final_results = valid_results
            outliers = []

        # Compute consensus values
        win_rates = [r.win_rate for r in final_results]
        games = [r.games_played for r in final_results]

        consensus_win_rate = statistics.median(win_rates)
        total_games = sum(games)
        evaluator_count = len(final_results)

        # Check if we have supermajority
        supermajority_count = int(len(results) * self.config.supermajority_fraction)
        has_consensus = (
            evaluator_count >= max(self.config.min_evaluators, supermajority_count)
        )

        if require_consensus and not has_consensus:
            return ConsensusEvaluationResult(
                model_hash=model_hash,
                has_consensus=False,
                evaluator_count=evaluator_count,
                individual_results=final_results,
                outliers=outliers,
            )

        # Compute confidence based on:
        # - Number of evaluators
        # - Total games played
        # - Win rate variance
        evaluator_confidence = min(1.0, evaluator_count / 5.0)  # Max at 5 evaluators
        games_confidence = min(1.0, total_games / 500.0)  # Max at 500 games

        variance = 0.0
        if len(win_rates) >= 2:
            variance = statistics.variance(win_rates)
        variance_confidence = max(0.0, 1.0 - variance * 10)  # Penalty for high variance

        confidence = (
            evaluator_confidence * 0.3
            + games_confidence * 0.4
            + variance_confidence * 0.3
        )

        consensus_result = ConsensusEvaluationResult(
            model_hash=model_hash,
            win_rate=consensus_win_rate,
            confidence=confidence,
            games_total=total_games,
            evaluator_count=evaluator_count,
            has_consensus=has_consensus,
            individual_results=final_results,
            consensus_timestamp=time.time(),
            outliers=outliers,
        )

        # Cache result
        if has_consensus:
            self._consensus_reached[model_hash] = consensus_result

        return consensus_result

    def get_pending_evaluations(self, model_hash: str) -> list[EvaluationResult]:
        """Get pending evaluation results for a model.

        Args:
            model_hash: Model hash to query

        Returns:
            List of pending evaluation results
        """
        return list(self._pending.get(model_hash, []))

    def get_consensus_result(self, model_hash: str) -> ConsensusEvaluationResult | None:
        """Get cached consensus result if available.

        Args:
            model_hash: Model hash to query

        Returns:
            Consensus result if reached, None otherwise
        """
        return self._consensus_reached.get(model_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dictionary with pending counts, consensus counts, etc.
        """
        return {
            "pending_models": len(self._pending),
            "consensus_reached": len(self._consensus_reached),
            "total_pending_results": sum(len(r) for r in self._pending.values()),
            "node_id": self.node_id,
        }

    def clear(self) -> None:
        """Clear all pending and cached results."""
        self._pending.clear()
        self._consensus_reached.clear()
        for waiters in self._consensus_waiters.values():
            for waiter in waiters:
                waiter.set()
        self._consensus_waiters.clear()


# Singleton instance
_evaluation_consensus_manager: EvaluationConsensusManager | None = None


def get_evaluation_consensus_manager(
    dag: HashgraphDAG | None = None,
    node_id: str = "",
    config: EvaluationConsensusConfig | None = None,
) -> EvaluationConsensusManager:
    """Get or create the singleton evaluation consensus manager.

    Args:
        dag: Hashgraph DAG (required on first call)
        node_id: Local node identifier
        config: Configuration

    Returns:
        The singleton manager instance

    Raises:
        ValueError: If dag is None on first call
    """
    global _evaluation_consensus_manager

    if _evaluation_consensus_manager is None:
        if dag is None:
            raise ValueError("DAG required for initial manager creation")
        _evaluation_consensus_manager = EvaluationConsensusManager(
            dag=dag,
            node_id=node_id,
            config=config,
        )

    return _evaluation_consensus_manager


def reset_evaluation_consensus_manager() -> None:
    """Reset the singleton manager (for testing)."""
    global _evaluation_consensus_manager
    _evaluation_consensus_manager = None


__all__ = [
    "EvaluationEventType",
    "EvaluationResult",
    "ConsensusEvaluationResult",
    "EvaluationConsensusConfig",
    "EvaluationConsensusManager",
    "get_evaluation_consensus_manager",
    "reset_evaluation_consensus_manager",
]

"""Work Distribution Adapter for Cluster-Wide Task Coordination.

This module bridges training/evaluation pipelines to the centralized work queue,
enabling distributed execution across 50+ cluster nodes.

Integration Points:
- Training requests → WorkQueue (WorkType.TRAINING)
- Evaluation requests → WorkQueue (WorkType.TOURNAMENT/GAUNTLET)
- CMAES optimization → WorkQueue (WorkType.GPU_CMAES/CPU_CMAES)
- Model sync → WorkQueue (WorkType.DATA_SYNC)

Event Integration:
- Emits WORK_SUBMITTED when work is queued
- Emits WORK_CLAIMED when work is assigned
- Subscribes to TRAINING_THRESHOLD_REACHED to auto-queue training

Usage:
    from app.coordination.work_distributor import get_work_distributor

    distributor = get_work_distributor()

    # Submit training work
    work_id = await distributor.submit_training(
        board="square8",
        num_players=2,
        epochs=100,
        priority=80,
    )

    # Submit evaluation work
    work_id = await distributor.submit_evaluation(
        candidate_model="nnue_v7",
        baseline_model="nnue_v6",
        games=200,
    )

    # Get status
    status = distributor.get_work_status(work_id)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_work_queue = None
_WorkItem = None
_WorkType = None
_WorkStatus = None


def _get_work_queue():
    """Lazy load WorkQueue to avoid circular imports."""
    global _work_queue, _WorkItem, _WorkType, _WorkStatus

    if _work_queue is None:
        try:
            from app.coordination.work_queue import (
                WorkItem,
                WorkQueue,
                WorkStatus,
                WorkType,
            )

            _WorkItem = WorkItem
            _WorkType = WorkType
            _WorkStatus = WorkStatus

            # Use shared work queue instance
            db_path = os.environ.get(
                "RINGRIFT_WORK_QUEUE_DB",
                "data/work_queue.db"
            )
            _work_queue = WorkQueue(db_path=db_path)
            logger.info(f"Work distributor connected to queue at {db_path}")

        except ImportError as e:
            logger.warning(f"WorkQueue not available: {e}")
            return None

    return _work_queue


@dataclass
class DistributedWorkConfig:
    """Configuration for distributed work submission."""

    # Node selection
    require_gpu: bool = False
    require_high_memory: bool = False  # For square19/hexagonal
    preferred_nodes: list[str] | None = None

    # Scheduling
    priority: int = 50  # 0-100, higher = more urgent
    timeout_seconds: float = 3600.0  # 1 hour default
    max_attempts: int = 3

    # Dependencies
    depends_on: list[str] | None = None


class WorkDistributor:
    """Distributes work across the cluster via the central work queue."""

    def __init__(self):
        self._queue = None
        self._local_submissions: dict[str, dict[str, Any]] = {}
        self._event_callbacks: list[callable] = []

    def _ensure_queue(self):
        """Ensure work queue is available."""
        if self._queue is None:
            self._queue = _get_work_queue()
        return self._queue is not None

    # =========================================================================
    # Training Submission
    # =========================================================================

    async def submit_training(
        self,
        board: str,
        num_players: int,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 3e-4,
        checkpoint_path: str | None = None,
        db_paths: list[str] | None = None,
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a training job to the cluster work queue.

        Args:
            board: Board type (square8, square19, hexagonal).
            num_players: Number of players (2, 3, 4).
            epochs: Training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            checkpoint_path: Optional path to resume from.
            db_paths: Optional list of database paths.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            logger.warning("Work queue not available, cannot submit training")
            return None

        config = config or DistributedWorkConfig()

        # Determine priority based on data availability
        priority = config.priority

        # Higher priority for underrepresented configs
        if board in ("square19", "hexagonal"):
            priority = min(100, priority + 20)
        if num_players in (3, 4):
            priority = min(100, priority + 10)

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "checkpoint_path": checkpoint_path,
            "db_paths": db_paths or [],
            "require_gpu": config.require_gpu,
            "require_high_memory": config.require_high_memory or board != "square8",
        }

        item = _WorkItem(
            work_type=_WorkType.TRAINING,
            priority=priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            depends_on=config.depends_on or [],
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted training work {work_id}: {board}_{num_players}p")

        # Track locally
        self._local_submissions[work_id] = {
            "type": "training",
            "submitted_at": time.time(),
            "config": work_config,
        }

        # Emit event
        await self._emit_work_submitted(work_id, "training", work_config)

        return work_id

    # =========================================================================
    # Evaluation Submission
    # =========================================================================

    async def submit_evaluation(
        self,
        candidate_model: str,
        baseline_model: str | None = None,
        games: int = 200,
        board: str = "square8",
        num_players: int = 2,
        evaluation_type: str = "gauntlet",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit an evaluation job to the cluster work queue.

        Args:
            candidate_model: Path/ID of candidate model.
            baseline_model: Path/ID of baseline model (optional).
            games: Number of games to play.
            board: Board type.
            num_players: Number of players.
            evaluation_type: 'gauntlet' or 'tournament'.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            logger.warning("Work queue not available, cannot submit evaluation")
            return None

        config = config or DistributedWorkConfig()

        work_type = (
            _WorkType.GAUNTLET if evaluation_type == "gauntlet"
            else _WorkType.TOURNAMENT
        )

        work_config = {
            "candidate_model": candidate_model,
            "baseline_model": baseline_model,
            "games": games,
            "board_type": board,
            "num_players": num_players,
        }

        item = _WorkItem(
            work_type=work_type,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
            depends_on=config.depends_on or [],
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted {evaluation_type} work {work_id}")

        self._local_submissions[work_id] = {
            "type": evaluation_type,
            "submitted_at": time.time(),
            "config": work_config,
        }

        await self._emit_work_submitted(work_id, evaluation_type, work_config)

        return work_id

    # =========================================================================
    # CMAES Optimization Submission
    # =========================================================================

    async def submit_cmaes(
        self,
        board: str,
        num_players: int,
        generations: int = 50,
        population_size: int = 20,
        use_gpu: bool = True,
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a CMAES optimization job.

        Args:
            board: Board type.
            num_players: Number of players.
            generations: Number of generations.
            population_size: Population size.
            use_gpu: Whether to use GPU acceleration.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_type = _WorkType.GPU_CMAES if use_gpu else _WorkType.CPU_CMAES

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "generations": generations,
            "population_size": population_size,
        }

        item = _WorkItem(
            work_type=work_type,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted CMAES work {work_id}")

        return work_id

    # =========================================================================
    # Selfplay Submission
    # =========================================================================

    async def submit_selfplay(
        self,
        board: str,
        num_players: int,
        games: int = 1000,
        ai_type: str = "gumbel-mcts",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a selfplay job.

        Args:
            board: Board type.
            num_players: Number of players.
            games: Number of games to generate.
            ai_type: AI type to use.
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_config = {
            "board_type": board,
            "num_players": num_players,
            "games": games,
            "ai_type": ai_type,
        }

        item = _WorkItem(
            work_type=_WorkType.SELFPLAY,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
            max_attempts=config.max_attempts,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted selfplay work {work_id}: {board}_{num_players}p, {games} games")

        return work_id

    # =========================================================================
    # Data Sync Submission
    # =========================================================================

    async def submit_data_sync(
        self,
        source_path: str,
        target_nodes: list[str] | None = None,
        sync_type: str = "model",
        config: DistributedWorkConfig | None = None,
    ) -> str | None:
        """Submit a data sync job to distribute data across cluster.

        Args:
            source_path: Path to data to sync.
            target_nodes: Specific nodes to sync to (None = all).
            sync_type: Type of sync ('model', 'database', 'checkpoint').
            config: Distributed work configuration.

        Returns:
            Work ID if submitted successfully, None otherwise.
        """
        if not self._ensure_queue():
            return None

        config = config or DistributedWorkConfig()

        work_config = {
            "source_path": source_path,
            "target_nodes": target_nodes,
            "sync_type": sync_type,
        }

        item = _WorkItem(
            work_type=_WorkType.DATA_SYNC,
            priority=config.priority,
            config=work_config,
            timeout_seconds=config.timeout_seconds,
        )

        work_id = self._queue.add_work(item)
        logger.info(f"Submitted data sync work {work_id}: {sync_type}")

        return work_id

    # =========================================================================
    # Status and Monitoring
    # =========================================================================

    def get_work_status(self, work_id: str) -> dict[str, Any] | None:
        """Get status of a submitted work item."""
        if not self._ensure_queue():
            return None

        item = self._queue.get_work_item(work_id)
        if item is None:
            return None

        return {
            "work_id": item.work_id,
            "status": item.status.value,
            "work_type": item.work_type.value,
            "priority": item.priority,
            "claimed_by": item.claimed_by,
            "attempts": item.attempts,
            "created_at": item.created_at,
            "result": item.result,
            "error": item.error,
        }

    def get_queue_stats(self) -> dict[str, Any]:
        """Get overall queue statistics."""
        if not self._ensure_queue():
            return {"available": False}

        return self._queue.get_stats()

    def get_pending_work(
        self,
        work_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get pending work items."""
        if not self._ensure_queue():
            return []

        items = self._queue.get_pending(limit=limit)

        if work_type:
            items = [i for i in items if i.work_type.value == work_type]

        return [i.to_dict() for i in items]

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def submit_multiconfig_training(
        self,
        configs: list[tuple[str, int]],  # List of (board, num_players)
        epochs: int = 100,
        base_priority: int = 50,
    ) -> list[str]:
        """Submit training for multiple configurations.

        Args:
            configs: List of (board, num_players) tuples.
            epochs: Training epochs.
            base_priority: Base priority (adjusted per config).

        Returns:
            List of work IDs.
        """
        work_ids = []

        for board, num_players in configs:
            work_id = await self.submit_training(
                board=board,
                num_players=num_players,
                epochs=epochs,
                config=DistributedWorkConfig(priority=base_priority),
            )
            if work_id:
                work_ids.append(work_id)

        return work_ids

    async def submit_crossboard_evaluation(
        self,
        candidate_model: str,
        games_per_config: int = 200,
    ) -> list[str]:
        """Submit evaluations for all 9 board/player configurations.

        Args:
            candidate_model: Model to evaluate.
            games_per_config: Games per configuration.

        Returns:
            List of work IDs.
        """
        from app.training.crossboard_strength import ALL_BOARD_CONFIGS

        work_ids = []

        for board, num_players in ALL_BOARD_CONFIGS:
            work_id = await self.submit_evaluation(
                candidate_model=candidate_model,
                games=games_per_config,
                board=board,
                num_players=num_players,
            )
            if work_id:
                work_ids.append(work_id)

        return work_ids

    # =========================================================================
    # Event Integration
    # =========================================================================

    async def _emit_work_submitted(
        self,
        work_id: str,
        work_type: str,
        config: dict[str, Any],
    ) -> None:
        """Emit event when work is submitted."""
        try:
            from app.distributed.event_helpers import emit_event_safe

            await emit_event_safe(
                "WORK_SUBMITTED",
                {
                    "work_id": work_id,
                    "work_type": work_type,
                    "config": config,
                },
                "work_distributor"
            )
        except Exception as e:
            logger.debug(f"Could not emit work submitted event: {e}")


# Global instance
_distributor_instance: WorkDistributor | None = None


def get_work_distributor() -> WorkDistributor:
    """Get the global WorkDistributor instance."""
    global _distributor_instance
    if _distributor_instance is None:
        _distributor_instance = WorkDistributor()
    return _distributor_instance


# Convenience functions for common operations
async def distribute_training(
    board: str,
    num_players: int,
    **kwargs,
) -> str | None:
    """Convenience function to distribute training work."""
    return await get_work_distributor().submit_training(
        board=board,
        num_players=num_players,
        **kwargs,
    )


async def distribute_evaluation(
    candidate_model: str,
    **kwargs,
) -> str | None:
    """Convenience function to distribute evaluation work."""
    return await get_work_distributor().submit_evaluation(
        candidate_model=candidate_model,
        **kwargs,
    )


async def distribute_selfplay(
    board: str,
    num_players: int,
    games: int = 1000,
    **kwargs,
) -> str | None:
    """Convenience function to distribute selfplay work."""
    return await get_work_distributor().submit_selfplay(
        board=board,
        num_players=num_players,
        games=games,
        **kwargs,
    )

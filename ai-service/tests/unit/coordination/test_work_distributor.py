"""Tests for work_distributor module.

Tests the WorkDistributor class that bridges training/evaluation pipelines
to the centralized work queue for distributed execution across cluster nodes.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.work_distributor import (
    # Data classes
    DistributedWorkConfig,
    # Main class
    WorkDistributor,
    get_work_distributor,
    # Convenience functions
    distribute_training,
    distribute_evaluation,
    distribute_selfplay,
)


# ============================================================================
# DistributedWorkConfig Tests
# ============================================================================


class TestDistributedWorkConfig:
    """Tests for DistributedWorkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = DistributedWorkConfig()
        assert config.require_gpu is False
        assert config.require_high_memory is False
        assert config.preferred_nodes is None
        assert config.priority == 50
        assert config.timeout_seconds == 3600.0
        assert config.max_attempts == 3
        assert config.depends_on is None

    def test_custom_values(self) -> None:
        """Test custom values are stored correctly."""
        config = DistributedWorkConfig(
            require_gpu=True,
            require_high_memory=True,
            preferred_nodes=["node1", "node2"],
            priority=90,
            timeout_seconds=7200.0,
            max_attempts=5,
            depends_on=["work-123"],
        )
        assert config.require_gpu is True
        assert config.require_high_memory is True
        assert config.preferred_nodes == ["node1", "node2"]
        assert config.priority == 90
        assert config.timeout_seconds == 7200.0
        assert config.max_attempts == 5
        assert config.depends_on == ["work-123"]


# ============================================================================
# WorkDistributor Initialization Tests
# ============================================================================


class TestWorkDistributorInit:
    """Tests for WorkDistributor initialization."""

    def test_init_creates_empty_state(self) -> None:
        """Test initialization creates empty state."""
        distributor = WorkDistributor()
        assert distributor._queue is None
        assert distributor._local_submissions == {}
        assert distributor._event_callbacks == []

    def test_ensure_queue_lazy_loads(self) -> None:
        """Test _ensure_queue lazy loads the work queue."""
        distributor = WorkDistributor()

        # Mock the lazy load function
        mock_queue = MagicMock()
        with patch("app.coordination.work_distributor._get_work_queue", return_value=mock_queue):
            result = distributor._ensure_queue()

        assert result is True
        assert distributor._queue is mock_queue

    def test_ensure_queue_returns_false_when_unavailable(self) -> None:
        """Test _ensure_queue returns False when queue unavailable."""
        distributor = WorkDistributor()

        with patch("app.coordination.work_distributor._get_work_queue", return_value=None):
            result = distributor._ensure_queue()

        assert result is False


# ============================================================================
# Training Submission Tests
# ============================================================================


class TestSubmitTraining:
    """Tests for training job submission."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "work-123"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_training_basic(self, distributor: WorkDistributor) -> None:
        """Test basic training submission."""
        # Mock WorkItem and WorkType
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.TRAINING = "training"

            work_id = await distributor.submit_training(
                board="square8",
                num_players=2,
                epochs=100,
            )

        assert work_id == "work-123"
        assert "work-123" in distributor._local_submissions
        assert distributor._local_submissions["work-123"]["type"] == "training"

    @pytest.mark.asyncio
    async def test_submit_training_with_custom_config(self, distributor: WorkDistributor) -> None:
        """Test training submission with custom config."""
        config = DistributedWorkConfig(
            require_gpu=True,
            priority=80,
            timeout_seconds=7200.0,
        )

        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.TRAINING = "training"

            work_id = await distributor.submit_training(
                board="hexagonal",
                num_players=4,
                epochs=200,
                config=config,
            )

        assert work_id == "work-123"

    @pytest.mark.asyncio
    async def test_submit_training_priority_boost_large_board(self, distributor: WorkDistributor) -> None:
        """Test that large boards get priority boost."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.TRAINING = "training"

            await distributor.submit_training(
                board="square19",
                num_players=2,
            )

            # Verify WorkItem was called with boosted priority
            call_args = mock_item.call_args
            # Priority should be boosted by 20 for square19 (50 + 20 = 70)
            assert call_args.kwargs["priority"] >= 70

    @pytest.mark.asyncio
    async def test_submit_training_priority_boost_multiplayer(self, distributor: WorkDistributor) -> None:
        """Test that 3/4 player configs get priority boost."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.TRAINING = "training"

            await distributor.submit_training(
                board="square8",
                num_players=4,
            )

            # Verify WorkItem was called with boosted priority
            call_args = mock_item.call_args
            # Priority should be boosted by 10 for 4 players (50 + 10 = 60)
            assert call_args.kwargs["priority"] >= 60

    @pytest.mark.asyncio
    async def test_submit_training_returns_none_when_queue_unavailable(self) -> None:
        """Test training submission returns None when queue unavailable."""
        distributor = WorkDistributor()

        with patch("app.coordination.work_distributor._get_work_queue", return_value=None):
            work_id = await distributor.submit_training(
                board="square8",
                num_players=2,
            )

        assert work_id is None


# ============================================================================
# Evaluation Submission Tests
# ============================================================================


class TestSubmitEvaluation:
    """Tests for evaluation job submission."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "eval-456"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_evaluation_gauntlet(self, distributor: WorkDistributor) -> None:
        """Test gauntlet evaluation submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.GAUNTLET = "gauntlet"
            mock_type.TOURNAMENT = "tournament"

            work_id = await distributor.submit_evaluation(
                candidate_model="nnue_v7",
                baseline_model="nnue_v6",
                games=200,
                evaluation_type="gauntlet",
            )

        assert work_id == "eval-456"
        assert distributor._local_submissions["eval-456"]["type"] == "gauntlet"

    @pytest.mark.asyncio
    async def test_submit_evaluation_tournament(self, distributor: WorkDistributor) -> None:
        """Test tournament evaluation submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.GAUNTLET = "gauntlet"
            mock_type.TOURNAMENT = "tournament"

            work_id = await distributor.submit_evaluation(
                candidate_model="nnue_v7",
                games=100,
                evaluation_type="tournament",
            )

        assert work_id == "eval-456"
        assert distributor._local_submissions["eval-456"]["type"] == "tournament"

    @pytest.mark.asyncio
    async def test_submit_evaluation_returns_none_when_unavailable(self) -> None:
        """Test evaluation submission returns None when queue unavailable."""
        distributor = WorkDistributor()

        with patch("app.coordination.work_distributor._get_work_queue", return_value=None):
            work_id = await distributor.submit_evaluation(
                candidate_model="test_model",
            )

        assert work_id is None


# ============================================================================
# CMAES Submission Tests
# ============================================================================


class TestSubmitCmaes:
    """Tests for CMAES optimization submission."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "cmaes-789"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_cmaes_gpu(self, distributor: WorkDistributor) -> None:
        """Test GPU CMAES submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.GPU_CMAES = "gpu_cmaes"
            mock_type.CPU_CMAES = "cpu_cmaes"

            work_id = await distributor.submit_cmaes(
                board="square8",
                num_players=2,
                generations=50,
                use_gpu=True,
            )

        assert work_id == "cmaes-789"
        # Verify GPU type was used
        call_args = mock_item.call_args
        assert call_args.kwargs["work_type"] == "gpu_cmaes"

    @pytest.mark.asyncio
    async def test_submit_cmaes_cpu(self, distributor: WorkDistributor) -> None:
        """Test CPU CMAES submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.GPU_CMAES = "gpu_cmaes"
            mock_type.CPU_CMAES = "cpu_cmaes"

            work_id = await distributor.submit_cmaes(
                board="hex8",
                num_players=2,
                use_gpu=False,
            )

        assert work_id == "cmaes-789"
        call_args = mock_item.call_args
        assert call_args.kwargs["work_type"] == "cpu_cmaes"


# ============================================================================
# Selfplay Submission Tests
# ============================================================================


class TestSubmitSelfplay:
    """Tests for selfplay job submission."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "selfplay-111"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_selfplay(self, distributor: WorkDistributor) -> None:
        """Test selfplay submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.SELFPLAY = "selfplay"

            work_id = await distributor.submit_selfplay(
                board="square8",
                num_players=2,
                games=1000,
                ai_type="gumbel-mcts",
            )

        assert work_id == "selfplay-111"

    @pytest.mark.asyncio
    async def test_submit_selfplay_custom_games(self, distributor: WorkDistributor) -> None:
        """Test selfplay with custom game count."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.SELFPLAY = "selfplay"

            await distributor.submit_selfplay(
                board="hexagonal",
                num_players=4,
                games=5000,
            )

            call_args = mock_item.call_args
            assert call_args.kwargs["config"]["games"] == 5000


# ============================================================================
# Data Sync Submission Tests
# ============================================================================


class TestSubmitDataSync:
    """Tests for data sync job submission."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "sync-222"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_data_sync_model(self, distributor: WorkDistributor) -> None:
        """Test model sync submission."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.DATA_SYNC = "data_sync"

            work_id = await distributor.submit_data_sync(
                source_path="/models/best.pth",
                sync_type="model",
            )

        assert work_id == "sync-222"

    @pytest.mark.asyncio
    async def test_submit_data_sync_with_targets(self, distributor: WorkDistributor) -> None:
        """Test data sync with specific target nodes."""
        with patch("app.coordination.work_distributor._WorkItem") as mock_item, \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.DATA_SYNC = "data_sync"

            await distributor.submit_data_sync(
                source_path="/data/games.db",
                target_nodes=["node1", "node2"],
                sync_type="database",
            )

            call_args = mock_item.call_args
            assert call_args.kwargs["config"]["target_nodes"] == ["node1", "node2"]


# ============================================================================
# Status and Monitoring Tests
# ============================================================================


class TestStatusAndMonitoring:
    """Tests for status and monitoring methods."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        return distributor

    def test_get_work_status_found(self, distributor: WorkDistributor) -> None:
        """Test getting status of existing work."""
        mock_item = MagicMock()
        mock_item.work_id = "work-123"
        mock_item.status.value = "pending"
        mock_item.work_type.value = "training"
        mock_item.priority = 50
        mock_item.claimed_by = None
        mock_item.attempts = 0
        mock_item.created_at = 1234567890.0
        mock_item.result = None
        mock_item.error = None

        distributor._queue.get_work_item.return_value = mock_item

        status = distributor.get_work_status("work-123")

        assert status is not None
        assert status["work_id"] == "work-123"
        assert status["status"] == "pending"
        assert status["work_type"] == "training"

    def test_get_work_status_not_found(self, distributor: WorkDistributor) -> None:
        """Test getting status of non-existent work."""
        distributor._queue.get_work_item.return_value = None

        status = distributor.get_work_status("missing-work")

        assert status is None

    def test_get_queue_stats(self, distributor: WorkDistributor) -> None:
        """Test getting queue statistics."""
        distributor._queue.get_queue_status.return_value = {
            "total_items": 100,
            "by_status": {"pending": 50, "running": 30, "completed": 20},
        }

        stats = distributor.get_queue_stats()

        assert stats["total_items"] == 100

    def test_get_queue_stats_unavailable(self) -> None:
        """Test queue stats when queue unavailable."""
        distributor = WorkDistributor()

        with patch("app.coordination.work_distributor._get_work_queue", return_value=None):
            stats = distributor.get_queue_stats()

        assert stats == {"available": False}

    def test_get_pending_work(self, distributor: WorkDistributor) -> None:
        """Test getting pending work items."""
        mock_item = MagicMock()
        mock_item.work_type.value = "training"
        mock_item.to_dict.return_value = {"work_id": "work-1", "type": "training"}

        distributor._queue.get_pending.return_value = [mock_item]

        pending = distributor.get_pending_work()

        assert len(pending) == 1
        assert pending[0]["work_id"] == "work-1"

    def test_get_pending_work_filtered(self, distributor: WorkDistributor) -> None:
        """Test getting pending work filtered by type."""
        mock_training = MagicMock()
        mock_training.work_type.value = "training"
        mock_training.to_dict.return_value = {"work_id": "work-1", "type": "training"}

        mock_eval = MagicMock()
        mock_eval.work_type.value = "evaluation"
        mock_eval.to_dict.return_value = {"work_id": "work-2", "type": "evaluation"}

        distributor._queue.get_pending.return_value = [mock_training, mock_eval]

        pending = distributor.get_pending_work(work_type="training")

        assert len(pending) == 1
        assert pending[0]["type"] == "training"


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestBatchOperations:
    """Tests for batch submission operations."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        # Generate sequential work IDs
        distributor._queue.add_work.side_effect = lambda _: f"work-{len(distributor._local_submissions)}"
        return distributor

    @pytest.mark.asyncio
    async def test_submit_multiconfig_training(self, distributor: WorkDistributor) -> None:
        """Test submitting training for multiple configurations."""
        with patch("app.coordination.work_distributor._WorkItem"), \
             patch("app.coordination.work_distributor._WorkType") as mock_type:
            mock_type.TRAINING = "training"

            work_ids = await distributor.submit_multiconfig_training(
                configs=[
                    ("square8", 2),
                    ("hex8", 2),
                    ("square19", 4),
                ],
                epochs=50,
            )

        assert len(work_ids) == 3

    @pytest.mark.asyncio
    async def test_submit_crossboard_evaluation(self, distributor: WorkDistributor) -> None:
        """Test submitting evaluations for all configurations."""
        with patch("app.coordination.work_distributor._WorkItem"), \
             patch("app.coordination.work_distributor._WorkType") as mock_type, \
             patch("app.training.crossboard_strength.ALL_BOARD_CONFIGS", [
                 ("square8", 2),
                 ("hex8", 2),
                 ("square19", 2),
             ]):
            mock_type.GAUNTLET = "gauntlet"
            mock_type.TOURNAMENT = "tournament"

            work_ids = await distributor.submit_crossboard_evaluation(
                candidate_model="test_model",
                games_per_config=100,
            )

        assert len(work_ids) == 3


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_queue_unavailable(self) -> None:
        """Test health check when queue unavailable."""
        distributor = WorkDistributor()

        with patch("app.coordination.work_distributor._get_work_queue", return_value=None):
            result = distributor.health_check()

        assert result.healthy is False
        assert "not available" in result.message.lower()

    def test_health_check_healthy(self) -> None:
        """Test health check when distributor is healthy."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.get_queue_status.return_value = {
            "total_items": 50,
            "by_status": {"pending": 10, "failed": 2},
        }

        result = distributor.health_check()

        assert result.healthy is True
        assert result.message == "Distributor healthy"

    def test_health_check_high_pending(self) -> None:
        """Test health check with high pending count."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.get_queue_status.return_value = {
            "total_items": 600,
            "by_status": {"pending": 550, "failed": 5},
        }

        result = distributor.health_check()

        assert result.healthy is False
        assert "High pending count" in result.message

    def test_health_check_high_failure(self) -> None:
        """Test health check with high failure count."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.get_queue_status.return_value = {
            "total_items": 100,
            "by_status": {"pending": 20, "failed": 60},
        }

        result = distributor.health_check()

        assert result.healthy is False
        assert "High failure count" in result.message


# ============================================================================
# Singleton Tests
# ============================================================================


class TestSingleton:
    """Tests for get_work_distributor singleton."""

    def test_returns_same_instance(self) -> None:
        """Test singleton returns same instance."""
        # Reset singleton
        import app.coordination.work_distributor as mod
        mod._distributor_instance = None

        dist1 = get_work_distributor()
        dist2 = get_work_distributor()

        assert dist1 is dist2

    def test_singleton_can_be_reset(self) -> None:
        """Test singleton can be reset for testing."""
        import app.coordination.work_distributor as mod

        mod._distributor_instance = None
        dist1 = get_work_distributor()

        mod._distributor_instance = None
        dist2 = get_work_distributor()

        assert dist1 is not dist2


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_distribute_training(self) -> None:
        """Test distribute_training convenience function."""
        import app.coordination.work_distributor as mod
        mod._distributor_instance = None

        with patch.object(WorkDistributor, "submit_training", new_callable=AsyncMock) as mock:
            mock.return_value = "work-123"

            work_id = await distribute_training(
                board="square8",
                num_players=2,
                epochs=100,
            )

        assert work_id == "work-123"
        mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_distribute_evaluation(self) -> None:
        """Test distribute_evaluation convenience function."""
        import app.coordination.work_distributor as mod
        mod._distributor_instance = None

        with patch.object(WorkDistributor, "submit_evaluation", new_callable=AsyncMock) as mock:
            mock.return_value = "eval-456"

            work_id = await distribute_evaluation(
                candidate_model="test_model",
                games=200,
            )

        assert work_id == "eval-456"

    @pytest.mark.asyncio
    async def test_distribute_selfplay(self) -> None:
        """Test distribute_selfplay convenience function."""
        import app.coordination.work_distributor as mod
        mod._distributor_instance = None

        with patch.object(WorkDistributor, "submit_selfplay", new_callable=AsyncMock) as mock:
            mock.return_value = "selfplay-789"

            work_id = await distribute_selfplay(
                board="hex8",
                num_players=4,
                games=500,
            )

        assert work_id == "selfplay-789"
        mock.assert_called_once_with(
            board="hex8",
            num_players=4,
            games=500,
        )


# ============================================================================
# Event Integration Tests
# ============================================================================


class TestEventIntegration:
    """Tests for event emission integration."""

    @pytest.fixture
    def distributor(self) -> WorkDistributor:
        """Create distributor with mocked queue."""
        distributor = WorkDistributor()
        distributor._queue = MagicMock()
        distributor._queue.add_work.return_value = "work-123"
        return distributor

    @pytest.mark.asyncio
    async def test_emits_work_submitted_event(self, distributor: WorkDistributor) -> None:
        """Test that WORK_SUBMITTED event is emitted."""
        with patch("app.coordination.work_distributor._WorkItem"), \
             patch("app.coordination.work_distributor._WorkType") as mock_type, \
             patch("app.distributed.event_helpers.emit_event_safe", new_callable=AsyncMock) as mock_emit:
            mock_type.TRAINING = "training"

            await distributor.submit_training(
                board="square8",
                num_players=2,
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "WORK_SUBMITTED"

    @pytest.mark.asyncio
    async def test_event_emission_failure_doesnt_break_submission(self, distributor: WorkDistributor) -> None:
        """Test that event emission failure doesn't break work submission."""
        with patch("app.coordination.work_distributor._WorkItem"), \
             patch("app.coordination.work_distributor._WorkType") as mock_type, \
             patch("app.distributed.event_helpers.emit_event_safe", side_effect=Exception("Event error")):
            mock_type.TRAINING = "training"

            # Should still succeed even if event emission fails
            work_id = await distributor.submit_training(
                board="square8",
                num_players=2,
            )

            assert work_id == "work-123"

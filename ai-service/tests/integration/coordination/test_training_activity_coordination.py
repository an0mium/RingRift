"""Integration tests for TrainingActivityDaemon coordination flow.

Tests the complete flow:
1. TrainingActivityDaemon detects training activity (local or P2P)
2. Emits TRAINING_STARTED event
3. Triggers priority sync via SyncFacade
4. SyncFacade calls AutoSyncDaemon.trigger_priority_sync()
5. Sync completion emits DATA_SYNC_COMPLETED
6. DataPipelineOrchestrator receives event and processes

Created: December 27, 2025
Purpose: Verify training detection → sync → pipeline coordination flow
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reset_singletons():
    """Reset all singleton instances before and after each test."""
    from app.coordination.training_activity_daemon import reset_training_activity_daemon

    reset_training_activity_daemon()
    yield
    reset_training_activity_daemon()


# =============================================================================
# Event Flow Tests
# =============================================================================


class TestTrainingActivityToSyncFlow:
    """Tests for TrainingActivityDaemon → SyncFacade flow."""

    @pytest.mark.asyncio
    async def test_training_detected_triggers_priority_sync(self, reset_singletons):
        """When training is detected, priority sync should be triggered."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        # Create daemon with sync enabled
        config = TrainingActivityConfig(
            check_interval_seconds=1,
            trigger_priority_sync=True,
        )
        daemon = TrainingActivityDaemon(config=config)

        # Track if priority sync was called
        sync_calls = []

        async def mock_trigger_priority_sync(reason: str, data_type: str = "games"):
            sync_calls.append({"reason": reason, "data_type": data_type})

        # Mock the sync facade and event emission
        with patch("app.coordination.sync_facade.get_sync_facade") as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock(
                side_effect=mock_trigger_priority_sync
            )
            mock_get_facade.return_value = mock_facade

            with patch(
                "app.distributed.data_events.emit_training_started"
            ) as mock_emit:
                mock_emit.return_value = AsyncMock()()

                # Simulate training detection
                new_training_nodes = {"node-1", "node-2"}
                await daemon._on_training_detected(new_training_nodes)

        # Verify sync was triggered
        assert len(sync_calls) == 1, "Priority sync should be triggered once"
        assert "node-1" in sync_calls[0]["reason"] or "node-2" in sync_calls[0]["reason"]
        assert sync_calls[0]["data_type"] == "games"

    @pytest.mark.asyncio
    async def test_training_detected_emits_event_per_node(self, reset_singletons):
        """TRAINING_STARTED event should be emitted for each training node."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig(
            check_interval_seconds=1,
            trigger_priority_sync=False,  # Disable sync for this test
        )
        daemon = TrainingActivityDaemon(config=config)

        emitted_events = []

        async def mock_emit(node_id: str, source: str):
            emitted_events.append({"node_id": node_id, "source": source})

        with patch(
            "app.distributed.data_events.emit_training_started",
            new=mock_emit,
        ):
            new_training_nodes = {"node-1", "node-2", "node-3"}
            await daemon._on_training_detected(new_training_nodes)

        # Verify event emitted for each node
        assert len(emitted_events) == 3, "Event should be emitted for each training node"
        node_ids = {e["node_id"] for e in emitted_events}
        assert node_ids == {"node-1", "node-2", "node-3"}

    @pytest.mark.asyncio
    async def test_graceful_shutdown_triggers_final_sync(self, reset_singletons):
        """Graceful shutdown should trigger final priority sync."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig(
            check_interval_seconds=1,
            trigger_priority_sync=True,
        )
        daemon = TrainingActivityDaemon(config=config)

        sync_calls = []

        async def mock_sync(reason: str, data_type: str = "games"):
            sync_calls.append({"reason": reason, "data_type": data_type})

        with patch("app.coordination.sync_facade.get_sync_facade") as mock_get_facade:
            mock_facade = MagicMock()
            mock_facade.trigger_priority_sync = AsyncMock(side_effect=mock_sync)
            mock_get_facade.return_value = mock_facade

            # Trigger graceful shutdown hook
            await daemon._on_graceful_shutdown()

        # Verify final sync was triggered
        assert len(sync_calls) == 1, "Final sync should be triggered on shutdown"
        assert "termination" in sync_calls[0]["reason"]


class TestP2PTrainingDetection:
    """Tests for P2P-based training detection."""

    @pytest.mark.asyncio
    async def test_parse_p2p_status_detects_training_jobs(self, reset_singletons):
        """P2P status with training jobs should detect training nodes."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig()
        daemon = TrainingActivityDaemon(config=config)

        # Mock P2P status response
        mock_status = {
            "peers": {
                "node-training": {
                    "running_jobs": [{"type": "training", "config": "hex8_2p"}],
                    "processes": [],
                },
                "node-selfplay": {
                    "running_jobs": [{"type": "selfplay", "config": "hex8_2p"}],
                    "processes": [],
                },
                "node-idle": {
                    "running_jobs": [],
                    "processes": [],
                },
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=mock_status):
            training_nodes = await daemon._check_p2p_training()

        # Only training node should be detected
        assert training_nodes == {"node-training"}

    @pytest.mark.asyncio
    async def test_parse_p2p_status_detects_training_processes(self, reset_singletons):
        """P2P status with training processes should detect training nodes."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig(
            training_process_patterns=["app.training.train", "train.py"]
        )
        daemon = TrainingActivityDaemon(config=config)

        mock_status = {
            "peers": {
                "node-training": {
                    "running_jobs": [],
                    "processes": ["python app.training.train --epochs 100"],
                },
                "node-other": {
                    "running_jobs": [],
                    "processes": ["python selfplay.py"],
                },
            }
        }

        with patch.object(daemon, "_get_p2p_status", return_value=mock_status):
            training_nodes = await daemon._check_p2p_training()

        assert training_nodes == {"node-training"}


class TestLocalTrainingDetection:
    """Tests for local process-based training detection."""

    @pytest.mark.asyncio
    async def test_local_training_adds_self_node(self, reset_singletons):
        """Local training detection should add self to training nodes."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig(check_interval_seconds=1)
        daemon = TrainingActivityDaemon(config=config)

        # Mock P2P returning no training nodes
        with patch.object(daemon, "_check_p2p_training", return_value=set()):
            # Mock local training detection returning True
            with patch.object(daemon, "detect_local_training", return_value=True):
                # Track detected training nodes
                detected_nodes = None

                async def capture_detection(nodes):
                    nonlocal detected_nodes
                    detected_nodes = nodes

                with patch.object(
                    daemon, "_on_training_detected", side_effect=capture_detection
                ):
                    await daemon._run_cycle()

        # Self node should be in training nodes
        assert daemon.node_id in daemon._training_nodes


class TestRunCycleIntegration:
    """Tests for the full run cycle integration."""

    @pytest.mark.asyncio
    async def test_run_cycle_tracks_state_changes(self, reset_singletons):
        """Run cycle should track training node state changes correctly."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig(
            check_interval_seconds=0.1,
            trigger_priority_sync=False,
        )
        daemon = TrainingActivityDaemon(config=config)

        # First cycle: node-1 starts training
        with patch.object(daemon, "_check_p2p_training", return_value={"node-1"}):
            with patch.object(daemon, "detect_local_training", return_value=False):
                with patch(
                    "app.distributed.data_events.emit_training_started"
                ) as mock_emit:
                    mock_emit.return_value = AsyncMock()()
                    await daemon._run_cycle()

        assert daemon._training_nodes == {"node-1"}

        # Second cycle: node-1 still training, node-2 joins
        with patch.object(
            daemon, "_check_p2p_training", return_value={"node-1", "node-2"}
        ):
            with patch.object(daemon, "detect_local_training", return_value=False):
                with patch(
                    "app.distributed.data_events.emit_training_started"
                ) as mock_emit:
                    mock_emit.return_value = AsyncMock()()
                    await daemon._run_cycle()

        assert daemon._training_nodes == {"node-1", "node-2"}

        # Third cycle: node-1 completes, node-2 still training
        with patch.object(daemon, "_check_p2p_training", return_value={"node-2"}):
            with patch.object(daemon, "detect_local_training", return_value=False):
                await daemon._run_cycle()

        assert daemon._training_nodes == {"node-2"}


class TestHealthCheckIntegration:
    """Tests for health check integration."""

    @pytest.mark.asyncio
    async def test_health_check_reports_tracking_count(self, reset_singletons):
        """Health check should report number of tracked training nodes."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig()
        daemon = TrainingActivityDaemon(config=config)
        daemon._running = True
        daemon._training_nodes = {"node-1", "node-2", "node-3"}

        health = daemon.health_check()

        assert health.healthy is True
        assert "3 training nodes" in health.message
        assert sorted(health.details["training_nodes"]) == ["node-1", "node-2", "node-3"]

    @pytest.mark.asyncio
    async def test_get_status_includes_sync_count(self, reset_singletons):
        """Status should include sync triggered count."""
        from app.coordination.training_activity_daemon import (
            TrainingActivityConfig,
            TrainingActivityDaemon,
        )

        config = TrainingActivityConfig()
        daemon = TrainingActivityDaemon(config=config)
        daemon._running = True
        daemon._syncs_triggered = 5

        status = daemon.get_status()

        assert status["syncs_triggered"] == 5
        assert status["config"]["trigger_priority_sync"] is True


class TestSyncFacadeIntegration:
    """Tests for SyncFacade.trigger_priority_sync integration."""

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_exists(self):
        """SyncFacade should have trigger_priority_sync method."""
        from app.coordination.sync_facade import SyncFacade

        facade = SyncFacade()
        assert hasattr(facade, "trigger_priority_sync")
        assert callable(facade.trigger_priority_sync)

    @pytest.mark.asyncio
    async def test_sync_facade_method_signature(self):
        """trigger_priority_sync should accept required parameters."""
        from app.coordination.sync_facade import SyncFacade
        import inspect

        sig = inspect.signature(SyncFacade.trigger_priority_sync)
        params = list(sig.parameters.keys())

        # Should accept reason at minimum
        assert "reason" in params or len(params) > 1  # self + reason or **kwargs


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_singleton_returns_same_instance(self, reset_singletons):
        """get_training_activity_daemon() should return singleton."""
        from app.coordination.training_activity_daemon import (
            get_training_activity_daemon,
        )

        daemon1 = get_training_activity_daemon()
        daemon2 = get_training_activity_daemon()

        assert daemon1 is daemon2

    def test_reset_singleton_clears_instance(self, reset_singletons):
        """reset_training_activity_daemon() should clear singleton."""
        from app.coordination.training_activity_daemon import (
            get_training_activity_daemon,
            reset_training_activity_daemon,
        )

        daemon1 = get_training_activity_daemon()
        reset_training_activity_daemon()
        daemon2 = get_training_activity_daemon()

        assert daemon1 is not daemon2

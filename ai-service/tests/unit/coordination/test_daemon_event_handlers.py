"""Tests for DaemonEventHandlers.

December 2025: Tests for event handlers extracted from daemon_manager.py.
"""

import asyncio
import socket
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class MockDaemonInfo:
    """Mock daemon info for testing."""

    state: Any
    restart_count: int = 0
    instance: Any = None


@dataclass
class MockEvent:
    """Mock event for testing."""

    payload: dict


class TestDaemonEventHandlersInit:
    """Tests for handler initialization."""

    def test_init_stores_manager_reference(self):
        """Test that __init__ stores manager reference."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        assert handlers._manager is mock_manager

    def test_init_sets_subscribed_false(self):
        """Test that _subscribed is False initially."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        assert handlers._subscribed is False


class TestSubscribeToEvents:
    """Tests for subscribe_to_events method."""

    @pytest.mark.asyncio
    async def test_subscribe_sets_subscribed_flag(self):
        """Test that subscribing sets _subscribed to True."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch("app.coordination.daemon_event_handlers.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            # Mock DataEventType with necessary attributes
            with patch("app.coordination.daemon_event_handlers.DataEventType", create=True) as mock_event_type:
                mock_event_type.REGRESSION_CRITICAL = MagicMock(value="regression_critical")
                mock_event_type.SELFPLAY_TARGET_UPDATED = MagicMock(value="selfplay_target_updated")
                mock_event_type.EXPLORATION_BOOST = MagicMock(value="exploration_boost")
                mock_event_type.DAEMON_STATUS_CHANGED = MagicMock(value="daemon_status_changed")
                mock_event_type.HOST_OFFLINE = MagicMock(value="host_offline")
                mock_event_type.HOST_ONLINE = MagicMock(value="host_online")
                mock_event_type.LEADER_ELECTED = MagicMock(value="leader_elected")
                mock_event_type.BACKPRESSURE_ACTIVATED = MagicMock(value="backpressure_activated")
                mock_event_type.BACKPRESSURE_RELEASED = MagicMock(value="backpressure_released")
                mock_event_type.DISK_SPACE_LOW = MagicMock(value="disk_space_low")

                await handlers.subscribe_to_events()

        assert handlers._subscribed is True

    @pytest.mark.asyncio
    async def test_subscribe_skips_if_already_subscribed(self):
        """Test that subscribing is skipped if already subscribed."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)
        handlers._subscribed = True

        with patch("app.coordination.daemon_event_handlers.get_router") as mock_get_router:
            await handlers.subscribe_to_events()

        # get_router should not be called if already subscribed
        mock_get_router.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_handles_no_router(self):
        """Test graceful handling when router is not available."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch("app.coordination.daemon_event_handlers.get_router") as mock_get_router:
            with patch("app.coordination.daemon_event_handlers.DataEventType", None):
                mock_get_router.return_value = None

                await handlers.subscribe_to_events()

        assert handlers._subscribed is False

    @pytest.mark.asyncio
    async def test_subscribe_handles_import_error(self):
        """Test graceful handling of ImportError."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch(
            "app.coordination.daemon_event_handlers.get_router",
            side_effect=ImportError("No module"),
        ):
            await handlers.subscribe_to_events()

        assert handlers._subscribed is False


class TestOnRegressionCritical:
    """Tests for _on_regression_critical handler."""

    @pytest.mark.asyncio
    async def test_regression_critical_logs_event(self):
        """Test that critical regression is logged."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "model_id": "model_123",
                "elo_drop": 50,
                "current_elo": 1450,
                "previous_elo": 1500,
            }
        )

        with patch.object(handlers, "_emit_cluster_alert", new_callable=AsyncMock):
            await handlers._on_regression_critical(event)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_regression_critical_emits_alert(self):
        """Test that cluster alert is emitted."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "model_id": "model_123",
                "elo_drop": 50,
                "current_elo": 1450,
                "previous_elo": 1500,
            }
        )

        with patch.object(handlers, "_emit_cluster_alert", new_callable=AsyncMock) as mock_emit:
            await handlers._on_regression_critical(event)

        mock_emit.assert_called_once()
        call_kwargs = mock_emit.call_args[1]
        assert call_kwargs["alert_type"] == "regression_critical"
        assert call_kwargs["config_key"] == "hex8_2p"
        assert call_kwargs["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_regression_critical_checks_rollback_daemon(self):
        """Test that MODEL_DISTRIBUTION daemon is checked."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_manager = MagicMock()
        mock_info = MockDaemonInfo(state=DaemonState.RUNNING)
        mock_manager._daemons = {DaemonType.MODEL_DISTRIBUTION: mock_info}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "model_id": "model_123",
                "elo_drop": 50,
                "current_elo": 1450,
                "previous_elo": 1500,
            }
        )

        with patch.object(handlers, "_emit_cluster_alert", new_callable=AsyncMock):
            await handlers._on_regression_critical(event)

        # Should access the daemon info
        assert DaemonType.MODEL_DISTRIBUTION in mock_manager._daemons

    @pytest.mark.asyncio
    async def test_regression_critical_handles_raw_payload(self):
        """Test handling event without payload attribute."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        # Event is just a dict (raw payload)
        event = {
            "config_key": "hex8_2p",
            "model_id": "model_123",
            "elo_drop": 50,
            "current_elo": 1450,
            "previous_elo": 1500,
        }

        with patch.object(handlers, "_emit_cluster_alert", new_callable=AsyncMock):
            await handlers._on_regression_critical(event)

        # Should not raise any errors


class TestOnSelfplayTargetUpdated:
    """Tests for _on_selfplay_target_updated handler."""

    @pytest.mark.asyncio
    async def test_selfplay_target_updated_logs_event(self):
        """Test that target update is logged."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "priority": "high",
                "reason": "data_stale",
                "target_jobs": 10,
            }
        )

        await handlers._on_selfplay_target_updated(event)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_selfplay_target_triggers_immediate_check_on_high_priority(self):
        """Test that urgent/high priority triggers immediate check."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.trigger_immediate_check = MagicMock()

        mock_info = MockDaemonInfo(state=DaemonState.RUNNING, instance=mock_daemon)
        mock_manager = MagicMock()
        mock_manager._daemons = {DaemonType.IDLE_RESOURCE: mock_info}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "priority": "urgent",
                "reason": "data_stale",
            }
        )

        await handlers._on_selfplay_target_updated(event)

        mock_daemon.trigger_immediate_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_selfplay_target_no_trigger_on_normal_priority(self):
        """Test that normal priority doesn't trigger immediate check."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.trigger_immediate_check = MagicMock()

        mock_info = MockDaemonInfo(state=DaemonState.RUNNING, instance=mock_daemon)
        mock_manager = MagicMock()
        mock_manager._daemons = {DaemonType.IDLE_RESOURCE: mock_info}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "priority": "normal",
                "reason": "routine",
            }
        )

        await handlers._on_selfplay_target_updated(event)

        mock_daemon.trigger_immediate_check.assert_not_called()


class TestOnExplorationBoost:
    """Tests for _on_exploration_boost handler."""

    @pytest.mark.asyncio
    async def test_exploration_boost_logs_event(self):
        """Test that exploration boost is logged."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "boost_factor": 1.5,
                "reason": "elo_plateau",
                "duration_seconds": 7200,
            }
        )

        await handlers._on_exploration_boost(event)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_exploration_boost_applies_to_scheduler(self):
        """Test that boost is applied to selfplay scheduler."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_scheduler = MagicMock()
        mock_scheduler.apply_exploration_boost = MagicMock()

        mock_info = MockDaemonInfo(state=DaemonState.RUNNING, instance=mock_scheduler)
        mock_manager = MagicMock()
        mock_manager._daemons = {DaemonType.SELFPLAY_SCHEDULER: mock_info}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "config_key": "hex8_2p",
                "boost_factor": 1.5,
                "reason": "elo_plateau",
                "duration_seconds": 7200,
            }
        )

        await handlers._on_exploration_boost(event)

        mock_scheduler.apply_exploration_boost.assert_called_once_with(
            config_key="hex8_2p",
            boost_factor=1.5,
            duration_seconds=7200,
        )


class TestOnDaemonStatusChanged:
    """Tests for _on_daemon_status_changed handler."""

    @pytest.mark.asyncio
    async def test_daemon_status_changed_logs_event(self):
        """Test that status change is logged."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "daemon_type": "AUTO_SYNC",
                "old_status": "RUNNING",
                "new_status": "STOPPED",
                "reason": "manual",
            }
        )

        await handlers._on_daemon_status_changed(event)

        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_daemon_status_changed_restarts_failed_daemon(self):
        """Test that failed daemon is restarted."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonType

        mock_lifecycle = AsyncMock()
        mock_info = MockDaemonInfo(state=MagicMock(), restart_count=0)

        mock_manager = MagicMock()
        mock_manager._daemons = {DaemonType.AUTO_SYNC: mock_info}
        mock_manager._lifecycle = mock_lifecycle
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "daemon_type": "AUTO_SYNC",
                "old_status": "RUNNING",
                "new_status": "FAILED",
                "reason": "crash",
            }
        )

        await handlers._on_daemon_status_changed(event)

        mock_lifecycle.restart_daemon.assert_called_once_with(DaemonType.AUTO_SYNC)

    @pytest.mark.asyncio
    async def test_daemon_status_changed_respects_max_restarts(self):
        """Test that max restart limit is respected."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonType

        mock_lifecycle = AsyncMock()
        mock_info = MockDaemonInfo(state=MagicMock(), restart_count=5)  # Exceeds limit

        mock_manager = MagicMock()
        mock_manager._daemons = {DaemonType.AUTO_SYNC: mock_info}
        mock_manager._lifecycle = mock_lifecycle
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "daemon_type": "AUTO_SYNC",
                "old_status": "RUNNING",
                "new_status": "FAILED",
                "reason": "crash",
            }
        )

        await handlers._on_daemon_status_changed(event)

        mock_lifecycle.restart_daemon.assert_not_called()

    @pytest.mark.asyncio
    async def test_daemon_status_changed_ignores_unknown_type(self):
        """Test that unknown daemon type is handled gracefully."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "daemon_type": "UNKNOWN_DAEMON",
                "old_status": "RUNNING",
                "new_status": "FAILED",
            }
        )

        # Should not raise
        await handlers._on_daemon_status_changed(event)


class TestOnHostOffline:
    """Tests for _on_host_offline handler."""

    @pytest.mark.asyncio
    async def test_host_offline_notifies_sync_daemons(self):
        """Test that sync daemons are notified of offline host."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_auto_sync = MagicMock()
        mock_auto_sync.mark_host_offline = MagicMock()

        mock_model_dist = MagicMock()
        mock_model_dist.mark_host_offline = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.AUTO_SYNC: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_auto_sync
            ),
            DaemonType.MODEL_DISTRIBUTION: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_model_dist
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "host_id": "worker-1",
                "reason": "heartbeat_timeout",
            }
        )

        await handlers._on_host_offline(event)

        mock_auto_sync.mark_host_offline.assert_called_once_with("worker-1")
        mock_model_dist.mark_host_offline.assert_called_once_with("worker-1")

    @pytest.mark.asyncio
    async def test_host_offline_handles_no_method(self):
        """Test graceful handling when daemon lacks mark_host_offline."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock(spec=[])  # No mark_host_offline

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.AUTO_SYNC: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(payload={"host_id": "worker-1"})

        # Should not raise
        await handlers._on_host_offline(event)


class TestOnHostOnline:
    """Tests for _on_host_online handler."""

    @pytest.mark.asyncio
    async def test_host_online_notifies_sync_daemons(self):
        """Test that sync daemons are notified of online host."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_auto_sync = MagicMock()
        mock_auto_sync.mark_host_online = MagicMock()

        mock_model_dist = MagicMock()
        mock_model_dist.mark_host_online = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.AUTO_SYNC: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_auto_sync
            ),
            DaemonType.MODEL_DISTRIBUTION: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_model_dist
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(payload={"host_id": "worker-1"})

        await handlers._on_host_online(event)

        mock_auto_sync.mark_host_online.assert_called_once_with("worker-1")
        mock_model_dist.mark_host_online.assert_called_once_with("worker-1")


class TestOnLeaderElected:
    """Tests for _on_leader_elected handler."""

    @pytest.mark.asyncio
    async def test_leader_elected_starts_leader_only_daemons(self):
        """Test that leader-only daemons are started when elected."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_manager = AsyncMock()
        mock_manager._daemons = {
            DaemonType.DATA_PIPELINE: MockDaemonInfo(state=DaemonState.STOPPED),
            DaemonType.AUTO_PROMOTION: MockDaemonInfo(state=DaemonState.STOPPED),
            DaemonType.EVALUATION: MockDaemonInfo(state=DaemonState.STOPPED),
            DaemonType.TRAINING_TRIGGER: MockDaemonInfo(state=DaemonState.STOPPED),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "leader_id": "this-node",
                "previous_leader_id": "old-leader",
                "is_self": True,
            }
        )

        await handlers._on_leader_elected(event)

        # Should call start for each leader-only daemon
        assert mock_manager.start.call_count == 4

    @pytest.mark.asyncio
    async def test_leader_elected_stops_leader_only_daemons_on_loss(self):
        """Test that leader-only daemons are stopped when losing leadership."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_manager = AsyncMock()
        mock_manager._daemons = {
            DaemonType.DATA_PIPELINE: MockDaemonInfo(state=DaemonState.RUNNING),
            DaemonType.AUTO_PROMOTION: MockDaemonInfo(state=DaemonState.RUNNING),
            DaemonType.EVALUATION: MockDaemonInfo(state=DaemonState.RUNNING),
            DaemonType.TRAINING_TRIGGER: MockDaemonInfo(state=DaemonState.RUNNING),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "leader_id": "other-node",
                "previous_leader_id": "this-node",
                "is_self": False,
            }
        )

        await handlers._on_leader_elected(event)

        # Should call stop for each leader-only daemon
        assert mock_manager.stop.call_count == 4

    @pytest.mark.asyncio
    async def test_leader_elected_notifies_running_daemons(self):
        """Test that running daemons are notified of leadership change."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.on_leader_changed = MagicMock()

        mock_manager = AsyncMock()
        mock_manager._daemons = {
            DaemonType.AUTO_SYNC: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "leader_id": "new-leader",
                "is_self": False,
            }
        )

        await handlers._on_leader_elected(event)

        mock_daemon.on_leader_changed.assert_called_once_with(
            leader_id="new-leader", is_self=False
        )


class TestOnBackpressureActivated:
    """Tests for _on_backpressure_activated handler."""

    @pytest.mark.asyncio
    async def test_backpressure_activated_pauses_daemons(self):
        """Test that non-essential daemons are paused."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.pause = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.IDLE_RESOURCE: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "reason": "high_memory",
                "threshold": 80,
                "current_value": 95,
            }
        )

        await handlers._on_backpressure_activated(event)

        assert mock_daemon.pause.call_count == 2

    @pytest.mark.asyncio
    async def test_backpressure_activated_handles_no_pause_method(self):
        """Test graceful handling when daemon lacks pause method."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock(spec=[])  # No pause method

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.IDLE_RESOURCE: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "reason": "high_memory",
                "threshold": 80,
                "current_value": 95,
            }
        )

        # Should not raise
        await handlers._on_backpressure_activated(event)


class TestOnBackpressureReleased:
    """Tests for _on_backpressure_released handler."""

    @pytest.mark.asyncio
    async def test_backpressure_released_resumes_daemons(self):
        """Test that paused daemons are resumed."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.resume = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.IDLE_RESOURCE: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(payload={"duration_seconds": 120.5})

        await handlers._on_backpressure_released(event)

        assert mock_daemon.resume.call_count == 2


class TestOnDiskSpaceLow:
    """Tests for _on_disk_space_low handler."""

    @pytest.mark.asyncio
    async def test_disk_space_low_pauses_daemons_at_critical(self):
        """Test that data-generating daemons are paused at 85%+."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.pause = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
            DaemonType.IDLE_RESOURCE: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        local_hostname = socket.gethostname()
        event = MockEvent(
            payload={
                "host": local_hostname,
                "usage_percent": 90.0,
                "free_gb": 5.0,
                "threshold": 70,
            }
        )

        await handlers._on_disk_space_low(event)

        assert mock_daemon.pause.call_count == 2

    @pytest.mark.asyncio
    async def test_disk_space_low_ignores_other_hosts(self):
        """Test that events for other hosts are ignored."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.pause = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        event = MockEvent(
            payload={
                "host": "other-host-xyz",
                "usage_percent": 90.0,
                "free_gb": 5.0,
                "threshold": 70,
            }
        )

        await handlers._on_disk_space_low(event)

        mock_daemon.pause.assert_not_called()

    @pytest.mark.asyncio
    async def test_disk_space_low_no_action_below_critical(self):
        """Test that no action is taken below 85%."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.pause = MagicMock()

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        local_hostname = socket.gethostname()
        event = MockEvent(
            payload={
                "host": local_hostname,
                "usage_percent": 75.0,  # Below 85% critical threshold
                "free_gb": 20.0,
                "threshold": 70,
            }
        )

        await handlers._on_disk_space_low(event)

        mock_daemon.pause.assert_not_called()


class TestEmitClusterAlert:
    """Tests for _emit_cluster_alert helper."""

    @pytest.mark.asyncio
    async def test_emit_cluster_alert_publishes_event(self):
        """Test that alert event is published."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch("app.coordination.daemon_event_handlers.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_router.publish_async = AsyncMock()
            mock_get_router.return_value = mock_router

            with patch("app.coordination.daemon_event_handlers.DataEventType") as mock_event_type:
                mock_event_type.HEALTH_ALERT = MagicMock(value="health_alert")

                with patch("app.coordination.daemon_event_handlers.DataEvent") as mock_data_event:
                    mock_data_event.return_value = MagicMock()

                    await handlers._emit_cluster_alert(
                        alert_type="test_alert",
                        config_key="hex8_2p",
                        message="Test message",
                        severity="warning",
                        extra_field="extra_value",
                    )

            mock_router.publish_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_cluster_alert_handles_no_router(self):
        """Test graceful handling when router is not available."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch("app.coordination.daemon_event_handlers.get_router") as mock_get_router:
            mock_get_router.return_value = None

            # Should not raise
            await handlers._emit_cluster_alert(
                alert_type="test_alert",
                config_key="hex8_2p",
                message="Test message",
                severity="warning",
            )

    @pytest.mark.asyncio
    async def test_emit_cluster_alert_handles_runtime_error(self):
        """Test graceful handling of RuntimeError."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch(
            "app.coordination.daemon_event_handlers.get_router",
            side_effect=RuntimeError("Router not available"),
        ):
            # Should not raise
            await handlers._emit_cluster_alert(
                alert_type="test_alert",
                config_key="hex8_2p",
                message="Test message",
                severity="warning",
            )


class TestWireRollbackHandler:
    """Tests for _wire_rollback_handler method."""

    def test_wire_rollback_handler_wires_successfully(self):
        """Test successful wiring of rollback handler."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch(
            "app.coordination.daemon_event_handlers.get_model_registry"
        ) as mock_get_registry:
            mock_registry = MagicMock()
            mock_get_registry.return_value = mock_registry

            with patch(
                "app.coordination.daemon_event_handlers.wire_regression_to_rollback"
            ) as mock_wire:
                mock_wire.return_value = MagicMock()

                handlers._wire_rollback_handler()

                mock_wire.assert_called_once_with(mock_registry)

    def test_wire_rollback_handler_handles_import_error(self):
        """Test graceful handling of ImportError."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch(
            "app.coordination.daemon_event_handlers.get_model_registry",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise
            handlers._wire_rollback_handler()

    def test_wire_rollback_handler_handles_runtime_error(self):
        """Test graceful handling of RuntimeError."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        handlers = DaemonEventHandlers(mock_manager)

        with patch(
            "app.coordination.daemon_event_handlers.get_model_registry",
            side_effect=RuntimeError("Registry not initialized"),
        ):
            # Should not raise
            handlers._wire_rollback_handler()


class TestEventPayloadHandling:
    """Tests for event payload handling edge cases."""

    @pytest.mark.asyncio
    async def test_handles_missing_payload_keys(self):
        """Test graceful handling of missing payload keys."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        # Minimal payload
        event = MockEvent(payload={})

        with patch.object(handlers, "_emit_cluster_alert", new_callable=AsyncMock):
            # Should use default values, not raise
            await handlers._on_regression_critical(event)

    @pytest.mark.asyncio
    async def test_handles_dict_event_without_payload_attr(self):
        """Test handling of dict event without payload attribute."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        mock_manager = MagicMock()
        mock_manager._daemons = {}
        handlers = DaemonEventHandlers(mock_manager)

        # Event as plain dict
        event = {"config_key": "hex8_2p", "priority": "normal", "reason": "test"}

        # Should not raise
        await handlers._on_selfplay_target_updated(event)

    @pytest.mark.asyncio
    async def test_handles_exception_in_handler(self):
        """Test that exceptions in handlers are caught."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers
        from app.coordination.daemon_types import DaemonState, DaemonType

        mock_daemon = MagicMock()
        mock_daemon.pause = MagicMock(side_effect=RuntimeError("Pause failed"))

        mock_manager = MagicMock()
        mock_manager._daemons = {
            DaemonType.SELFPLAY_COORDINATOR: MockDaemonInfo(
                state=DaemonState.RUNNING, instance=mock_daemon
            ),
        }
        handlers = DaemonEventHandlers(mock_manager)

        local_hostname = socket.gethostname()
        event = MockEvent(
            payload={
                "host": local_hostname,
                "usage_percent": 90.0,
                "free_gb": 5.0,
                "threshold": 70,
            }
        )

        # Should not raise despite the exception in pause()
        await handlers._on_disk_space_low(event)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_daemon_event_handlers(self):
        """Test that __all__ contains DaemonEventHandlers."""
        from app.coordination.daemon_event_handlers import __all__

        assert "DaemonEventHandlers" in __all__

    def test_can_import_daemon_event_handlers(self):
        """Test that DaemonEventHandlers can be imported."""
        from app.coordination.daemon_event_handlers import DaemonEventHandlers

        assert DaemonEventHandlers is not None

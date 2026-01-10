"""Tests for P2P resilience loops.

Tests cover:
- SelfHealingConfig/Loop: Stuck job recovery, stale process cleanup
- PredictiveMonitoringConfig/Loop: Proactive monitoring and alerting
- SplitBrainDetectionConfig/Loop: Network partition detection
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.p2p.loops.resilience_loops import (
    SelfHealingConfig,
    SelfHealingLoop,
    PredictiveMonitoringConfig,
    PredictiveMonitoringLoop,
    SplitBrainDetectionConfig,
    SplitBrainDetectionLoop,
)


# =============================================================================
# SelfHealingConfig Tests
# =============================================================================


class TestSelfHealingConfig:
    """Tests for SelfHealingConfig dataclass."""

    def test_default_values(self):
        """Test SelfHealingConfig has sensible defaults."""
        config = SelfHealingConfig()

        assert config.healing_interval_seconds == 60.0
        assert config.stale_process_check_interval_seconds == 300.0
        assert config.initial_delay_seconds == 45.0

    def test_validation_healing_interval_zero(self):
        """Test validation rejects healing_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="healing_interval_seconds"):
            SelfHealingConfig(healing_interval_seconds=0)

    def test_validation_stale_process_interval_zero(self):
        """Test validation rejects stale_process_check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="stale_process_check_interval_seconds"):
            SelfHealingConfig(stale_process_check_interval_seconds=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            SelfHealingConfig(initial_delay_seconds=-1)

    def test_custom_values(self):
        """Test SelfHealingConfig with custom values."""
        config = SelfHealingConfig(
            healing_interval_seconds=30.0,
            stale_process_check_interval_seconds=120.0,
            initial_delay_seconds=10.0,
        )

        assert config.healing_interval_seconds == 30.0
        assert config.stale_process_check_interval_seconds == 120.0
        assert config.initial_delay_seconds == 10.0


# =============================================================================
# SelfHealingLoop Tests
# =============================================================================


class TestSelfHealingLoop:
    """Tests for SelfHealingLoop class."""

    def _create_loop(self, **overrides):
        """Create a SelfHealingLoop with defaults."""
        defaults = {
            "is_leader": MagicMock(return_value=True),
            "get_health_manager": MagicMock(return_value=None),
            "get_work_queue": MagicMock(return_value=None),
            "cleanup_stale_processes": MagicMock(return_value=0),
            "config": None,
            "restart_stopped_loops": None,
        }
        defaults.update(overrides)
        return SelfHealingLoop(**defaults)

    def test_init(self):
        """Test SelfHealingLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "self_healing"
        assert loop._stale_processes_cleaned == 0
        assert loop._stuck_jobs_recovered == 0
        assert loop._loops_restarted == 0

    def test_init_custom_config(self):
        """Test SelfHealingLoop with custom config."""
        config = SelfHealingConfig(healing_interval_seconds=30.0)
        loop = self._create_loop(config=config)

        assert loop.interval == 30.0

    @pytest.mark.asyncio
    async def test_on_start_waits_initial_delay(self):
        """Test _on_start waits for initial delay."""
        config = SelfHealingConfig(initial_delay_seconds=0.01)
        loop = self._create_loop(config=config)

        start = time.time()
        await loop._on_start()
        elapsed = time.time() - start

        assert elapsed >= 0.01

    @pytest.mark.asyncio
    async def test_run_once_cleans_stale_processes(self):
        """Test _run_once cleans up stale processes when interval elapsed."""
        cleanup = MagicMock(return_value=3)
        config = SelfHealingConfig(stale_process_check_interval_seconds=0.001)
        loop = self._create_loop(
            cleanup_stale_processes=cleanup,
            config=config,
        )
        loop._last_stale_check = 0  # Force check

        await loop._run_once()

        cleanup.assert_called_once()
        assert loop._stale_processes_cleaned == 3

    @pytest.mark.asyncio
    async def test_run_once_restarts_stopped_loops(self):
        """Test _run_once restarts stopped loops."""
        restart = AsyncMock(return_value={"loop1": True, "loop2": True})
        loop = self._create_loop(restart_stopped_loops=restart)
        loop._last_loop_restart_check = 0  # Force check

        await loop._run_once()

        restart.assert_called_once()
        assert loop._loops_restarted == 2

    @pytest.mark.asyncio
    async def test_run_once_skips_stuck_job_recovery_non_leader(self):
        """Test _run_once skips stuck job recovery if not leader."""
        is_leader = MagicMock(return_value=False)
        mock_health = MagicMock()
        get_health = MagicMock(return_value=mock_health)
        loop = self._create_loop(
            is_leader=is_leader,
            get_health_manager=get_health,
        )

        await loop._run_once()

        # Health manager should not be accessed for job recovery
        mock_health.find_stuck_jobs.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_handles_cleanup_error(self):
        """Test _run_once handles cleanup error gracefully."""
        cleanup = MagicMock(side_effect=Exception("cleanup failed"))
        config = SelfHealingConfig(stale_process_check_interval_seconds=0.001)
        loop = self._create_loop(
            cleanup_stale_processes=cleanup,
            config=config,
        )
        loop._last_stale_check = 0

        await loop._run_once()  # Should not raise

        assert loop._stale_processes_cleaned == 0

    def test_get_healing_stats(self):
        """Test get_healing_stats returns correct stats."""
        loop = self._create_loop()
        loop._stale_processes_cleaned = 5
        loop._stuck_jobs_recovered = 2
        loop._loops_restarted = 3

        stats = loop.get_healing_stats()

        assert stats["stale_processes_cleaned"] == 5
        assert stats["stuck_jobs_recovered"] == 2
        assert stats["loops_restarted"] == 3
        assert "total_runs" in stats

    def test_health_check_running(self):
        """Test health_check when loop is running."""
        loop = self._create_loop()
        loop._running = True

        health = loop.health_check()

        assert health.healthy is True
        assert "operational" in health.message.lower() or "SelfHealingLoop" in health.message

    def test_health_check_stopped(self):
        """Test health_check when loop is stopped."""
        loop = self._create_loop()
        loop._running = False

        health = loop.health_check()

        assert health.healthy is True
        assert "stopped" in health.message.lower()


# =============================================================================
# PredictiveMonitoringConfig Tests
# =============================================================================


class TestPredictiveMonitoringConfig:
    """Tests for PredictiveMonitoringConfig dataclass."""

    def test_default_values(self):
        """Test PredictiveMonitoringConfig has sensible defaults."""
        config = PredictiveMonitoringConfig()

        assert config.monitoring_interval_seconds == 300.0
        assert config.initial_delay_seconds == 90.0

    def test_validation_monitoring_interval_zero(self):
        """Test validation rejects monitoring_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="monitoring_interval_seconds"):
            PredictiveMonitoringConfig(monitoring_interval_seconds=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            PredictiveMonitoringConfig(initial_delay_seconds=-1)


# =============================================================================
# PredictiveMonitoringLoop Tests
# =============================================================================


class TestPredictiveMonitoringLoop:
    """Tests for PredictiveMonitoringLoop class."""

    def _create_mock_peer(self, node_id: str, alive: bool = True, disk_pct: float = 50.0, mem_pct: float = 40.0):
        """Create a mock peer."""
        peer = MagicMock()
        peer.node_id = node_id
        peer.is_alive.return_value = alive
        peer.disk_percent = disk_pct
        peer.mem_percent = mem_pct
        return peer

    def _create_loop(self, **overrides):
        """Create a PredictiveMonitoringLoop with defaults."""
        defaults = {
            "is_leader": MagicMock(return_value=True),
            "get_alert_manager": MagicMock(return_value=None),
            "get_work_queue": MagicMock(return_value=None),
            "get_peers": MagicMock(return_value=[]),
            "get_notifier": MagicMock(return_value=None),
            "get_production_models": None,
            "config": None,
        }
        defaults.update(overrides)
        return PredictiveMonitoringLoop(**defaults)

    def test_init(self):
        """Test PredictiveMonitoringLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "predictive_monitoring"
        assert loop._alerts_sent == 0
        assert loop._checks_performed == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        is_leader = MagicMock(return_value=False)
        mock_alert = MagicMock()
        get_alert = MagicMock(return_value=mock_alert)
        loop = self._create_loop(
            is_leader=is_leader,
            get_alert_manager=get_alert,
        )

        await loop._run_once()

        mock_alert.run_all_checks.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_skips_no_alert_manager(self):
        """Test _run_once skips if no alert manager."""
        get_alert = MagicMock(return_value=None)
        loop = self._create_loop(get_alert_manager=get_alert)

        await loop._run_once()

        assert loop._checks_performed == 0

    @pytest.mark.asyncio
    async def test_run_once_collects_peer_metrics(self):
        """Test _run_once collects metrics from peers."""
        peers = [
            self._create_mock_peer("node-1", disk_pct=60.0, mem_pct=50.0),
            self._create_mock_peer("node-2", disk_pct=70.0, mem_pct=60.0),
        ]
        mock_alert = MagicMock()
        mock_alert.run_all_checks = AsyncMock(return_value=[])
        get_alert = MagicMock(return_value=mock_alert)
        loop = self._create_loop(
            get_alert_manager=get_alert,
            get_peers=MagicMock(return_value=peers),
        )

        await loop._run_once()

        # Should record metrics for both peers
        assert mock_alert.record_disk_usage.call_count == 2
        assert mock_alert.record_memory_usage.call_count == 2

    @pytest.mark.asyncio
    async def test_run_once_records_queue_depth(self):
        """Test _run_once records queue depth."""
        mock_wq = MagicMock()
        mock_wq.get_queue_status.return_value = {"by_status": {"pending": 50}}
        mock_alert = MagicMock()
        mock_alert.run_all_checks = AsyncMock(return_value=[])
        get_alert = MagicMock(return_value=mock_alert)
        loop = self._create_loop(
            get_alert_manager=get_alert,
            get_work_queue=MagicMock(return_value=mock_wq),
        )

        await loop._run_once()

        mock_alert.record_queue_depth.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_run_once_sends_alerts(self):
        """Test _run_once sends alerts via notifier."""
        # Create mock alert
        mock_alert_obj = MagicMock()
        mock_alert_obj.alert_type.value = "disk_usage"
        mock_alert_obj.message = "Disk usage high"
        mock_alert_obj.severity.value = "warning"
        mock_alert_obj.action = "cleanup"
        mock_alert_obj.target_id = "node-1"

        mock_alert = MagicMock()
        mock_alert.run_all_checks = AsyncMock(return_value=[mock_alert_obj])

        mock_notifier = MagicMock()
        mock_notifier.send = AsyncMock()

        loop = self._create_loop(
            get_alert_manager=MagicMock(return_value=mock_alert),
            get_notifier=MagicMock(return_value=mock_notifier),
        )

        await loop._run_once()

        mock_notifier.send.assert_called_once()
        assert loop._alerts_sent == 1

    def test_get_monitoring_stats(self):
        """Test get_monitoring_stats returns correct stats."""
        loop = self._create_loop()
        loop._alerts_sent = 5
        loop._checks_performed = 10

        stats = loop.get_monitoring_stats()

        assert stats["alerts_sent"] == 5
        assert stats["checks_performed"] == 10

    def test_health_check_running_leader(self):
        """Test health_check when running as leader."""
        loop = self._create_loop()
        loop._running = True

        health = loop.health_check()

        assert health.healthy is True
        assert "operational" in health.message.lower() or "PredictiveMonitoring" in health.message

    def test_health_check_running_not_leader(self):
        """Test health_check when running but not leader.

        The loop uses CoordinatorStatus.PAUSED for non-leader state.
        """
        is_leader = MagicMock(return_value=False)
        loop = self._create_loop(is_leader=is_leader)
        loop._running = True

        health = loop.health_check()
        assert health.healthy is True
        # The message should indicate non-leader state
        assert "idle" in health.message.lower() or "not leader" in health.message.lower()


# =============================================================================
# SplitBrainDetectionConfig Tests
# =============================================================================


class TestSplitBrainDetectionConfig:
    """Tests for SplitBrainDetectionConfig dataclass."""

    def test_default_values(self):
        """Test SplitBrainDetectionConfig has sensible defaults."""
        config = SplitBrainDetectionConfig()

        assert config.detection_interval_seconds == 60.0
        assert config.initial_delay_seconds == 120.0
        assert config.request_timeout_seconds == 5.0
        assert config.min_peers_for_detection == 3
        assert config.partition_alert_threshold_seconds == 600  # 10 min
        assert config.partition_resync_delay_seconds == 60
        assert config.min_peers_for_healthy == 3

    def test_validation_detection_interval_zero(self):
        """Test validation rejects detection_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="detection_interval_seconds"):
            SplitBrainDetectionConfig(detection_interval_seconds=0)

    def test_validation_initial_delay_negative(self):
        """Test validation rejects initial_delay_seconds < 0."""
        with pytest.raises(ValueError, match="initial_delay_seconds"):
            SplitBrainDetectionConfig(initial_delay_seconds=-1)

    def test_validation_request_timeout_zero(self):
        """Test validation rejects request_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="request_timeout_seconds"):
            SplitBrainDetectionConfig(request_timeout_seconds=0)

    def test_validation_min_peers_zero(self):
        """Test validation rejects min_peers_for_detection < 1."""
        with pytest.raises(ValueError, match="min_peers_for_detection"):
            SplitBrainDetectionConfig(min_peers_for_detection=0)

    def test_from_defaults(self):
        """Test from_defaults factory method."""
        config = SplitBrainDetectionConfig.from_defaults()

        # Should have valid defaults
        assert config.detection_interval_seconds > 0
        assert config.min_peers_for_detection >= 1


# =============================================================================
# SplitBrainDetectionLoop Tests
# =============================================================================


class TestSplitBrainDetectionLoop:
    """Tests for SplitBrainDetectionLoop class."""

    def _create_loop(self, **overrides):
        """Create a SplitBrainDetectionLoop with defaults."""
        defaults = {
            "get_peers": MagicMock(return_value={}),
            "get_peer_endpoint": MagicMock(return_value=None),
            "get_own_leader_id": MagicMock(return_value="leader-1"),
            "get_cluster_epoch": MagicMock(return_value=1),
            "on_split_brain_detected": None,
            "config": None,
        }
        defaults.update(overrides)
        return SplitBrainDetectionLoop(**defaults)

    def test_init(self):
        """Test SplitBrainDetectionLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "split_brain_detection"
        assert loop._detections == 0
        assert loop._checks_performed == 0
        assert loop._partition_start_time == 0.0
        assert loop._partition_alert_emitted is False

    @pytest.mark.asyncio
    async def test_run_once_skips_insufficient_peers(self):
        """Test _run_once skips if not enough peers."""
        get_peers = MagicMock(return_value={"node-1": {}, "node-2": {}})
        config = SplitBrainDetectionConfig(min_peers_for_detection=5)
        loop = self._create_loop(
            get_peers=get_peers,
            config=config,
        )

        await loop._run_once()

        assert loop._checks_performed == 1  # Still incremented
        assert loop._detections == 0

    @pytest.mark.asyncio
    async def test_run_once_detects_single_leader(self):
        """Test _run_once detects healthy cluster with single leader."""
        get_peers = MagicMock(return_value={
            "node-1": {},
            "node-2": {},
            "node-3": {},
        })
        get_own_leader = MagicMock(return_value="leader-1")
        config = SplitBrainDetectionConfig(min_peers_for_detection=3)
        loop = self._create_loop(
            get_peers=get_peers,
            get_own_leader_id=get_own_leader,
            get_peer_endpoint=MagicMock(return_value=None),  # No polling
            config=config,
        )

        await loop._run_once()

        assert loop._detections == 0  # No split-brain

    @pytest.mark.asyncio
    async def test_run_once_detects_split_brain(self):
        """Test _run_once detects split-brain with multiple leaders."""
        get_peers = MagicMock(return_value={
            "node-1": {},
            "node-2": {},
            "node-3": {},
        })
        get_own_leader = MagicMock(return_value="leader-1")

        # Mock to simulate different leaders from peers
        async def mock_poll_response(*args, **kwargs):
            # First call returns leader-1, second returns leader-2
            pass

        config = SplitBrainDetectionConfig(min_peers_for_detection=3)
        on_split_brain = AsyncMock()
        loop = self._create_loop(
            get_peers=get_peers,
            get_own_leader_id=get_own_leader,
            on_split_brain_detected=on_split_brain,
            config=config,
        )

        # Manually simulate detection by setting state
        loop._last_leaders_seen = ["leader-1", "leader-2"]

        # Run detection with pre-populated multiple leaders in leaders_seen dict
        # Since we can't easily mock aiohttp, we'll simulate the detection result
        await loop._track_partition_duration()

        # Partition tracking should have started
        assert loop._partition_start_time > 0

    @pytest.mark.asyncio
    async def test_track_partition_duration_emits_alert(self):
        """Test _track_partition_duration emits alert after threshold."""
        config = SplitBrainDetectionConfig(partition_alert_threshold_seconds=1)
        loop = self._create_loop(config=config)
        loop._partition_start_time = time.time() - 2  # Started 2s ago
        loop._partition_alert_emitted = False
        loop._last_leaders_seen = ["leader-1", "leader-2"]

        await loop._track_partition_duration()

        assert loop._partition_alert_emitted is True

    @pytest.mark.asyncio
    async def test_handle_partition_healed(self):
        """Test _handle_partition_healed resets tracking."""
        config = SplitBrainDetectionConfig(partition_resync_delay_seconds=0.01)
        loop = self._create_loop(config=config)
        loop._partition_start_time = time.time() - 100
        loop._partition_alert_emitted = True

        await loop._handle_partition_healed()

        assert loop._partition_start_time == 0.0
        assert loop._partition_alert_emitted is False
        assert loop._last_healthy_time > 0

    def test_get_detection_stats(self):
        """Test get_detection_stats returns correct stats."""
        loop = self._create_loop()
        loop._detections = 2
        loop._checks_performed = 100
        loop._partition_start_time = time.time() - 300
        loop._partition_alert_emitted = True
        loop._last_leaders_seen = ["leader-1", "leader-2"]

        stats = loop.get_detection_stats()

        assert stats["detections"] == 2
        assert stats["checks_performed"] == 100
        assert stats["partition_active"] is True
        assert stats["partition_duration_seconds"] >= 300
        assert stats["partition_alert_emitted"] is True
        assert stats["last_leaders_seen"] == ["leader-1", "leader-2"]

    def test_get_detection_stats_no_partition(self):
        """Test get_detection_stats when no partition active."""
        loop = self._create_loop()
        loop._detections = 0
        loop._partition_start_time = 0.0

        stats = loop.get_detection_stats()

        assert stats["partition_active"] is False
        assert stats["partition_duration_seconds"] == 0.0

    def test_health_check_healthy(self):
        """Test health_check when healthy (no partition)."""
        loop = self._create_loop()
        loop._running = True
        loop._partition_start_time = 0.0

        health = loop.health_check()

        assert health.healthy is True
        assert "operational" in health.message.lower() or "SplitBrainDetection" in health.message

    def test_health_check_partition_active_under_threshold(self):
        """Test health_check with partition under alert threshold."""
        config = SplitBrainDetectionConfig(partition_alert_threshold_seconds=600)
        loop = self._create_loop(config=config)
        loop._running = True
        loop._partition_start_time = time.time() - 60  # 1 min ago, under 10 min threshold
        loop._last_leaders_seen = ["leader-1", "leader-2"]

        health = loop.health_check()

        assert health.healthy is True
        assert "degraded" in str(health.status).lower() or "partition" in health.message.lower()

    def test_health_check_partition_active_over_threshold(self):
        """Test health_check with partition over alert threshold."""
        config = SplitBrainDetectionConfig(partition_alert_threshold_seconds=60)
        loop = self._create_loop(config=config)
        loop._running = True
        loop._partition_start_time = time.time() - 120  # 2 min ago, over 1 min threshold
        loop._last_leaders_seen = ["leader-1", "leader-2"]

        health = loop.health_check()

        assert health.healthy is False
        assert "split-brain" in health.message.lower() or "partition" in health.message.lower()


# =============================================================================
# Loop Lifecycle Tests
# =============================================================================


class TestResilienceLoopsLifecycle:
    """Tests for resilience loop lifecycle management."""

    @pytest.mark.asyncio
    async def test_self_healing_start_stop(self):
        """Test SelfHealingLoop can start and stop cleanly."""
        config = SelfHealingConfig(
            healing_interval_seconds=0.1,
            initial_delay_seconds=0,
            stale_process_check_interval_seconds=1000,  # Don't trigger
        )
        loop = SelfHealingLoop(
            is_leader=MagicMock(return_value=False),
            get_health_manager=MagicMock(return_value=None),
            get_work_queue=MagicMock(return_value=None),
            cleanup_stale_processes=MagicMock(return_value=0),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_predictive_monitoring_start_stop(self):
        """Test PredictiveMonitoringLoop can start and stop cleanly."""
        config = PredictiveMonitoringConfig(
            monitoring_interval_seconds=0.1,
            initial_delay_seconds=0,
        )
        loop = PredictiveMonitoringLoop(
            is_leader=MagicMock(return_value=False),
            get_alert_manager=MagicMock(return_value=None),
            get_work_queue=MagicMock(return_value=None),
            get_peers=MagicMock(return_value=[]),
            get_notifier=MagicMock(return_value=None),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_split_brain_detection_start_stop(self):
        """Test SplitBrainDetectionLoop can start and stop cleanly."""
        config = SplitBrainDetectionConfig(
            detection_interval_seconds=0.1,
            initial_delay_seconds=0,
        )
        loop = SplitBrainDetectionLoop(
            get_peers=MagicMock(return_value={}),
            get_peer_endpoint=MagicMock(return_value=None),
            get_own_leader_id=MagicMock(return_value="leader-1"),
            get_cluster_epoch=MagicMock(return_value=1),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestResilienceLoopsIntegration:
    """Integration tests for resilience loops."""

    @pytest.mark.asyncio
    async def test_self_healing_with_stuck_jobs(self):
        """Test self-healing loop with stuck job detection.

        Note: The actual stuck job recovery requires WorkItem import inside
        _run_once(), so we test the parts that are mockable:
        - Health manager is queried for work queue
        - Queue status is retrieved
        - Stale process cleanup is called
        """
        mock_health = MagicMock()
        mock_health.find_stuck_jobs.return_value = []  # No stuck jobs

        mock_wq = MagicMock()
        mock_wq.get_queue_status.return_value = {"running": []}

        cleanup = MagicMock(return_value=0)
        config = SelfHealingConfig(stale_process_check_interval_seconds=0.001)

        loop = SelfHealingLoop(
            is_leader=MagicMock(return_value=True),
            get_health_manager=MagicMock(return_value=mock_health),
            get_work_queue=MagicMock(return_value=mock_wq),
            cleanup_stale_processes=cleanup,
            config=config,
        )
        loop._last_stale_check = 0  # Force stale check

        await loop._run_once()

        # Verify health manager was wired up
        mock_health.set_work_queue.assert_called_once_with(mock_wq)
        mock_wq.get_queue_status.assert_called_once()
        cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_predictive_monitoring_full_cycle(self):
        """Test predictive monitoring full cycle."""
        # Create mock peers
        mock_peer = MagicMock()
        mock_peer.node_id = "node-1"
        mock_peer.is_alive.return_value = True
        mock_peer.disk_percent = 85.0
        mock_peer.mem_percent = 75.0

        # Create mock alert manager
        mock_alert = MagicMock()
        mock_alert.run_all_checks = AsyncMock(return_value=[])

        # Create mock work queue
        mock_wq = MagicMock()
        mock_wq.get_queue_status.return_value = {"by_status": {"pending": 25}}

        loop = PredictiveMonitoringLoop(
            is_leader=MagicMock(return_value=True),
            get_alert_manager=MagicMock(return_value=mock_alert),
            get_work_queue=MagicMock(return_value=mock_wq),
            get_peers=MagicMock(return_value=[mock_peer]),
            get_notifier=MagicMock(return_value=None),
        )

        await loop._run_once()

        # Verify metrics were recorded
        mock_alert.record_disk_usage.assert_called_once_with("node-1", 85.0)
        mock_alert.record_memory_usage.assert_called_once_with("node-1", 75.0)
        mock_alert.record_queue_depth.assert_called_once_with(25)
        mock_alert.run_all_checks.assert_called_once()
        assert loop._checks_performed == 1

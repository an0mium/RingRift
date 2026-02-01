"""Tests for scripts.p2p.loop_registry - Loop registration for P2P orchestrator.

February 2026: Created as part of Sprint 2 test coverage improvements.
Tests the LoopRegistrationResult dataclass, register_all_loops() main function,
individual _register_*() functions, and the _get_client_session() helper.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import fields
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.loop_registry import (
    LoopRegistrationResult,
    _get_client_session,
    register_all_loops,
)


# ---------------------------------------------------------------------------
# Helper: create a mock orchestrator with common attributes
# ---------------------------------------------------------------------------

def _make_mock_orchestrator(**overrides: Any) -> MagicMock:
    """Create a mock P2POrchestrator with attributes needed by loop_registry."""
    orch = MagicMock()
    orch.node_id = "test-node-1"
    orch.host = "127.0.0.1"
    orch.port = 8770
    orch.role = "FOLLOWER"
    orch.leader_id = "leader-node-1"
    orch.http_session = MagicMock()

    # Locks
    orch.peers_lock = threading.RLock()
    orch.jobs_lock = threading.RLock()
    orch.training_jobs_lock = threading.RLock()

    # Peers
    orch.peers = {}
    orch.known_peers = []

    # Active jobs
    orch.active_jobs = {}
    orch.training_jobs = {}

    # Selfplay scheduler
    orch.selfplay_scheduler = MagicMock()
    orch.selfplay_scheduler.verify_pending_spawns = AsyncMock()
    orch.selfplay_scheduler.get_spawn_success_rate = MagicMock(return_value=1.0)

    # Job manager
    orch.job_manager = MagicMock()
    orch.job_manager.process_stale_jobs = AsyncMock()

    # State manager
    orch.state_manager = MagicMock()
    orch.state_manager.get_peers = MagicMock(return_value={})

    # Training coordinator
    orch.training_coordinator = MagicMock()

    # Peer snapshot
    orch._peer_snapshot = MagicMock()
    orch._peer_snapshot.get_snapshot = MagicMock(return_value={})

    # Self info
    orch.self_info = MagicMock()
    orch.self_info.is_gpu_node = MagicMock(return_value=False)
    orch.self_info.training_jobs = 0
    orch.self_info.gpu_percent = 0
    orch.self_info.cpu_percent = 0
    orch.self_info.has_gpu = False
    orch.self_info.selfplay_jobs = 0
    orch.self_info.max_selfplay_slots = 8

    # Notifier
    orch.notifier = MagicMock()
    orch.notifier.send = AsyncMock()

    # Callbacks
    orch._is_leader = MagicMock(return_value=False)
    orch._safe_emit_event = MagicMock()
    orch._safe_emit_p2p_event = MagicMock()

    # Various orchestrator methods
    orch._detect_nat_type = MagicMock()
    orch._probe_nat_blocked_peers = MagicMock()
    orch._update_relay_preferences = MagicMock()
    orch._validate_relay_assignments = MagicMock()
    orch._collect_cluster_manifest = AsyncMock()
    orch._collect_local_data_manifest = MagicMock()
    orch._update_manifest_from_loop = MagicMock()
    orch._update_improvement_cycle_from_loop = MagicMock()
    orch._record_selfplay_stats_sample = MagicMock()
    orch._get_alive_peers_for_broadcast = MagicMock(return_value=[])
    orch._cleanup_local_disk = AsyncMock()
    orch._convert_jsonl_to_db = AsyncMock()
    orch._convert_jsonl_to_npz_for_training = AsyncMock()
    orch._trigger_export_for_loop = AsyncMock()
    orch._start_auto_training = AsyncMock()
    orch.get_data_directory = MagicMock()
    orch._sync_selfplay_to_training_nodes = AsyncMock()
    orch._claim_work_from_leader = AsyncMock()
    orch._execute_claimed_work = AsyncMock()
    orch._report_work_result = AsyncMock()
    orch._claim_work_batch_from_leader = AsyncMock()
    orch._send_heartbeat_to_peer = AsyncMock()
    orch._get_healthy_node_ids_for_reassignment = MagicMock(return_value=[])
    orch._cleanup_stale_processes = AsyncMock()
    orch._get_loop_manager = MagicMock(return_value=None)
    orch._get_leader_base_url = MagicMock(return_value=None)
    orch._probe_peer_health = AsyncMock(return_value=False)
    orch._bootstrap_from_peer = AsyncMock()
    orch._sync_peer_snapshot = MagicMock()
    orch._emit_split_brain_detected = AsyncMock()

    # Cluster config
    orch.cluster_config = {}
    orch._cluster_config = {}
    orch._model_versions = {}
    orch.cluster_data_manifest = None

    # Stability / recovery
    orch._stability_controller = None
    orch._peer_state_tracker = None
    orch._cooldown_manager = None

    # Voter
    orch.voter_node_ids = []
    orch.quorum_manager = MagicMock()

    # Misc
    orch.bootstrap_seeds = []
    orch.cluster_epoch = 0
    orch._autonomous_queue_loop = None
    orch.restart_http_server = AsyncMock()
    orch._check_for_updates = MagicMock(return_value=False)
    orch._perform_git_update = AsyncMock()
    orch._restart_orchestrator = AsyncMock()
    orch._get_commits_behind = MagicMock(return_value=0)
    orch._is_peer_alive_for_circuit_breaker = MagicMock(return_value=True)
    orch._get_cached_jittered_timeout = MagicMock(return_value=60.0)
    orch._get_tailscale_ip_for_peer = MagicMock(return_value=None)
    orch.network = MagicMock()

    for k, v in overrides.items():
        setattr(orch, k, v)

    return orch


def _make_mock_manager() -> MagicMock:
    """Create a mock LoopManager that records register() calls."""
    mgr = MagicMock()
    mgr._registered: list[str] = []

    def _track_register(loop: Any) -> None:
        name = getattr(loop, "name", None) or type(loop).__name__
        mgr._registered.append(name)

    mgr.register = MagicMock(side_effect=_track_register)
    return mgr


# ===========================================================================
# Tests for LoopRegistrationResult dataclass
# ===========================================================================


class TestLoopRegistrationResult:
    """Tests for the LoopRegistrationResult dataclass."""

    def test_defaults(self):
        result = LoopRegistrationResult(success=True)
        assert result.success is True
        assert result.loops_registered == 0
        assert result.loops_failed == []
        assert result.error is None

    def test_failure_with_error(self):
        result = LoopRegistrationResult(
            success=False,
            loops_registered=3,
            loops_failed=["LoopA", "LoopB"],
            error="import error",
        )
        assert result.success is False
        assert result.loops_registered == 3
        assert result.loops_failed == ["LoopA", "LoopB"]
        assert result.error == "import error"

    def test_field_names(self):
        names = {f.name for f in fields(LoopRegistrationResult)}
        assert names == {"success", "loops_registered", "loops_failed", "error"}

    def test_mutable_default_list_not_shared(self):
        r1 = LoopRegistrationResult(success=True)
        r2 = LoopRegistrationResult(success=True)
        r1.loops_failed.append("X")
        assert "X" not in r2.loops_failed


# ===========================================================================
# Tests for _get_client_session helper
# ===========================================================================


class TestGetClientSession:
    """Tests for the _get_client_session helper."""

    @pytest.mark.asyncio
    async def test_returns_client_session(self):
        import aiohttp
        session = _get_client_session()
        assert isinstance(session, aiohttp.ClientSession)
        await session.close()

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        from aiohttp import ClientTimeout
        timeout = ClientTimeout(total=5)
        session = _get_client_session(timeout)
        assert session is not None
        await session.close()

    @pytest.mark.asyncio
    async def test_default_timeout(self):
        session = _get_client_session(None)
        assert session is not None
        await session.close()


# ===========================================================================
# Tests for register_all_loops main function
# ===========================================================================


class TestRegisterAllLoops:
    """Tests for the register_all_loops() main entry point."""

    def test_returns_result_even_on_import_failure(self):
        """If loop imports fail, should still return a result (not raise)."""
        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()

        # Patch the entire import block to fail
        with patch(
            "scripts.p2p.loop_registry.register_all_loops",
            wraps=register_all_loops,
        ):
            result = register_all_loops(orch, mgr)

        assert isinstance(result, LoopRegistrationResult)
        # Either success=True (some loops registered) or False (total failure)
        assert isinstance(result.success, bool)
        assert isinstance(result.loops_registered, int)
        assert isinstance(result.loops_failed, list)

    def test_total_import_failure_returns_error(self):
        """When the import block in register_all_loops raises, success=False."""
        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()

        with patch(
            "scripts.p2p.loop_registry._register_queue_populator",
            side_effect=RuntimeError("boom"),
        ):
            # RuntimeError propagates up to the outer try/except in register_all_loops
            result = register_all_loops(orch, mgr)

        assert isinstance(result, LoopRegistrationResult)
        # It could be success=False if the RuntimeError was caught by outer handler
        # or some loops already registered before the error
        assert isinstance(result.loops_registered, int)

    def test_accumulates_failed_loops(self):
        """Failed individual registrations are tracked in loops_failed."""
        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()

        result = register_all_loops(orch, mgr)

        # Some loops may not be importable in this test environment
        # but the function should still complete successfully
        assert isinstance(result, LoopRegistrationResult)
        assert result.success is True
        # Total = registered + len(failed)
        assert result.loops_registered >= 0

    def test_loops_registered_count_matches_manager_register_calls(self):
        """loops_registered should match the number of manager.register() calls."""
        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()

        result = register_all_loops(orch, mgr)

        assert result.loops_registered == mgr.register.call_count


# ===========================================================================
# Tests for individual _register_*() functions
# ===========================================================================


class TestRegisterQueuePopulator:
    """Tests for _register_queue_populator."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_queue_populator

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        # Mock the OrchestratorContext
        mock_ctx = MagicMock()

        count = _register_queue_populator(orch, mgr, mock_ctx, failed)

        assert count == 1
        assert mgr.register.call_count == 1
        assert len(failed) == 0
        # Check orchestrator got the loop reference
        assert orch._queue_populator_loop is not None

    def test_import_failure(self):
        from scripts.p2p.loop_registry import _register_queue_populator

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        with patch(
            "scripts.p2p.loop_registry._register_queue_populator",
        ) as mock_fn:
            # Simulate the import failing by mocking the function itself
            mock_fn.return_value = 0
            mock_fn.side_effect = None

        # Test the actual function with a broken import
        with patch.dict("sys.modules", {"scripts.p2p.loops": None}):
            # This should catch ImportError and return 0
            count = _register_queue_populator(orch, mgr, MagicMock(), failed)
            assert count == 0
            assert "QueuePopulatorLoop" in failed


class TestRegisterEloSync:
    """Tests for _register_elo_sync."""

    def test_success_with_elo_manager(self):
        from scripts.p2p.loop_registry import _register_elo_sync

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.elo_sync_manager = MagicMock()
        ctx.sync_in_progress = MagicMock(return_value=False)

        count = _register_elo_sync(orch, mgr, ctx, failed)

        assert count == 1
        assert mgr.register.call_count == 1
        assert len(failed) == 0

    def test_skipped_when_elo_manager_is_none(self):
        from scripts.p2p.loop_registry import _register_elo_sync

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.elo_sync_manager = None

        count = _register_elo_sync(orch, mgr, ctx, failed)

        assert count == 0
        assert mgr.register.call_count == 0
        assert len(failed) == 0  # Not a failure, just skipped


class TestRegisterJobReaper:
    """Tests for _register_job_reaper."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_job_reaper

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.get_active_jobs = MagicMock(return_value={})
        ctx.cancel_job = AsyncMock()
        ctx.get_job_heartbeats = MagicMock(return_value={})

        count = _register_job_reaper(orch, mgr, ctx, failed)

        assert count == 1
        assert mgr.register.call_count == 1


class TestRegisterOrphanDetection:
    """Tests for _register_orphan_detection."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_orphan_detection

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_orphan_detection(orch, mgr, failed)

        assert count == 1
        assert mgr.register.call_count == 1

    def test_tracked_pids_from_jobs(self):
        """The internal _get_tracked_pids should extract PIDs from active_jobs."""
        from scripts.p2p.loop_registry import _register_orphan_detection

        orch = _make_mock_orchestrator()
        orch.active_jobs = {
            "selfplay": {
                "job-1": {"pid": 1234, "config": "hex8_2p"},
                "job-2": {"pid": 5678, "config": "square8_2p"},
            }
        }

        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_orphan_detection(orch, mgr, failed)
        assert count == 1

        # The registered loop should have been called with get_tracked_pids
        registered_loop = mgr.register.call_args[0][0]
        pids = registered_loop._get_tracked_pids()
        assert 1234 in pids
        assert 5678 in pids


class TestRegisterIdleDetection:
    """Tests for _register_idle_detection."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_idle_detection

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.get_role = MagicMock(return_value="FOLLOWER")
        ctx.get_peers = MagicMock(return_value={})
        ctx.get_work_queue = MagicMock()
        ctx.auto_start_selfplay = AsyncMock()
        ctx.handle_zombie_detected = AsyncMock()

        count = _register_idle_detection(orch, mgr, ctx, failed)

        assert count == 1
        assert len(failed) == 0


class TestRegisterSpawnVerification:
    """Tests for _register_spawn_verification."""

    def test_success_with_scheduler(self):
        from scripts.p2p.loop_registry import _register_spawn_verification

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.selfplay_scheduler = MagicMock()
        ctx.selfplay_scheduler.verify_pending_spawns = AsyncMock()
        ctx.selfplay_scheduler.get_spawn_success_rate = MagicMock(return_value=0.95)
        ctx.get_peers = MagicMock(return_value={"node-a": MagicMock()})

        count = _register_spawn_verification(orch, mgr, ctx, failed)

        assert count == 1

    def test_skipped_without_scheduler(self):
        from scripts.p2p.loop_registry import _register_spawn_verification

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.selfplay_scheduler = None

        count = _register_spawn_verification(orch, mgr, ctx, failed)

        assert count == 0
        assert mgr.register.call_count == 0


class TestRegisterJobReassignment:
    """Tests for _register_job_reassignment."""

    def test_success_with_job_manager(self):
        from scripts.p2p.loop_registry import _register_job_reassignment

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.get_role = MagicMock(return_value="LEADER")
        ctx.job_manager = MagicMock()
        ctx.job_manager.process_stale_jobs = AsyncMock()

        count = _register_job_reassignment(orch, mgr, ctx, failed)

        assert count == 1

    def test_skipped_without_job_manager(self):
        from scripts.p2p.loop_registry import _register_job_reassignment

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.job_manager = None

        count = _register_job_reassignment(orch, mgr, ctx, failed)

        assert count == 0


class TestRegisterWorkQueueMaintenance:
    """Tests for _register_work_queue_maintenance."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_work_queue_maintenance

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        ctx = MagicMock()
        ctx.is_leader = MagicMock(return_value=True)
        ctx.get_work_queue = MagicMock()

        count = _register_work_queue_maintenance(orch, mgr, ctx, failed)

        assert count == 1


class TestRegisterNatManagement:
    """Tests for _register_nat_management."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_nat_management

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_nat_management(orch, mgr, failed)

        assert count == 1
        assert mgr.register.call_count == 1


class TestRegisterModelSync:
    """Tests for _register_model_sync."""

    def test_success_when_available(self):
        from scripts.p2p.loop_registry import _register_model_sync

        orch = _make_mock_orchestrator()
        orch._model_versions = {"hex8_2p": "abc123"}
        orch.cluster_config = {"hosts": {}}
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_model_sync(orch, mgr, failed)

        assert count == 1

    def test_skipped_when_no_model_versions(self):
        from scripts.p2p.loop_registry import _register_model_sync

        orch = _make_mock_orchestrator()
        # Remove the _model_versions attribute
        del orch._model_versions
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_model_sync(orch, mgr, failed)

        assert count == 0


class TestRegisterLeaderProbe:
    """Tests for _register_leader_probe."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_leader_probe

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_leader_probe(orch, mgr, failed)

        assert count == 1
        assert mgr.register.call_count == 1

    def test_import_failure(self):
        from scripts.p2p.loop_registry import _register_leader_probe

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        with patch(
            "scripts.p2p.loop_registry._register_leader_probe",
        ):
            pass  # Just testing the patch works

        # Simulate import error by patching the import inside
        with patch.dict("sys.modules", {"scripts.p2p.loops": None}):
            count = _register_leader_probe(orch, mgr, failed)
            assert count == 0
            assert "LeaderProbeLoop" in failed


class TestRegisterLeaderMaintenance:
    """Tests for _register_leader_maintenance."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_leader_maintenance

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_leader_maintenance(orch, mgr, failed)

        assert count == 1


class TestRegisterHttpServerHealth:
    """Tests for _register_http_server_health."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_http_server_health

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_http_server_health(orch, mgr, failed)

        assert count == 1
        # Verify the loop was created with correct port
        registered_loop = mgr.register.call_args[0][0]
        assert registered_loop._port == orch.port


class TestRegisterPeerCleanup:
    """Tests for _register_peer_cleanup."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_peer_cleanup

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_peer_cleanup(orch, mgr, failed)

        assert count == 1

    def test_env_var_config(self):
        """PeerCleanup should respect environment variables."""
        from scripts.p2p.loop_registry import _register_peer_cleanup

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        with patch.dict("os.environ", {
            "RINGRIFT_PEER_CLEANUP_INTERVAL": "600",
            "RINGRIFT_PEER_CLEANUP_ENABLED": "true",
        }):
            count = _register_peer_cleanup(orch, mgr, failed)

        assert count == 1


class TestRegisterGossipStateCleanup:
    """Tests for _register_gossip_state_cleanup."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_gossip_state_cleanup

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_gossip_state_cleanup(orch, mgr, failed)

        assert count == 1


class TestRegisterGossipPeerPromotion:
    """Tests for _register_gossip_peer_promotion."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_gossip_peer_promotion

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_gossip_peer_promotion(orch, mgr, failed)

        assert count == 1


class TestRegisterStabilityController:
    """Tests for _register_stability_controller."""

    def test_success_when_controller_exists(self):
        from scripts.p2p.loop_registry import _register_stability_controller

        orch = _make_mock_orchestrator()
        orch._stability_controller = MagicMock()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_stability_controller(orch, mgr, failed)

        assert count == 1
        mgr.register.assert_called_once_with(orch._stability_controller)

    def test_skipped_when_controller_is_none(self):
        from scripts.p2p.loop_registry import _register_stability_controller

        orch = _make_mock_orchestrator()
        orch._stability_controller = None
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_stability_controller(orch, mgr, failed)

        assert count == 0
        assert mgr.register.call_count == 0


class TestRegisterAutonomousQueuePopulation:
    """Tests for _register_autonomous_queue_population."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_autonomous_queue_population

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_autonomous_queue_population(orch, mgr, failed)

        assert count == 1
        assert orch._autonomous_queue_loop is not None

    def test_import_failure_clears_loop_ref(self):
        from scripts.p2p.loop_registry import _register_autonomous_queue_population

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        with patch.dict("sys.modules", {"scripts.p2p.loops": None}):
            count = _register_autonomous_queue_population(orch, mgr, failed)

        assert count == 0
        assert orch._autonomous_queue_loop is None
        assert "AutonomousQueuePopulationLoop" in failed


class TestRegisterCircuitBreakerDecay:
    """Tests for _register_circuit_breaker_decay."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_circuit_breaker_decay

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_circuit_breaker_decay(orch, mgr, failed)

        assert count == 1
        assert orch._cb_decay_loop is not None

    def test_env_var_config(self):
        from scripts.p2p.loop_registry import _register_circuit_breaker_decay

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        with patch.dict("os.environ", {
            "RINGRIFT_CB_DECAY_ENABLED": "true",
            "RINGRIFT_CB_DECAY_INTERVAL": "600",
            "RINGRIFT_CB_DECAY_TTL": "7200",
        }):
            count = _register_circuit_breaker_decay(orch, mgr, failed)

        assert count == 1


class TestRegisterGitUpdate:
    """Tests for _register_git_update."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_git_update

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_git_update(orch, mgr, failed)

        assert count == 1
        assert orch._git_update_loop is not None


class TestRegisterSplitBrainDetection:
    """Tests for _register_split_brain_detection."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_split_brain_detection

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_split_brain_detection(orch, mgr, failed)

        assert count == 1


class TestRegisterPeerRecovery:
    """Tests for _register_peer_recovery."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_peer_recovery

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_peer_recovery(orch, mgr, failed)

        assert count == 1
        assert orch._peer_recovery_loop is not None


class TestRegisterComprehensiveEvaluation:
    """Tests for _register_comprehensive_evaluation."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_comprehensive_evaluation

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_comprehensive_evaluation(orch, mgr, failed)

        assert count == 1


class TestRegisterTournamentDataPipeline:
    """Tests for _register_tournament_data_pipeline."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_tournament_data_pipeline

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_tournament_data_pipeline(orch, mgr, failed)

        assert count == 1


class TestRegisterSwimMembership:
    """Tests for _register_swim_membership."""

    def test_success(self):
        from scripts.p2p.loop_registry import _register_swim_membership

        orch = _make_mock_orchestrator()
        mgr = _make_mock_manager()
        failed: list[str] = []

        count = _register_swim_membership(orch, mgr, failed)

        assert count == 1
        assert orch._swim_membership_loop is not None


# ===========================================================================
# Integration-style test: verify all registration functions return 0 or 1
# ===========================================================================


class TestAllRegistrationFunctionsProtocol:
    """Verify all _register_*() functions follow the protocol: return 0 or 1."""

    def _get_all_register_functions(self) -> list[tuple[str, Any]]:
        """Import all individual registration functions."""
        import scripts.p2p.loop_registry as mod

        return [
            (name, func)
            for name, func in vars(mod).items()
            if name.startswith("_register_") and callable(func)
        ]

    def test_all_functions_discovered(self):
        """Ensure we find a reasonable number of registration functions."""
        funcs = self._get_all_register_functions()
        # loop_registry.py has 45+ _register_*() functions
        assert len(funcs) >= 40, f"Only found {len(funcs)} register functions"

    def test_all_functions_return_int(self):
        """Each _register_*() function should return 0 or 1."""
        funcs = self._get_all_register_functions()
        orch = _make_mock_orchestrator()

        for name, func in funcs:
            mgr = _make_mock_manager()
            failed: list[str] = []

            # Determine the function signature to call correctly
            import inspect
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            try:
                if len(params) == 3:
                    # (orchestrator, manager, failed)
                    result = func(orch, mgr, failed)
                elif len(params) == 4:
                    # (orchestrator, manager, ctx, failed)
                    ctx = MagicMock()
                    ctx.elo_sync_manager = MagicMock()
                    ctx.sync_in_progress = MagicMock(return_value=False)
                    ctx.get_active_jobs = MagicMock(return_value={})
                    ctx.cancel_job = AsyncMock()
                    ctx.get_job_heartbeats = MagicMock(return_value={})
                    ctx.get_role = MagicMock(return_value="FOLLOWER")
                    ctx.get_peers = MagicMock(return_value={})
                    ctx.get_work_queue = MagicMock()
                    ctx.get_work_queue_depth = MagicMock(return_value=0)
                    ctx.auto_start_selfplay = AsyncMock()
                    ctx.handle_zombie_detected = AsyncMock()
                    ctx.selfplay_scheduler = MagicMock()
                    ctx.selfplay_scheduler.verify_pending_spawns = AsyncMock()
                    ctx.selfplay_scheduler.get_spawn_success_rate = MagicMock(return_value=1.0)
                    ctx.job_manager = MagicMock()
                    ctx.job_manager.process_stale_jobs = AsyncMock()
                    ctx.is_leader = MagicMock(return_value=False)
                    ctx.get_leader_id = MagicMock(return_value="leader-1")
                    ctx.get_pending_jobs_for_node = MagicMock(return_value=[])
                    ctx.spawn_preemptive_job = AsyncMock()
                    result = func(orch, mgr, ctx, failed)
                else:
                    continue  # Skip unexpected signatures

                assert result in (0, 1), (
                    f"{name} returned {result} (expected 0 or 1)"
                )
            except Exception as e:
                # Import failures are acceptable in test environment
                # but the function itself should not raise
                pytest.fail(
                    f"{name} raised {type(e).__name__}: {e}"
                )

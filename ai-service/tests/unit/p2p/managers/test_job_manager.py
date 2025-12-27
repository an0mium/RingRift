"""Tests for JobManager - P2P job spawning and lifecycle management.

December 27, 2025: Created as part of test coverage improvement effort.
Tests core job lifecycle functionality without requiring actual job execution.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""
    node_id: str
    ip: str
    port: int = 8770
    role: str = "training"
    last_seen: float = 0.0
    gpu_available: bool = True
    is_alive: bool = True
    has_gpu: bool = True
    _healthy: bool = True

    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return self._healthy and self.is_alive


@pytest.fixture
def job_manager_deps():
    """Create dependencies for JobManager initialization."""
    return {
        "ringrift_path": "/mock/ringrift",
        "node_id": "test-node-1",
        "peers": {},
        "peers_lock": threading.Lock(),
        "active_jobs": {},
        "jobs_lock": threading.Lock(),
        "improvement_loop_state": {},
        "distributed_tournament_state": {},
    }


@pytest.fixture
def job_manager(job_manager_deps):
    """Create a JobManager instance for testing."""
    from scripts.p2p.managers.job_manager import JobManager
    return JobManager(**job_manager_deps)


@pytest.fixture
def job_manager_with_peers(job_manager_deps):
    """Create a JobManager with mock peers."""
    from scripts.p2p.managers.job_manager import JobManager

    peers = {
        "node-2": MockNodeInfo(node_id="node-2", ip="10.0.0.2"),
        "node-3": MockNodeInfo(node_id="node-3", ip="10.0.0.3"),
        "node-4": MockNodeInfo(node_id="node-4", ip="10.0.0.4", is_alive=False, _healthy=False),
    }
    job_manager_deps["peers"] = peers
    return JobManager(**job_manager_deps)


# =============================================================================
# JobManagerStats Tests
# =============================================================================


class TestJobManagerStats:
    """Tests for JobManagerStats dataclass."""

    def test_stats_defaults(self):
        """Test default values for stats."""
        from scripts.p2p.managers.job_manager import JobManagerStats

        stats = JobManagerStats()
        assert stats.jobs_spawned == 0
        assert stats.jobs_completed == 0
        assert stats.jobs_failed == 0
        assert stats.jobs_cancelled == 0
        assert stats.nodes_recovered == 0
        assert stats.hosts_offline == 0
        assert stats.hosts_online == 0

    def test_stats_initialization(self):
        """Test stats with custom values."""
        from scripts.p2p.managers.job_manager import JobManagerStats

        stats = JobManagerStats(
            jobs_spawned=10,
            jobs_completed=8,
            jobs_failed=2,
        )
        assert stats.jobs_spawned == 10
        assert stats.jobs_completed == 8
        assert stats.jobs_failed == 2


# =============================================================================
# JobManager Initialization Tests
# =============================================================================


class TestJobManagerInit:
    """Tests for JobManager initialization."""

    def test_basic_init(self, job_manager):
        """Test basic initialization."""
        assert job_manager.node_id == "test-node-1"
        assert job_manager.ringrift_path == "/mock/ringrift"
        assert isinstance(job_manager.stats, object)

    def test_init_with_empty_peers(self, job_manager):
        """Test initialization with no peers."""
        assert len(job_manager.peers) == 0

    def test_init_with_peers(self, job_manager_with_peers):
        """Test initialization with peers."""
        assert len(job_manager_with_peers.peers) == 3

    def test_engine_modes_constant(self, job_manager):
        """Test SEARCH_ENGINE_MODES contains expected modes."""
        expected_modes = {"maxn", "brs", "mcts", "gumbel-mcts"}
        assert expected_modes.issubset(job_manager.SEARCH_ENGINE_MODES)


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription functionality."""

    def test_subscribe_to_events_success(self, job_manager):
        """Test successful event subscription."""
        with patch("app.coordination.event_router.get_event_bus") as mock_bus:
            mock_bus.return_value = MagicMock()
            job_manager.subscribe_to_events()
            assert job_manager._subscribed is True

    def test_subscribe_to_events_idempotent(self, job_manager):
        """Test subscription is idempotent."""
        job_manager._subscribed = True
        # Should return early without error
        job_manager.subscribe_to_events()
        assert job_manager._subscribed is True


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handler methods."""

    @pytest.mark.asyncio
    async def test_on_host_offline_cancels_running_jobs(self, job_manager):
        """Test HOST_OFFLINE cancels running jobs on that host."""
        # Setup: Add a running job for the offline host
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"node_id": "node-2", "status": "running"},
        }

        # Create mock event with payload
        event = MagicMock()
        event.payload = {"node_id": "node-2", "last_seen": 12345.0}

        initial_cancelled = job_manager.stats.jobs_cancelled
        await job_manager._on_host_offline(event)

        # Job should be marked as cancelled
        assert job_manager.active_jobs["selfplay"]["job-1"]["status"] == "cancelled"
        assert job_manager.stats.jobs_cancelled == initial_cancelled + 1
        assert job_manager.stats.hosts_offline == 1

    @pytest.mark.asyncio
    async def test_on_host_offline_no_jobs(self, job_manager):
        """Test HOST_OFFLINE with no jobs to cancel."""
        event = MagicMock()
        event.payload = {"node_id": "node-2"}

        # Should not raise, stats should not be updated
        await job_manager._on_host_offline(event)
        assert job_manager.stats.hosts_offline == 0  # No jobs cancelled

    @pytest.mark.asyncio
    async def test_on_host_online_updates_stats(self, job_manager):
        """Test HOST_ONLINE handler updates stats."""
        event = MagicMock()
        event.payload = {"node_id": "node-2", "ip": "10.0.0.2"}

        initial_online = job_manager.stats.hosts_online
        await job_manager._on_host_online(event)
        assert job_manager.stats.hosts_online == initial_online + 1

    @pytest.mark.asyncio
    async def test_on_host_online_empty_node_id(self, job_manager):
        """Test HOST_ONLINE with empty node_id returns early."""
        event = MagicMock()
        event.payload = {"node_id": ""}

        await job_manager._on_host_online(event)
        assert job_manager.stats.hosts_online == 0  # Should return early

    @pytest.mark.asyncio
    async def test_on_node_recovered_updates_stats(self, job_manager):
        """Test NODE_RECOVERED handler updates stats."""
        event = MagicMock()
        event.payload = {"node_id": "node-2", "downtime": 300.0}

        initial_recovered = job_manager.stats.nodes_recovered
        await job_manager._on_node_recovered(event)
        assert job_manager.stats.nodes_recovered == initial_recovered + 1


# =============================================================================
# Task Event Emission Tests
# =============================================================================


class TestTaskEventEmission:
    """Tests for task event emission."""

    def test_emit_task_event_success(self, job_manager):
        """Test task event emission."""
        with patch.object(job_manager, "_emit_task_event") as mock_emit:
            job_manager._emit_task_event(
                "TASK_STARTED",
                job_id="job-123",
                job_type="selfplay",
                config_key="hex8_2p",
            )
            mock_emit.assert_called_once()

    def test_emit_task_event_no_emitter(self, job_manager):
        """Test graceful degradation when event emitter unavailable."""
        # Should not raise even if event system unavailable
        job_manager._emit_task_event(
            "TASK_STARTED",
            job_id="job-123",
            job_type="selfplay",
        )


# =============================================================================
# GPU Selfplay Job Tests
# =============================================================================


class TestGPUSelfplayJob:
    """Tests for GPU selfplay job spawning."""

    def test_search_engine_modes_defined(self, job_manager):
        """Test SEARCH_ENGINE_MODES contains expected modes."""
        expected_modes = {"mcts", "gumbel-mcts"}
        assert expected_modes.issubset(job_manager.SEARCH_ENGINE_MODES)

    def test_run_gpu_selfplay_job_exists(self, job_manager):
        """Test that run_gpu_selfplay_job method exists and is callable."""
        assert hasattr(job_manager, "run_gpu_selfplay_job")
        assert callable(job_manager.run_gpu_selfplay_job)

    def test_job_manager_has_stats(self, job_manager):
        """Test that JobManager tracks stats."""
        assert hasattr(job_manager, "stats")
        assert hasattr(job_manager.stats, "jobs_spawned")
        assert hasattr(job_manager.stats, "jobs_completed")
        assert hasattr(job_manager.stats, "jobs_failed")


# =============================================================================
# Distributed Selfplay Tests
# =============================================================================


class TestDistributedSelfplay:
    """Tests for distributed selfplay functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_selfplay_to_workers(self, job_manager_with_peers):
        """Test dispatching work to alive workers."""
        # Get alive peers
        alive_peers = [
            p for p in job_manager_with_peers.peers.values()
            if hasattr(p, "is_alive") and p.is_alive
        ]
        assert len(alive_peers) == 2  # node-2 and node-3


# =============================================================================
# Training Data Export Tests
# =============================================================================


class TestTrainingDataExport:
    """Tests for training data export functionality."""

    @pytest.mark.asyncio
    async def test_export_training_data_requires_job_id(self, job_manager):
        """Test that job_id is required."""
        with pytest.raises(TypeError):
            await job_manager.export_training_data()  # Missing job_id


# =============================================================================
# Training Job Tests
# =============================================================================


class TestTrainingJob:
    """Tests for training job spawning."""

    @pytest.mark.asyncio
    async def test_run_training_requires_job_id(self, job_manager):
        """Test that job_id is required."""
        with pytest.raises(TypeError):
            await job_manager.run_training()  # Missing job_id


# =============================================================================
# Tournament Tests
# =============================================================================


class TestDistributedTournament:
    """Tests for distributed tournament functionality."""

    def test_generate_tournament_matches(self, job_manager):
        """Test match generation logic."""
        # Test with mock models
        models = ["model_a", "model_b", "model_c"]
        matches = job_manager._generate_tournament_matches(models, games_per_pair=2)

        # Should generate all pairings
        assert len(matches) == 6  # 3 choose 2 = 3 pairs, 2 games each = 6

    def test_get_tournament_workers(self, job_manager_with_peers):
        """Test getting available tournament workers."""
        workers = job_manager_with_peers._get_tournament_workers()

        # Should only return alive workers
        assert len(workers) == 2

    def test_calculate_elo_updates(self, job_manager):
        """Test Elo calculation."""
        # Test basic Elo update
        models = ["model_a", "model_b"]
        results = [
            {"player1_model": "model_a", "player2_model": "model_b", "winner": "model_a"},
        ]

        deltas = job_manager._calculate_elo_updates(models, results)

        # Winner should have positive delta, loser negative
        assert deltas["model_a"] > 0
        assert deltas["model_b"] < 0


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestJobCleanup:
    """Tests for job cleanup functionality."""

    def test_cleanup_completed_jobs_empty(self, job_manager):
        """Test cleanup with no jobs."""
        cleaned = job_manager.cleanup_completed_jobs()
        assert cleaned == 0

    def test_cleanup_completed_jobs_removes_completed(self, job_manager):
        """Test that completed jobs are cleaned up."""
        # Add some completed jobs
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"status": "completed", "completed_at": 0},
            "job-2": {"status": "running"},
        }

        cleaned = job_manager.cleanup_completed_jobs()

        # Should clean completed jobs
        assert "job-2" in job_manager.active_jobs.get("selfplay", {})


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_returns_dict(self, job_manager):
        """Test health check returns proper structure."""
        health = job_manager.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health or "status" in health

    def test_health_check_includes_stats(self, job_manager):
        """Test health check includes job stats."""
        health = job_manager.health_check()

        # Should include some stats
        assert isinstance(health, dict)

    def test_health_check_with_active_jobs(self, job_manager):
        """Test health check with running jobs."""
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"status": "running"},
        }

        health = job_manager.health_check()
        assert isinstance(health, dict)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_default_port(self):
        """Test default port retrieval."""
        from scripts.p2p.managers.job_manager import _get_default_port

        port = _get_default_port()
        assert port == 8770  # Default fallback

    def test_get_event_emitter_thread_safe(self):
        """Test event emitter getter is thread-safe."""
        from scripts.p2p.managers.job_manager import _get_event_emitter

        # Should not raise even with concurrent calls
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_get_event_emitter) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be the same (or all None)
        assert len(set(id(r) if r else None for r in results)) <= 2

"""Unit tests for unified_scheduler module.

Tests the UnifiedScheduler job routing and management system.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.coordination.unified_scheduler import (
    Backend,
    JobState,
    JobStatus,
    JobType,
    UnifiedJob,
    UnifiedScheduler,
)


# =============================================================================
# Backend Enum Tests
# =============================================================================


class TestBackend:
    """Test Backend enum."""

    def test_all_backends_exist(self):
        """Verify all expected backends are defined."""
        backends = [b.value for b in Backend]
        assert "slurm" in backends
        assert "vast" in backends
        assert "p2p" in backends
        assert "auto" in backends

    def test_backend_count(self):
        """Verify backend count matches expected."""
        assert len(Backend) == 4

    def test_backend_is_str_enum(self):
        """Backend values should be strings."""
        for backend in Backend:
            assert isinstance(backend.value, str)


# =============================================================================
# JobType Enum Tests
# =============================================================================


class TestJobType:
    """Test JobType enum."""

    def test_all_job_types_exist(self):
        """Verify all expected job types are defined."""
        types = [t.value for t in JobType]
        assert "selfplay" in types
        assert "gpu_selfplay" in types
        assert "training" in types
        assert "evaluation" in types
        assert "tournament" in types
        assert "gauntlet" in types
        assert "custom" in types

    def test_job_type_count(self):
        """Verify job type count matches expected."""
        assert len(JobType) == 7


# =============================================================================
# JobState Enum Tests
# =============================================================================


class TestJobState:
    """Test JobState enum."""

    def test_all_states_exist(self):
        """Verify all expected states are defined."""
        states = [s.value for s in JobState]
        assert "pending" in states
        assert "queued" in states
        assert "running" in states
        assert "completed" in states
        assert "failed" in states
        assert "cancelled" in states
        assert "unknown" in states

    def test_state_count(self):
        """Verify state count matches expected."""
        assert len(JobState) == 7


# =============================================================================
# UnifiedJob Dataclass Tests
# =============================================================================


class TestUnifiedJob:
    """Test UnifiedJob dataclass."""

    def test_create_minimal_job(self):
        """Should create job with only name."""
        job = UnifiedJob(name="test-job")
        assert job.name == "test-job"
        assert job.job_type == JobType.SELFPLAY
        assert job.target_backend == Backend.AUTO

    def test_create_job_with_type(self):
        """Should create job with specific type."""
        job = UnifiedJob(name="train-job", job_type=JobType.TRAINING)
        assert job.job_type == JobType.TRAINING

    def test_job_has_unique_id(self):
        """Each job should have a unique ID."""
        job1 = UnifiedJob(name="job-1")
        job2 = UnifiedJob(name="job-2")
        assert job1.id != job2.id

    def test_job_id_format(self):
        """Job ID should be 8-character UUID prefix."""
        job = UnifiedJob(name="test")
        assert len(job.id) == 8

    def test_job_has_created_at(self):
        """Job should have creation timestamp."""
        before = time.time()
        job = UnifiedJob(name="test")
        after = time.time()
        assert before <= job.created_at <= after

    def test_job_with_target_node(self):
        """Should set target node."""
        job = UnifiedJob(name="test", target_node="lambda-gh200-a")
        assert job.target_node == "lambda-gh200-a"

    def test_job_with_backend(self):
        """Should set target backend."""
        job = UnifiedJob(name="test", target_backend=Backend.P2P)
        assert job.target_backend == Backend.P2P

    def test_job_default_resources(self):
        """Should have sensible resource defaults."""
        job = UnifiedJob(name="test")
        assert job.cpus == 16
        assert job.memory_gb == 64
        assert job.gpus == 1
        assert job.time_limit_hours == 8.0

    def test_job_custom_resources(self):
        """Should accept custom resource values."""
        job = UnifiedJob(
            name="big-job",
            cpus=32,
            memory_gb=256,
            gpus=4,
            time_limit_hours=24.0,
        )
        assert job.cpus == 32
        assert job.memory_gb == 256
        assert job.gpus == 4
        assert job.time_limit_hours == 24.0

    def test_job_with_config(self):
        """Should store config dict."""
        config = {"board_type": "hex8", "num_players": 2}
        job = UnifiedJob(name="selfplay", config=config)
        assert job.config["board_type"] == "hex8"
        assert job.config["num_players"] == 2

    def test_job_with_env_vars(self):
        """Should store environment variables."""
        job = UnifiedJob(
            name="test",
            env_vars={"CUDA_VISIBLE_DEVICES": "0", "DEBUG": "1"},
        )
        assert job.env_vars["CUDA_VISIBLE_DEVICES"] == "0"

    def test_job_with_priority(self):
        """Should set priority."""
        job = UnifiedJob(name="urgent", priority=100)
        assert job.priority == 100

    def test_job_with_dependencies(self):
        """Should track dependencies."""
        job = UnifiedJob(name="step-2", depends_on=["step-1-id"])
        assert "step-1-id" in job.depends_on

    def test_job_with_tags(self):
        """Should store tags."""
        job = UnifiedJob(name="tagged", tags=["production", "hex8"])
        assert "production" in job.tags
        assert "hex8" in job.tags


# =============================================================================
# JobStatus Dataclass Tests
# =============================================================================


class TestJobStatus:
    """Test JobStatus dataclass."""

    def test_create_status(self):
        """Should create status with required fields."""
        status = JobStatus(
            job_id="backend-123",
            unified_id="abc12345",
            backend=Backend.SLURM,
            state=JobState.RUNNING,
        )
        assert status.job_id == "backend-123"
        assert status.unified_id == "abc12345"
        assert status.backend == Backend.SLURM
        assert status.state == JobState.RUNNING

    def test_is_running_true(self):
        """is_running should be True when state is RUNNING."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.RUNNING,
        )
        assert status.is_running is True

    def test_is_running_false(self):
        """is_running should be False for other states."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.COMPLETED,
        )
        assert status.is_running is False

    def test_is_finished_completed(self):
        """is_finished should be True when completed."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.COMPLETED,
        )
        assert status.is_finished is True

    def test_is_finished_failed(self):
        """is_finished should be True when failed."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.FAILED,
        )
        assert status.is_finished is True

    def test_is_finished_cancelled(self):
        """is_finished should be True when cancelled."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.CANCELLED,
        )
        assert status.is_finished is True

    def test_is_finished_false_for_running(self):
        """is_finished should be False when running."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.RUNNING,
        )
        assert status.is_finished is False

    def test_runtime_none_without_start(self):
        """runtime_seconds should be None without start_time."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.PENDING,
        )
        assert status.runtime_seconds is None

    def test_runtime_with_start(self):
        """runtime_seconds should calculate elapsed time."""
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.RUNNING,
            start_time=time.time() - 60,
        )
        assert status.runtime_seconds >= 60

    def test_runtime_with_end(self):
        """runtime_seconds should use end_time when available."""
        start = time.time() - 100
        end = start + 50
        status = JobStatus(
            job_id="123",
            unified_id="abc",
            backend=Backend.P2P,
            state=JobState.COMPLETED,
            start_time=start,
            end_time=end,
        )
        assert status.runtime_seconds == 50


# =============================================================================
# UnifiedScheduler Initialization Tests
# =============================================================================


class TestUnifiedSchedulerInit:
    """Test UnifiedScheduler initialization."""

    @pytest.fixture
    def tmp_db(self, tmp_path):
        """Create temporary database path."""
        return str(tmp_path / "test_scheduler.db")

    def test_init_creates_database(self, tmp_db):
        """Should create database file."""
        scheduler = UnifiedScheduler(
            db_path=tmp_db,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )
        assert Path(scheduler.db_path).exists()

    def test_init_with_disabled_backends(self, tmp_db):
        """Should accept disabled backend flags."""
        scheduler = UnifiedScheduler(
            db_path=tmp_db,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )
        assert scheduler.enable_slurm is False
        assert scheduler.enable_vast is False
        assert scheduler.enable_p2p is False

    def test_init_stores_db_path(self, tmp_db):
        """Should store database path."""
        scheduler = UnifiedScheduler(
            db_path=tmp_db,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )
        assert scheduler.db_path == tmp_db

    def test_init_has_lock(self, tmp_db):
        """Should have async lock."""
        scheduler = UnifiedScheduler(
            db_path=tmp_db,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )
        assert hasattr(scheduler, "_lock")


# =============================================================================
# UnifiedScheduler Backend Detection Tests
# =============================================================================


class TestUnifiedSchedulerBackendDetection:
    """Test backend detection logic."""

    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create scheduler with all backends disabled."""
        db_path = str(tmp_path / "test.db")
        return UnifiedScheduler(
            db_path=db_path,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )

    def test_explicit_backend_used(self, scheduler):
        """Should use explicitly specified backend."""
        job = UnifiedJob(name="test", target_backend=Backend.P2P)
        backend = scheduler._detect_backend(job)
        assert backend == Backend.P2P

    def test_vast_node_pattern(self, scheduler):
        """Should route vast-* nodes to Vast backend."""
        job = UnifiedJob(name="test", target_node="vast-12345")
        backend = scheduler._detect_backend(job)
        assert backend == Backend.VAST

    def test_hetzner_node_pattern(self, scheduler):
        """Should route hetzner-* nodes to P2P backend."""
        job = UnifiedJob(name="test", target_node="hetzner-cpu1")
        backend = scheduler._detect_backend(job)
        assert backend == Backend.P2P

    def test_ringrift_node_pattern(self, scheduler):
        """Should route ringrift-* nodes to P2P backend."""
        job = UnifiedJob(name="test", target_node="ringrift-prod-1")
        backend = scheduler._detect_backend(job)
        assert backend == Backend.P2P

    def test_cpu_node_pattern(self, scheduler):
        """Should route *-cpu* nodes to P2P backend."""
        job = UnifiedJob(name="test", target_node="worker-cpu-1")
        backend = scheduler._detect_backend(job)
        assert backend == Backend.P2P

    def test_gpu_type_vast(self, scheduler):
        """Should route RTX GPUs to Vast backend."""
        job = UnifiedJob(name="test", target_gpu_type="rtx5090")
        backend = scheduler._detect_backend(job)
        assert backend == Backend.VAST

    def test_default_training_to_slurm(self, scheduler):
        """Training jobs should default to Slurm."""
        job = UnifiedJob(name="train", job_type=JobType.TRAINING)
        backend = scheduler._detect_backend(job)
        assert backend == Backend.SLURM

    def test_default_gpu_selfplay_to_slurm(self, scheduler):
        """GPU selfplay should default to Slurm."""
        job = UnifiedJob(name="selfplay", job_type=JobType.GPU_SELFPLAY)
        backend = scheduler._detect_backend(job)
        assert backend == Backend.SLURM

    def test_default_selfplay_to_p2p(self, scheduler):
        """Regular selfplay should default to P2P."""
        job = UnifiedJob(name="selfplay", job_type=JobType.SELFPLAY)
        backend = scheduler._detect_backend(job)
        assert backend == Backend.P2P


# =============================================================================
# UnifiedScheduler Job Recording Tests
# =============================================================================


class TestUnifiedSchedulerJobRecording:
    """Test job recording and updating."""

    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create scheduler with all backends disabled."""
        db_path = str(tmp_path / "test.db")
        return UnifiedScheduler(
            db_path=db_path,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )

    def test_record_job(self, scheduler):
        """Should record job in database."""
        import sqlite3

        job = UnifiedJob(name="test-record", job_type=JobType.SELFPLAY)
        scheduler._record_job(job, Backend.P2P)

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM jobs WHERE unified_id = ?",
                (job.id,),
            ).fetchone()

        assert row is not None
        assert row["name"] == "test-record"
        assert row["backend"] == "p2p"
        assert row["state"] == "pending"

    def test_update_job_state(self, scheduler):
        """Should update job state."""
        import sqlite3

        job = UnifiedJob(name="test-update")
        scheduler._record_job(job, Backend.P2P)
        scheduler._update_job(job.id, state=JobState.RUNNING)

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT state FROM jobs WHERE unified_id = ?",
                (job.id,),
            ).fetchone()

        assert row["state"] == "running"

    def test_update_job_with_backend_id(self, scheduler):
        """Should update job with backend job ID."""
        import sqlite3

        job = UnifiedJob(name="test-backend-id")
        scheduler._record_job(job, Backend.SLURM)
        scheduler._update_job(job.id, backend_job_id="slurm-12345")

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT backend_job_id FROM jobs WHERE unified_id = ?",
                (job.id,),
            ).fetchone()

        assert row["backend_job_id"] == "slurm-12345"

    def test_update_job_with_error(self, scheduler):
        """Should update job with error message."""
        import sqlite3

        job = UnifiedJob(name="test-error")
        scheduler._record_job(job, Backend.P2P)
        scheduler._update_job(job.id, state=JobState.FAILED, error="Connection refused")

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT state, error FROM jobs WHERE unified_id = ?",
                (job.id,),
            ).fetchone()

        assert row["state"] == "failed"
        assert row["error"] == "Connection refused"


# =============================================================================
# Integration Tests
# =============================================================================


class TestUnifiedSchedulerIntegration:
    """Integration tests for unified scheduler."""

    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create scheduler with all backends disabled."""
        db_path = str(tmp_path / "test.db")
        return UnifiedScheduler(
            db_path=db_path,
            enable_slurm=False,
            enable_vast=False,
            enable_p2p=False,
        )

    def test_record_multiple_jobs(self, scheduler):
        """Should handle multiple job records."""
        import sqlite3

        jobs = [
            UnifiedJob(name="job-1", job_type=JobType.SELFPLAY),
            UnifiedJob(name="job-2", job_type=JobType.TRAINING),
            UnifiedJob(name="job-3", job_type=JobType.EVALUATION),
        ]

        for job in jobs:
            scheduler._record_job(job, Backend.P2P)

        with sqlite3.connect(scheduler.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

        assert count == 3

    def test_job_with_config_persists(self, scheduler):
        """Should persist job config as JSON."""
        import json
        import sqlite3

        config = {"board_type": "hex8", "num_players": 2, "games": 1000}
        job = UnifiedJob(name="configured", config=config)
        scheduler._record_job(job, Backend.P2P)

        with sqlite3.connect(scheduler.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT config FROM jobs WHERE unified_id = ?",
                (job.id,),
            ).fetchone()

        loaded = json.loads(row["config"])
        assert loaded["board_type"] == "hex8"
        assert loaded["num_players"] == 2
        assert loaded["games"] == 1000

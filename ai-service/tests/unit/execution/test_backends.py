"""Comprehensive unit tests for app.execution.backends module.

Tests cover:
- BackendType enum
- WorkerStatus dataclass
- JobResult dataclass
- OrchestratorBackend interface
- LocalBackend implementation
- SSHBackend implementation (mocked SSH)
- SlurmBackend implementation (mocked sbatch)
- P2PBackend implementation (mocked HTTP)

December 2025: Created as part of test coverage improvement initiative.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.execution.backends import (
    BackendType,
    WorkerStatus,
    JobResult,
    OrchestratorBackend,
    LocalBackend,
    SSHBackend,
    SlurmBackend,
    P2PBackend,
    get_backend,
)


class TestBackendType:
    """Tests for BackendType enum."""

    def test_all_backend_types_defined(self):
        """Test that all expected backend types are defined."""
        assert BackendType.LOCAL == "local"
        assert BackendType.SSH == "ssh"
        assert BackendType.P2P == "p2p"
        assert BackendType.SLURM == "slurm"
        assert BackendType.HYBRID == "hybrid"

    def test_backend_type_is_str_enum(self):
        """Test that BackendType values are strings."""
        for bt in BackendType:
            assert isinstance(bt.value, str)

    def test_backend_type_from_string(self):
        """Test creating BackendType from string."""
        assert BackendType("local") == BackendType.LOCAL
        assert BackendType("ssh") == BackendType.SSH
        assert BackendType("p2p") == BackendType.P2P

    def test_backend_type_invalid_raises(self):
        """Test that invalid backend type raises ValueError."""
        with pytest.raises(ValueError):
            BackendType("invalid_backend")


class TestWorkerStatus:
    """Tests for WorkerStatus dataclass."""

    def test_worker_status_creation(self):
        """Test creating a WorkerStatus instance."""
        status = WorkerStatus(
            name="worker-1",
            available=True,
            cpu_percent=50.0,
            memory_percent=30.0,
            active_jobs=2,
        )
        assert status.name == "worker-1"
        assert status.available is True
        assert status.cpu_percent == 50.0
        assert status.memory_percent == 30.0
        assert status.active_jobs == 2

    def test_worker_status_defaults(self):
        """Test WorkerStatus default values."""
        status = WorkerStatus(name="worker-2", available=False)
        assert status.cpu_percent == 0.0
        assert status.memory_percent == 0.0
        assert status.active_jobs == 0
        assert status.last_seen is None
        assert status.metadata == {}

    def test_worker_status_with_metadata(self):
        """Test WorkerStatus with metadata."""
        status = WorkerStatus(
            name="worker-3",
            available=True,
            metadata={"gpu": "RTX 4090", "provider": "vast.ai"},
        )
        assert status.metadata["gpu"] == "RTX 4090"
        assert status.metadata["provider"] == "vast.ai"


class TestJobResult:
    """Tests for JobResult dataclass."""

    def test_job_result_success(self):
        """Test creating a successful JobResult."""
        result = JobResult(
            job_id="job-123",
            success=True,
            worker="worker-1",
            output={"games": 100, "samples": 5000},
            duration_seconds=3600.0,
        )
        assert result.job_id == "job-123"
        assert result.success is True
        assert result.worker == "worker-1"
        assert result.output["games"] == 100
        assert result.duration_seconds == 3600.0
        assert result.error is None

    def test_job_result_failure(self):
        """Test creating a failed JobResult."""
        result = JobResult(
            job_id="job-456",
            success=False,
            worker="worker-2",
            output=None,
            duration_seconds=10.0,
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestOrchestratorBackendInterface:
    """Tests for OrchestratorBackend abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that OrchestratorBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OrchestratorBackend()

    def test_interface_has_required_methods(self):
        """Test that OrchestratorBackend defines all required abstract methods."""
        required_methods = [
            "get_available_workers",
            "run_selfplay",
            "run_tournament",
            "run_training",
            "sync_models",
            "sync_data",
        ]
        for method in required_methods:
            assert hasattr(OrchestratorBackend, method)


class TestLocalBackend:
    """Tests for LocalBackend implementation."""

    @pytest.fixture
    def local_backend(self, tmp_path):
        """Create a LocalBackend instance with temporary directory."""
        return LocalBackend(working_dir=str(tmp_path))

    def test_initialization(self, tmp_path):
        """Test LocalBackend initialization."""
        backend = LocalBackend(working_dir=str(tmp_path))
        # LocalBackend stores executor, not _working_dir directly
        assert hasattr(backend, "executor")
        assert hasattr(backend, "_ai_service_root")

    def test_initialization_default_working_dir(self):
        """Test LocalBackend uses current directory as default."""
        backend = LocalBackend()
        # Should not raise, executor handles default
        assert hasattr(backend, "executor")

    @pytest.mark.asyncio
    async def test_get_available_workers(self, local_backend):
        """Test getting available workers returns local worker."""
        workers = await local_backend.get_available_workers()
        assert len(workers) == 1
        assert workers[0].name == "local"
        assert workers[0].available is True

    @pytest.mark.asyncio
    async def test_get_available_workers_cpu_info(self, local_backend):
        """Test that worker status includes CPU info."""
        workers = await local_backend.get_available_workers()
        # CPU percent should be a valid number
        assert 0 <= workers[0].cpu_percent <= 100

    @pytest.mark.asyncio
    async def test_sync_models_local(self, local_backend):
        """Test sync_models on local backend."""
        # LocalBackend.sync_models may be implemented or no-op
        try:
            result = await local_backend.sync_models(model_paths=["/path/to/model.pth"])
            # Should complete without error
        except NotImplementedError:
            pass  # Acceptable for local backend

    @pytest.mark.asyncio
    async def test_sync_data_local(self, local_backend):
        """Test sync_data on local backend (no-op for local backend)."""
        # sync_data takes source_workers and target_path, not source/destination
        result = await local_backend.sync_data(source_workers=None, target_path="/tmp/test")
        # LocalBackend.sync_data is a no-op and returns {"local": 0}
        assert result == {"local": 0}


class TestSSHBackend:
    """Tests for SSHBackend implementation (mocked SSH)."""

    @pytest.fixture
    def ssh_backend(self):
        """Create an SSHBackend instance with mocked hosts."""
        with patch.object(SSHBackend, "_load_hosts"):
            backend = SSHBackend(hosts_config_path=None)
            backend._hosts = {
                "worker-1": {"host": "192.168.1.1", "user": "root", "port": 22},
                "worker-2": {"host": "192.168.1.2", "user": "root", "port": 22},
            }
            return backend

    def test_initialization(self):
        """Test SSHBackend initialization."""
        with patch.object(SSHBackend, "_load_hosts") as mock_load:
            backend = SSHBackend(hosts_config_path="/path/to/hosts.yaml")
            mock_load.assert_called_once()

    def test_has_hosts_attribute(self):
        """Test SSHBackend has _hosts attribute after init."""
        with patch.object(SSHBackend, "_load_hosts"):
            backend = SSHBackend(hosts_config_path=None)
            assert hasattr(backend, "_hosts")

    @pytest.mark.asyncio
    async def test_get_available_workers_returns_list(self, ssh_backend):
        """Test getting available workers returns a list."""
        workers = await ssh_backend.get_available_workers()
        assert isinstance(workers, list)

    @pytest.mark.asyncio
    async def test_run_selfplay_with_no_workers(self, ssh_backend):
        """Test run_selfplay falls back to local when no SSH workers available."""
        # Mock get_available_workers to return empty list
        ssh_backend._hosts = {}
        with patch.object(ssh_backend, "get_available_workers", new=AsyncMock(return_value=[])):
            # SSHBackend falls back to LocalBackend when no workers available
            with patch("app.execution.backends.LocalBackend") as MockLocalBackend:
                mock_local_instance = MagicMock()
                mock_local_instance.run_selfplay = AsyncMock(return_value=[
                    MagicMock(success=True, worker="local")
                ])
                MockLocalBackend.return_value = mock_local_instance

                result = await ssh_backend.run_selfplay(
                    games=100,
                    board_type="square8",
                    num_players=2,
                )
                # Should have called LocalBackend as fallback
                MockLocalBackend.assert_called_once()
                mock_local_instance.run_selfplay.assert_called_once()


class TestSlurmBackend:
    """Tests for SlurmBackend implementation (mocked Slurm)."""

    @pytest.fixture
    def slurm_config(self):
        """Create a mock Slurm configuration."""
        config = MagicMock()
        config.partition = "gpu"
        config.account = "research"
        config.time_limit = "24:00:00"
        config.nodes = 1
        config.ntasks = 1
        config.cpus_per_task = 4
        config.mem = "32G"
        config.gpus = 1
        config.output_pattern = "logs/%j.out"
        config.error_pattern = "logs/%j.err"
        # Required attributes for initialization
        config.job_dir = "jobs"
        config.log_dir = "logs"
        config.shared_root = None
        config.poll_interval_seconds = 20
        config.venv_activate = None
        config.venv_activate_arm64 = None
        config.setup_commands = []
        config.extra_sbatch_args = []
        return config

    @pytest.fixture
    def slurm_backend(self, slurm_config, tmp_path):
        """Create a SlurmBackend instance."""
        return SlurmBackend(config=slurm_config, working_dir=str(tmp_path))

    def test_initialization(self, slurm_config, tmp_path):
        """Test SlurmBackend initialization."""
        backend = SlurmBackend(config=slurm_config, working_dir=str(tmp_path))
        # SlurmBackend stores config as self.config, not self._config
        assert backend.config == slurm_config
        # Check it has repo_root and directories
        assert hasattr(backend, "repo_root")
        assert hasattr(backend, "job_dir")
        assert hasattr(backend, "log_dir")

    def test_normalize_job_name(self, slurm_backend):
        """Test job name normalization."""
        result = slurm_backend._normalize_job_name("my-job_123")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_sbatch_args_returns_list(self, slurm_backend):
        """Test building sbatch arguments returns list."""
        # Signature is: _build_sbatch_args(job_name, work_type, overrides=None)
        args = slurm_backend._build_sbatch_args(
            job_name="test_job",
            work_type="selfplay",
            overrides=None,
        )
        assert isinstance(args, list)
        # Should have some arguments (at least --job-name, --output, --error)
        assert len(args) >= 6  # --job-name, value, --output, value, --error, value

    @pytest.mark.asyncio
    async def test_get_available_workers_returns_list(self, slurm_backend):
        """Test get_available_workers returns a list."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"node1 idle\n", b""))
            mock_exec.return_value = mock_process

            workers = await slurm_backend.get_available_workers()
            assert isinstance(workers, list)


class TestP2PBackend:
    """Tests for P2PBackend implementation (mocked HTTP)."""

    @pytest.fixture
    def p2p_backend(self):
        """Create a P2PBackend instance."""
        return P2PBackend(leader_url="http://localhost:8770")

    def test_initialization(self):
        """Test P2PBackend initialization."""
        backend = P2PBackend(leader_url="http://localhost:8770")
        assert backend.leader_url == "http://localhost:8770"

    def test_initialization_default_url(self):
        """Test P2PBackend uses None by default (auto-detect)."""
        backend = P2PBackend()
        # leader_url is None by default, will be resolved via _get_leader_url
        assert backend.leader_url is None

    def test_initialization_poll_interval(self):
        """Test P2PBackend poll_interval parameter."""
        backend = P2PBackend(poll_interval=10.0)
        assert backend.poll_interval == 10.0

    def test_initialization_timeout(self):
        """Test P2PBackend timeout parameter."""
        backend = P2PBackend(timeout=7200.0)
        assert backend.default_timeout == 7200.0

    @pytest.mark.asyncio
    async def test_get_leader_url_with_explicit(self, p2p_backend):
        """Test getting leader URL when explicitly set."""
        url = await p2p_backend._get_leader_url()
        assert url == "http://localhost:8770"

    @pytest.mark.asyncio
    async def test_close_session_when_none(self):
        """Test closing when session is None (no-op)."""
        backend = P2PBackend(leader_url="http://localhost:8770")
        # Should not raise when session is None
        await backend.close()

    @pytest.mark.asyncio
    async def test_close_session_when_active(self):
        """Test closing an active session."""
        backend = P2PBackend(leader_url="http://localhost:8770")
        mock_session = AsyncMock()
        mock_session.close = AsyncMock()
        backend._session = mock_session
        await backend.close()
        # Session's close method should have been called
        mock_session.close.assert_called_once()
        # And _session should now be None
        assert backend._session is None


class TestGetBackend:
    """Tests for get_backend factory function."""

    def test_get_backend_local(self):
        """Test getting local backend."""
        backend = get_backend(BackendType.LOCAL, force_new=True)
        assert isinstance(backend, LocalBackend)

    def test_get_backend_p2p(self):
        """Test getting P2P backend."""
        backend = get_backend(BackendType.P2P, force_new=True)
        assert isinstance(backend, P2PBackend)

    def test_get_backend_ssh(self):
        """Test getting SSH backend."""
        with patch.object(SSHBackend, "_load_hosts"):
            backend = get_backend(BackendType.SSH, force_new=True)
            assert isinstance(backend, SSHBackend)

    def test_get_backend_caches_instance(self):
        """Test that get_backend caches instances."""
        backend1 = get_backend(BackendType.LOCAL, force_new=True)
        backend2 = get_backend(BackendType.LOCAL, force_new=False)
        # Same instance should be returned (cached)
        assert backend1 is backend2

    def test_get_backend_force_new(self):
        """Test that force_new creates new instance."""
        backend1 = get_backend(BackendType.LOCAL, force_new=True)
        backend2 = get_backend(BackendType.LOCAL, force_new=True)
        # Different instances should be returned
        assert backend1 is not backend2


class TestBackendIntegration:
    """Integration tests for backend cooperation."""

    @pytest.mark.asyncio
    async def test_local_backend_selfplay_integration(self, tmp_path):
        """Test LocalBackend can execute selfplay (smoke test)."""
        backend = LocalBackend(working_dir=str(tmp_path))

        # This is a smoke test - actual selfplay would require full setup
        workers = await backend.get_available_workers()
        assert len(workers) > 0
        assert workers[0].available

    @pytest.mark.asyncio
    async def test_backend_worker_status_serializable(self, tmp_path):
        """Test that WorkerStatus can be serialized."""
        import json

        backend = LocalBackend(working_dir=str(tmp_path))
        workers = await backend.get_available_workers()

        # Convert to dict and back
        worker_dict = {
            "name": workers[0].name,
            "available": workers[0].available,
            "cpu_percent": workers[0].cpu_percent,
        }
        json_str = json.dumps(worker_dict)
        loaded = json.loads(json_str)
        assert loaded["name"] == workers[0].name

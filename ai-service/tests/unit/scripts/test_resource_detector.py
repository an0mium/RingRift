"""Tests for ResourceDetector from p2p module.

Tests cover:
- GPU detection (NVIDIA, Apple MPS)
- Memory detection (Linux, macOS)
- Resource usage monitoring
- NFS accessibility checking
- External work detection
- Startup grace period
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from scripts.p2p.resource_detector import ResourceDetector, ResourceDetectorMixin


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def detector(temp_dir):
    """Create a ResourceDetector instance with test settings."""
    return ResourceDetector(
        ringrift_path=temp_dir,
        start_time=time.time(),
        startup_grace_period=30.0,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """Test ResourceDetector initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        detector = ResourceDetector()

        assert detector.ringrift_path == Path.cwd()
        assert detector.startup_grace_period == 30.0
        assert detector._cached_gpu is None
        assert detector._cached_memory is None

    def test_init_custom_values(self, temp_dir):
        """Test initialization with custom values."""
        start = time.time()
        detector = ResourceDetector(
            ringrift_path=temp_dir,
            start_time=start,
            startup_grace_period=60.0,
        )

        assert detector.ringrift_path == temp_dir
        assert detector.start_time == start
        assert detector.startup_grace_period == 60.0


# =============================================================================
# GPU Detection Tests
# =============================================================================


class TestDetectGpu:
    """Test GPU detection."""

    def test_detect_nvidia_gpu(self, detector):
        """Test detecting NVIDIA GPU via nvidia-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090\n"

        with patch("subprocess.run", return_value=mock_result):
            has_gpu, gpu_name = detector.detect_gpu()

        assert has_gpu is True
        assert gpu_name == "NVIDIA GeForce RTX 4090"

    def test_detect_nvidia_multi_gpu(self, detector):
        """Test detecting multiple NVIDIA GPUs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA A100\nNVIDIA A100\n"

        with patch("subprocess.run", return_value=mock_result):
            has_gpu, gpu_name = detector.detect_gpu()

        assert has_gpu is True
        assert gpu_name == "NVIDIA A100"  # Returns first GPU

    def test_detect_no_nvidia_fallback_mps(self, detector):
        """Test fallback to Apple MPS when nvidia-smi fails."""
        # Clear cache to force re-detection
        detector._cached_gpu = None

        nvidia_result = MagicMock()
        nvidia_result.returncode = 1
        nvidia_result.stdout = ""

        mps_result = MagicMock()
        mps_result.returncode = 0
        mps_result.stdout = "True"

        def mock_run(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                return nvidia_result
            return mps_result

        with patch("subprocess.run", side_effect=mock_run):
            has_gpu, gpu_name = detector.detect_gpu()

        assert has_gpu is True
        assert gpu_name == "Apple MPS"

    def test_detect_no_gpu(self, detector):
        """Test when no GPU is available."""
        detector._cached_gpu = None

        def mock_run(cmd, **kwargs):
            if "nvidia-smi" in cmd:
                raise FileNotFoundError()
            result = MagicMock()
            result.stdout = "False"
            return result

        with patch("subprocess.run", side_effect=mock_run):
            has_gpu, gpu_name = detector.detect_gpu()

        assert has_gpu is False
        assert gpu_name == ""

    def test_detect_gpu_caching(self, detector):
        """Test that GPU detection result is cached."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test GPU\n"

        with patch("subprocess.run", return_value=mock_result) as mock:
            # First call
            detector.detect_gpu()
            # Second call should use cache
            detector.detect_gpu()

        # subprocess.run should only be called once
        assert mock.call_count == 1


# =============================================================================
# Memory Detection Tests
# =============================================================================


class TestDetectMemory:
    """Test memory detection."""

    def test_detect_memory_linux(self, detector):
        """Test memory detection on Linux."""
        detector._cached_memory = None
        meminfo_content = "MemTotal:       32768000 kB\nMemFree:        16384000 kB\n"

        with patch("sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=meminfo_content)):
                memory_gb = detector.detect_memory()

        assert memory_gb == 31  # 32768000 kB / 1024^2 = ~31 GB

    def test_detect_memory_macos(self, detector):
        """Test memory detection on macOS."""
        detector._cached_memory = None
        mock_result = MagicMock()
        mock_result.stdout = "34359738368\n"  # 32 GB in bytes

        with patch("sys.platform", "darwin"):
            with patch("subprocess.run", return_value=mock_result):
                memory_gb = detector.detect_memory()

        assert memory_gb == 32

    def test_detect_memory_fallback(self, detector):
        """Test memory detection fallback to default."""
        detector._cached_memory = None

        with patch("sys.platform", "linux"):
            with patch("builtins.open", side_effect=OSError()):
                memory_gb = detector.detect_memory()

        assert memory_gb == 16  # Default value

    def test_detect_memory_caching(self, detector):
        """Test that memory detection result is cached."""
        detector._cached_memory = None
        meminfo_content = "MemTotal:       16777216 kB\n"

        with patch("sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=meminfo_content)) as mock:
                detector.detect_memory()
                detector.detect_memory()

        assert mock.call_count == 1


# =============================================================================
# Network Detection Tests
# =============================================================================


class TestGetLocalIp:
    """Test local IP detection."""

    def test_get_local_ip_success(self, detector):
        """Test successful local IP detection."""
        mock_socket = MagicMock()
        mock_socket.getsockname.return_value = ("192.168.1.100", 12345)

        with patch("socket.socket", return_value=mock_socket):
            ip = detector.get_local_ip()

        assert ip == "192.168.1.100"

    def test_get_local_ip_failure(self, detector):
        """Test fallback when socket fails."""
        with patch("socket.socket", side_effect=OSError()):
            ip = detector.get_local_ip()

        assert ip == "127.0.0.1"


class TestGetTailscaleIp:
    """Test Tailscale IP detection."""

    def test_get_tailscale_ip_success(self, detector):
        """Test successful Tailscale IP detection."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "100.64.1.100\n"

        with patch("subprocess.run", return_value=mock_result):
            ip = detector.get_tailscale_ip()

        assert ip == "100.64.1.100"

    def test_get_tailscale_ip_not_installed(self, detector):
        """Test when Tailscale is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            ip = detector.get_tailscale_ip()

        assert ip == ""

    def test_get_tailscale_ip_not_connected(self, detector):
        """Test when Tailscale is installed but not connected."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            ip = detector.get_tailscale_ip()

        assert ip == ""


# =============================================================================
# Startup Grace Period Tests
# =============================================================================


class TestStartupGracePeriod:
    """Test startup grace period checking."""

    def test_in_grace_period(self, temp_dir):
        """Test detection when in grace period."""
        detector = ResourceDetector(
            ringrift_path=temp_dir,
            start_time=time.time(),
            startup_grace_period=30.0,
        )

        assert detector.is_in_startup_grace_period() is True

    def test_after_grace_period(self, temp_dir):
        """Test detection after grace period."""
        detector = ResourceDetector(
            ringrift_path=temp_dir,
            start_time=time.time() - 60.0,  # 60 seconds ago
            startup_grace_period=30.0,
        )

        assert detector.is_in_startup_grace_period() is False


# =============================================================================
# Resource Usage Tests
# =============================================================================


class TestGetResourceUsage:
    """Test resource usage monitoring."""

    def test_get_resource_usage_structure(self, detector):
        """Test that resource usage returns expected structure."""
        # Mock all subprocess calls to avoid real system calls
        with patch("subprocess.run"):
            with patch("shutil.disk_usage") as mock_disk:
                mock_disk.return_value = MagicMock(used=50e9, total=100e9, free=50e9)
                with patch("builtins.open", mock_open(read_data="0.5 0.4 0.3\n")):
                    with patch("sys.platform", "linux"):
                        with patch("os.cpu_count", return_value=8):
                            usage = detector.get_resource_usage()

        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "disk_percent" in usage
        assert "gpu_percent" in usage
        assert "gpu_memory_percent" in usage

    def test_get_resource_usage_disk(self, temp_dir):
        """Test disk usage calculation with real temp directory."""
        # Use a real temp directory so we get actual disk stats
        detector = ResourceDetector(ringrift_path=temp_dir)
        usage = detector.get_resource_usage()

        # Should have disk stats for the temp directory
        assert "disk_percent" in usage
        assert "disk_free_gb" in usage
        assert 0 <= usage["disk_percent"] <= 100
        assert usage["disk_free_gb"] >= 0

    def test_get_resource_usage_returns_defaults_on_error(self, detector):
        """Test that resource usage returns defaults when detection fails."""
        # Force an error by using an invalid path
        detector.ringrift_path = Path("/nonexistent/path/that/does/not/exist")
        usage = detector.get_resource_usage()

        # Should still return a valid structure with safe defaults
        assert "cpu_percent" in usage
        assert "memory_percent" in usage
        assert "disk_percent" in usage
        assert "gpu_percent" in usage
        assert "gpu_memory_percent" in usage


# =============================================================================
# NFS Accessibility Tests
# =============================================================================


class TestCheckNfsAccessible:
    """Test NFS accessibility checking."""

    def test_nfs_accessible_returns_bool(self, detector):
        """Test that NFS check returns a boolean."""
        # In test environment, NFS paths won't exist, so this tests the graceful fallback
        accessible = detector.check_nfs_accessible()
        assert isinstance(accessible, bool)
        # In test environment without NFS, should return False
        assert accessible is False

    def test_nfs_accessible_with_env_var(self, detector, temp_dir):
        """Test NFS check with custom env var path."""
        with patch.dict("os.environ", {"RINGRIFT_NFS_PATH": str(temp_dir)}):
            accessible = detector.check_nfs_accessible()
        # temp_dir exists and is a directory, but may not be considered NFS
        # depending on whether it can be iterated
        assert isinstance(accessible, bool)

    def test_nfs_not_accessible_on_error(self, detector):
        """Test that NFS check returns False when paths can't be accessed."""
        # Test with paths that don't exist (the default case in test environment)
        accessible = detector.check_nfs_accessible()
        assert accessible is False


# =============================================================================
# External Work Detection Tests
# =============================================================================


class TestDetectLocalExternalWork:
    """Test external work detection."""

    def test_detect_cmaes_running(self, detector):
        """Test detecting CMA-ES process."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n"

        def mock_run(cmd, **kwargs):
            if "pgrep" in cmd and "cmaes" in cmd[2].lower():
                return mock_result
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            work = detector.detect_local_external_work()

        assert work["cmaes_running"] is True

    def test_detect_gauntlet_running(self, detector):
        """Test detecting gauntlet process."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n"

        def mock_run(cmd, **kwargs):
            if "pgrep" in cmd and "gauntlet" in cmd[2].lower():
                return mock_result
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            return result

        with patch("subprocess.run", side_effect=mock_run):
            work = detector.detect_local_external_work()

        assert work["gauntlet_running"] is True

    def test_detect_no_external_work(self, detector):
        """Test when no external work is running."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            work = detector.detect_local_external_work()

        assert work["cmaes_running"] is False
        assert work["gauntlet_running"] is False
        assert work["tournament_running"] is False
        assert work["data_merge_running"] is False

    def test_detect_external_work_error_handling(self, detector):
        """Test error handling in external work detection."""
        with patch("subprocess.run", side_effect=subprocess.SubprocessError()):
            work = detector.detect_local_external_work()

        # Should return default False values on error
        assert work["cmaes_running"] is False
        assert work["gauntlet_running"] is False


# =============================================================================
# Mixin Tests
# =============================================================================


class TestResourceDetectorMixin:
    """Test ResourceDetectorMixin."""

    def test_mixin_delegates_to_detector(self, temp_dir):
        """Test that mixin methods delegate to detector."""

        class TestOrchestrator(ResourceDetectorMixin):
            def __init__(self, path):
                self._resource_detector = ResourceDetector(ringrift_path=path)

        orch = TestOrchestrator(temp_dir)

        # Test delegation - these should not raise
        assert isinstance(orch._detect_memory(), int)
        assert isinstance(orch._get_local_ip(), str)
        assert isinstance(orch._is_in_startup_grace_period(), bool)
        assert isinstance(orch._get_resource_usage(), dict)
        assert isinstance(orch._check_nfs_accessible(), bool)
        assert isinstance(orch._detect_local_external_work(), dict)

    def test_mixin_gpu_detection(self, temp_dir):
        """Test GPU detection through mixin."""

        class TestOrchestrator(ResourceDetectorMixin):
            def __init__(self, path):
                self._resource_detector = ResourceDetector(ringrift_path=path)

        orch = TestOrchestrator(temp_dir)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Test GPU\n"

        with patch("subprocess.run", return_value=mock_result):
            has_gpu, gpu_name = orch._detect_gpu()

        assert has_gpu is True
        assert gpu_name == "Test GPU"

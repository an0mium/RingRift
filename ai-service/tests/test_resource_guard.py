"""Tests for app/utils/resource_guard.py - Unified resource checking."""

import pytest
from unittest.mock import patch, MagicMock

from app.utils.resource_guard import (
    LIMITS,
    ResourceLimits,
    get_disk_usage,
    check_disk_space,
    check_disk_for_write,
    get_memory_usage,
    check_memory,
    get_cpu_usage,
    check_cpu,
    get_gpu_memory_usage,
    check_gpu_memory,
    can_proceed,
    get_resource_status,
    ResourceGuard,
    get_degradation_level,
    should_proceed_with_priority,
    OperationPriority,
    AsyncResourceLimiter,
)


class TestResourceLimits:
    """Test ResourceLimits configuration."""

    def test_limits_are_frozen(self):
        """Limits should be immutable."""
        with pytest.raises(Exception):  # FrozenInstanceError
            LIMITS.DISK_MAX_PERCENT = 50.0

    def test_disk_limits(self):
        """Verify disk limit values."""
        assert LIMITS.DISK_MAX_PERCENT == 80.0
        assert LIMITS.DISK_WARN_PERCENT == 75.0

    def test_cpu_limits(self):
        """Verify CPU limit values."""
        assert LIMITS.CPU_MAX_PERCENT == 80.0
        assert LIMITS.CPU_WARN_PERCENT == 70.0

    def test_gpu_limits(self):
        """Verify GPU limit values."""
        assert LIMITS.GPU_MAX_PERCENT == 80.0
        assert LIMITS.GPU_WARN_PERCENT == 70.0

    def test_memory_limits(self):
        """Verify memory limit values."""
        assert LIMITS.MEMORY_MAX_PERCENT == 90.0
        assert LIMITS.MEMORY_WARN_PERCENT == 80.0

    def test_load_factor(self):
        """Verify load factor limit."""
        assert LIMITS.LOAD_MAX_FACTOR == 1.5


class TestDiskUsage:
    """Test disk usage functions."""

    def test_get_disk_usage_returns_tuple(self):
        """get_disk_usage should return (percent, available_gb, total_gb)."""
        result = get_disk_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3
        used_pct, avail_gb, total_gb = result
        assert 0 <= used_pct <= 100
        assert avail_gb >= 0
        assert total_gb > 0

    def test_get_disk_usage_with_path(self, tmp_path):
        """get_disk_usage should work with custom path."""
        result = get_disk_usage(str(tmp_path))
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch('app.utils.resource_guard.shutil.disk_usage')
    def test_get_disk_usage_handles_error(self, mock_disk_usage):
        """get_disk_usage should return safe defaults on error."""
        mock_disk_usage.side_effect = OSError("disk error")
        result = get_disk_usage("/nonexistent")
        assert result == (0.0, 999.0, 999.0)

    @patch('app.utils.resource_guard.get_disk_usage')
    def test_check_disk_space_passes_when_ok(self, mock_get_disk):
        """check_disk_space should return True when space is available."""
        mock_get_disk.return_value = (50.0, 100.0, 200.0)  # 50% used, 100GB free
        assert check_disk_space(required_gb=10.0) is True

    @patch('app.utils.resource_guard.get_disk_usage')
    def test_check_disk_space_fails_on_percentage(self, mock_get_disk):
        """check_disk_space should return False when usage exceeds limit."""
        mock_get_disk.return_value = (85.0, 50.0, 200.0)  # 85% used
        assert check_disk_space(required_gb=1.0, log_warning=False) is False

    @patch('app.utils.resource_guard.get_disk_usage')
    def test_check_disk_space_fails_on_absolute(self, mock_get_disk):
        """check_disk_space should return False when free space insufficient."""
        mock_get_disk.return_value = (50.0, 1.0, 200.0)  # Only 1GB free
        assert check_disk_space(required_gb=10.0, log_warning=False) is False

    @patch('app.utils.resource_guard.check_disk_space')
    def test_check_disk_for_write(self, mock_check):
        """check_disk_for_write should add safety margin."""
        mock_check.return_value = True
        result = check_disk_for_write(estimated_size_mb=1024)  # 1GB
        mock_check.assert_called_once()
        call_args = mock_check.call_args
        assert call_args[1]['required_gb'] == 2.0


class TestMemoryUsage:
    """Test memory usage functions."""

    def test_get_memory_usage_returns_tuple(self):
        """get_memory_usage should return (percent, available_gb, total_gb)."""
        result = get_memory_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch('app.utils.resource_guard.get_memory_usage')
    def test_check_memory_passes_when_ok(self, mock_get_mem):
        """check_memory should return True when memory is available."""
        mock_get_mem.return_value = (50.0, 16.0, 32.0)
        assert check_memory(required_gb=4.0) is True

    @patch('app.utils.resource_guard.get_memory_usage')
    def test_check_memory_fails_on_percentage(self, mock_get_mem):
        """check_memory should return False when usage exceeds limit."""
        mock_get_mem.return_value = (95.0, 1.0, 32.0)
        assert check_memory(required_gb=0.5, log_warning=False) is False

    @patch('app.utils.resource_guard.get_memory_usage')
    def test_check_memory_fails_on_absolute(self, mock_get_mem):
        """check_memory should return False when free memory insufficient."""
        mock_get_mem.return_value = (50.0, 0.5, 32.0)
        assert check_memory(required_gb=4.0, log_warning=False) is False


class TestCPUUsage:
    """Test CPU usage functions."""

    def test_get_cpu_usage_returns_tuple(self):
        """get_cpu_usage should return (percent, load_per_cpu, cpu_count)."""
        result = get_cpu_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_check_cpu_passes_when_ok(self, mock_get_cpu):
        """check_cpu should return True when CPU usage is acceptable."""
        mock_get_cpu.return_value = (50.0, 0.5, 8)
        assert check_cpu() is True

    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_check_cpu_fails_on_percentage(self, mock_get_cpu):
        """check_cpu should return False when usage exceeds limit."""
        mock_get_cpu.return_value = (85.0, 0.5, 8)
        assert check_cpu(log_warning=False) is False

    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_check_cpu_fails_on_load(self, mock_get_cpu):
        """check_cpu should return False when load exceeds factor."""
        mock_get_cpu.return_value = (50.0, 2.0, 8)
        assert check_cpu(log_warning=False) is False


class TestGPUMemory:
    """Test GPU memory functions."""

    def test_get_gpu_memory_returns_tuple(self):
        """get_gpu_memory_usage should return tuple."""
        result = get_gpu_memory_usage()
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch('app.utils.resource_guard.get_gpu_memory_usage')
    def test_check_gpu_memory_passes_without_gpu(self, mock_get_gpu):
        """check_gpu_memory should pass when no GPU available."""
        mock_get_gpu.return_value = (0.0, 0.0, 0.0)
        assert check_gpu_memory(required_gb=8.0) is True

    @patch('app.utils.resource_guard.get_gpu_memory_usage')
    def test_check_gpu_memory_passes_when_ok(self, mock_get_gpu):
        """check_gpu_memory should return True when memory available."""
        mock_get_gpu.return_value = (50.0, 12.0, 24.0)
        assert check_gpu_memory(required_gb=8.0) is True

    @patch('app.utils.resource_guard.get_gpu_memory_usage')
    def test_check_gpu_memory_fails_on_percentage(self, mock_get_gpu):
        """check_gpu_memory should return False when exceeds limit."""
        mock_get_gpu.return_value = (85.0, 3.6, 24.0)
        assert check_gpu_memory(required_gb=1.0, log_warning=False) is False


class TestCanProceed:
    """Test combined resource check."""

    @patch('app.utils.resource_guard.check_disk_space')
    @patch('app.utils.resource_guard.check_memory')
    @patch('app.utils.resource_guard.check_cpu')
    def test_can_proceed_all_ok(self, mock_cpu, mock_mem, mock_disk):
        """can_proceed should return True when all checks pass."""
        mock_disk.return_value = True
        mock_mem.return_value = True
        mock_cpu.return_value = True
        assert can_proceed() is True

    @patch('app.utils.resource_guard.check_disk_space')
    @patch('app.utils.resource_guard.check_memory')
    @patch('app.utils.resource_guard.check_cpu')
    def test_can_proceed_fails_on_disk(self, mock_cpu, mock_mem, mock_disk):
        """can_proceed should return False when disk check fails."""
        mock_disk.return_value = False
        mock_mem.return_value = True
        mock_cpu.return_value = True
        assert can_proceed() is False

    @patch('app.utils.resource_guard.check_disk_space')
    @patch('app.utils.resource_guard.check_memory')
    @patch('app.utils.resource_guard.check_cpu')
    def test_can_proceed_skips_disabled_checks(self, mock_cpu, mock_mem, mock_disk):
        """can_proceed should skip disabled checks."""
        mock_disk.return_value = False
        mock_mem.return_value = True
        mock_cpu.return_value = True
        assert can_proceed(check_disk=False) is True


class TestResourceGuard:
    """Test ResourceGuard context manager."""

    @patch('app.utils.resource_guard.can_proceed')
    def test_resource_guard_sets_ok_when_available(self, mock_can_proceed):
        """ResourceGuard should set ok=True when resources available."""
        mock_can_proceed.return_value = True
        with ResourceGuard() as guard:
            assert guard.ok is True

    @patch('app.utils.resource_guard.can_proceed')
    @patch('app.utils.resource_guard.get_resource_status')
    def test_resource_guard_sets_ok_false_when_unavailable(
        self, mock_status, mock_can_proceed
    ):
        """ResourceGuard should set ok=False when resources unavailable."""
        mock_can_proceed.return_value = False
        mock_status.return_value = {"test": "status"}
        with ResourceGuard() as guard:
            assert guard.ok is False


class TestDegradationLevel:
    """Test graceful degradation functions."""

    @patch('app.utils.resource_guard.get_disk_usage')
    @patch('app.utils.resource_guard.get_memory_usage')
    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_degradation_level_0_normal(self, mock_cpu, mock_mem, mock_disk):
        """Level 0 when resources below 70% of limit."""
        mock_disk.return_value = (40.0, 100.0, 200.0)
        mock_mem.return_value = (45.0, 16.0, 32.0)
        mock_cpu.return_value = (40.0, 0.5, 8)
        assert get_degradation_level() == 0

    @patch('app.utils.resource_guard.get_disk_usage')
    @patch('app.utils.resource_guard.get_memory_usage')
    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_degradation_level_1_light(self, mock_cpu, mock_mem, mock_disk):
        """Level 1 when resources at 70-85% of limit."""
        mock_disk.return_value = (60.0, 80.0, 200.0)
        mock_mem.return_value = (45.0, 16.0, 32.0)
        mock_cpu.return_value = (40.0, 0.5, 8)
        assert get_degradation_level() == 1

    @patch('app.utils.resource_guard.get_disk_usage')
    @patch('app.utils.resource_guard.get_memory_usage')
    @patch('app.utils.resource_guard.get_cpu_usage')
    def test_degradation_level_4_critical(self, mock_cpu, mock_mem, mock_disk):
        """Level 4 when resources exceed limit."""
        mock_disk.return_value = (85.0, 30.0, 200.0)
        mock_mem.return_value = (45.0, 16.0, 32.0)
        mock_cpu.return_value = (40.0, 0.5, 8)
        assert get_degradation_level() == 4

    @patch('app.utils.resource_guard.get_degradation_level')
    def test_should_proceed_critical_runs_at_level_3(self, mock_degradation):
        """CRITICAL priority should proceed at degradation level 3."""
        mock_degradation.return_value = 3
        assert should_proceed_with_priority(OperationPriority.CRITICAL) is True
        # At level 4, even CRITICAL is paused (per implementation: priority > degradation)
        mock_degradation.return_value = 4
        assert should_proceed_with_priority(OperationPriority.CRITICAL) is False

    @patch('app.utils.resource_guard.get_degradation_level')
    def test_should_proceed_background_pauses_first(self, mock_degradation):
        """BACKGROUND priority should pause at level 1."""
        mock_degradation.return_value = 1
        assert should_proceed_with_priority(OperationPriority.BACKGROUND) is False
        assert should_proceed_with_priority(OperationPriority.LOW) is False
        assert should_proceed_with_priority(OperationPriority.NORMAL) is True


class TestResourceStatus:
    """Test resource status reporting."""

    def test_get_resource_status_returns_dict(self):
        """get_resource_status should return complete status dict."""
        status = get_resource_status(export_prometheus=False)
        assert isinstance(status, dict)
        assert "disk" in status
        assert "memory" in status
        assert "cpu" in status
        assert "gpu" in status
        assert "can_proceed" in status

    def test_get_resource_status_disk_keys(self):
        """Disk status should have required keys."""
        status = get_resource_status(export_prometheus=False)
        disk = status["disk"]
        assert "used_percent" in disk
        assert "available_gb" in disk
        assert "total_gb" in disk
        assert "ok" in disk
        assert "limit_percent" in disk


class TestAsyncResourceLimiter:
    """Test AsyncResourceLimiter class."""

    def test_limiter_initialization(self):
        """Limiter should initialize with defaults."""
        limiter = AsyncResourceLimiter()
        assert limiter.disk_required_gb == 2.0
        assert limiter.mem_required_gb == 1.0
        assert limiter.gpu_required_gb == 0.0

    def test_limiter_custom_values(self):
        """Limiter should accept custom values."""
        limiter = AsyncResourceLimiter(
            disk_required_gb=10.0,
            mem_required_gb=4.0,
            gpu_required_gb=8.0,
        )
        assert limiter.disk_required_gb == 10.0
        assert limiter.mem_required_gb == 4.0
        assert limiter.gpu_required_gb == 8.0

    @patch('app.utils.resource_guard.check_disk_space')
    @patch('app.utils.resource_guard.check_memory')
    @patch('app.utils.resource_guard.check_cpu')
    def test_limiter_check_resources(self, mock_cpu, mock_mem, mock_disk):
        """Limiter _check_resources should return ok and issues."""
        mock_disk.return_value = True
        mock_mem.return_value = True
        mock_cpu.return_value = True

        limiter = AsyncResourceLimiter()
        ok, issues = limiter._check_resources()
        assert ok is True
        assert issues == []

"""Tests for scripts/lib/health.py module.

Tests cover:
- DiskHealth, MemoryHealth, CPUHealth dataclasses
- System health checks
- HTTP service health checks
- Process health verification
"""

import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.lib.health import (
    CPUHealth,
    DiskHealth,
    GPUInfo,
    MemoryHealth,
    ServiceHealth,
    SystemHealth,
    check_cpu,
    check_disk_space,
    check_gpus,
    check_http_health,
    check_memory,
    check_port_open,
    check_process_health,
    check_system_health,
    wait_for_healthy,
)


class TestDiskHealth:
    """Tests for DiskHealth dataclass."""

    def test_gb_conversions(self):
        """Test byte to GB conversions."""
        health = DiskHealth(
            path="/",
            total_bytes=100 * 1024**3,  # 100 GB
            used_bytes=60 * 1024**3,    # 60 GB
            free_bytes=40 * 1024**3,    # 40 GB
            percent_used=60.0,
        )
        assert health.total_gb == 100.0
        assert health.used_gb == 60.0
        assert health.free_gb == 40.0

    def test_is_critical(self):
        """Test critical threshold detection."""
        # Critical by percent
        health = DiskHealth(
            path="/", total_bytes=100, used_bytes=95, free_bytes=5,
            percent_used=95.0
        )
        assert health.is_critical is True

        # Critical by free space
        health = DiskHealth(
            path="/",
            total_bytes=100 * 1024**3,
            used_bytes=99 * 1024**3,
            free_bytes=1 * 1024**3,  # 1 GB free
            percent_used=99.0
        )
        assert health.is_critical is True

        # Not critical
        health = DiskHealth(
            path="/",
            total_bytes=100 * 1024**3,
            used_bytes=50 * 1024**3,
            free_bytes=50 * 1024**3,
            percent_used=50.0
        )
        assert health.is_critical is False

    def test_is_warning(self):
        """Test warning threshold detection."""
        # Warning (>70%)
        health = DiskHealth(
            path="/", total_bytes=100, used_bytes=75, free_bytes=25,
            percent_used=75.0
        )
        assert health.is_warning is True

        # Not warning
        health = DiskHealth(
            path="/", total_bytes=100, used_bytes=50, free_bytes=50,
            percent_used=50.0
        )
        assert health.is_warning is False


class TestMemoryHealth:
    """Tests for MemoryHealth dataclass."""

    def test_gb_conversions(self):
        """Test byte to GB conversions."""
        health = MemoryHealth(
            total_bytes=16 * 1024**3,
            available_bytes=8 * 1024**3,
            used_bytes=8 * 1024**3,
            percent_used=50.0,
        )
        assert health.total_gb == 16.0
        assert health.available_gb == 8.0

    def test_is_critical(self):
        """Test critical threshold detection."""
        health = MemoryHealth(
            total_bytes=100, available_bytes=4, used_bytes=96,
            percent_used=96.0
        )
        assert health.is_critical is True

    def test_is_warning(self):
        """Test warning threshold detection."""
        health = MemoryHealth(
            total_bytes=100, available_bytes=10, used_bytes=90,
            percent_used=90.0
        )
        assert health.is_warning is True


class TestCPUHealth:
    """Tests for CPUHealth dataclass."""

    def test_load_per_core(self):
        """Test load per core calculation."""
        health = CPUHealth(
            percent_used=50.0,
            load_1min=4.0,
            load_5min=3.5,
            load_15min=3.0,
            core_count=4,
        )
        assert health.load_per_core == 1.0

    def test_is_overloaded(self):
        """Test overload detection."""
        # Overloaded (load > 2x cores)
        health = CPUHealth(
            percent_used=100.0,
            load_1min=10.0,
            load_5min=9.0,
            load_15min=8.0,
            core_count=4,
        )
        assert health.is_overloaded is True

        # Not overloaded
        health = CPUHealth(
            percent_used=50.0,
            load_1min=4.0,
            load_5min=3.5,
            load_15min=3.0,
            core_count=4,
        )
        assert health.is_overloaded is False


class TestGPUInfo:
    """Tests for GPUInfo dataclass."""

    def test_memory_calculations(self):
        """Test GPU memory calculations."""
        gpu = GPUInfo(
            index=0,
            name="NVIDIA RTX 3090",
            memory_total_mb=24576,
            memory_used_mb=12288,
            utilization_percent=50,
            temperature_c=65,
        )
        assert gpu.memory_free_mb == 12288
        assert gpu.memory_percent == 50.0


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_properties(self):
        """Test property accessors."""
        health = SystemHealth(
            disk=DiskHealth("/", 100, 60, 40, 60.0),
            memory=MemoryHealth(100, 50, 50, 50.0),
            cpu=CPUHealth(50.0, 2.0, 1.5, 1.0, 4),
            gpus=[GPUInfo(0, "GPU", 1000, 500, 50, 60)],
        )
        assert health.disk_percent == 60.0
        assert health.memory_percent == 50.0
        assert health.cpu_percent == 50.0
        assert health.gpu_count == 1


class TestCheckDiskSpace:
    """Tests for check_disk_space function."""

    def test_check_root(self):
        """Test checking root filesystem."""
        health = check_disk_space("/")
        assert health.total_bytes > 0
        assert health.path == "/"

    def test_check_tmp(self, tmp_path):
        """Test checking temp directory."""
        health = check_disk_space(tmp_path)
        assert health.total_bytes > 0

    def test_nonexistent_path(self):
        """Test checking nonexistent path."""
        health = check_disk_space("/nonexistent/path/xyz")
        # Should return zeros rather than raising
        assert health.total_bytes == 0


class TestCheckMemory:
    """Tests for check_memory function."""

    def test_returns_memory_info(self):
        """Test that memory info is returned."""
        health = check_memory()
        # Should return some memory info on any system
        assert health.total_bytes >= 0


class TestCheckCPU:
    """Tests for check_cpu function."""

    def test_returns_cpu_info(self):
        """Test that CPU info is returned."""
        health = check_cpu()
        assert health.core_count >= 1
        assert health.load_1min >= 0


class TestCheckGPUs:
    """Tests for check_gpus function."""

    @patch("scripts.lib.health.subprocess.run")
    def test_parses_nvidia_smi(self, mock_run):
        """Test parsing nvidia-smi output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="0, NVIDIA RTX 3090, 24576, 12288, 50, 65\n"
                   "1, NVIDIA RTX 3090, 24576, 8192, 30, 60\n"
        )

        gpus = check_gpus()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA RTX 3090"
        assert gpus[0].memory_total_mb == 24576

    @patch("scripts.lib.health.subprocess.run")
    def test_handles_no_gpus(self, mock_run):
        """Test handling when nvidia-smi fails."""
        mock_run.side_effect = FileNotFoundError()

        gpus = check_gpus()
        assert gpus == []


class TestCheckSystemHealth:
    """Tests for check_system_health function."""

    def test_returns_health_status(self):
        """Test that system health is returned."""
        health = check_system_health(include_gpus=False)

        assert health.disk is not None
        assert health.memory is not None
        assert health.cpu is not None
        assert isinstance(health.is_healthy, bool)


class TestCheckHttpHealth:
    """Tests for check_http_health function."""

    @patch("scripts.lib.health.urllib.request.urlopen")
    def test_healthy_service(self, mock_urlopen):
        """Test checking healthy HTTP service."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"healthy": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        health = check_http_health("http://localhost:8080/health")

        assert health.is_healthy is True
        assert health.status_code == 200

    @patch("scripts.lib.health.urllib.request.urlopen")
    def test_unhealthy_status(self, mock_urlopen):
        """Test checking service with bad status."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://test", 500, "Server Error", {}, None
        )

        health = check_http_health("http://localhost:8080/health")

        assert health.is_healthy is False
        assert health.status_code == 500

    @patch("scripts.lib.health.urllib.request.urlopen")
    def test_connection_error(self, mock_urlopen):
        """Test handling connection error."""
        mock_urlopen.side_effect = OSError("Connection refused")

        health = check_http_health("http://localhost:9999/health")

        assert health.is_healthy is False
        assert health.error != ""

    @patch("scripts.lib.health.urllib.request.urlopen")
    def test_json_field_check(self, mock_urlopen):
        """Test checking specific JSON field."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        health = check_http_health(
            "http://localhost:8080/health",
            json_field="status"
        )

        assert health.is_healthy is True


class TestCheckPortOpen:
    """Tests for check_port_open function."""

    def test_closed_port(self):
        """Test checking closed port."""
        # Port 59999 is very unlikely to be open
        result = check_port_open("localhost", 59999, timeout=0.5)
        assert result is False


class TestCheckProcessHealth:
    """Tests for check_process_health function."""

    def test_current_process(self):
        """Test checking current process."""
        is_healthy, count = check_process_health(pid=os.getpid())
        assert is_healthy is True
        assert count == 1

    def test_nonexistent_process(self):
        """Test checking nonexistent process."""
        is_healthy, count = check_process_health(pid=999999999)
        assert is_healthy is False
        assert count == 0

    @patch("scripts.lib.process.count_processes_by_pattern")
    def test_pattern_match(self, mock_count):
        """Test checking process by pattern."""
        mock_count.return_value = 3

        is_healthy, count = check_process_health(pattern="python", min_count=2)

        assert is_healthy is True
        assert count == 3


class TestWaitForHealthy:
    """Tests for wait_for_healthy function."""

    def test_immediate_success(self):
        """Test when check passes immediately."""
        def always_healthy():
            return True

        result = wait_for_healthy(always_healthy, timeout=1.0)
        assert result is True

    def test_eventual_success(self):
        """Test when check passes after retries."""
        call_count = [0]

        def eventually_healthy():
            call_count[0] += 1
            return call_count[0] >= 3

        result = wait_for_healthy(eventually_healthy, timeout=5.0, interval=0.1)
        assert result is True
        assert call_count[0] >= 3

    def test_timeout(self):
        """Test when check never passes."""
        def never_healthy():
            return False

        result = wait_for_healthy(never_healthy, timeout=0.2, interval=0.05)
        assert result is False

    def test_with_object_result(self):
        """Test with object that has is_healthy attribute."""
        class HealthResult:
            is_healthy = True

        def check():
            return HealthResult()

        result = wait_for_healthy(check, timeout=1.0)
        assert result is True

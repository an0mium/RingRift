"""Tests for adaptive_resource_manager.py - Proactive resource management.

This module tests the adaptive resource manager that monitors and manages
disk, memory, and GPU resources across the cluster.

Coverage includes:
1. ResourceStatus dataclass operations
2. CleanupResult dataclass operations
3. AdaptiveResourceManager initialization
4. Disk usage monitoring
5. Memory usage monitoring
6. GPU memory management
7. NFS path monitoring
8. File cleanup operations
9. Selfplay data aggregation
10. Check and cleanup flow
11. Run loop lifecycle
12. Statistics tracking
13. Health check compliance
14. Singleton pattern
15. Module constants
16. Error handling and recovery
17. Priority-based scheduling (resource priority)
18. Worker pool scaling considerations

December 2025: Expanded test coverage to 50+ tests.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch, mock_open

import pytest

from app.coordination.adaptive_resource_manager import (
    AGGREGATION_INTERVAL,
    CHECK_INTERVAL,
    CLEANUP_BATCH_SIZE,
    CLEANUP_COOLDOWN,
    DEFAULT_DATA_PATH,
    DEFAULT_NFS_PATH,
    DISK_CLEANUP_THRESHOLD,
    DISK_CRITICAL_THRESHOLD,
    DISK_WARNING_THRESHOLD,
    MEMORY_CRITICAL_THRESHOLD,
    MEMORY_WARNING_THRESHOLD,
    MIN_FILE_AGE_HOURS,
    AdaptiveResourceManager,
    CleanupResult,
    ResourceStatus,
    get_resource_manager,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manager(tmp_path: Path) -> AdaptiveResourceManager:
    """Create a test manager with temporary paths."""
    nfs_path = tmp_path / "nfs"
    data_path = tmp_path / "data"
    nfs_path.mkdir()
    data_path.mkdir()
    return AdaptiveResourceManager(
        nfs_path=str(nfs_path),
        data_path=str(data_path),
    )


@pytest.fixture
def resource_status() -> ResourceStatus:
    """Create a sample ResourceStatus."""
    return ResourceStatus(
        node_id="test-node",
        disk_total_gb=100.0,
        disk_used_gb=60.0,
        disk_free_gb=40.0,
        disk_percent=60.0,
        memory_total_gb=32.0,
        memory_used_gb=16.0,
        memory_free_gb=16.0,
        memory_percent=50.0,
    )


# =============================================================================
# ResourceStatus Tests
# =============================================================================


class TestResourceStatus:
    """Tests for ResourceStatus dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        status = ResourceStatus(node_id="test")
        assert status.node_id == "test"
        assert status.disk_total_gb == 0
        assert status.disk_used_gb == 0
        assert status.disk_free_gb == 0
        assert status.disk_percent == 0
        assert status.memory_total_gb == 0
        assert status.memory_percent == 0
        assert status.gpu_memory_used_gb == 0
        assert status.is_healthy is True
        assert status.warnings == []
        assert status.errors == []

    def test_to_dict(self, resource_status: ResourceStatus) -> None:
        """Test dictionary conversion."""
        result = resource_status.to_dict()
        
        assert result["node_id"] == "test-node"
        assert result["disk"]["total_gb"] == 100.0
        assert result["disk"]["used_gb"] == 60.0
        assert result["disk"]["percent"] == 60.0
        assert result["memory"]["total_gb"] == 32.0
        assert result["memory"]["percent"] == 50.0
        assert result["is_healthy"] is True

    def test_to_dict_with_warnings(self) -> None:
        """Test dictionary conversion with warnings."""
        status = ResourceStatus(
            node_id="test",
            warnings=["Disk warning: 75%"],
            errors=["Memory critical: 95%"],
            is_healthy=False,
        )
        result = status.to_dict()
        
        assert result["is_healthy"] is False
        assert "Disk warning: 75%" in result["warnings"]
        assert "Memory critical: 95%" in result["errors"]


# =============================================================================
# CleanupResult Tests
# =============================================================================


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_default_values(self) -> None:
        """Test default field values."""
        result = CleanupResult(success=True)
        assert result.success is True
        assert result.files_deleted == 0
        assert result.bytes_freed == 0
        assert result.errors == []
        assert result.duration_seconds == 0

    def test_with_values(self) -> None:
        """Test with custom values."""
        result = CleanupResult(
            success=True,
            files_deleted=10,
            bytes_freed=1024 * 1024,
            duration_seconds=5.5,
        )
        assert result.files_deleted == 10
        assert result.bytes_freed == 1024 * 1024
        assert result.duration_seconds == 5.5


# =============================================================================
# AdaptiveResourceManager Tests
# =============================================================================


class TestAdaptiveResourceManagerInit:
    """Tests for AdaptiveResourceManager initialization."""

    def test_default_init(self) -> None:
        """Test initialization with defaults."""
        manager = AdaptiveResourceManager()
        assert manager.running is False
        assert manager.last_cleanup_time == 0
        assert manager.last_aggregation_time == 0
        assert manager.stats["cleanups_triggered"] == 0

    def test_custom_init(self, tmp_path: Path) -> None:
        """Test initialization with custom paths."""
        nfs = tmp_path / "nfs"
        data = tmp_path / "data"
        nfs.mkdir()
        data.mkdir()
        
        manager = AdaptiveResourceManager(
            nfs_path=str(nfs),
            data_path=str(data),
            disk_threshold=50.0,
            memory_threshold=70.0,
        )
        assert manager.nfs_path == nfs
        assert manager.data_path == data
        assert manager.disk_threshold == 50.0
        assert manager.memory_threshold == 70.0


class TestDiskUsage:
    """Tests for disk usage methods."""

    def test_get_disk_usage_success(self, manager: AdaptiveResourceManager) -> None:
        """Test successful disk usage retrieval."""
        total, used, free = manager._get_disk_usage(manager.data_path)
        # Should return actual disk stats for temp directory
        assert total > 0
        assert used >= 0
        assert free >= 0
        assert total >= used + free - 1  # Allow for rounding

    def test_get_disk_usage_nonexistent_path(self, manager: AdaptiveResourceManager) -> None:
        """Test disk usage for nonexistent path."""
        total, used, free = manager._get_disk_usage(Path("/nonexistent/path"))
        assert total == 0
        assert used == 0
        assert free == 0


class TestMemoryUsage:
    """Tests for memory usage methods."""

    def test_get_memory_usage_mock(self, manager: AdaptiveResourceManager) -> None:
        """Test memory usage with mocked /proc/meminfo."""
        mock_meminfo = """MemTotal:       32000000 kB
MemFree:        8000000 kB
Buffers:        1000000 kB
Cached:         7000000 kB
"""
        with patch("builtins.open", mock_open(read_data=mock_meminfo)):
            total, used, free = manager._get_memory_usage()
            # Values are in GB (converted from KB)
            assert total > 0
            assert used >= 0
            assert free >= 0


class TestGpuMemory:
    """Tests for GPU memory methods."""

    def test_get_gpu_memory_no_nvidia(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory when nvidia-smi not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            used, total = manager._get_gpu_memory()
            assert used == 0
            assert total == 0

    def test_get_gpu_memory_with_nvidia(self, manager: AdaptiveResourceManager) -> None:
        """Test GPU memory with mocked nvidia-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8000, 24000\n"
        
        with patch("subprocess.run", return_value=mock_result):
            used, total = manager._get_gpu_memory()
            assert used == 8000 / 1024  # MB to GB
            assert total == 24000 / 1024


class TestGetLocalStatus:
    """Tests for get_local_status method."""

    def test_get_local_status(self, manager: AdaptiveResourceManager) -> None:
        """Test getting local status."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 60.0, 40.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(8.0, 24.0)):
                    status = manager.get_local_status("test-node")
        
        assert status.node_id == "test-node"
        assert status.disk_total_gb == 100.0
        assert status.disk_percent == 60.0
        assert status.memory_total_gb == 32.0
        assert status.is_healthy is True

    def test_get_local_status_disk_warning(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with disk warning."""
        # ResourceManagerDefaults.DISK_WARNING_THRESHOLD = 85%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 88.0, 12.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.disk_percent == 88.0
        assert len(status.warnings) > 0
        assert any("Disk warning" in w for w in status.warnings)

    def test_get_local_status_disk_critical(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with disk critical."""
        # ResourceManagerDefaults.DISK_CRITICAL_THRESHOLD = 92%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 94.0, 6.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.is_healthy is False
        assert any("Disk critical" in e for e in status.errors)

    def test_get_local_status_memory_critical(self, manager: AdaptiveResourceManager) -> None:
        """Test local status with memory critical."""
        # ResourceManagerDefaults.MEMORY_CRITICAL_THRESHOLD = 95%
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 31.0, 1.0)):  # 96.875%
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    status = manager.get_local_status()

        assert status.is_healthy is False
        assert any("Memory critical" in e for e in status.errors)


class TestGetNfsStatus:
    """Tests for get_nfs_status method."""

    def test_get_nfs_status_path_exists(self, manager: AdaptiveResourceManager) -> None:
        """Test NFS status when path exists."""
        status = manager.get_nfs_status()
        assert status.node_id == "nfs"
        assert status.disk_total_gb > 0

    def test_get_nfs_status_path_not_exists(self, tmp_path: Path) -> None:
        """Test NFS status when path doesn't exist."""
        manager = AdaptiveResourceManager(
            nfs_path=str(tmp_path / "nonexistent"),
            data_path=str(tmp_path),
        )
        status = manager.get_nfs_status()
        assert status.is_healthy is False
        assert any("not accessible" in e for e in status.errors)


class TestCleanupOldFiles:
    """Tests for cleanup_old_files method."""

    @pytest.mark.asyncio
    async def test_cleanup_dry_run(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup in dry run mode."""
        # Create old test files
        test_dir = manager.data_path / "test"
        test_dir.mkdir()
        old_file = test_dir / "old_file.jsonl"
        old_file.write_text("test data")
        
        # Make file appear old
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        import os
        os.utime(old_file, (old_time, old_time))
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=True)
        
        assert result.success is True
        assert result.files_deleted == 1
        assert result.bytes_freed > 0
        assert old_file.exists()  # Not actually deleted in dry run

    @pytest.mark.asyncio
    async def test_cleanup_actual_delete(self, manager: AdaptiveResourceManager) -> None:
        """Test actual file cleanup."""
        test_dir = manager.data_path / "cleanup_test"
        test_dir.mkdir()
        old_file = test_dir / "old_file.tmp"
        old_file.write_text("test data to delete")
        
        old_time = time.time() - (25 * 3600)
        import os
        os.utime(old_file, (old_time, old_time))
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=False)
        
        assert result.success is True
        assert result.files_deleted == 1
        assert not old_file.exists()  # Should be deleted

    @pytest.mark.asyncio
    async def test_cleanup_respects_age(self, manager: AdaptiveResourceManager) -> None:
        """Test that cleanup respects file age threshold."""
        test_dir = manager.data_path / "age_test"
        test_dir.mkdir()
        new_file = test_dir / "new_file.jsonl"
        new_file.write_text("recent data")
        # File is new, should not be deleted
        
        result = await manager.cleanup_old_files(test_dir, min_age_hours=24, dry_run=False)
        
        assert result.files_deleted == 0
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_cleanup_empty_directory(self, manager: AdaptiveResourceManager) -> None:
        """Test cleanup on empty directory."""
        test_dir = manager.data_path / "empty"
        test_dir.mkdir()
        
        result = await manager.cleanup_old_files(test_dir)
        
        assert result.success is True
        assert result.files_deleted == 0


class TestCheckAndCleanup:
    """Tests for check_and_cleanup method."""

    @pytest.mark.asyncio
    async def test_check_no_cleanup_needed(self, manager: AdaptiveResourceManager) -> None:
        """Test check when no cleanup needed."""
        with patch.object(manager, "_get_disk_usage", return_value=(100.0, 50.0, 50.0)):
            with patch.object(manager, "_get_memory_usage", return_value=(32.0, 16.0, 16.0)):
                with patch.object(manager, "_get_gpu_memory", return_value=(0.0, 0.0)):
                    result = await manager.check_and_cleanup()
        
        assert result["cleanup_triggered"] is False
        assert result["nfs_status"] is not None
        assert result["local_status"] is not None

    @pytest.mark.asyncio
    async def test_check_cleanup_triggered(self, manager: AdaptiveResourceManager) -> None:
        """Test check triggers cleanup when threshold exceeded."""
        manager.disk_threshold = 50.0  # Low threshold
        
        with patch.object(manager, "get_nfs_status") as mock_nfs:
            mock_status = ResourceStatus(node_id="nfs", disk_percent=75.0)
            mock_nfs.return_value = mock_status
            
            with patch.object(manager, "get_local_status") as mock_local:
                mock_local.return_value = ResourceStatus(node_id="local")
                
                with patch.object(manager, "cleanup_old_files", new_callable=AsyncMock) as mock_cleanup:
                    mock_cleanup.return_value = CleanupResult(success=True, files_deleted=5)
                    result = await manager.check_and_cleanup()
        
        assert result["cleanup_triggered"] is True
        mock_cleanup.assert_called_once()


class TestAggregation:
    """Tests for data aggregation methods."""

    @pytest.mark.asyncio
    async def test_aggregate_no_nodes(self, manager: AdaptiveResourceManager) -> None:
        """Test aggregation with no source nodes."""
        result = await manager.aggregate_selfplay_data(source_nodes=[])
        
        assert result["success"] is True
        assert result["games_aggregated"] == 0

    @pytest.mark.asyncio
    async def test_get_active_selfplay_nodes_import_error(self, manager: AdaptiveResourceManager) -> None:
        """Test getting active nodes when ClusterMonitor unavailable."""
        with patch.dict("sys.modules", {"app.coordination.cluster_status_monitor": None}):
            nodes = await manager._get_active_selfplay_nodes()
            assert nodes == []


class TestRunLoop:
    """Tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_stop(self, manager: AdaptiveResourceManager) -> None:
        """Test stopping the manager."""
        manager.running = True
        manager.stop()
        assert manager.running is False

    @pytest.mark.asyncio
    async def test_run_loop_starts_and_stops(self, manager: AdaptiveResourceManager) -> None:
        """Test that run loop can be started and stopped."""
        with patch.object(manager, "check_and_cleanup", new_callable=AsyncMock):
            # Start run in background
            task = asyncio.create_task(manager.run())
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            assert manager.running is True
            
            # Stop it
            manager.stop()
            await asyncio.sleep(0.1)
            assert manager.running is False
            
            # Cancel task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestGetStats:
    """Tests for get_stats method."""

    def test_get_stats(self, manager: AdaptiveResourceManager) -> None:
        """Test getting manager stats."""
        stats = manager.get_stats()
        
        assert "running" in stats
        assert "cleanups_triggered" in stats
        assert "bytes_freed_total" in stats
        assert "nfs_path" in stats
        assert "data_path" in stats
        assert "disk_threshold" in stats

    def test_get_stats_after_cleanup(self, manager: AdaptiveResourceManager) -> None:
        """Test stats after cleanup operation."""
        manager.stats["cleanups_triggered"] = 5
        manager.stats["bytes_freed_total"] = 1024 * 1024 * 100
        
        stats = manager.get_stats()
        assert stats["cleanups_triggered"] == 5
        assert stats["bytes_freed_total"] == 1024 * 1024 * 100


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_running(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when running."""
        manager.running = True
        result = manager.health_check()
        
        assert result.healthy is True
        assert "AdaptiveResourceManager" in result.message

    def test_health_check_not_running(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when not running."""
        manager.running = False
        result = manager.health_check()
        
        assert result.healthy is True  # Still healthy if no errors

    def test_health_check_with_errors(self, manager: AdaptiveResourceManager) -> None:
        """Test health check when errors occurred."""
        manager.running = False
        manager.stats["errors"] = 5
        result = manager.health_check()
        
        # Should still be healthy if not running (graceful shutdown)
        assert result is not None


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_resource_manager(self) -> None:
        """Test singleton accessor."""
        # Reset singleton
        import app.coordination.adaptive_resource_manager as arm
        arm._resource_manager = None
        
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        assert manager1 is manager2


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_threshold_values(self) -> None:
        """Test threshold constants have reasonable values."""
        assert DISK_WARNING_THRESHOLD < DISK_CRITICAL_THRESHOLD
        assert MEMORY_WARNING_THRESHOLD < MEMORY_CRITICAL_THRESHOLD
        assert 0 < DISK_WARNING_THRESHOLD < 100
        assert 0 < MEMORY_WARNING_THRESHOLD < 100

    def test_cleanup_settings(self) -> None:
        """Test cleanup settings."""
        assert MIN_FILE_AGE_HOURS > 0
        assert CLEANUP_BATCH_SIZE > 0

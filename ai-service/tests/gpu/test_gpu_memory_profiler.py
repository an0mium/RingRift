"""Tests for GPU memory profiling utilities."""

import pytest
import torch

from app.ai.gpu_memory_profiler import (
    MemoryProfiler,
    MemorySnapshot,
    get_tensor_memory_usage,
    profile_memory,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_snapshot_creation(self):
        """Snapshot can be created with required fields."""
        snap = MemorySnapshot(
            name="test",
            timestamp=1234567890.0,
            allocated_bytes=1024 * 1024,
            reserved_bytes=2 * 1024 * 1024,
            max_allocated_bytes=3 * 1024 * 1024,
            num_tensors=10,
        )

        assert snap.name == "test"
        assert snap.allocated_mb == 1.0
        assert snap.reserved_mb == 2.0
        assert snap.max_allocated_mb == 3.0
        assert snap.num_tensors == 10

    def test_snapshot_with_tensor_sizes(self):
        """Snapshot can include tensor sizes by dtype."""
        snap = MemorySnapshot(
            name="test",
            timestamp=0.0,
            allocated_bytes=0,
            reserved_bytes=0,
            max_allocated_bytes=0,
            num_tensors=2,
            tensor_sizes={"torch.float32": 1024, "torch.int64": 512},
        )

        assert len(snap.tensor_sizes) == 2
        assert snap.tensor_sizes["torch.float32"] == 1024


class TestMemoryProfiler:
    """Tests for MemoryProfiler class."""

    @pytest.fixture
    def profiler(self):
        """Create a profiler for CPU (always available)."""
        return MemoryProfiler(device=torch.device("cpu"))

    def test_profiler_creation(self, profiler):
        """Profiler can be created."""
        assert profiler.device == torch.device("cpu")
        assert len(profiler.snapshots) == 0

    def test_snapshot_capture(self, profiler):
        """Profiler can capture snapshots."""
        snap = profiler.snapshot("test")

        assert snap.name == "test"
        assert snap.timestamp > 0
        assert "test" in profiler.snapshots

    def test_multiple_snapshots(self, profiler):
        """Profiler can capture multiple snapshots."""
        profiler.snapshot("first")
        profiler.snapshot("second")
        profiler.snapshot("third")

        assert len(profiler.snapshots) == 3
        assert "first" in profiler.snapshots
        assert "second" in profiler.snapshots
        assert "third" in profiler.snapshots

    def test_compare_snapshots(self, profiler):
        """Profiler can compare two snapshots."""
        profiler.snapshot("before")
        # Create a tensor to change memory state
        _ = torch.zeros(1000, 1000)
        profiler.snapshot("after")

        comparison = profiler.compare("before", "after")

        assert "before" in comparison
        assert "after" in comparison
        assert "elapsed_seconds" in comparison
        assert "delta_allocated_mb" in comparison
        assert comparison["elapsed_seconds"] >= 0

    def test_compare_missing_snapshot(self, profiler):
        """Comparing with missing snapshot returns error."""
        profiler.snapshot("exists")
        result = profiler.compare("exists", "missing")

        assert "error" in result

    def test_report_empty(self, profiler):
        """Report works with no snapshots."""
        report = profiler.report()
        assert "No snapshots captured" in report

    def test_report_with_snapshots(self, profiler):
        """Report includes all snapshots."""
        profiler.snapshot("snap1")
        profiler.snapshot("snap2")

        report = profiler.report()

        assert "snap1" in report
        assert "snap2" in report
        assert "Allocated" in report


class TestProfileMemoryContextManager:
    """Tests for profile_memory context manager."""

    def test_context_manager_captures_snapshots(self):
        """Context manager captures before/after snapshots."""
        with profile_memory("test") as prof:
            _ = torch.zeros(100, 100)

        assert "before" in prof.snapshots
        assert "after" in prof.snapshots

    def test_context_manager_timing(self):
        """Context manager measures elapsed time."""
        import time

        with profile_memory("test") as prof:
            time.sleep(0.01)  # 10ms

        comparison = prof.compare("before", "after")
        assert comparison["elapsed_seconds"] >= 0.01


class TestGetTensorMemoryUsage:
    """Tests for get_tensor_memory_usage function."""

    def test_returns_valid_structure(self):
        """Function returns expected structure."""
        result = get_tensor_memory_usage(torch.device("cpu"))

        assert "device" in result
        assert "tensors_by_dtype" in result
        assert "tensors_by_size" in result
        assert "total_count" in result
        assert "total_bytes" in result
        assert "total_mb" in result

    def test_counts_tensors(self):
        """Function counts existing tensors."""
        # Create some tensors
        t1 = torch.zeros(100, 100)
        t2 = torch.ones(50, 50, dtype=torch.int64)

        result = get_tensor_memory_usage(torch.device("cpu"))

        # Should find at least these tensors
        assert result["total_count"] >= 2
        assert result["total_bytes"] > 0

        # Clean up
        del t1, t2

    def test_auto_detect_device(self):
        """Function auto-detects device when not specified."""
        result = get_tensor_memory_usage()
        assert "device" in result


class TestMemoryProfilingIntegration:
    """Integration tests for memory profiling with GPU operations."""

    def test_profile_batch_state_creation(self):
        """Profile memory during BatchGameState creation."""
        try:
            from app.ai.gpu_batch_state import BatchGameState
        except ImportError:
            pytest.skip("GPU modules not available")

        with profile_memory("batch_state") as prof:
            state = BatchGameState.create_batch(
                batch_size=100,
                board_size=8,
                num_players=2,
                device=torch.device("cpu"),
            )

        comparison = prof.compare("before", "after")

        # Should have allocated some memory
        assert comparison["after_allocated_mb"] >= 0
        # Should have created tensors
        assert prof.snapshots["after"].num_tensors >= prof.snapshots["before"].num_tensors

        del state

    def test_profile_game_simulation(self):
        """Profile memory during game simulation."""
        try:
            from app.ai.gpu_game_types import GameStatus
            from app.ai.gpu_parallel_games import ParallelGameRunner
        except ImportError:
            pytest.skip("GPU modules not available")

        with profile_memory("game_sim") as prof:
            runner = ParallelGameRunner(
                batch_size=10,
                board_size=8,
                num_players=2,
                device=torch.device("cpu"),
                shadow_validation=False,
                state_validation=False,
            )

            # Run a few steps
            weights = [runner._default_weights() for _ in range(10)]
            for _ in range(5):
                runner._step_games(weights)

        report = prof.report()
        assert "before" in report
        assert "after" in report

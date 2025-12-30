"""Unit tests for task_coordinator_reservations module.

Tests the ReservationManager class for gauntlet and training node reservations.
Verifies thread safety, expiry handling, and singleton behavior.

Created: December 30, 2025
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from app.coordination.task_coordinator_reservations import (
    ReservationManager,
    ReservationStats,
    get_reservation_manager,
    reset_reservation_manager,
)


class TestReservationStats:
    """Tests for ReservationStats dataclass."""

    def test_default_values(self) -> None:
        """Test default initialization."""
        stats = ReservationStats()
        assert stats.gauntlet_count == 0
        assert stats.training_count == 0
        assert stats.gauntlet_nodes == set()
        assert stats.training_nodes == set()

    def test_total_reserved_empty(self) -> None:
        """Test total_reserved with no reservations."""
        stats = ReservationStats()
        assert stats.total_reserved == 0

    def test_total_reserved_gauntlet_only(self) -> None:
        """Test total_reserved with only gauntlet reservations."""
        stats = ReservationStats(
            gauntlet_count=2,
            gauntlet_nodes={"node-1", "node-2"},
        )
        assert stats.total_reserved == 2

    def test_total_reserved_training_only(self) -> None:
        """Test total_reserved with only training reservations."""
        stats = ReservationStats(
            training_count=3,
            training_nodes={"gpu-1", "gpu-2", "gpu-3"},
        )
        assert stats.total_reserved == 3

    def test_total_reserved_overlapping_nodes(self) -> None:
        """Test total_reserved with overlapping nodes (union)."""
        stats = ReservationStats(
            gauntlet_count=2,
            training_count=2,
            gauntlet_nodes={"node-1", "node-2"},
            training_nodes={"node-2", "node-3"},  # node-2 overlaps
        )
        # Union: {node-1, node-2, node-3} = 3
        assert stats.total_reserved == 3

    def test_total_reserved_disjoint_nodes(self) -> None:
        """Test total_reserved with disjoint node sets."""
        stats = ReservationStats(
            gauntlet_count=2,
            training_count=2,
            gauntlet_nodes={"cpu-1", "cpu-2"},
            training_nodes={"gpu-1", "gpu-2"},
        )
        assert stats.total_reserved == 4


class TestReservationManagerGauntlet:
    """Tests for gauntlet reservation functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()
        self.manager = get_reservation_manager()

    def test_reserve_for_gauntlet_single_node(self) -> None:
        """Test reserving a single node for gauntlet."""
        reserved = self.manager.reserve_for_gauntlet(["node-1"])
        assert reserved == ["node-1"]
        assert self.manager.is_reserved_for_gauntlet("node-1")

    def test_reserve_for_gauntlet_multiple_nodes(self) -> None:
        """Test reserving multiple nodes for gauntlet."""
        reserved = self.manager.reserve_for_gauntlet(["node-1", "node-2", "node-3"])
        assert reserved == ["node-1", "node-2", "node-3"]
        assert self.manager.is_reserved_for_gauntlet("node-1")
        assert self.manager.is_reserved_for_gauntlet("node-2")
        assert self.manager.is_reserved_for_gauntlet("node-3")

    def test_reserve_for_gauntlet_duplicate_skipped(self) -> None:
        """Test that already-reserved nodes are skipped."""
        self.manager.reserve_for_gauntlet(["node-1"])
        reserved = self.manager.reserve_for_gauntlet(["node-1", "node-2"])
        # node-1 already reserved, so only node-2 is newly reserved
        assert reserved == ["node-2"]

    def test_reserve_for_gauntlet_empty_list(self) -> None:
        """Test reserving with empty list."""
        reserved = self.manager.reserve_for_gauntlet([])
        assert reserved == []

    def test_release_from_gauntlet(self) -> None:
        """Test releasing nodes from gauntlet."""
        self.manager.reserve_for_gauntlet(["node-1", "node-2"])
        self.manager.release_from_gauntlet(["node-1"])
        assert not self.manager.is_reserved_for_gauntlet("node-1")
        assert self.manager.is_reserved_for_gauntlet("node-2")

    def test_release_from_gauntlet_nonexistent(self) -> None:
        """Test releasing a node that wasn't reserved (no-op)."""
        self.manager.release_from_gauntlet(["node-1"])
        # Should not raise

    def test_release_all_gauntlet(self) -> None:
        """Test releasing all gauntlet reservations."""
        self.manager.reserve_for_gauntlet(["node-1", "node-2", "node-3"])
        count = self.manager.release_all_gauntlet()
        assert count == 3
        assert not self.manager.is_reserved_for_gauntlet("node-1")
        assert self.manager.get_gauntlet_reserved() == set()

    def test_release_all_gauntlet_empty(self) -> None:
        """Test releasing all when no reservations exist."""
        count = self.manager.release_all_gauntlet()
        assert count == 0

    def test_get_gauntlet_reserved(self) -> None:
        """Test getting set of reserved nodes."""
        self.manager.reserve_for_gauntlet(["node-1", "node-2"])
        reserved = self.manager.get_gauntlet_reserved()
        assert reserved == {"node-1", "node-2"}
        # Verify it's a copy
        reserved.add("node-3")
        assert self.manager.get_gauntlet_reserved() == {"node-1", "node-2"}

    def test_get_available_for_gauntlet_prefers_cpu(self) -> None:
        """Test that CPU nodes are preferred for gauntlet."""
        all_nodes = ["gpu-1", "cpu-1", "gpu-2", "cpu-2", "cpu-3"]
        available = self.manager.get_available_for_gauntlet(all_nodes, count=2)
        # Should prefer CPU nodes
        assert available == ["cpu-1", "cpu-2"]

    def test_get_available_for_gauntlet_excludes_reserved(self) -> None:
        """Test that already-reserved nodes are excluded."""
        all_nodes = ["cpu-1", "cpu-2", "cpu-3"]
        self.manager.reserve_for_gauntlet(["cpu-1"])
        available = self.manager.get_available_for_gauntlet(all_nodes, count=2)
        assert "cpu-1" not in available
        assert available == ["cpu-2", "cpu-3"]

    def test_get_available_for_gauntlet_fallback_to_gpu(self) -> None:
        """Test fallback to GPU nodes when not enough CPU nodes."""
        all_nodes = ["cpu-1", "gpu-1", "gpu-2"]
        available = self.manager.get_available_for_gauntlet(all_nodes, count=3)
        # Should take CPU first, then GPU
        assert available == ["cpu-1", "gpu-1", "gpu-2"]


class TestReservationManagerTraining:
    """Tests for training reservation functionality."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()
        self.manager = get_reservation_manager()

    def test_reserve_for_training_single_node(self) -> None:
        """Test reserving a single node for training."""
        reserved = self.manager.reserve_for_training(["gpu-1"])
        assert reserved == ["gpu-1"]
        assert self.manager.is_reserved_for_training("gpu-1")

    def test_reserve_for_training_with_duration(self) -> None:
        """Test reserving with custom duration."""
        reserved = self.manager.reserve_for_training(
            ["gpu-1"],
            duration_seconds=3600,
            config_key="hex8_2p",
        )
        assert reserved == ["gpu-1"]

    def test_reserve_for_training_duplicate_skipped(self) -> None:
        """Test that already-reserved nodes are skipped."""
        self.manager.reserve_for_training(["gpu-1"])
        reserved = self.manager.reserve_for_training(["gpu-1", "gpu-2"])
        assert reserved == ["gpu-2"]

    def test_release_from_training(self) -> None:
        """Test releasing nodes from training."""
        self.manager.reserve_for_training(["gpu-1", "gpu-2"])
        self.manager.release_from_training(["gpu-1"])
        assert not self.manager.is_reserved_for_training("gpu-1")
        assert self.manager.is_reserved_for_training("gpu-2")

    def test_release_all_training(self) -> None:
        """Test releasing all training reservations."""
        self.manager.reserve_for_training(["gpu-1", "gpu-2"])
        count = self.manager.release_all_training()
        assert count == 2
        assert self.manager.get_training_reserved() == set()

    def test_training_reservation_expiry(self) -> None:
        """Test that expired training reservations are cleaned up."""
        # Reserve with very short duration
        self.manager.reserve_for_training(["gpu-1"], duration_seconds=0.1)
        assert self.manager.is_reserved_for_training("gpu-1")

        # Wait for expiry
        time.sleep(0.15)

        # Should be expired now
        assert not self.manager.is_reserved_for_training("gpu-1")

    def test_get_training_reserved_cleans_expired(self) -> None:
        """Test that get_training_reserved cleans up expired reservations."""
        self.manager.reserve_for_training(["gpu-1"], duration_seconds=0.1)
        time.sleep(0.15)
        reserved = self.manager.get_training_reserved()
        assert reserved == set()

    def test_get_available_for_training_gpu_only(self) -> None:
        """Test GPU-only filtering for training availability."""
        all_nodes = ["cpu-1", "gpu-1", "h100-node", "a100-node", "cpu-2"]
        available = self.manager.get_available_for_training(
            all_nodes, gpu_nodes_only=True
        )
        # Should only include GPU indicators
        assert "cpu-1" not in available
        assert "cpu-2" not in available
        assert "gpu-1" in available
        assert "h100-node" in available
        assert "a100-node" in available

    def test_get_available_for_training_excludes_gauntlet(self) -> None:
        """Test exclusion of gauntlet-reserved nodes."""
        self.manager.reserve_for_gauntlet(["gpu-1"])
        all_nodes = ["gpu-1", "gpu-2"]
        available = self.manager.get_available_for_training(
            all_nodes, gpu_nodes_only=False, exclude_gauntlet=True
        )
        assert "gpu-1" not in available
        assert "gpu-2" in available

    def test_get_available_for_training_includes_gauntlet(self) -> None:
        """Test inclusion of gauntlet-reserved nodes when excluded=False."""
        self.manager.reserve_for_gauntlet(["gpu-1"])
        all_nodes = ["gpu-1", "gpu-2"]
        available = self.manager.get_available_for_training(
            all_nodes, gpu_nodes_only=False, exclude_gauntlet=False
        )
        assert "gpu-1" in available
        assert "gpu-2" in available

    def test_get_available_for_training_excludes_training_reserved(self) -> None:
        """Test exclusion of already training-reserved nodes."""
        self.manager.reserve_for_training(["gpu-1"])
        all_nodes = ["gpu-1", "gpu-2"]
        available = self.manager.get_available_for_training(
            all_nodes, gpu_nodes_only=False
        )
        assert "gpu-1" not in available
        assert "gpu-2" in available


class TestReservationManagerCombined:
    """Tests for combined reservation operations."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()
        self.manager = get_reservation_manager()

    def test_is_any_reserved_gauntlet(self) -> None:
        """Test is_any_reserved for gauntlet-only reservation."""
        self.manager.reserve_for_gauntlet(["node-1"])
        assert self.manager.is_any_reserved("node-1")
        assert not self.manager.is_any_reserved("node-2")

    def test_is_any_reserved_training(self) -> None:
        """Test is_any_reserved for training-only reservation."""
        self.manager.reserve_for_training(["gpu-1"])
        assert self.manager.is_any_reserved("gpu-1")
        assert not self.manager.is_any_reserved("gpu-2")

    def test_is_any_reserved_both(self) -> None:
        """Test is_any_reserved for both reservation types."""
        self.manager.reserve_for_gauntlet(["node-1"])
        self.manager.reserve_for_training(["gpu-1"])
        assert self.manager.is_any_reserved("node-1")
        assert self.manager.is_any_reserved("gpu-1")
        assert not self.manager.is_any_reserved("other")

    def test_get_stats(self) -> None:
        """Test getting reservation statistics."""
        self.manager.reserve_for_gauntlet(["cpu-1", "cpu-2"])
        self.manager.reserve_for_training(["gpu-1", "gpu-2", "gpu-3"])
        stats = self.manager.get_stats()
        assert stats.gauntlet_count == 2
        assert stats.training_count == 3
        assert stats.gauntlet_nodes == {"cpu-1", "cpu-2"}
        assert stats.training_nodes == {"gpu-1", "gpu-2", "gpu-3"}
        assert stats.total_reserved == 5

    def test_get_stats_with_expired_training(self) -> None:
        """Test that stats excludes expired training reservations."""
        self.manager.reserve_for_training(["gpu-1"], duration_seconds=0.1)
        time.sleep(0.15)
        stats = self.manager.get_stats()
        assert stats.training_count == 0
        assert "gpu-1" not in stats.training_nodes

    def test_release_all(self) -> None:
        """Test releasing all reservations of both types."""
        self.manager.reserve_for_gauntlet(["cpu-1", "cpu-2"])
        self.manager.reserve_for_training(["gpu-1"])
        gauntlet_count, training_count = self.manager.release_all()
        assert gauntlet_count == 2
        assert training_count == 1
        assert self.manager.get_gauntlet_reserved() == set()
        assert self.manager.get_training_reserved() == set()


class TestSingletonBehavior:
    """Tests for singleton pattern."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()

    def test_get_reservation_manager_same_instance(self) -> None:
        """Test that get_reservation_manager returns the same instance."""
        manager1 = get_reservation_manager()
        manager2 = get_reservation_manager()
        assert manager1 is manager2

    def test_reset_reservation_manager(self) -> None:
        """Test that reset creates a new instance."""
        manager1 = get_reservation_manager()
        manager1.reserve_for_gauntlet(["node-1"])
        reset_reservation_manager()
        manager2 = get_reservation_manager()
        # New instance should not have reservations
        assert not manager2.is_reserved_for_gauntlet("node-1")

    def test_reset_releases_existing_reservations(self) -> None:
        """Test that reset releases existing reservations."""
        manager = get_reservation_manager()
        manager.reserve_for_gauntlet(["node-1"])
        manager.reserve_for_training(["gpu-1"])
        reset_reservation_manager()
        # After reset, all reservations should be gone
        new_manager = get_reservation_manager()
        assert new_manager.get_gauntlet_reserved() == set()
        assert new_manager.get_training_reserved() == set()


class TestThreadSafety:
    """Tests for thread safety of ReservationManager."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()
        self.manager = get_reservation_manager()

    def test_concurrent_gauntlet_reservations(self) -> None:
        """Test concurrent gauntlet reservation is thread-safe."""
        all_nodes = [f"node-{i}" for i in range(100)]

        def reserve_batch(start: int, count: int) -> list[str]:
            nodes = all_nodes[start : start + count]
            return self.manager.reserve_for_gauntlet(nodes)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(reserve_batch, i * 10, 10) for i in range(10)
            ]
            results = [f.result() for f in futures]

        # All nodes should be reserved (no duplicates due to locking)
        total_reserved = sum(len(r) for r in results)
        assert total_reserved == 100
        assert len(self.manager.get_gauntlet_reserved()) == 100

    def test_concurrent_training_reservations(self) -> None:
        """Test concurrent training reservation is thread-safe."""
        all_nodes = [f"gpu-{i}" for i in range(100)]

        def reserve_batch(start: int, count: int) -> list[str]:
            nodes = all_nodes[start : start + count]
            return self.manager.reserve_for_training(nodes)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(reserve_batch, i * 10, 10) for i in range(10)
            ]
            results = [f.result() for f in futures]

        total_reserved = sum(len(r) for r in results)
        assert total_reserved == 100
        assert len(self.manager.get_training_reserved()) == 100

    def test_concurrent_reserve_and_release(self) -> None:
        """Test concurrent reservation and release is thread-safe."""
        errors: list[Exception] = []
        iterations = 100

        def reserve_and_release(thread_id: int) -> None:
            try:
                for i in range(iterations):
                    node_id = f"node-{thread_id}-{i}"
                    self.manager.reserve_for_gauntlet([node_id])
                    self.manager.release_from_gauntlet([node_id])
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(reserve_and_release, range(5)))

        assert len(errors) == 0
        # All nodes should be released
        assert len(self.manager.get_gauntlet_reserved()) == 0

    def test_concurrent_is_any_reserved(self) -> None:
        """Test concurrent is_any_reserved checks are thread-safe."""
        # Reserve some nodes
        self.manager.reserve_for_gauntlet(["node-1", "node-2"])
        self.manager.reserve_for_training(["gpu-1", "gpu-2"])

        errors: list[Exception] = []
        results: list[bool] = []

        def check_reservation(node_id: str) -> None:
            try:
                for _ in range(100):
                    result = self.manager.is_any_reserved(node_id)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(check_reservation, ["node-1", "gpu-1", "node-2", "other"]))

        assert len(errors) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Reset singleton before each test."""
        reset_reservation_manager()
        self.manager = get_reservation_manager()

    def test_reserve_same_node_gauntlet_and_training(self) -> None:
        """Test reserving the same node for both purposes."""
        self.manager.reserve_for_gauntlet(["node-1"])
        # Same node can also be reserved for training (different purpose)
        reserved = self.manager.reserve_for_training(["node-1"])
        assert reserved == ["node-1"]
        assert self.manager.is_reserved_for_gauntlet("node-1")
        assert self.manager.is_reserved_for_training("node-1")

    def test_release_gauntlet_does_not_affect_training(self) -> None:
        """Test that releasing gauntlet doesn't affect training reservation."""
        self.manager.reserve_for_gauntlet(["node-1"])
        self.manager.reserve_for_training(["node-1"])
        self.manager.release_from_gauntlet(["node-1"])
        assert not self.manager.is_reserved_for_gauntlet("node-1")
        assert self.manager.is_reserved_for_training("node-1")

    def test_release_training_does_not_affect_gauntlet(self) -> None:
        """Test that releasing training doesn't affect gauntlet reservation."""
        self.manager.reserve_for_gauntlet(["node-1"])
        self.manager.reserve_for_training(["node-1"])
        self.manager.release_from_training(["node-1"])
        assert self.manager.is_reserved_for_gauntlet("node-1")
        assert not self.manager.is_reserved_for_training("node-1")

    def test_empty_node_list_handling(self) -> None:
        """Test handling of empty node lists."""
        assert self.manager.reserve_for_gauntlet([]) == []
        assert self.manager.reserve_for_training([]) == []
        self.manager.release_from_gauntlet([])
        self.manager.release_from_training([])
        # Should not raise

    def test_get_available_with_all_reserved(self) -> None:
        """Test availability when all nodes are reserved."""
        all_nodes = ["node-1", "node-2"]
        self.manager.reserve_for_gauntlet(all_nodes)
        available = self.manager.get_available_for_gauntlet(all_nodes, count=2)
        assert available == []

    def test_gpu_indicators_case_insensitive(self) -> None:
        """Test GPU node detection is case-insensitive."""
        all_nodes = ["GPU-1", "H100-Node", "A100-FAST", "cuda-dev"]
        available = self.manager.get_available_for_training(
            all_nodes, gpu_nodes_only=True
        )
        assert len(available) == 4

    def test_very_short_training_expiry(self) -> None:
        """Test very short training expiry (edge case)."""
        self.manager.reserve_for_training(["gpu-1"], duration_seconds=0.001)
        time.sleep(0.01)
        assert not self.manager.is_reserved_for_training("gpu-1")

    def test_long_training_expiry(self) -> None:
        """Test long training expiry doesn't immediately expire."""
        self.manager.reserve_for_training(["gpu-1"], duration_seconds=86400)  # 24h
        assert self.manager.is_reserved_for_training("gpu-1")

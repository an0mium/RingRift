"""
Task Coordinator Reservation Manager.

Extracted from TaskCoordinator (December 30, 2025) as part of god class refactoring.
Handles worker/node reservation for both gauntlet evaluation and training jobs.

This module provides a single-responsibility class for managing node reservations,
which were previously interleaved with admission control and task lifecycle code.

Usage:
    from app.coordination.task_coordinator_reservations import (
        ReservationManager,
        get_reservation_manager,
    )

    manager = get_reservation_manager()

    # Reserve nodes for gauntlet evaluation
    reserved = manager.reserve_for_gauntlet(["node-1", "node-2"])

    # Reserve GPU nodes for training
    reserved = manager.reserve_for_training(["gpu-node-1"], duration_seconds=7200)

    # Check if a node is reserved
    if manager.is_any_reserved(node_id):
        print("Node is reserved, skip for selfplay")
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from app.coordination.singleton_mixin import SingletonMixin

logger = logging.getLogger(__name__)


@dataclass
class ReservationStats:
    """Statistics about current reservations."""

    gauntlet_count: int = 0
    training_count: int = 0
    gauntlet_nodes: set[str] = field(default_factory=set)
    training_nodes: set[str] = field(default_factory=set)

    @property
    def total_reserved(self) -> int:
        """Total unique nodes reserved."""
        return len(self.gauntlet_nodes | self.training_nodes)


class ReservationManager(SingletonMixin):
    """Manager for node reservations (gauntlet and training).

    Thread-safe manager for reserving nodes for specific purposes,
    preventing them from being used for general selfplay tasks.

    Features:
    - Gauntlet reservations: Simple reserve/release for evaluation workers
    - Training reservations: Time-limited with automatic expiry
    - CPU-node preference for gauntlet (saves GPU for selfplay)
    - GPU-node preference for training

    Thread Safety:
    - Uses separate locks for gauntlet and training reservations
    - All public methods are thread-safe
    """

    def __init__(self) -> None:
        """Initialize the reservation manager."""
        # Gauntlet reservations (simple set)
        self._gauntlet_reserved: set[str] = set()
        self._gauntlet_lock = threading.RLock()

        # Training reservations (with expiry)
        self._training_reserved: set[str] = set()
        self._training_reservation_expiry: dict[str, float] = {}
        self._training_lock = threading.RLock()

        logger.debug("[ReservationManager] Initialized")

    # ==========================================
    # Gauntlet Worker Reservation
    # ==========================================

    def reserve_for_gauntlet(self, node_ids: list[str]) -> list[str]:
        """Reserve workers for gauntlet evaluation.

        Reserved workers are excluded from selfplay task assignment
        until released.

        Args:
            node_ids: List of node IDs to reserve

        Returns:
            List of successfully reserved node IDs
        """
        reserved = []
        with self._gauntlet_lock:
            for node_id in node_ids:
                if node_id not in self._gauntlet_reserved:
                    self._gauntlet_reserved.add(node_id)
                    reserved.append(node_id)
                    logger.info(f"[Gauntlet] Reserved worker: {node_id}")
        return reserved

    def release_from_gauntlet(self, node_ids: list[str]) -> None:
        """Release workers from gauntlet reservation.

        Args:
            node_ids: List of node IDs to release
        """
        with self._gauntlet_lock:
            for node_id in node_ids:
                if node_id in self._gauntlet_reserved:
                    self._gauntlet_reserved.discard(node_id)
                    logger.info(f"[Gauntlet] Released worker: {node_id}")

    def release_all_gauntlet(self) -> int:
        """Release all workers from gauntlet reservation.

        Returns:
            Number of workers released
        """
        with self._gauntlet_lock:
            count = len(self._gauntlet_reserved)
            self._gauntlet_reserved.clear()
            if count > 0:
                logger.info(f"[Gauntlet] Released all {count} workers")
            return count

    def is_reserved_for_gauntlet(self, node_id: str) -> bool:
        """Check if a worker is reserved for gauntlet.

        Args:
            node_id: Node ID to check

        Returns:
            True if reserved
        """
        with self._gauntlet_lock:
            return node_id in self._gauntlet_reserved

    def get_gauntlet_reserved(self) -> set[str]:
        """Get set of all workers reserved for gauntlet."""
        with self._gauntlet_lock:
            return self._gauntlet_reserved.copy()

    def get_available_for_gauntlet(self, all_nodes: list[str], count: int = 2) -> list[str]:
        """Get available nodes that can be reserved for gauntlet.

        Prefers CPU-only nodes over GPU nodes for gauntlet evaluation.

        Args:
            all_nodes: List of all known node IDs
            count: Number of nodes to reserve

        Returns:
            List of node IDs to reserve
        """
        with self._gauntlet_lock:
            # Filter out already reserved nodes
            available = [n for n in all_nodes if n not in self._gauntlet_reserved]

            # Prefer nodes with "cpu" in their ID
            cpu_nodes = [n for n in available if "cpu" in n.lower()]
            other_nodes = [n for n in available if "cpu" not in n.lower()]

            # Take from CPU nodes first, then others
            candidates = cpu_nodes + other_nodes
            return candidates[:count]

    # ==========================================
    # Training Reservation (December 2025)
    # ==========================================

    def reserve_for_training(
        self,
        node_ids: list[str],
        duration_seconds: float = 7200.0,  # Default 2 hours
        config_key: str = "",
    ) -> list[str]:
        """Reserve GPU nodes for training jobs.

        Reserved nodes are excluded from selfplay task assignment
        until released or the reservation expires. This ensures training
        jobs get priority access to GPU resources.

        December 2025: Implements Phase 4 of the Training Loop Improvement Plan
        to ensure training jobs get dedicated GPU access.

        Args:
            node_ids: List of node IDs to reserve
            duration_seconds: How long to reserve (default 2 hours)
            config_key: Optional config key for logging

        Returns:
            List of successfully reserved node IDs
        """
        reserved = []
        expiry = time.time() + duration_seconds
        with self._training_lock:
            for node_id in node_ids:
                if node_id not in self._training_reserved:
                    self._training_reserved.add(node_id)
                    self._training_reservation_expiry[node_id] = expiry
                    reserved.append(node_id)
                    logger.info(
                        f"[Training] Reserved node {node_id} for training "
                        f"(config={config_key or 'any'}, expires in {duration_seconds/60:.0f}min)"
                    )
        return reserved

    def release_from_training(self, node_ids: list[str]) -> None:
        """Release nodes from training reservation.

        Args:
            node_ids: List of node IDs to release
        """
        with self._training_lock:
            for node_id in node_ids:
                if node_id in self._training_reserved:
                    self._training_reserved.discard(node_id)
                    self._training_reservation_expiry.pop(node_id, None)
                    logger.info(f"[Training] Released node: {node_id}")

    def release_all_training(self) -> int:
        """Release all nodes from training reservation.

        Returns:
            Number of nodes released
        """
        with self._training_lock:
            count = len(self._training_reserved)
            self._training_reserved.clear()
            self._training_reservation_expiry.clear()
            if count > 0:
                logger.info(f"[Training] Released all {count} reserved nodes")
            return count

    def is_reserved_for_training(self, node_id: str) -> bool:
        """Check if a node is reserved for training.

        Also cleans up expired reservations.

        Args:
            node_id: Node ID to check

        Returns:
            True if reserved and not expired
        """
        with self._training_lock:
            self._cleanup_expired_training_reservations()
            return node_id in self._training_reserved

    def get_training_reserved(self) -> set[str]:
        """Get set of all nodes reserved for training."""
        with self._training_lock:
            self._cleanup_expired_training_reservations()
            return self._training_reserved.copy()

    def _cleanup_expired_training_reservations(self) -> None:
        """Remove expired training reservations (internal helper).

        Called internally when checking reservations.
        Must be called with _training_lock held.
        """
        now = time.time()
        expired = [
            node_id for node_id, expiry in self._training_reservation_expiry.items()
            if expiry < now
        ]
        for node_id in expired:
            self._training_reserved.discard(node_id)
            self._training_reservation_expiry.pop(node_id, None)
            logger.debug(f"[Training] Reservation expired for node: {node_id}")

    def get_available_for_training(
        self,
        all_nodes: list[str],
        gpu_nodes_only: bool = True,
        exclude_gauntlet: bool = True,
    ) -> list[str]:
        """Get available nodes that can be reserved for training.

        Prefers GPU nodes and excludes already reserved nodes.

        Args:
            all_nodes: List of all known node IDs
            gpu_nodes_only: Whether to filter for GPU nodes only
            exclude_gauntlet: Whether to exclude gauntlet-reserved nodes

        Returns:
            List of available node IDs for training
        """
        with self._training_lock:
            self._cleanup_expired_training_reservations()
            available = [n for n in all_nodes if n not in self._training_reserved]

        if exclude_gauntlet:
            with self._gauntlet_lock:
                available = [n for n in available if n not in self._gauntlet_reserved]

        if gpu_nodes_only:
            # Filter for nodes with GPU indicators in their ID
            gpu_indicators = ["gpu", "cuda", "h100", "a100", "l40", "4090", "3090", "gh200"]
            available = [
                n for n in available
                if any(ind in n.lower() for ind in gpu_indicators)
            ]

        return available

    # ==========================================
    # Combined Operations
    # ==========================================

    def is_any_reserved(self, node_id: str) -> bool:
        """Check if a node is reserved for any purpose (training or gauntlet).

        Args:
            node_id: Node ID to check

        Returns:
            True if reserved for either purpose
        """
        return self.is_reserved_for_gauntlet(node_id) or self.is_reserved_for_training(node_id)

    def get_stats(self) -> ReservationStats:
        """Get current reservation statistics.

        Returns:
            ReservationStats with current counts and node sets
        """
        with self._gauntlet_lock:
            gauntlet_nodes = self._gauntlet_reserved.copy()

        with self._training_lock:
            self._cleanup_expired_training_reservations()
            training_nodes = self._training_reserved.copy()

        return ReservationStats(
            gauntlet_count=len(gauntlet_nodes),
            training_count=len(training_nodes),
            gauntlet_nodes=gauntlet_nodes,
            training_nodes=training_nodes,
        )

    def release_all(self) -> tuple[int, int]:
        """Release all reservations (both gauntlet and training).

        Returns:
            Tuple of (gauntlet_released, training_released)
        """
        gauntlet = self.release_all_gauntlet()
        training = self.release_all_training()
        return gauntlet, training


# Module-level singleton accessor
_reservation_manager: Optional[ReservationManager] = None
_manager_lock = threading.Lock()


def get_reservation_manager() -> ReservationManager:
    """Get the singleton ReservationManager instance.

    Returns:
        The global ReservationManager instance
    """
    global _reservation_manager
    if _reservation_manager is None:
        with _manager_lock:
            if _reservation_manager is None:
                _reservation_manager = ReservationManager()
    return _reservation_manager


def reset_reservation_manager() -> None:
    """Reset the singleton instance (for testing).

    This clears all reservations and creates a fresh instance.
    """
    global _reservation_manager
    with _manager_lock:
        if _reservation_manager is not None:
            _reservation_manager.release_all()
        _reservation_manager = None

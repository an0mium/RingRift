"""Thread-safe peers snapshot helper for copy-on-write pattern.

Jan 12, 2026: Created as part of Phase 2 lock contention fixes.

This module provides the PeersSnapshot class for safely iterating over
peers without holding the peers_lock for extended periods. The copy-on-write
pattern prevents lock contention in high-traffic scenarios.

Usage:
    from scripts.p2p.peers_snapshot import PeersSnapshot

    # In P2P orchestrator methods:
    snapshot = PeersSnapshot.create(self.peers, self.peers_lock)
    for node_id, peer in snapshot.items():
        # Safe iteration - lock already released
        process_peer(peer)

    # For alive peers only:
    for node_id, peer in snapshot.alive_peers():
        send_message(peer)

    # With custom timeout:
    snapshot = PeersSnapshot.create(self.peers, self.peers_lock, timeout=2.0)
    if snapshot.is_empty:
        logger.warning("Failed to acquire peers lock or no peers")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Default lock acquisition timeout (seconds)
DEFAULT_SNAPSHOT_TIMEOUT = 1.0

# How long a peer can be without heartbeat before considered dead
PEER_ALIVE_THRESHOLD_SECONDS = 120.0


@dataclass
class PeersSnapshot:
    """Immutable snapshot of peers dict for thread-safe iteration.

    This class implements the copy-on-write pattern for the peers dictionary.
    It takes a quick snapshot under lock, then releases the lock immediately,
    allowing iteration to proceed without blocking other operations.

    Attributes:
        peers: Dictionary mapping node_id to peer info (NodeInfo or dict)
        snapshot_time: When this snapshot was taken
        lock_acquired: Whether the lock was successfully acquired
    """

    peers: dict[str, Any] = field(default_factory=dict)
    snapshot_time: float = 0.0
    lock_acquired: bool = True

    @classmethod
    def create(
        cls,
        peers: dict[str, Any],
        peers_lock: threading.RLock,
        timeout: float = DEFAULT_SNAPSHOT_TIMEOUT,
    ) -> "PeersSnapshot":
        """Create snapshot with timeout-based lock acquisition.

        Takes a quick snapshot of the peers dictionary while holding the lock,
        then immediately releases it. The snapshot is immutable and safe to
        iterate over without blocking other threads.

        Args:
            peers: The peers dictionary to snapshot
            peers_lock: The RLock protecting the peers dictionary
            timeout: Maximum time to wait for lock (seconds). Default 1.0s.

        Returns:
            PeersSnapshot with copy of peers dict, or empty snapshot on timeout.

        Example:
            snapshot = PeersSnapshot.create(self.peers, self.peers_lock)
            for peer_id, peer in snapshot.items():
                # Process peer safely - lock is released
                await send_heartbeat(peer)
        """
        acquired = peers_lock.acquire(blocking=True, timeout=timeout)
        if acquired:
            try:
                # Quick copy under lock - typically <100 microseconds for 50 peers
                return cls(
                    peers=dict(peers),
                    snapshot_time=time.time(),
                    lock_acquired=True,
                )
            finally:
                peers_lock.release()
        else:
            # Lock acquisition timed out - return empty snapshot
            logger.warning(
                f"PeersSnapshot: lock acquisition timed out after {timeout}s"
            )
            return cls(
                peers={},
                snapshot_time=time.time(),
                lock_acquired=False,
            )

    @classmethod
    def create_nonblocking(
        cls,
        peers: dict[str, Any],
        peers_lock: threading.RLock,
    ) -> "PeersSnapshot":
        """Create snapshot without blocking if lock is unavailable.

        Useful for best-effort operations that can proceed with stale data
        or skip processing if the lock is contended.

        Args:
            peers: The peers dictionary to snapshot
            peers_lock: The RLock protecting the peers dictionary

        Returns:
            PeersSnapshot with copy of peers, or empty snapshot if lock busy.
        """
        acquired = peers_lock.acquire(blocking=False)
        if acquired:
            try:
                return cls(
                    peers=dict(peers),
                    snapshot_time=time.time(),
                    lock_acquired=True,
                )
            finally:
                peers_lock.release()
        else:
            return cls(
                peers={},
                snapshot_time=time.time(),
                lock_acquired=False,
            )

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over all peers (node_id, peer_info) pairs."""
        return iter(self.peers.items())

    def values(self) -> Iterator[Any]:
        """Iterate over all peer info objects."""
        return iter(self.peers.values())

    def keys(self) -> Iterator[str]:
        """Iterate over all node IDs."""
        return iter(self.peers.keys())

    def get(self, node_id: str, default: Any = None) -> Any:
        """Get a peer by node_id."""
        return self.peers.get(node_id, default)

    def __contains__(self, node_id: str) -> bool:
        """Check if node_id is in snapshot."""
        return node_id in self.peers

    def __len__(self) -> int:
        """Return number of peers in snapshot."""
        return len(self.peers)

    def __iter__(self) -> Iterator[str]:
        """Iterate over node IDs."""
        return iter(self.peers)

    @property
    def is_empty(self) -> bool:
        """Return True if snapshot has no peers or lock failed."""
        return len(self.peers) == 0

    @property
    def age_seconds(self) -> float:
        """Return how old this snapshot is in seconds."""
        return time.time() - self.snapshot_time

    def alive_peers(
        self,
        threshold_seconds: float = PEER_ALIVE_THRESHOLD_SECONDS,
    ) -> Iterator[tuple[str, Any]]:
        """Iterate over peers that are considered alive.

        A peer is alive if:
        1. It has an is_alive() method that returns True, OR
        2. Its last_heartbeat is within threshold_seconds of now

        Args:
            threshold_seconds: Max seconds since last heartbeat. Default 120s.

        Yields:
            Tuples of (node_id, peer_info) for alive peers.
        """
        now = time.time()
        for node_id, peer in self.peers.items():
            # Try is_alive() method first (NodeInfo objects)
            if hasattr(peer, "is_alive") and callable(peer.is_alive):
                if peer.is_alive():
                    yield node_id, peer
            # Fall back to checking last_heartbeat attribute
            elif hasattr(peer, "last_heartbeat"):
                if (now - peer.last_heartbeat) < threshold_seconds:
                    yield node_id, peer
            # If peer is a dict, check for last_heartbeat key
            elif isinstance(peer, dict) and "last_heartbeat" in peer:
                if (now - peer["last_heartbeat"]) < threshold_seconds:
                    yield node_id, peer
            else:
                # Can't determine aliveness - include by default
                yield node_id, peer

    def dead_peers(
        self,
        threshold_seconds: float = PEER_ALIVE_THRESHOLD_SECONDS,
    ) -> Iterator[tuple[str, Any]]:
        """Iterate over peers that are considered dead/unreachable.

        Inverse of alive_peers() - yields peers that haven't been seen
        within threshold_seconds.

        Args:
            threshold_seconds: Max seconds since last heartbeat. Default 120s.

        Yields:
            Tuples of (node_id, peer_info) for dead peers.
        """
        now = time.time()
        alive_set = set(node_id for node_id, _ in self.alive_peers(threshold_seconds))
        for node_id, peer in self.peers.items():
            if node_id not in alive_set:
                yield node_id, peer

    def filter_by_role(self, role: str) -> Iterator[tuple[str, Any]]:
        """Filter peers by role (e.g., 'leader', 'follower', 'voter').

        Args:
            role: Role to filter by (case-insensitive)

        Yields:
            Tuples of (node_id, peer_info) matching the role.
        """
        role_lower = role.lower()
        for node_id, peer in self.peers.items():
            peer_role = None
            if hasattr(peer, "role"):
                peer_role = peer.role
                if hasattr(peer_role, "value"):
                    peer_role = peer_role.value
            elif isinstance(peer, dict):
                peer_role = peer.get("role")

            if peer_role and str(peer_role).lower() == role_lower:
                yield node_id, peer

    def count_alive(
        self,
        threshold_seconds: float = PEER_ALIVE_THRESHOLD_SECONDS,
    ) -> int:
        """Count number of alive peers."""
        return sum(1 for _ in self.alive_peers(threshold_seconds))

    def to_dict(self) -> dict[str, Any]:
        """Return the peers dictionary (for serialization)."""
        return dict(self.peers)

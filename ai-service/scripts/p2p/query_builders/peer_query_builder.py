"""Peer query builder for P2P orchestrator.

Phase 3.2 Code Quality Cleanup: Peer-specific query methods.

This consolidates the repetitive peer iteration patterns found
in methods like _get_alive_peers_for_broadcast, _get_healthy_node_ids,
_get_peer_health_summary, etc.

Example usage:
    # In P2POrchestrator.__init__:
    self._peer_query = PeerQueryBuilder(self.peers, self.peers_lock, self.node_id)

    # Replace _get_alive_peers_for_broadcast:
    def _get_alive_peers_for_broadcast(self):
        return self._peer_query.alive_non_retired().unwrap_or([])
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from scripts.p2p.query_builders.base_query_builder import (
    BaseQueryBuilder,
    QueryResult,
    SummaryResult,
)

if TYPE_CHECKING:
    # Avoid circular imports - NodeInfo is defined in p2p_orchestrator
    NodeInfo = Any

logger = logging.getLogger(__name__)


class PeerQueryBuilder(BaseQueryBuilder):
    """Query builder for peer collections.

    Provides peer-specific query methods with consistent error handling
    and thread-safe access to the peers dictionary.

    Attributes:
        _self_node_id: The current node's ID (to exclude from queries).
    """

    def __init__(
        self,
        peers: Dict[str, "NodeInfo"],
        lock: threading.RLock,
        self_node_id: str,
    ):
        """Initialize peer query builder.

        Args:
            peers: The peers dictionary from P2POrchestrator.
            lock: The peers_lock from P2POrchestrator.
            self_node_id: The current node's ID.
        """
        super().__init__()
        self._items = peers
        self._lock = lock
        self._self_node_id = self_node_id

    def _is_alive(self, peer: "NodeInfo") -> bool:
        """Check if peer is alive, handling method or property."""
        is_alive = getattr(peer, "is_alive", lambda: False)
        if callable(is_alive):
            return is_alive()
        return bool(is_alive)

    def _is_retired(self, peer: "NodeInfo") -> bool:
        """Check if peer is retired."""
        return bool(getattr(peer, "retired", False))

    def _is_self(self, peer: "NodeInfo") -> bool:
        """Check if peer is this node."""
        return getattr(peer, "node_id", None) == self._self_node_id

    def _has_gpu(self, peer: "NodeInfo") -> bool:
        """Check if peer has GPU."""
        return bool(getattr(peer, "gpu_type", None))

    def _is_healthy(self, peer: "NodeInfo") -> bool:
        """Check if peer is healthy (method or score-based).

        Tries is_healthy() method first, then falls back to is_alive() + health_score.
        """
        is_healthy = getattr(peer, "is_healthy", None)
        if callable(is_healthy):
            return is_healthy()
        # Fallback: alive + health score >= 0.5
        return self._is_alive(peer) and getattr(peer, "health_score", 1.0) >= 0.5

    def _is_training_enabled(self, peer: "NodeInfo") -> bool:
        """Check if peer has training enabled."""
        return bool(getattr(peer, "training_enabled", False))

    # =========================================================================
    # Simple filter methods - replacements for _get_* methods
    # =========================================================================

    def alive(self, *, exclude_self: bool = True) -> QueryResult[List["NodeInfo"]]:
        """Get all alive peers.

        Replaces patterns like:
            with self.peers_lock:
                return [p for p in self.peers.values() if p.is_alive()]

        Args:
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of alive NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer)

        return self.filter(predicate)

    def alive_non_retired(
        self, *, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get alive, non-retired peers.

        Replaces: _get_alive_peers_for_broadcast()

        Args:
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of alive, non-retired NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer) and not self._is_retired(peer)

        return self.filter(predicate)

    def alive_with_gpu(
        self, *, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get alive peers with GPU capability.

        Replaces patterns like:
            [p for p in self.peers.values() if p.is_alive() and p.gpu_type]

        Args:
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of alive GPU-capable NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer) and self._has_gpu(peer)

        return self.filter(predicate)

    def by_role(
        self, role: str, *, alive_only: bool = True, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get peers with specific role.

        Args:
            role: Role to filter by (e.g., "leader", "voter").
            alive_only: If True, only include alive peers.
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of matching NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            if alive_only and not self._is_alive(peer):
                return False
            return getattr(peer, "role", None) == role

        return self.filter(predicate)

    def healthy(
        self, *, threshold: float = 0.5, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get healthy peers based on health score.

        Args:
            threshold: Minimum health score (0.0-1.0) to be considered healthy.
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of healthy NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            if not self._is_alive(peer):
                return False
            score = getattr(peer, "health_score", 1.0)
            return score >= threshold

        return self.filter(predicate)

    # =========================================================================
    # ID extraction methods
    # =========================================================================

    def alive_ids(self, *, exclude_self: bool = True) -> QueryResult[List[str]]:
        """Get IDs of all alive peers.

        Replaces patterns like:
            [p.node_id for p in self.peers.values() if p.is_alive()]

        Args:
            exclude_self: If True, exclude this node's ID.

        Returns:
            QueryResult containing list of node IDs.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer)

        return self.map(
            mapper=lambda p: getattr(p, "node_id", ""),
            predicate=predicate,
        )

    def healthy_ids(
        self, *, threshold: float = 0.5, exclude_self: bool = True
    ) -> QueryResult[List[str]]:
        """Get IDs of healthy peers.

        Replaces: _get_healthy_node_ids_for_reassignment()

        Args:
            threshold: Minimum health score to be considered healthy.
            exclude_self: If True, exclude this node's ID.

        Returns:
            QueryResult containing list of healthy node IDs.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            if not self._is_alive(peer):
                return False
            score = getattr(peer, "health_score", 1.0)
            return score >= threshold

        return self.map(
            mapper=lambda p: getattr(p, "node_id", ""),
            predicate=predicate,
        )

    # =========================================================================
    # Count methods
    # =========================================================================

    def alive_count(self, *, exclude_self: bool = True) -> QueryResult[int]:
        """Count alive peers.

        Replaces patterns like:
            sum(1 for p in self.peers.values() if p.is_alive())

        Args:
            exclude_self: If True, don't count this node.

        Returns:
            QueryResult containing count.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer)

        return self.count(predicate)

    def gpu_count(self, *, alive_only: bool = True) -> QueryResult[int]:
        """Count peers with GPU capability.

        Args:
            alive_only: If True, only count alive GPU peers.

        Returns:
            QueryResult containing count.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if alive_only and not self._is_alive(peer):
                return False
            return self._has_gpu(peer)

        return self.count(predicate)

    # =========================================================================
    # Summary methods
    # =========================================================================

    def status_summary(self) -> SummaryResult:
        """Get peer status summary.

        Replaces common summary patterns that count alive/dead/GPU peers.

        Returns:
            SummaryResult with counts and per-peer details.
        """
        return self.summarize(
            count_fn=lambda p: {
                "total": True,
                "alive": self._is_alive(p),
                "retired": self._is_retired(p),
                "gpu": self._has_gpu(p),
                "alive_with_gpu": self._is_alive(p) and self._has_gpu(p),
            },
            detail_fn=lambda p: {
                "alive": self._is_alive(p),
                "retired": self._is_retired(p),
                "gpu_type": getattr(p, "gpu_type", None),
                "role": getattr(p, "role", None),
            },
            key_fn=lambda p: getattr(p, "node_id", "unknown"),
        )

    def health_summary(self) -> SummaryResult:
        """Get peer health summary.

        Provides aggregated health metrics for monitoring.

        Returns:
            SummaryResult with health counts and scores.
        """
        return self.summarize(
            count_fn=lambda p: {
                "total": True,
                "healthy": self._is_alive(p) and getattr(p, "health_score", 1.0) >= 0.5,
                "degraded": self._is_alive(p) and 0.25 <= getattr(p, "health_score", 1.0) < 0.5,
                "unhealthy": self._is_alive(p) and getattr(p, "health_score", 1.0) < 0.25,
                "dead": not self._is_alive(p),
            },
            detail_fn=lambda p: {
                "health_score": getattr(p, "health_score", 1.0),
                "last_heartbeat": getattr(p, "last_heartbeat", 0.0),
                "consecutive_failures": getattr(p, "consecutive_failures", 0),
            },
            key_fn=lambda p: getattr(p, "node_id", "unknown"),
        )

    # =========================================================================
    # Single-peer lookups
    # =========================================================================

    def get_peer(self, node_id: str) -> QueryResult[Optional["NodeInfo"]]:
        """Get a specific peer by ID.

        Replaces: self.peers.get(node_id)

        Args:
            node_id: The peer's node ID.

        Returns:
            QueryResult containing the peer or None.
        """
        return self.safe_get(node_id)

    def get_leader(self) -> QueryResult[Optional["NodeInfo"]]:
        """Get the current leader peer.

        Replaces patterns like:
            leader = next((p for p in self.peers.values() if p.role == 'leader'), None)

        Returns:
            QueryResult containing the leader peer or None.
        """
        return self.first(lambda p: getattr(p, "role", None) == "leader")

    # =========================================================================
    # Extended filter methods (Phase 3.2 Step 6 continuation)
    # =========================================================================

    def healthy_with_gpu(
        self, *, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get healthy peers with GPU capability.

        Replaces: _get_gpu_workers_for_cmaes() pattern
            [p for p in peers if p.is_healthy() and p.has_gpu]

        Args:
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of healthy GPU-capable NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_healthy(peer) and self._has_gpu(peer)

        return self.filter(predicate)

    def training_enabled(
        self, *, alive_only: bool = True, exclude_self: bool = True
    ) -> QueryResult[List["NodeInfo"]]:
        """Get peers with training enabled.

        Args:
            alive_only: If True, only include alive peers.
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of training-enabled NodeInfo objects.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            if alive_only and not self._is_alive(peer):
                return False
            return self._is_training_enabled(peer)

        return self.filter(predicate)

    def available_for_reassignment(
        self,
        *,
        cpu_threshold: float = 90.0,
        gpu_mem_threshold: float = 95.0,
        stale_seconds: float = 120.0,
    ) -> QueryResult[List[str]]:
        """Get node IDs available for job reassignment.

        Replaces: _get_healthy_node_ids_for_reassignment() pattern
        Filters out nodes that are offline/failing/retired, overloaded, or stale.

        Args:
            cpu_threshold: Max CPU percent to be considered available.
            gpu_mem_threshold: Max GPU memory percent to be considered available.
            stale_seconds: Max seconds since last heartbeat.

        Returns:
            QueryResult containing list of available node IDs.
        """
        import time
        now = time.time()

        def predicate(peer: "NodeInfo") -> bool:
            # Skip offline/failing/retired
            status = getattr(peer, "status", "unknown")
            if status in ("offline", "failing", "retired", "archived"):
                return False
            if self._is_retired(peer):
                return False
            # Skip overloaded nodes
            if getattr(peer, "cpu_percent", 0.0) > cpu_threshold:
                return False
            if getattr(peer, "gpu_memory_percent", 0.0) > gpu_mem_threshold:
                return False
            # Skip stale nodes
            last_seen = getattr(peer, "last_seen", 0.0) or getattr(peer, "last_heartbeat", 0.0)
            if last_seen > 0 and (now - last_seen) > stale_seconds:
                return False
            return True

        return self.map(
            mapper=lambda p: getattr(p, "node_id", ""),
            predicate=predicate,
        )

    def to_endpoint_dicts(
        self, *, limit: int = 0, exclude_self: bool = True
    ) -> QueryResult[List[Dict[str, Any]]]:
        """Get alive, non-retired peers as endpoint dictionaries for gossip.

        Replaces: _get_peer_endpoints_for_gossip() pattern

        Args:
            limit: Maximum number of endpoints to return (0 = unlimited).
            exclude_self: If True, exclude this node from results.

        Returns:
            QueryResult containing list of endpoint dicts with node_id, host, port, etc.
        """
        def mapper(peer: "NodeInfo") -> Dict[str, Any]:
            return {
                "node_id": getattr(peer, "node_id", ""),
                "host": str(getattr(peer, "host", "") or ""),
                "port": int(getattr(peer, "port", 8770) or 8770),
                "tailscale_ip": str(getattr(peer, "tailscale_ip", "") or ""),
                "is_alive": True,
                "last_heartbeat": float(getattr(peer, "last_heartbeat", 0) or 0),
            }

        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer) and not self._is_retired(peer)

        result = self.map(mapper=mapper, predicate=predicate)
        if limit > 0 and result.success and len(result.data) > limit:
            result.data = result.data[:limit]
            result.count = len(result.data)
        return result

    # =========================================================================
    # Extended count methods (Phase 3.2 Step 6 continuation)
    # =========================================================================

    def alive_non_retired_count(self, *, exclude_self: bool = True) -> QueryResult[int]:
        """Count alive, non-retired peers.

        Replaces: sum(1 for p in peers if p.is_alive() and not p.retired)

        Args:
            exclude_self: If True, don't count this node.

        Returns:
            QueryResult containing count.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer) and not self._is_retired(peer)

        return self.count(predicate)

    def alive_gpu_count(self, *, exclude_self: bool = True) -> QueryResult[int]:
        """Count alive, non-retired peers with GPU.

        Replaces: sum(1 for p in peers if p.is_alive() and not p.retired and p.gpu_type)

        Args:
            exclude_self: If True, don't count this node.

        Returns:
            QueryResult containing count.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return self._is_alive(peer) and not self._is_retired(peer) and self._has_gpu(peer)

        return self.count(predicate)

    def training_enabled_count(self, *, exclude_self: bool = True) -> QueryResult[int]:
        """Count alive, non-retired peers with training enabled.

        Replaces: sum(1 for p in peers if p.is_alive() and not p.retired and p.training_enabled)

        Args:
            exclude_self: If True, don't count this node.

        Returns:
            QueryResult containing count.
        """
        def predicate(peer: "NodeInfo") -> bool:
            if exclude_self and self._is_self(peer):
                return False
            return (
                self._is_alive(peer)
                and not self._is_retired(peer)
                and self._is_training_enabled(peer)
            )

        return self.count(predicate)

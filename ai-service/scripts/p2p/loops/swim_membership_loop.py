"""SWIM membership integration loop.

January 2026: Extracted from p2p_orchestrator.py for modularity.

This loop integrates SWIM (Scalable Weakly-consistent Infection-style Membership)
protocol with the P2P peer tracking system. SWIM provides fast failure detection
through gossip-based membership updates.

Integration:
    1. Starts the SWIM manager if the swim-p2p library is available
    2. Periodically syncs SWIM membership with the peers dictionary
    3. Uses SWIM failure detection to mark peers as failed faster than HTTP heartbeats

Note: SWIM peer IDs use port 7947, while P2P uses port 8770. This loop filters
SWIM-format entries to prevent pollution of peer tracking.

Usage:
    from scripts.p2p.loops import SwimMembershipLoop, SwimMembershipConfig

    loop = SwimMembershipLoop(
        get_swim_manager=lambda: orchestrator._swim_manager,
        get_peers=lambda: orchestrator.peers,
        get_peers_lock=lambda: orchestrator.peers_lock,
    )
    await loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from scripts.p2p.loops.base import BaseLoop

if TYPE_CHECKING:
    import threading

logger = logging.getLogger(__name__)


# SWIM uses port 7947, P2P uses port 8770
SWIM_PORT = 7947
P2P_PORT = 8770


@dataclass
class SwimMembershipConfig:
    """Configuration for SWIM membership loop.

    Attributes:
        sync_interval_seconds: How often to sync SWIM membership (default: 10s)
        startup_timeout_seconds: Max time to wait for SWIM manager start (default: 30s)
        enable_raft_init: Whether to try deferred Raft initialization (default: True)
    """

    sync_interval_seconds: float = 10.0
    startup_timeout_seconds: float = 30.0
    enable_raft_init: bool = True


@dataclass
class NodeInfo:
    """Minimal node info for SWIM-discovered peers.

    Note: This is a simplified version - the full NodeInfo is discovered
    via HTTP gossip handshake.
    """

    node_id: str
    host: str
    port: int = P2P_PORT
    last_heartbeat: float = field(default_factory=time.time)


class SwimMembershipLoop(BaseLoop):
    """Integrates SWIM membership protocol with P2P peer tracking.

    This loop provides faster failure detection than HTTP heartbeats alone.
    SWIM uses gossip-based infection-style membership updates to detect
    node failures in O(log N) time.

    The loop:
    1. Starts the SWIM manager on first run
    2. Periodically syncs alive peers from SWIM to the peers dict
    3. Filters SWIM-format peer IDs (IP:7947) to prevent pollution
    4. Optionally triggers deferred Raft initialization after peers discovered

    Attributes:
        config: Configuration for sync intervals and behavior
    """

    def __init__(
        self,
        get_swim_manager: Callable[[], Any],
        get_peers: Callable[[], dict[str, Any]],
        get_peers_lock: Callable[[], "threading.RLock"],
        config: SwimMembershipConfig | None = None,
        try_raft_init: Callable[[], None] | None = None,
        get_raft_initialized: Callable[[], bool] | None = None,
    ):
        """Initialize the SWIM membership loop.

        Args:
            get_swim_manager: Callback to get SWIM manager instance
            get_peers: Callback to get peers dictionary
            get_peers_lock: Callback to get peers lock
            config: Configuration for loop behavior
            try_raft_init: Optional callback for deferred Raft initialization
            get_raft_initialized: Optional callback to check Raft init status
        """
        self._config = config or SwimMembershipConfig()
        super().__init__(
            name="swim_membership",
            interval=self._config.sync_interval_seconds,
        )
        self._get_swim_manager = get_swim_manager
        self._get_peers = get_peers
        self._get_peers_lock = get_peers_lock
        self._try_raft_init = try_raft_init
        self._get_raft_initialized = get_raft_initialized

        self._swim_started = False
        self._peers_synced = 0
        self._sync_errors = 0

    async def _on_start(self) -> None:
        """Start the SWIM manager when loop starts."""
        swim_manager = self._get_swim_manager()
        if not swim_manager:
            logger.info(f"[{self.name}] Disabled (swim-p2p not available)")
            return

        try:
            started = await swim_manager.start()
            if started:
                self._swim_started = True
                logger.info(f"[{self.name}] SWIM manager started successfully")
            else:
                logger.warning(f"[{self.name}] Failed to start SWIM manager")
        except Exception as e:
            logger.error(f"[{self.name}] Error starting SWIM manager: {e}")

    async def _on_stop(self) -> None:
        """Stop the SWIM manager when loop stops."""
        if self._swim_started:
            swim_manager = self._get_swim_manager()
            if swim_manager:
                try:
                    await swim_manager.stop()
                    logger.info(f"[{self.name}] SWIM manager stopped")
                except Exception as e:
                    logger.warning(f"[{self.name}] Error stopping SWIM manager: {e}")
            self._swim_started = False

    async def _run_once(self) -> None:
        """Execute one SWIM membership sync iteration."""
        swim_manager = self._get_swim_manager()
        if not swim_manager or not self._swim_started:
            return

        try:
            alive_peers = swim_manager.get_alive_peers()
            self._sync_peers(alive_peers)
            self._try_deferred_raft_init()
        except Exception as e:
            self._sync_errors += 1
            logger.warning(f"[{self.name}] Sync error: {e}")

    def _sync_peers(self, alive_peers: list[str]) -> None:
        """Sync SWIM alive peers to the peers dictionary.

        Args:
            alive_peers: List of peer IDs from SWIM
        """
        now = time.time()
        peers = self._get_peers()
        peers_lock = self._get_peers_lock()

        with peers_lock:
            for peer_id in alive_peers:
                # Filter SWIM protocol entries (IP:7947 format)
                # These should NOT be added to peers - they pollute VoterHealth,
                # Elo sync, and other peer iteration points
                if self._is_swim_format_peer(peer_id):
                    continue

                if peer_id not in peers:
                    # New peer detected by SWIM
                    host, port = self._parse_peer_address(peer_id)
                    peers[peer_id] = NodeInfo(
                        node_id=peer_id,
                        host=host,
                        port=P2P_PORT,  # Always use P2P port
                        last_heartbeat=now,
                    )
                    self._peers_synced += 1
                else:
                    # Update existing peer's heartbeat
                    peer = peers[peer_id]
                    if hasattr(peer, "last_heartbeat"):
                        peer.last_heartbeat = now

    def _is_swim_format_peer(self, peer_id: str) -> bool:
        """Check if peer ID is in SWIM format (IP:7947).

        SWIM peer IDs like "100.126.21.102:7947" should be filtered.
        Proper peers are discovered via HTTP gossip with node names or IP:8770.

        Args:
            peer_id: Peer identifier to check

        Returns:
            True if this is a SWIM-format peer ID that should be filtered
        """
        if ":" not in peer_id:
            return False
        _, port_str = peer_id.rsplit(":", 1)
        return port_str == str(SWIM_PORT)

    def _parse_peer_address(self, peer_id: str) -> tuple[str, int]:
        """Parse peer address from peer ID.

        Args:
            peer_id: Peer identifier (may be "host:port" or just "host")

        Returns:
            Tuple of (host, port)
        """
        if ":" in peer_id:
            parts = peer_id.rsplit(":", 1)
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                port = P2P_PORT
        else:
            host = peer_id
            port = P2P_PORT
        return host or "unknown", port

    def _try_deferred_raft_init(self) -> None:
        """Try deferred Raft initialization after peers discovered.

        Raft needs peer addresses which aren't available at startup.
        """
        if not self._config.enable_raft_init:
            return

        if not self._try_raft_init:
            return

        # Check if already initialized
        if self._get_raft_initialized and self._get_raft_initialized():
            return

        try:
            from scripts.p2p.constants import RAFT_ENABLED

            if RAFT_ENABLED:
                self._try_raft_init()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[{self.name}] Deferred Raft init attempt: {e}")

    def health_check(self) -> "HealthCheckResult":
        """Check loop health for DaemonManager integration.

        Returns:
            HealthCheckResult with SWIM membership statistics
        """
        try:
            from app.coordination.protocols import CoordinatorStatus, HealthCheckResult
        except ImportError:
            return {  # type: ignore[return-value]
                "healthy": self._running,
                "status": "running" if self._running else "stopped",
                "message": f"SWIM loop: {self._peers_synced} peers synced",
                "details": self._get_health_details(),
            }

        if not self._running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message=f"Loop {self.name} is stopped",
            )

        swim_manager = self._get_swim_manager()
        if not swim_manager:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.IDLE,
                message="SWIM not available (swim-p2p not installed)",
            )

        if not self._swim_started:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="SWIM manager failed to start",
                details=self._get_health_details(),
            )

        # Check error rate
        total_ops = self._peers_synced + self._sync_errors
        if total_ops > 0:
            error_rate = self._sync_errors / total_ops
            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"High SWIM sync error rate: {error_rate:.1%}",
                    details=self._get_health_details(),
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message=f"SWIM healthy ({self._peers_synced} peers synced)",
            details=self._get_health_details(),
        )

    def _get_health_details(self) -> dict[str, Any]:
        """Get detailed health statistics."""
        return {
            "swim_started": self._swim_started,
            "peers_synced": self._peers_synced,
            "sync_errors": self._sync_errors,
            "sync_interval": self._config.sync_interval_seconds,
            "raft_init_enabled": self._config.enable_raft_init,
        }

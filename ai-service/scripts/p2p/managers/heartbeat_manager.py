"""Heartbeat Manager for P2P Orchestrator.

January 2026: Phase 14 of P2P Orchestrator Decomposition.
Consolidates heartbeat, bootstrap, relay, and voter heartbeat methods.

This module provides:
- Core heartbeat sending (HTTP + SSH fallback)
- Bootstrap from seed peers
- Relay heartbeats for NAT-blocked nodes
- Voter-specific high-frequency heartbeats
- Dead peer reconnection

Extracted from p2p_orchestrator.py to reduce complexity and improve testability.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator
    from scripts.p2p.models import NodeInfo

logger = logging.getLogger(__name__)

# Constants (from orchestrator)
DEFAULT_PORT = 8770
PEER_BOOTSTRAP_INTERVAL = 60  # Seconds between bootstrap attempts
STARTUP_GRACE_PERIOD = 30  # Grace period for startup
ISOLATED_BOOTSTRAP_INTERVAL = 120  # Interval for continuous bootstrap
MIN_CONNECTED_PEERS = 3  # Minimum peers before considered isolated


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat behavior."""
    heartbeat_interval: float = 15.0
    peer_timeout: float = 60.0
    voter_heartbeat_interval: float = 10.0
    voter_heartbeat_timeout: float = 8.0
    peer_bootstrap_interval: float = 60.0
    relay_heartbeat_interval: float = 30.0
    dead_peer_probe_interval: float = 15.0
    # Bootstrap settings
    bootstrap_max_seeds_per_run: int = 8
    bootstrap_timeout: float = 8.0
    # Voter settings
    voter_max_reconnect_per_cycle: int = 3
    voter_mesh_refresh_interval: float = 300.0


@dataclass
class HeartbeatStats:
    """Statistics about heartbeat operations."""
    heartbeats_sent: int = 0
    heartbeat_failures: int = 0
    ssh_fallback_attempts: int = 0
    ssh_fallback_successes: int = 0
    voter_heartbeats_sent: int = 0
    bootstrap_attempts: int = 0
    bootstrap_successes: int = 0
    peers_imported: int = 0
    relay_heartbeats_sent: int = 0
    relay_heartbeat_failures: int = 0
    dead_peer_reconnections: int = 0
    reconnection_failures: int = 0


# ============================================================================
# Singleton Pattern
# ============================================================================

_heartbeat_manager: HeartbeatManager | None = None


def get_heartbeat_manager() -> HeartbeatManager | None:
    """Get the global HeartbeatManager singleton."""
    return _heartbeat_manager


def set_heartbeat_manager(manager: HeartbeatManager) -> None:
    """Set the global HeartbeatManager singleton."""
    global _heartbeat_manager
    _heartbeat_manager = manager


def reset_heartbeat_manager() -> None:
    """Reset the global HeartbeatManager singleton (for testing)."""
    global _heartbeat_manager
    _heartbeat_manager = None


def create_heartbeat_manager(
    config: HeartbeatConfig | None = None,
    orchestrator: P2POrchestrator | None = None,
) -> HeartbeatManager:
    """Factory function to create and register a HeartbeatManager.

    Args:
        config: Optional configuration. Uses defaults if not provided.
        orchestrator: The P2P orchestrator instance.

    Returns:
        The created HeartbeatManager instance.
    """
    manager = HeartbeatManager(
        config=config or HeartbeatConfig(),
        orchestrator=orchestrator,
    )
    set_heartbeat_manager(manager)
    return manager


# ============================================================================
# HeartbeatManager
# ============================================================================

class HeartbeatManager:
    """Manages heartbeat operations for P2P cluster.

    This manager handles:
    - Sending heartbeats to peers (HTTP + SSH fallback)
    - Bootstrapping from seed peers
    - Relay heartbeats for NAT-blocked nodes
    - Voter-specific heartbeats
    - Dead peer reconnection

    Jan 27, 2026: Phase 14 decomposition from p2p_orchestrator.py.
    """

    def __init__(
        self,
        config: HeartbeatConfig | None = None,
        orchestrator: P2POrchestrator | None = None,
    ):
        """Initialize the HeartbeatManager.

        Args:
            config: Heartbeat configuration.
            orchestrator: The P2P orchestrator instance.
        """
        self.config = config or HeartbeatConfig()
        self._orchestrator = orchestrator
        self._stats = HeartbeatStats()

        # State tracking
        self._last_peer_bootstrap: float = 0.0
        self._last_voter_mesh_refresh: float = 0.0
        self._voter_reconnect_backoff: dict[str, float] = {}

        # Relay state
        self._relay_heartbeat_errors: int = 0
        self._last_relay_heartbeat: float = 0.0

    # =========================================================================
    # Properties (delegate to orchestrator)
    # =========================================================================

    @property
    def _peers(self) -> dict[str, Any]:
        """Get peers dict from orchestrator."""
        return getattr(self._orchestrator, "peers", {})

    @property
    def _peers_lock(self) -> Any:
        """Get peers lock from orchestrator."""
        return getattr(self._orchestrator, "peers_lock", None)

    @property
    def _node_id(self) -> str:
        """Get this node's ID."""
        return getattr(self._orchestrator, "node_id", "unknown")

    @property
    def _self_info(self) -> Any:
        """Get this node's info."""
        return getattr(self._orchestrator, "self_info", None)

    @property
    def _leader_id(self) -> str | None:
        """Get current leader ID."""
        return getattr(self._orchestrator, "leader_id", None)

    @property
    def _voter_node_ids(self) -> list[str]:
        """Get voter node IDs."""
        return list(getattr(self._orchestrator, "voter_node_ids", []) or [])

    @property
    def _running(self) -> bool:
        """Check if orchestrator is running."""
        return getattr(self._orchestrator, "running", False)

    # =========================================================================
    # Helper Methods (delegate to orchestrator)
    # =========================================================================

    def _auth_headers(self) -> dict[str, str]:
        """Get auth headers from orchestrator."""
        if self._orchestrator and hasattr(self._orchestrator, "_auth_headers"):
            return self._orchestrator._auth_headers()
        return {}

    def _parse_peer_address(self, addr: str) -> tuple[str, str, int]:
        """Parse peer address into scheme, host, port."""
        if self._orchestrator and hasattr(self._orchestrator, "_parse_peer_address"):
            return self._orchestrator._parse_peer_address(addr)
        # Fallback parsing
        scheme = "http"
        if "://" in addr:
            scheme, addr = addr.split("://", 1)
        if ":" in addr:
            host, port_str = addr.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                port = DEFAULT_PORT
        else:
            host = addr
            port = DEFAULT_PORT
        return scheme, host, port

    async def _update_self_info_async(self) -> None:
        """Update self info asynchronously."""
        if self._orchestrator and hasattr(self._orchestrator, "_update_self_info_async"):
            await self._orchestrator._update_self_info_async()

    def _maybe_adopt_voter_node_ids(self, voters: list[str], source: str = "unknown") -> None:
        """Adopt voter node IDs from peer."""
        if self._orchestrator and hasattr(self._orchestrator, "_maybe_adopt_voter_node_ids"):
            self._orchestrator._maybe_adopt_voter_node_ids(voters, source=source)

    def _save_peer_to_cache(self, node_id: str, host: str, port: int, tailscale_ip: str) -> None:
        """Save peer to cache."""
        if self._orchestrator and hasattr(self._orchestrator, "_save_peer_to_cache"):
            self._orchestrator._save_peer_to_cache(node_id, host, port, tailscale_ip)

    def _update_peer_reputation(self, node_id: str, success: bool = True) -> None:
        """Update peer reputation."""
        if self._orchestrator and hasattr(self._orchestrator, "_update_peer_reputation"):
            self._orchestrator._update_peer_reputation(node_id, success=success)

    def _sync_peer_snapshot(self) -> None:
        """Sync peer snapshot."""
        if self._orchestrator and hasattr(self._orchestrator, "_sync_peer_snapshot"):
            self._orchestrator._sync_peer_snapshot()

    def _set_leader(self, leader_id: str, reason: str = "", save_state: bool = True) -> None:
        """Set leader atomically."""
        if self._orchestrator and hasattr(self._orchestrator, "_set_leader"):
            self._orchestrator._set_leader(leader_id, reason=reason, save_state=save_state)

    def _save_state(self) -> None:
        """Save orchestrator state."""
        if self._orchestrator and hasattr(self._orchestrator, "_save_state"):
            self._orchestrator._save_state()

    def _maybe_adopt_leader_from_peers(self) -> None:
        """Try to adopt leader from peers."""
        if self._orchestrator and hasattr(self._orchestrator, "_maybe_adopt_leader_from_peers"):
            self._orchestrator._maybe_adopt_leader_from_peers()

    def _has_voter_quorum(self) -> bool:
        """Check if voter quorum is available."""
        if self._orchestrator and hasattr(self._orchestrator, "_has_voter_quorum"):
            return self._orchestrator._has_voter_quorum()
        return True

    async def _start_election(self) -> None:
        """Start leader election."""
        if self._orchestrator and hasattr(self._orchestrator, "_start_election"):
            await self._orchestrator._start_election()

    async def _discover_tailscale_peers(self) -> None:
        """Discover peers via Tailscale."""
        if self._orchestrator and hasattr(self._orchestrator, "_discover_tailscale_peers"):
            await self._orchestrator._discover_tailscale_peers()

    # =========================================================================
    # Core Heartbeat Methods
    # =========================================================================

    async def send_heartbeat_to_peer(
        self,
        peer_host: str,
        peer_port: int,
        scheme: str = "http",
        timeout: int = 15,
    ) -> Any | None:
        """Send heartbeat to a peer and return their info.

        Args:
            peer_host: Target peer hostname or IP
            peer_port: Target peer port
            scheme: HTTP or HTTPS scheme
            timeout: Request timeout in seconds

        Returns:
            NodeInfo if successful, None otherwise

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        if aiohttp is None:
            return None

        from scripts.p2p.models import NodeInfo
        from scripts.p2p.network import get_client_session, safe_json_response
        from app.distributed.circuit_breaker import CircuitState

        target = f"{peer_host}:{peer_port}"
        circuit_registry = getattr(self._orchestrator, "_circuit_registry", None)

        if circuit_registry:
            breaker = circuit_registry.get_breaker("p2p")
            if not breaker.can_execute(target):
                state = breaker.get_state(target)
                if state == CircuitState.OPEN:
                    return None
        else:
            breaker = None

        http_failed = False

        # Prepare payload
        await self._update_self_info_async()
        if self._self_info:
            payload = self._self_info.to_dict()
        else:
            payload = {}

        voter_node_ids = self._voter_node_ids
        if voter_node_ids:
            payload["voter_node_ids"] = voter_node_ids
            payload["voter_quorum_size"] = int(getattr(self._orchestrator, "voter_quorum_size", 0) or 0)
            payload["voter_config_source"] = str(getattr(self._orchestrator, "voter_config_source", "") or "")

        try:
            effective_timeout = float(timeout)
            if circuit_registry:
                effective_timeout = circuit_registry.get_timeout("p2p", target, float(timeout))
            client_timeout = ClientTimeout(total=effective_timeout)

            async with get_client_session(client_timeout) as session:
                scheme = (scheme or "http").lower()
                url = f"{scheme}://{peer_host}:{peer_port}/heartbeat"
                async with session.post(url, json=payload, headers=self._auth_headers()) as resp:
                    data, json_error = await safe_json_response(resp, default=None, log_errors=False)
                    if json_error or data is None:
                        if breaker:
                            breaker.record_failure(target)
                        http_failed = True
                    else:
                        # Process voter info
                        incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
                        if incoming_voters:
                            voters_list: list[str] = []
                            if isinstance(incoming_voters, list):
                                voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                            elif isinstance(incoming_voters, str):
                                voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                            if voters_list:
                                self._maybe_adopt_voter_node_ids(voters_list, source="learned")

                        info = NodeInfo.from_dict(data)
                        if not info.reported_host:
                            info.reported_host = info.host
                        if not info.reported_port:
                            info.reported_port = info.port
                        info.scheme = scheme
                        info.host = peer_host
                        info.port = peer_port

                        if breaker:
                            breaker.record_success(target)

                        self._save_peer_to_cache(
                            info.node_id,
                            peer_host,
                            peer_port,
                            str(getattr(info, "tailscale_ip", "") or "")
                        )
                        self._update_peer_reputation(info.node_id, success=True)
                        self._stats.heartbeats_sent += 1

                        return info

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError):
            if breaker:
                breaker.record_failure(target)
            http_failed = True

        # SSH fallback
        if http_failed:
            self._stats.heartbeat_failures += 1
            hybrid_transport = getattr(self._orchestrator, "hybrid_transport", None)
            if hybrid_transport:
                self._stats.ssh_fallback_attempts += 1
                info = await self._send_heartbeat_via_ssh_fallback(peer_host, peer_port, payload)
                if info:
                    if breaker:
                        breaker.record_success(target)
                    self._stats.ssh_fallback_successes += 1
                    return info

        return None

    async def _send_heartbeat_via_ssh_fallback(
        self,
        peer_host: str,
        peer_port: int,
        payload: dict[str, Any],
    ) -> Any | None:
        """Send heartbeat via SSH when HTTP fails.

        Args:
            peer_host: Target peer hostname or IP
            peer_port: Target peer port
            payload: Heartbeat payload dict

        Returns:
            NodeInfo if successful, None otherwise

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        from scripts.p2p.models import NodeInfo

        hybrid_transport = getattr(self._orchestrator, "hybrid_transport", None)
        if not hybrid_transport:
            return None

        node_id = self.find_node_id_for_host(peer_host)
        if not node_id:
            return None

        try:
            success, response = await hybrid_transport.send_heartbeat(
                node_id=node_id,
                host=peer_host,
                port=peer_port,
                self_info=payload,
            )

            if success and response:
                info = NodeInfo.from_dict(response)
                if not info.reported_host:
                    info.reported_host = info.host
                if not info.reported_port:
                    info.reported_port = info.port
                info.scheme = "http"
                info.host = peer_host
                info.port = peer_port

                self._save_peer_to_cache(
                    info.node_id,
                    peer_host,
                    peer_port,
                    str(getattr(info, "tailscale_ip", "") or "")
                )
                self._update_peer_reputation(info.node_id, success=True)

                logger.debug(f"[Heartbeat] SSH fallback successful to {node_id}")
                return info

        except Exception as e:  # noqa: BLE001
            logger.debug(f"[Heartbeat] SSH fallback failed for {node_id}: {e}")

        return None

    def find_node_id_for_host(self, host: str) -> str | None:
        """Find node_id for a given host address.

        Args:
            host: Hostname or IP address

        Returns:
            node_id if found, None otherwise

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        # Check peers first
        if self._peers_lock:
            with self._peers_lock:
                for peer in self._peers.values():
                    if peer.host == host or getattr(peer, "tailscale_ip", None) == host:
                        return peer.node_id

        # Check cluster_hosts.yaml
        try:
            from app.sync.cluster_hosts import get_cluster_nodes
            configured_hosts = get_cluster_nodes()
            for name, cfg in configured_hosts.items():
                if cfg.tailscale_ip == host or cfg.ssh_host == host or cfg.best_ip == host:
                    return name
        except (AttributeError, ImportError):
            pass

        return None

    # =========================================================================
    # Bootstrap Methods
    # =========================================================================

    async def bootstrap_from_known_peers(self) -> bool:
        """Import cluster membership from seed peers via `/relay/peers`.

        Returns:
            True if any peers were imported, False otherwise.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        if aiohttp is None:
            return False

        from scripts.p2p.models import NodeInfo, NodeRole
        from scripts.p2p.network import get_client_session

        known_peers = getattr(self._orchestrator, "known_peers", []) or []
        known_seed_peers: list[str] = [p for p in known_peers if p]
        discovered_seed_peers: list[str] = []

        # Get snapshot of current peers
        if self._peers_lock:
            with self._peers_lock:
                peers_snapshot = [p for p in self._peers.values() if p.node_id != self._node_id]
        else:
            peers_snapshot = []
        peers_snapshot.sort(key=lambda p: str(getattr(p, "node_id", "") or ""))

        for peer in peers_snapshot:
            if getattr(peer, "nat_blocked", False):
                continue
            if not peer.should_retry():
                continue

            scheme = (getattr(peer, "scheme", "http") or "http").lower()
            host = str(getattr(peer, "host", "") or "").strip()
            try:
                port = int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT)
            except ValueError:
                port = DEFAULT_PORT
            if host:
                discovered_seed_peers.append(f"{scheme}://{host}:{port}")

            rh = str(getattr(peer, "reported_host", "") or "").strip()
            try:
                rp = int(getattr(peer, "reported_port", 0) or 0)
            except ValueError:
                rp = 0
            if rh and rp:
                discovered_seed_peers.append(f"{scheme}://{rh}:{rp}")

        # Interleave known and discovered peers
        seen: set[str] = set()
        seed_peers: list[str] = []
        ki = 0
        di = 0
        while ki < len(known_seed_peers) or di < len(discovered_seed_peers):
            if ki < len(known_seed_peers):
                candidate = known_seed_peers[ki]
                ki += 1
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    seed_peers.append(candidate)
            if di < len(discovered_seed_peers):
                candidate = discovered_seed_peers[di]
                di += 1
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    seed_peers.append(candidate)

        if not seed_peers:
            return False

        now = time.time()
        if now - self._last_peer_bootstrap < self.config.peer_bootstrap_interval:
            return False

        max_seeds = self.config.bootstrap_max_seeds_per_run
        timeout = ClientTimeout(total=self.config.bootstrap_timeout)
        bootstrapped = False
        imported_any = False

        self._stats.bootstrap_attempts += 1

        async with get_client_session(timeout) as session:
            for idx, peer_addr in enumerate(seed_peers):
                if idx >= max_seeds:
                    break
                try:
                    scheme, host, port = self._parse_peer_address(peer_addr)
                    scheme = (scheme or "http").lower()
                    url = f"{scheme}://{host}:{port}/relay/peers"
                    async with session.get(url, headers=self._auth_headers()) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()

                    if not isinstance(data, dict) or not data.get("success"):
                        continue

                    bootstrapped = True

                    # Process voter info
                    incoming_voters = data.get("voter_node_ids") or data.get("voters") or None
                    if incoming_voters:
                        voters_list: list[str] = []
                        if isinstance(incoming_voters, list):
                            voters_list = [str(v).strip() for v in incoming_voters if str(v).strip()]
                        elif isinstance(incoming_voters, str):
                            voters_list = [t.strip() for t in incoming_voters.split(",") if t.strip()]
                        if voters_list:
                            self._maybe_adopt_voter_node_ids(voters_list, source="learned")

                    peers_data = data.get("peers") or {}
                    if not isinstance(peers_data, dict):
                        continue

                    # Import peers
                    if self._peers_lock:
                        with self._peers_lock:
                            before = len(self._peers)
                            for node_id, peer_dict in peers_data.items():
                                if not node_id or node_id == self._node_id:
                                    continue
                                try:
                                    info = NodeInfo.from_dict(peer_dict)
                                except AttributeError:
                                    continue
                                existing = self._peers.get(info.node_id)
                                if existing:
                                    # Preserve state when merging
                                    if getattr(existing, "nat_blocked", False) and not getattr(info, "nat_blocked", False):
                                        info.nat_blocked = True
                                        info.nat_blocked_since = float(getattr(existing, "nat_blocked_since", 0.0) or 0.0) or time.time()
                                        info.last_nat_probe = float(getattr(existing, "last_nat_probe", 0.0) or 0.0)
                                    if (getattr(existing, "relay_via", "") or "") and not (getattr(info, "relay_via", "") or ""):
                                        info.relay_via = str(getattr(existing, "relay_via", "") or "")
                                    if getattr(existing, "retired", False):
                                        info.retired = True
                                        info.retired_at = float(getattr(existing, "retired_at", 0.0) or 0.0)
                                    info.consecutive_failures = int(getattr(existing, "consecutive_failures", 0) or 0)
                                    info.last_failure_time = float(getattr(existing, "last_failure_time", 0.0) or 0.0)

                                self._peers[info.node_id] = info
                            after = len(self._peers)

                        self._sync_peer_snapshot()
                        new = max(0, after - before)
                        if new:
                            imported_any = True
                            self._stats.peers_imported += new
                            logger.info(f"Bootstrap: imported {new} new peers from {host}:{port}")

                    # Process leader info
                    leader_id = str(data.get("leader_id") or "").strip()
                    if leader_id and leader_id != self._node_id:
                        role = getattr(self._orchestrator, "role", None)
                        if role == NodeRole.LEADER and leader_id > self._node_id:
                            logger.info(f"Bootstrap: stepping down for leader {leader_id}")
                        self._set_leader(leader_id, reason="bootstrap_discover_leader", save_state=False)

                except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, KeyError, ValueError):
                    continue

        self._last_peer_bootstrap = now
        if bootstrapped:
            self._stats.bootstrap_successes += 1
            self._maybe_adopt_leader_from_peers()
            self._save_state()
        return imported_any

    async def continuous_bootstrap_loop(self) -> None:
        """Continuously attempt to join cluster when isolated.

        This loop runs on ALL nodes and ensures isolated nodes can rejoin.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        known_peers = getattr(self._orchestrator, "known_peers", []) or []
        if known_peers:
            logger.debug(f"[ContinuousBootstrap] Waiting {STARTUP_GRACE_PERIOD}s grace period")
            await asyncio.sleep(STARTUP_GRACE_PERIOD)
        else:
            logger.info("[ContinuousBootstrap] No known peers, skipping startup grace period")
            await asyncio.sleep(5)

        while self._running:
            try:
                await asyncio.sleep(ISOLATED_BOOTSTRAP_INTERVAL)

                # Count alive peers
                peers_alive = 0
                if self._peers_lock:
                    with self._peers_lock:
                        peers_alive = sum(
                            1 for p in self._peers.values()
                            if p.node_id != self._node_id and p.is_alive()
                        )

                is_isolated = peers_alive < MIN_CONNECTED_PEERS
                no_leader = self._leader_id is None or (
                    self._leader_id != self._node_id and
                    self._leader_id not in self._peers
                )

                if is_isolated or no_leader:
                    if is_isolated:
                        logger.warning(f"Isolated: only {peers_alive} alive peers, attempting bootstrap...")
                    elif no_leader:
                        logger.warning(f"No valid leader (current: {self._leader_id}), attempting bootstrap...")

                    bootstrapped = await self.bootstrap_from_multiple_seeds()

                    if bootstrapped:
                        logger.info(f"Bootstrap successful!")
                        self._maybe_adopt_leader_from_peers()

                        if not self._leader_id:
                            if self._voter_node_ids and not self._has_voter_quorum():
                                logger.warning("Skipping election after bootstrap: no voter quorum")
                            else:
                                await self._start_election()
                    else:
                        logger.info("Bootstrap from seeds failed, trying Tailscale discovery...")
                        await self._discover_tailscale_peers()

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error in continuous bootstrap loop: {e}")
                await asyncio.sleep(30)

    async def bootstrap_from_multiple_seeds(self) -> bool:
        """Bootstrap from multiple seed sources.

        Sources tried in order:
        1. Cached peers (highest reputation first)
        2. CLI-provided peers
        3. Hardcoded BOOTSTRAP_SEEDS
        4. Tailscale network scan

        Returns:
            True if any peers were imported.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        # Delegate to orchestrator's implementation for now
        # This method has complex interactions with peer cache
        if self._orchestrator and hasattr(self._orchestrator, "_bootstrap_from_multiple_seeds"):
            return await self._orchestrator._bootstrap_from_multiple_seeds()
        return await self.bootstrap_from_known_peers()

    def load_bootstrap_seeds_from_config(self) -> list[str]:
        """Load bootstrap seeds from configuration.

        Returns:
            List of seed peer addresses.

        Jan 27, 2026: Migrated from p2p_orchestrator.py (Phase 14).
        """
        if self._orchestrator and hasattr(self._orchestrator, "_load_bootstrap_seeds_from_config"):
            return self._orchestrator._load_bootstrap_seeds_from_config()
        return []

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            dict with healthy status, message, and details.
        """
        is_healthy = self._running
        stats = asdict(self._stats)

        # Calculate health metrics
        if self._stats.heartbeats_sent > 0:
            failure_rate = self._stats.heartbeat_failures / self._stats.heartbeats_sent
            if failure_rate > 0.5:
                is_healthy = False

        message = "Heartbeat manager healthy" if is_healthy else "High heartbeat failure rate"

        return {
            "healthy": is_healthy,
            "message": message,
            "details": stats,
        }

    def get_stats(self) -> HeartbeatStats:
        """Get current statistics."""
        return self._stats

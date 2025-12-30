"""Tailscale Peer Discovery Loop for P2P Orchestrator.

December 2025: Bridges the gap between Tailscale connectivity and P2P peer discovery.

Problem: P2P orchestrator only discovers peers via heartbeat propagation. If a peer
restarts or loses heartbeat chain, it becomes invisible even though Tailscale shows
it as online. This causes clusters to show 5-7 connected peers when 40 are online.

Solution: Actively query `tailscale status --json` to discover peers that are online
in Tailscale but missing from the P2P mesh, then proactively reconnect to them.

Usage:
    from scripts.p2p.loops.tailscale_discovery_loop import (
        TailscalePeerDiscoveryLoop,
        TailscaleDiscoveryConfig,
    )

    discovery_loop = TailscalePeerDiscoveryLoop(
        get_tailscale_peers=orchestrator.get_tailscale_status,
        get_config_hosts=lambda: load_hosts().get('hosts', {}),
        get_current_peers=lambda: dict(orchestrator.peers),
        reconnect_peer=orchestrator.reconnect_discovered_peer,
    )
    await discovery_loop.run_forever()

Events:
    PEER_DISCOVERED: Emitted when a peer is found in Tailscale but missing from P2P
    PEER_RECONNECTED: Emitted when a missing peer is successfully reconnected
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from .base import BaseLoop, LoopStats

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TailscaleDiscoveryConfig:
    """Configuration for Tailscale peer discovery loop."""

    # Interval between discovery cycles (seconds)
    discovery_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_P2P_TAILSCALE_DISCOVERY_INTERVAL", "60")
        )
    )

    # Timeout for Tailscale CLI command (seconds)
    tailscale_timeout_seconds: float = 10.0

    # Timeout for HTTP probe to peer (seconds)
    connect_timeout_seconds: float = 5.0

    # Whether to automatically reconnect to missing peers
    auto_reconnect: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_P2P_TAILSCALE_AUTO_RECONNECT", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Maximum peers to attempt reconnection per cycle (limit load)
    max_reconnect_per_cycle: int = 10

    # P2P port for health probe
    p2p_port: int = 8770

    # Whether the loop is enabled
    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_P2P_TAILSCALE_DISCOVERY_ENABLED", "true"
        ).lower()
        in {"1", "true", "yes", "on"}
    )

    # Emit events for monitoring
    emit_events: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.discovery_interval_seconds <= 0:
            raise ValueError("discovery_interval_seconds must be > 0")
        if self.tailscale_timeout_seconds <= 0:
            raise ValueError("tailscale_timeout_seconds must be > 0")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be > 0")
        if self.max_reconnect_per_cycle <= 0:
            raise ValueError("max_reconnect_per_cycle must be > 0")


@dataclass
class TailscaleDiscoveryStats(LoopStats):
    """Extended statistics for Tailscale discovery loop."""

    peers_discovered: int = 0
    peers_reconnected: int = 0
    peers_failed: int = 0
    tailscale_online_count: int = 0
    p2p_connected_count: int = 0
    last_health_score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for JSON serialization."""
        base = super().to_dict()
        base.update({
            "peers_discovered": self.peers_discovered,
            "peers_reconnected": self.peers_reconnected,
            "peers_failed": self.peers_failed,
            "tailscale_online_count": self.tailscale_online_count,
            "p2p_connected_count": self.p2p_connected_count,
            "last_health_score": round(self.last_health_score, 2),
        })
        return base


# =============================================================================
# Tailscale Peer Discovery Loop
# =============================================================================


class TailscalePeerDiscoveryLoop(BaseLoop):
    """Background loop that discovers peers from Tailscale and reconnects missing ones.

    Key features:
    - Queries `tailscale status --json` at configurable interval (default: 60s)
    - Compares Tailscale online peers against P2P connected peers
    - Proactively reconnects to peers online in Tailscale but missing from P2P
    - Reports health score (P2P connected / Tailscale online)
    - Emits events for monitoring and alerting
    """

    def __init__(
        self,
        get_tailscale_peers: Callable[[], Coroutine[Any, Any, dict[str, bool]]],
        get_config_hosts: Callable[[], dict[str, dict[str, Any]]],
        get_current_peers: Callable[[], dict[str, Any]],
        reconnect_peer: Callable[[str, str, int], Coroutine[Any, Any, bool]],
        emit_event: Callable[[str, dict[str, Any]], None] | None = None,
        config: TailscaleDiscoveryConfig | None = None,
    ):
        """Initialize Tailscale peer discovery loop.

        Args:
            get_tailscale_peers: Async function returning {tailscale_ip: is_online}
            get_config_hosts: Function returning hosts config from distributed_hosts.yaml
            get_current_peers: Function returning current P2P peers dict
            reconnect_peer: Async function to reconnect a peer (node_id, host, port)
            emit_event: Optional function to emit events (event_type, data)
            config: Optional configuration (uses defaults if not provided)
        """
        self._config = config or TailscaleDiscoveryConfig()

        super().__init__(
            name="tailscale_discovery",
            interval=self._config.discovery_interval_seconds,
        )

        self._get_tailscale_peers = get_tailscale_peers
        self._get_config_hosts = get_config_hosts
        self._get_current_peers = get_current_peers
        self._reconnect_peer = reconnect_peer
        self._emit_event = emit_event

        # Extended stats
        self._stats = TailscaleDiscoveryStats(name="tailscale_discovery")

        # Check if tailscale CLI is available
        self._tailscale_path = shutil.which("tailscale")
        if not self._tailscale_path:
            logger.warning("Tailscale CLI not found - discovery loop will be limited")

        # Track last discovery results for debugging
        self._last_missing_peers: list[str] = []
        self._last_reconnection_results: dict[str, bool] = {}

    @property
    def stats(self) -> TailscaleDiscoveryStats:
        """Get loop statistics."""
        return self._stats

    @property
    def is_enabled(self) -> bool:
        """Check if loop is enabled."""
        return self._config.enabled

    async def _run_once(self) -> None:
        """Execute one discovery cycle."""
        if not self._config.enabled:
            return

        start_time = time.time()

        # 1. Get Tailscale online peers
        ts_peers = await self._get_tailscale_peers()
        ts_online_ips = {ip for ip, online in ts_peers.items() if online}
        self._stats.tailscale_online_count = len(ts_online_ips)

        # 2. Get configured hosts with tailscale_ip
        config_hosts = self._get_config_hosts()
        ip_to_node: dict[str, tuple[str, dict]] = {}
        for node_name, node_config in config_hosts.items():
            ts_ip = node_config.get("tailscale_ip")
            if ts_ip and node_config.get("p2p_enabled", True):
                ip_to_node[ts_ip] = (node_name, node_config)

        # 3. Get current P2P peers
        current_peers = self._get_current_peers()
        current_peer_ids = set()
        for peer_id, peer_info in current_peers.items():
            # Check if peer is alive (has recent heartbeat)
            is_alive = getattr(peer_info, "is_alive", lambda: True)
            if callable(is_alive):
                if is_alive():
                    current_peer_ids.add(peer_id)
            elif is_alive:
                current_peer_ids.add(peer_id)

        self._stats.p2p_connected_count = len(current_peer_ids)

        # 4. Find missing peers (online in Tailscale but not in P2P)
        missing_peers: list[tuple[str, str, dict]] = []
        for ts_ip in ts_online_ips:
            if ts_ip in ip_to_node:
                node_name, node_config = ip_to_node[ts_ip]
                if node_name not in current_peer_ids:
                    missing_peers.append((node_name, ts_ip, node_config))

        self._last_missing_peers = [name for name, _, _ in missing_peers]
        self._stats.peers_discovered += len(missing_peers)

        # 5. Calculate health score
        if self._stats.tailscale_online_count > 0:
            self._stats.last_health_score = (
                self._stats.p2p_connected_count / self._stats.tailscale_online_count
            )
        else:
            self._stats.last_health_score = 1.0

        # 6. Log discrepancy
        if missing_peers:
            logger.info(
                f"Tailscale discovery: {len(missing_peers)} peers online in Tailscale "
                f"but missing from P2P: {[n for n, _, _ in missing_peers[:5]]}..."
                if len(missing_peers) > 5
                else f"Tailscale discovery: {len(missing_peers)} peers missing: "
                f"{[n for n, _, _ in missing_peers]}"
            )

            # Emit discovery event
            if self._emit_event and self._config.emit_events:
                self._emit_event(
                    "PEER_DISCOVERY_GAP",
                    {
                        "tailscale_online": self._stats.tailscale_online_count,
                        "p2p_connected": self._stats.p2p_connected_count,
                        "missing_count": len(missing_peers),
                        "missing_peers": [n for n, _, _ in missing_peers],
                        "health_score": self._stats.last_health_score,
                    },
                )

        # 7. Attempt reconnection if enabled
        if self._config.auto_reconnect and missing_peers:
            self._last_reconnection_results = {}

            for node_name, ts_ip, node_config in missing_peers[
                : self._config.max_reconnect_per_cycle
            ]:
                port = node_config.get("p2p_port", self._config.p2p_port)
                try:
                    success = await self._reconnect_peer(node_name, ts_ip, port)
                    self._last_reconnection_results[node_name] = success

                    if success:
                        self._stats.peers_reconnected += 1
                        logger.info(
                            f"Reconnected to {node_name} via Tailscale discovery "
                            f"({ts_ip}:{port})"
                        )

                        # Emit reconnection event
                        if self._emit_event and self._config.emit_events:
                            self._emit_event(
                                "PEER_RECONNECTED",
                                {
                                    "node_id": node_name,
                                    "tailscale_ip": ts_ip,
                                    "port": port,
                                    "source": "tailscale_discovery",
                                },
                            )
                    else:
                        self._stats.peers_failed += 1
                        logger.debug(f"Failed to reconnect to {node_name} ({ts_ip})")

                except Exception as e:
                    self._stats.peers_failed += 1
                    self._last_reconnection_results[node_name] = False
                    logger.warning(f"Error reconnecting to {node_name}: {e}")

        # Log summary
        duration = time.time() - start_time
        logger.debug(
            f"Tailscale discovery cycle completed in {duration:.2f}s: "
            f"TS={self._stats.tailscale_online_count}, "
            f"P2P={self._stats.p2p_connected_count}, "
            f"missing={len(missing_peers)}, "
            f"health={self._stats.last_health_score:.2f}"
        )

    def get_health_report(self) -> dict[str, Any]:
        """Get current health report for monitoring."""
        return {
            "enabled": self._config.enabled,
            "tailscale_available": self._tailscale_path is not None,
            "tailscale_online": self._stats.tailscale_online_count,
            "p2p_connected": self._stats.p2p_connected_count,
            "missing_from_p2p": self._last_missing_peers,
            "health_score": round(self._stats.last_health_score, 2),
            "status": "healthy" if self._stats.last_health_score > 0.8 else "degraded",
            "stats": self._stats.to_dict(),
            "last_reconnection_results": self._last_reconnection_results,
        }


# =============================================================================
# Standalone Tailscale Query (for P2P Orchestrator)
# =============================================================================


async def get_tailscale_status(timeout: float = 10.0) -> dict[str, bool]:
    """Query Tailscale status and return peer online status.

    This is a standalone function that can be used by the P2P orchestrator
    without the full discovery loop.

    Args:
        timeout: Timeout for tailscale CLI command in seconds

    Returns:
        Dict mapping Tailscale IP to online status {ip: is_online}
    """
    tailscale_path = shutil.which("tailscale")
    if not tailscale_path:
        logger.debug("Tailscale CLI not found")
        return {}

    try:
        proc = await asyncio.create_subprocess_exec(
            "tailscale",
            "status",
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )

        if proc.returncode != 0:
            logger.warning(f"Tailscale status failed: {stderr.decode()}")
            return {}

        data = json.loads(stdout.decode())

        # Extract peer IPs and online status
        result: dict[str, bool] = {}
        for peer_key, peer_data in data.get("Peer", {}).items():
            is_online = peer_data.get("Online", False)
            # Extract Tailscale IPs
            for ip in peer_data.get("TailscaleIPs", []):
                result[ip] = is_online

        return result

    except asyncio.TimeoutError:
        logger.warning(f"Tailscale status timed out after {timeout}s")
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Tailscale status JSON: {e}")
        return {}
    except FileNotFoundError:
        logger.debug("Tailscale command not found")
        return {}
    except Exception as e:
        logger.warning(f"Error querying Tailscale status: {e}")
        return {}

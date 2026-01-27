"""IPDiscoveryManager: Cloud and mesh IP discovery for cluster nodes.

January 2026: Extracted from p2p_orchestrator.py for better modularity.
Handles Tailscale, Vast.ai, and AWS IP discovery and reconnection.

This manager complements the P2P orchestrator by providing:
- Periodic IP refresh from cloud providers (Vast, AWS)
- Tailscale mesh network discovery
- Missing peer reconnection sweeps
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import aiohttp
    from scripts.p2p.models import PeerInfo

logger = logging.getLogger(__name__)

# Default P2P port
DEFAULT_PORT = 8770


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class IPDiscoveryConfig:
    """Configuration for IPDiscoveryManager.

    Attributes:
        vast_refresh_interval: Interval for Vast IP refresh (seconds)
        aws_refresh_interval: Interval for AWS IP refresh (seconds)
        tailscale_refresh_interval: Interval for Tailscale IP refresh (seconds)
        connection_timeout: Timeout for HTTP connections (seconds)
        subprocess_timeout: Timeout for subprocess commands (seconds)
    """

    vast_refresh_interval: float = 300.0  # 5 minutes
    aws_refresh_interval: float = 300.0  # 5 minutes
    tailscale_refresh_interval: float = 120.0  # 2 minutes
    connection_timeout: float = 5.0
    subprocess_timeout: float = 10.0


@dataclass
class IPDiscoveryStats:
    """Statistics for IPDiscoveryManager operations."""

    vast_updates: int = 0
    vast_errors: int = 0
    aws_updates: int = 0
    aws_errors: int = 0
    tailscale_updates: int = 0
    tailscale_errors: int = 0
    discovery_sweeps: int = 0
    reconnection_sweeps: int = 0
    peers_reconnected: int = 0
    last_vast_update: float = 0.0
    last_aws_update: float = 0.0
    last_tailscale_update: float = 0.0
    last_discovery_sweep: float = 0.0
    last_reconnection_sweep: float = 0.0


# ============================================================================
# Singleton management
# ============================================================================

_instance: IPDiscoveryManager | None = None


def get_ip_discovery_manager() -> IPDiscoveryManager | None:
    """Get the singleton IPDiscoveryManager instance."""
    return _instance


def set_ip_discovery_manager(manager: IPDiscoveryManager) -> None:
    """Set the singleton IPDiscoveryManager instance."""
    global _instance
    _instance = manager


def reset_ip_discovery_manager() -> None:
    """Reset the singleton IPDiscoveryManager instance (for testing)."""
    global _instance
    _instance = None


def create_ip_discovery_manager(
    config: IPDiscoveryConfig | None = None,
    orchestrator: Any | None = None,
) -> IPDiscoveryManager:
    """Create and register an IPDiscoveryManager instance.

    Args:
        config: Optional configuration
        orchestrator: P2P orchestrator reference (for callbacks)

    Returns:
        The created IPDiscoveryManager instance
    """
    manager = IPDiscoveryManager(config=config, orchestrator=orchestrator)
    set_ip_discovery_manager(manager)
    return manager


# ============================================================================
# Dynamic registry import (optional dependency)
# ============================================================================

try:
    from app.distributed.dynamic_host_registry import (
        DynamicHostRegistry,
        get_registry,
    )
    HAS_DYNAMIC_REGISTRY = True
except ImportError:
    HAS_DYNAMIC_REGISTRY = False
    get_registry = None  # type: ignore[assignment]


# ============================================================================
# IPDiscoveryManager
# ============================================================================


class IPDiscoveryManager:
    """Manager for cloud and mesh network IP discovery.

    This class handles:
    - Periodic IP refresh from Vast.ai API
    - Periodic IP refresh from AWS CLI
    - Tailscale mesh network IP discovery
    - One-shot peer discovery via Tailscale
    - Reconnection sweeps for missing peers
    """

    # Compute node hostname patterns for Tailscale discovery
    COMPUTE_PATTERNS = [
        "lambda-", "vast-", "gh200", "h100", "a100", "a10",
        "nebius-", "runpod-", "vultr-", "hetzner-",
    ]

    def __init__(
        self,
        config: IPDiscoveryConfig | None = None,
        orchestrator: Any | None = None,
    ):
        """Initialize IPDiscoveryManager.

        Args:
            config: Configuration for the manager
            orchestrator: P2P orchestrator reference (for callbacks)
        """
        self.config = config or IPDiscoveryConfig()
        self._orchestrator = orchestrator
        self._stats = IPDiscoveryStats()
        self._running = False

    @property
    def stats(self) -> IPDiscoveryStats:
        """Get current statistics."""
        return self._stats

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the P2P orchestrator reference.

        Called during orchestrator initialization.
        """
        self._orchestrator = orchestrator

    def start(self) -> None:
        """Mark manager as running (called by orchestrator)."""
        self._running = True

    def stop(self) -> None:
        """Mark manager as stopped."""
        self._running = False

    # ========================================================================
    # Force refresh
    # ========================================================================

    async def force_ip_refresh_all_sources(self) -> int:
        """Force immediate refresh of IPs from all CLI sources (Tailscale, Vast, AWS).

        Called when network partition is detected to aggressively discover
        alternative paths to reach peers.

        Returns:
            Total number of IPs updated across all sources
        """
        if not HAS_DYNAMIC_REGISTRY or get_registry is None:
            return 0

        registry = get_registry()
        total_updated = 0

        logger.info("Force-refreshing all IP sources for partition recovery...")

        # Refresh Tailscale first (most likely to help in partition)
        try:
            registry._last_tailscale_check = 0
            updated = await registry.update_tailscale_ips()
            if updated > 0:
                logger.info(f"Tailscale refresh: {updated} IPs updated")
                total_updated += updated
                self._stats.tailscale_updates += updated
        except Exception as e:  # noqa: BLE001
            logger.info(f"Tailscale refresh error: {e}")
            self._stats.tailscale_errors += 1

        # Refresh Vast IPs
        try:
            registry._last_vast_check = 0
            updated = await registry.update_vast_ips()
            if updated > 0:
                logger.info(f"Vast refresh: {updated} IPs updated")
                total_updated += updated
                self._stats.vast_updates += updated
        except Exception as e:  # noqa: BLE001
            logger.info(f"Vast refresh error: {e}")
            self._stats.vast_errors += 1

        # Refresh AWS IPs
        try:
            registry._last_aws_check = 0
            updated = await registry.update_aws_ips()
            if updated > 0:
                logger.info(f"AWS refresh: {updated} IPs updated")
                total_updated += updated
                self._stats.aws_updates += updated
        except Exception as e:  # noqa: BLE001
            logger.info(f"AWS refresh error: {e}")
            self._stats.aws_errors += 1

        if total_updated > 0:
            logger.info(f"Force refresh complete: {total_updated} total IPs updated")
        return total_updated

    # ========================================================================
    # Background IP update loops
    # ========================================================================

    async def vast_ip_update_loop(self) -> None:
        """Background loop to periodically refresh Vast instance connection info.

        Uses VAST_API_KEY when available, otherwise falls back to the `vastai`
        CLI if installed (see DynamicHostRegistry.update_vast_ips).
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        logger.info("Vast IP update loop started")
        registry = get_registry()

        while self._running:
            try:
                await asyncio.sleep(self.config.vast_refresh_interval)

                updated = await registry.update_vast_ips()
                if updated > 0:
                    logger.info(f"Updated {updated} Vast instance IPs from API")
                    self._stats.vast_updates += updated
                self._stats.last_vast_update = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.info(f"Vast IP update loop error: {e}")
                self._stats.vast_errors += 1
                await asyncio.sleep(60)

    async def aws_ip_update_loop(self) -> None:
        """Background loop to periodically refresh AWS instance connection info.

        Uses the `aws` CLI (see DynamicHostRegistry.update_aws_ips). No-op when
        no AWS instances are configured in distributed_hosts.yaml properties.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        logger.info("AWS IP update loop started")
        registry = get_registry()

        while self._running:
            try:
                await asyncio.sleep(self.config.aws_refresh_interval)

                updated = await registry.update_aws_ips()
                if updated > 0:
                    logger.info(f"Updated {updated} AWS instance IPs via CLI")
                    self._stats.aws_updates += updated
                self._stats.last_aws_update = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.info(f"AWS IP update loop error: {e}")
                self._stats.aws_errors += 1
                await asyncio.sleep(60)

    async def tailscale_ip_update_loop(self) -> None:
        """Background loop to discover and update Tailscale IPs for cluster nodes.

        Uses `tailscale status --json` to discover mesh network peers.
        Tailscale provides reliable connectivity even when public IPs change.
        """
        if not HAS_DYNAMIC_REGISTRY:
            return

        logger.info("Tailscale IP update loop started")
        registry = get_registry()

        while self._running:
            try:
                await asyncio.sleep(self.config.tailscale_refresh_interval)

                updated = await registry.update_tailscale_ips()
                if updated > 0:
                    logger.info(f"Updated {updated} node Tailscale IPs")
                    self._stats.tailscale_updates += updated
                self._stats.last_tailscale_update = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                logger.info(f"Tailscale IP update loop error: {e}")
                self._stats.tailscale_errors += 1
                await asyncio.sleep(60)

    # ========================================================================
    # Tailscale peer discovery
    # ========================================================================

    async def discover_tailscale_peers(
        self,
        peers_lock: Any,
        peers: dict[str, Any],
        send_heartbeat_callback: Callable[..., Any],
        run_subprocess_callback: Callable[..., Any],
    ) -> int:
        """One-shot Tailscale peer discovery for bootstrap fallback.

        Called when bootstrap from seeds fails. Discovers peers via
        `tailscale status --json` and attempts to connect.

        Args:
            peers_lock: Lock protecting the peers dict
            peers: Dict of peer_id -> PeerInfo
            send_heartbeat_callback: Async function to send heartbeat to peer
            run_subprocess_callback: Async function to run subprocess commands

        Returns:
            Number of new peers discovered
        """
        logger.info("Running one-shot Tailscale peer discovery...")
        self._stats.discovery_sweeps += 1
        self._stats.last_discovery_sweep = time.time()

        try:
            returncode, stdout, stderr = await run_subprocess_callback(
                ["tailscale", "status", "--json"],
                timeout=self.config.subprocess_timeout
            )
            if returncode != 0:
                logger.warning(f"Tailscale status failed: {stderr}")
                return 0

            ts_data = json.loads(stdout)
            ts_peers = ts_data.get("Peer", {})

            # Get current peer node_ids
            current_peers = set()
            with peers_lock:
                current_peers = {p.node_id for p in peers.values()}

            discovered = 0
            for peer_info in ts_peers.values():
                hostname = peer_info.get("HostName", "").lower()
                is_compute = any(pat in hostname for pat in self.COMPUTE_PATTERNS)
                if not is_compute:
                    continue

                # Get IP from TailscaleIPs (prefer IPv4)
                ts_ips = peer_info.get("TailscaleIPs", [])
                ipv4s = [ip for ip in ts_ips if "." in ip]
                if not ipv4s:
                    continue
                ip = ipv4s[0]

                # Skip if we already know this IP
                known = False
                with peers_lock:
                    for p in peers.values():
                        if getattr(p, "tailscale_ip", None) == ip or p.host == ip:
                            known = True
                            break
                if known:
                    continue

                # Try to connect via HTTP
                try:
                    import aiohttp
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.config.connection_timeout)
                    ) as session:
                        async with session.get(f"http://{ip}:{DEFAULT_PORT}/status") as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                node_id = data.get("node_id", hostname)
                                if node_id not in current_peers:
                                    logger.info(f"Discovered peer {node_id} via Tailscale at {ip}")
                                    await send_heartbeat_callback(ip, DEFAULT_PORT)
                                    discovered += 1
                except (Exception,) as e:  # noqa: BLE001
                    logger.debug(f"Tailscale discovery failed for {hostname}/{ip}: {type(e).__name__}")

            if discovered > 0:
                logger.info(f"Tailscale discovery: connected to {discovered} new peer(s)")
            else:
                logger.info("Tailscale discovery: no new peers found")

            return discovered

        except FileNotFoundError:
            logger.debug("Tailscale not installed on this node")
            return 0
        except json.JSONDecodeError as e:
            logger.warning(f"Tailscale status JSON parse error: {e}")
            return 0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Tailscale discovery error: {e}")
            return 0

    async def reconnect_missing_tailscale_peers(
        self,
        peers_lock: Any,
        peers: dict[str, Any],
        load_distributed_hosts_callback: Callable[[], dict[str, Any]],
        reconnect_peer_callback: Callable[..., Any],
        run_subprocess_callback: Callable[..., Any],
        node_id: str,
    ) -> int:
        """Force reconnect to peers online in Tailscale but missing from P2P mesh.

        January 2026: Fixes peer discovery asymmetry where P2P shows fewer peers
        than Tailscale. Performs a targeted reconnection sweep.

        Args:
            peers_lock: Lock protecting the peers dict
            peers: Dict of peer_id -> PeerInfo
            load_distributed_hosts_callback: Callback to load distributed_hosts.yaml
            reconnect_peer_callback: Async callback to reconnect to a discovered peer
            run_subprocess_callback: Async function to run subprocess commands
            node_id: This node's ID (for event emission)

        Returns:
            Number of peers successfully reconnected.
        """
        logger.info("[NetworkHealth] Running Tailscale-to-P2P reconnection sweep...")
        self._stats.reconnection_sweeps += 1
        self._stats.last_reconnection_sweep = time.time()

        try:
            # Get Tailscale status
            returncode, stdout, stderr = await run_subprocess_callback(
                ["tailscale", "status", "--json"],
                timeout=self.config.subprocess_timeout
            )
            if returncode != 0:
                logger.warning(f"Tailscale status failed: {stderr}")
                return 0

            ts_data = json.loads(stdout)
            ts_peers = ts_data.get("Peer", {})

            # Build map of Tailscale IPs to online status
            ts_online: dict[str, bool] = {}
            for peer_info in ts_peers.values():
                ts_ips = peer_info.get("TailscaleIPs", [])
                is_online = peer_info.get("Online", False)
                for ip in ts_ips:
                    if "." in ip:  # IPv4
                        ts_online[ip] = is_online

            # Get config hosts for IP-to-node mapping
            config_hosts = load_distributed_hosts_callback().get("hosts", {})
            ip_to_node = {
                h.get("tailscale_ip"): (name, h)
                for name, h in config_hosts.items()
                if h.get("tailscale_ip") and h.get("p2p_enabled", True)
            }

            # Get current P2P peer IDs
            p2p_peer_ids = set()
            with peers_lock:
                for peer_id, peer_info in peers.items():
                    is_alive = getattr(peer_info, "is_alive", lambda: True)
                    if (callable(is_alive) and is_alive()) or (not callable(is_alive) and is_alive):
                        p2p_peer_ids.add(peer_id)

            # Find and reconnect missing peers
            reconnected = 0
            attempted = 0

            for ts_ip, is_online in ts_online.items():
                if not is_online:
                    continue

                if ts_ip not in ip_to_node:
                    continue

                node_name, node_config = ip_to_node[ts_ip]

                # Skip if already connected in P2P
                if node_name in p2p_peer_ids:
                    continue

                attempted += 1
                port = node_config.get("p2p_port", DEFAULT_PORT)

                try:
                    success = await reconnect_peer_callback(node_name, ts_ip, port)
                    if success:
                        reconnected += 1
                        self._stats.peers_reconnected += 1
                        logger.debug(f"[NetworkHealth] Reconnected {node_name}")
                except (Exception,) as e:  # noqa: BLE001
                    logger.debug(f"[NetworkHealth] Failed to reconnect {node_name}: {e}")

            if reconnected > 0:
                logger.info(
                    f"[NetworkHealth] Reconnection sweep complete: "
                    f"{reconnected}/{attempted} peers reconnected"
                )
            elif attempted > 0:
                logger.warning(
                    f"[NetworkHealth] Reconnection sweep: "
                    f"0/{attempted} peers reconnected (check network connectivity)"
                )
            else:
                logger.debug("[NetworkHealth] No missing peers to reconnect")

            # Emit metric for observability
            try:
                from app.coordination.event_router import safe_emit_event
                safe_emit_event(
                    "P2P_DISCOVERY_GAP",
                    {
                        "tailscale_online": sum(1 for v in ts_online.values() if v),
                        "p2p_connected": len(p2p_peer_ids),
                        "gap": attempted,
                        "reconnected": reconnected,
                        "node_id": node_id,
                    },
                    source="ip_discovery_manager",
                )
            except ImportError:
                pass

            return reconnected

        except FileNotFoundError:
            logger.debug("[NetworkHealth] Tailscale not installed")
            return 0
        except json.JSONDecodeError as e:
            logger.warning(f"[NetworkHealth] Tailscale status JSON parse error: {e}")
            return 0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[NetworkHealth] Reconnection sweep error: {e}")
            return 0

    # ========================================================================
    # Health check
    # ========================================================================

    def health_check(self) -> dict[str, Any]:
        """Return health check information for DaemonManager integration.

        Returns:
            Dict with health status and statistics
        """
        now = time.time()

        # Check if updates are happening
        vast_stale = (now - self._stats.last_vast_update) > self.config.vast_refresh_interval * 3
        aws_stale = (now - self._stats.last_aws_update) > self.config.aws_refresh_interval * 3
        tailscale_stale = (now - self._stats.last_tailscale_update) > self.config.tailscale_refresh_interval * 3

        # Determine overall health
        if not HAS_DYNAMIC_REGISTRY:
            status = "DISABLED"
        elif vast_stale and aws_stale and tailscale_stale:
            status = "DEGRADED"
        else:
            status = "HEALTHY"

        return {
            "status": status,
            "has_dynamic_registry": HAS_DYNAMIC_REGISTRY,
            "running": self._running,
            "stats": {
                "vast_updates": self._stats.vast_updates,
                "vast_errors": self._stats.vast_errors,
                "aws_updates": self._stats.aws_updates,
                "aws_errors": self._stats.aws_errors,
                "tailscale_updates": self._stats.tailscale_updates,
                "tailscale_errors": self._stats.tailscale_errors,
                "discovery_sweeps": self._stats.discovery_sweeps,
                "reconnection_sweeps": self._stats.reconnection_sweeps,
                "peers_reconnected": self._stats.peers_reconnected,
            },
            "last_updates": {
                "vast": self._stats.last_vast_update,
                "aws": self._stats.last_aws_update,
                "tailscale": self._stats.last_tailscale_update,
                "discovery": self._stats.last_discovery_sweep,
                "reconnection": self._stats.last_reconnection_sweep,
            },
        }

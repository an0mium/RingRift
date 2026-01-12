"""Quorum Crisis Discovery Loop for P2P Orchestrator.

January 2026: Aggressive peer discovery triggered by quorum loss.

Problem: During quorum loss, the normal bootstrap interval (60s) is too slow for
quick recovery. The cluster can sit leaderless for minutes waiting for peer discovery.

Solution: When quorum drops to MINIMUM or LOST, this loop:
1. Reduces discovery interval from 60s to 10s (configurable via CRISIS_INTERVAL)
2. Probes all configured bootstrap seeds in parallel
3. Attempts direct TCP connections to known voter IPs
4. Emits PEER_DISCOVERY_EMERGENCY events for observability

The loop subscribes to quorum events:
- QUORUM_LOST, QUORUM_AT_RISK -> Enter crisis mode
- QUORUM_RESTORED -> Exit crisis mode

Usage:
    from scripts.p2p.loops import QuorumCrisisDiscoveryLoop

    crisis_loop = QuorumCrisisDiscoveryLoop(
        get_bootstrap_seeds=lambda: ["100.64.x.y:8770", ...],
        get_voter_endpoints=lambda: [("100.64.x.y", 8770), ...],
        probe_endpoint=lambda addr: orchestrator.probe_peer_health(addr),
        on_peer_discovered=lambda peer_id, addr: orchestrator.add_peer(peer_id, addr),
        emit_event=lambda event, data: orchestrator.emit_event(event, data),
    )
    await crisis_loop.run_forever()

Events:
    Subscribes to: QUORUM_LOST, QUORUM_AT_RISK, QUORUM_RESTORED
    Emits: PEER_DISCOVERY_EMERGENCY
"""

from __future__ import annotations

__all__ = [
    "QuorumCrisisDiscoveryLoop",
    "QuorumCrisisConfig",
    "CrisisStats",
]

import asyncio
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Protocol

from .base import BaseLoop

logger = logging.getLogger(__name__)


# =============================================================================
# Environment Variable Configuration
# =============================================================================

def _env_bool(name: str, default: bool = False) -> bool:
    """Read boolean from environment variable."""
    val = os.environ.get(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _env_float(name: str, default: float) -> float:
    """Read float from environment variable."""
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read int from environment variable."""
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


# =============================================================================
# Configuration Constants
# =============================================================================

# Feature flag
QUORUM_CRISIS_DISCOVERY_ENABLED = _env_bool("RINGRIFT_QUORUM_CRISIS_DISCOVERY", True)

# Crisis mode interval (aggressive)
QUORUM_CRISIS_INTERVAL = _env_float("RINGRIFT_QUORUM_CRISIS_INTERVAL", 10.0)

# Normal mode interval (disabled - we skip runs when not in crisis)
QUORUM_NORMAL_INTERVAL = _env_float("RINGRIFT_QUORUM_NORMAL_INTERVAL", 60.0)

# TCP probe timeout for voter probing
QUORUM_CRISIS_TCP_TIMEOUT = _env_float("RINGRIFT_QUORUM_CRISIS_TCP_TIMEOUT", 3.0)

# HTTP probe timeout for bootstrap seeds
QUORUM_CRISIS_HTTP_TIMEOUT = _env_float("RINGRIFT_QUORUM_CRISIS_HTTP_TIMEOUT", 5.0)

# Maximum parallel probes
QUORUM_CRISIS_MAX_PARALLEL = _env_int("RINGRIFT_QUORUM_CRISIS_MAX_PARALLEL", 10)


# =============================================================================
# Protocols for Callbacks
# =============================================================================

class ProbeCallback(Protocol):
    """Protocol for endpoint probing callback."""

    async def __call__(self, address: str) -> tuple[bool, str | None]:
        """Probe an endpoint and return (success, peer_id or error)."""
        ...


class DiscoveredCallback(Protocol):
    """Protocol for peer discovered callback."""

    async def __call__(self, peer_id: str, address: str) -> None:
        """Called when a peer is successfully discovered."""
        ...


class EventEmitCallback(Protocol):
    """Protocol for event emission callback."""

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event with the given type and data."""
        ...


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class QuorumCrisisConfig:
    """Configuration for quorum crisis discovery loop.

    January 2026: Created for aggressive peer discovery during quorum loss.
    """

    # Feature enablement
    enabled: bool = QUORUM_CRISIS_DISCOVERY_ENABLED

    # Timing
    crisis_interval: float = QUORUM_CRISIS_INTERVAL  # Interval during crisis
    normal_interval: float = QUORUM_NORMAL_INTERVAL  # Interval when healthy (used for skip logic)
    tcp_probe_timeout: float = QUORUM_CRISIS_TCP_TIMEOUT
    http_probe_timeout: float = QUORUM_CRISIS_HTTP_TIMEOUT

    # Parallelism
    max_parallel_probes: int = QUORUM_CRISIS_MAX_PARALLEL

    # Crisis exit cooldown (avoid flapping)
    cooldown_after_restore: float = 30.0  # Wait 30s after QUORUM_RESTORED before fully disabling

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.crisis_interval <= 0:
            raise ValueError("crisis_interval must be > 0")
        if self.tcp_probe_timeout <= 0:
            raise ValueError("tcp_probe_timeout must be > 0")
        if self.http_probe_timeout <= 0:
            raise ValueError("http_probe_timeout must be > 0")
        if self.max_parallel_probes < 1:
            raise ValueError("max_parallel_probes must be >= 1")


@dataclass
class CrisisStats:
    """Statistics for quorum crisis discovery operations."""

    # Run counts
    crisis_runs: int = 0
    skipped_runs: int = 0  # Skipped because not in crisis

    # Discovery results
    seeds_probed: int = 0
    seeds_successful: int = 0
    voters_tcp_probed: int = 0
    voters_tcp_successful: int = 0
    peers_discovered: int = 0

    # Crisis mode tracking
    crisis_entries: int = 0  # How many times we entered crisis mode
    crisis_duration_total: float = 0.0  # Total time spent in crisis mode
    last_crisis_entry: float = 0.0
    last_crisis_exit: float = 0.0

    # Events
    emergency_events_emitted: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "crisis_runs": self.crisis_runs,
            "skipped_runs": self.skipped_runs,
            "seeds_probed": self.seeds_probed,
            "seeds_successful": self.seeds_successful,
            "voters_tcp_probed": self.voters_tcp_probed,
            "voters_tcp_successful": self.voters_tcp_successful,
            "peers_discovered": self.peers_discovered,
            "crisis_entries": self.crisis_entries,
            "crisis_duration_total_seconds": self.crisis_duration_total,
            "emergency_events_emitted": self.emergency_events_emitted,
        }


# =============================================================================
# Main Loop Implementation
# =============================================================================

class QuorumCrisisDiscoveryLoop(BaseLoop):
    """Aggressive peer discovery loop triggered by quorum loss.

    January 2026: Created to speed up cluster recovery during quorum crisis.

    When quorum drops to MINIMUM or LOST, this loop:
    1. Accelerates discovery interval from 60s to 10s
    2. Probes all configured bootstrap seeds in parallel
    3. Attempts direct TCP connections to known voter IPs
    4. Emits PEER_DISCOVERY_EMERGENCY events for monitoring

    The loop subscribes to quorum events to toggle crisis mode:
    - QUORUM_LOST, QUORUM_AT_RISK -> Enter crisis mode
    - QUORUM_RESTORED -> Exit crisis mode (with cooldown)
    """

    def __init__(
        self,
        get_bootstrap_seeds: Callable[[], list[str]],
        get_voter_endpoints: Callable[[], list[tuple[str, int]]],
        probe_endpoint: ProbeCallback | Callable[[str], Coroutine[Any, Any, tuple[bool, str | None]]],
        on_peer_discovered: DiscoveredCallback | Callable[[str, str], Coroutine[Any, Any, None]],
        emit_event: EventEmitCallback | Callable[[str, dict[str, Any]], None],
        config: QuorumCrisisConfig | None = None,
    ):
        """Initialize quorum crisis discovery loop.

        Args:
            get_bootstrap_seeds: Returns list of bootstrap seed addresses ("host:port")
            get_voter_endpoints: Returns list of voter (host, port) tuples
            probe_endpoint: Async callback to probe an endpoint, returns (success, peer_id/error)
            on_peer_discovered: Async callback when a peer is discovered
            emit_event: Callback to emit events
            config: Loop configuration
        """
        self.config = config or QuorumCrisisConfig()
        super().__init__(
            name="quorum_crisis_discovery",
            interval=self.config.crisis_interval,  # Use crisis interval as base
            enabled=self.config.enabled,
        )

        # Callbacks
        self._get_bootstrap_seeds = get_bootstrap_seeds
        self._get_voter_endpoints = get_voter_endpoints
        self._probe_endpoint = probe_endpoint
        self._on_peer_discovered = on_peer_discovered
        self._emit_event = emit_event

        # Crisis mode state
        self._in_crisis_mode = False
        self._crisis_entry_time: float = 0.0
        self._cooldown_until: float = 0.0

        # Statistics
        self._crisis_stats = CrisisStats()

    # =========================================================================
    # Crisis Mode Management
    # =========================================================================

    def enter_crisis_mode(self, reason: str = "quorum_event") -> None:
        """Enter crisis mode for aggressive peer discovery.

        Called by external event handlers when QUORUM_LOST or QUORUM_AT_RISK.

        Args:
            reason: Why we're entering crisis mode
        """
        if self._in_crisis_mode:
            logger.debug(f"[QuorumCrisis] Already in crisis mode, reason: {reason}")
            return

        now = time.time()
        self._in_crisis_mode = True
        self._crisis_entry_time = now
        self._crisis_stats.crisis_entries += 1
        self._crisis_stats.last_crisis_entry = now

        logger.warning(
            f"[QuorumCrisis] ENTERING CRISIS MODE (reason: {reason}). "
            f"Discovery interval: {self.config.crisis_interval}s"
        )

        # Emit emergency event
        self._emit_emergency_event("crisis_entered", reason)

    def exit_crisis_mode(self, reason: str = "quorum_restored") -> None:
        """Exit crisis mode after quorum is restored.

        Called by external event handlers when QUORUM_RESTORED.

        Args:
            reason: Why we're exiting crisis mode
        """
        if not self._in_crisis_mode:
            logger.debug(f"[QuorumCrisis] Not in crisis mode, ignoring exit: {reason}")
            return

        now = time.time()
        crisis_duration = now - self._crisis_entry_time
        self._crisis_stats.crisis_duration_total += crisis_duration
        self._crisis_stats.last_crisis_exit = now

        # Set cooldown period to avoid flapping
        self._cooldown_until = now + self.config.cooldown_after_restore
        self._in_crisis_mode = False

        logger.info(
            f"[QuorumCrisis] EXITING CRISIS MODE (reason: {reason}). "
            f"Duration: {crisis_duration:.1f}s, cooldown until: {self.config.cooldown_after_restore}s"
        )

    @property
    def in_crisis_mode(self) -> bool:
        """Check if loop is in crisis mode."""
        return self._in_crisis_mode

    # =========================================================================
    # Event Subscription Helpers
    # =========================================================================

    def get_event_subscriptions(self) -> dict[str, Callable[[dict[str, Any]], None]]:
        """Return event type -> handler mappings for external subscription.

        The orchestrator should call these handlers when the corresponding
        events are received.

        Returns:
            Dict mapping event type strings to handler callables
        """
        return {
            "quorum_lost": lambda data: self.enter_crisis_mode("QUORUM_LOST"),
            "quorum_at_risk": lambda data: self.enter_crisis_mode("QUORUM_AT_RISK"),
            "quorum_restored": lambda data: self.exit_crisis_mode("QUORUM_RESTORED"),
            # Also handle DataEventType enum values
            "QUORUM_LOST": lambda data: self.enter_crisis_mode("QUORUM_LOST"),
            "QUORUM_AT_RISK": lambda data: self.enter_crisis_mode("QUORUM_AT_RISK"),
            "QUORUM_RESTORED": lambda data: self.exit_crisis_mode("QUORUM_RESTORED"),
        }

    # =========================================================================
    # BaseLoop Implementation
    # =========================================================================

    async def _on_start(self) -> None:
        """Log startup configuration."""
        if not self.config.enabled:
            logger.info("[QuorumCrisis] Loop disabled via configuration")
            return

        logger.info(
            f"[QuorumCrisis] Started (crisis_interval: {self.config.crisis_interval}s, "
            f"tcp_timeout: {self.config.tcp_probe_timeout}s, "
            f"http_timeout: {self.config.http_probe_timeout}s)"
        )

    async def _run_once(self) -> None:
        """Execute one discovery cycle.

        If not in crisis mode, this method returns early (no-op).
        In crisis mode, it performs aggressive parallel discovery.
        """
        if not self.config.enabled:
            return

        # Check cooldown period
        now = time.time()
        if now < self._cooldown_until:
            logger.debug("[QuorumCrisis] In cooldown period, skipping run")
            self._crisis_stats.skipped_runs += 1
            return

        # Skip if not in crisis mode
        if not self._in_crisis_mode:
            self._crisis_stats.skipped_runs += 1
            return

        self._crisis_stats.crisis_runs += 1
        logger.info(
            f"[QuorumCrisis] Crisis discovery run #{self._crisis_stats.crisis_runs}"
        )

        # Run parallel discovery
        try:
            await asyncio.gather(
                self._aggressive_bootstrap(),
                self._tcp_voter_probing(),
                return_exceptions=True,
            )
        except Exception as e:
            logger.warning(f"[QuorumCrisis] Error in discovery run: {e}")

        # Emit periodic emergency event for monitoring
        if self._crisis_stats.crisis_runs % 3 == 1:  # Every 3rd run
            self._emit_emergency_event(
                "crisis_active",
                f"run_{self._crisis_stats.crisis_runs}"
            )

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    async def _aggressive_bootstrap(self) -> None:
        """Probe all bootstrap seeds in parallel.

        January 2026: During quorum crisis, we don't wait for normal
        bootstrap intervals. Instead, we immediately probe all known seeds.
        """
        seeds = self._get_bootstrap_seeds()
        if not seeds:
            logger.debug("[QuorumCrisis] No bootstrap seeds configured")
            return

        logger.info(f"[QuorumCrisis] Probing {len(seeds)} bootstrap seeds in parallel")

        # Create probe tasks with semaphore for parallelism control
        semaphore = asyncio.Semaphore(self.config.max_parallel_probes)

        async def probe_with_limit(seed: str) -> tuple[str, bool, str | None]:
            async with semaphore:
                try:
                    success, result = await asyncio.wait_for(
                        self._probe_endpoint(seed),
                        timeout=self.config.http_probe_timeout,
                    )
                    return seed, success, result
                except asyncio.TimeoutError:
                    return seed, False, "timeout"
                except Exception as e:
                    return seed, False, str(e)

        # Run all probes
        tasks = [probe_with_limit(seed) for seed in seeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = 0
        for result in results:
            self._crisis_stats.seeds_probed += 1

            if isinstance(result, Exception):
                logger.debug(f"[QuorumCrisis] Seed probe exception: {result}")
                continue

            seed, success, peer_id_or_error = result
            if success and peer_id_or_error:
                successful += 1
                self._crisis_stats.seeds_successful += 1
                self._crisis_stats.peers_discovered += 1

                logger.info(
                    f"[QuorumCrisis] Discovered peer via bootstrap: {peer_id_or_error} at {seed}"
                )

                # Notify orchestrator
                try:
                    await self._on_peer_discovered(peer_id_or_error, seed)
                except Exception as e:
                    logger.warning(f"[QuorumCrisis] Error notifying peer discovered: {e}")

        logger.info(
            f"[QuorumCrisis] Bootstrap probe complete: {successful}/{len(seeds)} successful"
        )

    async def _tcp_voter_probing(self) -> None:
        """Attempt direct TCP connections to known voter IPs.

        January 2026: Fast TCP probes to detect if voters are reachable
        before attempting full HTTP probes. This catches cases where
        the peer is alive but the P2P server is slow to respond.
        """
        endpoints = self._get_voter_endpoints()
        if not endpoints:
            logger.debug("[QuorumCrisis] No voter endpoints configured")
            return

        logger.info(f"[QuorumCrisis] TCP probing {len(endpoints)} voter endpoints")

        # Create probe tasks with semaphore
        semaphore = asyncio.Semaphore(self.config.max_parallel_probes)

        async def tcp_probe_with_limit(
            host: str, port: int
        ) -> tuple[str, int, bool]:
            async with semaphore:
                success = await self._tcp_connectivity_check(host, port)
                return host, port, success

        # Run all TCP probes
        tasks = [tcp_probe_with_limit(host, port) for host, port in endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        reachable = 0
        for result in results:
            self._crisis_stats.voters_tcp_probed += 1

            if isinstance(result, Exception):
                logger.debug(f"[QuorumCrisis] TCP probe exception: {result}")
                continue

            host, port, success = result
            if success:
                reachable += 1
                self._crisis_stats.voters_tcp_successful += 1
                logger.debug(f"[QuorumCrisis] Voter reachable via TCP: {host}:{port}")

                # Do full HTTP probe for reachable voters
                try:
                    probe_success, peer_id = await asyncio.wait_for(
                        self._probe_endpoint(f"{host}:{port}"),
                        timeout=self.config.http_probe_timeout,
                    )
                    if probe_success and peer_id:
                        self._crisis_stats.peers_discovered += 1
                        logger.info(
                            f"[QuorumCrisis] Discovered voter: {peer_id} at {host}:{port}"
                        )
                        await self._on_peer_discovered(peer_id, f"{host}:{port}")
                except asyncio.TimeoutError:
                    logger.debug(f"[QuorumCrisis] HTTP probe timeout for {host}:{port}")
                except Exception as e:
                    logger.debug(f"[QuorumCrisis] HTTP probe error for {host}:{port}: {e}")

        logger.info(
            f"[QuorumCrisis] TCP voter probe complete: {reachable}/{len(endpoints)} reachable"
        )

    async def _tcp_connectivity_check(self, host: str, port: int) -> bool:
        """Check raw TCP connectivity to an endpoint.

        Args:
            host: Target hostname or IP
            port: Target port

        Returns:
            True if TCP connection succeeds
        """
        try:
            # Run blocking socket operation in thread pool
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._sync_tcp_check,
                    host,
                    port,
                ),
                timeout=self.config.tcp_probe_timeout,
            )
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    def _sync_tcp_check(self, host: str, port: int) -> bool:
        """Synchronous TCP connectivity check (runs in thread pool).

        Args:
            host: Target hostname or IP
            port: Target port

        Returns:
            True if TCP connection succeeds
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.tcp_probe_timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    # =========================================================================
    # Event Emission
    # =========================================================================

    def _emit_emergency_event(self, trigger: str, details: str) -> None:
        """Emit PEER_DISCOVERY_EMERGENCY event for observability.

        Args:
            trigger: What triggered the event (crisis_entered, crisis_active, etc.)
            details: Additional details
        """
        self._crisis_stats.emergency_events_emitted += 1

        event_data = {
            "trigger": trigger,
            "details": details,
            "crisis_runs": self._crisis_stats.crisis_runs,
            "peers_discovered": self._crisis_stats.peers_discovered,
            "crisis_duration": time.time() - self._crisis_entry_time if self._in_crisis_mode else 0,
            "timestamp": time.time(),
        }

        try:
            self._emit_event("PEER_DISCOVERY_EMERGENCY", event_data)
            logger.debug(f"[QuorumCrisis] Emitted PEER_DISCOVERY_EMERGENCY: {trigger}")
        except Exception as e:
            logger.warning(f"[QuorumCrisis] Failed to emit event: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_crisis_stats(self) -> CrisisStats:
        """Get current crisis discovery statistics."""
        return self._crisis_stats

    def get_stats_dict(self) -> dict[str, Any]:
        """Get statistics as a dictionary."""
        stats = self._crisis_stats.to_dict()
        stats["in_crisis_mode"] = self._in_crisis_mode
        stats["enabled"] = self.config.enabled
        return stats

    def health_check(self) -> dict[str, Any]:
        """Return health check information for DaemonManager integration.

        Returns:
            Health check result with is_healthy, status, and details
        """
        stats = self._crisis_stats

        # Determine health based on crisis mode effectiveness
        is_healthy = True
        status = "healthy"

        if self._in_crisis_mode:
            # In crisis mode, check if we're making progress
            if stats.crisis_runs > 10 and stats.peers_discovered == 0:
                status = "degraded"
            else:
                status = "crisis_active"

        return {
            "is_healthy": is_healthy,
            "status": status,
            "details": {
                "enabled": self.config.enabled,
                "in_crisis_mode": self._in_crisis_mode,
                "crisis_runs": stats.crisis_runs,
                "peers_discovered": stats.peers_discovered,
                "crisis_entries": stats.crisis_entries,
            },
        }

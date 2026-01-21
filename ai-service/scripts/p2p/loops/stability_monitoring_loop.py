"""Stability Monitoring Loop for P2P Network.

January 19, 2026 - P2P Network Stability Plan Phase 3:
    Dedicated loop that tracks mesh stability metrics to ensure
    20+ nodes remain connected for multi-hour operation.

This loop monitors:
- Total peer count vs target (20+ nodes)
- Per-provider health (online/total per provider)
- Peer churn rate (joins + leaves per hour)
- Leader stability (changes per hour)
- Relay health (healthy/unhealthy count)

Events Emitted:
- P2P_MESH_DEGRADED: When mesh coverage ratio < 0.5
- P2P_RELAY_SHORTAGE: When healthy relays < 3
- P2P_HIGH_CHURN: When peer churn exceeds threshold
- P2P_LEADER_UNSTABLE: When leader changes > 2/hour
- P2P_STABILITY_HEALTHY: When metrics are within target ranges

Usage:
    from scripts.p2p.loops.stability_monitoring_loop import StabilityMonitoringLoop

    loop = StabilityMonitoringLoop(
        get_peer_status=lambda: orchestrator.get_all_peers(),
        get_leader_id=lambda: orchestrator.leader_id,
        get_relay_health=lambda: orchestrator.get_relay_health(),
        emit_event=lambda event, data: orchestrator.emit_event(event, data),
    )
    await loop.run_forever()
"""
from __future__ import annotations

__all__ = [
    "StabilityMonitoringLoop",
    "StabilityConfig",
    "StabilityMetrics",
    "ProviderHealth",
]

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol

from .base import BaseLoop

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealth:
    """Health metrics for a single provider."""
    provider: str
    online_count: int
    total_count: int
    avg_peer_count: float = 0.0

    @property
    def online_ratio(self) -> float:
        """Ratio of online nodes to total nodes."""
        if self.total_count == 0:
            return 0.0
        return self.online_count / self.total_count


@dataclass
class StabilityMetrics:
    """Stability metrics for the P2P mesh network."""
    alive_peer_count: int
    target_peer_count: int = 20

    # Mesh health
    mesh_coverage_ratio: float = 0.0  # alive/target

    # Per-provider health
    provider_health: Dict[str, ProviderHealth] = field(default_factory=dict)

    # Stability over time (tracked externally)
    peer_churn_last_hour: int = 0  # Peers joined + left
    leader_changes_last_hour: int = 0  # Should be 0-2

    # Relay health
    healthy_relays: int = 0
    unhealthy_relays: int = 0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.target_peer_count > 0:
            self.mesh_coverage_ratio = self.alive_peer_count / self.target_peer_count

    @property
    def is_healthy(self) -> bool:
        """Check if all metrics are within healthy ranges."""
        return (
            self.mesh_coverage_ratio >= 0.5
            and self.healthy_relays >= 3
            and self.peer_churn_last_hour <= 10
            and self.leader_changes_last_hour <= 2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alive_peer_count": self.alive_peer_count,
            "target_peer_count": self.target_peer_count,
            "mesh_coverage_ratio": round(self.mesh_coverage_ratio, 3),
            "provider_health": {
                name: {
                    "online": ph.online_count,
                    "total": ph.total_count,
                    "online_ratio": round(ph.online_ratio, 3),
                    "avg_peer_count": round(ph.avg_peer_count, 1),
                }
                for name, ph in self.provider_health.items()
            },
            "peer_churn_last_hour": self.peer_churn_last_hour,
            "leader_changes_last_hour": self.leader_changes_last_hour,
            "healthy_relays": self.healthy_relays,
            "unhealthy_relays": self.unhealthy_relays,
            "is_healthy": self.is_healthy,
            "timestamp": self.timestamp,
        }


@dataclass
class StabilityConfig:
    """Configuration for stability monitoring."""
    # Monitoring interval
    check_interval: float = float(os.environ.get(
        "RINGRIFT_STABILITY_CHECK_INTERVAL", "60"
    ))

    # Target metrics
    target_peer_count: int = int(os.environ.get(
        "RINGRIFT_TARGET_PEER_COUNT", "20"
    ))
    min_healthy_relays: int = int(os.environ.get(
        "RINGRIFT_MIN_HEALTHY_RELAYS", "3"
    ))

    # Alert thresholds
    mesh_degraded_threshold: float = float(os.environ.get(
        "RINGRIFT_MESH_DEGRADED_THRESHOLD", "0.5"
    ))
    max_churn_per_hour: int = int(os.environ.get(
        "RINGRIFT_MAX_CHURN_PER_HOUR", "10"
    ))
    max_leader_changes_per_hour: int = int(os.environ.get(
        "RINGRIFT_MAX_LEADER_CHANGES_PER_HOUR", "2"
    ))

    # Event cooldowns (seconds) to prevent alert storms
    event_cooldown: float = float(os.environ.get(
        "RINGRIFT_STABILITY_EVENT_COOLDOWN", "300"
    ))


class PeerStatusProtocol(Protocol):
    """Protocol for peer status data."""
    node_id: str
    state: str  # 'alive', 'suspect', 'dead'
    last_heartbeat: float


class StabilityMonitoringLoop(BaseLoop):
    """Loop that tracks P2P mesh stability metrics.

    Monitors:
    - Peer count vs target (20+)
    - Per-provider health
    - Peer churn rate
    - Leader stability
    - Relay health

    Emits events when metrics fall outside target ranges.
    """

    def __init__(
        self,
        get_peer_status: Callable[[], Dict[str, Any]],
        get_leader_id: Callable[[], Optional[str]],
        get_relay_health: Callable[[], Dict[str, bool]] | None = None,
        emit_event: Callable[[str, Dict[str, Any]], None] | None = None,
        config: StabilityConfig | None = None,
        **kwargs,
    ):
        """Initialize the stability monitoring loop.

        Args:
            get_peer_status: Callback to get current peer status dict
            get_leader_id: Callback to get current leader ID
            get_relay_health: Callback to get relay health (relay_id -> is_healthy)
            emit_event: Callback to emit events
            config: Monitoring configuration
        """
        self.config = config or StabilityConfig()
        super().__init__(
            name="stability_monitoring",
            interval=self.config.check_interval,
            **kwargs,
        )

        self._get_peer_status = get_peer_status
        self._get_leader_id = get_leader_id
        self._get_relay_health = get_relay_health
        self._emit_event = emit_event

        # Historical tracking for churn/stability
        self._peer_history: deque[set[str]] = deque(maxlen=60)  # 60 samples = 1 hour at 1min interval
        self._leader_history: deque[str] = deque(maxlen=60)
        self._last_peer_set: set[str] = set()
        self._last_leader_id: Optional[str] = None

        # Event cooldowns
        self._last_event_times: Dict[str, float] = {}

        # Current metrics
        self._current_metrics: Optional[StabilityMetrics] = None

    async def _run_once(self) -> None:
        """Collect and evaluate stability metrics."""
        try:
            metrics = await self._collect_metrics()
            self._current_metrics = metrics

            # Log current state
            logger.info(
                f"Stability: {metrics.alive_peer_count}/{metrics.target_peer_count} peers "
                f"({metrics.mesh_coverage_ratio:.0%}), "
                f"churn={metrics.peer_churn_last_hour}/hr, "
                f"leader_changes={metrics.leader_changes_last_hour}/hr, "
                f"relays={metrics.healthy_relays}/{metrics.healthy_relays + metrics.unhealthy_relays}"
            )

            # Emit alerts as needed
            await self._emit_alerts(metrics)

        except Exception as e:
            logger.error(f"Error collecting stability metrics: {e}")
            raise

    async def _collect_metrics(self) -> StabilityMetrics:
        """Collect current stability metrics."""
        # Get peer status
        peers = self._get_peer_status()
        current_peer_set = set()
        provider_counts: Dict[str, Dict[str, int]] = {}  # provider -> {online, total}
        provider_peer_sums: Dict[str, float] = {}  # provider -> sum of peer counts

        for peer_id, peer_info in peers.items():
            # Extract provider from node ID
            provider = self._extract_provider(peer_id)

            if provider not in provider_counts:
                provider_counts[provider] = {"online": 0, "total": 0}
                provider_peer_sums[provider] = 0.0

            provider_counts[provider]["total"] += 1

            # Check if peer is alive
            state = peer_info.get("state", "unknown") if isinstance(peer_info, dict) else getattr(peer_info, "state", "unknown")
            if state in ("alive", "suspect"):
                current_peer_set.add(peer_id)
                provider_counts[provider]["online"] += 1

                # Track peer's own peer count if available
                peer_count = peer_info.get("peer_count", 0) if isinstance(peer_info, dict) else getattr(peer_info, "peer_count", 0)
                provider_peer_sums[provider] += peer_count

        # Build provider health
        provider_health = {}
        for provider, counts in provider_counts.items():
            online = counts["online"]
            avg_peers = provider_peer_sums[provider] / online if online > 0 else 0.0
            provider_health[provider] = ProviderHealth(
                provider=provider,
                online_count=online,
                total_count=counts["total"],
                avg_peer_count=avg_peers,
            )

        # Track peer churn
        self._peer_history.append(current_peer_set)
        churn = self._calculate_churn()

        # Track leader changes
        current_leader = self._get_leader_id()
        if current_leader != self._last_leader_id:
            self._leader_history.append(current_leader or "unknown")
            self._last_leader_id = current_leader
        leader_changes = self._calculate_leader_changes()

        # Get relay health
        healthy_relays = 0
        unhealthy_relays = 0
        if self._get_relay_health:
            try:
                relay_health = self._get_relay_health()
                for relay_id, is_healthy in relay_health.items():
                    if is_healthy:
                        healthy_relays += 1
                    else:
                        unhealthy_relays += 1
            except Exception as e:
                logger.warning(f"Failed to get relay health: {e}")

        self._last_peer_set = current_peer_set

        return StabilityMetrics(
            alive_peer_count=len(current_peer_set),
            target_peer_count=self.config.target_peer_count,
            provider_health=provider_health,
            peer_churn_last_hour=churn,
            leader_changes_last_hour=leader_changes,
            healthy_relays=healthy_relays,
            unhealthy_relays=unhealthy_relays,
        )

    def _extract_provider(self, node_id: str) -> str:
        """Extract provider name from node ID."""
        providers = ["lambda", "vast", "nebius", "runpod", "vultr", "hetzner", "mac"]
        for provider in providers:
            if node_id.lower().startswith(provider):
                return provider
        return "other"

    def _calculate_churn(self) -> int:
        """Calculate peer churn over the last hour."""
        if len(self._peer_history) < 2:
            return 0

        churn = 0
        prev_set = None
        for peer_set in self._peer_history:
            if prev_set is not None:
                joined = len(peer_set - prev_set)
                left = len(prev_set - peer_set)
                churn += joined + left
            prev_set = peer_set

        return churn

    def _calculate_leader_changes(self) -> int:
        """Calculate leader changes over the last hour."""
        if len(self._leader_history) < 2:
            return 0

        changes = 0
        prev_leader = None
        for leader in self._leader_history:
            if prev_leader is not None and leader != prev_leader:
                changes += 1
            prev_leader = leader

        return changes

    async def _emit_alerts(self, metrics: StabilityMetrics) -> None:
        """Emit alerts based on metrics."""
        if not self._emit_event:
            return

        now = time.time()

        # Check mesh degradation
        if metrics.mesh_coverage_ratio < self.config.mesh_degraded_threshold:
            await self._emit_with_cooldown(
                "P2P_MESH_DEGRADED",
                {
                    "alive": metrics.alive_peer_count,
                    "target": metrics.target_peer_count,
                    "coverage_ratio": metrics.mesh_coverage_ratio,
                    "provider_health": {
                        k: {"online": v.online_count, "total": v.total_count}
                        for k, v in metrics.provider_health.items()
                    },
                },
            )

        # Check relay shortage
        if metrics.healthy_relays < self.config.min_healthy_relays:
            await self._emit_with_cooldown(
                "P2P_RELAY_SHORTAGE",
                {
                    "healthy": metrics.healthy_relays,
                    "unhealthy": metrics.unhealthy_relays,
                    "required": self.config.min_healthy_relays,
                },
            )

        # Check high churn
        if metrics.peer_churn_last_hour > self.config.max_churn_per_hour:
            await self._emit_with_cooldown(
                "P2P_HIGH_CHURN",
                {
                    "churn": metrics.peer_churn_last_hour,
                    "threshold": self.config.max_churn_per_hour,
                },
            )

        # Check leader instability
        if metrics.leader_changes_last_hour > self.config.max_leader_changes_per_hour:
            await self._emit_with_cooldown(
                "P2P_LEADER_UNSTABLE",
                {
                    "changes": metrics.leader_changes_last_hour,
                    "threshold": self.config.max_leader_changes_per_hour,
                },
            )

        # Emit healthy status if all checks pass
        if metrics.is_healthy:
            await self._emit_with_cooldown(
                "P2P_STABILITY_HEALTHY",
                metrics.to_dict(),
            )

    async def _emit_with_cooldown(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event with cooldown to prevent alert storms."""
        now = time.time()
        last_emit = self._last_event_times.get(event_type, 0)

        if now - last_emit >= self.config.event_cooldown:
            try:
                self._emit_event(event_type, data)
                self._last_event_times[event_type] = now
                logger.info(f"Emitted {event_type}: {data}")
            except Exception as e:
                logger.error(f"Failed to emit {event_type}: {e}")

    def get_current_metrics(self) -> Optional[StabilityMetrics]:
        """Get the most recent stability metrics."""
        return self._current_metrics

    def health_check(self) -> Dict[str, Any]:
        """Return health check for DaemonManager integration."""
        metrics = self._current_metrics
        if metrics is None:
            return {
                "healthy": True,
                "details": {
                    "status": "initializing",
                    "loop_stats": self._stats.to_dict() if hasattr(self, "_stats") else {},
                },
            }

        return {
            "healthy": metrics.is_healthy,
            "details": {
                "alive_peers": metrics.alive_peer_count,
                "target_peers": metrics.target_peer_count,
                "mesh_coverage": round(metrics.mesh_coverage_ratio, 3),
                "peer_churn_per_hour": metrics.peer_churn_last_hour,
                "leader_changes_per_hour": metrics.leader_changes_last_hour,
                "healthy_relays": metrics.healthy_relays,
                "loop_stats": self._stats.to_dict() if hasattr(self, "_stats") else {},
            },
        }

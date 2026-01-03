"""
P2P relay transport implementation.

Tier 4 (RELAY): Route through P2P leader or other relay nodes.

Jan 2, 2026: Enhanced with per-relay health tracking and health-based selection
to avoid single relay bottleneck and cascade failures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import aiohttp
    from aiohttp import ClientTimeout
except ImportError:
    aiohttp = None  # type: ignore
    ClientTimeout = None  # type: ignore

from ..transport_cascade import BaseTransport, TransportResult, TransportTier

logger = logging.getLogger(__name__)


# =============================================================================
# Per-Relay Health Tracking (Jan 2, 2026)
# =============================================================================


@dataclass
class RelayHealth:
    """Health tracking for a single relay node.

    Jan 2, 2026: Added to enable health-based relay selection and avoid
    cascade failures when a relay becomes unhealthy.
    """

    node_id: str
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    total_latency_ms: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_error: str = ""

    # Circuit breaker settings
    failure_threshold: int = 3  # Failures before marking unhealthy
    recovery_timeout: float = 60.0  # Seconds before retrying unhealthy relay

    @property
    def is_healthy(self) -> bool:
        """Check if relay is considered healthy."""
        if self.consecutive_failures >= self.failure_threshold:
            # Check if recovery timeout has passed
            if self.last_failure_time > 0:
                elapsed = time.time() - self.last_failure_time
                if elapsed < self.recovery_timeout:
                    return False
        return True

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total = self.successes + self.failures
        if total == 0:
            return 1.0  # Assume healthy if no data
        return self.successes / total

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes

    @property
    def health_score(self) -> float:
        """Calculate health score for relay selection (higher = better).

        Score factors:
        - Success rate (0-1) weighted by 0.6
        - Recency of success (0-1) weighted by 0.3
        - Low latency bonus (0-1) weighted by 0.1
        """
        # Success rate component
        success_component = self.success_rate * 0.6

        # Recency component (how recently did we succeed?)
        if self.last_success_time > 0:
            since_success = time.time() - self.last_success_time
            # Decay over 5 minutes
            recency = max(0.0, 1.0 - (since_success / 300.0))
        else:
            recency = 0.5  # Unknown - assume moderate
        recency_component = recency * 0.3

        # Latency component (lower is better, normalize to 100-2000ms range)
        if self.avg_latency_ms > 0:
            latency_score = max(0.0, 1.0 - (self.avg_latency_ms - 100) / 1900)
        else:
            latency_score = 0.5  # Unknown
        latency_component = latency_score * 0.1

        return success_component + recency_component + latency_component

    def record_success(self, latency_ms: float) -> None:
        """Record a successful relay request."""
        self.successes += 1
        self.consecutive_failures = 0
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()

    def record_failure(self, error: str) -> None:
        """Record a failed relay request."""
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error = error


class P2PRelayTransport(BaseTransport):
    """
    Relay transport through P2P leader or relay nodes.

    Tier 4 (RELAY): For nodes that can't be reached directly.
    The payload is sent to a relay node which forwards to the target.

    Jan 2, 2026: Enhanced with per-relay health tracking. Relays are sorted
    by health score before use, avoiding cascade failures from unhealthy relays.
    """

    name = "p2p_relay"
    tier = TransportTier.TIER_4_RELAY

    def __init__(
        self,
        relay_nodes: list[str] | None = None,
        port: int = 8770,
        timeout: float = 20.0,
    ):
        self._relay_nodes = relay_nodes or []
        self._port = port
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._leader_node: str | None = None
        # Per-relay health tracking (Jan 2, 2026)
        self._relay_health: dict[str, RelayHealth] = {}

    def set_leader_node(self, leader_node: str) -> None:
        """Set the current P2P leader for relay."""
        self._leader_node = leader_node

    def add_relay_node(self, node: str) -> None:
        """Add a relay node."""
        if node not in self._relay_nodes:
            self._relay_nodes.append(node)

    def _get_or_create_relay_health(self, node_id: str) -> RelayHealth:
        """Get or create health tracking for a relay node."""
        if node_id not in self._relay_health:
            self._relay_health[node_id] = RelayHealth(node_id=node_id)
        return self._relay_health[node_id]

    def _get_sorted_relays(self, target: str) -> list[str]:
        """Get relay list sorted by health score (best first).

        Jan 2, 2026: Replaces simple leader-first ordering with health-based
        sorting. Healthy relays with good success rates are tried first.
        """
        # Build candidate list: leader + configured relays (excluding target)
        candidates = []
        if self._leader_node and self._leader_node != target:
            candidates.append(self._leader_node)
        candidates.extend([r for r in self._relay_nodes if r != target and r not in candidates])

        if not candidates:
            return []

        # Filter to healthy relays first
        healthy_relays = []
        unhealthy_relays = []

        for node_id in candidates:
            health = self._get_or_create_relay_health(node_id)
            if health.is_healthy:
                healthy_relays.append((node_id, health.health_score))
            else:
                # Keep unhealthy relays as fallback (may have recovered)
                unhealthy_relays.append((node_id, health.health_score))

        # Sort healthy relays by score (descending), keep unhealthy as fallback
        healthy_relays.sort(key=lambda x: x[1], reverse=True)
        unhealthy_relays.sort(key=lambda x: x[1], reverse=True)

        return [node_id for node_id, _ in healthy_relays] + [node_id for node_id, _ in unhealthy_relays]

    def get_relay_health_summary(self) -> dict[str, Any]:
        """Get summary of all relay health states.

        Returns dict with overall stats and per-relay details.
        """
        total = len(self._relay_health)
        healthy = sum(1 for h in self._relay_health.values() if h.is_healthy)
        return {
            "total_relays": total,
            "healthy_relays": healthy,
            "unhealthy_relays": total - healthy,
            "relays": {
                node_id: {
                    "is_healthy": h.is_healthy,
                    "health_score": round(h.health_score, 3),
                    "success_rate": round(h.success_rate, 3),
                    "avg_latency_ms": round(h.avg_latency_ms, 1),
                    "consecutive_failures": h.consecutive_failures,
                    "last_error": h.last_error,
                }
                for node_id, h in self._relay_health.items()
            },
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def send(self, target: str, payload: bytes) -> TransportResult:
        """Send payload via relay.

        Jan 2, 2026: Enhanced to use health-based relay selection. Healthy relays
        with good success rates are tried first, reducing cascade failures.
        """
        if aiohttp is None:
            return self._make_result(
                success=False, latency_ms=0, error="aiohttp not installed"
            )

        # Get relays sorted by health (healthy first, best scores first)
        relay_list = self._get_sorted_relays(target)

        if not relay_list:
            return self._make_result(
                success=False,
                latency_ms=0,
                error="No relay nodes available",
            )

        # Try each relay in health order
        errors = []
        for relay in relay_list:
            result = await self._send_via_relay(relay, target, payload)
            if result.success:
                return result
            errors.append(f"{relay}: {result.error}")

        return self._make_result(
            success=False,
            latency_ms=0,
            error=f"All relays failed: {'; '.join(errors)}",
        )

    async def _send_via_relay(
        self, relay: str, target: str, payload: bytes
    ) -> TransportResult:
        """Send payload through a specific relay node.

        Jan 2, 2026: Records success/failure in per-relay health tracking.
        """
        # Relay endpoint expects target in header
        url = f"http://{relay}:{self._port}/relay/forward"
        start_time = time.time()
        health = self._get_or_create_relay_health(relay)

        try:
            session = await self._get_session()
            async with session.post(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Relay-Target": target,
                },
            ) as resp:
                latency_ms = (time.time() - start_time) * 1000
                response_data = await resp.read()

                if resp.status == 200:
                    health.record_success(latency_ms)
                    return self._make_result(
                        success=True,
                        latency_ms=latency_ms,
                        response=response_data,
                        relay_node=relay,
                    )
                else:
                    error_msg = f"Relay HTTP {resp.status}"
                    health.record_failure(error_msg)
                    return self._make_result(
                        success=False,
                        latency_ms=latency_ms,
                        error=error_msg,
                    )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            health.record_failure(error_msg)
            return self._make_result(
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=error_msg,
            )

    async def is_available(self, target: str) -> bool:
        """Check if relay transport is available."""
        # Available if we have any relay nodes
        return bool(self._leader_node) or bool(self._relay_nodes)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

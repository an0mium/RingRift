"""Centralized provider-specific timeout multipliers.

January 16, 2026: Created to consolidate scattered timeout configurations.

Previously, provider timeout multipliers were defined in 3 places:
- scripts/p2p/loops/loop_constants.py (LoopTimeouts.get_provider_multiplier)
- scripts/p2p/loops/peer_recovery_loop.py (PROVIDER_PROBE_TIMEOUTS)
- scripts/p2p/connection_pool.py (PROVIDER_TIMEOUT_MULTIPLIERS)

This module provides a single source of truth for provider-aware timeouts,
reducing technical debt and ensuring consistent behavior across all P2P
operations.

Usage:
    from app.config.provider_timeouts import ProviderTimeouts

    # Get multiplier for a node
    mult = ProviderTimeouts.get_multiplier("lambda-gh200-1")  # Returns 2.0

    # Get adjusted timeout
    timeout = ProviderTimeouts.get_timeout("vast-12345", base_timeout=10.0)  # Returns 25.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderTimeouts:
    """Centralized provider-aware timeout configuration.

    Timeout multipliers are based on observed network characteristics:
    - vast: Consumer networks with high variance, NAT traversal issues
    - lambda: NAT-blocked nodes require relay hops, adding latency
    - runpod: Container networking overhead
    - nebius: Stable cloud infrastructure
    - vultr: vGPU shared infrastructure
    - hetzner: Bare metal, direct connectivity
    - local: Local machines on same network

    The multipliers are applied to base timeouts (e.g., 8.0s health check)
    to get provider-adjusted timeouts (e.g., 20.0s for Vast.ai).
    """

    # Provider-specific timeout multipliers
    # Higher values = longer timeouts for slower/less reliable networks
    # Jan 25, 2026: Changed local/mac from 0.5 to 1.0 to match cluster nodes
    # A 0.5x multiplier caused 108s local timeout vs 180-360s cluster,
    # triggering timeout disagreement retirements (ratio > 2.0 = instant retire)
    MULTIPLIERS: ClassVar[dict[str, float]] = {
        "vast": 2.5,       # Consumer networks, high variance, NAT issues
        "lambda": 2.0,     # NAT-blocked GH200s, relay latency
        "runpod": 2.0,     # Container networking overhead
        "nebius": 1.5,     # Stable cloud, some variance
        "vultr": 1.5,      # vGPU shared infrastructure
        "hetzner": 1.0,    # Bare metal, direct IP
        "local": 1.0,      # Local machines (was 0.5, caused disagreement)
        "mac": 1.0,        # Local Macs (was 0.5, caused disagreement)
    }

    # Default multiplier for unknown providers (conservative)
    DEFAULT_MULTIPLIER: ClassVar[float] = 1.2

    # Provider-specific probe timeouts (absolute values in seconds)
    # Used when base timeout isn't appropriate (e.g., peer recovery probes)
    PROBE_TIMEOUTS: ClassVar[dict[str, float]] = {
        "vast": 35.0,      # Consumer networks need longer probes
        "lambda": 25.0,    # NAT relay adds latency
        "runpod": 30.0,    # Container networking
        "nebius": 20.0,    # Stable but variable
        "vultr": 20.0,     # vGPU overhead
        "hetzner": 15.0,   # Bare metal, fast
        "local": 10.0,     # Local machines
        "mac": 10.0,       # Local Macs
    }

    DEFAULT_PROBE_TIMEOUT: ClassVar[float] = 20.0

    @classmethod
    def extract_provider(cls, node_id: str) -> str:
        """Extract provider name from node_id.

        Node IDs follow pattern: {provider}-{identifier}
        Examples:
            lambda-gh200-1 -> lambda
            vast-29031159 -> vast
            runpod-h100 -> runpod
            mac-studio -> mac
            local-mac -> local

        Args:
            node_id: Node identifier string

        Returns:
            Provider name in lowercase, or "unknown" if not recognized
        """
        if not node_id:
            return "unknown"

        node_lower = node_id.lower()

        # Check for known provider prefixes
        for provider in cls.MULTIPLIERS:
            if node_lower.startswith(provider):
                return provider

        # Special cases
        if "mac" in node_lower or "local" in node_lower:
            return "local"

        return "unknown"

    @classmethod
    def get_multiplier(cls, node_id: str) -> float:
        """Get timeout multiplier for a node.

        Args:
            node_id: Node identifier (e.g., "lambda-gh200-1")

        Returns:
            Multiplier to apply to base timeouts (e.g., 2.0 for Lambda)
        """
        provider = cls.extract_provider(node_id)
        return cls.MULTIPLIERS.get(provider, cls.DEFAULT_MULTIPLIER)

    @classmethod
    def get_timeout(cls, node_id: str, base_timeout: float) -> float:
        """Get provider-adjusted timeout.

        Args:
            node_id: Node identifier
            base_timeout: Base timeout in seconds

        Returns:
            Adjusted timeout (base_timeout * multiplier)

        Example:
            >>> ProviderTimeouts.get_timeout("lambda-gh200-1", 10.0)
            20.0  # 10.0 * 2.0 multiplier
        """
        multiplier = cls.get_multiplier(node_id)
        return base_timeout * multiplier

    @classmethod
    def get_probe_timeout(cls, node_id: str) -> float:
        """Get absolute probe timeout for a node.

        Use this when you need a fixed timeout rather than a multiplied one.
        Useful for peer recovery probes and health checks.

        Args:
            node_id: Node identifier

        Returns:
            Probe timeout in seconds
        """
        provider = cls.extract_provider(node_id)
        return cls.PROBE_TIMEOUTS.get(provider, cls.DEFAULT_PROBE_TIMEOUT)

    @classmethod
    def get_connection_timeout(cls, node_id: str, base_timeout: float = 30.0) -> float:
        """Get connection timeout for establishing new connections.

        Args:
            node_id: Node identifier
            base_timeout: Base connection timeout (default: 30s)

        Returns:
            Adjusted connection timeout
        """
        return cls.get_timeout(node_id, base_timeout)

    @classmethod
    def get_health_check_timeout(cls, node_id: str, base_timeout: float = 8.0) -> float:
        """Get health check timeout for a node.

        Args:
            node_id: Node identifier
            base_timeout: Base health check timeout (default: 8s)

        Returns:
            Adjusted health check timeout
        """
        return cls.get_timeout(node_id, base_timeout)


# Convenience functions for common operations
def get_provider_multiplier(node_id: str) -> float:
    """Get timeout multiplier for a node. Convenience wrapper."""
    return ProviderTimeouts.get_multiplier(node_id)


def get_provider_timeout(node_id: str, base_timeout: float) -> float:
    """Get provider-adjusted timeout. Convenience wrapper."""
    return ProviderTimeouts.get_timeout(node_id, base_timeout)


def get_provider_probe_timeout(node_id: str) -> float:
    """Get absolute probe timeout for a node. Convenience wrapper."""
    return ProviderTimeouts.get_probe_timeout(node_id)

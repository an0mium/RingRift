"""Unified Node Identity System.

Single source of truth for node identification across the cluster.
Resolves identity using multiple sources with clear priority order.

Usage:
    from app.config.node_identity import NodeIdentity, get_node_identity

    # Get canonical node identity
    identity = get_node_identity()
    print(f"This node is: {identity.canonical_id}")
    print(f"Tailscale IP: {identity.tailscale_ip}")

Priority Order:
    1. RINGRIFT_NODE_ID environment variable (explicit override)
    2. Hostname match against distributed_hosts.yaml
    3. Tailscale IP match against distributed_hosts.yaml

This module solves the identity mismatch problem where:
- socket.gethostname() returns "ip-192-222-57-210" (Lambda cloud hostname)
- Config expects "lambda-gh200-1" (human-readable name)
- P2P voters use config names, causing quorum failures
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

__all__ = [
    "NodeIdentity",
    "IdentityError",
    "get_node_identity",
    "get_tailscale_ip",
    "resolve_node_id",
]

logger = logging.getLogger(__name__)


class IdentityError(Exception):
    """Raised when node identity cannot be resolved.

    This is a fail-fast error that should stop the process rather than
    allow it to run with an unknown identity that could cause data
    corruption or cluster instability.
    """

    pass


@dataclass
class NodeIdentity:
    """Canonical node identity - single source of truth.

    Attributes:
        canonical_id: Config key from distributed_hosts.yaml (authoritative)
        tailscale_ip: Tailscale VPN IP if available
        ssh_host: SSH host (may be public IP or Tailscale IP)
        hostname: System hostname from socket.gethostname()
        role: Node role from config (coordinator, gpu_training, etc.)
        resolution_method: How the identity was resolved (env_var, hostname, tailscale)
    """

    canonical_id: str
    tailscale_ip: str | None = None
    ssh_host: str | None = None
    hostname: str = field(default_factory=socket.gethostname)
    role: str | None = None
    resolution_method: str = "unknown"

    @classmethod
    def resolve(cls, config: dict[str, Any] | None = None) -> NodeIdentity:
        """Resolve identity with fail-fast validation.

        Args:
            config: Optional pre-loaded config dict. If None, loads from YAML.

        Returns:
            Resolved NodeIdentity

        Raises:
            IdentityError: If identity cannot be resolved
        """
        # Load config if not provided
        if config is None:
            config = _load_hosts_config()

        hosts = config.get("hosts", {})
        hostname = socket.gethostname()

        # Priority 1: Explicit env var
        env_id = os.environ.get("RINGRIFT_NODE_ID")
        if env_id:
            if env_id in hosts:
                return cls._from_config(env_id, hosts, "env_var")

            # Env var set but not in config - this is a configuration error
            logger.warning(
                f"RINGRIFT_NODE_ID={env_id} not in config, "
                "trying other resolution methods"
            )

        # Priority 2: Hostname match (direct config key match)
        if hostname in hosts:
            return cls._from_config(hostname, hosts, "hostname")

        # Priority 3: Hostname pattern match (e.g., "ip-192-222-57-210" patterns)
        for node_id, node_cfg in hosts.items():
            if isinstance(node_cfg, dict):
                # Check ssh_host match
                if node_cfg.get("ssh_host") == hostname:
                    return cls._from_config(node_id, hosts, "ssh_host")

        # Priority 4: Tailscale IP match
        ts_ip = get_tailscale_ip()
        if ts_ip:
            for node_id, node_cfg in hosts.items():
                if isinstance(node_cfg, dict):
                    if node_cfg.get("tailscale_ip") == ts_ip:
                        return cls._from_config(node_id, hosts, "tailscale")

        # Priority 5: If we have an env var that's not in config, use it anyway
        # This allows new nodes to join before config is updated
        if env_id:
            logger.warning(
                f"Using RINGRIFT_NODE_ID={env_id} despite not being in config"
            )
            return cls(
                canonical_id=env_id,
                tailscale_ip=ts_ip,
                hostname=hostname,
                resolution_method="env_var_unverified",
            )

        # Cannot resolve - fail fast with helpful message
        node_ids = list(hosts.keys()) if hosts else []
        raise IdentityError(
            f"Cannot resolve node identity.\n"
            f"Hostname: {hostname}\n"
            f"Tailscale IP: {ts_ip}\n"
            f"Known nodes: {node_ids[:10]}{'...' if len(node_ids) > 10 else ''}\n\n"
            f"Fix: Set RINGRIFT_NODE_ID to one of the known node IDs,\n"
            f"or add this node to distributed_hosts.yaml"
        )

    @classmethod
    def _from_config(
        cls, node_id: str, hosts: dict[str, Any], method: str
    ) -> NodeIdentity:
        """Create NodeIdentity from config entry."""
        node_cfg = hosts.get(node_id, {})
        if not isinstance(node_cfg, dict):
            node_cfg = {}

        return cls(
            canonical_id=node_id,
            tailscale_ip=node_cfg.get("tailscale_ip"),
            ssh_host=node_cfg.get("ssh_host"),
            hostname=socket.gethostname(),
            role=node_cfg.get("role"),
            resolution_method=method,
        )

    @classmethod
    def resolve_safe(cls, config: dict[str, Any] | None = None) -> NodeIdentity | None:
        """Resolve identity without raising exceptions.

        Returns None if resolution fails, allowing callers to handle gracefully.
        """
        try:
            return cls.resolve(config)
        except IdentityError as e:
            logger.error(f"Node identity resolution failed: {e}")
            return None

    def matches_peer(self, peer_info: dict[str, Any]) -> bool:
        """Check if this identity matches a peer info dict.

        Useful for matching P2P peers to voter configuration.

        Args:
            peer_info: Dict with 'addresses', 'node_id', etc.

        Returns:
            True if any identifier matches
        """
        # Check node_id match
        if peer_info.get("node_id") == self.canonical_id:
            return True

        # Check address match
        peer_addresses = set(peer_info.get("addresses", []))
        my_addresses = {self.tailscale_ip, self.ssh_host} - {None}

        return bool(peer_addresses & my_addresses)


def get_tailscale_ip() -> str | None:
    """Get the local Tailscale IP address.

    Returns:
        Tailscale IP (100.x.x.x) or None if not available
    """
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.debug(f"tailscale status failed: {result.stderr}")
            return None

        status = json.loads(result.stdout)
        self_info = status.get("Self", {})
        ips = self_info.get("TailscaleIPs", [])

        # Return first IPv4 (100.x.x.x)
        for ip in ips:
            if ip.startswith("100."):
                return ip

        return ips[0] if ips else None

    except subprocess.TimeoutExpired:
        logger.debug("tailscale status timed out")
        return None
    except FileNotFoundError:
        logger.debug("tailscale command not found")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse tailscale status: {e}")
        return None


def _load_hosts_config() -> dict[str, Any]:
    """Load distributed_hosts.yaml configuration.

    Returns:
        Config dict with 'hosts' key
    """
    # Try multiple locations
    search_paths = [
        Path(__file__).parent.parent.parent / "config" / "distributed_hosts.yaml",
        Path.cwd() / "config" / "distributed_hosts.yaml",
        Path.home() / "ringrift" / "ai-service" / "config" / "distributed_hosts.yaml",
    ]

    # Also check env var override
    env_config = os.environ.get("RINGRIFT_CONFIG_PATH")
    if env_config:
        search_paths.insert(0, Path(env_config))

    for path in search_paths:
        if path.exists():
            try:
                import yaml

                with open(path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")
                continue

    logger.warning("No distributed_hosts.yaml found")
    return {}


# Cached singleton accessor
_identity_cache: NodeIdentity | None = None


def get_node_identity(force_refresh: bool = False) -> NodeIdentity:
    """Get the canonical node identity (cached).

    Args:
        force_refresh: If True, re-resolve identity

    Returns:
        NodeIdentity for this node

    Raises:
        IdentityError: If identity cannot be resolved
    """
    global _identity_cache

    if _identity_cache is None or force_refresh:
        _identity_cache = NodeIdentity.resolve()

    return _identity_cache


def resolve_node_id() -> str:
    """Convenience function to get just the canonical node ID.

    This is the primary function to use when you just need the node ID string.

    Returns:
        Canonical node ID string

    Raises:
        IdentityError: If identity cannot be resolved
    """
    return get_node_identity().canonical_id


@lru_cache(maxsize=1)
def get_node_id_safe() -> str:
    """Get node ID with fallback to hostname if resolution fails.

    Use this for non-critical code paths where a fallback is acceptable.
    For critical code paths, use resolve_node_id() which will fail fast.

    Returns:
        Canonical node ID or hostname as fallback
    """
    identity = NodeIdentity.resolve_safe()
    if identity:
        return identity.canonical_id

    # Fallback to old behavior
    return os.environ.get("RINGRIFT_NODE_ID", socket.gethostname())

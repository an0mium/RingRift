"""Network Utilities for P2P Orchestrator.

Extracted from p2p_orchestrator.py on December 26, 2025.

This module provides:
- Peer address parsing (host:port, URLs)
- Tailscale host detection
- URL building utilities for peer communication

Usage as standalone:
    from scripts.p2p.network_utils import NetworkUtils

    # Parse peer address
    scheme, host, port = NetworkUtils.parse_peer_address("http://example.com:8770")

    # Check if host is Tailscale
    is_ts = NetworkUtils.is_tailscale_host("100.64.1.100")

    # Build URLs for peer
    urls = NetworkUtils.urls_for_peer(peer_info, "/status", local_has_tailscale=True)
"""

from __future__ import annotations

import ipaddress
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .constants import DEFAULT_PORT, TAILSCALE_CGNAT_NETWORK

if TYPE_CHECKING:
    from .models import NodeInfo

logger = logging.getLogger(__name__)


class NetworkUtils:
    """Standalone network utilities for P2P communication.

    All methods are static and can be used without instantiation.
    """

    @staticmethod
    def parse_peer_address(peer_addr: str) -> tuple[str, str, int]:
        """Parse peer address from various formats.

        Supports:
        - `host`
        - `host:port`
        - `http://host[:port]`
        - `https://host[:port]`

        Args:
            peer_addr: Peer address string

        Returns:
            Tuple of (scheme, host, port)

        Raises:
            ValueError: If address is empty or invalid
        """
        peer_addr = (peer_addr or "").strip()
        if not peer_addr:
            raise ValueError("Empty peer address")

        if "://" in peer_addr:
            parsed = urlparse(peer_addr)
            scheme = (parsed.scheme or "http").lower()
            host = parsed.hostname or ""
            if not host:
                raise ValueError(f"Invalid peer URL: {peer_addr}")
            if parsed.port is not None:
                port = int(parsed.port)
            else:
                port = 443 if scheme == "https" else DEFAULT_PORT
            return scheme, host, port

        # Back-compat: host[:port]
        parts = peer_addr.split(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_PORT
        return "http", host, port

    @staticmethod
    def is_tailscale_host(host: str) -> bool:
        """Check if host is a Tailscale mesh endpoint.

        Args:
            host: Hostname or IP address to check

        Returns:
            True if host is a Tailscale endpoint (100.x.x.x or .ts.net)
        """
        h = (host or "").strip()
        if not h:
            return False
        if h.endswith(".ts.net"):
            return True
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            return False
        if not isinstance(ip, ipaddress.IPv4Address):
            return False
        return ip in TAILSCALE_CGNAT_NETWORK

    @staticmethod
    def build_url(scheme: str, host: str, port: int, path: str) -> str:
        """Build a URL from components.

        Args:
            scheme: URL scheme (http/https)
            host: Hostname or IP
            port: Port number
            path: URL path (should start with /)

        Returns:
            Formatted URL string
        """
        return f"{scheme}://{host}:{port}{path}"

    @staticmethod
    def url_for_peer(
        peer: "NodeInfo",
        path: str,
        local_has_tailscale: bool = False,
    ) -> str:
        """Build a single URL for reaching a peer.

        Args:
            peer: NodeInfo object with host/port info
            path: URL path (e.g., "/status")
            local_has_tailscale: Whether local node has Tailscale

        Returns:
            URL string for peer communication
        """
        scheme = (getattr(peer, "scheme", None) or "http").lower()
        host = str(getattr(peer, "host", "") or "").strip()
        try:
            port = int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT

        rh = (getattr(peer, "reported_host", "") or "").strip()
        try:
            rp = int(getattr(peer, "reported_port", 0) or 0)
        except ValueError:
            rp = 0

        if rh and rp:
            # Prefer reported endpoints when the observed endpoint is loopback
            # (proxy/relay artifacts).
            is_loopback = host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
            prefer_tailscale = local_has_tailscale and NetworkUtils.is_tailscale_host(rh)
            if is_loopback or prefer_tailscale:
                host, port = rh, rp

        return f"{scheme}://{host}:{port}{path}"

    @staticmethod
    def urls_for_peer(
        peer: "NodeInfo",
        path: str,
        local_has_tailscale: bool = False,
    ) -> list[str]:
        """Build candidate URLs for reaching a peer.

        Includes both the observed reachable endpoint (`host`/`port`) and the
        peer's self-reported endpoint (`reported_host`/`reported_port`) when
        available. This improves resilience in mixed network environments
        (public IP vs overlay networks like Tailscale, port-mapped listeners).

        Args:
            peer: NodeInfo object with host/port info
            path: URL path (e.g., "/status")
            local_has_tailscale: Whether local node has Tailscale

        Returns:
            List of candidate URLs to try
        """
        scheme = (getattr(peer, "scheme", None) or "http").lower()
        urls: list[str] = []

        def _add(h: Any, p: Any) -> None:
            try:
                host_str = str(h or "").strip()
                port_int = int(p)
            except (ValueError, AttributeError):
                return
            if not host_str or port_int <= 0:
                return
            url = f"{scheme}://{host_str}:{port_int}{path}"
            if url not in urls:
                urls.append(url)

        rh = (getattr(peer, "reported_host", "") or "").strip()
        try:
            rp = int(getattr(peer, "reported_port", 0) or 0)
        except ValueError:
            rp = 0

        host = str(getattr(peer, "host", "") or "").strip()
        try:
            port = int(getattr(peer, "port", 0) or 0)
        except ValueError:
            port = 0

        # Prefer Tailscale endpoints first when available locally
        reported_preferred = False
        if rh and rp and local_has_tailscale and NetworkUtils.is_tailscale_host(rh):
            _add(rh, rp)
            reported_preferred = True

        _add(host, port)

        if rh and rp and (not reported_preferred) and (rh != host or rp != port):
            _add(rh, rp)

        return urls

    @staticmethod
    def endpoint_key(
        peer: "NodeInfo",
        local_has_tailscale: bool = False,
    ) -> tuple[str, str, int] | None:
        """Get normalized endpoint key for a peer.

        Used for detecting NAT/port collisions.

        Args:
            peer: NodeInfo object
            local_has_tailscale: Whether local node has Tailscale

        Returns:
            Tuple of (scheme, host, port) or None if invalid
        """
        scheme = (getattr(peer, "scheme", None) or "http").lower()
        host = str(getattr(peer, "host", "") or "").strip()
        try:
            port = int(getattr(peer, "port", DEFAULT_PORT) or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT

        rh = (getattr(peer, "reported_host", "") or "").strip()
        try:
            rp = int(getattr(peer, "reported_port", 0) or 0)
        except ValueError:
            rp = 0

        # Use reported_host when observed host is loopback/relay
        if rh and rp > 0:
            is_loopback = host in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}
            prefer_tailscale = local_has_tailscale and NetworkUtils.is_tailscale_host(rh)
            if is_loopback or prefer_tailscale:
                host, port = rh, rp

        if not host or port <= 0:
            return None

        return (scheme, host, port)

    @staticmethod
    def find_endpoint_conflicts(
        peers: list["NodeInfo"],
        local_has_tailscale: bool = False,
    ) -> set[tuple[str, str, int]]:
        """Find duplicate endpoints (NAT collisions).

        Args:
            peers: List of NodeInfo objects
            local_has_tailscale: Whether local node has Tailscale

        Returns:
            Set of endpoint keys that appear more than once
        """
        from collections import Counter

        keys = []
        for peer in peers:
            if hasattr(peer, "is_alive") and callable(peer.is_alive):
                if not peer.is_alive():
                    continue
            key = NetworkUtils.endpoint_key(peer, local_has_tailscale)
            if key:
                keys.append(key)

        counts = Counter(keys)
        return {k for k, count in counts.items() if count > 1}


class NetworkUtilsMixin:
    """Mixin class for adding network utilities to P2POrchestrator.

    Provides backward-compatible method names delegating to NetworkUtils.
    """

    def _parse_peer_address(self, peer_addr: str) -> tuple[str, str, int]:
        """Parse peer address (delegates to NetworkUtils)."""
        return NetworkUtils.parse_peer_address(peer_addr)

    def _is_tailscale_host(self, host: str) -> bool:
        """Check if host is Tailscale (delegates to NetworkUtils)."""
        return NetworkUtils.is_tailscale_host(host)

    def _url_for_peer(self, peer: "NodeInfo", path: str) -> str:
        """Build URL for peer (delegates to NetworkUtils)."""
        return NetworkUtils.url_for_peer(
            peer, path, local_has_tailscale=self._local_has_tailscale()
        )

    def _urls_for_peer(self, peer: "NodeInfo", path: str) -> list[str]:
        """Build URLs for peer (delegates to NetworkUtils)."""
        return NetworkUtils.urls_for_peer(
            peer, path, local_has_tailscale=self._local_has_tailscale()
        )

    def _local_has_tailscale(self) -> bool:
        """Check if local node has Tailscale.

        Override this in subclass to provide actual implementation.
        """
        return False

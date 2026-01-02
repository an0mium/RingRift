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

from .constants import DEFAULT_PORT, TAILSCALE_CGNAT_NETWORK, TAILSCALE_IPV6_NETWORK

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

        Jan 2, 2026: Added IPv6 support. Handles bracketed notation [::1]:port.

        Supports:
        - `host`
        - `host:port`
        - `[ipv6]:port`
        - `http://host[:port]`
        - `https://host[:port]`
        - `http://[ipv6][:port]`

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

        # Handle IPv6 bracket notation: [ipv6]:port or [ipv6]
        if peer_addr.startswith("["):
            bracket_end = peer_addr.find("]")
            if bracket_end == -1:
                raise ValueError(f"Invalid IPv6 address (unclosed bracket): {peer_addr}")
            host = peer_addr[1:bracket_end]
            rest = peer_addr[bracket_end + 1:]
            if rest.startswith(":"):
                try:
                    port = int(rest[1:])
                except ValueError:
                    port = DEFAULT_PORT
            else:
                port = DEFAULT_PORT
            return "http", host, port

        # Check if this is an unbracketed IPv6 address (contains multiple colons)
        # IPv6 addresses have 7 colons (full form) or :: (compressed form)
        if peer_addr.count(":") > 1:
            # This is likely an IPv6 address without brackets
            # Try to parse as IP to validate
            try:
                ipaddress.ip_address(peer_addr)
                return "http", peer_addr, DEFAULT_PORT
            except ValueError:
                # Could be IPv6:port, try to split on last colon
                # But this is ambiguous, so prefer explicit bracket notation
                last_colon = peer_addr.rfind(":")
                maybe_port = peer_addr[last_colon + 1:]
                if maybe_port.isdigit():
                    host = peer_addr[:last_colon]
                    try:
                        ipaddress.ip_address(host)
                        return "http", host, int(maybe_port)
                    except ValueError:
                        pass
                # Fall through to treat as hostname

        # Back-compat: host[:port] for IPv4 and hostnames
        parts = peer_addr.split(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 and parts[1] else DEFAULT_PORT
        return "http", host, port

    @staticmethod
    def is_tailscale_host(host: str) -> bool:
        """Check if host is a Tailscale mesh endpoint.

        Jan 2026: Added IPv6 support. Recognizes both IPv4 (100.x.x.x) and
        IPv6 (fd7a:115c:a1e0::*) Tailscale addresses.

        Args:
            host: Hostname or IP address to check

        Returns:
            True if host is a Tailscale endpoint (100.x.x.x, fd7a:..., or .ts.net)
        """
        h = (host or "").strip()
        if not h:
            return False
        if h.endswith(".ts.net"):
            return True

        # Handle IPv6 with brackets (e.g., [fd7a:115c:a1e0::1]:8770)
        if h.startswith("["):
            h = h.split("]")[0][1:]

        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            return False

        # Check IPv4 CGNAT range (100.64.0.0/10)
        if isinstance(ip, ipaddress.IPv4Address):
            return ip in TAILSCALE_CGNAT_NETWORK

        # Check IPv6 Tailscale range (fd7a:115c:a1e0::/48)
        if isinstance(ip, ipaddress.IPv6Address):
            return ip in TAILSCALE_IPV6_NETWORK

        return False

    @staticmethod
    def build_url(scheme: str, host: str, port: int, path: str) -> str:
        """Build a URL from components.

        Jan 2026: Added IPv6 support. IPv6 addresses are wrapped in brackets.

        Args:
            scheme: URL scheme (http/https)
            host: Hostname or IP
            port: Port number
            path: URL path (should start with /)

        Returns:
            Formatted URL string
        """
        # Wrap IPv6 addresses in brackets
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
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
            # Jan 2, 2026: Wrap IPv6 addresses in brackets for URL construction
            if ":" in host_str and not host_str.startswith("["):
                host_str = f"[{host_str}]"
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

    @staticmethod
    def get_local_tailscale_ipv6() -> str:
        """Get local Tailscale IPv6 address if available.

        Jan 2, 2026: Added for IPv6 support. Tailscale assigns each node an IPv6
        address in the fd7a:115c:a1e0::/48 range in addition to the IPv4 100.x.x.x.

        Returns:
            Tailscale IPv6 address (e.g., "fd7a:115c:a1e0:ab12:4843:cd96:6260:1234")
            or empty string if not available.
        """
        import socket
        try:
            # Get all network interfaces
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET6):
                addr = info[4][0]
                # Check if this is a Tailscale IPv6 address
                try:
                    ip = ipaddress.ip_address(addr.split("%")[0])  # Strip zone ID
                    if isinstance(ip, ipaddress.IPv6Address) and ip in TAILSCALE_IPV6_NETWORK:
                        return str(ip)
                except ValueError:
                    continue
        except (socket.gaierror, OSError):
            pass

        # Fallback: try reading from tailscale status
        try:
            import subprocess
            result = subprocess.run(
                ["tailscale", "ip", "-6"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                ipv6 = result.stdout.strip().split()[0]
                try:
                    ip = ipaddress.ip_address(ipv6)
                    if isinstance(ip, ipaddress.IPv6Address) and ip in TAILSCALE_IPV6_NETWORK:
                        return str(ip)
                except ValueError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return ""

    @staticmethod
    def get_local_tailscale_ipv4() -> str:
        """Get local Tailscale IPv4 address if available.

        Returns:
            Tailscale IPv4 address (e.g., "100.64.1.100") or empty string.
        """
        import socket
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                addr = info[4][0]
                try:
                    ip = ipaddress.ip_address(addr)
                    if isinstance(ip, ipaddress.IPv4Address) and ip in TAILSCALE_CGNAT_NETWORK:
                        return str(ip)
                except ValueError:
                    continue
        except (socket.gaierror, OSError):
            pass

        # Fallback: try reading from tailscale status
        try:
            import subprocess
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                ipv4 = result.stdout.strip().split()[0]
                try:
                    ip = ipaddress.ip_address(ipv4)
                    if isinstance(ip, ipaddress.IPv4Address) and ip in TAILSCALE_CGNAT_NETWORK:
                        return str(ip)
                except ValueError:
                    pass
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        return ""

    @staticmethod
    def get_local_tailscale_ips() -> tuple[str, str]:
        """Get both local Tailscale IPv4 and IPv6 addresses.

        Jan 2, 2026: Added for dual-stack support.

        Returns:
            Tuple of (ipv4, ipv6) where each is empty string if not available.
        """
        return (
            NetworkUtils.get_local_tailscale_ipv4(),
            NetworkUtils.get_local_tailscale_ipv6(),
        )


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

    def _get_tailscale_ip_for_peer(self, node_id: str) -> str:
        """Look up a peer's Tailscale IP from dynamic registry or cluster.yaml.

        Enables automatic fallback to Tailscale mesh when public IPs fail.
        Falls back to static config in cluster.yaml if dynamic registry unavailable.

        Args:
            node_id: The peer's node identifier

        Returns:
            Tailscale IP (100.x.x.x) if available, else empty string
        """
        # Try dynamic registry first
        try:
            from app.distributed.dynamic_registry import get_registry
            registry = get_registry()
            if registry is not None:
                with registry._lock:
                    if node_id in registry._nodes:
                        ts_ip = registry._nodes[node_id].tailscale_ip or ""
                        if ts_ip:
                            return ts_ip
        except (ImportError, KeyError, IndexError, AttributeError):
            pass

        # Fall back to static cluster.yaml config
        try:
            from scripts.p2p.cluster_config import get_cluster_config
            cluster_config = get_cluster_config()
            ts_ip = cluster_config.get_tailscale_ip(node_id)
            if ts_ip:
                return ts_ip
        except (ImportError, AttributeError):
            pass

        return ""

    def _tailscale_urls_for_voter(self, voter: "NodeInfo", path: str) -> list[str]:
        """Return Tailscale-exclusive URLs for voter communication.

        For election/lease operations between voter nodes, NAT-blocked public IPs
        cause split-brain issues. This method ensures voter communication uses only
        Tailscale mesh IPs (100.x.x.x) which bypass NAT.

        Falls back to regular `_urls_for_peer()` if no Tailscale IP is available.

        Args:
            voter: NodeInfo of the voter peer
            path: URL path (e.g., "/lease/request")

        Returns:
            List of Tailscale-only URLs, or fallback to regular URLs
        """
        import contextlib

        scheme = (getattr(voter, "scheme", None) or "http").lower()
        urls: list[str] = []

        voter_id = str(getattr(voter, "node_id", "") or "").strip()
        port = 0
        with contextlib.suppress(Exception):
            port = int(getattr(voter, "port", 0) or 0)
        if port <= 0:
            try:
                port = int(getattr(voter, "reported_port", DEFAULT_PORT) or DEFAULT_PORT)
            except ValueError:
                port = DEFAULT_PORT

        # Priority 1: Dynamic registry Tailscale IP lookup
        ts_ip = self._get_tailscale_ip_for_peer(voter_id)
        if ts_ip:
            urls.append(f"{scheme}://{ts_ip}:{port}{path}")

        # Priority 2: Check if reported_host is a Tailscale IP
        rh = str(getattr(voter, "reported_host", "") or "").strip()
        if rh and self._is_tailscale_host(rh):
            try:
                rp = int(getattr(voter, "reported_port", 0) or 0)
            except ValueError:
                rp = 0
            if rp > 0:
                url = f"{scheme}://{rh}:{rp}{path}"
                if url not in urls:
                    urls.append(url)

        # Priority 3: Check if host is a Tailscale IP
        host = str(getattr(voter, "host", "") or "").strip()
        if host and self._is_tailscale_host(host):
            url = f"{scheme}://{host}:{port}{path}"
            if url not in urls:
                urls.append(url)

        # VOTER COMMUNICATION FIX: Do NOT fall back to non-Tailscale URLs for voters.
        # Voter lease operations MUST use Tailscale to avoid NAT/loopback issues.
        # If no Tailscale URLs available, return empty list and let caller handle it
        # (the lease request will fail gracefully and try next voter).
        if not urls:
            logger.debug(f"No Tailscale URLs found for voter {voter_id}, skipping")
            return []

        return urls

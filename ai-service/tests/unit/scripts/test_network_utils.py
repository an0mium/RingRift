"""Tests for NetworkUtils from p2p module.

Tests cover:
- Peer address parsing (URLs, host:port)
- Tailscale host detection
- URL building utilities
- Endpoint key generation
- Conflict detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from scripts.p2p.network_utils import NetworkUtils, NetworkUtilsMixin


# =============================================================================
# Mock NodeInfo for testing
# =============================================================================


@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing URL building."""

    host: str = ""
    port: int = 8770
    scheme: str = "http"
    reported_host: str = ""
    reported_port: int = 0
    _alive: bool = True

    def is_alive(self) -> bool:
        return self._alive


# =============================================================================
# Parse Peer Address Tests
# =============================================================================


class TestParsePeerAddress:
    """Test peer address parsing."""

    def test_parse_host_only(self):
        """Test parsing host without port."""
        scheme, host, port = NetworkUtils.parse_peer_address("example.com")
        assert scheme == "http"
        assert host == "example.com"
        assert port == 8770  # DEFAULT_PORT

    def test_parse_host_with_port(self):
        """Test parsing host:port format."""
        scheme, host, port = NetworkUtils.parse_peer_address("example.com:9000")
        assert scheme == "http"
        assert host == "example.com"
        assert port == 9000

    def test_parse_http_url(self):
        """Test parsing HTTP URL."""
        scheme, host, port = NetworkUtils.parse_peer_address("http://example.com:8770")
        assert scheme == "http"
        assert host == "example.com"
        assert port == 8770

    def test_parse_https_url(self):
        """Test parsing HTTPS URL."""
        scheme, host, port = NetworkUtils.parse_peer_address("https://secure.example.com")
        assert scheme == "https"
        assert host == "secure.example.com"
        assert port == 443  # HTTPS default

    def test_parse_https_url_with_port(self):
        """Test parsing HTTPS URL with explicit port."""
        scheme, host, port = NetworkUtils.parse_peer_address("https://secure.example.com:8443")
        assert scheme == "https"
        assert host == "secure.example.com"
        assert port == 8443

    def test_parse_ip_address(self):
        """Test parsing IP address."""
        scheme, host, port = NetworkUtils.parse_peer_address("192.168.1.100:8770")
        assert scheme == "http"
        assert host == "192.168.1.100"
        assert port == 8770

    def test_parse_empty_raises(self):
        """Test that empty address raises ValueError."""
        with pytest.raises(ValueError, match="Empty peer address"):
            NetworkUtils.parse_peer_address("")

    def test_parse_invalid_url_raises(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid peer URL"):
            NetworkUtils.parse_peer_address("http://")

    def test_parse_whitespace_stripped(self):
        """Test that whitespace is stripped."""
        scheme, host, port = NetworkUtils.parse_peer_address("  example.com:8770  ")
        assert host == "example.com"


# =============================================================================
# Tailscale Host Detection Tests
# =============================================================================


class TestIsTailscaleHost:
    """Test Tailscale host detection."""

    def test_tailscale_ip(self):
        """Test Tailscale CGNAT IP detection."""
        assert NetworkUtils.is_tailscale_host("100.64.1.100") is True
        assert NetworkUtils.is_tailscale_host("100.100.100.100") is True

    def test_tailscale_domain(self):
        """Test Tailscale domain detection."""
        assert NetworkUtils.is_tailscale_host("mynode.ts.net") is True
        assert NetworkUtils.is_tailscale_host("subdomain.mynode.ts.net") is True

    def test_public_ip_not_tailscale(self):
        """Test public IPs are not Tailscale."""
        assert NetworkUtils.is_tailscale_host("8.8.8.8") is False
        assert NetworkUtils.is_tailscale_host("192.168.1.100") is False

    def test_hostname_not_tailscale(self):
        """Test regular hostnames are not Tailscale."""
        assert NetworkUtils.is_tailscale_host("example.com") is False
        assert NetworkUtils.is_tailscale_host("myserver.local") is False

    def test_empty_not_tailscale(self):
        """Test empty string is not Tailscale."""
        assert NetworkUtils.is_tailscale_host("") is False
        assert NetworkUtils.is_tailscale_host(None) is False

    def test_ipv6_not_tailscale(self):
        """Test IPv6 addresses are not Tailscale (only IPv4 supported)."""
        assert NetworkUtils.is_tailscale_host("::1") is False
        assert NetworkUtils.is_tailscale_host("2001:db8::1") is False


# =============================================================================
# URL Building Tests
# =============================================================================


class TestBuildUrl:
    """Test URL building."""

    def test_build_url_basic(self):
        """Test basic URL building."""
        url = NetworkUtils.build_url("http", "example.com", 8770, "/status")
        assert url == "http://example.com:8770/status"

    def test_build_url_https(self):
        """Test HTTPS URL building."""
        url = NetworkUtils.build_url("https", "secure.example.com", 443, "/api")
        assert url == "https://secure.example.com:443/api"


class TestUrlForPeer:
    """Test single URL building for peer."""

    def test_url_for_peer_basic(self):
        """Test basic URL for peer."""
        peer = MockNodeInfo(host="192.168.1.100", port=8770)
        url = NetworkUtils.url_for_peer(peer, "/status")
        assert url == "http://192.168.1.100:8770/status"

    def test_url_for_peer_with_scheme(self):
        """Test URL with custom scheme."""
        peer = MockNodeInfo(host="secure.example.com", port=443, scheme="https")
        url = NetworkUtils.url_for_peer(peer, "/api")
        assert url == "https://secure.example.com:443/api"

    def test_url_for_peer_loopback_uses_reported(self):
        """Test that loopback addresses use reported endpoint."""
        peer = MockNodeInfo(
            host="127.0.0.1",
            port=8770,
            reported_host="192.168.1.100",
            reported_port=9000,
        )
        url = NetworkUtils.url_for_peer(peer, "/status")
        assert url == "http://192.168.1.100:9000/status"

    def test_url_for_peer_tailscale_preferred(self):
        """Test Tailscale endpoint preferred when local has Tailscale."""
        peer = MockNodeInfo(
            host="192.168.1.100",
            port=8770,
            reported_host="100.64.1.50",
            reported_port=8770,
        )
        url = NetworkUtils.url_for_peer(peer, "/status", local_has_tailscale=True)
        assert url == "http://100.64.1.50:8770/status"


class TestUrlsForPeer:
    """Test multiple URL building for peer."""

    def test_urls_for_peer_basic(self):
        """Test basic URLs for peer."""
        peer = MockNodeInfo(host="192.168.1.100", port=8770)
        urls = NetworkUtils.urls_for_peer(peer, "/status")
        assert urls == ["http://192.168.1.100:8770/status"]

    def test_urls_for_peer_with_reported(self):
        """Test URLs include both observed and reported."""
        peer = MockNodeInfo(
            host="192.168.1.100",
            port=8770,
            reported_host="10.0.0.50",
            reported_port=9000,
        )
        urls = NetworkUtils.urls_for_peer(peer, "/status")
        assert len(urls) == 2
        assert "http://192.168.1.100:8770/status" in urls
        assert "http://10.0.0.50:9000/status" in urls

    def test_urls_for_peer_tailscale_first(self):
        """Test Tailscale URL comes first when local has Tailscale."""
        peer = MockNodeInfo(
            host="192.168.1.100",
            port=8770,
            reported_host="100.64.1.50",
            reported_port=8770,
        )
        urls = NetworkUtils.urls_for_peer(peer, "/status", local_has_tailscale=True)
        assert urls[0] == "http://100.64.1.50:8770/status"

    def test_urls_for_peer_no_duplicates(self):
        """Test no duplicate URLs."""
        peer = MockNodeInfo(
            host="192.168.1.100",
            port=8770,
            reported_host="192.168.1.100",
            reported_port=8770,
        )
        urls = NetworkUtils.urls_for_peer(peer, "/status")
        assert len(urls) == 1


# =============================================================================
# Endpoint Key Tests
# =============================================================================


class TestEndpointKey:
    """Test endpoint key generation."""

    def test_endpoint_key_basic(self):
        """Test basic endpoint key."""
        peer = MockNodeInfo(host="192.168.1.100", port=8770)
        key = NetworkUtils.endpoint_key(peer)
        assert key == ("http", "192.168.1.100", 8770)

    def test_endpoint_key_loopback_uses_reported(self):
        """Test loopback uses reported endpoint."""
        peer = MockNodeInfo(
            host="127.0.0.1",
            port=8770,
            reported_host="192.168.1.100",
            reported_port=9000,
        )
        key = NetworkUtils.endpoint_key(peer)
        assert key == ("http", "192.168.1.100", 9000)

    def test_endpoint_key_invalid_returns_none(self):
        """Test invalid peer returns None."""
        peer = MockNodeInfo(host="", port=0)
        key = NetworkUtils.endpoint_key(peer)
        assert key is None


class TestFindEndpointConflicts:
    """Test endpoint conflict detection."""

    def test_no_conflicts(self):
        """Test no conflicts with unique endpoints."""
        peers = [
            MockNodeInfo(host="192.168.1.100", port=8770),
            MockNodeInfo(host="192.168.1.101", port=8770),
            MockNodeInfo(host="192.168.1.102", port=8770),
        ]
        conflicts = NetworkUtils.find_endpoint_conflicts(peers)
        assert len(conflicts) == 0

    def test_detects_nat_collision(self):
        """Test detection of NAT collision (same endpoint)."""
        peers = [
            MockNodeInfo(host="192.168.1.100", port=8770),
            MockNodeInfo(host="192.168.1.100", port=8770),  # Duplicate
            MockNodeInfo(host="192.168.1.101", port=8770),
        ]
        conflicts = NetworkUtils.find_endpoint_conflicts(peers)
        assert ("http", "192.168.1.100", 8770) in conflicts

    def test_excludes_dead_peers(self):
        """Test dead peers are excluded from conflict detection."""
        peers = [
            MockNodeInfo(host="192.168.1.100", port=8770, _alive=True),
            MockNodeInfo(host="192.168.1.100", port=8770, _alive=False),  # Dead
        ]
        conflicts = NetworkUtils.find_endpoint_conflicts(peers)
        assert len(conflicts) == 0


# =============================================================================
# Mixin Tests
# =============================================================================


class TestNetworkUtilsMixin:
    """Test NetworkUtilsMixin."""

    def test_mixin_parse_peer_address(self):
        """Test mixin delegates parse_peer_address."""

        class TestClass(NetworkUtilsMixin):
            pass

        obj = TestClass()
        scheme, host, port = obj._parse_peer_address("example.com:8770")
        assert host == "example.com"

    def test_mixin_is_tailscale_host(self):
        """Test mixin delegates is_tailscale_host."""

        class TestClass(NetworkUtilsMixin):
            pass

        obj = TestClass()
        assert obj._is_tailscale_host("100.64.1.100") is True
        assert obj._is_tailscale_host("example.com") is False

    def test_mixin_url_for_peer(self):
        """Test mixin delegates url_for_peer."""

        class TestClass(NetworkUtilsMixin):
            def _local_has_tailscale(self) -> bool:
                return False

        obj = TestClass()
        peer = MockNodeInfo(host="192.168.1.100", port=8770)
        url = obj._url_for_peer(peer, "/status")
        assert url == "http://192.168.1.100:8770/status"

    def test_mixin_urls_for_peer(self):
        """Test mixin delegates urls_for_peer."""

        class TestClass(NetworkUtilsMixin):
            def _local_has_tailscale(self) -> bool:
                return True

        obj = TestClass()
        peer = MockNodeInfo(
            host="192.168.1.100",
            port=8770,
            reported_host="100.64.1.50",
            reported_port=8770,
        )
        urls = obj._urls_for_peer(peer, "/status")
        assert urls[0] == "http://100.64.1.50:8770/status"

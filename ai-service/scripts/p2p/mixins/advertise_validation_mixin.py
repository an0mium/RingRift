"""Advertise Host Validation Mixin - IP validation and advertise host management.

January 2026: Extracted from p2p_orchestrator.py to reduce file size.

This mixin provides IP validation and advertise host management functionality:
- _validate_and_fix_advertise_host(): Validate/fix advertise host IP
- _periodic_ip_validation_loop(): Background IP validation loop
- _is_advertising_private_ip(): Check if advertising private IP
- _try_get_tailscale_ip(): Try to get Tailscale IP
- _safe_emit_private_ip_alert(): Emit event for private IP detection
- _discover_all_ips(): Discover all IPs for this node (IPv4 + IPv6)
- _select_primary_advertise_host(): Select best primary address
- _set_advertise_host(): Atomically update advertise_host
- _get_yaml_tailscale_ip(): Get Tailscale IP from distributed_hosts.yaml

Usage:
    class P2POrchestrator(AdvertiseValidationMixin, ...):
        pass

Dependencies on parent class attributes:
    - advertise_host: str
    - alternate_ips: set[str]
    - node_id: str
    - self_info: NodeInfo
    - running: bool
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from scripts.p2p.p2p_mixin_base import P2PMixinBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AdvertiseValidationMixin(P2PMixinBase):
    """Mixin providing IP validation and advertise host management for P2P orchestrator.

    This mixin handles:
    - Validating and fixing advertise host IP addresses
    - Periodic IP validation for late Tailscale availability
    - Private IP detection and Tailscale enforcement
    - Multi-IP discovery for dual-stack (IPv4/IPv6) support

    Inherits from P2PMixinBase for shared helper methods.
    """

    MIXIN_TYPE = "advertise_validation"

    # Type hints for parent class attributes
    advertise_host: str
    advertise_port: int
    alternate_ips: set[str]
    node_id: str
    self_info: Any  # NodeInfo
    running: bool
    start_time: float

    def _set_advertise_host(self, new_host: str, reason: str = "") -> bool:
        """Atomically update advertise_host and self.self_info.host.

        Jan 12, 2026: Centralized setter to prevent host desync.

        Previously, multiple code paths could set advertise_host without
        updating self.self_info.host, causing heartbeats to broadcast stale IPs.
        This setter ensures both are always in sync.

        Args:
            new_host: New IP/hostname to advertise
            reason: Why the change is happening (for logging)

        Returns:
            True if host changed, False if no change needed
        """
        if not new_host or new_host == self.advertise_host:
            return False

        old_host = self.advertise_host
        self.advertise_host = new_host

        # CRITICAL: Update self.self_info snapshot immediately
        if hasattr(self, "self_info") and self.self_info:
            self.self_info.host = new_host
            self.self_info.last_heartbeat = time.time()

        reason_str = f" ({reason})" if reason else ""
        logger.info(f"[P2P] advertise_host changed: {old_host} -> {new_host}{reason_str}")
        return True

    def _validate_and_fix_advertise_host(self) -> None:
        """Validate advertise_host, fix private IP issues, and populate alternate_ips.

        December 30, 2025: Added to prevent P2P quorum loss caused by nodes
        advertising private LAN IPs (10.x, 192.168.x, 172.16-31.x) that other
        nodes in the mesh cannot reach.

        January 2, 2026: Extended for dual-stack IPv4/IPv6 support.
        - Discovers all IPs (IPv4 + IPv6) for this node
        - Prefers IPv4 for primary (broader compatibility)
        - Populates alternate_ips with all other addresses including IPv6

        This method:
        1. Discovers all reachable IPs (both address families)
        2. Selects best primary (prefer Tailscale IPv4, then any IPv4, then IPv6)
        3. Populates alternate_ips with remaining addresses
        4. Emits warnings/errors for operator awareness if using private IP
        """
        import ipaddress

        # Discover all IPs for this node (IPv4 + IPv6)
        all_ips = self._discover_all_ips()

        if not all_ips:
            logger.warning("[P2P] No IPs discovered for this node")
            return

        # If we already have a valid advertise_host in our discovered IPs, just update alternates
        if self.advertise_host and self.advertise_host in all_ips:
            # Check if current host is IPv6 but we have IPv4 available
            is_current_ipv6 = ":" in self.advertise_host
            ipv4_ips = {ip for ip in all_ips if ":" not in ip}

            if is_current_ipv6 and ipv4_ips:
                # Prefer IPv4 for primary
                primary, alternates = self._select_primary_advertise_host(all_ips)
                if primary and primary != self.advertise_host:
                    # Jan 12, 2026: Use setter to ensure self.self_info.host is also updated
                    self._set_advertise_host(primary, "ipv6_to_ipv4_switch")
                    self.alternate_ips = alternates
                    return

            # Current host is valid, just update alternates
            self.alternate_ips = all_ips - {self.advertise_host}
            logger.debug(f"[P2P] Updated alternate_ips: {len(self.alternate_ips)} addresses")
            return

        # Need to select a new primary host
        if self.advertise_host:
            # Current advertise_host is not in discovered IPs (possibly private/loopback)
            try:
                ip = ipaddress.ip_address(self.advertise_host)
                is_unreachable = ip.is_private or ip.is_loopback
                if is_unreachable:
                    ip_type = "loopback" if ip.is_loopback else "private"
                    logger.warning(
                        f"[P2P] Current advertise_host {self.advertise_host} is {ip_type}, selecting new primary"
                    )
            except ValueError:
                pass  # Not an IP address (maybe hostname)

        # Select best primary (IPv4 preferred)
        primary, alternates = self._select_primary_advertise_host(all_ips)

        if primary:
            old_host = self.advertise_host
            # Jan 12, 2026: Use setter to ensure self.self_info.host is also updated
            changed = self._set_advertise_host(primary, "primary_selection")
            self.alternate_ips = alternates

            if changed and old_host:
                print(
                    f"[P2P] WARNING: advertise_host auto-fixed: {old_host} -> {primary} "
                    f"(alternate IPs: {len(alternates)})"
                )
                logger.warning(
                    f"P2P advertise_host auto-fixed: {old_host} -> {primary} "
                    f"(discovered {len(all_ips)} IPs, {len(alternates)} alternates)"
                )
            elif not old_host:
                logger.info(
                    f"[P2P] advertise_host set to {primary} with {len(alternates)} alternate IPs"
                )
        else:
            # No valid IPs found - emit error
            print(
                f"[P2P] ERROR: advertise_host {self.advertise_host} is unreachable by peers! "
                f"Set RINGRIFT_P2P_ADVERTISE_HOST to your public/Tailscale IP."
            )
            logger.error(
                f"P2P advertise_host {self.advertise_host} is unreachable - mesh connectivity will fail!"
            )

    async def _periodic_ip_validation_loop(self) -> None:
        """Periodically revalidate advertise_host for late Tailscale availability.

        Dec 31, 2025: Added for 48-hour autonomous operation.
        Jan 12, 2026: Enhanced with aggressive startup checking and Tailscale enforcement.

        Problem: If Tailscale is not ready at startup, coordinator advertises private
        IP (10.x.x.x) which breaks mesh connectivity. Tailscale may become available
        later but the private IP persists.

        Solution: Check every 15s for first 5 minutes (aggressive startup), then
        every 5 minutes. If advertising private IP and Tailscale is available, switch
        to Tailscale IP, update peer info, and re-announce to bootstrap seeds.

        Jan 12, 2026 Enhancement: Added explicit Tailscale enforcement that checks
        specifically for private IP + Tailscale availability scenario.
        """
        interval = 15.0  # Very fast checking during startup (was 30s)
        stable_count = 0
        startup_fast_period = 300.0  # 5 minutes of fast checking (was 3 min)
        start_time = time.time()

        await asyncio.sleep(5)  # Brief initial delay (was 10s)

        while self.running:
            try:
                old_host = self.advertise_host
                is_private = self._is_advertising_private_ip()

                # Jan 12, 2026: Explicit Tailscale enforcement for private IP scenarios
                if is_private:
                    logger.info(f"[IP_ENFORCE] Advertising private IP {old_host}, checking for Tailscale...")
                    ts_ip = self._try_get_tailscale_ip()
                    if ts_ip:
                        # Tailscale is available - switch immediately
                        self._set_advertise_host(ts_ip, "tailscale_enforcement")
                        logger.warning(f"[IP_ENFORCE] Switched from private to Tailscale: {old_host} -> {ts_ip}")

                        # Emit event for monitoring/alerting
                        self._safe_emit_private_ip_alert(old_host, ts_ip, switched=True)

                        # Re-announce to bootstrap seeds
                        try:
                            await self._announce_to_bootstrap_seeds()
                            logger.info("[IP_ENFORCE] Re-announced to bootstrap seeds with Tailscale IP")
                        except Exception as announce_err:  # noqa: BLE001
                            logger.debug(f"[IP_ENFORCE] Re-announce failed: {announce_err}")

                        stable_count = 0
                        await asyncio.sleep(interval)
                        continue
                    else:
                        # Still no Tailscale - emit warning event periodically
                        if stable_count % 10 == 0:  # Every ~150s during startup
                            self._safe_emit_private_ip_alert(old_host, None, switched=False)
                            logger.warning(f"[IP_ENFORCE] Still advertising private IP {old_host}, Tailscale unavailable")

                # Run normal validation
                self._validate_and_fix_advertise_host()

                if old_host != self.advertise_host:
                    logger.warning(f"[P2P] IP revalidation detected change: {old_host} -> {self.advertise_host}")

                    # Re-announce to bootstrap seeds with corrected IP
                    try:
                        await self._announce_to_bootstrap_seeds()
                        logger.info("[P2P] Re-announced to bootstrap seeds with corrected IP")
                    except Exception as announce_err:  # noqa: BLE001
                        logger.debug(f"[P2P] Failed to re-announce after IP correction: {announce_err}")

                    stable_count = 0
                else:
                    stable_count += 1

                # Slow down after startup fast period and stable checks
                elapsed = time.time() - start_time
                if elapsed > startup_fast_period and stable_count >= 6:
                    if interval < 300.0:
                        interval = 300.0  # Slow to 5-minute checks
                        logger.debug("[P2P] IP validation: switching to 5-minute interval")

            except Exception as e:  # noqa: BLE001
                logger.debug(f"[P2P] IP revalidation error: {e}")

            await asyncio.sleep(interval)

    def _is_advertising_private_ip(self) -> bool:
        """Check if currently advertising a private/unreachable IP.

        Jan 12, 2026: Helper for Tailscale enforcement loop.

        Returns:
            True if advertise_host is a private IP (10.x, 192.168.x, 172.16-31.x)
        """
        import ipaddress

        if not self.advertise_host:
            return False

        try:
            ip = ipaddress.ip_address(self.advertise_host)
            # Tailscale CGNAT (100.x.x.x) is "private" technically but globally routable via mesh
            if self.advertise_host.startswith("100."):
                return False
            return ip.is_private or ip.is_loopback
        except ValueError:
            return False

    def _try_get_tailscale_ip(self) -> str:
        """Try to get Tailscale IP without waiting/blocking.

        Jan 12, 2026: Helper for Tailscale enforcement loop.

        Returns:
            Tailscale IP if available, else empty string
        """
        try:
            from scripts.p2p.resource_detector import ResourceDetector
            detector = ResourceDetector()
            # Prefer IPv4 for broader compatibility
            ts_ipv4 = detector.get_tailscale_ipv4()
            if ts_ipv4:
                return ts_ipv4
            ts_ipv6 = detector.get_tailscale_ipv6()
            if ts_ipv6:
                return ts_ipv6
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _safe_emit_private_ip_alert(self, private_ip: str, tailscale_ip: str | None, switched: bool) -> None:
        """Emit event for private IP detection (for monitoring/alerting).

        Jan 12, 2026: Part of Tailscale enforcement for autonomous operation.
        """
        try:
            from app.coordination.data_events import DataEventType
            event_type = DataEventType.PRIVATE_IP_ADVERTISED if hasattr(DataEventType, "PRIVATE_IP_ADVERTISED") else None
            if event_type:
                self._emit_event(
                    str(event_type.value),
                    {
                        "node_id": self.node_id,
                        "private_ip": private_ip,
                        "tailscale_ip": tailscale_ip,
                        "switched": switched,
                        "severity": "info" if switched else "warning",
                    },
                )
        except Exception:  # noqa: BLE001
            pass  # Event emission is best-effort

    def _discover_all_ips(self, exclude_primary: str | None = None) -> set[str]:
        """Discover all IP addresses this node can be reached at (IPv4 AND IPv6).

        January 2026: Multi-IP advertising for improved mesh resilience.
        January 2, 2026: Extended for dual-stack IPv4/IPv6 support.

        Collects IPs from:
        1. Tailscale IPs (100.x.x.x IPv4, fd7a:115c:a1e0:: IPv6)
        2. Hostname resolution (both address families)
        3. Local network interfaces (both address families)
        4. YAML config (tailscale_ip, ssh_host if resolvable)

        Returns:
            Set of IP addresses (excluding the primary advertise_host)
        """
        import ipaddress
        import socket

        ips: set[str] = set()

        # 1. Tailscale IPs (both IPv4 and IPv6)
        # IMPORTANT: Explicitly fetch BOTH address families since _get_tailscale_ip()
        # defaults to IPv6, which may not be reachable from IPv4-only peers.
        try:
            from scripts.p2p.resource_detector import ResourceDetector
            detector = ResourceDetector()
            # Get Tailscale IPv4 explicitly (100.x.x.x)
            ts_ipv4 = detector.get_tailscale_ipv4()
            if ts_ipv4:
                ips.add(ts_ipv4)
                logger.debug(f"[P2P] Discovered Tailscale IPv4: {ts_ipv4}")
            # Get Tailscale IPv6 explicitly (fd7a:115c:a1e0::)
            ts_ipv6 = detector.get_tailscale_ipv6()
            if ts_ipv6:
                ips.add(ts_ipv6)
                logger.debug(f"[P2P] Discovered Tailscale IPv6: {ts_ipv6}")
        except Exception as e:
            logger.debug(f"[P2P] ResourceDetector Tailscale lookup failed: {e}")
            # Fall back to legacy method
            ts_ip = self._get_tailscale_ip()
            if ts_ip:
                ips.add(ts_ip)

        # 2. Try to get IPs from hostname - BOTH address families
        try:
            hostname = socket.gethostname()
            # IPv4
            for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = addr_info[4][0]
                if ip and ip != "127.0.0.1":
                    ips.add(ip)
            # IPv6
            for addr_info in socket.getaddrinfo(hostname, None, socket.AF_INET6):
                ip = addr_info[4][0]
                # Skip link-local (fe80::) and loopback (::1)
                if ip and not ip.startswith("fe80:") and ip != "::1":
                    ips.add(ip)
        except Exception:
            pass

        # 3. Get IPs from network interfaces - BOTH address families
        try:
            import netifaces
            for iface in netifaces.interfaces():
                # Skip loopback interfaces
                if iface.startswith("lo"):
                    continue
                addrs = netifaces.ifaddresses(iface)
                # IPv4
                for addr_info in addrs.get(netifaces.AF_INET, []):
                    ip = addr_info.get("addr")
                    if ip and ip != "127.0.0.1":
                        try:
                            ip_obj = ipaddress.ip_address(ip)
                            # Include Tailscale and public IPs, skip other private
                            if not ip_obj.is_private or ip.startswith("100."):
                                ips.add(ip)
                        except ValueError:
                            pass
                # IPv6 (NEW - dual-stack support)
                for addr_info in addrs.get(netifaces.AF_INET6, []):
                    ip = addr_info.get("addr", "")
                    if ip:
                        # Strip zone ID (e.g., "fe80::1%eth0" -> "fe80::1")
                        ip = ip.split("%")[0]
                        # Skip link-local (fe80::) and loopback (::1)
                        if not ip.startswith("fe80:") and ip != "::1":
                            ips.add(ip)
        except ImportError:
            # netifaces not available, try socket approach
            try:
                # Get primary outbound IPv4
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                if local_ip and local_ip != "127.0.0.1":
                    ips.add(local_ip)
            except Exception:
                pass
        except Exception:
            pass

        # 4. Check YAML config for this node
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(self.node_id, {})

            # Add Tailscale IP from config
            cfg_ts_ip = node_cfg.get("tailscale_ip")
            if cfg_ts_ip:
                ips.add(cfg_ts_ip)

            # Jan 23, 2026: Add ssh_host directly if it looks like an IP address (public IP)
            # This is critical for RINGRIFT_PREFER_PUBLIC_IP to work
            ssh_host = node_cfg.get("ssh_host")
            if ssh_host and not ssh_host.startswith("ssh"):  # Skip vast.ai ssh gateway
                # Check if ssh_host is already an IP address (IPv4)
                import re
                ipv4_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
                if ipv4_pattern.match(ssh_host):
                    ips.add(ssh_host)
                    logger.debug(f"[P2P] Added ssh_host IP from YAML: {ssh_host}")

            # Also try to resolve ssh_host if it's a hostname - both address families
            if ssh_host and not ssh_host.startswith("ssh"):  # Skip vast.ai ssh gateway
                try:
                    # IPv4
                    for addr_info in socket.getaddrinfo(ssh_host, None, socket.AF_INET):
                        ip = addr_info[4][0]
                        if ip and ip != "127.0.0.1":
                            ips.add(ip)
                    # IPv6
                    for addr_info in socket.getaddrinfo(ssh_host, None, socket.AF_INET6):
                        ip = addr_info[4][0]
                        if ip and not ip.startswith("fe80:") and ip != "::1":
                            ips.add(ip)
                except Exception:
                    pass
        except Exception:
            pass

        # Remove primary host to avoid duplication
        if exclude_primary and exclude_primary in ips:
            ips.discard(exclude_primary)

        # Remove localhost variants (127.0.0.0/8 loopback range and bind-all)
        ips = {ip for ip in ips if not ip.startswith("127.") and ip != "0.0.0.0" and ip != "::1"}

        logger.debug(f"[P2P] Discovered alternate IPs: {ips}")
        return ips

    def _select_primary_advertise_host(self, all_ips: set[str]) -> tuple[str, set[str]]:
        """Select best primary address (prefer IPv4 for compatibility) and return alternates.

        January 2, 2026: Added for dual-stack IPv4/IPv6 support.
        January 23, 2026: Added RINGRIFT_PREFER_PUBLIC_IP to prioritize public IPs over Tailscale.
        Ensures primary advertise_host is IPv4 when available (broader compatibility),
        with IPv6 addresses available in alternate_ips for dual-stack peers.

        Preference order for primary (default):
        1. Tailscale CGNAT IPv4 (100.x.x.x) - globally routable via Tailscale mesh
        2. Other IPv4 addresses
        3. Tailscale IPv6 (fd7a:115c:a1e0::) - if no IPv4 available
        4. Other IPv6 addresses

        If RINGRIFT_PREFER_PUBLIC_IP=1, preference order becomes:
        1. Public IPv4 addresses (non-100.x.x.x, non-private)
        2. Tailscale CGNAT IPv4 (100.x.x.x)
        3. Other IPv4 addresses
        4. IPv6 addresses

        Args:
            all_ips: Set of all discovered IP addresses

        Returns:
            Tuple of (primary_host, alternate_ips)
        """
        if not all_ips:
            return "", set()

        ipv4_ips: set[str] = set()
        ipv6_ips: set[str] = set()

        for ip in all_ips:
            if ":" in ip:
                ipv6_ips.add(ip)
            else:
                ipv4_ips.add(ip)

        # Check if we should prefer public IPs over Tailscale
        prefer_public = os.environ.get("RINGRIFT_PREFER_PUBLIC_IP", "").strip().lower() in ("1", "true", "yes")

        # Separate public and Tailscale IPs
        tailscale_v4 = [ip for ip in ipv4_ips if ip.startswith("100.")]
        public_v4 = [ip for ip in ipv4_ips if not ip.startswith("100.") and not ip.startswith("10.") and not ip.startswith("172.") and not ip.startswith("192.168.")]

        if prefer_public:
            # Preference 1: Public IPv4 (non-Tailscale, non-private)
            if public_v4:
                primary = public_v4[0]
                alternates = all_ips - {primary}
                logger.info(f"[P2P] Preferring public IP {primary} over Tailscale (RINGRIFT_PREFER_PUBLIC_IP=1)")
                return primary, alternates

        # Preference 1/2: Tailscale CGNAT IPv4 (100.x.x.x)
        if tailscale_v4:
            primary = tailscale_v4[0]
            alternates = all_ips - {primary}
            return primary, alternates

        # Preference 2/3: Any other IPv4
        if ipv4_ips:
            primary = next(iter(ipv4_ips))
            alternates = all_ips - {primary}
            return primary, alternates

        # Preference 3: Tailscale IPv6 (fd7a:115c:a1e0::)
        tailscale_v6 = [ip for ip in ipv6_ips if ip.startswith("fd7a:115c:a1e0:")]
        if tailscale_v6:
            primary = tailscale_v6[0]
            alternates = all_ips - {primary}
            return primary, alternates

        # Preference 4: Any other IPv6
        if ipv6_ips:
            primary = next(iter(ipv6_ips))
            alternates = all_ips - {primary}
            return primary, alternates

        return "", set()

    def _get_yaml_tailscale_ip(self) -> str | None:
        """Get Tailscale IP from distributed_hosts.yaml for this node.

        Jan 12, 2026: Added to fix IP advertisement timing issue. When Tailscale
        CLI isn't ready at startup, we now fall back to the pre-configured
        tailscale_ip from YAML before falling back to local IP.

        This provides a reliable source for the correct IP even when Tailscale
        daemon is slow to start, preventing nodes from advertising unreachable
        local IPs (e.g., 10.0.0.62).

        Returns:
            Tailscale IP from config if available for this node, else None.
        """
        try:
            from app.config.cluster_config import load_cluster_config
            config = load_cluster_config()
            nodes = getattr(config, "hosts_raw", {}) or {}
            node_cfg = nodes.get(self.node_id, {})

            tailscale_ip = node_cfg.get("tailscale_ip")
            if tailscale_ip:
                logger.debug(f"[P2P] Found tailscale_ip in YAML config: {tailscale_ip}")
                return tailscale_ip
        except ImportError:
            logger.debug("[P2P] cluster_config not available for tailscale_ip lookup")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[P2P] Failed to load tailscale_ip from config: {e}")

        return None

    # Abstract methods that must be implemented by parent class
    def _get_tailscale_ip(self) -> str:
        """Get Tailscale IP for this node. Must be implemented by parent."""
        raise NotImplementedError("Parent class must implement _get_tailscale_ip()")

    async def _announce_to_bootstrap_seeds(self) -> None:
        """Announce to bootstrap seeds. Must be implemented by parent."""
        raise NotImplementedError("Parent class must implement _announce_to_bootstrap_seeds()")

#!/usr/bin/env python3
"""Provision node ID for P2P cluster membership.

This script writes the canonical node ID to /etc/ringrift/node-id,
establishing the single source of truth for this node's identity.

Usage:
    # Provision with explicit node ID (preferred)
    sudo python scripts/provision_node_id.py --node-id lambda-gh200-1

    # Auto-detect node ID from IP/hostname
    sudo python scripts/provision_node_id.py --auto-detect

    # Verify current provisioning status
    python scripts/provision_node_id.py --verify

    # Dry run (show what would be written)
    python scripts/provision_node_id.py --node-id lambda-gh200-1 --dry-run

Part of Phase 1: Node Identity Management (P2P Cluster Stability Plan)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.node_identity import (
    NODE_ID_FILE,
    LEGACY_P2P_CONFIG,
    get_tailscale_ip,
    validate_identity_claim,
)


def load_config() -> dict:
    """Load distributed_hosts.yaml configuration."""
    import yaml

    config_paths = [
        Path(__file__).parent.parent / "config" / "distributed_hosts.yaml",
        Path.cwd() / "config" / "distributed_hosts.yaml",
    ]

    for path in config_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}

    raise FileNotFoundError("Could not find distributed_hosts.yaml")


def get_local_ips() -> set[str]:
    """Get all IP addresses for this machine."""
    ips = set()

    # Get Tailscale IP
    ts_ip = get_tailscale_ip()
    if ts_ip:
        ips.add(ts_ip)

    # Get hostname-based IPs
    hostname = socket.gethostname()
    try:
        # Get all IPs for hostname
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ips.add(info[4][0])
    except socket.gaierror:
        pass

    # Try to get all network interface IPs
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr.get("addr")
                    if ip and not ip.startswith("127."):
                        ips.add(ip)
    except ImportError:
        # netifaces not available, try ip command
        try:
            result = subprocess.run(
                ["ip", "-4", "addr", "show"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.splitlines():
                if "inet " in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1].split("/")[0]
                        if not ip.startswith("127."):
                            ips.add(ip)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return ips


def auto_detect_node_id(config: dict) -> str | None:
    """Auto-detect node ID from local IPs and hostname."""
    hosts = config.get("hosts", {})
    local_ips = get_local_ips()
    hostname = socket.gethostname()

    print(f"Auto-detecting node ID...")
    print(f"  Hostname: {hostname}")
    print(f"  Local IPs: {local_ips}")

    # Priority 1: Direct hostname match
    if hostname in hosts:
        print(f"  Matched by hostname: {hostname}")
        return hostname

    # Priority 2: IP match
    for node_id, node_cfg in hosts.items():
        if not isinstance(node_cfg, dict):
            continue

        node_ips = set()
        if node_cfg.get("tailscale_ip"):
            node_ips.add(node_cfg["tailscale_ip"])
        if node_cfg.get("ssh_host"):
            ssh_host = node_cfg["ssh_host"]
            if ssh_host.replace(".", "").isdigit():
                node_ips.add(ssh_host)
        if node_cfg.get("public_ip"):
            node_ips.add(node_cfg["public_ip"])

        if local_ips & node_ips:
            matching_ip = (local_ips & node_ips).pop()
            print(f"  Matched by IP ({matching_ip}): {node_id}")
            return node_id

    print("  No match found!")
    return None


def verify_node_id(node_id: str, config: dict) -> tuple[bool, str]:
    """Verify that node ID is valid and IPs match."""
    hosts = config.get("hosts", {})

    if node_id not in hosts:
        return False, f"Node ID '{node_id}' not found in config"

    # Check IP match
    local_ips = get_local_ips()
    valid, reason = validate_identity_claim(node_id, local_ips, config)

    if not valid:
        return False, reason

    return True, "OK"


def write_node_id(node_id: str, dry_run: bool = False) -> bool:
    """Write node ID to the canonical file atomically."""
    target_dir = NODE_ID_FILE.parent

    if dry_run:
        print(f"[DRY RUN] Would write '{node_id}' to {NODE_ID_FILE}")
        return True

    try:
        # Create directory if needed
        if not target_dir.exists():
            target_dir.mkdir(parents=True, mode=0o755)

        # Write atomically using temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=target_dir,
            prefix="node-id.",
            delete=False,
        ) as f:
            f.write(node_id + "\n")
            temp_path = Path(f.name)

        # Set permissions
        os.chmod(temp_path, 0o644)

        # Atomic rename
        temp_path.rename(NODE_ID_FILE)

        print(f"Successfully wrote node ID to {NODE_ID_FILE}")
        return True

    except PermissionError:
        print(f"ERROR: Permission denied. Run with sudo to write to {NODE_ID_FILE}")
        return False
    except OSError as e:
        print(f"ERROR: Failed to write node ID: {e}")
        return False


def show_current_status() -> None:
    """Show current node identity provisioning status."""
    print("=== Node Identity Status ===\n")

    # Check canonical file
    print(f"Canonical file ({NODE_ID_FILE}):")
    if NODE_ID_FILE.exists():
        try:
            content = NODE_ID_FILE.read_text().strip()
            print(f"  Present: {content}")
        except PermissionError:
            print("  Present but unreadable (permission denied)")
    else:
        print("  Not present")

    # Check legacy file
    print(f"\nLegacy file ({LEGACY_P2P_CONFIG}):")
    if LEGACY_P2P_CONFIG.exists():
        try:
            for line in LEGACY_P2P_CONFIG.read_text().splitlines():
                if line.strip().startswith("NODE_ID="):
                    print(f"  Present: {line.strip()}")
                    break
            else:
                print("  Present but no NODE_ID line")
        except PermissionError:
            print("  Present but unreadable (permission denied)")
    else:
        print("  Not present")

    # Check env var
    env_id = os.environ.get("RINGRIFT_NODE_ID")
    print(f"\nEnvironment (RINGRIFT_NODE_ID):")
    if env_id:
        print(f"  Set: {env_id}")
    else:
        print("  Not set")

    # Show current resolution
    print("\n--- Current Resolution ---")
    try:
        from app.config.node_identity import get_node_identity

        identity = get_node_identity(force_refresh=True)
        print(f"  Canonical ID: {identity.canonical_id}")
        print(f"  Resolution method: {identity.resolution_method}")
        print(f"  Tailscale IP: {identity.tailscale_ip}")
        print(f"  Role: {identity.role}")
    except Exception as e:
        print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Provision node ID for P2P cluster membership",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Provision with explicit node ID (preferred)
    sudo python scripts/provision_node_id.py --node-id lambda-gh200-1

    # Auto-detect from IPs and hostname
    sudo python scripts/provision_node_id.py --auto-detect

    # Verify current status
    python scripts/provision_node_id.py --verify

    # Show what would be written
    python scripts/provision_node_id.py --node-id lambda-gh200-1 --dry-run
""",
    )

    parser.add_argument(
        "--node-id",
        help="Explicit node ID to provision",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect node ID from IPs and hostname",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify current provisioning status",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip IP validation (use with caution)",
    )

    args = parser.parse_args()

    # Handle verify mode
    if args.verify:
        show_current_status()
        return 0

    # Require either --node-id or --auto-detect
    if not args.node_id and not args.auto_detect:
        parser.print_help()
        print("\nError: Must specify either --node-id or --auto-detect")
        return 1

    # Load config
    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    # Determine node ID
    if args.auto_detect:
        node_id = auto_detect_node_id(config)
        if not node_id:
            print("\nERROR: Could not auto-detect node ID.")
            print("Please use --node-id to specify explicitly.")
            return 1
    else:
        node_id = args.node_id

    # Verify node ID
    if not args.force:
        valid, reason = verify_node_id(node_id, config)
        if not valid:
            print(f"\nERROR: Validation failed - {reason}")
            print("Use --force to skip validation (not recommended)")
            return 1

    # Write node ID
    print(f"\nProvisioning node ID: {node_id}")

    if not write_node_id(node_id, dry_run=args.dry_run):
        return 1

    # Show final status
    if not args.dry_run:
        print("\n--- Verification ---")
        show_current_status()

    return 0


if __name__ == "__main__":
    sys.exit(main())

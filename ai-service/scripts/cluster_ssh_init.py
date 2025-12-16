#!/usr/bin/env python3
"""Cluster SSH Initialization - Secure host key management.

This script manages SSH host keys for cluster hosts using ssh-keyscan.
It uses a secure Trust On First Use (TOFU) model:
1. Scan hosts to collect their public keys
2. Add keys to ~/.ssh/known_hosts (hashed for privacy)
3. Future connections verify against these known keys

Usage:
    # Scan all hosts from distributed_hosts.yaml and add to known_hosts
    python scripts/cluster_ssh_init.py --scan

    # Scan specific hosts only
    python scripts/cluster_ssh_init.py --scan --hosts lambda-gh200-a lambda-gh200-b

    # Dry run - show what would be added
    python scripts/cluster_ssh_init.py --scan --dry-run

    # Remove stale entries for hosts that changed keys
    python scripts/cluster_ssh_init.py --refresh

    # Verify all hosts are in known_hosts
    python scripts/cluster_ssh_init.py --verify
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_hosts_config() -> Dict[str, dict]:
    """Load host configurations from distributed_hosts.yaml."""
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    hosts = {}
    for name, data in config.get("hosts", {}).items():
        if data.get("enabled", True):
            ssh_host = data.get("ssh_host", data.get("host", ""))
            ssh_port = data.get("ssh_port", data.get("port", 22))
            if ssh_host:
                hosts[name] = {
                    "host": ssh_host,
                    "port": int(ssh_port),
                }
    return hosts


def get_known_hosts_path() -> Path:
    """Get the path to known_hosts file."""
    return Path.home() / ".ssh" / "known_hosts"


def scan_host_key(host: str, port: int = 22, timeout: int = 5) -> Optional[str]:
    """Scan a host for its SSH public key using ssh-keyscan.

    Returns the host key entry (hashed) or None if unreachable.
    """
    try:
        cmd = ["ssh-keyscan", "-H", "-T", str(timeout)]
        if port != 22:
            cmd.extend(["-p", str(port)])
        cmd.append(host)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )

        # ssh-keyscan outputs to stdout, errors to stderr
        if result.stdout.strip():
            return result.stdout.strip()
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"  Error scanning {host}: {e}")
        return None


def get_existing_hosts(known_hosts_path: Path) -> Set[str]:
    """Get set of hosts already in known_hosts.

    Note: When hosts are hashed (-H), we can't easily determine which hosts
    are present. This function returns unhashed host entries only.
    """
    existing = set()
    if not known_hosts_path.exists():
        return existing

    with open(known_hosts_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Hashed entries start with |1|
            if line.startswith("|1|"):
                continue  # Can't extract host from hashed entry
            # Format: host[,host...] key-type key [comment]
            parts = line.split()
            if len(parts) >= 2:
                hosts_part = parts[0]
                for h in hosts_part.split(","):
                    # Handle [host]:port format
                    if h.startswith("["):
                        h = h.split("]")[0][1:]
                    existing.add(h)
    return existing


def add_to_known_hosts(key_entry: str, known_hosts_path: Path, dry_run: bool = False) -> bool:
    """Add a host key entry to known_hosts."""
    if dry_run:
        print(f"  Would add: {key_entry[:80]}...")
        return True

    # Ensure .ssh directory exists with correct permissions
    ssh_dir = known_hosts_path.parent
    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    # Append to known_hosts
    with open(known_hosts_path, "a") as f:
        f.write(key_entry + "\n")
    return True


def remove_host_from_known_hosts(host: str, port: int, known_hosts_path: Path) -> bool:
    """Remove a host from known_hosts using ssh-keygen -R."""
    try:
        if port != 22:
            target = f"[{host}]:{port}"
        else:
            target = host

        result = subprocess.run(
            ["ssh-keygen", "-R", target, "-f", str(known_hosts_path)],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def verify_host_in_known_hosts(host: str, port: int = 22) -> bool:
    """Verify a host's key is in known_hosts by attempting a connection."""
    try:
        cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",  # Strict for verification
            "-o", "ConnectTimeout=5",
        ]
        if port != 22:
            cmd.extend(["-p", str(port)])
        cmd.extend([f"dummy@{host}", "exit"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Exit code 255 with "Host key verification failed" means not in known_hosts
        # Exit code 255 with "Permission denied" means key is known but auth failed (good!)
        stderr = result.stderr.lower()
        if "host key verification failed" in stderr:
            return False
        return True  # Key is known (even if auth fails)
    except Exception:
        return False


def scan_hosts(
    hosts: Dict[str, dict],
    filter_hosts: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Scan hosts and add their keys to known_hosts.

    Returns (success_count, fail_count).
    """
    known_hosts_path = get_known_hosts_path()
    success = 0
    failed = 0

    targets = hosts if not filter_hosts else {k: v for k, v in hosts.items() if k in filter_hosts}

    print(f"Scanning {len(targets)} hosts...")
    for name, config in sorted(targets.items()):
        host = config["host"]
        port = config["port"]

        port_str = f":{port}" if port != 22 else ""
        print(f"  {name} ({host}{port_str})...", end=" ", flush=True)

        key_entry = scan_host_key(host, port)
        if key_entry:
            add_to_known_hosts(key_entry, known_hosts_path, dry_run)
            print("OK")
            success += 1
        else:
            print("UNREACHABLE")
            failed += 1

    return success, failed


def refresh_hosts(hosts: Dict[str, dict], dry_run: bool = False) -> Tuple[int, int]:
    """Remove and re-scan all hosts to refresh their keys.

    Use this when host keys have changed (e.g., VM recreated).
    Returns (success_count, fail_count).
    """
    known_hosts_path = get_known_hosts_path()

    print(f"Refreshing keys for {len(hosts)} hosts...")
    for name, config in sorted(hosts.items()):
        host = config["host"]
        port = config["port"]

        if not dry_run:
            remove_host_from_known_hosts(host, port, known_hosts_path)
        print(f"  Removed old key for {name}")

    return scan_hosts(hosts, dry_run=dry_run)


def verify_hosts(hosts: Dict[str, dict]) -> Tuple[int, int, List[str]]:
    """Verify all hosts have keys in known_hosts.

    Returns (verified_count, missing_count, missing_hosts).
    """
    verified = 0
    missing = 0
    missing_hosts = []

    print(f"Verifying {len(hosts)} hosts...")
    for name, config in sorted(hosts.items()):
        host = config["host"]
        port = config["port"]

        port_str = f":{port}" if port != 22 else ""
        print(f"  {name} ({host}{port_str})...", end=" ", flush=True)

        if verify_host_in_known_hosts(host, port):
            print("OK")
            verified += 1
        else:
            print("MISSING")
            missing += 1
            missing_hosts.append(name)

    return verified, missing, missing_hosts


def main():
    parser = argparse.ArgumentParser(
        description="Cluster SSH initialization - secure host key management"
    )
    parser.add_argument("--scan", action="store_true", help="Scan hosts and add keys to known_hosts")
    parser.add_argument("--refresh", action="store_true", help="Remove and re-scan all host keys")
    parser.add_argument("--verify", action="store_true", help="Verify hosts are in known_hosts")
    parser.add_argument("--hosts", nargs="+", help="Specific hosts to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    if not (args.scan or args.refresh or args.verify):
        parser.print_help()
        return 1

    hosts = load_hosts_config()
    if not hosts:
        print("No hosts found in config")
        return 1

    print(f"Loaded {len(hosts)} hosts from config")

    if args.verify:
        verified, missing, missing_hosts = verify_hosts(hosts)
        print(f"\nResults: {verified} verified, {missing} missing")
        if missing_hosts:
            print(f"Missing hosts: {', '.join(missing_hosts)}")
            print("\nRun with --scan to add missing host keys")
        return 0 if missing == 0 else 1

    if args.refresh:
        success, failed = refresh_hosts(hosts, dry_run=args.dry_run)
    else:  # --scan
        success, failed = scan_hosts(hosts, filter_hosts=args.hosts, dry_run=args.dry_run)

    print(f"\nResults: {success} successful, {failed} failed")

    if args.dry_run:
        print("\n[DRY RUN] No changes made. Remove --dry-run to apply.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

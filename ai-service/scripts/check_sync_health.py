#!/usr/bin/env python3
"""Sync Health Monitoring - Quick check of data sync status.

Usage:
    python scripts/check_sync_health.py
    python scripts/check_sync_health.py --watch
    python scripts/check_sync_health.py --verbose

December 2025: Created for monitoring data sync between cluster and OWC drive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Terminal colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class SyncStatus:
    """Status of a sync target."""
    name: str
    healthy: bool
    last_sync: datetime | None
    file_count: int
    total_size_gb: float
    message: str


def run_ssh_command(host: str, command: str, timeout: int = 10) -> tuple[bool, str]:
    """Run SSH command and return (success, output)."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def check_daemon_status(host: str) -> dict[str, bool]:
    """Check if sync daemons are running."""
    daemons = {}

    success, output = run_ssh_command(
        host,
        "ps aux | grep -E 'external_drive_sync|s3_backup_daemon|auto_sync_daemon' | grep -v grep | wc -l"
    )

    if success:
        count = int(output.strip() or 0)
        daemons["sync_daemons_running"] = count > 0
        daemons["daemon_count"] = count
    else:
        daemons["sync_daemons_running"] = False
        daemons["daemon_count"] = 0

    return daemons


def check_owc_drive(host: str) -> SyncStatus:
    """Check OWC external drive sync status."""
    success, output = run_ssh_command(
        host,
        "df -h /Volumes/RingRift-Data 2>/dev/null | tail -1"
    )

    if not success or "RingRift-Data" not in output:
        return SyncStatus(
            name="OWC Drive",
            healthy=False,
            last_sync=None,
            file_count=0,
            total_size_gb=0,
            message="Drive not mounted"
        )

    # Get space info
    parts = output.split()
    if len(parts) >= 4:
        used = parts[2]
        avail = parts[3]
        capacity = parts[4]
    else:
        used = avail = capacity = "?"

    # Check recent sync activity
    success, sync_time = run_ssh_command(
        host,
        "stat -f '%Sm' /Volumes/RingRift-Data/selfplay_repository 2>/dev/null"
    )

    last_sync = None
    if success and sync_time:
        try:
            last_sync = datetime.strptime(sync_time.strip(), "%b %d %H:%M:%S %Y")
        except ValueError:
            pass

    # Count files (check multiple locations)
    success, count = run_ssh_command(
        host,
        "find /Volumes/RingRift-Data/selfplay_repository /Volumes/RingRift-Data/canonical_games /Volumes/RingRift-Data/canonical_data -type f 2>/dev/null | wc -l",
        timeout=30
    )
    file_count = int(count.strip() or 0) if success else 0

    # Get size
    success, size_output = run_ssh_command(
        host,
        "du -sh /Volumes/RingRift-Data 2>/dev/null | cut -f1"
    )
    size_str = size_output.strip() if success else "0G"

    # Parse size to GB
    total_gb = 0
    if size_str.endswith("T"):
        total_gb = float(size_str[:-1]) * 1024
    elif size_str.endswith("G"):
        total_gb = float(size_str[:-1])
    elif size_str.endswith("M"):
        total_gb = float(size_str[:-1]) / 1024

    age_hours = 0
    if last_sync:
        age_hours = (datetime.now() - last_sync).total_seconds() / 3600

    healthy = age_hours < 1.0 if last_sync else False

    return SyncStatus(
        name="OWC Drive",
        healthy=healthy,
        last_sync=last_sync,
        file_count=file_count,
        total_size_gb=total_gb,
        message=f"Used: {used}, Avail: {avail}, Capacity: {capacity}"
    )


def check_disk_usage(host: str) -> dict[str, Any]:
    """Check local disk usage."""
    success, output = run_ssh_command(
        host,
        "df -h /Users/armand 2>/dev/null | tail -1"
    )

    if success and output:
        parts = output.split()
        if len(parts) >= 5:
            capacity = parts[4].rstrip("%")
            try:
                return {
                    "disk_usage_percent": int(capacity),
                    "healthy": int(capacity) < 70,
                    "message": f"Disk at {capacity}%"
                }
            except ValueError:
                pass

    return {"disk_usage_percent": 0, "healthy": False, "message": "Unable to check"}


def check_s3_backup() -> SyncStatus:
    """Check S3 backup status (requires aws cli)."""
    try:
        bucket = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
        result = subprocess.run(
            ["aws", "s3", "ls", f"s3://{bucket}/", "--summarize"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Find total objects line
            total_objects = 0
            total_size = 0
            for line in lines:
                if "Total Objects:" in line:
                    total_objects = int(line.split(":")[1].strip())
                elif "Total Size:" in line:
                    size_bytes = int(line.split(":")[1].strip())
                    total_size = size_bytes / (1024**3)  # Convert to GB

            return SyncStatus(
                name="S3 Backup",
                healthy=total_objects > 0,
                last_sync=None,  # Would need to check object timestamps
                file_count=total_objects,
                total_size_gb=total_size,
                message=f"Bucket: {bucket}"
            )
        else:
            return SyncStatus(
                name="S3 Backup",
                healthy=False,
                last_sync=None,
                file_count=0,
                total_size_gb=0,
                message=f"Error: {result.stderr[:100]}"
            )
    except FileNotFoundError:
        return SyncStatus(
            name="S3 Backup",
            healthy=False,
            last_sync=None,
            file_count=0,
            total_size_gb=0,
            message="aws cli not installed"
        )
    except Exception as e:
        return SyncStatus(
            name="S3 Backup",
            healthy=False,
            last_sync=None,
            file_count=0,
            total_size_gb=0,
            message=str(e)
        )


def print_status(status: SyncStatus, verbose: bool = False) -> None:
    """Print sync status with colors."""
    icon = f"{GREEN}✓{RESET}" if status.healthy else f"{RED}✗{RESET}"
    print(f"{icon} {BOLD}{status.name}{RESET}")

    if status.last_sync:
        age = datetime.now() - status.last_sync
        age_str = f"{age.total_seconds() / 60:.0f} min ago"
        if age.total_seconds() > 3600:
            age_str = f"{age.total_seconds() / 3600:.1f} hours ago"
        color = GREEN if age.total_seconds() < 3600 else YELLOW
        print(f"   Last sync: {color}{age_str}{RESET}")

    if verbose or not status.healthy:
        print(f"   Files: {status.file_count:,}")
        print(f"   Size: {status.total_size_gb:.1f} GB")

    print(f"   {status.message}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check sync health status")
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all details")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval (seconds)")
    args = parser.parse_args()

    coordinator = "mac-studio"

    while True:
        os.system("clear" if sys.platform != "win32" else "cls")
        print(f"{BOLD}{CYAN}=== RingRift Sync Health ==={RESET}")
        print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Check daemon status
        daemon_status = check_daemon_status(coordinator)
        daemon_icon = f"{GREEN}✓{RESET}" if daemon_status.get("sync_daemons_running") else f"{RED}✗{RESET}"
        print(f"{daemon_icon} {BOLD}Sync Daemons{RESET}")
        print(f"   Running: {daemon_status.get('daemon_count', 0)} daemon(s)\n")

        # Check disk usage
        disk = check_disk_usage(coordinator)
        disk_icon = f"{GREEN}✓{RESET}" if disk["healthy"] else f"{YELLOW}⚠{RESET}"
        print(f"{disk_icon} {BOLD}Local Disk{RESET}")
        print(f"   {disk['message']}\n")

        # Check OWC drive
        owc_status = check_owc_drive(coordinator)
        print_status(owc_status, args.verbose)

        # Check S3 backup
        s3_status = check_s3_backup()
        print_status(s3_status, args.verbose)

        # Summary
        all_healthy = (
            daemon_status.get("sync_daemons_running", False)
            and disk["healthy"]
            and owc_status.healthy
            and s3_status.healthy
        )

        if all_healthy:
            print(f"{GREEN}{BOLD}All systems healthy!{RESET}")
        else:
            issues = []
            if not daemon_status.get("sync_daemons_running"):
                issues.append("Sync daemons not running")
            if not disk["healthy"]:
                issues.append("Disk usage high")
            if not owc_status.healthy:
                issues.append("OWC sync stale or unmounted")
            if not s3_status.healthy:
                issues.append("S3 backup issue")

            print(f"{YELLOW}Issues: {', '.join(issues)}{RESET}")

        if not args.watch:
            break

        print(f"\n{CYAN}Refreshing in {args.interval}s... (Ctrl+C to exit){RESET}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()

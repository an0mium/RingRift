#!/usr/bin/env python3
"""
Sync Monitor - Monitors cluster data sync to Mac Studio OWC drive.

This script checks the sync progress and reports status for all cluster nodes.
Designed to run on Mac Studio or any machine with SSH access to cluster nodes.

Usage:
    python scripts/sync_monitor.py [--dest-path PATH] [--interval SECONDS]

Example:
    python scripts/sync_monitor.py --dest-path /Volumes/RingRift-Data/canonical_data/cluster_20251222

Configuration:
    Hosts are loaded from config/distributed_hosts.yaml
    Copy config/distributed_hosts.yaml.example to get started.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.distributed.hosts import load_remote_hosts, HostConfig, DEFAULT_SSH_KEY


def check_node_db_size(host_config: HostConfig, db_path: str = "~/ringrift/ai-service/data/games/*.db") -> Optional[dict]:
    """Check DB sizes on a remote node."""
    try:
        ssh_key = os.path.expanduser(host_config.ssh_key or DEFAULT_SSH_KEY)
        ssh_user = host_config.ssh_user or "ubuntu"
        ssh_host = host_config.tailscale_ip or host_config.ssh_host

        cmd = [
            "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
            "-i", ssh_key, f"{ssh_user}@{ssh_host}",
            f"ls -lh {db_path} 2>/dev/null | tail -5"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            dbs = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 9:
                    size = parts[4]
                    name = parts[-1].split('/')[-1]
                    dbs.append({"name": name, "size": size})
            return {"status": "ok", "dbs": dbs}
        return {"status": "no_dbs", "dbs": []}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "dbs": []}
    except Exception as e:
        return {"status": f"error: {e}", "dbs": []}


def check_dest_progress(dest_path: str) -> dict:
    """Check sync destination progress."""
    dest = Path(dest_path)
    if not dest.exists():
        return {"status": "not_found", "files": [], "total_size": "0"}

    files = list(dest.glob("*.db"))
    total_size = sum(f.stat().st_size for f in files)

    def format_size(size_bytes: int) -> str:
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.1f}G"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.1f}M"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f}K"
        return f"{size_bytes}B"

    file_info = [{"name": f.name, "size": format_size(f.stat().st_size)} for f in files]
    return {"status": "ok", "files": file_info, "total_size": format_size(total_size)}


def check_running_syncs() -> list:
    """Check for running rsync/scp processes."""
    try:
        cmd = ["ps", "aux"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        syncs = []
        for line in result.stdout.split('\n'):
            if 'rsync' in line and 'grep' not in line:
                parts = line.split()
                if len(parts) >= 11:
                    syncs.append({
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "cmd": ' '.join(parts[10:])[:80]
                    })
        return syncs
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(description="Monitor cluster data sync progress")
    parser.add_argument("--dest-path", default="/Volumes/RingRift-Data/canonical_data/cluster_20251222",
                       help="Destination path for synced data")
    parser.add_argument("--interval", type=int, default=0,
                       help="Monitoring interval in seconds (0 for one-shot)")
    parser.add_argument("--check-nodes", action="store_true",
                       help="Also check source node DB sizes")
    args = parser.parse_args()

    while True:
        print(f"\n{'='*60}")
        print(f"Sync Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # Check destination
        print(f"\n📁 Destination: {args.dest_path}")
        dest_info = check_dest_progress(args.dest_path)
        print(f"   Status: {dest_info['status']}")
        print(f"   Total size: {dest_info['total_size']}")
        if dest_info['files']:
            print(f"   Files ({len(dest_info['files'])}):")
            for f in dest_info['files'][:10]:
                print(f"     - {f['name']}: {f['size']}")
            if len(dest_info['files']) > 10:
                print(f"     ... and {len(dest_info['files']) - 10} more")

        # Check running syncs
        print(f"\n🔄 Active sync processes:")
        syncs = check_running_syncs()
        if syncs:
            for s in syncs[:5]:
                print(f"   PID {s['pid']} (CPU: {s['cpu']}%, MEM: {s['mem']}%)")
                print(f"   └─ {s['cmd']}")
        else:
            print("   No active rsync processes")

        # Optionally check source nodes
        if args.check_nodes:
            print(f"\n🖥️  Source nodes:")
            hosts = load_remote_hosts()

            if not hosts:
                print("   Warning: No hosts found in config/distributed_hosts.yaml")
                print("   Copy config/distributed_hosts.yaml.example to get started.")
            else:
                for name, host_config in hosts.items():
                    info = check_node_db_size(host_config)
                    status = "✓" if info['status'] == 'ok' else "⚠"
                    host_ip = host_config.tailscale_ip or host_config.ssh_host
                    print(f"   {status} {name} ({host_ip}):")
                    if info['dbs']:
                        for db in info['dbs'][:3]:
                            print(f"      - {db['name']}: {db['size']}")
                    else:
                        print(f"      {info['status']}")

        print(f"\n{'='*60}")

        if args.interval <= 0:
            break

        print(f"Next check in {args.interval} seconds...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

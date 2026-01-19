#!/usr/bin/env python3
"""Deploy P2P supervisor (cron-based) to cluster nodes for automatic restart on crash.

For Docker containers (Vast.ai, RunPod) that don't have systemd, this uses a
cron-based keepalive approach instead.

This script:
1. Creates a keepalive script in the ringrift directory
2. Adds cron entry to run it every minute
3. Starts P2P immediately

Usage:
    python scripts/deploy_p2p_supervisor.py                  # Deploy to all p2p_enabled nodes
    python scripts/deploy_p2p_supervisor.py --nodes vast-*   # Deploy to matching nodes
    python scripts/deploy_p2p_supervisor.py --dry-run        # Preview actions
    python scripts/deploy_p2p_supervisor.py --check          # Check P2P status
    python scripts/deploy_p2p_supervisor.py --seeds ...      # Override seed peers
"""

import argparse
import asyncio
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

import yaml


KEEPALIVE_SCRIPT = '''#!/bin/bash
# RingRift P2P Keepalive Script
# Ensures P2P orchestrator is always running
# Jan 2026: Fixed race condition with flock-based locking

NODE_ID="{node_id}"
RINGRIFT_PATH="{ringrift_path}"
P2P_PORT=8770
P2P_SEEDS="{seed_peers}"
LOGFILE="$RINGRIFT_PATH/logs/p2p_keepalive.log"
PIDFILE="$RINGRIFT_PATH/logs/p2p.pid"
LOCKFILE="$RINGRIFT_PATH/logs/p2p_keepalive.lock"
STARTUP_GRACE=30  # Seconds to wait for P2P to start before considering it failed

# Ensure log directory exists
mkdir -p "$RINGRIFT_PATH/logs"

# Use flock for mutual exclusion - prevents multiple cron jobs from racing
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    # Another instance is already running, exit silently
    exit 0
fi

# Check if P2P is responding to health checks
check_health() {{
    curl -s --connect-timeout 5 http://localhost:$P2P_PORT/health > /dev/null 2>&1
}}

# Check if P2P process is running via PID file
check_pid() {{
    if [ -f "$PIDFILE" ]; then
        pid=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}}

# If health check passes, we're good
if check_health; then
    exit 0
fi

# Health check failed - check if process is starting up
if check_pid; then
    pid=$(cat "$PIDFILE")
    # Check how long the process has been running
    if [ -f "/proc/$pid/stat" ]; then
        # Linux: check process start time
        start_time=$(stat -c %Y /proc/$pid 2>/dev/null || echo 0)
        now=$(date +%s)
        age=$((now - start_time))
        if [ $age -lt $STARTUP_GRACE ]; then
            echo "[$(date)] P2P process $pid starting up (age: ${{age}}s), waiting..." >> "$LOGFILE"
            exit 0
        fi
    elif [ "$(uname)" = "Darwin" ]; then
        # macOS: use ps to get elapsed time
        elapsed=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        if [ -n "$elapsed" ]; then
            # Parse elapsed time (formats: SS, MM:SS, HH:MM:SS, D-HH:MM:SS)
            if ! echo "$elapsed" | grep -q ':'; then
                # Just seconds
                secs=$elapsed
            elif ! echo "$elapsed" | grep -q '-'; then
                # MM:SS or HH:MM:SS
                secs=$(echo "$elapsed" | awk -F: '{{if(NF==2) print $1*60+$2; else print $1*3600+$2*60+$3}}')
            else
                # Already running for days, definitely not starting up
                secs=99999
            fi
            if [ "$secs" -lt $STARTUP_GRACE ]; then
                echo "[$(date)] P2P process $pid starting up (age: ${{secs}}s), waiting..." >> "$LOGFILE"
                exit 0
            fi
        fi
    fi
    # Process exists but health check failed after grace period - kill it
    echo "[$(date)] P2P process $pid unhealthy after grace period, killing..." >> "$LOGFILE"
    kill -15 "$pid" 2>/dev/null
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
fi

# No healthy process running - clean up any orphans and start fresh
echo "[$(date)] P2P not running or unhealthy, starting..." >> "$LOGFILE"

# Kill any zombie/orphan processes (use SIGTERM first, then SIGKILL)
pgrep -f "p2p_orchestrator.py.*--node-id.*$NODE_ID" | while read -r pid; do
    echo "[$(date)] Killing orphan P2P process $pid" >> "$LOGFILE"
    kill -15 "$pid" 2>/dev/null
done
sleep 2
pkill -9 -f "p2p_orchestrator.py.*--node-id.*$NODE_ID" 2>/dev/null || true

# Find Python
if [ -f "$RINGRIFT_PATH/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/venv/bin/python"
else
    PYTHON="/usr/bin/python3"
fi

# Start P2P in background
cd "$RINGRIFT_PATH"
export PYTHONPATH="$RINGRIFT_PATH"
nohup $PYTHON scripts/p2p_orchestrator.py \\
    --node-id "$NODE_ID" \\
    --port $P2P_PORT \\
    --peers "$P2P_SEEDS" \\
    --ringrift-path "${{RINGRIFT_PATH%/ai-service}}" \\
    >> "$RINGRIFT_PATH/logs/p2p.log" 2>&1 &

P2P_PID=$!
echo "$P2P_PID" > "$PIDFILE"
echo "[$(date)] P2P started with PID $P2P_PID" >> "$LOGFILE"

# Wait a moment and verify it's still running
sleep 3
if kill -0 "$P2P_PID" 2>/dev/null; then
    echo "[$(date)] P2P process $P2P_PID confirmed running" >> "$LOGFILE"
else
    echo "[$(date)] WARNING: P2P process $P2P_PID died immediately after start" >> "$LOGFILE"
    # Check the log for errors
    tail -20 "$RINGRIFT_PATH/logs/p2p.log" >> "$LOGFILE" 2>/dev/null
fi
'''


def load_hosts_config():
    """Load distributed_hosts.yaml configuration."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_seed_peers(config: dict) -> list[str]:
    """Build seed peers from the p2p_voters list in distributed_hosts.yaml."""
    voters = config.get("p2p_voters", []) or []
    hosts = config.get("hosts", {}) or {}
    peers: list[str] = []
    for node_id in voters:
        host_config = hosts.get(node_id, {})
        host = host_config.get("p2p_public_host") or host_config.get("ssh_host")
        if not host:
            continue
        port = int(host_config.get("p2p_port", 8770) or 8770)
        if "://" in str(host):
            peer = str(host)
        else:
            peer = f"http://{host}:{port}"
        if peer not in peers:
            peers.append(peer)
    return peers


def build_ssh_cmd(host_config: dict) -> list[str]:
    """Build SSH command for a host."""
    ssh_host = host_config.get("ssh_host")
    ssh_port = host_config.get("ssh_port", 22)
    ssh_user = host_config.get("ssh_user", "root")
    ssh_key = host_config.get("ssh_key", "~/.ssh/id_ed25519")
    ssh_key = os.path.expanduser(ssh_key)

    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(ssh_port),
        "-i", ssh_key,
        f"{ssh_user}@{ssh_host}",
    ]


def get_keepalive_script(node_id: str, host_config: dict, seed_peers: list[str]) -> str:
    """Generate keepalive script for a node."""
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    # Expand ~ for the script
    ringrift_path = ringrift_path.replace("~", "$HOME")
    seed_list = ",".join(seed_peers)
    return KEEPALIVE_SCRIPT.format(
        node_id=node_id,
        ringrift_path=ringrift_path,
        seed_peers=seed_list,
    )


async def deploy_to_node(
    node_id: str,
    host_config: dict,
    seed_peers: list[str],
    dry_run: bool = False,
) -> tuple[str, bool, str]:
    """Deploy keepalive supervisor to a single node."""
    ssh_cmd = build_ssh_cmd(host_config)
    ringrift_path = host_config.get("ringrift_path", "~/ringrift/ai-service")
    keepalive_script = get_keepalive_script(node_id, host_config, seed_peers)

    # Need sudo for nebius (ubuntu user)
    sudo_prefix = ""
    if host_config.get("ssh_user") == "ubuntu":
        sudo_prefix = "sudo "

    commands = f"""
set -e

# Expand home directory
RINGRIFT_PATH="{ringrift_path}"
RINGRIFT_PATH="${{RINGRIFT_PATH/#~/$HOME}}"

# Create directories
mkdir -p "$RINGRIFT_PATH/logs"
mkdir -p "$RINGRIFT_PATH/scripts"

# Write keepalive script
cat > "$RINGRIFT_PATH/scripts/p2p_keepalive.sh" << 'KEEPALIVE_EOF'
{keepalive_script}
KEEPALIVE_EOF

chmod +x "$RINGRIFT_PATH/scripts/p2p_keepalive.sh"

# Add cron entry (if not already present)
CRON_CMD="* * * * * $RINGRIFT_PATH/scripts/p2p_keepalive.sh"
(crontab -l 2>/dev/null | grep -v "p2p_keepalive" || true; echo "$CRON_CMD") | crontab -

# Run keepalive immediately to start P2P
$RINGRIFT_PATH/scripts/p2p_keepalive.sh

# Wait for P2P to start
sleep 5

# Check if running
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "P2P_RUNNING"
else
    echo "P2P_NOT_RUNNING"
fi
"""

    if dry_run:
        return (node_id, True, f"Would deploy keepalive to {node_id}")

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            commands,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=90)
        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        if "P2P_RUNNING" in stdout_text:
            return (node_id, True, "Deployed and running")
        elif "P2P_NOT_RUNNING" in stdout_text:
            # Cron installed but P2P not yet running (might take a moment)
            return (node_id, True, "Deployed, awaiting start")
        else:
            return (node_id, False, f"Exit {result.returncode}: {stderr_text[:200] or stdout_text[:200]}")
    except asyncio.TimeoutError:
        return (node_id, False, "Timeout")
    except Exception as e:
        return (node_id, False, str(e))


async def check_node_status(node_id: str, host_config: dict) -> tuple[str, str, str]:
    """Check P2P status on a node."""
    ssh_cmd = build_ssh_cmd(host_config)

    command = """
if curl -s --connect-timeout 5 http://localhost:8770/health > /dev/null 2>&1; then
    echo "active"
    curl -s http://localhost:8770/status 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('node_id','?'))" 2>/dev/null || echo "?"
else
    echo "inactive"
    echo "-"
fi
crontab -l 2>/dev/null | grep -q "p2p_keepalive" && echo "cron_ok" || echo "no_cron"
"""

    try:
        result = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(result.communicate(), timeout=20)
        lines = stdout.decode().strip().split('\n')
        status = lines[0] if lines else "unknown"
        node_reported = lines[1] if len(lines) > 1 else "?"
        has_cron = lines[2] if len(lines) > 2 else "?"
        return (node_id, status, f"cron:{has_cron}")
    except asyncio.TimeoutError:
        return (node_id, "timeout", "")
    except Exception as e:
        return (node_id, f"error: {e}", "")


async def main():
    parser = argparse.ArgumentParser(description="Deploy P2P supervisor (cron-based) to cluster nodes")
    parser.add_argument("--nodes", help="Node pattern to match (e.g., 'vast-*')")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without executing")
    parser.add_argument("--check", action="store_true", help="Check P2P status on nodes")
    parser.add_argument("--seeds", help="Override seed peers (comma-separated)")
    parser.add_argument("--skip-local", action="store_true", default=True, help="Skip local nodes (default: true)")
    args = parser.parse_args()

    config = load_hosts_config()
    hosts = config.get("hosts", {})
    seed_peers = [s.strip() for s in (args.seeds or "").split(",") if s.strip()]
    if not seed_peers:
        seed_peers = build_seed_peers(config)
    if not seed_peers:
        print("Warning: No seed peers resolved from config; P2P bootstrap may be slow.", file=sys.stderr)

    # Filter to p2p_enabled nodes
    nodes_to_deploy = []
    for node_id, host_config in hosts.items():
        if not host_config.get("p2p_enabled", False):
            continue

        # Skip local nodes
        if args.skip_local:
            if node_id in ("local-mac", "mac-studio"):
                continue
            if host_config.get("ssh_host") == "localhost":
                continue

        # Pattern match if specified
        if args.nodes and not fnmatch(node_id, args.nodes):
            continue

        nodes_to_deploy.append((node_id, host_config))

    if not nodes_to_deploy:
        print("No nodes matched the criteria")
        sys.exit(1)

    print(f"Target nodes: {len(nodes_to_deploy)}")

    if args.check:
        # Check status mode
        print("\nChecking P2P status...\n")
        tasks = [check_node_status(nid, cfg) for nid, cfg in nodes_to_deploy]
        results = await asyncio.gather(*tasks)

        print(f"{'Node':<30} {'Status':<15} {'Supervisor'}")
        print("-" * 60)
        for node_id, status, supervisor in sorted(results):
            status_icon = "✓" if status == "active" else "✗" if status == "inactive" else "?"
            print(f"{node_id:<30} {status_icon} {status:<12} {supervisor}")

        active = sum(1 for _, s, _ in results if s == "active")
        print(f"\nActive: {active}/{len(results)}")
    else:
        # Deploy mode
        print("\nDeploying P2P keepalive supervisor...\n")

        # Deploy in batches of 10 for stability
        batch_size = 10
        all_results = []

        for i in range(0, len(nodes_to_deploy), batch_size):
            batch = nodes_to_deploy[i:i + batch_size]
            tasks = [deploy_to_node(nid, cfg, seed_peers, args.dry_run) for nid, cfg in batch]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

            for node_id, success, message in results:
                icon = "✓" if success else "✗"
                print(f"  {icon} {node_id}: {message}")

        succeeded = sum(1 for _, s, _ in all_results if s)
        print(f"\nDeployed: {succeeded}/{len(all_results)}")


if __name__ == "__main__":
    asyncio.run(main())

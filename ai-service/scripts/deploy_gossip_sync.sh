#!/bin/bash
# Deploy P2P Gossip Sync to all cluster hosts
#
# This script:
# 1. Pushes the gossip_sync.py to all hosts
# 2. Starts the gossip daemon on each host
# 3. Opens firewall port 8771 for gossip protocol
#
# Usage:
#   ./scripts/deploy_gossip_sync.sh          # Deploy to all hosts
#   ./scripts/deploy_gossip_sync.sh --status # Check status on all hosts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"
GOSSIP_PORT=8771

# Hosts to deploy to (excluding Tailscale duplicates)
HOSTS=(
    "ubuntu@192.222.51.29"    # gh200_a
    "ubuntu@192.222.51.167"   # gh200_b
    "ubuntu@192.222.51.162"   # gh200_c
    "ubuntu@192.222.58.122"   # gh200_d
    "ubuntu@192.222.57.162"   # gh200_e
    "ubuntu@192.222.57.178"   # gh200_f
    "ubuntu@192.222.57.79"    # gh200_g
    "ubuntu@192.222.56.123"   # gh200_h
    "ubuntu@192.222.50.112"   # gh200_i
    "ubuntu@192.222.50.210"   # gh200_j
    "ubuntu@192.222.51.150"   # gh200_k
    "ubuntu@192.222.51.233"   # gh200_l
    "ubuntu@209.20.157.81"    # lambda_h100
    "ubuntu@192.222.53.22"    # lambda_2xh100
    "ubuntu@150.136.65.197"   # lambda_a10
)

HOST_NAMES=(
    "gh200_a" "gh200_b" "gh200_c" "gh200_d" "gh200_e" "gh200_f"
    "gh200_g" "gh200_h" "gh200_i" "gh200_j" "gh200_k" "gh200_l"
    "lambda_h100" "lambda_2xh100" "lambda_a10"
)

check_status() {
    echo "=== Gossip Sync Status ==="
    for i in "${!HOSTS[@]}"; do
        host="${HOSTS[$i]}"
        name="${HOST_NAMES[$i]}"
        echo -n "$name: "
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" \
            "pgrep -f 'gossip_sync' > /dev/null && echo 'RUNNING' || echo 'STOPPED'" 2>/dev/null \
            || echo "UNREACHABLE"
    done
}

deploy() {
    echo "=== Deploying Gossip Sync to ${#HOSTS[@]} hosts ==="

    for i in "${!HOSTS[@]}"; do
        host="${HOSTS[$i]}"
        name="${HOST_NAMES[$i]}"
        echo ""
        echo "--- Deploying to $name ($host) ---"

        # Check connectivity
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo OK" > /dev/null 2>&1; then
            echo "  SKIP: Host unreachable"
            continue
        fi

        # Sync gossip_sync.py
        echo "  Syncing gossip_sync.py..."
        rsync -avz --progress \
            "$AI_SERVICE_ROOT/app/distributed/gossip_sync.py" \
            "$host:~/ringrift/ai-service/app/distributed/" 2>/dev/null || true

        # Sync remote_hosts.yaml (needed for peer discovery)
        rsync -avz \
            "$AI_SERVICE_ROOT/config/remote_hosts.yaml" \
            "$host:~/ringrift/ai-service/config/" 2>/dev/null || true

        # Stop existing daemon
        echo "  Stopping existing daemon..."
        ssh "$host" "pkill -f 'gossip_sync' || true" 2>/dev/null

        # Start daemon
        echo "  Starting gossip daemon..."
        ssh "$host" "cd ~/ringrift/ai-service && \
            source venv/bin/activate && \
            nohup python -m app.distributed.gossip_sync --start --node-id $name \
            > logs/gossip_sync.log 2>&1 &" 2>/dev/null

        # Verify started
        sleep 1
        if ssh -o ConnectTimeout=5 "$host" "pgrep -f 'gossip_sync' > /dev/null" 2>/dev/null; then
            echo "  ✓ Gossip daemon started"
        else
            echo "  ✗ Failed to start daemon"
        fi
    done

    echo ""
    echo "=== Deployment Complete ==="
    echo "Run '$0 --status' to check daemon status"
}

stop_all() {
    echo "=== Stopping Gossip Sync on all hosts ==="
    for i in "${!HOSTS[@]}"; do
        host="${HOSTS[$i]}"
        name="${HOST_NAMES[$i]}"
        echo -n "$name: "
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" \
            "pkill -f 'gossip_sync' && echo 'STOPPED' || echo 'NOT RUNNING'" 2>/dev/null \
            || echo "UNREACHABLE"
    done
}

case "${1:-deploy}" in
    --status|-s)
        check_status
        ;;
    --stop)
        stop_all
        ;;
    --deploy|deploy|"")
        deploy
        ;;
    *)
        echo "Usage: $0 [--status|--stop|--deploy]"
        exit 1
        ;;
esac

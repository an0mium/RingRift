#!/bin/bash
# Deploy Elo sync services to cluster nodes
# Usage: ./scripts/deploy_elo_sync.sh [coordinator|worker|both] [host1 host2 ...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
HOSTS_CONFIG="$AI_SERVICE_DIR/config/distributed_hosts.yaml"

MODE="${1:-worker}"
shift || true

# Parse hosts from config or use provided list
if [ $# -gt 0 ]; then
    HOSTS=("$@")
else
    # Extract tailscale IPs from config (nodes with status: ready)
    HOSTS=(
        "100.78.101.123"   # lambda-h100
        "100.97.104.89"    # lambda-2xh100
        "100.123.183.70"   # lambda-gh200-a
        "100.104.34.73"    # lambda-gh200-b
        "100.88.35.19"     # lambda-gh200-c
        "100.75.84.47"     # lambda-gh200-d
        "100.88.176.74"    # lambda-gh200-e
        "100.104.165.116"  # lambda-gh200-f
        "100.104.126.58"   # lambda-gh200-g
        "100.65.88.62"     # lambda-gh200-h
        "100.99.27.56"     # lambda-gh200-i
        "100.96.142.42"    # lambda-gh200-k
        "100.76.145.60"    # lambda-gh200-l
        "100.91.25.13"     # lambda-a10
        "100.115.97.24"    # aws-staging
    )
fi

echo "=== Deploying Elo Sync ($MODE) to ${#HOSTS[@]} hosts ==="

deploy_to_host() {
    local host=$1
    local mode=$2

    echo "[$host] Deploying $mode..."

    # Copy latest sync script
    scp -q "$AI_SERVICE_DIR/scripts/elo_db_sync.py" "ubuntu@$host:~/ringrift/ai-service/scripts/" 2>/dev/null || {
        echo "[$host] Failed to copy script"
        return 1
    }

    # Copy systemd service file
    if [ "$mode" = "coordinator" ] || [ "$mode" = "both" ]; then
        scp -q "$AI_SERVICE_DIR/config/systemd/elo-sync-coordinator.service" "ubuntu@$host:/tmp/" 2>/dev/null
        ssh -q "ubuntu@$host" "sudo mv /tmp/elo-sync-coordinator.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable elo-sync-coordinator && sudo systemctl restart elo-sync-coordinator" 2>/dev/null || {
            echo "[$host] Failed to setup coordinator service"
        }
    fi

    if [ "$mode" = "worker" ] || [ "$mode" = "both" ]; then
        scp -q "$AI_SERVICE_DIR/config/systemd/elo-sync-worker.service" "ubuntu@$host:/tmp/" 2>/dev/null
        ssh -q "ubuntu@$host" "sudo mv /tmp/elo-sync-worker.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable elo-sync-worker && sudo systemctl restart elo-sync-worker" 2>/dev/null || {
            echo "[$host] Failed to setup worker service"
        }
    fi

    echo "[$host] Done"
}

# Deploy in parallel
for host in "${HOSTS[@]}"; do
    deploy_to_host "$host" "$MODE" &
done

wait
echo ""
echo "=== Deployment complete ==="
echo "Check status with: python scripts/elo_db_sync.py --mode status"

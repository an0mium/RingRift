#!/usr/bin/env bash
# Deploy P2P systemd service to Hetzner CPU nodes
# Usage: ./scripts/deploy_p2p_systemd.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../config/systemd"
SSH_KEY="${HOME}/.ssh/id_cluster"

# Node configurations
NODES="hetzner-cpu1:46.62.147.150 hetzner-cpu2:135.181.39.239 hetzner-cpu3:46.62.217.168"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

deploy_to_node() {
    local node_name="$1"
    local ssh_host="$2"

    echo ""
    echo "=== Deploying to ${node_name} (${ssh_host}) ==="

    local env_file="${CONFIG_DIR}/p2p-env-${node_name}"
    if [[ ! -f "$env_file" ]]; then
        echo "ERROR: Environment file not found: ${env_file}"
        return 1
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] Would deploy to ${node_name}"
        echo "[DRY RUN] Service file: ${CONFIG_DIR}/ringrift-p2p.service"
        echo "[DRY RUN] Env file: ${env_file}"
        return 0
    fi

    # Create logs directory
    ssh -i "$SSH_KEY" "root@${ssh_host}" "mkdir -p /root/ringrift/ai-service/logs"

    # Copy service file
    scp -i "$SSH_KEY" "${CONFIG_DIR}/ringrift-p2p.service" "root@${ssh_host}:/etc/systemd/system/"

    # Copy environment file
    scp -i "$SSH_KEY" "${env_file}" "root@${ssh_host}:/etc/default/ringrift-p2p"

    # Update service to use environment file
    ssh -i "$SSH_KEY" "root@${ssh_host}" bash <<'REMOTE_SCRIPT'
        # Add EnvironmentFile to service
        if ! grep -q "EnvironmentFile" /etc/systemd/system/ringrift-p2p.service; then
            sed -i '/\[Service\]/a EnvironmentFile=/etc/default/ringrift-p2p' /etc/systemd/system/ringrift-p2p.service
        fi

        # Stop existing P2P process
        pkill -f "p2p_orchestrator.py" 2>/dev/null || true
        sleep 2

        # Reload and enable service
        systemctl daemon-reload
        systemctl enable ringrift-p2p
        systemctl start ringrift-p2p

        # Wait and check status
        sleep 3
        systemctl status ringrift-p2p --no-pager || true
REMOTE_SCRIPT

    echo "=== ${node_name} deployment complete ==="
}

echo "Deploying P2P systemd service to Hetzner nodes..."
echo "Service file: ${CONFIG_DIR}/ringrift-p2p.service"

for node_entry in $NODES; do
    node_name="${node_entry%%:*}"
    ssh_host="${node_entry##*:}"
    deploy_to_node "$node_name" "$ssh_host"
done

echo ""
echo "=== Deployment complete ==="
echo ""
echo "To check status on any node:"
echo "  ssh -i ~/.ssh/id_cluster root@<host> 'systemctl status ringrift-p2p'"
echo ""
echo "To view logs:"
echo "  ssh -i ~/.ssh/id_cluster root@<host> 'journalctl -u ringrift-p2p -f'"

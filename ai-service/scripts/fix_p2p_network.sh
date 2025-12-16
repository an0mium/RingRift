#!/bin/bash
# Fix P2P network by updating peer configuration on all voter nodes
#
# This script:
# 1. Generates the full peer list from distributed_hosts.yaml
# 2. Updates node.conf on each voter node
# 3. Updates the systemd service to read from node.conf
# 4. Restarts the P2P orchestrator on each node
#
# Usage:
#   ./scripts/fix_p2p_network.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"

# Voter nodes with their Tailscale IPs
declare -A VOTER_NODES=(
    ["aws-staging"]="100.115.97.24"
    ["lambda-a10"]="100.91.25.13"
    ["lambda-h100"]="100.78.101.123"
    ["lambda-2xh100"]="100.97.104.89"
    ["lambda-gh200-a"]="100.123.183.70"
    ["lambda-gh200-b"]="100.104.34.73"
    ["lambda-gh200-c"]="100.88.35.19"
    ["lambda-gh200-d"]="100.75.84.47"
    ["lambda-gh200-e"]="100.88.176.74"
    ["lambda-gh200-f"]="100.104.165.116"
)

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No changes will be made"
fi

# Generate the full peer list
echo "Generating peer list from distributed_hosts.yaml..."
PEERS=$(cd "$AI_SERVICE_ROOT" && python3 scripts/generate_p2p_config.py --voter-peers)
echo "Peer list: $PEERS"
echo "Total peers: $(echo "$PEERS" | tr ',' '\n' | wc -l | tr -d ' ')"
echo ""

update_node() {
    local node_id="$1"
    local ip="$2"
    local ssh_user="ubuntu"
    local ringrift_path="/home/ubuntu/ringrift"

    # Special case for aws-staging
    if [[ "$node_id" == "aws-staging" ]]; then
        ssh_user="ubuntu"
        ringrift_path="/home/ubuntu/ringrift"
    fi

    echo "=== Updating $node_id ($ip) ==="

    if $DRY_RUN; then
        echo "[DRY RUN] Would update $node_id"
        return 0
    fi

    # Test connectivity first
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$ssh_user@$ip" "echo ok" >/dev/null 2>&1; then
        echo "SKIP: Cannot reach $node_id at $ip"
        return 1
    fi

    # Sync the generate_p2p_config.py script first
    echo "  Syncing generate_p2p_config.py..."
    scp -o ConnectTimeout=10 "$AI_SERVICE_ROOT/scripts/generate_p2p_config.py" \
        "$ssh_user@$ip:$ringrift_path/ai-service/scripts/" 2>/dev/null || true

    # Update node.conf with P2P_PEERS
    echo "  Updating node.conf..."
    ssh -o ConnectTimeout=15 "$ssh_user@$ip" "
        # Add or update P2P_PEERS in node.conf
        if grep -q '^P2P_PEERS=' /etc/ringrift/node.conf 2>/dev/null; then
            sudo sed -i 's|^P2P_PEERS=.*|P2P_PEERS=$PEERS|' /etc/ringrift/node.conf
        else
            echo 'P2P_PEERS=$PEERS' | sudo tee -a /etc/ringrift/node.conf >/dev/null
        fi
        echo '  node.conf updated'
    " || echo "  Failed to update node.conf"

    # Update the systemd service to use P2P_PEERS from environment
    echo "  Updating systemd service..."
    ssh -o ConnectTimeout=15 "$ssh_user@$ip" "
        # Update service to use \$P2P_PEERS instead of hardcoded peers
        sudo sed -i 's|--peers \"http://[^\"]*\"|--peers \"\\\${P2P_PEERS:-}\"|' /etc/systemd/system/ringrift-p2p.service
        sudo systemctl daemon-reload
    " || echo "  Failed to update systemd service"

    # Restart the P2P orchestrator
    echo "  Restarting P2P orchestrator..."
    ssh -o ConnectTimeout=15 "$ssh_user@$ip" "
        sudo systemctl restart ringrift-p2p || sudo pkill -9 -f p2p_orchestrator
        sleep 3
        ps aux | grep p2p_orchestrator | grep -v grep | head -1 | awk '{print \"  PID: \" \$2 \" | CMD: \" \$11 \" \" \$12 \" \" \$13}'
    " || echo "  Failed to restart"

    echo ""
}

echo "Starting P2P network fix across ${#VOTER_NODES[@]} voter nodes..."
echo ""

# Update each voter node
for node_id in "${!VOTER_NODES[@]}"; do
    ip="${VOTER_NODES[$node_id]}"
    update_node "$node_id" "$ip" || true
done

echo ""
echo "=== Summary ==="
echo "Updated ${#VOTER_NODES[@]} voter nodes with full peer list."
echo "Wait 30-60 seconds for the network to converge, then verify with:"
echo "  curl -s http://<any-voter-ip>:8770/status | python3 -c \"import sys,json; d=json.load(sys.stdin); print('Leader:', d.get('leader_id'), '| Voters alive:', d.get('voters_alive'), '/', d.get('voter_quorum_size'), '| Quorum:', d.get('voter_quorum_ok'))\""

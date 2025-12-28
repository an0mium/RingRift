#!/bin/bash
# Automatic cluster data sync script
# Syncs canonical game databases from cluster nodes to local coordinator
# Run via cron: */30 * * * * /path/to/sync_cluster_data.sh >> /path/to/logs/sync.log 2>&1

set -e

# Configuration
LOCAL_DIR="/Users/armand/Development/RingRift/ai-service/data/games"
LOG_DIR="/Users/armand/Development/RingRift/ai-service/logs"
SSH_KEY="$HOME/.ssh/id_ed25519"
RSYNC_OPTS="-avz --progress --timeout=120"

# Cluster nodes with game data
declare -a NODES=(
    "ubuntu@89.169.111.139:22:~/ringrift/ai-service/data/games"      # nebius-h100-1
    "ubuntu@89.169.110.128:22:~/ringrift/ai-service/data/games"      # nebius-h100-3
    "ubuntu@89.169.112.47:22:~/ringrift/ai-service/data/games"       # nebius-backbone-1
    "root@38.128.233.145:33085:/workspace/ringrift/ai-service/data/games"  # runpod-a100-1
    "root@104.255.9.187:11681:/workspace/ringrift/ai-service/data/games"   # runpod-a100-2
    "root@208.167.249.164:22:/root/ringrift/ai-service/data/games"   # vultr-a100-20gb
)

# Ensure directories exist
mkdir -p "$LOCAL_DIR" "$LOG_DIR"

echo "=============================================="
echo "Cluster Data Sync - $(date)"
echo "=============================================="

TOTAL_SYNCED=0
FAILED_NODES=0

for node_spec in "${NODES[@]}"; do
    # Parse node specification
    USER_HOST=$(echo "$node_spec" | cut -d: -f1)
    PORT=$(echo "$node_spec" | cut -d: -f2)
    REMOTE_PATH=$(echo "$node_spec" | cut -d: -f3)
    NODE_NAME=$(echo "$USER_HOST" | cut -d@ -f2 | cut -d. -f1)

    echo ""
    echo "--- Syncing from $NODE_NAME ---"

    # Test connectivity first
    if ! timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        -p "$PORT" -i "$SSH_KEY" "$USER_HOST" "echo ok" > /dev/null 2>&1; then
        echo "  SKIP: Cannot reach $NODE_NAME"
        ((FAILED_NODES++))
        continue
    fi

    # Sync canonical databases
    if rsync $RSYNC_OPTS \
        -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p $PORT -i $SSH_KEY" \
        --include='canonical_*.db' \
        --include='selfplay_*.db' \
        --exclude='*.db-shm' \
        --exclude='*.db-wal' \
        --exclude='*.db-journal' \
        --exclude='*' \
        "$USER_HOST:$REMOTE_PATH/" \
        "$LOCAL_DIR/" 2>&1; then

        # Count synced files
        COUNT=$(ls -1 "$LOCAL_DIR"/canonical_*.db 2>/dev/null | wc -l)
        echo "  OK: Synced databases from $NODE_NAME"
        ((TOTAL_SYNCED += COUNT))
    else
        echo "  WARN: Rsync failed for $NODE_NAME"
        ((FAILED_NODES++))
    fi
done

echo ""
echo "=============================================="
echo "Sync Complete - $(date)"
echo "  Nodes failed: $FAILED_NODES"
echo "=============================================="

# Print current game counts
echo ""
echo "Current game counts:"
cd "$LOCAL_DIR"
for db in canonical_*.db; do
    if [ -f "$db" ]; then
        count=$(sqlite3 "$db" 'SELECT COUNT(*) FROM games' 2>/dev/null || echo "?")
        printf "  %-30s %s games\n" "$db" "$count"
    fi
done

echo ""
echo "Done."

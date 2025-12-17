#!/bin/bash
# Automated training data sync from GH200 cluster
# Run via cron: */15 * * * * /path/to/sync_training_data_cron.sh

set -e
cd "$(dirname "$0")/.."

LOG_FILE="logs/sync_cron.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# GH200-a: Tailscale IP (preferred) and direct IP (fallback)
GH200A_TS="100.123.183.70"
GH200A_DIRECT="192.222.51.29"

echo "[$TIMESTAMP] Starting data sync..." >> "$LOG_FILE"

# Try Tailscale first, fallback to direct
sync_success=false
for host in "$GH200A_TS" "$GH200A_DIRECT"; do
  if ssh -o ConnectTimeout=5 ubuntu@$host "test -f ~/ringrift/ai-service/data/games/jsonl_aggregated.db" 2>/dev/null; then
    echo "[$TIMESTAMP] Using host $host for sync" >> "$LOG_FILE"
    rsync -avz --timeout=120 ubuntu@$host:~/ringrift/ai-service/data/games/jsonl_aggregated.db \
      data/games/gh200a_synced.db >> "$LOG_FILE" 2>&1
    sync_success=true
    break
  fi
done

if [ "$sync_success" = false ]; then
  echo "[$TIMESTAMP] GH200-a sync failed - no accessible host" >> "$LOG_FILE"
fi

# Update symlinks
ln -sf gh200a_synced.db data/games/all_jsonl_training.db
ln -sf gh200a_synced.db data/games/jsonl_aggregated.db

# Sync MCTS data if available
mkdir -p data/selfplay/mcts_cluster
for host in "$GH200A_TS" "$GH200A_DIRECT"; do
  rsync -avz --timeout=60 ubuntu@$host:~/ringrift/ai-service/data/selfplay/mcts_*/games.jsonl \
    data/selfplay/mcts_cluster/ >> "$LOG_FILE" 2>&1 && break || true
done

echo "[$TIMESTAMP] Sync complete" >> "$LOG_FILE"

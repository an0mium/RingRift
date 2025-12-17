#!/bin/bash
# Automated training data sync from GH200 cluster
# Run via cron: */15 * * * * /path/to/sync_training_data_cron.sh

set -e
cd "$(dirname "$0")/.."

LOG_FILE="logs/sync_cron.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$TIMESTAMP] Starting data sync..." >> "$LOG_FILE"

# Sync from primary data aggregator (GH200-a)
rsync -avz --timeout=60 ubuntu@192.222.51.29:~/ringrift/ai-service/data/games/jsonl_aggregated.db \
  data/games/gh200a_synced.db >> "$LOG_FILE" 2>&1 || echo "[$TIMESTAMP] GH200-a sync failed" >> "$LOG_FILE"

# Update symlink
ln -sf gh200a_synced.db data/games/all_jsonl_training.db

# Sync MCTS data if available
rsync -avz --timeout=60 ubuntu@192.222.51.29:~/ringrift/ai-service/data/selfplay/mcts_*/games.jsonl \
  data/selfplay/mcts_cluster/ >> "$LOG_FILE" 2>&1 || true

echo "[$TIMESTAMP] Sync complete" >> "$LOG_FILE"

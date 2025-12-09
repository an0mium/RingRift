#!/bin/bash
# Vast.ai Streaming Selfplay Script
# Runs selfplay and streams completed games to Lambda in real-time
#
# Usage: ./vast_streaming_selfplay.sh [--games 100] [--workers 8] [--sync-interval 30]
#
# This script runs on the Vast.ai instance and:
# 1. Runs selfplay workers that write to individual JSONL files
# 2. A background sync process that tails the JSONL files and streams new lines to Lambda
# 3. Uses rsync with append mode for efficient incremental transfer

set -e

# Configuration
GAMES_PER_WORKER=${GAMES_PER_WORKER:-50}
WORKERS=${WORKERS:-8}
SYNC_INTERVAL=${SYNC_INTERVAL:-30}  # Seconds between syncs
LAMBDA_HOST="lambda-gpu"
LAMBDA_DATA_DIR="/home/ubuntu/ringrift/ai-service/data/collected/streaming"
INSTANCE_NAME=$(hostname | head -c 20)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --games) GAMES_PER_WORKER="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --sync-interval) SYNC_INTERVAL="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

log() {
    echo "[$(date '+%H:%M:%S')] $1"
}

log "=== Vast.ai Streaming Selfplay ==="
log "Instance: $INSTANCE_NAME"
log "Games per worker: $GAMES_PER_WORKER"
log "Workers: $WORKERS"
log "Sync interval: ${SYNC_INTERVAL}s"
log "Lambda target: $LAMBDA_HOST:$LAMBDA_DATA_DIR"

# Setup directories
cd ~/ringrift/ai-service
source venv/bin/activate 2>/dev/null || true
mkdir -p logs/selfplay/streaming data/streaming

# Ensure Lambda directory exists
ssh $LAMBDA_HOST "mkdir -p $LAMBDA_DATA_DIR" 2>/dev/null || log "Warning: Could not create Lambda directory"

# Create master output file that all workers append to
MASTER_JSONL="logs/selfplay/streaming/${INSTANCE_NAME}_$(date '+%Y%m%d_%H%M%S').jsonl"
touch "$MASTER_JSONL"

# Track sync position for each file
SYNC_POS_FILE="/tmp/sync_pos_${INSTANCE_NAME}"
echo "0" > "$SYNC_POS_FILE"

# Background sync function - runs continuously
sync_to_lambda() {
    local master_file="$1"
    local remote_file="${LAMBDA_DATA_DIR}/$(basename $master_file)"

    while true; do
        # Get current file size
        local current_size=$(stat -c%s "$master_file" 2>/dev/null || echo "0")
        local last_pos=$(cat "$SYNC_POS_FILE" 2>/dev/null || echo "0")

        if [ "$current_size" -gt "$last_pos" ]; then
            # New data available - sync it
            local new_bytes=$((current_size - last_pos))
            log "Syncing $new_bytes new bytes to Lambda..."

            # Use tail to get new content and append to remote
            tail -c +$((last_pos + 1)) "$master_file" | ssh $LAMBDA_HOST "cat >> $remote_file" 2>/dev/null

            if [ $? -eq 0 ]; then
                echo "$current_size" > "$SYNC_POS_FILE"
                local lines=$(wc -l < "$master_file")
                log "Synced. Total games: $lines"
            else
                log "Sync failed, will retry"
            fi
        fi

        sleep "$SYNC_INTERVAL"
    done
}

# Start background sync process
log "Starting background sync process..."
sync_to_lambda "$MASTER_JSONL" &
SYNC_PID=$!
log "Sync PID: $SYNC_PID"

# Cleanup on exit
cleanup() {
    log "Shutting down..."
    kill $SYNC_PID 2>/dev/null || true

    # Final sync
    log "Final sync..."
    local current_size=$(stat -c%s "$MASTER_JSONL" 2>/dev/null || echo "0")
    local last_pos=$(cat "$SYNC_POS_FILE" 2>/dev/null || echo "0")
    if [ "$current_size" -gt "$last_pos" ]; then
        tail -c +$((last_pos + 1)) "$MASTER_JSONL" | ssh $LAMBDA_HOST "cat >> ${LAMBDA_DATA_DIR}/$(basename $MASTER_JSONL)" 2>/dev/null
    fi

    log "Done. Total games: $(wc -l < $MASTER_JSONL)"
}
trap cleanup EXIT

# Start selfplay workers - all append to the same master file
log "Starting $WORKERS selfplay workers..."
BASE_SEED=$((RANDOM * 1000 + $$))

for i in $(seq 1 $WORKERS); do
    python3 scripts/run_self_play_soak.py \
        --num-games $GAMES_PER_WORKER \
        --board-type square8 \
        --engine-mode descent-only \
        --num-players 2 \
        --max-moves 500 \
        --seed $((BASE_SEED + i)) \
        --log-jsonl "$MASTER_JSONL" \
        --no-record-db \
        --include-training-data \
        >> logs/selfplay/streaming/worker_${i}.log 2>&1 &
    log "Started worker $i (PID: $!)"
done

log "All workers started. Waiting for completion..."
wait

log "All workers complete!"

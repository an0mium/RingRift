#!/bin/bash
# Simple Elo DB synchronization using rsync
# Add to cron: */5 * * * * /path/to/sync_elo_db.sh
#
# Usage:
#   ./sync_elo_db.sh pull lambda-h100   # Pull from coordinator
#   ./sync_elo_db.sh push lambda-h100   # Push to coordinator

set -e

MODE=${1:-pull}
COORDINATOR=${2:-lambda-h100}
LOCAL_DB="${HOME}/ringrift/ai-service/data/unified_elo.db"
REMOTE_DB="~/ringrift/ai-service/data/unified_elo.db"
LOCK_FILE="/tmp/elo_sync.lock"

# Prevent concurrent syncs
exec 200>"$LOCK_FILE"
flock -n 200 || { echo "Sync already running"; exit 0; }

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

case "$MODE" in
    pull)
        log "Pulling Elo DB from $COORDINATOR"
        rsync -avz --timeout=30 "$COORDINATOR:$REMOTE_DB" "$LOCAL_DB.tmp" && \
            mv "$LOCAL_DB.tmp" "$LOCAL_DB"
        log "Pull complete"
        ;;
    push)
        log "Pushing Elo DB to $COORDINATOR"
        rsync -avz --timeout=30 "$LOCAL_DB" "$COORDINATOR:$REMOTE_DB"
        log "Push complete"
        ;;
    *)
        echo "Usage: $0 [pull|push] [coordinator_host]"
        exit 1
        ;;
esac

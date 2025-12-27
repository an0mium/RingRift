#!/bin/bash
# P2P Supervisor Loop - keeps P2P running with automatic restart
#
# Usage: ./p2p_supervisor.sh [node-id] [ai-service-path] [port]
#
# This script runs in the foreground and restarts P2P if it crashes.
# Best used with: nohup ./p2p_supervisor.sh &
# Or as a systemd service.

set -e

# Parse arguments with defaults
NODE_ID="${1:-${RINGRIFT_NODE_ID:-$(hostname -s)}}"
RINGRIFT_AI_SERVICE="${2:-${RINGRIFT_AI_SERVICE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
P2P_PORT="${3:-${RINGRIFT_P2P_PORT:-8770}}"

# Compute parent path (RingRift root)
RINGRIFT_PATH="${RINGRIFT_AI_SERVICE%/ai-service}"
if [ "$RINGRIFT_PATH" = "$RINGRIFT_AI_SERVICE" ]; then
    RINGRIFT_PATH="$RINGRIFT_AI_SERVICE"
fi

# Default peer seeds (voter nodes)
P2P_SEEDS="${RINGRIFT_P2P_SEEDS:-http://89.169.112.47:8770,http://135.181.39.239:8770,http://135.181.39.201:8770,http://208.167.249.164:8770}"

# Find Python executable
if [ -f "$RINGRIFT_AI_SERVICE/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_AI_SERVICE/venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="/usr/bin/python3"
fi

# Ensure log directory exists
mkdir -p "$RINGRIFT_AI_SERVICE/logs"

cd "$RINGRIFT_AI_SERVICE"
export PYTHONPATH="$RINGRIFT_AI_SERVICE"

echo "[$(date)] P2P Supervisor starting"
echo "[$(date)] Node ID: $NODE_ID"
echo "[$(date)] AI Service: $RINGRIFT_AI_SERVICE"
echo "[$(date)] Port: $P2P_PORT"
echo "[$(date)] Python: $PYTHON"

RESTART_COUNT=0
RESTART_DELAY=10

while true; do
    echo "[$(date)] Starting P2P orchestrator (restart #$RESTART_COUNT)..."

    # Run P2P orchestrator, capturing output
    $PYTHON scripts/p2p_orchestrator.py \
        --node-id "$NODE_ID" \
        --port "$P2P_PORT" \
        --peers "$P2P_SEEDS" \
        --ringrift-path "$RINGRIFT_PATH" \
        2>&1 | tee -a "$RINGRIFT_AI_SERVICE/logs/p2p.log" || true

    EXIT_CODE=$?
    RESTART_COUNT=$((RESTART_COUNT + 1))

    echo "[$(date)] P2P exited with code $EXIT_CODE"

    # Exponential backoff on repeated failures (max 5 minutes)
    if [ $RESTART_COUNT -gt 5 ]; then
        RESTART_DELAY=$((RESTART_DELAY * 2))
        if [ $RESTART_DELAY -gt 300 ]; then
            RESTART_DELAY=300
        fi
    fi

    echo "[$(date)] Restarting in $RESTART_DELAY seconds..."
    sleep $RESTART_DELAY

    # Reset delay after successful run (>60s)
    if [ $(($(date +%s) - $(date -d "60 seconds ago" +%s 2>/dev/null || echo 0))) -gt 60 ]; then
        RESTART_DELAY=10
    fi
done

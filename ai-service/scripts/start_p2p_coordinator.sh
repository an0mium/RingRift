#!/bin/bash
# Start P2P orchestrator in coordinator-only mode
# This script sources .env.local to pick up RINGRIFT_IS_COORDINATOR=true

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

# Source .env.local for local machine configuration
if [ -f "$AI_SERVICE_DIR/.env.local" ]; then
    set -a  # Export all variables
    source "$AI_SERVICE_DIR/.env.local"
    set +a
    echo "Loaded $AI_SERVICE_DIR/.env.local"
fi

# Verify coordinator mode
if [ "$RINGRIFT_IS_COORDINATOR" = "true" ]; then
    echo "Running in COORDINATOR-ONLY mode (no compute tasks)"
else
    echo "Warning: RINGRIFT_IS_COORDINATOR is not set to 'true'"
fi

# Set required paths
export PYTHONPATH="$AI_SERVICE_DIR"
cd "$AI_SERVICE_DIR"

# Use the correct Python interpreter
PYTHON="${PYTHON:-/Users/armand/.pyenv/versions/3.10.13/bin/python}"

# Default settings
NODE_ID="${RINGRIFT_NODE_ID:-local-mac}"
PORT="${RINGRIFT_P2P_PORT:-8770}"
PEERS="${RINGRIFT_P2P_PEERS:-89.169.112.47:8770,46.62.147.150:8770,135.181.39.239:8770}"
LOG_FILE="${AI_SERVICE_DIR}/logs/p2p_coordinator.log"

echo "Starting P2P orchestrator..."
echo "  Node ID: $NODE_ID"
echo "  Port: $PORT"
echo "  Coordinator mode: $RINGRIFT_IS_COORDINATOR"
echo "  Log file: $LOG_FILE"

# Ensure logs directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Start in foreground or background based on argument
if [ "$1" = "--daemon" ] || [ "$1" = "-d" ]; then
    echo "Starting as daemon..."
    nohup "$PYTHON" scripts/p2p_orchestrator.py \
        --node-id "$NODE_ID" \
        --port "$PORT" \
        --peers "$PEERS" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Started with PID $PID"
    echo "$PID" > "$AI_SERVICE_DIR/logs/p2p_coordinator.pid"
else
    exec "$PYTHON" scripts/p2p_orchestrator.py \
        --node-id "$NODE_ID" \
        --port "$PORT" \
        --peers "$PEERS"
fi

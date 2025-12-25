#!/bin/bash
# RingRift P2P Orchestrator Startup Script
# This script is called by systemd to start the P2P orchestrator.
# It handles environment variable normalization and path detection.

set -e

# Always read from config file to ensure variables are set
if [ -f /etc/ringrift/node.conf ]; then
    source /etc/ringrift/node.conf
fi

# Normalize path - handle both /home/ubuntu/ringrift and /home/ubuntu/ringrift/ai-service
RINGRIFT_PATH="${RINGRIFT_PATH:-/home/ubuntu/ringrift}"
RINGRIFT_PATH="${RINGRIFT_PATH%/ai-service}"
AI_SERVICE_PATH="$RINGRIFT_PATH/ai-service"

export PYTHONPATH="$AI_SERVICE_PATH"

# Find python - prefer venv, fall back to system
PY="$AI_SERVICE_PATH/venv/bin/python"
if [ ! -x "$PY" ]; then
    PY="/usr/bin/python3"
fi

cd "$AI_SERVICE_PATH"
exec "$PY" scripts/p2p_orchestrator.py \
    --node-id "${NODE_ID:-$(hostname)}" \
    --port "${P2P_PORT:-8770}" \
    --peers "${COORDINATOR_URL:-https://p2p.ringrift.ai,http://100.78.101.123:8770,http://100.88.176.74:8770}" \
    --ringrift-path "$RINGRIFT_PATH"

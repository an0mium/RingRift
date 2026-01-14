#!/bin/bash
# RingRift P2P Keepalive Script
# Ensures P2P orchestrator is always running
#
# Usage:
#   ./scripts/p2p_keepalive.sh
#
# Add to crontab for automatic recovery:
#   */5 * * * * /path/to/ringrift/ai-service/scripts/p2p_keepalive.sh
#
# Environment variables:
#   RINGRIFT_NODE_ID - Override auto-detected node ID
#   RINGRIFT_P2P_PORT - Override default port (8770)
#   RINGRIFT_P2P_PEERS - Override default peer list

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGRIFT_PATH="${SCRIPT_DIR%/scripts}"
P2P_PORT="${RINGRIFT_P2P_PORT:-8770}"
LOGFILE="$RINGRIFT_PATH/logs/p2p_keepalive.log"

# Auto-detect node ID from canonical file or fallbacks
# Priority (Jan 13, 2026 - P2P Cluster Stability Plan Phase 1):
#   0) /etc/ringrift/node-id (canonical, written by deployment)
#   1) RINGRIFT_NODE_ID env var
#   2) /etc/default/ringrift-p2p NODE_ID (legacy)
#   3) hostname -s (fallback with WARNING)
if [ -f /etc/ringrift/node-id ]; then
    # Priority 0: Canonical file (single source of truth)
    NODE_ID="$(cat /etc/ringrift/node-id 2>/dev/null | tr -d '[:space:]')"
    if [ -z "$NODE_ID" ]; then
        echo "[$(date)] WARNING: /etc/ringrift/node-id exists but is empty" >> "$LOGFILE"
    fi
fi

if [ -z "$NODE_ID" ] && [ -n "$RINGRIFT_NODE_ID" ]; then
    # Priority 1: Environment variable
    NODE_ID="$RINGRIFT_NODE_ID"
fi

if [ -z "$NODE_ID" ] && [ -f /etc/default/ringrift-p2p ]; then
    # Priority 2: Legacy systemd service config
    source /etc/default/ringrift-p2p 2>/dev/null
fi

if [ -z "$NODE_ID" ]; then
    # Priority 3: Hostname fallback (with warning)
    NODE_ID="$(hostname -s)"
    echo "[$(date)] WARNING: No canonical node ID found, using hostname '$NODE_ID'" >> "$LOGFILE"
    echo "[$(date)] Run 'sudo python scripts/provision_node_id.py --auto-detect' to fix" >> "$LOGFILE"
fi

# Validate node ID looks correct (warn if it looks like a hostname that doesn't match expected patterns)
if [[ "$NODE_ID" == ringrift-* ]] && [[ ! "$NODE_ID" =~ ^(lambda-|vast-|runpod-|nebius-|hetzner-|vultr-|mac-) ]]; then
    echo "[$(date)] WARNING: NODE_ID '$NODE_ID' appears to be hostname-derived. Set RINGRIFT_NODE_ID for proper identification." >> "$LOGFILE"
fi

# Default peers (coordinator nodes)
DEFAULT_PEERS="http://100.78.101.123:8770,http://100.88.176.74:8770,http://100.107.168.125:8770"
PEERS="${RINGRIFT_P2P_PEERS:-$DEFAULT_PEERS}"

# Ensure log directory exists
mkdir -p "$RINGRIFT_PATH/logs"

# Check if P2P is running and healthy
check_health() {
    curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1
}

if check_health; then
    # Already running and healthy
    exit 0
fi

# Log restart attempt
echo "[$(date)] P2P not running or unhealthy, starting..." >> "$LOGFILE"

# Kill any zombie process
pkill -9 -f "p2p_orchestrator.py" 2>/dev/null || true
sleep 1

# Find Python interpreter
if [ -f "$RINGRIFT_PATH/venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/venv/bin/python"
elif [ -f "$RINGRIFT_PATH/.venv/bin/python" ]; then
    PYTHON="$RINGRIFT_PATH/.venv/bin/python"
else
    PYTHON="$(which python3)"
fi

# Verify Python exists
if [ ! -x "$PYTHON" ]; then
    echo "[$(date)] ERROR: Python not found" >> "$LOGFILE"
    exit 1
fi

# Start P2P in background
cd "$RINGRIFT_PATH"
export PYTHONPATH="$RINGRIFT_PATH"

nohup "$PYTHON" scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$P2P_PORT" \
    --peers "$PEERS" \
    --ringrift-path "${RINGRIFT_PATH%/ai-service}" \
    >> "$RINGRIFT_PATH/logs/p2p.log" 2>&1 &

P2P_PID=$!
echo "[$(date)] P2P started with PID $P2P_PID" >> "$LOGFILE"

# Wait briefly and verify it started
sleep 3
if check_health; then
    echo "[$(date)] P2P health check passed" >> "$LOGFILE"
else
    echo "[$(date)] WARNING: P2P started but health check failed" >> "$LOGFILE"
fi

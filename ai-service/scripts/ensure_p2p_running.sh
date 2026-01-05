#!/bin/bash
# ensure_p2p_running.sh - Auto-recovery script for P2P orchestrator
#
# Session 17.28 (Jan 5, 2026) - Prevent cluster node disconnections
#
# Purpose: Automatically restart P2P orchestrator if it's not running.
# This prevents nodes from becoming unreachable after reboots or crashes.
#
# Installation (add to crontab):
#   */5 * * * * /path/to/ai-service/scripts/ensure_p2p_running.sh >> /path/to/logs/p2p_recovery.log 2>&1
#
# Example cron entries for different providers:
#   Lambda GH200:  */5 * * * * cd ~/ringrift/ai-service && ./scripts/ensure_p2p_running.sh >> logs/p2p_recovery.log 2>&1
#   Vast.ai:       */5 * * * * cd /workspace/ringrift/ai-service && ./scripts/ensure_p2p_running.sh >> logs/p2p_recovery.log 2>&1
#   Nebius:        */5 * * * * cd ~/ringrift/ai-service && ./scripts/ensure_p2p_running.sh >> logs/p2p_recovery.log 2>&1

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${AI_SERVICE_DIR}/logs"
LOG_FILE="${LOG_DIR}/p2p_recovery.log"
P2P_PORT="${RINGRIFT_P2P_PORT:-8770}"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [P2P_RECOVERY] $1" >> "$LOG_FILE"
}

# Check if P2P orchestrator is running
check_p2p_running() {
    if pgrep -f "p2p_orchestrator" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Check if P2P is responding on the health endpoint
check_p2p_health() {
    if curl -s --connect-timeout 5 "http://localhost:${P2P_PORT}/health" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Restart P2P orchestrator
restart_p2p() {
    log "Starting P2P orchestrator..."

    cd "$AI_SERVICE_DIR"

    # Find the correct Python interpreter
    PYTHON=""
    if [ -f "venv/bin/python" ]; then
        PYTHON="venv/bin/python"
    elif command -v python3 > /dev/null 2>&1; then
        PYTHON="python3"
    elif command -v python > /dev/null 2>&1; then
        PYTHON="python"
    else
        log "ERROR: No Python interpreter found"
        return 1
    fi

    # Kill any zombie processes first
    pkill -f "p2p_orchestrator" 2>/dev/null || true
    sleep 2

    # Start the orchestrator
    PYTHONPATH="${AI_SERVICE_DIR}" nohup "$PYTHON" scripts/p2p_orchestrator.py >> "${LOG_DIR}/p2p_orchestrator.log" 2>&1 &

    # Wait for startup
    sleep 10

    # Verify it started
    if check_p2p_running; then
        log "P2P orchestrator started successfully (PID: $(pgrep -f 'p2p_orchestrator'))"
        return 0
    else
        log "ERROR: P2P orchestrator failed to start"
        return 1
    fi
}

# Main logic
main() {
    # Check if already running and healthy
    if check_p2p_running; then
        if check_p2p_health; then
            # All good, nothing to do
            exit 0
        else
            log "WARNING: P2P process exists but health check failed, restarting..."
        fi
    else
        log "P2P orchestrator not running, starting recovery..."
    fi

    # Attempt restart
    if restart_p2p; then
        log "Recovery completed successfully"

        # Wait and verify health
        sleep 5
        if check_p2p_health; then
            log "P2P orchestrator is healthy"
        else
            log "WARNING: P2P started but health check still failing"
        fi
    else
        log "ERROR: Recovery failed"
        exit 1
    fi
}

main "$@"

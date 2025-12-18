#!/bin/bash
# Mac Keepalive Script - Prevents Mac nodes from sleeping and keeps services running
# Deploy on all Mac cluster nodes (mac-studio, macbook-pro-1, macbook-pro-2, etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$AI_SERVICE_ROOT/logs/mac_keepalive.log"
PID_FILE="/tmp/ringrift_mac_keepalive.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Prevent Mac from sleeping
prevent_sleep() {
    # Check if caffeinate is already running for this script
    if pgrep -f "caffeinate.*ringrift" >/dev/null 2>&1; then
        return 0
    fi

    # Start caffeinate in background to prevent sleep
    # -d: prevent display sleep
    # -i: prevent idle sleep
    # -s: prevent system sleep (on AC power)
    caffeinate -dis &
    CAFFEINATE_PID=$!
    log "Started caffeinate (PID: $CAFFEINATE_PID) to prevent sleep"
    echo $CAFFEINATE_PID > /tmp/ringrift_caffeinate.pid
}

# Disable system sleep via pmset (requires sudo, run once manually)
configure_power_settings() {
    if command -v pmset >/dev/null 2>&1; then
        log "Power settings configuration (run manually with sudo if needed):"
        log "  sudo pmset -a sleep 0           # Disable sleep"
        log "  sudo pmset -a disksleep 0       # Disable disk sleep"
        log "  sudo pmset -a displaysleep 15   # Display can sleep after 15 min"
        log "  sudo pmset -a powernap 0        # Disable Power Nap"
    fi
}

# Check and restart P2P orchestrator if needed
check_p2p_orchestrator() {
    if ! pgrep -f "p2p_orchestrator" >/dev/null 2>&1; then
        log "P2P orchestrator not running, starting..."
        cd "$AI_SERVICE_ROOT"

        # Try to find Python
        PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
        if [ -z "$PYTHON" ]; then
            log "ERROR: Python not found"
            return 1
        fi

        # Get hostname for node-id
        NODE_ID=$(hostname | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9-]/-/g')

        nohup $PYTHON scripts/p2p_orchestrator.py --node-id "$NODE_ID" >> "$AI_SERVICE_ROOT/logs/p2p_orchestrator.log" 2>&1 &
        log "Started P2P orchestrator with node-id: $NODE_ID (PID: $!)"
    fi
}

# Check Tailscale connectivity
check_tailscale() {
    if command -v tailscale >/dev/null 2>&1; then
        STATUS=$(tailscale status --json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('BackendState', 'Unknown'))" 2>/dev/null || echo "Unknown")
        if [ "$STATUS" != "Running" ]; then
            log "WARNING: Tailscale status is $STATUS, attempting restart..."
            if [ "$(uname)" = "Darwin" ]; then
                # macOS - restart via launchctl
                sudo launchctl kickstart -k system/com.tailscale.tailscaled 2>/dev/null || true
            fi
        fi
    fi
}

# Send keepalive ping to prevent network timeout
send_keepalive_ping() {
    # Ping Lambda to keep Tailscale connection alive
    LAMBDA_IP="100.97.104.89"  # lambda-2xh100
    if ping -c 1 -W 2 "$LAMBDA_IP" >/dev/null 2>&1; then
        : # Connection alive
    else
        log "WARNING: Cannot reach Lambda ($LAMBDA_IP)"
    fi
}

# Main keepalive loop
main_loop() {
    log "Starting Mac keepalive daemon..."

    # Store PID
    echo $$ > "$PID_FILE"

    # Prevent sleep
    prevent_sleep

    # Show power config instructions
    configure_power_settings

    while true; do
        check_p2p_orchestrator
        check_tailscale
        send_keepalive_ping

        # Sleep for 60 seconds between checks
        sleep 60
    done
}

# Cleanup on exit
cleanup() {
    log "Shutting down Mac keepalive daemon..."
    if [ -f /tmp/ringrift_caffeinate.pid ]; then
        kill $(cat /tmp/ringrift_caffeinate.pid) 2>/dev/null || true
        rm -f /tmp/ringrift_caffeinate.pid
    fi
    rm -f "$PID_FILE"
}
trap cleanup EXIT

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Mac keepalive already running (PID: $OLD_PID)"
        exit 0
    fi
fi

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Run main loop
main_loop

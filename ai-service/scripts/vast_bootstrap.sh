#!/bin/bash
# ==============================================================================
# Vast.ai Instance Bootstrap Script
# ==============================================================================
#
# This script sets up a vast.ai instance for RingRift selfplay and P2P cluster.
# It handles:
#   1. Code deployment (git clone or update)
#   2. Dependency installation
#   3. P2P orchestrator setup and startup
#   4. Selfplay worker configuration
#
# Usage:
#   # Run on a vast instance via SSH:
#   curl -sL https://raw.githubusercontent.com/.../vast_bootstrap.sh | bash
#
#   # Or copy and run:
#   scp scripts/vast_bootstrap.sh root@<vast-ip>:/tmp/ && ssh root@<vast-ip> "bash /tmp/vast_bootstrap.sh"
#
# Environment variables (optional):
#   RINGRIFT_NODE_ID    - Override auto-detected node ID
#   RINGRIFT_PEERS      - Comma-separated peer URLs (default: lambda-a10)
#   RINGRIFT_GIT_REPO   - Git repo URL (if using git clone)
#   RINGRIFT_SKIP_GIT   - Set to 1 to skip git operations
#
# ==============================================================================

set -e

# Configuration
RINGRIFT_ROOT="${RINGRIFT_ROOT:-/root/RingRift}"
AI_SERVICE="${RINGRIFT_ROOT}/ai-service"
P2P_PORT="${P2P_PORT:-8770}"
LOG_DIR="/var/log/ringrift"
CONF_DIR="/etc/ringrift"

# Default peers for vast nodes
# Priority order:
#   1. Cloudflare tunnel (most reliable - bypasses NAT, stable URL)
#   2. AWS proxy (relay hub for NAT-blocked nodes)
#   3. Public IPs (fallback if tunnel is down)
#
# Cloudflare tunnel: p2p.ringrift.ai -> lambda-gh200-e:8770
# AWS proxy: 52.15.114.79 (relay hub)
# lambda-gh200-e: 192.222.57.162 (direct fallback)
# lambda-gh200-a: 192.222.51.29 (direct fallback)
DEFAULT_PEERS="https://p2p.ringrift.ai,http://52.15.114.79:8770,http://192.222.57.162:8770,http://192.222.51.29:8770"

# Relay peers - these receive relay heartbeats for NAT-blocked vast nodes
# Both Cloudflare tunnel and AWS proxy can relay heartbeats to the cluster
RELAY_PEERS="https://p2p.ringrift.ai,http://52.15.114.79:8770"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ==============================================================================
# Node ID Detection
# ==============================================================================
detect_node_id() {
    if [ -n "$RINGRIFT_NODE_ID" ]; then
        echo "$RINGRIFT_NODE_ID"
        return
    fi

    # Try Tailscale hostname first
    if command -v tailscale &> /dev/null; then
        local ts_name
        ts_name=$(tailscale status --self --json 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('Self',{}).get('DNSName','').split('.')[0])" 2>/dev/null || true)
        if [ -n "$ts_name" ] && [ "$ts_name" != "null" ]; then
            echo "$ts_name"
            return
        fi
    fi

    # Fall back to hostname with vast prefix
    local hostname_short
    hostname_short=$(hostname | cut -c1-12)
    echo "vast-${hostname_short}"
}

# ==============================================================================
# Dependency Check/Install
# ==============================================================================
install_dependencies() {
    log_info "Checking dependencies..."

    # Python packages needed for P2P
    local packages="pyyaml aiohttp psutil requests"

    pip3 install -q $packages 2>/dev/null || {
        log_warn "pip install failed, trying with --user"
        pip3 install --user -q $packages
    }

    log_info "Dependencies installed"
}

# ==============================================================================
# Code Deployment
# ==============================================================================
deploy_code() {
    log_info "Checking code deployment..."

    if [ -d "$AI_SERVICE/scripts/p2p_orchestrator.py" ] || [ -f "$AI_SERVICE/scripts/p2p_orchestrator.py" ]; then
        log_info "Code already deployed at $AI_SERVICE"

        # Update if git is available and not skipped
        if [ -z "$RINGRIFT_SKIP_GIT" ] && [ -d "$AI_SERVICE/.git" ]; then
            log_info "Pulling latest code..."
            cd "$AI_SERVICE"
            git pull --ff-only 2>/dev/null || log_warn "Git pull failed (non-fatal)"
        fi
    else
        log_info "Code not found, attempting deployment..."
        mkdir -p "$RINGRIFT_ROOT"

        if [ -n "$RINGRIFT_GIT_REPO" ]; then
            log_info "Cloning from $RINGRIFT_GIT_REPO"
            git clone --depth 1 "$RINGRIFT_GIT_REPO" "$RINGRIFT_ROOT"
        else
            log_error "Code not deployed and no git repo specified."
            log_error "Please deploy code manually or set RINGRIFT_GIT_REPO"
            exit 1
        fi
    fi
}

# ==============================================================================
# P2P Service Setup
# ==============================================================================
setup_p2p_service() {
    local node_id="$1"
    local peers="${RINGRIFT_PEERS:-$DEFAULT_PEERS}"

    log_info "Setting up P2P service for node: $node_id"

    # Create directories
    mkdir -p "$LOG_DIR" "$CONF_DIR"

    # Write node configuration
    cat > "$CONF_DIR/node.conf" << EOF
# RingRift Node Configuration
# Generated by vast_bootstrap.sh on $(date)
NODE_ID=$node_id
P2P_PORT=$P2P_PORT
PEERS=$peers
RELAY_PEERS=$RELAY_PEERS
RINGRIFT_PATH=$RINGRIFT_ROOT
AI_SERVICE_PATH=$AI_SERVICE
EOF

    log_info "Configuration written to $CONF_DIR/node.conf"

    # Create startup script
    cat > "$CONF_DIR/start_p2p.sh" << 'STARTSCRIPT'
#!/bin/bash
source /etc/ringrift/node.conf

# Kill any existing P2P process
pkill -f "p2p_orchestrator.py" 2>/dev/null || true
sleep 2

# Find Python - try venv, then conda, then system
PY="${AI_SERVICE_PATH}/venv/bin/python"
if [ ! -x "$PY" ]; then PY="/opt/conda/bin/python3"; fi
if [ ! -x "$PY" ]; then PY="/usr/bin/python3"; fi

# Start P2P orchestrator
cd "$AI_SERVICE_PATH"
export PYTHONPATH="$AI_SERVICE_PATH:$PYTHONPATH"

# Build relay-peers argument if supported and set
RELAY_ARG=""
if [ -n "$RELAY_PEERS" ]; then
    # Check if --relay-peers is supported by checking help output
    if "$PY" scripts/p2p_orchestrator.py --help 2>&1 | grep -q "relay-peers"; then
        RELAY_ARG="--relay-peers $RELAY_PEERS"
    fi
fi

exec "$PY" scripts/p2p_orchestrator.py \
    --node-id "$NODE_ID" \
    --port "$P2P_PORT" \
    --peers "$PEERS" \
    $RELAY_ARG \
    --ringrift-path "$RINGRIFT_PATH" \
    >> /var/log/ringrift/p2p.log 2>&1
STARTSCRIPT

    chmod +x "$CONF_DIR/start_p2p.sh"

    # Create cron job for auto-restart (works in Docker containers without systemd)
    cat > /etc/cron.d/ringrift-p2p << EOF
# RingRift P2P auto-restart
# Check every 5 minutes if P2P is running, restart if not
*/5 * * * * root pgrep -f p2p_orchestrator.py > /dev/null || /etc/ringrift/start_p2p.sh &
EOF

    log_info "Cron job created for P2P auto-restart"
}

# ==============================================================================
# Start P2P Service
# ==============================================================================
start_p2p() {
    log_info "Starting P2P service..."

    # Kill any existing
    pkill -f "p2p_orchestrator.py" 2>/dev/null || true
    sleep 2

    # Start in background
    nohup /etc/ringrift/start_p2p.sh &

    # Wait and verify
    sleep 5
    if pgrep -f "p2p_orchestrator.py" > /dev/null; then
        log_info "P2P service started successfully"

        # Show status
        local status
        status=$(curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/status" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Role: {d.get(\"role\")}, Leader: {d.get(\"leader_id\")}')" 2>/dev/null || echo "Status check failed")
        log_info "P2P Status: $status"
    else
        log_error "P2P service failed to start. Check /var/log/ringrift/p2p.log"
        tail -20 /var/log/ringrift/p2p.log 2>/dev/null || true
        exit 1
    fi
}

# ==============================================================================
# Main
# ==============================================================================
main() {
    log_info "=============================================="
    log_info "RingRift Vast.ai Bootstrap"
    log_info "=============================================="

    # Detect node ID
    NODE_ID=$(detect_node_id)
    log_info "Node ID: $NODE_ID"

    # Install dependencies
    install_dependencies

    # Deploy/update code
    deploy_code

    # Setup P2P service
    setup_p2p_service "$NODE_ID"

    # Start P2P
    start_p2p

    log_info "=============================================="
    log_info "Bootstrap complete!"
    log_info "Node ID: $NODE_ID"
    log_info "P2P Port: $P2P_PORT"
    log_info "Log file: $LOG_DIR/p2p.log"
    log_info "=============================================="
}

main "$@"

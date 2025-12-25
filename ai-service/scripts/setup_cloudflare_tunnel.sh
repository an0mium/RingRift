#!/bin/bash
# ==============================================================================
# Cloudflare Tunnel Setup for RingRift P2P
# ==============================================================================
#
# This script sets up a Cloudflare tunnel for the P2P orchestrator.
#
# Two modes:
#   1. Quick tunnel (default): Free, no account needed, URL changes on restart
#   2. Named tunnel (--token): Persistent URL, requires Cloudflare account
#
# Usage:
#   ./setup_cloudflare_tunnel.sh [port]              # Quick tunnel
#   ./setup_cloudflare_tunnel.sh --token TOKEN       # Named tunnel with token
#
# To create a named tunnel:
#   1. Go to https://one.dash.cloudflare.com/
#   2. Navigate to: Access > Tunnels > Create a tunnel
#   3. Name it 'ringrift-p2p', copy the token
#   4. Add public hostname: p2p.yourdomain.com -> http://localhost:8770
#
# ==============================================================================

set -e

P2P_PORT="8770"
TUNNEL_TOKEN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            TUNNEL_TOKEN="$2"
            shift 2
            ;;
        --port)
            P2P_PORT="$2"
            shift 2
            ;;
        [0-9]*)
            P2P_PORT="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Try to get token from file if not provided
if [ -z "$TUNNEL_TOKEN" ]; then
    for tokenfile in /workspace/ringrift/ai-service/config/.cloudflare-tunnel-token \
                     ~/ringrift/ai-service/config/.cloudflare-tunnel-token \
                     /etc/ringrift/.cloudflare-tunnel-token; do
        if [ -f "$tokenfile" ]; then
            TUNNEL_TOKEN=$(cat "$tokenfile")
            echo "Using tunnel token from $tokenfile"
            break
        fi
    done
fi
LOG_FILE="/var/log/ringrift/cloudflared.log"
TUNNEL_URL_FILE="/etc/ringrift/tunnel_url"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    log_info "Installing cloudflared..."
    curl -fsSL https://pkg.cloudflare.com/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    dpkg -i /tmp/cloudflared.deb || apt-get install -f -y
    rm /tmp/cloudflared.deb
fi

# Check if P2P is running
if ! curl -s --connect-timeout 5 "http://localhost:$P2P_PORT/health" > /dev/null 2>&1; then
    log_error "P2P orchestrator not running on port $P2P_PORT"
    exit 1
fi

# Kill any existing tunnels
log_info "Stopping any existing tunnels..."
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 2

mkdir -p "$(dirname $LOG_FILE)" "$(dirname $TUNNEL_URL_FILE)" 2>/dev/null || true

if [ -n "$TUNNEL_TOKEN" ]; then
    # Named tunnel with token (persistent URL)
    log_info "Starting Cloudflare named tunnel..."
    nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" > "$LOG_FILE" 2>&1 &
    TUNNEL_PID=$!
    TUNNEL_MODE="named"

    # Wait for tunnel to connect
    log_info "Waiting for tunnel to connect..."
    for i in {1..30}; do
        if grep -q "Connection.*registered" "$LOG_FILE" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    TUNNEL_URL="(configured in Cloudflare dashboard)"
else
    # Quick tunnel (ephemeral URL)
    log_info "Starting Cloudflare quick tunnel for port $P2P_PORT..."
    nohup cloudflared tunnel --url "http://127.0.0.1:$P2P_PORT" > "$LOG_FILE" 2>&1 &
    TUNNEL_PID=$!
    TUNNEL_MODE="quick"

    # Wait for tunnel URL
    log_info "Waiting for tunnel URL..."
    for i in {1..30}; do
        TUNNEL_URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" "$LOG_FILE" 2>/dev/null | head -1)
        if [ -n "$TUNNEL_URL" ]; then
            break
        fi
        sleep 1
    done

    if [ -z "$TUNNEL_URL" ]; then
        log_error "Failed to get tunnel URL. Check $LOG_FILE"
        cat "$LOG_FILE" 2>/dev/null | tail -20
        exit 1
    fi

    # Save tunnel URL
    echo "$TUNNEL_URL" > "$TUNNEL_URL_FILE"

    # Test tunnel
    log_info "Testing tunnel..."
    sleep 3
    if curl -s --connect-timeout 10 "$TUNNEL_URL/health" > /dev/null 2>&1; then
        log_info "Tunnel is working!"
    else
        log_warn "Tunnel may take a few more seconds to propagate"
    fi
fi

log_info "=============================================="
log_info "Cloudflare Tunnel Setup Complete"
log_info "=============================================="
log_info "Mode: $TUNNEL_MODE"
log_info "Tunnel PID: $TUNNEL_PID"
log_info "Log file: $LOG_FILE"
if [ "$TUNNEL_MODE" = "quick" ]; then
    log_info "Tunnel URL: $TUNNEL_URL"
    log_info ""
    log_info "To use this tunnel from Vast nodes:"
    log_info "  export P2P_RELAY_URL=$TUNNEL_URL"
    log_info ""
    log_info "Note: Quick tunnel URLs change on restart."
    log_info "For persistent URLs, use --token with a named tunnel."
else
    log_info ""
    log_info "Named tunnel connected. Configure public hostname in Cloudflare dashboard:"
    log_info "  https://one.dash.cloudflare.com/ > Access > Tunnels"
    log_info ""
    log_info "Add public hostname: p2p.yourdomain.com -> http://localhost:$P2P_PORT"
fi
log_info "=============================================="

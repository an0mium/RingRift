#!/bin/bash
# Setup Tailscale on Vast.ai container for P2P connectivity
# Usage: ./setup_vast_tailscale.sh [--authkey KEY] [--hostname NAME]

set -e

# Parse arguments
AUTHKEY=""
HOSTNAME="vast-$(hostname | cut -d- -f2 2>/dev/null || echo $$)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --authkey)
            AUTHKEY="$2"
            shift 2
            ;;
        --hostname)
            HOSTNAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Try to get authkey from file if not provided
if [ -z "$AUTHKEY" ]; then
    for keyfile in /workspace/ringrift/ai-service/config/.tailscale-authkey \
                   ~/ringrift/ai-service/config/.tailscale-authkey \
                   ~/.tailscale-authkey; do
        if [ -f "$keyfile" ]; then
            AUTHKEY=$(cat "$keyfile")
            echo "Using authkey from $keyfile"
            break
        fi
    done
fi

if [ -z "$AUTHKEY" ]; then
    echo "ERROR: No Tailscale auth key provided"
    echo "Usage: $0 --authkey tskey-auth-xxxxx"
    echo ""
    echo "Generate an auth key at: https://login.tailscale.com/admin/settings/keys"
    echo "- Reusable: Yes"
    echo "- Ephemeral: Yes"
    echo "- Tags: tag:vast-nodes (optional)"
    exit 1
fi

echo "=== Setting up Tailscale on Vast container ==="
echo "Hostname: $HOSTNAME"

# Install Tailscale if not present
if ! command -v tailscale &> /dev/null; then
    echo "Installing Tailscale..."
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Check if tailscaled is running
if ! pgrep -x tailscaled > /dev/null; then
    echo "Starting tailscaled in userspace mode..."
    mkdir -p /var/run/tailscale /var/lib/tailscale

    # Kill any existing tailscaled
    pkill tailscaled 2>/dev/null || true
    sleep 1

    # Start tailscaled in background
    nohup tailscaled \
        --tun=userspace-networking \
        --statedir=/var/lib/tailscale \
        --socket=/var/run/tailscale/tailscaled.sock \
        > /tmp/tailscaled.log 2>&1 &

    echo "Waiting for tailscaled to start..."
    sleep 3
fi

# Check tailscaled status
SOCKET="/var/run/tailscale/tailscaled.sock"
if [ ! -S "$SOCKET" ]; then
    echo "ERROR: tailscaled socket not found at $SOCKET"
    echo "Check /tmp/tailscaled.log for errors"
    cat /tmp/tailscaled.log 2>/dev/null | tail -20
    exit 1
fi

# Connect to Tailscale
echo "Connecting to Tailscale..."
tailscale --socket="$SOCKET" up \
    --authkey="$AUTHKEY" \
    --hostname="$HOSTNAME" \
    --accept-routes \
    --accept-dns=false

# Verify connection
echo ""
echo "=== Tailscale Status ==="
tailscale --socket="$SOCKET" status

# Get our Tailscale IP
TAILSCALE_IP=$(tailscale --socket="$SOCKET" ip -4 2>/dev/null || echo "unknown")
echo ""
echo "Tailscale IP: $TAILSCALE_IP"

# Test connectivity to Lambda nodes
echo ""
echo "=== Testing Lambda connectivity ==="
for ip in 100.78.101.123 100.97.104.89 100.83.234.82; do
    if curl -s --connect-timeout 3 "http://$ip:8770/health" > /dev/null 2>&1; then
        echo "  $ip: OK (P2P reachable)"
    else
        echo "  $ip: FAILED"
    fi
done

echo ""
echo "=== Setup Complete ==="
echo "Your Vast node is now connected to the Tailscale network"
echo "Update P2P config to use this node's Tailscale IP: $TAILSCALE_IP"

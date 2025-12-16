#!/bin/bash
#
# Setup reverse SSH tunnel from Vast instance to H100 relay
#
# This creates a persistent reverse tunnel so the H100 (coordinator) can:
# - SSH into Vast instances through the tunnel
# - Send commands and sync data
# - Coordinate distributed tasks
#
# Architecture:
#   Vast Instance <--reverse tunnel--> H100 (Relay/Coordinator)
#   - Vast opens tunnel: ssh -R <port>:localhost:22 ubuntu@H100
#   - H100 can then: ssh -p <port> root@localhost
#
# Usage on Vast instance:
#   RELAY_HOST=209.20.157.81 TUNNEL_PORT=20001 bash setup_vast_relay.sh
#
# Or run remotely:
#   ssh root@ssh5.vast.ai -p 14364 "RELAY_HOST=209.20.157.81 TUNNEL_PORT=20001 bash -s" < setup_vast_relay.sh

set -e

# Configuration
RELAY_HOST="${RELAY_HOST:-209.20.157.81}"  # H100 IP
RELAY_USER="${RELAY_USER:-ubuntu}"
TUNNEL_PORT="${TUNNEL_PORT:-}"  # Will be auto-assigned if not set
SSH_KEY="${SSH_KEY:-/root/.ssh/id_ed25519}"
AUTOSSH_LOG="/tmp/autossh.log"

echo "=== Vast Instance Reverse Tunnel Setup ==="
echo "Relay Host: $RELAY_HOST"
echo "Relay User: $RELAY_USER"

# ============================================
# Step 1: Generate SSH key if needed
# ============================================
setup_ssh_key() {
    echo ""
    echo "=== Setting up SSH key ==="

    if [ ! -f "$SSH_KEY" ]; then
        echo "Generating new SSH key..."
        ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "vast-$(hostname)"
    fi

    echo "SSH public key:"
    cat "${SSH_KEY}.pub"

    echo ""
    echo "==> Add this key to $RELAY_USER@$RELAY_HOST:~/.ssh/authorized_keys"
    echo "    Or run: ssh-copy-id -i $SSH_KEY $RELAY_USER@$RELAY_HOST"
}

# ============================================
# Step 2: Install autossh for persistent tunnels
# ============================================
install_autossh() {
    echo ""
    echo "=== Installing autossh ==="

    if command -v autossh &> /dev/null; then
        echo "autossh already installed"
        return 0
    fi

    apt-get update -qq
    apt-get install -y -qq autossh

    echo "autossh installed"
}

# ============================================
# Step 3: Assign unique tunnel port
# ============================================
assign_tunnel_port() {
    echo ""
    echo "=== Assigning tunnel port ==="

    if [ -n "$TUNNEL_PORT" ]; then
        echo "Using specified port: $TUNNEL_PORT"
        return 0
    fi

    # Generate port based on hostname hash (20001-20999)
    HASH=$(echo "$(hostname)" | md5sum | cut -c1-4)
    TUNNEL_PORT=$((20001 + (0x$HASH % 999)))

    echo "Auto-assigned port: $TUNNEL_PORT"
}

# ============================================
# Step 4: Create tunnel service
# ============================================
create_tunnel_service() {
    echo ""
    echo "=== Creating reverse tunnel ==="

    # Kill any existing autossh
    pkill -f "autossh.*$RELAY_HOST" 2>/dev/null || true

    # Create tunnel command
    TUNNEL_CMD="autossh -M 0 -f -N -o 'ServerAliveInterval 30' -o 'ServerAliveCountMax 3' \
        -o 'StrictHostKeyChecking no' -o 'ExitOnForwardFailure yes' \
        -i $SSH_KEY -R $TUNNEL_PORT:localhost:22 $RELAY_USER@$RELAY_HOST"

    echo "Starting tunnel: $TUNNEL_CMD"

    # Start the tunnel
    eval "$TUNNEL_CMD" >> "$AUTOSSH_LOG" 2>&1

    sleep 2

    # Verify tunnel is running
    if pgrep -f "autossh.*$RELAY_HOST" > /dev/null; then
        echo "Tunnel established successfully!"
        echo ""
        echo "From H100, you can now SSH to this instance:"
        echo "  ssh -p $TUNNEL_PORT root@localhost"
    else
        echo "ERROR: Tunnel failed to start. Check $AUTOSSH_LOG"
        return 1
    fi
}

# ============================================
# Step 5: Register with P2P coordinator
# ============================================
register_tunnel() {
    echo ""
    echo "=== Registering tunnel with coordinator ==="

    # Get instance info
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "0")

    # Write tunnel info to file for P2P registration
    TUNNEL_INFO="/tmp/tunnel_info.json"
    cat > "$TUNNEL_INFO" << EOF
{
    "node_id": "vast-$(hostname | cut -c1-8)",
    "tunnel_port": $TUNNEL_PORT,
    "relay_host": "$RELAY_HOST",
    "relay_user": "$RELAY_USER",
    "gpu_name": "$GPU_NAME",
    "gpu_memory": "$GPU_MEM",
    "ssh_command": "ssh -p $TUNNEL_PORT root@localhost"
}
EOF

    echo "Tunnel info saved to $TUNNEL_INFO"
    cat "$TUNNEL_INFO"

    # Try to register with P2P API
    P2P_URL="http://$RELAY_HOST:8765"
    echo ""
    echo "Registering with P2P at $P2P_URL..."

    curl -s -X POST "$P2P_URL/api/v1/nodes/register" \
        -H "Content-Type: application/json" \
        -d @"$TUNNEL_INFO" 2>/dev/null && echo "Registered!" || echo "Registration failed (P2P may not be running)"
}

# ============================================
# Step 6: Create startup script
# ============================================
create_startup_script() {
    echo ""
    echo "=== Creating startup script ==="

    STARTUP_SCRIPT="/root/start_tunnel.sh"
    cat > "$STARTUP_SCRIPT" << EOF
#!/bin/bash
# Auto-start reverse tunnel to H100

# Kill existing tunnels
pkill -f "autossh.*$RELAY_HOST" 2>/dev/null || true

# Wait for network
sleep 5

# Start tunnel
autossh -M 0 -f -N -o 'ServerAliveInterval 30' -o 'ServerAliveCountMax 3' \\
    -o 'StrictHostKeyChecking no' -o 'ExitOnForwardFailure yes' \\
    -i $SSH_KEY -R $TUNNEL_PORT:localhost:22 $RELAY_USER@$RELAY_HOST

echo "Tunnel started on port $TUNNEL_PORT"
EOF

    chmod +x "$STARTUP_SCRIPT"
    echo "Startup script created: $STARTUP_SCRIPT"

    # Add to crontab for persistence
    (crontab -l 2>/dev/null | grep -v "start_tunnel.sh"; echo "@reboot $STARTUP_SCRIPT") | crontab -

    echo "Added to crontab for auto-restart"
}

# ============================================
# Main
# ============================================
main() {
    setup_ssh_key
    install_autossh
    assign_tunnel_port
    create_tunnel_service
    register_tunnel
    create_startup_script

    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "Node: vast-$(hostname | cut -c1-8)"
    echo "Tunnel Port: $TUNNEL_PORT"
    echo "Relay Host: $RELAY_HOST"
    echo ""
    echo "To connect from H100:"
    echo "  ssh -p $TUNNEL_PORT root@localhost"
    echo ""
    echo "To check tunnel status:"
    echo "  pgrep -a autossh"
}

main "$@"

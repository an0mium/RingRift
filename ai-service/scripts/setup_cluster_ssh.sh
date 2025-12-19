#!/bin/bash
# Setup SSH keys for cluster access
# Run this script at the start of each session to ensure SSH connectivity

set -e

echo "=== RingRift Cluster SSH Setup ==="

# Start SSH agent if not running
if [ -z "$SSH_AUTH_SOCK" ]; then
    echo "Starting SSH agent..."
    eval "$(ssh-agent -s)"
fi

# Add cluster key (no passphrase)
if ! ssh-add -l 2>/dev/null | grep -q "id_cluster"; then
    echo "Adding cluster key..."
    ssh-add ~/.ssh/id_cluster 2>/dev/null || true
fi

# Verify keys loaded
echo "Loaded SSH keys:"
ssh-add -l 2>/dev/null || echo "No keys loaded"

# Quick connectivity test
echo ""
echo "Testing cluster connectivity..."
NODES="lambda-gh200-d lambda-gh200-e lambda-gh200-f lambda-gh200-g lambda-h100"
for node in $NODES; do
    if ssh -o ConnectTimeout=3 -o BatchMode=yes "$node" "echo OK" 2>/dev/null; then
        echo "  ✓ $node"
    else
        echo "  ✗ $node (unreachable)"
    fi
done

echo ""
echo "SSH setup complete."

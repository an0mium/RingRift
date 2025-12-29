#!/bin/bash
# Setup Node.js on cluster nodes for TypeScript parity gates
#
# Usage:
#   # Install on a specific node
#   ssh <node> 'bash -s' < scripts/setup_node_on_cluster.sh
#
#   # Or run remotely
#   ssh <node> "curl -sL https://raw.githubusercontent.com/.../setup_node_on_cluster.sh | bash"
#
# December 28, 2025

set -e

echo "=========================================="
echo "Setting up Node.js for RingRift parity gates"
echo "=========================================="

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS=$(uname -s)
fi

echo "Detected OS: $OS"

# Install Node.js based on OS
case "$OS" in
    ubuntu|debian)
        echo "Installing Node.js via NodeSource..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
        ;;
    centos|rhel|fedora)
        echo "Installing Node.js via NodeSource..."
        curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
        sudo yum install -y nodejs
        ;;
    Darwin)
        echo "Installing Node.js via Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Installing..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install node
        ;;
    *)
        echo "Unknown OS: $OS"
        echo "Trying generic Node.js installation via nvm..."

        # Install nvm
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

        # Load nvm
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

        # Install Node.js
        nvm install 20
        nvm use 20
        ;;
esac

# Verify installation
echo ""
echo "Verifying installation..."
node --version
npm --version
npx --version

# Install TypeScript globally
echo ""
echo "Installing TypeScript and ts-node..."
npm install -g typescript ts-node

# Verify ts-node
ts-node --version

echo ""
echo "=========================================="
echo "Node.js setup complete!"
echo "=========================================="
echo ""
echo "You can now run parity gates with:"
echo "  cd ~/ringrift && TS_NODE_PROJECT=tsconfig.server.json npx ts-node -T scripts/selfplay-db-ts-replay.ts --help"

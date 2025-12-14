#!/bin/bash
# RingRift Cluster Update Script
# Updates all AWS and local cluster machines to latest code from GitHub
#
# Usage: ./scripts/cluster-update.sh [--dry-run] [--aws-only] [--local-only]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_URL="git@github.com:yourusername/RingRift.git"
BRANCH="main"
REMOTE_DIR="/home/ubuntu/ringrift"
LOCAL_REMOTE_DIR="~/Development/RingRift"

# AWS instances (from ~/.ssh/config)
AWS_HOSTS=(
    "ringrift-staging"
    "ringrift-selfplay-extra"
)

# Lambda GPU instances (primary training)
LAMBDA_HOSTS=(
    "lambda-h100"
    "lambda-2xh100"
    "lambda-a10"
)

# Local cluster machines (from ~/.ssh/config)
LOCAL_HOSTS=(
    "mac-studio"
    "m1-pro"
    "macbook-pro-3"
)

# Parse arguments
DRY_RUN=false
AWS_ONLY=false
LOCAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --aws-only)
            AWS_ONLY=true
            shift
            ;;
        --local-only)
            LOCAL_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--aws-only] [--local-only]"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if host is reachable
check_host() {
    local host=$1
    ssh -o ConnectTimeout=5 -o BatchMode=yes "$host" "echo ok" &>/dev/null
}

# Update a remote host
update_host() {
    local host=$1
    local remote_dir=$2

    log_info "Updating $host..."

    if ! check_host "$host"; then
        log_warning "$host is not reachable, skipping"
        return 1
    fi

    if $DRY_RUN; then
        log_info "[DRY RUN] Would update $host"
        return 0
    fi

    # Check for running Python processes
    running=$(ssh -o ConnectTimeout=10 "$host" "ps aux | grep -E 'python.*selfplay|python.*train' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    if [[ "$running" -gt 0 ]]; then
        log_warning "$host has $running running Python processes - update may interrupt work"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping $host"
            return 0
        fi
    fi

    # Perform git pull
    ssh -o ConnectTimeout=30 "$host" "cd $remote_dir && git fetch origin && git reset --hard origin/$BRANCH" 2>&1

    if [[ $? -eq 0 ]]; then
        log_success "$host updated successfully"

        # Show latest commit
        ssh -o ConnectTimeout=10 "$host" "cd $remote_dir && git log -1 --oneline" 2>&1
        return 0
    else
        log_error "Failed to update $host"
        return 1
    fi
}

# Sync from local to remote (for machines without git access)
sync_host() {
    local host=$1
    local remote_dir=$2

    log_info "Syncing to $host via rsync..."

    if ! check_host "$host"; then
        log_warning "$host is not reachable, skipping"
        return 1
    fi

    if $DRY_RUN; then
        log_info "[DRY RUN] Would sync to $host"
        return 0
    fi

    # Sync ai-service directory (most important for selfplay/training)
    rsync -avz --delete \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='venv' \
        --exclude='*.db' \
        --exclude='*.npz' \
        --exclude='logs/' \
        --exclude='data/games/' \
        --exclude='models/*.pth' \
        ai-service/ "$host:$remote_dir/ai-service/"

    if [[ $? -eq 0 ]]; then
        log_success "$host synced successfully"
        return 0
    else
        log_error "Failed to sync to $host"
        return 1
    fi
}

echo "============================================"
echo "RingRift Cluster Update Script"
echo "============================================"
echo ""

if $DRY_RUN; then
    log_warning "DRY RUN MODE - no changes will be made"
    echo ""
fi

# Summary of what will be updated
echo "Targets:"
if ! $LOCAL_ONLY; then
    echo "  AWS: ${AWS_HOSTS[*]}"
fi
if ! $AWS_ONLY; then
    echo "  Local: ${LOCAL_HOSTS[*]}"
fi
echo ""

# Update AWS instances
if ! $LOCAL_ONLY; then
    echo "=== AWS Instances ==="
    for host in "${AWS_HOSTS[@]}"; do
        update_host "$host" "$REMOTE_DIR" || true
    done
    echo ""
fi

# Update Lambda GPU instances (primary training hosts)
if ! $LOCAL_ONLY; then
    echo "=== Lambda GPU Instances ==="
    for host in "${LAMBDA_HOSTS[@]}"; do
        update_host "$host" "~/ringrift" || true
    done
    echo ""
fi

# Update local cluster machines
if ! $AWS_ONLY; then
    echo "=== Local Cluster ==="
    for host in "${LOCAL_HOSTS[@]}"; do
        sync_host "$host" "$LOCAL_REMOTE_DIR" || true
    done
    echo ""
fi

echo "============================================"
log_success "Cluster update complete!"
echo "============================================"

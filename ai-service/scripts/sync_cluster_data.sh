#!/bin/bash
# sync_cluster_data.sh - Sync selfplay game data from all cluster hosts
#
# This script syncs selfplay game databases (.db) and JSONL files from all
# cluster hosts defined in config/distributed_hosts.yaml to a target directory.
#
# Common use cases:
#   1. Sync to external drive (OWS/WD): --target /Volumes/RingRift-Data/selfplay_repository/raw
#   2. Sync to local data directory: --target ./data/games/synced
#   3. Sync to training machine: --target /path/to/training/data
#
# Usage:
#   ./scripts/sync_cluster_data.sh [OPTIONS]
#
# Options:
#   --target PATH   Target directory for synced data (default: ./data/games/cluster_sync)
#   --config FILE   Hosts config file (default: config/distributed_hosts.yaml)
#   --dry-run       Show what would be synced without actually syncing
#   --parallel N    Sync from N hosts in parallel (default: 4)
#   -h, --help      Show this help message
#
# Configuration:
#   Hosts are read from config/distributed_hosts.yaml. Each host entry should have:
#     - ssh_host: IP or hostname
#     - ssh_user: SSH username
#     - ssh_port: (optional) SSH port, default 22
#     - ssh_key: (optional) path to SSH key
#     - ringrift_path: path to ringrift/ai-service on the remote host
#
# The script automatically handles:
#   - Standard SSH hosts (port 22, persistent storage)
#   - Vast.ai instances (custom ports, RAM storage at /dev/shm)
#   - Failed/unreachable hosts (continues with remaining hosts)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$AI_SERVICE_DIR/config/distributed_hosts.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default target (local cluster sync directory)
TARGET_DIR="$AI_SERVICE_DIR/data/games/cluster_sync"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --target) TARGET_DIR="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --help|-h)
            head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "  RingRift Cluster Data Sync"
echo "  Timestamp: $TIMESTAMP"
echo "  Target: $TARGET_DIR"
echo "  Config: $CONFIG_FILE"
[[ "$DRY_RUN" == "true" ]] && echo "  Mode: DRY RUN"
echo "============================================="

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    log_info "Copy config/distributed_hosts.template.yaml to config/distributed_hosts.yaml"
    exit 1
fi

# Verify target parent is accessible (create if needed)
PARENT_DIR="$(dirname "$TARGET_DIR")"
if [[ ! -d "$PARENT_DIR" ]]; then
    log_info "Creating parent directory: $PARENT_DIR"
    mkdir -p "$PARENT_DIR" || {
        log_error "Cannot create target parent directory: $PARENT_DIR"
        log_info "If syncing to external drive, ensure it's mounted"
        exit 1
    }
fi

mkdir -p "$TARGET_DIR"

# Track stats
SYNC_SUCCESS=0
SYNC_FAILED=0
TOTAL_FILES=0

# Generic sync function
sync_from_host() {
    local name="$1"
    local ssh_host="$2"
    local ssh_user="$3"
    local remote_path="$4"
    local ssh_key="${5:-}"
    local ssh_port="${6:-22}"

    log_info "Syncing from $name..."
    local dest="$TARGET_DIR/$name"
    mkdir -p "$dest"

    local ssh_opts="-o ConnectTimeout=15 -o StrictHostKeyChecking=no -o BatchMode=yes"
    [[ -n "$ssh_key" ]] && ssh_opts="$ssh_opts -i $ssh_key"
    [[ "$ssh_port" != "22" ]] && ssh_opts="$ssh_opts -p $ssh_port"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] rsync -avz -e 'ssh $ssh_opts' $ssh_user@$ssh_host:$remote_path/data/games/*.db $dest/"
        return 0
    fi

    local synced_any=false

    # Try standard path first
    if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_path/data/games/"*.db "$dest/" 2>/dev/null; then
        synced_any=true
    fi
    if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:$remote_path/data/games/"*.jsonl "$dest/" 2>/dev/null; then
        synced_any=true
    fi

    # For Vast.ai instances, also try RAM storage at /dev/shm
    if [[ "$ssh_user" == "root" ]]; then
        if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:/dev/shm/games/"*.db "$dest/" 2>/dev/null; then
            synced_any=true
        fi
        if rsync -avz --progress -e "ssh $ssh_opts" "$ssh_user@$ssh_host:/dev/shm/games/"*.jsonl "$dest/" 2>/dev/null; then
            synced_any=true
        fi
    fi

    if [[ "$synced_any" == "true" ]]; then
        local db_count=$(find "$dest" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
        local jsonl_count=$(find "$dest" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
        log_success "$name: $db_count DB(s), $jsonl_count JSONL"
        ((SYNC_SUCCESS++)) || true
        ((TOTAL_FILES += db_count + jsonl_count)) || true
    else
        log_warning "$name: no data or unreachable"
        ((SYNC_FAILED++)) || true
    fi
}

# ============================================
# PARSE CONFIG AND SYNC
# ============================================

log_section "Reading host configuration"

# Use Python to parse YAML and extract host details
if ! command -v python3 &>/dev/null; then
    log_error "Python3 required for YAML parsing"
    exit 1
fi

# Generate sync commands from config
SYNC_COMMANDS=$(python3 - "$CONFIG_FILE" << 'PYTHON_SCRIPT'
import sys
import os

config_file = sys.argv[1]

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

with open(config_file) as f:
    config = yaml.safe_load(f)

hosts = config.get("hosts", {})

for name, host_cfg in hosts.items():
    # Skip disabled/commented hosts
    status = host_cfg.get("status", "ready")
    if status not in ("ready", "setup"):
        continue

    ssh_host = host_cfg.get("ssh_host", "")
    ssh_user = host_cfg.get("ssh_user", "")
    ssh_port = host_cfg.get("ssh_port", 22)
    ssh_key = host_cfg.get("ssh_key", "")
    ringrift_path = host_cfg.get("ringrift_path", "~/ringrift/ai-service")

    # Expand ~ in paths
    ssh_key = ssh_key.replace("~", os.environ.get("HOME", "~"))

    if ssh_host and ssh_user:
        print(f'sync_from_host "{name}" "{ssh_host}" "{ssh_user}" "{ringrift_path}" "{ssh_key}" "{ssh_port}"')
PYTHON_SCRIPT
) || {
    log_error "Failed to parse config file"
    exit 1
}

HOST_COUNT=$(echo "$SYNC_COMMANDS" | grep -c 'sync_from_host' || echo "0")
log_info "Found $HOST_COUNT hosts in configuration"

log_section "Syncing from all hosts"

# Execute sync commands
while IFS= read -r cmd; do
    [[ -n "$cmd" ]] && eval "$cmd"
done <<< "$SYNC_COMMANDS"

# ============================================
# SUMMARY
# ============================================

log_section "Sync Summary"
echo "  Successful: $SYNC_SUCCESS hosts"
echo "  Failed/Empty: $SYNC_FAILED hosts"
echo "  Total files synced: $TOTAL_FILES"
echo ""

# Disk usage summary
if [[ -d "$TARGET_DIR" ]]; then
    log_info "Target directory status:"
    du -sh "$TARGET_DIR" 2>/dev/null || true

    # Count total games if sqlite3 is available
    if command -v sqlite3 &>/dev/null; then
        log_info "Counting games in synced databases..."
        TOTAL_GAMES=0
        while IFS= read -r db; do
            count=$(sqlite3 "$db" "SELECT COUNT(*) FROM games;" 2>/dev/null || echo "0")
            ((TOTAL_GAMES += count)) || true
        done < <(find "$TARGET_DIR" -name "*.db" 2>/dev/null)
        log_success "Total games across all synced DBs: $TOTAL_GAMES"
    fi
fi

log_section "Done! $(date)"

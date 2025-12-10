#!/bin/bash
# sync_to_mac_studio.sh - Sync selfplay data from cloud instances directly to Mac Studio
#
# This script syncs game databases from all remote cloud hosts directly to Mac Studio,
# bypassing the local laptop to save disk space.
#
# Usage:
#   ./scripts/sync_to_mac_studio.sh [OPTIONS]
#
# Options:
#   --merge           After syncing, merge all DBs into a single merged.db on Mac Studio
#   --cleanup         Remove synced subdirectories after merging (keeps only merged DB)
#   --dry-run         Show what would be synced without actually syncing
#   -h, --help        Show this help message
#
# Prerequisites:
#   - SSH config entry for 'mac-studio' (see ~/.ssh/config)
#   - SSH access to all cloud instances from this machine (for tunneling)
#   - Mac Studio must be reachable from cloud instances OR we use this machine as relay
#
# Data Flow:
#   Cloud Host -> (this laptop as relay) -> Mac Studio
#   OR
#   Cloud Host -> Mac Studio (if direct SSH access exists)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

# Mac Studio configuration
MAC_STUDIO_HOST="${MAC_STUDIO_HOST:-mac-studio}"
MAC_STUDIO_DATA_DIR="~/Development/RingRift/ai-service/data/games"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAC_STUDIO_SYNC_DIR="$MAC_STUDIO_DATA_DIR/synced_${TIMESTAMP}"

# Colors for output
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
DO_MERGE=false
DO_CLEANUP=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --merge) DO_MERGE=true; shift ;;
        --cleanup) DO_CLEANUP=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
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
echo "  RingRift Selfplay Data Sync -> Mac Studio"
echo "  Timestamp: $TIMESTAMP"
echo "============================================="

# Check SSH connection to Mac Studio
log_info "Testing connection to Mac Studio..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$MAC_STUDIO_HOST" echo "Connected" 2>/dev/null; then
    log_error "Cannot connect to $MAC_STUDIO_HOST"
    echo ""
    echo "Make sure Mac Studio is reachable and SSH is configured:"
    echo "  ssh $MAC_STUDIO_HOST"
    exit 1
fi
log_success "Mac Studio is reachable"

# Create sync directory on Mac Studio
if [[ "$DRY_RUN" != "true" ]]; then
    ssh "$MAC_STUDIO_HOST" "mkdir -p $MAC_STUDIO_SYNC_DIR"
fi

# Track success/failure
SYNC_SUCCESS=0
SYNC_FAILED=0

# Function to sync from a cloud host through this machine to Mac Studio
# This uses a "relay" approach: pull to temp dir, then push to Mac Studio
sync_via_relay() {
    local source_host="$1"
    local source_path="$2"
    local dest_subdir="$3"
    local ssh_opts="${4:-}"

    log_info "Syncing from $source_host to Mac Studio..."

    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" RETURN

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] rsync $ssh_opts -avz $source_host:$source_path/*.db $tmp_dir/"
        echo "  [DRY-RUN] rsync -avz $tmp_dir/ $MAC_STUDIO_HOST:$MAC_STUDIO_SYNC_DIR/$dest_subdir/"
        return 0
    fi

    local synced_any=false

    # Pull DB files from source to temp
    if rsync -avz --progress $ssh_opts "$source_host:$source_path/"*.db "$tmp_dir/" 2>/dev/null; then
        synced_any=true
    fi

    # Pull JSONL files from source to temp
    if rsync -avz --progress $ssh_opts "$source_host:$source_path/"*.jsonl "$tmp_dir/" 2>/dev/null; then
        synced_any=true
    fi

    if [[ "$synced_any" == "true" ]]; then
        # Push from temp to Mac Studio
        ssh "$MAC_STUDIO_HOST" "mkdir -p $MAC_STUDIO_SYNC_DIR/$dest_subdir"
        if rsync -avz --progress "$tmp_dir/" "$MAC_STUDIO_HOST:$MAC_STUDIO_SYNC_DIR/$dest_subdir/"; then
            local db_count=$(find "$tmp_dir" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
            local jsonl_count=$(find "$tmp_dir" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
            log_success "$source_host -> Mac Studio: $db_count database(s), $jsonl_count JSONL file(s)"
            ((SYNC_SUCCESS++)) || true
        else
            log_error "Failed to push to Mac Studio from $source_host"
            ((SYNC_FAILED++)) || true
        fi
    else
        log_warning "$source_host: no data found or sync failed"
        ((SYNC_FAILED++)) || true
    fi
}

# Function to sync from Vast instances (custom SSH port)
sync_vast_via_relay() {
    local host="$1"
    local port="$2"
    local name="$3"
    local remote_path="${4:-/dev/shm/games}"

    log_info "Syncing from $name (Vast, port $port) to Mac Studio..."

    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" RETURN

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY-RUN] rsync -e \"ssh -p $port\" -avz root@$host:$remote_path/*.db $tmp_dir/"
        echo "  [DRY-RUN] rsync -avz $tmp_dir/ $MAC_STUDIO_HOST:$MAC_STUDIO_SYNC_DIR/$name/"
        return 0
    fi

    local synced_any=false

    # Pull from Vast to temp
    if rsync -avz --progress -e "ssh -p $port -o ConnectTimeout=15" "root@$host:$remote_path/"*.db "$tmp_dir/" 2>/dev/null; then
        synced_any=true
    fi

    if rsync -avz --progress -e "ssh -p $port -o ConnectTimeout=15" "root@$host:$remote_path/"*.jsonl "$tmp_dir/" 2>/dev/null; then
        synced_any=true
    fi

    if [[ "$synced_any" == "true" ]]; then
        # Push from temp to Mac Studio
        ssh "$MAC_STUDIO_HOST" "mkdir -p $MAC_STUDIO_SYNC_DIR/$name"
        if rsync -avz --progress "$tmp_dir/" "$MAC_STUDIO_HOST:$MAC_STUDIO_SYNC_DIR/$name/"; then
            local db_count=$(find "$tmp_dir" -name "*.db" 2>/dev/null | wc -l | tr -d ' ')
            local jsonl_count=$(find "$tmp_dir" -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
            log_success "$name -> Mac Studio: $db_count database(s), $jsonl_count JSONL file(s)"
            ((SYNC_SUCCESS++)) || true
        else
            log_error "Failed to push to Mac Studio from $name"
            ((SYNC_FAILED++)) || true
        fi
    else
        log_warning "$name: no data found or instance may be terminated"
        ((SYNC_FAILED++)) || true
    fi
}

# ============================================
# SYNC FROM CLOUD HOSTS TO MAC STUDIO
# ============================================

log_section "Syncing from cloud hosts to Mac Studio"

# AWS/Lambda instances
sync_via_relay "ringrift-staging" "~/ringrift/ai-service/data/games" "aws_staging"
sync_via_relay "ringrift-selfplay-extra" "~/ringrift/ai-service/data/games" "aws_extra"

# Lambda GPU
sync_via_relay "ubuntu@150.136.65.197" "~/ringrift/ai-service/data/games" "lambda"

# Vast instances (ephemeral, update ports as needed)
# NOTE: Vast instances change frequently - update these entries when provisioning new ones
# sync_vast_via_relay "211.72.13.202" "45875" "vast_4x5090"
# sync_vast_via_relay "178.43.61.252" "18080" "vast_2x5090"
# sync_vast_via_relay "79.116.93.241" "47070" "vast_1x3090"

# ============================================
# SUMMARY
# ============================================

log_section "Sync Summary"
echo "  Successful: $SYNC_SUCCESS"
echo "  Failed/Empty: $SYNC_FAILED"
echo "  Output directory: $MAC_STUDIO_HOST:$MAC_STUDIO_SYNC_DIR"

if [[ "$DRY_RUN" != "true" ]]; then
    # List all synced files on Mac Studio
    TOTAL_DBS=$(ssh "$MAC_STUDIO_HOST" "find $MAC_STUDIO_SYNC_DIR -name '*.db' 2>/dev/null | wc -l" | tr -d ' ')
    TOTAL_JSONL=$(ssh "$MAC_STUDIO_HOST" "find $MAC_STUDIO_SYNC_DIR -name '*.jsonl' 2>/dev/null | wc -l" | tr -d ' ')
    log_info "Total synced to Mac Studio: $TOTAL_DBS database(s), $TOTAL_JSONL JSONL file(s)"
fi

# ============================================
# MERGE ON MAC STUDIO (optional)
# ============================================

if [[ "$DO_MERGE" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
    log_section "Merging Databases on Mac Studio"

    MERGED_DB="$MAC_STUDIO_DATA_DIR/merged_${TIMESTAMP}.db"

    # Run merge script on Mac Studio
    if ssh "$MAC_STUDIO_HOST" "cd ~/Development/RingRift/ai-service && python scripts/merge_game_dbs.py --output $MERGED_DB --glob '$MAC_STUDIO_SYNC_DIR/**/*.db' --on-conflict skip" 2>/dev/null; then
        log_success "Merged database: $MAC_STUDIO_HOST:$MERGED_DB"

        # Get game count
        GAME_COUNT=$(ssh "$MAC_STUDIO_HOST" "sqlite3 '$MERGED_DB' 'SELECT COUNT(*) FROM games;'" 2>/dev/null || echo "?")
        log_info "Total games in merged DB: $GAME_COUNT"

        # Create symlink to latest
        ssh "$MAC_STUDIO_HOST" "rm -f '$MAC_STUDIO_DATA_DIR/merged_latest.db' && ln -s '$MERGED_DB' '$MAC_STUDIO_DATA_DIR/merged_latest.db'"
        log_info "Created symlink: merged_latest.db -> $(basename "$MERGED_DB")"

        # Cleanup synced directories if requested
        if [[ "$DO_CLEANUP" == "true" ]]; then
            log_info "Cleaning up synced directories..."
            ssh "$MAC_STUDIO_HOST" "rm -rf '$MAC_STUDIO_SYNC_DIR'"
            log_success "Removed $MAC_STUDIO_SYNC_DIR"
        fi
    else
        log_error "Failed to merge databases on Mac Studio"
    fi
fi

log_section "Done!"
echo ""
echo "Data is now on Mac Studio at: $MAC_STUDIO_SYNC_DIR"
echo ""
echo "To run training on Mac Studio:"
echo "  ssh $MAC_STUDIO_HOST"
echo "  cd ~/Development/RingRift/ai-service"
echo "  python app/training/train.py --device mps"

#!/bin/bash
# RingRift AI Training Automation - Cron Job Installer
#
# December 2025: Comprehensive automation cron setup.
# This script installs all recommended cron jobs for the AI training pipeline.
#
# Usage:
#   ./scripts/install_cron_jobs.sh           # Show recommended cron config
#   ./scripts/install_cron_jobs.sh --install # Install cron jobs
#   ./scripts/install_cron_jobs.sh --remove  # Remove RingRift cron jobs
#
# Categories:
#   1. Cluster Health      - Hourly health reports
#   2. Lambda Watchdog     - Hourly Lambda instance restart check
#   3. Data Sync           - Every 30 min selfplay data sync
#   4. Training Freshness  - Every 30 min data freshness check
#   5. Vast Maintenance    - Every 15 min Vast.ai keepalive
#   6. Model Distribution  - Hourly model sync across cluster
#   7. Diverse Selfplay    - Every 4 hours for priority configs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Generate recommended cron entries
generate_cron_entries() {
    cat << EOF
# ============================================================================
# RingRift AI Training Automation - Cron Jobs
# Generated: $(date)
# AI Service Path: $AI_SERVICE_DIR
# ============================================================================

# ----------------------------------------------------------------------------
# 1. CLUSTER HEALTH (Hourly) - Monitor cluster and alert on issues
# ----------------------------------------------------------------------------
0 * * * * cd $AI_SERVICE_DIR && python3 scripts/cluster_health_cli.py --report >> logs/cluster_health.log 2>&1

# ----------------------------------------------------------------------------
# 2. LAMBDA WATCHDOG (Hourly) - Restart terminated Lambda instances
# ----------------------------------------------------------------------------
0 * * * * cd $AI_SERVICE_DIR && python3 scripts/lambda_watchdog.py --auto-restart >> logs/lambda_watchdog.log 2>&1

# ----------------------------------------------------------------------------
# 3. DATA SYNC (Every 30 min) - Sync selfplay data across cluster
# ----------------------------------------------------------------------------
*/30 * * * * cd $AI_SERVICE_DIR && python3 scripts/unified_data_sync.py --once >> logs/data_sync.log 2>&1

# ----------------------------------------------------------------------------
# 4. TRAINING FRESHNESS (Every 30 min) - Check data freshness on training nodes
# This runs via the train.py --check-data-freshness flag during training
# The cron job is optional, but useful for proactive alerts
# ----------------------------------------------------------------------------
*/30 * * * * cd $AI_SERVICE_DIR && python3 -c "from app.coordination.training_freshness import get_freshness_checker; import asyncio; asyncio.run(get_freshness_checker().check_all_configs())" >> logs/training_freshness.log 2>&1

# ----------------------------------------------------------------------------
# 5. VAST.AI KEEPALIVE (Every 15 min) - Prevent idle termination
# ----------------------------------------------------------------------------
*/15 * * * * cd $AI_SERVICE_DIR && python3 scripts/vast_keepalive.py --auto >> logs/vast_keepalive.log 2>&1

# ----------------------------------------------------------------------------
# 6. VAST.AI P2P SYNC (Every 10 min) - Full orchestration cycle
# ----------------------------------------------------------------------------
*/10 * * * * cd $AI_SERVICE_DIR && python3 scripts/vast_p2p_sync.py --full >> logs/vast_p2p.log 2>&1

# ----------------------------------------------------------------------------
# 7. MODEL DISTRIBUTION (Hourly) - Sync promoted models across cluster
# Note: ModelDistributionDaemon also runs continuously via daemon_manager
# This cron is a backup for the daemon
# ----------------------------------------------------------------------------
0 * * * * cd $AI_SERVICE_DIR && python3 scripts/sync_models.py --distribute >> logs/model_sync.log 2>&1

# ----------------------------------------------------------------------------
# 8. DIVERSE SELFPLAY (Every 4 hours) - Priority configs selfplay
# Targets: hexagonal_2p, hexagonal_4p (fewest games)
# ----------------------------------------------------------------------------
0 */4 * * * cd $AI_SERVICE_DIR && ./scripts/cron_diverse_selfplay.sh >> logs/diverse_selfplay.log 2>&1

# ----------------------------------------------------------------------------
# 9. TRAINING PIPELINE (Daily at 4 AM) - Full training cycle
# ----------------------------------------------------------------------------
0 4 * * * cd $AI_SERVICE_DIR && ./scripts/cron_training.sh >> logs/cron_training.log 2>&1

# ----------------------------------------------------------------------------
# 10. LOG ROTATION (Daily at midnight) - Compress and archive old logs
# ----------------------------------------------------------------------------
0 0 * * * find $AI_SERVICE_DIR/logs -name "*.log" -mtime +7 -exec gzip {} \; 2>/dev/null

# ============================================================================
EOF
}

show_cron_config() {
    echo ""
    log_info "Recommended cron configuration for RingRift AI automation:"
    echo ""
    generate_cron_entries
    echo ""
    log_info "To install, run: $0 --install"
    log_info "To remove,  run: $0 --remove"
}

install_cron_jobs() {
    # Create logs directory
    mkdir -p "$AI_SERVICE_DIR/logs"

    log_info "Installing RingRift cron jobs..."

    # Backup existing crontab
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S) 2>/dev/null || true

    # Remove existing RingRift entries and add new ones
    (
        # Keep non-RingRift entries
        crontab -l 2>/dev/null | grep -v "# RingRift" | grep -v "ringrift" | grep -v "$AI_SERVICE_DIR" || true

        # Add new entries
        generate_cron_entries
    ) | crontab -

    log_info "Cron jobs installed successfully!"
    log_info "View with: crontab -l"
    log_info "Logs will be in: $AI_SERVICE_DIR/logs/"
}

remove_cron_jobs() {
    log_info "Removing RingRift cron jobs..."

    # Keep only non-RingRift entries
    (crontab -l 2>/dev/null | grep -v "# RingRift" | grep -v "ringrift" | grep -v "$AI_SERVICE_DIR" || true) | crontab -

    log_info "RingRift cron jobs removed."
}

# Main
case "${1:-}" in
    --install)
        install_cron_jobs
        ;;
    --remove)
        remove_cron_jobs
        ;;
    --help|-h)
        echo "Usage: $0 [--install|--remove|--help]"
        echo ""
        echo "Options:"
        echo "  (none)     Show recommended cron configuration"
        echo "  --install  Install cron jobs"
        echo "  --remove   Remove RingRift cron jobs"
        echo "  --help     Show this help"
        ;;
    *)
        show_cron_config
        ;;
esac

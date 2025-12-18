#!/bin/bash
# ============================================================================
# RingRift Work Rebalancing Cron Setup
# ============================================================================
#
# Sets up automated cluster work rebalancing to ensure:
# - GPU-heavy nodes run GPU work (training, tournaments)
# - CPU-rich nodes run CPU work (CMA-ES, selfplay)
# - Idle nodes get assigned appropriate work
#
# Usage:
#   ./scripts/setup_work_rebalancing_cron.sh          # Install cron jobs
#   ./scripts/setup_work_rebalancing_cron.sh --remove # Remove cron jobs
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

# Cron schedule:
# - Rebalance check every 10 minutes
# - Auto-training check every 5 minutes
# - Unified work orchestrator every 2 minutes

REBALANCE_CRON="*/10 * * * * cd $AI_SERVICE_DIR && python3 scripts/smart_work_router.py --rebalance --quiet >> /tmp/work_rebalance.log 2>&1"
AUTO_TRAIN_CRON="*/5 * * * * cd $AI_SERVICE_DIR && python3 scripts/auto_training_trigger.py >> /tmp/auto_training.log 2>&1"
ORCHESTRATOR_CRON="*/2 * * * * cd $AI_SERVICE_DIR && python3 scripts/unified_work_orchestrator.py --node-id \$(hostname) --once >> /tmp/work_orchestrator.log 2>&1"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

install_cron() {
    echo -e "${GREEN}Installing RingRift work rebalancing cron jobs...${NC}"

    # Get current crontab (suppress error if empty)
    crontab -l 2>/dev/null > /tmp/current_cron || true

    # Remove any existing RingRift cron entries
    grep -v "smart_work_router\|auto_training_trigger\|unified_work_orchestrator" /tmp/current_cron > /tmp/new_cron || true

    # Add header comment
    echo "" >> /tmp/new_cron
    echo "# RingRift Cluster Work Rebalancing" >> /tmp/new_cron

    # Add new cron jobs
    echo "$REBALANCE_CRON" >> /tmp/new_cron
    echo "$AUTO_TRAIN_CRON" >> /tmp/new_cron
    echo "$ORCHESTRATOR_CRON" >> /tmp/new_cron

    # Install new crontab
    crontab /tmp/new_cron
    rm -f /tmp/current_cron /tmp/new_cron

    echo -e "${GREEN}Cron jobs installed:${NC}"
    echo "  - Smart work router: every 10 minutes"
    echo "  - Auto training trigger: every 5 minutes"
    echo "  - Unified orchestrator: every 2 minutes"
    echo ""
    echo -e "${YELLOW}Verify with:${NC} crontab -l | grep -E 'smart_work|auto_train|orchestrator'"
}

remove_cron() {
    echo -e "${YELLOW}Removing RingRift work rebalancing cron jobs...${NC}"

    # Get current crontab
    crontab -l 2>/dev/null > /tmp/current_cron || true

    # Remove RingRift entries
    grep -v "smart_work_router\|auto_training_trigger\|unified_work_orchestrator\|RingRift Cluster Work" /tmp/current_cron > /tmp/new_cron || true

    # Install cleaned crontab
    crontab /tmp/new_cron
    rm -f /tmp/current_cron /tmp/new_cron

    echo -e "${GREEN}Cron jobs removed.${NC}"
}

deploy_to_cluster() {
    echo -e "${GREEN}Deploying cron setup to cluster nodes...${NC}"

    # Lambda nodes with Tailscale IPs
    LAMBDA_NODES=(
        "ubuntu@100.91.25.13"    # lambda-a10
        "ubuntu@100.78.101.123"  # lambda-h100
        "ubuntu@100.97.104.89"   # lambda-2xh100
        "ubuntu@100.123.183.70"  # lambda-gh200-a
        "ubuntu@100.104.165.116" # lambda-gh200-f
    )

    for node in "${LAMBDA_NODES[@]}"; do
        echo -n "  Deploying to ${node##*@}... "
        if ssh -o ConnectTimeout=10 -o BatchMode=yes $node "
            cd ~/ringrift/ai-service 2>/dev/null || cd ~/ai-service 2>/dev/null || exit 1

            # Install cron jobs
            crontab -l 2>/dev/null | grep -v 'smart_work_router\|auto_training_trigger\|unified_work_orchestrator' > /tmp/cron.tmp || true
            echo '# RingRift Work Rebalancing' >> /tmp/cron.tmp
            echo '*/10 * * * * cd ~/ringrift/ai-service && python3 scripts/smart_work_router.py --rebalance --quiet >> /tmp/work_rebalance.log 2>&1' >> /tmp/cron.tmp
            echo '*/5 * * * * cd ~/ringrift/ai-service && python3 scripts/auto_training_trigger.py >> /tmp/auto_training.log 2>&1' >> /tmp/cron.tmp
            echo '*/2 * * * * cd ~/ringrift/ai-service && python3 scripts/unified_work_orchestrator.py --node-id \$(hostname) --once >> /tmp/work_orchestrator.log 2>&1' >> /tmp/cron.tmp
            crontab /tmp/cron.tmp
            rm /tmp/cron.tmp
        " 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}FAILED${NC}"
        fi
    done
}

case "${1:-}" in
    --remove)
        remove_cron
        ;;
    --deploy)
        install_cron
        deploy_to_cluster
        ;;
    *)
        install_cron
        echo ""
        echo -e "${YELLOW}To deploy to all cluster nodes:${NC}"
        echo "  $0 --deploy"
        ;;
esac

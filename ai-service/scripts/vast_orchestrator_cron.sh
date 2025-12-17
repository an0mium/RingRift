#!/bin/bash
# Vast.ai instance orchestrator cron job
# Runs every 30 minutes to:
# 1. Health check all vast instances
# 2. Maintain P2P network connectivity
# 3. Sync game data from instances with games
# 4. Restart workers on idle/stuck instances
# 5. Send keepalive to prevent idle termination

set -e
cd /Users/armand/Development/RingRift/ai-service
source venv/bin/activate 2>/dev/null || true

LOG_FILE="logs/vast_orchestrator_cron.log"
mkdir -p logs

echo "========================================" >> "$LOG_FILE"
echo "$(date -Iseconds) Vast Orchestrator Starting" >> "$LOG_FILE"

# 1. Keepalive (fast, prevents termination)
echo "$(date -Iseconds) Running keepalive..." >> "$LOG_FILE"
timeout 60 python -B scripts/vast_keepalive.py --keepalive >> "$LOG_FILE" 2>&1 || true

# 2. P2P network maintenance (ensure mesh connectivity)
echo "$(date -Iseconds) Checking P2P network..." >> "$LOG_FILE"
timeout 120 python -B scripts/vast_p2p_setup.py --check-status >> "$LOG_FILE" 2>&1 || true

# 3. Full lifecycle management (health check, restart, sync)
echo "$(date -Iseconds) Running lifecycle manager..." >> "$LOG_FILE"
timeout 600 python -B scripts/vast_lifecycle.py --auto >> "$LOG_FILE" 2>&1 || true

# 4. Deploy P2P to any new instances
echo "$(date -Iseconds) Deploying P2P to new instances..." >> "$LOG_FILE"
timeout 180 python -B scripts/vast_p2p_setup.py --deploy-to-vast --components p2p >> "$LOG_FILE" 2>&1 || true

echo "$(date -Iseconds) Vast Orchestrator Complete" >> "$LOG_FILE"

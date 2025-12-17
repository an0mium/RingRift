#!/bin/bash
# Vast.ai instance orchestrator cron job
# Runs every 15-30 minutes to:
# 1. Health check all vast instances
# 2. Maintain P2P network connectivity
# 3. Sync game data from instances with games
# 4. Restart workers on idle/stuck instances
# 5. Send keepalive to prevent idle termination
# 6. Update distributed_hosts.yaml with current IPs
#
# Cron example (every 15 minutes):
#   */15 * * * * /Users/armand/Development/RingRift/ai-service/scripts/vast_orchestrator_cron.sh

set -e
cd /Users/armand/Development/RingRift/ai-service
source venv/bin/activate 2>/dev/null || true

LOG_FILE="logs/vast_orchestrator_cron.log"
mkdir -p logs

# Rotate log if > 10MB
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null || echo 0) -gt 10485760 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
fi

echo "========================================" >> "$LOG_FILE"
echo "$(date -Iseconds) Vast Orchestrator Starting" >> "$LOG_FILE"

# 1. Keepalive (fast, prevents termination)
echo "$(date -Iseconds) [1/5] Running keepalive..." >> "$LOG_FILE"
timeout 60 python -B scripts/vast_keepalive.py --keepalive >> "$LOG_FILE" 2>&1 || true

# 2. P2P Sync (update IPs, unretire nodes, ensure P2P running)
echo "$(date -Iseconds) [2/5] Running P2P sync..." >> "$LOG_FILE"
timeout 180 python -B scripts/vast_p2p_sync.py --full >> "$LOG_FILE" 2>&1 || true

# 3. P2P network maintenance (ensure mesh connectivity)
echo "$(date -Iseconds) [3/5] Checking P2P network..." >> "$LOG_FILE"
timeout 120 python -B scripts/vast_p2p_setup.py --check-status >> "$LOG_FILE" 2>&1 || true

# 4. Full lifecycle management (health check, restart, sync)
echo "$(date -Iseconds) [4/5] Running lifecycle manager..." >> "$LOG_FILE"
timeout 600 python -B scripts/vast_lifecycle.py --auto >> "$LOG_FILE" 2>&1 || true

# 5. Deploy P2P to any new instances
echo "$(date -Iseconds) [5/5] Deploying P2P to new instances..." >> "$LOG_FILE"
timeout 180 python -B scripts/vast_p2p_setup.py --deploy-to-vast --components p2p >> "$LOG_FILE" 2>&1 || true

echo "$(date -Iseconds) Vast Orchestrator Complete" >> "$LOG_FILE"

# Summary stats
VAST_RUNNING=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for i in d if i.get('actual_status')=='running'))" 2>/dev/null || echo "?")
echo "$(date -Iseconds) Summary: $VAST_RUNNING running Vast instances" >> "$LOG_FILE"

#!/bin/bash
# 10-hour cluster monitoring script
# Created by Claude Code
# Runs every 3-5 minutes, checks cluster health, restores to optimal state

set -e

LOG_FILE="/Users/armand/Development/RingRift/ai-service/logs/cluster_monitor_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

MONITOR_DURATION_HOURS=10
INTERVAL_SECONDS=240  # 4 minutes
END_TIME=$(($(date +%s) + MONITOR_DURATION_HOURS * 3600))

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_tailscale_nodes() {
    log "=== Tailscale Node Status ==="
    tailscale status 2>/dev/null | tee -a "$LOG_FILE" || log "Tailscale check failed"
}

check_cluster_nodes() {
    log "=== Cluster Node Status ==="
    ssh ubuntu@100.123.183.70 "cd ~/ringrift/ai-service && source venv/bin/activate && python scripts/cluster_submit.py status 2>/dev/null || echo 'Status check unavailable'" 2>&1 | tee -a "$LOG_FILE"
}

check_vast_instances() {
    log "=== Vast.ai Instance Status ==="
    vast show instances --raw 2>/dev/null | jq -r '.[] | "\(.id): \(.status) - GPU: \(.gpu_name) - Util: \(.gpu_util // "N/A")%"' 2>/dev/null | tee -a "$LOG_FILE" || log "Vast check skipped (no CLI or no instances)"
}

check_gpu_utilization() {
    log "=== GPU Utilization on Primary Node ==="
    ssh ubuntu@100.123.183.70 "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo 'GPU check failed'" 2>&1 | tee -a "$LOG_FILE"
}

check_running_jobs() {
    log "=== Running Python Processes ==="
    ssh ubuntu@100.123.183.70 "ps aux | grep -E 'python.*(selfplay|train|gauntlet|elo)' | grep -v grep | head -10 || echo 'No training jobs found'" 2>&1 | tee -a "$LOG_FILE"
}

restart_idle_workers() {
    log "=== Checking for Idle Workers ==="
    # Check if any ELO workers are running, restart if needed
    local worker_count=$(ssh ubuntu@100.123.183.70 "pgrep -f elo_db_sync | wc -l" 2>/dev/null || echo 0)
    if [ "$worker_count" -lt 1 ]; then
        log "WARNING: No ELO workers running, attempting restart..."
        ssh ubuntu@100.123.183.70 "cd ~/ringrift/ai-service && source venv/bin/activate && nohup python scripts/elo_db_sync.py --mode worker --coordinator 100.77.77.122 > /tmp/elo_worker.log 2>&1 &" 2>/dev/null
        log "ELO worker restart attempted"
    fi

    # Check for cluster worker
    local cluster_worker=$(ssh ubuntu@100.123.183.70 "pgrep -f cluster_worker | wc -l" 2>/dev/null || echo 0)
    log "Cluster workers running: $cluster_worker"
}

fill_idle_nodes() {
    log "=== Attempting to Fill Idle Nodes ==="
    ssh ubuntu@100.123.183.70 "cd ~/ringrift/ai-service && source venv/bin/activate && python scripts/cluster_submit.py fill-idle --job-type selfplay --board-type square8 --player-count 2 --dry-run 2>/dev/null || echo 'fill-idle not available'" 2>&1 | head -20 | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting 10-hour cluster monitoring"
log "End time: $(date -r $END_TIME '+%Y-%m-%d %H:%M:%S')"
log "Interval: ${INTERVAL_SECONDS} seconds"
log "=========================================="

iteration=0
while [ $(date +%s) -lt $END_TIME ]; do
    iteration=$((iteration + 1))
    log ""
    log "=========================================="
    log "Iteration #$iteration - $(date '+%Y-%m-%d %H:%M:%S')"
    log "=========================================="

    # Run all checks
    check_tailscale_nodes
    check_gpu_utilization
    check_running_jobs
    check_cluster_nodes

    # Restoration actions
    restart_idle_workers

    # Calculate remaining time
    remaining=$((END_TIME - $(date +%s)))
    remaining_hours=$((remaining / 3600))
    remaining_mins=$(((remaining % 3600) / 60))
    log "Remaining: ${remaining_hours}h ${remaining_mins}m"

    # Sleep until next iteration
    log "Sleeping for ${INTERVAL_SECONDS} seconds..."
    sleep $INTERVAL_SECONDS
done

log "=========================================="
log "Monitoring complete after $iteration iterations"
log "=========================================="

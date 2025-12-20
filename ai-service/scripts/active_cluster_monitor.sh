#!/bin/bash
# Active Cluster Monitor - Checks health AND fills idle nodes
# Created by Claude Code

set -e

LOG_FILE="/Users/armand/Development/RingRift/ai-service/logs/active_monitor_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

PRIMARY_NODE="ubuntu@100.123.183.70"
MONITOR_DURATION_HOURS=10
INTERVAL_SECONDS=300  # 5 minutes
END_TIME=$(($(date +%s) + MONITOR_DURATION_HOURS * 3600))

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_gpu_util() {
    local node=$1
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$node" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null || echo "-1"
}

check_running_jobs() {
    ssh -o ConnectTimeout=10 "$PRIMARY_NODE" \
        "ps aux | grep -E 'python.*(selfplay|train|gauntlet|elo|soak|tournament)' | grep -v grep | wc -l" 2>/dev/null || echo "0"
}

start_selfplay_on_node() {
    local node=$1
    local board=$2
    local players=$3
    log "  Starting selfplay on $node ($board ${players}p)..."
    ssh -o ConnectTimeout=30 "$node" "cd ~/ringrift/ai-service && source venv/bin/activate && \
        nohup python scripts/run_self_play_soak.py \
            --num-games 20 \
            --board-type $board \
            --num-players $players \
            --max-moves 2000 \
            --engine-mode mixed \
            --difficulty-band canonical \
            --log-jsonl data/selfplay/monitor_fill_${board}_${players}p.jsonl \
            --verbose 5 \
            > /tmp/monitor_selfplay_${board}_${players}p.log 2>&1 &" 2>/dev/null
    log "  Started selfplay job on $node"
}

fill_idle_primary() {
    local gpu_util=$(check_gpu_util "$PRIMARY_NODE")
    local job_count=$(check_running_jobs)

    log "Primary node: GPU=${gpu_util}%, Jobs=${job_count}"

    if [ "$gpu_util" != "-1" ] && [ "$gpu_util" -lt 20 ] && [ "$job_count" -lt 2 ]; then
        log "  Primary node underutilized, starting selfplay..."
        start_selfplay_on_node "$PRIMARY_NODE" "square8" "2"
    fi
}

check_lambda_nodes() {
    log "=== Lambda GH200 Fleet Status ==="
    local active=0
    local idle=0

    for node in lambda-gh200-{a,b,e,f,g,h,i,k,l,m,n,o}; do
        local ip=$(tailscale status 2>/dev/null | grep "$node" | awk '{print $1}')
        if [ -n "$ip" ]; then
            local util=$(check_gpu_util "ubuntu@$ip")
            if [ "$util" = "-1" ]; then
                log "  $node: offline/unreachable"
            elif [ "$util" -lt 10 ]; then
                log "  $node: idle (${util}%)"
                ((idle++))
            else
                log "  $node: active (${util}%)"
                ((active++))
            fi
        fi
    done

    log "Summary: ${active} active, ${idle} idle"
}

log "=========================================="
log "Starting Active Cluster Monitor"
log "Duration: ${MONITOR_DURATION_HOURS}h, Interval: ${INTERVAL_SECONDS}s"
log "=========================================="

iteration=0
while [ $(date +%s) -lt $END_TIME ]; do
    iteration=$((iteration + 1))
    log ""
    log "=== Iteration #$iteration ==="

    # Check and fill primary node
    fill_idle_primary

    # Check Lambda fleet (quick summary)
    check_lambda_nodes

    # Calculate remaining time
    remaining=$((END_TIME - $(date +%s)))
    remaining_hours=$((remaining / 3600))
    remaining_mins=$(((remaining % 3600) / 60))
    log "Remaining: ${remaining_hours}h ${remaining_mins}m"

    sleep $INTERVAL_SECONDS
done

log "=========================================="
log "Monitoring complete after $iteration iterations"
log "=========================================="

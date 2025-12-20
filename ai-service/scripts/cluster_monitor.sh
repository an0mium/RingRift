#!/bin/bash
# Cluster Monitor - Runs every 5 minutes for 10 hours
# Checks cluster health and takes corrective action

LOGFILE="/Users/armand/Development/RingRift/ai-service/logs/cluster_monitor.log"
mkdir -p "$(dirname "$LOGFILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

check_tailscale() {
    log "=== Tailscale Status ==="
    tailscale status 2>/dev/null | grep -E "(vast|lambda)" | head -20 >> "$LOGFILE"
    OFFLINE_COUNT=$(tailscale status 2>/dev/null | grep -E "(vast|lambda)" | grep -c "offline")
    log "Offline nodes: $OFFLINE_COUNT"
    return $OFFLINE_COUNT
}

check_vast_nodes() {
    log "=== Vast.ai Nodes ==="
    # Check key training nodes
    for node in "100.97.157.45" "100.74.40.31" "100.79.143.125"; do
        if ssh -o ConnectTimeout=10 -o BatchMode=yes root@$node "echo OK" >/dev/null 2>&1; then
            log "Node $node: ONLINE"
            # Check GPU utilization
            GPU_UTIL=$(ssh root@$node "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)
            log "  GPU: $GPU_UTIL"

            # Check for running training
            PROCS=$(ssh root@$node "ps aux | grep -E 'train|selfplay' | grep -v grep | wc -l" 2>/dev/null)
            log "  Training procs: $PROCS"

            # If GPU idle and no training, suggest starting work
            if [[ "$GPU_UTIL" == *"0 %"* ]] && [[ "$PROCS" == "0" ]]; then
                log "  WARNING: Node idle - consider starting training"
            fi
        else
            log "Node $node: OFFLINE/UNREACHABLE"
        fi
    done
}

check_lambda_nodes() {
    log "=== Lambda Nodes ==="
    for node in "100.88.176.74" "100.97.104.89"; do
        if ssh -o ConnectTimeout=10 -o BatchMode=yes ubuntu@$node "echo OK" >/dev/null 2>&1; then
            log "Lambda $node: ONLINE"
        else
            log "Lambda $node: OFFLINE/UNREACHABLE"
        fi
    done
}

check_generation_progress() {
    log "=== Canonical Generation ==="
    # Check A40 for canonical data generation
    SQ19_COUNT=$(ssh root@100.97.157.45 "cd /root/ringrift/ai-service && sqlite3 data/canonical/canonical_square19.db 'SELECT COUNT(*) FROM game_records' 2>/dev/null" 2>/dev/null)
    HEX_COUNT=$(ssh root@100.97.157.45 "cd /root/ringrift/ai-service && sqlite3 data/canonical/canonical_hexagonal.db 'SELECT COUNT(*) FROM game_records' 2>/dev/null" 2>/dev/null)
    log "Square19 games: ${SQ19_COUNT:-0}"
    log "Hexagonal games: ${HEX_COUNT:-0}"
}

take_corrective_action() {
    log "=== Corrective Actions ==="
    # If A40 is idle, restart GPU selfplay
    GPU_UTIL=$(ssh root@100.97.157.45 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)
    if [[ "$GPU_UTIL" == *"0 %"* ]] || [[ "$GPU_UTIL" == *"1 %"* ]] || [[ "$GPU_UTIL" == *"2 %"* ]]; then
        log "A40 GPU idle, checking if more games needed..."
        SQ19_COUNT=$(ssh root@100.97.157.45 "cd /root/ringrift/ai-service && sqlite3 data/canonical/canonical_square19.db 'SELECT COUNT(*) FROM game_records' 2>/dev/null" 2>/dev/null)
        if [[ "${SQ19_COUNT:-0}" -lt 200 ]]; then
            log "Starting GPU selfplay for square19..."
            ssh root@100.97.157.45 "cd /root/ringrift/ai-service && source venv/bin/activate && PYTHONPATH=/root/ringrift/ai-service nohup python scripts/run_gpu_selfplay.py --board square19 --num-games 300 --batch-size 64 --engine-mode random-only --output-db data/canonical/canonical_square19.db > /tmp/gpu_selfplay.log 2>&1 &" 2>/dev/null
            log "GPU selfplay started"
        fi
    fi
}

run_monitoring_cycle() {
    log ""
    log "========================================"
    log "CLUSTER HEALTH CHECK"
    log "========================================"

    check_tailscale
    check_vast_nodes
    check_lambda_nodes
    check_generation_progress
    take_corrective_action

    log "========================================"
    log "Check complete"
    log "========================================"
}

# Main loop - run for 10 hours (120 cycles of 5 minutes)
log "Starting cluster monitoring - 10 hour duration"
log "Interval: 5 minutes"

for i in $(seq 1 120); do
    log ""
    log "*** Cycle $i/120 ***"
    run_monitoring_cycle

    if [ $i -lt 120 ]; then
        log "Sleeping for 5 minutes..."
        sleep 300
    fi
done

log "Monitoring complete after 10 hours"

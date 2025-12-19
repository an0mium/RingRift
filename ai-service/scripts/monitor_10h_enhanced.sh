#!/bin/bash
#
# Enhanced 10-Hour Monitoring Script (2-minute intervals)
# Ensures all cluster nodes are online, active, and well-utilized
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$AI_SERVICE_DIR/logs/monitor_10h.log"
START_TIME=$(date +%s)
DURATION_HOURS=10
DURATION_SECONDS=$((DURATION_HOURS * 3600))
CHECK_INTERVAL=120  # 2 minutes

mkdir -p "$AI_SERVICE_DIR/logs"

# All cluster instances
CLUSTER_INSTANCES=(
    "28925166|ssh1.vast.ai|15166|RTX5090"
    "28928169|ssh5.vast.ai|18168|RTX5090-2"
    "28889943|ssh1.vast.ai|19942|RTX5080"
    "28918742|ssh8.vast.ai|38742|A40"
    "28889941|ssh3.vast.ai|19940|RTX4080S"
    "28844401|ssh1.vast.ai|14400|RTX4060Ti"
    "28889768|ssh2.vast.ai|19768|RTX4060Ti-2"
    "28844365|ssh5.vast.ai|14364|RTX3070"
    "28889766|ssh3.vast.ai|19766|RTX3060Ti"
    "28918740|ssh3.vast.ai|38740|RTX3060"
    "28890015|ssh9.vast.ai|10014|RTX2080Ti"
    "28844370|ssh2.vast.ai|14370|RTX2060S"
    "28920043|ssh2.vast.ai|10042|RTX5070"
)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_status() {
    printf "[$(date '+%Y-%m-%d %H:%M:%S')] %-12s %-10s %s\n" "$1" "$2" "$3" | tee -a "$LOG_FILE"
}

# Check single node status
check_node() {
    local id=$1 host=$2 port=$3 gpu=$4
    
    result=$(timeout 20 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" "
cd /root/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service 2>/dev/null || echo 'NO_PATH'
if [ -d . ]; then
    games=\$(wc -l data/gumbel_selfplay/sq8*.jsonl 2>/dev/null | tail -1 | awk '{print \$1}' || echo 0)
    procs=\$(pgrep -f 'generate_gumbel.*square8' 2>/dev/null | wc -l || echo 0)
    gpu_util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'N/A')
    echo \"OK|\$games|\$procs|\$gpu_util\"
else
    echo 'NO_PATH|0|0|0'
fi
" 2>&1 | grep -v "Welcome\|Have fun\|Warning" | tail -1)
    
    echo "$result"
}

# Start Gumbel on a node
start_gumbel_on_node() {
    local id=$1 host=$2 port=$3 gpu=$4
    
    timeout 30 ssh -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" "
cd /root/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service
export PYTHONPATH=\$(pwd)
source venv/bin/activate 2>/dev/null || true
mkdir -p logs data/gumbel_selfplay
timestamp=\$(date +%Y%m%d_%H%M%S)
nohup python3 scripts/generate_gumbel_selfplay.py \
    --num-games 300 \
    --board-type square8 \
    --num-players 2 \
    --gumbel-sims 64 \
    --output data/gumbel_selfplay/sq8_${gpu}_\${timestamp}.jsonl \
    > logs/gumbel_sq8.log 2>&1 &
echo 'STARTED'
" 2>&1 | grep -v "Welcome\|Have fun" | tail -1
}

# Check curriculum training
check_curriculum() {
    local stages_complete=$(ls "$AI_SERVICE_DIR"/curriculum_runs/sq8_2p_optimal_*/stage_*/nnue_policy_training_report.json 2>/dev/null | wc -l | tr -d ' ')
    local training_running=$(pgrep -f "train_nnue_policy_curriculum" 2>/dev/null | wc -l | tr -d ' ')
    local latest_stage=""
    
    if [ "$training_running" -gt 0 ]; then
        latest_stage=$(tail -5 /tmp/claude/tasks/bbfc915.output 2>/dev/null | grep -oE "Stage: [A-Z-]+" | tail -1 || echo "")
    fi
    
    echo "$stages_complete|$training_running|$latest_stage"
}

# Sync Gumbel data from cluster
sync_gumbel_data() {
    log "SYNC: Syncing Gumbel data from cluster..."
    mkdir -p "$AI_SERVICE_DIR/data/gumbel_selfplay/cluster"
    
    local synced=0
    for instance in "${CLUSTER_INSTANCES[@]}"; do
        IFS='|' read -r id host port gpu <<< "$instance"
        rsync -avz --timeout=30 -e "ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -p $port" \
            "root@$host:/root/ringrift/ai-service/data/gumbel_selfplay/sq8*.jsonl" \
            "$AI_SERVICE_DIR/data/gumbel_selfplay/cluster/" 2>/dev/null && ((synced++)) || true
    done
    
    local total_files=$(ls "$AI_SERVICE_DIR/data/gumbel_selfplay/cluster/"*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    local total_games=$(wc -l "$AI_SERVICE_DIR/data/gumbel_selfplay/cluster/"*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
    log "SYNC: $synced nodes synced, $total_files files, $total_games total games"
}

# Run A/B tests on cluster
run_cluster_ab_tests() {
    log "TEST: Running A/B tests on cluster..."
    
    # Find best available node
    for instance in "${CLUSTER_INSTANCES[@]}"; do
        IFS='|' read -r id host port gpu <<< "$instance"
        
        if timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no -p "$port" "root@$host" "echo OK" 2>/dev/null | grep -q "OK"; then
            log "TEST: Using $gpu ($id) for A/B tests"
            
            # Sync model
            scp -o ConnectTimeout=15 -o StrictHostKeyChecking=no -P "$port" \
                "$AI_SERVICE_DIR/models/nnue/nnue_policy_square8_2p.pt" \
                "root@$host:/root/ringrift/ai-service/models/nnue/" 2>/dev/null
            
            # Run tests
            ssh -o ConnectTimeout=600 -o StrictHostKeyChecking=no -p "$port" "root@$host" "
cd /root/ringrift/ai-service
export PYTHONPATH=\$(pwd)
source venv/bin/activate 2>/dev/null || true

echo '=== A/B Test (50 games @ 500ms) ==='
python3 scripts/ab_test_policy_models.py --model-a models/nnue/nnue_policy_square8_2p.pt --games 50 --think-time 500

echo ''
echo '=== Multi-Time A/B Test ==='
python3 scripts/ab_test_policy_models.py --model-a models/nnue/nnue_policy_square8_2p.pt --games 30 --quick-times
" 2>&1 | grep -v "Welcome\|Have fun" | tee -a "$LOG_FILE"
            
            return 0
        fi
    done
    
    log "TEST: ERROR - No nodes available for A/B tests"
    return 1
}

# Main monitoring loop
log "========================================"
log "ENHANCED 10-HOUR MONITORING STARTED"
log "========================================"
log "Check interval: 2 minutes"
log "Nodes monitored: ${#CLUSTER_INSTANCES[@]}"
log "Log file: $LOG_FILE"
log ""

CURRICULUM_DONE=false
TESTS_RUN=false
LAST_DATA_SYNC=0
CHECK_NUM=0

while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - START_TIME))
    remaining=$((DURATION_SECONDS - elapsed))
    hours_remaining=$((remaining / 3600))
    mins_remaining=$(((remaining % 3600) / 60))
    ((CHECK_NUM++))
    
    if [ $elapsed -ge $DURATION_SECONDS ]; then
        log "========================================"
        log "10-HOUR MONITORING COMPLETE"
        log "========================================"
        break
    fi
    
    log ""
    log "========================================"
    log "CHECK #$CHECK_NUM | ${hours_remaining}h ${mins_remaining}m remaining"
    log "========================================"
    
    # Check curriculum training
    curriculum_status=$(check_curriculum)
    IFS='|' read -r stages_complete training_running latest_stage <<< "$curriculum_status"
    log "CURRICULUM: $stages_complete/5 stages complete | Process: $training_running | $latest_stage"
    
    if [ "$stages_complete" -ge 5 ] && [ "$CURRICULUM_DONE" = false ]; then
        log ">>> CURRICULUM TRAINING COMPLETE <<<"
        CURRICULUM_DONE=true
    fi
    
    # Check all cluster nodes
    log ""
    log "CLUSTER NODE STATUS:"
    log "----------------------------------------"
    printf "%-12s %-8s %-8s %-8s %-8s\n" "GPU" "STATUS" "GAMES" "PROCS" "GPU%" | tee -a "$LOG_FILE"
    log "----------------------------------------"
    
    total_games=0
    total_procs=0
    nodes_online=0
    nodes_idle=0
    
    for instance in "${CLUSTER_INSTANCES[@]}"; do
        IFS='|' read -r id host port gpu <<< "$instance"
        
        result=$(check_node "$id" "$host" "$port" "$gpu")
        
        if [[ "$result" == OK* ]]; then
            IFS='|' read -r status games procs gpu_util <<< "$result"
            ((nodes_online++))
            total_games=$((total_games + games))
            total_procs=$((total_procs + procs))
            
            if [ "$procs" -eq 0 ]; then
                ((nodes_idle++))
                printf "%-12s %-8s %-8s %-8s %-8s" "$gpu" "IDLE" "$games" "$procs" "$gpu_util" | tee -a "$LOG_FILE"
                
                # Start Gumbel on idle node
                start_result=$(start_gumbel_on_node "$id" "$host" "$port" "$gpu")
                if [[ "$start_result" == *"STARTED"* ]]; then
                    echo " -> RESTARTED" | tee -a "$LOG_FILE"
                else
                    echo " -> FAILED" | tee -a "$LOG_FILE"
                fi
            else
                printf "%-12s %-8s %-8s %-8s %-8s\n" "$gpu" "RUNNING" "$games" "$procs" "$gpu_util%" | tee -a "$LOG_FILE"
            fi
        else
            printf "%-12s %-8s %-8s %-8s %-8s\n" "$gpu" "OFFLINE" "-" "-" "-" | tee -a "$LOG_FILE"
        fi
    done
    
    log "----------------------------------------"
    log "SUMMARY: $nodes_online/${#CLUSTER_INSTANCES[@]} online | $total_procs jobs | $total_games games | $nodes_idle restarted"
    
    # Sync data every hour
    if [ $((current_time - LAST_DATA_SYNC)) -ge 3600 ]; then
        sync_gumbel_data
        LAST_DATA_SYNC=$current_time
    fi
    
    # Run A/B tests when curriculum complete
    if [ "$CURRICULUM_DONE" = true ] && [ "$TESTS_RUN" = false ]; then
        run_cluster_ab_tests
        TESTS_RUN=true
    fi
    
    sleep $CHECK_INTERVAL
done

#!/bin/bash
# 10-Hour Cluster Automation Script
# Ensures all nodes are utilized for selfplay + training toward 2000+ Elo

set -e
cd /Users/armand/Development/RingRift/ai-service
export PYTHONPATH=.

LOG_DIR="logs/automation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Starting 10-Hour Cluster Automation ===" | tee "$LOG_DIR/main.log"
echo "Start time: $(date)" | tee -a "$LOG_DIR/main.log"
echo "Log directory: $LOG_DIR" | tee -a "$LOG_DIR/main.log"

# All 12 configs to train
CONFIGS=(
    "hex8:2"
    "hex8:3"
    "hex8:4"
    "square8:2"
    "square8:3"
    "square8:4"
    "square19:2"
    "square19:3"
    "square19:4"
    "hexagonal:2"
    "hexagonal:3"
    "hexagonal:4"
)

# GPU nodes for training
GPU_TRAINING_NODES=(
    "nebius-h100-1:ubuntu:89.169.111.139:22"
    "runpod-h100:root:102.210.171.65:30178"
    "runpod-a100-1:root:38.128.233.145:33085"
    "runpod-a100-2:root:104.255.9.187:11681"
)

# GPU nodes for selfplay
GPU_SELFPLAY_NODES=(
    "nebius-backbone-1:ubuntu:89.169.112.47:22"
    "nebius-l40s-2:ubuntu:158.160.64.58:22"
    "runpod-l40s-2:root:193.183.22.62:1630"
    "vultr-a100-20gb:root:208.167.249.164:22"
)

# Vast.ai nodes (mixed GPUs)
VAST_NODES=(
    "vast-29129529:root:ssh5.vast.ai:10853"
    "vast-29118471:root:ssh8.vast.ai:38470"
    "vast-29128352:root:ssh5.vast.ai:18352"
    "vast-28925166:root:ssh6.vast.ai:19528"
    "vast-29128356:root:ssh8.vast.ai:18360"
    "vast-28918742:root:ssh5.vast.ai:19234"
    "vast-29031159:root:ssh5.vast.ai:37654"
    "vast-29126088:root:ssh5.vast.ai:14766"
    "vast-29031161:root:ssh5.vast.ai:37658"
    "vast-28890015:root:ssh5.vast.ai:18134"
    "vast-28889766:root:ssh4.vast.ai:18004"
    "vast-29046315:root:ssh5.vast.ai:11174"
)

# CPU nodes for heuristic selfplay
CPU_NODES=(
    "hetzner-cpu1:root:138.201.198.49:22"
    "hetzner-cpu2:root:159.69.155.75:22"
    "hetzner-cpu3:root:159.69.155.106:22"
)

# Function to run command on remote node
run_remote() {
    local node=$1
    local user=$2
    local host=$3
    local port=$4
    local cmd=$5
    local timeout=${6:-30}

    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes \
        -i ~/.ssh/id_cluster -p "$port" "$user@$host" \
        "cd ~/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service 2>/dev/null; $cmd" \
        2>/dev/null
}

# Function to check if node is alive
check_node() {
    local node=$1
    local user=$2
    local host=$3
    local port=$4

    if timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        -i ~/.ssh/id_cluster -p "$port" "$user@$host" "echo ok" 2>/dev/null | grep -q "ok"; then
        return 0
    fi
    return 1
}

# Function to start selfplay on a node
start_selfplay() {
    local node=$1
    local user=$2
    local host=$3
    local port=$4
    local board=$5
    local players=$6
    local engine=${7:-gumbel}
    local games=${8:-1000}

    echo "[$(date +%H:%M:%S)] Starting $engine selfplay on $node: ${board}_${players}p" | tee -a "$LOG_DIR/main.log"

    run_remote "$node" "$user" "$host" "$port" "
        pkill -f 'selfplay.py.*${board}.*${players}' 2>/dev/null || true
        sleep 1
        export PYTHONPATH=.
        nohup python3 scripts/selfplay.py \
            --board $board --num-players $players \
            --engine $engine --num-games $games \
            --output-dir data/games/selfplay_${board}_${players}p \
            > logs/selfplay_${board}_${players}p.log 2>&1 &
        echo 'Started selfplay PID:' \$!
    " 60
}

# Function to start training on a node
start_training() {
    local node=$1
    local user=$2
    local host=$3
    local port=$4
    local board=$5
    local players=$6

    echo "[$(date +%H:%M:%S)] Starting training on $node: ${board}_${players}p" | tee -a "$LOG_DIR/main.log"

    run_remote "$node" "$user" "$host" "$port" "
        pkill -f 'app.training.train.*${board}' 2>/dev/null || true
        sleep 1
        export PYTHONPATH=.

        # Find best training data
        DATA_FILE=\$(ls -t data/training/*${board}*${players}p*.npz 2>/dev/null | head -1)
        if [ -z \"\$DATA_FILE\" ]; then
            echo 'No training data found for ${board}_${players}p'
            exit 1
        fi

        # Find init weights if available
        INIT_WEIGHTS=''
        if [ -f models/canonical_${board}_${players}p.pth ]; then
            INIT_WEIGHTS='--init-weights models/canonical_${board}_${players}p.pth'
        elif [ -f models/ringrift_best_${board}_${players}p.pth ]; then
            INIT_WEIGHTS='--init-weights models/ringrift_best_${board}_${players}p.pth'
        fi

        nohup python3 -m app.training.train \
            --board-type $board --num-players $players \
            --data-path \"\$DATA_FILE\" \
            --model-version v2 \
            --batch-size 1024 \
            --epochs 50 \
            --save-path models/${board}_${players}p_trained.pth \
            \$INIT_WEIGHTS \
            --skip-freshness-check \
            > logs/train_${board}_${players}p.log 2>&1 &
        echo 'Started training PID:' \$!
    " 120
}

# Function to check and restart stalled jobs
check_and_restart() {
    echo "[$(date +%H:%M:%S)] Checking cluster status..." | tee -a "$LOG_DIR/main.log"

    local config_idx=0

    # Check GPU training nodes
    for node_info in "${GPU_TRAINING_NODES[@]}"; do
        IFS=':' read -r node user host port <<< "$node_info"

        if check_node "$node" "$user" "$host" "$port"; then
            # Check if training is running
            if ! run_remote "$node" "$user" "$host" "$port" "pgrep -f 'app.training.train'" 2>/dev/null | grep -q .; then
                # Get next config
                local config="${CONFIGS[$config_idx]}"
                IFS=':' read -r board players <<< "$config"
                start_training "$node" "$user" "$host" "$port" "$board" "$players"
                config_idx=$(( (config_idx + 1) % ${#CONFIGS[@]} ))
            else
                echo "[$(date +%H:%M:%S)] $node: Training active" | tee -a "$LOG_DIR/main.log"
            fi
        else
            echo "[$(date +%H:%M:%S)] $node: OFFLINE" | tee -a "$LOG_DIR/main.log"
        fi
    done

    # Check GPU selfplay nodes
    for node_info in "${GPU_SELFPLAY_NODES[@]}"; do
        IFS=':' read -r node user host port <<< "$node_info"

        if check_node "$node" "$user" "$host" "$port"; then
            if ! run_remote "$node" "$user" "$host" "$port" "pgrep -f 'selfplay.py'" 2>/dev/null | grep -q .; then
                local config="${CONFIGS[$config_idx]}"
                IFS=':' read -r board players <<< "$config"
                start_selfplay "$node" "$user" "$host" "$port" "$board" "$players" "gumbel" 2000
                config_idx=$(( (config_idx + 1) % ${#CONFIGS[@]} ))
            else
                echo "[$(date +%H:%M:%S)] $node: Selfplay active" | tee -a "$LOG_DIR/main.log"
            fi
        else
            echo "[$(date +%H:%M:%S)] $node: OFFLINE" | tee -a "$LOG_DIR/main.log"
        fi
    done

    # Check Vast.ai nodes
    for node_info in "${VAST_NODES[@]}"; do
        IFS=':' read -r node user host port <<< "$node_info"

        if check_node "$node" "$user" "$host" "$port"; then
            if ! run_remote "$node" "$user" "$host" "$port" "pgrep -f 'selfplay.py|app.training'" 2>/dev/null | grep -q .; then
                local config="${CONFIGS[$config_idx]}"
                IFS=':' read -r board players <<< "$config"
                start_selfplay "$node" "$user" "$host" "$port" "$board" "$players" "gumbel" 1000
                config_idx=$(( (config_idx + 1) % ${#CONFIGS[@]} ))
            else
                echo "[$(date +%H:%M:%S)] $node: Active" | tee -a "$LOG_DIR/main.log"
            fi
        else
            echo "[$(date +%H:%M:%S)] $node: OFFLINE" | tee -a "$LOG_DIR/main.log"
        fi
    done

    # Check CPU nodes (heuristic selfplay)
    for node_info in "${CPU_NODES[@]}"; do
        IFS=':' read -r node user host port <<< "$node_info"

        if check_node "$node" "$user" "$host" "$port"; then
            if ! run_remote "$node" "$user" "$host" "$port" "pgrep -f 'selfplay.py'" 2>/dev/null | grep -q .; then
                local config="${CONFIGS[$config_idx]}"
                IFS=':' read -r board players <<< "$config"
                start_selfplay "$node" "$user" "$host" "$port" "$board" "$players" "heuristic" 5000
                config_idx=$(( (config_idx + 1) % ${#CONFIGS[@]} ))
            else
                echo "[$(date +%H:%M:%S)] $node: Selfplay active" | tee -a "$LOG_DIR/main.log"
            fi
        else
            echo "[$(date +%H:%M:%S)] $node: OFFLINE" | tee -a "$LOG_DIR/main.log"
        fi
    done
}

# Function to sync data periodically
sync_data() {
    echo "[$(date +%H:%M:%S)] Syncing data from cluster..." | tee -a "$LOG_DIR/main.log"

    # Run sync script if available
    if [ -f scripts/sync_cluster_data.py ]; then
        timeout 600 python3 scripts/sync_cluster_data.py --direction pull 2>&1 | tee -a "$LOG_DIR/sync.log" || true
    fi
}

# Function to run gauntlet evaluation
run_evaluation() {
    echo "[$(date +%H:%M:%S)] Running gauntlet evaluations..." | tee -a "$LOG_DIR/main.log"

    for model in models/*_trained.pth; do
        if [ -f "$model" ]; then
            # Extract board type and players from filename
            local basename=$(basename "$model" .pth)
            local board=$(echo "$basename" | sed 's/_[0-9]p_trained//' | sed 's/_trained//')
            local players=$(echo "$basename" | grep -oE '[0-9]p' | sed 's/p//')

            if [ -n "$board" ] && [ -n "$players" ]; then
                echo "[$(date +%H:%M:%S)] Evaluating $model" | tee -a "$LOG_DIR/main.log"
                timeout 600 python3 scripts/auto_promote.py --gauntlet \
                    --model "$model" \
                    --board-type "$board" \
                    --num-players "$players" \
                    --games 30 2>&1 | tee -a "$LOG_DIR/gauntlet.log" || true
            fi
        fi
    done
}

# Main loop - runs for 10 hours
END_TIME=$(($(date +%s) + 36000))  # 10 hours from now

echo "=== Starting main automation loop ===" | tee -a "$LOG_DIR/main.log"
echo "Will run until: $(date -r $END_TIME)" | tee -a "$LOG_DIR/main.log"

ITERATION=0
while [ $(date +%s) -lt $END_TIME ]; do
    ITERATION=$((ITERATION + 1))
    echo "" | tee -a "$LOG_DIR/main.log"
    echo "=== Iteration $ITERATION ($(date)) ===" | tee -a "$LOG_DIR/main.log"

    # Check and restart stalled jobs
    check_and_restart

    # Sync data every 30 minutes
    if [ $((ITERATION % 6)) -eq 0 ]; then
        sync_data
    fi

    # Run evaluations every hour
    if [ $((ITERATION % 12)) -eq 0 ]; then
        run_evaluation
    fi

    # Print status summary
    echo "[$(date +%H:%M:%S)] Sleeping 5 minutes until next check..." | tee -a "$LOG_DIR/main.log"
    sleep 300
done

echo "" | tee -a "$LOG_DIR/main.log"
echo "=== 10-Hour Automation Complete ===" | tee -a "$LOG_DIR/main.log"
echo "End time: $(date)" | tee -a "$LOG_DIR/main.log"

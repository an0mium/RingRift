#!/bin/bash
# Monitor for completed NPZ files and auto-start training
# Usage: ./monitor_and_train.sh

TRAINING_DIR=/lambda/nfs/RingRift/selfplay_data/training
OUTPUT_DIR=/lambda/nfs/RingRift/models/v4_large
LOG_DIR=/tmp/training_logs

mkdir -p $OUTPUT_DIR $LOG_DIR

cd ~/ringrift/ai-service
source venv/bin/activate

declare -A CONFIGS=(
    ["sq8_2p"]="square8:2"
    ["sq8_3p"]="square8:3"
    ["sq8_4p"]="square8:4"
    ["sq19_2p"]="square19:2"
    ["sq19_3p"]="square19:3"
    ["sq19_4p"]="square19:4"
    ["hex_2p"]="hexagonal:2"
    ["hex_3p"]="hexagonal:3"
    ["hex_4p"]="hexagonal:4"
)

declare -A TRAINED=()

while true; do
    for config in "${!CONFIGS[@]}"; do
        npz_file="${TRAINING_DIR}/${config}_with_policy.npz"
        model_file="${OUTPUT_DIR}/${config}_v4_large.pth"
        
        # Skip if already trained or training
        if [[ -n "${TRAINED[$config]}" ]]; then
            continue
        fi
        
        # Check if NPZ exists and model doesnt
        if [[ -f "$npz_file" ]] && [[ ! -f "$model_file" ]]; then
            IFS=":" read -r board_type num_players <<< "${CONFIGS[$config]}"
            
            echo "[$(date)] Starting training for $config ($board_type, ${num_players}p)"
            
            PYTHONPATH=. python -m app.training.train \
                --data-path "$npz_file" \
                --board-type "$board_type" \
                --num-players "$num_players" \
                --model-version v3 \
                --num-res-blocks 20 \
                --num-filters 256 \
                --epochs 50 \
                --lr-scheduler cosine \
                --early-stopping-patience 10 \
                --save-path "$model_file" \
                2>&1 | tee "$LOG_DIR/train_${config}.log" &
            
            TRAINED[$config]=1
            echo "[$(date)] Training started for $config (PID: $!)"
        fi
    done
    
    sleep 60  # Check every minute
done

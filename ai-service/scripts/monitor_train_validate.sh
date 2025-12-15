#!/bin/bash
# Monitor for completed NPZ files, auto-start training, and validate with Elo
# Usage: ./monitor_train_validate.sh

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
declare -A TRAINING_PIDS=()

log() {
    echo "[$(date +%H:%M:%S)] $1"
}

while true; do
    for config in "${!CONFIGS[@]}"; do
        npz_file="${TRAINING_DIR}/${config}_with_policy.npz"
        model_file="${OUTPUT_DIR}/${config}_v4_large.pth"
        
        # Skip if already started training
        if [[ -n "${TRAINED[$config]}" ]]; then
            # Check if training finished
            if [[ -n "${TRAINING_PIDS[$config]}" ]]; then
                if ! kill -0 "${TRAINING_PIDS[$config]}" 2>/dev/null; then
                    if [[ -f "$model_file" ]]; then
                        log "✅ Training COMPLETE for $config - model saved"
                        # Could trigger Elo validation here
                    else
                        log "❌ Training FAILED for $config"
                    fi
                    unset TRAINING_PIDS[$config]
                fi
            fi
            continue
        fi
        
        # Check if NPZ exists and model doesnt
        if [[ -f "$npz_file" ]] && [[ ! -f "$model_file" ]]; then
            IFS=":" read -r board_type num_players <<< "${CONFIGS[$config]}"
            
            log "🚀 Starting training for $config ($board_type, ${num_players}p)"
            
            PYTHONPATH=. RINGRIFT_SKIP_SHADOW_CONTRACTS=true python -m app.training.train \
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
                > "$LOG_DIR/train_${config}.log" 2>&1 &
            
            TRAINING_PIDS[$config]=$!
            TRAINED[$config]=1
            log "Training started for $config (PID: ${TRAINING_PIDS[$config]})"
        fi
    done
    
    # Status report every 5 minutes
    sleep 300
    log "--- Status check ---"
    for config in "${!TRAINED[@]}"; do
        if [[ -n "${TRAINING_PIDS[$config]}" ]]; then
            if kill -0 "${TRAINING_PIDS[$config]}" 2>/dev/null; then
                log "  $config: training (PID ${TRAINING_PIDS[$config]})"
            fi
        fi
    done
done

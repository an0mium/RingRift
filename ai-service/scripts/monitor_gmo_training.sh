#!/bin/bash
# Monitor GMO training and run evaluation when complete
# Usage: ./scripts/monitor_gmo_training.sh

set -e

CHECKPOINT_DIR="models/gmo/sq8_2p_playerrel"
LOG_PATTERN="logs/gmo_train_playerrel_v2*.log"
EVAL_OUTPUT="results/gmo_playerrel_eval_$(date +%Y%m%d_%H%M%S).json"

echo "=========================================="
echo "GMO Training Monitor"
echo "=========================================="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log pattern: $LOG_PATTERN"
echo ""

# Function to check if training is still running
is_training_running() {
    pgrep -f "train_gmo.*combined" > /dev/null 2>&1
}

# Function to check if training completed successfully
training_completed() {
    # Check for checkpoint file
    if [ -f "$CHECKPOINT_DIR/gmo_best.pt" ]; then
        return 0
    fi
    return 1
}

# Function to check if training failed
training_failed() {
    local log_file=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
    if [ -n "$log_file" ] && grep -q "Traceback\|Error\|Exception" "$log_file" 2>/dev/null; then
        if ! is_training_running; then
            return 0
        fi
    fi
    return 1
}

# Monitor loop
echo "Starting monitoring loop..."
while true; do
    if training_completed; then
        echo ""
        echo "=========================================="
        echo "Training COMPLETED!"
        echo "=========================================="
        echo "Checkpoint found at: $CHECKPOINT_DIR/gmo_best.pt"

        # Show final training stats
        echo ""
        echo "Final training log:"
        tail -30 $(ls -t $LOG_PATTERN 2>/dev/null | head -1)

        # Run evaluation
        echo ""
        echo "=========================================="
        echo "Running post-training evaluation..."
        echo "=========================================="

        python scripts/gmo_post_training_eval.py \
            --checkpoint "$CHECKPOINT_DIR/gmo_best.pt" \
            --output "$EVAL_OUTPUT" \
            --device cuda

        echo ""
        echo "Evaluation complete! Results saved to: $EVAL_OUTPUT"
        break

    elif training_failed; then
        echo ""
        echo "=========================================="
        echo "Training FAILED!"
        echo "=========================================="
        echo "Check log for errors:"
        tail -50 $(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        exit 1

    elif is_training_running; then
        # Show progress
        local log_file=$(ls -t $LOG_PATTERN 2>/dev/null | head -1)
        if [ -n "$log_file" ]; then
            echo -n "."
            # Every 5 minutes, show last few lines
            if [ $(($(date +%s) % 300)) -lt 10 ]; then
                echo ""
                echo "[$(date)] Training progress:"
                tail -5 "$log_file" 2>/dev/null || true
            fi
        fi
        sleep 10
    else
        echo ""
        echo "Training process not found. Checking if completed..."
        sleep 5
    fi
done

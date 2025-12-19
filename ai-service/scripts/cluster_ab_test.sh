#!/bin/bash
# Cluster A/B Test Job Script
# Runs comprehensive policy A/B tests on a remote cluster host
#
# Usage:
#   ./scripts/cluster_ab_test.sh [HOST]
#
# HOST defaults to lambda_h100 (fastest GPU for A/B testing)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default host
HOST="${1:-lambda_h100}"

# Get host configuration
case "$HOST" in
    lambda_h100)
        SSH_HOST="ubuntu@209.20.157.81"
        REMOTE_PATH="~/ringrift/ai-service"
        ;;
    lambda_2xh100)
        SSH_HOST="ubuntu@192.222.53.22"
        REMOTE_PATH="~/ringrift/ai-service"
        ;;
    lambda_a10)
        SSH_HOST="ubuntu@150.136.65.197"
        REMOTE_PATH="~/ringrift/ai-service"
        ;;
    gh200_a)
        SSH_HOST="ubuntu@192.222.51.29"
        REMOTE_PATH="~/ringrift/ai-service"
        ;;
    vast_512cpu)
        SSH_HOST="root@vast-512cpu"
        REMOTE_PATH="/workspace/ringrift/ai-service"
        ;;
    vast_rtx4060ti)
        SSH_HOST="root@vast-rtx4060ti"
        REMOTE_PATH="/workspace/ringrift/ai-service"
        ;;
    *)
        echo "Unknown host: $HOST"
        echo "Available hosts: lambda_h100, lambda_2xh100, lambda_a10, gh200_a, vast_512cpu, vast_rtx4060ti"
        exit 1
        ;;
esac

echo "========================================"
echo "Cluster A/B Test Job"
echo "========================================"
echo "Host: $HOST ($SSH_HOST)"
echo "Remote path: $REMOTE_PATH"
echo ""

# Step 1: Sync ONLY the necessary code files (no data directories)
echo "[1/4] Syncing code to cluster..."
rsync -avz --delete \
    --include='app/***' \
    --include='scripts/***' \
    --include='config/***' \
    --include='requirements*.txt' \
    --include='pyproject.toml' \
    --include='setup.py' \
    --exclude='*' \
    "$PROJECT_ROOT/" "$SSH_HOST:$REMOTE_PATH/"

# Step 2: Sync required model files
echo "[2/4] Syncing model files..."
ssh "$SSH_HOST" "mkdir -p $REMOTE_PATH/models"
rsync -avz "$PROJECT_ROOT/models/nnue_kl_full_mixed_sq8_2p.pth" \
    "$SSH_HOST:$REMOTE_PATH/models/" 2>/dev/null || echo "Model sync skipped"

# Ensure output directory exists
ssh "$SSH_HOST" "mkdir -p $REMOTE_PATH/data/training/kl_full_mixed"

# Step 3: Run A/B test on cluster
echo "[3/4] Running comprehensive A/B test on cluster..."
ssh "$SSH_HOST" "cd $REMOTE_PATH && python scripts/ab_test_policy_models.py \
    --model-a models/nnue_kl_full_mixed_sq8_2p.pth \
    --model-b none \
    --multi-time \
    --multi-time-values 50 100 200 500 \
    --num-games 100 \
    --output data/training/kl_full_mixed/comprehensive_ab_test.json"

# Step 4: Sync results back
echo "[4/4] Syncing results back..."
mkdir -p "$PROJECT_ROOT/data/training/kl_full_mixed"
rsync -avz "$SSH_HOST:$REMOTE_PATH/data/training/kl_full_mixed/comprehensive_ab_test.json" \
    "$PROJECT_ROOT/data/training/kl_full_mixed/"

echo ""
echo "========================================"
echo "A/B Test Complete!"
echo "========================================"
echo "Results saved to: data/training/kl_full_mixed/comprehensive_ab_test.json"

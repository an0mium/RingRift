#!/bin/bash
# Distributed Two-Stage Gauntlet Launcher
# Distributes gauntlet shards across Lambda and Vast nodes

set -e

BOARD=${1:-square8}
PLAYERS=${2:-2}
STAGE1_GAMES=${3:-10}
STAGE2_GAMES=${4:-50}

# Lambda nodes with their parallelism levels
LAMBDA_NODES=(
    "lambda-gh200-b-new:ubuntu:64"
    "lambda-gh200-m:ubuntu:64"
    "lambda-gh200-n:ubuntu:64"
    "lambda-gh200-o:ubuntu:64"
    "lambda-gh200-k:ubuntu:64"
    "lambda-gh200-l:ubuntu:64"
    "lambda-2xh100:ubuntu:52"
    "lambda-a10:ubuntu:30"
)

# Vast nodes (high CPU) with SSH info
VAST_NODES=(
    "ssh1.vast.ai:14400:root:128"
    "ssh3.vast.ai:38740:root:64"
    "ssh8.vast.ai:38742:root:64"
)

NUM_SHARDS=$((${#LAMBDA_NODES[@]} + ${#VAST_NODES[@]}))
SHARD=0

echo "========================================"
echo "Distributed Two-Stage Gauntlet"
echo "Board: $BOARD, Players: $PLAYERS"
echo "Stage 1: $STAGE1_GAMES games, Stage 2: $STAGE2_GAMES games"
echo "Total shards: $NUM_SHARDS"
echo "========================================"

# Launch on Lambda nodes
for node_info in "${LAMBDA_NODES[@]}"; do
    IFS=':' read -r node user parallel <<< "$node_info"
    echo "[$SHARD/$NUM_SHARDS] Launching on $node (${parallel} workers)..."

    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$user@$node" "
        cd ~/ringrift/ai-service
        source venv/bin/activate 2>/dev/null || true
        # Kill any existing gauntlet
        pkill -f two_stage_gauntlet || true
        # Launch new gauntlet shard
        nohup bash -c 'PYTHONPATH=. python scripts/two_stage_gauntlet.py \
            --run \
            --board $BOARD \
            --players $PLAYERS \
            --stage1-games $STAGE1_GAMES \
            --stage2-games $STAGE2_GAMES \
            --shard $SHARD \
            --num-shards $NUM_SHARDS \
            -j $parallel \
            2>&1 | tee logs/gauntlet_${BOARD}_${PLAYERS}p_shard${SHARD}.log' &>/dev/null &
        echo 'Started'
    " 2>/dev/null &

    SHARD=$((SHARD + 1))
done

# Launch on Vast nodes
for node_info in "${VAST_NODES[@]}"; do
    IFS=':' read -r host port user parallel <<< "$node_info"
    echo "[$SHARD/$NUM_SHARDS] Launching on $host:$port (${parallel} workers)..."

    ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no -p "$port" "$user@$host" "
        cd /root/ringrift/ai-service
        source venv/bin/activate 2>/dev/null || true
        pkill -f two_stage_gauntlet || true
        nohup bash -c 'PYTHONPATH=. python scripts/two_stage_gauntlet.py \
            --run \
            --board $BOARD \
            --players $PLAYERS \
            --stage1-games $STAGE1_GAMES \
            --stage2-games $STAGE2_GAMES \
            --shard $SHARD \
            --num-shards $NUM_SHARDS \
            -j $parallel \
            2>&1 | tee logs/gauntlet_${BOARD}_${PLAYERS}p_shard${SHARD}.log' &>/dev/null &
        echo 'Started'
    " 2>/dev/null &

    SHARD=$((SHARD + 1))
done

wait
echo ""
echo "All $NUM_SHARDS shards launched!"
echo "Monitor with: for node in lambda-gh200-{b-new,m,n,o,k,l}; do ssh ubuntu@\$node 'tail -5 ~/ringrift/ai-service/logs/gauntlet_*.log'; done"

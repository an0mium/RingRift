#!/bin/bash
# Auto-maintain optimal selfplay load on a node
#
# Usage: ./maintain_selfplay_load.sh [target-jobs] [board] [players]
#
# This script continuously monitors selfplay processes and spawns new ones
# to maintain a target number of concurrent jobs.

set -e

# Configuration with defaults
TARGET_JOBS="${1:-${RINGRIFT_TARGET_SELFPLAY_JOBS:-10}}"
BOARD="${2:-${RINGRIFT_SELFPLAY_BOARD:-hex8}}"
PLAYERS="${3:-${RINGRIFT_SELFPLAY_PLAYERS:-2}}"
ENGINE="${RINGRIFT_SELFPLAY_ENGINE:-gumbel}"
BUDGET="${RINGRIFT_SELFPLAY_BUDGET:-200}"
GAMES_PER_JOB="${RINGRIFT_GAMES_PER_JOB:-30}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RINGRIFT_AI_SERVICE="${RINGRIFT_AI_SERVICE:-$(dirname $(dirname "$SCRIPT_DIR"))}"

# Check interval (seconds)
CHECK_INTERVAL=60

echo "[$(date)] Selfplay load maintainer starting"
echo "[$(date)] Target jobs: $TARGET_JOBS"
echo "[$(date)] Board: $BOARD, Players: $PLAYERS"
echo "[$(date)] Engine: $ENGINE, Budget: $BUDGET"
echo "[$(date)] Games per job: $GAMES_PER_JOB"

cd "$RINGRIFT_AI_SERVICE"

# Activate venv if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$RINGRIFT_AI_SERVICE"

while true; do
    # Count current selfplay processes (excluding this script and grep)
    current=$(ps aux | grep -E 'selfplay\.py|run_gpu_selfplay|run_hybrid_selfplay' | grep -v grep | grep -v maintain | wc -l)
    needed=$((TARGET_JOBS - current))

    if [ $needed -gt 0 ]; then
        echo "[$(date)] Spawning $needed jobs (current: $current, target: $TARGET_JOBS)"

        for i in $(seq 1 $needed); do
            # Generate unique output suffix
            SUFFIX="${BOARD}_${PLAYERS}p_$(date +%s)_$RANDOM"
            LOGFILE="/tmp/selfplay_auto_$SUFFIX.log"

            nohup python scripts/selfplay.py \
                --board "$BOARD" \
                --num-players "$PLAYERS" \
                --engine "$ENGINE" \
                --simulation-budget "$BUDGET" \
                --num-games "$GAMES_PER_JOB" \
                --output-dir "data/games" \
                > "$LOGFILE" 2>&1 &

            echo "[$(date)] Started job $i/$needed (PID: $!)"

            # Small delay between spawns to avoid thundering herd
            sleep 2
        done
    else
        echo "[$(date)] Load OK (current: $current, target: $TARGET_JOBS)"
    fi

    sleep $CHECK_INTERVAL
done

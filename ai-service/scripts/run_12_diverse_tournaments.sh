#!/bin/bash
# Run 12 diverse tournaments: 4 boards × 3 player counts
# Each tournament uses 10 games per pairing for statistical significance

set -e

BOARDS="square8 square19 hex8 hexagonal"
PLAYERS="2 3 4"
GAMES=10
LOG=/tmp/diverse_tournaments_$(date +%Y%m%d_%H%M%S).log

echo "Starting 12 diverse tournaments at $(date)" | tee $LOG
echo "Boards: $BOARDS" | tee -a $LOG
echo "Players: $PLAYERS" | tee -a $LOG
echo "Games per pairing: $GAMES" | tee -a $LOG
echo "" | tee -a $LOG

cd ~/ringrift/ai-service
source venv/bin/activate

for board in $BOARDS; do
  for players in $PLAYERS; do
    echo "======================================" | tee -a $LOG
    echo "Running: $board with $players players" | tee -a $LOG
    echo "======================================" | tee -a $LOG

    python3 scripts/run_model_elo_tournament.py \
      --board "$board" \
      --players "$players" \
      --games "$GAMES" \
      --top-n 10 \
      --include-baselines \
      --run \
      2>&1 | tee -a $LOG

    echo "" | tee -a $LOG
    sleep 5  # Brief pause between tournaments
  done
done

echo "All 12 tournaments completed at $(date)" | tee -a $LOG
echo "Log saved to: $LOG"

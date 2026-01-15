#!/bin/bash
# Quality Training Pipeline - Jan 14, 2026
# Run after quality selfplay completes on GH200 nodes

set -e
cd /Users/armand/Development/RingRift/ai-service

echo "=== Step 1: Sync JSONL files from cluster ==="
# Copy quality selfplay data from GH200 nodes
mkdir -p data/selfplay/quality
scp ubuntu@100.71.89.91:/home/ubuntu/ringrift/ai-service/data/selfplay/quality_sq8_2p_*.jsonl data/selfplay/quality/ 2>/dev/null || echo "No sq8 files yet"
scp ubuntu@100.121.230.110:/home/ubuntu/ringrift/ai-service/data/selfplay/quality_hex8_2p_*.jsonl data/selfplay/quality/ 2>/dev/null || echo "No hex8 files yet"

echo ""
echo "=== Step 2: Convert JSONL to SQLite database ==="
# square8_2p
python scripts/chunked_jsonl_converter.py \
  --input-dir data/selfplay/quality \
  --output-dir data/games/quality \
  --board-type square8 \
  --num-players 2 \
  --force

# hex8_2p
python scripts/chunked_jsonl_converter.py \
  --input-dir data/selfplay/quality \
  --output-dir data/games/quality \
  --board-type hex8 \
  --num-players 2 \
  --force

echo ""
echo "=== Step 3: Export to NPZ training data ==="
# square8_2p
PYTHONPATH=. python scripts/export_replay_dataset.py \
  --db data/games/quality/square8_2p_*.db \
  --board-type square8 \
  --num-players 2 \
  --output data/training/quality_sq8_2p.npz \
  --allow-noncanonical

# hex8_2p
PYTHONPATH=. python scripts/export_replay_dataset.py \
  --db data/games/quality/hex8_2p_*.db \
  --board-type hex8 \
  --num-players 2 \
  --output data/training/quality_hex8_2p.npz \
  --allow-noncanonical

echo ""
echo "=== Step 4: Train models ==="
# square8_2p (on local or cluster)
PYTHONPATH=. python -m app.training.train \
  --board-type square8 \
  --num-players 2 \
  --data-path data/training/quality_sq8_2p.npz \
  --model-version v2 \
  --epochs 50 \
  --save-path models/quality_sq8_2p.pth

# hex8_2p
PYTHONPATH=. python -m app.training.train \
  --board-type hex8 \
  --num-players 2 \
  --data-path data/training/quality_hex8_2p.npz \
  --model-version v2 \
  --epochs 50 \
  --save-path models/quality_hex8_2p.pth

echo ""
echo "=== Step 5: Run gauntlet verification ==="
# square8_2p
PYTHONPATH=. python -c "
from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent
from app.models.core import BoardType

print('=== square8_2p Gauntlet ===')
results = run_baseline_gauntlet(
    model_path='models/quality_sq8_2p.pth',
    board_type=BoardType.SQUARE8,
    num_players=2,
    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
    games_per_opponent=30,
    verbose=True
)
print(f'vs Random: {results.baseline_results[\"random\"].wins}/{results.baseline_results[\"random\"].games_played}')
print(f'vs Heuristic: {results.baseline_results[\"heuristic\"].wins}/{results.baseline_results[\"heuristic\"].games_played}')
"

# hex8_2p
PYTHONPATH=. python -c "
from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent
from app.models.core import BoardType

print('=== hex8_2p Gauntlet ===')
results = run_baseline_gauntlet(
    model_path='models/quality_hex8_2p.pth',
    board_type=BoardType.HEX8,
    num_players=2,
    opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
    games_per_opponent=30,
    verbose=True
)
print(f'vs Random: {results.baseline_results[\"random\"].wins}/{results.baseline_results[\"random\"].games_played}')
print(f'vs Heuristic: {results.baseline_results[\"heuristic\"].wins}/{results.baseline_results[\"heuristic\"].games_played}')
"

echo ""
echo "=== Pipeline Complete ==="
echo "Success criteria:"
echo "  - vs Random: 85%+ (25.5/30 wins)"
echo "  - vs Heuristic: 60%+ (18/30 wins)"

#!/usr/bin/env bash
#
# Export NN training datasets from selfplay_* GameReplayDBs.
#
# This script runs export_replay_dataset.py over all known selfplay DBs:
#   data/games/selfplay_<board>_<players>p.db
#
# For each board in {square8, square19, hexagonal} it:
#   - Looks for DBs for num_players ∈ {2,3,4}.
#   - If the DB exists, exports samples into a per-board NPZ:
#       data/training/from_replays.<board>.npz
#   - Multiple player-counts and multiple invocations append into the same
#     NPZ file, using the append semantics in export_replay_dataset.py.
#
# Tunable via environment variables:
#   EXPORT_SAMPLE_EVERY       – use every Nth move (default: 1)
#   EXPORT_HISTORY_LENGTH     – history length for stacking (default: 3)
#   EXPORT_MAX_GAMES_PER_DB   – cap games per DB (default: unlimited)
#
# Usage (from ai-service/):
#   chmod +x scripts/export_replay_matrix.sh
#   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. scripts/export_replay_matrix.sh

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

BOARD_TYPES=(square8 square19 hexagonal)
PLAYER_COUNTS=(2 3 4)

SAMPLE_EVERY="${EXPORT_SAMPLE_EVERY:-1}"
HISTORY_LENGTH="${EXPORT_HISTORY_LENGTH:-3}"
MAX_GAMES_PER_DB="${EXPORT_MAX_GAMES_PER_DB:-}"

mkdir -p "data/training"

echo "Exporting replay datasets from selfplay_* DBs..."
echo "  Boards:        ${BOARD_TYPES[*]}"
echo "  Players:       ${PLAYER_COUNTS[*]}"
echo "  sample_every:  ${SAMPLE_EVERY}"
echo "  history_length:${HISTORY_LENGTH}"
if [[ -n "${MAX_GAMES_PER_DB}" ]]; then
  echo "  max_games/db:  ${MAX_GAMES_PER_DB}"
fi
echo

for board in "${BOARD_TYPES[@]}"; do
  output_path="data/training/from_replays.${board}.npz"

  for players in "${PLAYER_COUNTS[@]}"; do
    db_path="data/games/selfplay_${board}_${players}p.db"
    if [[ ! -f "${db_path}" ]]; then
      continue
    fi

    echo "=== Exporting ${board} ${players}p from ${db_path} → ${output_path} ==="

    args=(
      scripts/export_replay_dataset.py
      --db "${db_path}"
      --board-type "${board}"
      --num-players "${players}"
      --output "${output_path}"
      --append
      --history-length "${HISTORY_LENGTH}"
      --sample-every "${SAMPLE_EVERY}"
    )
    if [[ -n "${MAX_GAMES_PER_DB}" ]]; then
      args+=(--max-games "${MAX_GAMES_PER_DB}")
    fi

    PYTHONPATH=. python "${args[@]}"
    echo
  done
done

echo "Replay dataset export complete."

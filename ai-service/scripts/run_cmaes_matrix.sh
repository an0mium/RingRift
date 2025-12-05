#!/usr/bin/env bash
#
# Sequential CMA-ES runs to generate games.db replays.
#
# Runs small CMA-ES optimisation jobs for each board type, recording all
# evaluation games into per-run SQLite DBs under logs/cmaes/runs/<run_id>/games.db.
#
# Boards covered:
#   - square8
#   - square19
#   - hex (HEXAGONAL)
#
# All runs are sequential to avoid excessive memory usage.
#
# Tunable via environment variables:
#   CMAES_GENS_SQUARE8   – generations for square8 (default 3)
#   CMAES_GENS_SQUARE19  – generations for square19 (default 2)
#   CMAES_GENS_HEX       – generations for hex (default 2)
#   CMAES_POP_SQUARE8    – population size for square8 (default 8)
#   CMAES_POP_SQUARE19   – population size for square19 (default 6)
#   CMAES_POP_HEX        – population size for hex (default 6)
#   CMAES_GAMES_PER_EVAL – games per candidate evaluation (default 4)
#   CMAES_MAX_MOVES      – max moves per game (default 220)
#   CMAES_SEED           – base random seed (default 20251205)
#
# Usage (from ai-service/):
#   chmod +x scripts/run_cmaes_matrix.sh
#   PYTHONPATH=. scripts/run_cmaes_matrix.sh

set -euo pipefail

# Limit threads to avoid memory exhaustion and OMP crashes on macOS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# NOTE: hex is excluded due to PyTorch MPS adaptive pooling crash
# on Apple Silicon. Re-add once neural_net.py architecture is fixed.
BOARD_TYPES=(square8 square19)

# Conservative defaults to avoid memory exhaustion
GENS_SQUARE8="${CMAES_GENS_SQUARE8:-2}"
GENS_SQUARE19="${CMAES_GENS_SQUARE19:-1}"

POP_SQUARE8="${CMAES_POP_SQUARE8:-4}"
POP_SQUARE19="${CMAES_POP_SQUARE19:-3}"

GAMES_PER_EVAL="${CMAES_GAMES_PER_EVAL:-2}"
MAX_MOVES="${CMAES_MAX_MOVES:-100}"
SEED="${CMAES_SEED:-20251205}"

BASE_OUTPUT_DIR="logs/cmaes"
mkdir -p "${BASE_OUTPUT_DIR}"

echo "Starting CMA-ES matrix runs..."
echo "  Boards:          ${BOARD_TYPES[*]}"
echo "  Games per eval:  ${GAMES_PER_EVAL}"
echo "  Max moves:       ${MAX_MOVES}"
echo "  Seed:            ${SEED}"
echo

for board in "${BOARD_TYPES[@]}"; do
  case "${board}" in
    square8)
      generations="${GENS_SQUARE8}"
      pop="${POP_SQUARE8}"
      ;;
    square19)
      generations="${GENS_SQUARE19}"
      pop="${POP_SQUARE19}"
      ;;
    *)
      echo "Unknown board: ${board}" >&2
      exit 1
      ;;
  esac

  output_path="${BASE_OUTPUT_DIR}/${board}_2p.cmaes_test_weights.json"

  echo "=== Running CMA-ES on ${board} (gens=${generations}, pop=${pop}) ==="

  RINGRIFT_SKIP_SHADOW_CONTRACTS="${RINGRIFT_SKIP_SHADOW_CONTRACTS:-true}" \
  PYTHONPATH=. \
    python scripts/run_cmaes_optimization.py \
      --generations "${generations}" \
      --population-size "${pop}" \
      --games-per-eval "${GAMES_PER_EVAL}" \
      --max-moves "${MAX_MOVES}" \
      --board "${board}" \
      --eval-boards "${board}" \
      --num-players 2 \
      --eval-mode multi-start \
      --state-pool-id v1 \
      --seed "${SEED}" \
      --output "${output_path}"

  echo "Completed CMA-ES run for ${board}."
  echo
done

echo "CMA-ES matrix runs complete."


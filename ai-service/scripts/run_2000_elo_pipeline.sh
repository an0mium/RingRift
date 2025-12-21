#!/bin/bash
# 2000+ Elo Training Pipeline
# ===========================
# Orchestrates the full pipeline for training models to exceed 2000 Elo
# across all 12 board/player configurations using Gumbel MCTS.
#
# This script runs the following phases:
# 1. Hexagonal ANM parity diagnosis (if needed)
# 2. Gumbel MCTS selfplay data generation for all 12 configs
# 3. Experimental tier tournament (D12-D19 algorithms)
# 4. Canonical training data aggregation
#
# Usage:
#   ./scripts/run_2000_elo_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-anm          Skip ANM diagnosis phase
#   --skip-selfplay     Skip selfplay generation phase
#   --skip-tournament   Skip experimental tournament phase
#   --games N           Number of games per config (default: 200)
#   --budget N          Gumbel simulation budget (default: 200)
#   --dry-run           Show what would be done without executing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PROJECT_ROOT}/data/2000_elo_pipeline_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/logs"

# Default settings
GAMES_PER_CONFIG=200
SIMULATION_BUDGET=200
SKIP_ANM=false
SKIP_SELFPLAY=false
SKIP_TOURNAMENT=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-anm)
            SKIP_ANM=true
            shift
            ;;
        --skip-selfplay)
            SKIP_SELFPLAY=true
            shift
            ;;
        --skip-tournament)
            SKIP_TOURNAMENT=true
            shift
            ;;
        --games)
            GAMES_PER_CONFIG="$2"
            shift 2
            ;;
        --budget)
            SIMULATION_BUDGET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure we're in the right directory
cd "${PROJECT_ROOT}"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}/selfplay"
mkdir -p "${OUTPUT_DIR}/tournaments"
mkdir -p "${OUTPUT_DIR}/anm_diagnosis"

echo "============================================================"
echo "2000+ Elo Training Pipeline"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Games per config: ${GAMES_PER_CONFIG}"
echo "Simulation budget: ${SIMULATION_BUDGET}"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN MODE - No commands will be executed]"
    echo ""
fi

# Phase 1: Hexagonal ANM Parity Diagnosis
# ----------------------------------------
if [ "$SKIP_ANM" = false ]; then
    echo "============================================================"
    echo "Phase 1: Hexagonal ANM Parity Diagnosis"
    echo "============================================================"

    if [ "$DRY_RUN" = true ]; then
        echo "[Would run] ./scripts/run_hex_anm_diagnosis.sh"
    else
        echo "Running ANM diagnosis..."

        # Run the diagnosis
        PYTHONPATH="${PROJECT_ROOT}" bash "${SCRIPT_DIR}/run_hex_anm_diagnosis.sh" \
            2>&1 | tee "${LOG_DIR}/phase1_anm_diagnosis.log"

        # Copy results
        ANM_DIR=$(ls -td "${PROJECT_ROOT}/data/anm_diagnosis_"* 2>/dev/null | head -1)
        if [ -n "$ANM_DIR" ]; then
            cp -r "$ANM_DIR"/* "${OUTPUT_DIR}/anm_diagnosis/" 2>/dev/null || true
        fi

        echo "Phase 1 complete."
    fi
    echo ""
else
    echo "Skipping Phase 1 (ANM diagnosis)"
    echo ""
fi

# Phase 2: Gumbel MCTS Selfplay Generation
# ----------------------------------------
if [ "$SKIP_SELFPLAY" = false ]; then
    echo "============================================================"
    echo "Phase 2: Gumbel MCTS Selfplay Generation"
    echo "============================================================"
    echo "Generating ${GAMES_PER_CONFIG} games for each of 12 configurations"
    echo "Simulation budget: ${SIMULATION_BUDGET}"
    echo ""

    BOARDS="square8 square19 hex8 hexagonal"
    PLAYERS="2 3 4"

    for board in $BOARDS; do
        for players in $PLAYERS; do
            CONFIG="${board}_${players}p"
            echo "--------------------------------------------------------------"
            echo "Generating: ${CONFIG}"
            echo "--------------------------------------------------------------"

            if [ "$DRY_RUN" = true ]; then
                echo "[Would run] python generate_gumbel_selfplay.py --board ${board} --num-players ${players} --num-games ${GAMES_PER_CONFIG}"
            else
                PYTHONPATH="${PROJECT_ROOT}" python "${SCRIPT_DIR}/generate_gumbel_selfplay.py" \
                    --board "${board}" \
                    --num-players "${players}" \
                    --num-games "${GAMES_PER_CONFIG}" \
                    --simulation-budget "${SIMULATION_BUDGET}" \
                    --output "${OUTPUT_DIR}/selfplay/gumbel_${CONFIG}.jsonl" \
                    --db "${OUTPUT_DIR}/selfplay/gumbel_${CONFIG}.db" \
                    2>&1 | tee -a "${LOG_DIR}/phase2_selfplay_${CONFIG}.log"
            fi

            echo ""
        done
    done

    echo "Phase 2 complete."
    echo ""
else
    echo "Skipping Phase 2 (Selfplay generation)"
    echo ""
fi

# Phase 3: Experimental Tier Tournament (D12-D19)
# ------------------------------------------------
if [ "$SKIP_TOURNAMENT" = false ]; then
    echo "============================================================"
    echo "Phase 3: Experimental Tier Tournament (D12-D19)"
    echo "============================================================"
    echo "Running tournaments for all 12 configurations"
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "[Would run] ./scripts/run_12_diverse_tournaments.sh"
    else
        # Run the 12 diverse tournaments
        bash "${SCRIPT_DIR}/run_12_diverse_tournaments.sh" \
            2>&1 | tee "${LOG_DIR}/phase3_tournaments.log"

        echo "Phase 3 complete."
    fi
    echo ""
else
    echo "Skipping Phase 3 (Experimental tournament)"
    echo ""
fi

# Phase 4: Summary and Aggregation
# --------------------------------
echo "============================================================"
echo "Phase 4: Summary and Aggregation"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "[Would aggregate results]"
else
    # Count generated games
    TOTAL_GAMES=0
    for f in "${OUTPUT_DIR}/selfplay/"*.jsonl; do
        if [ -f "$f" ]; then
            COUNT=$(wc -l < "$f" 2>/dev/null || echo 0)
            TOTAL_GAMES=$((TOTAL_GAMES + COUNT))
        fi
    done

    # Generate summary
    cat > "${OUTPUT_DIR}/summary.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "games_per_config": ${GAMES_PER_CONFIG},
    "simulation_budget": ${SIMULATION_BUDGET},
    "total_games_generated": ${TOTAL_GAMES},
    "phases_completed": {
        "anm_diagnosis": $([ "$SKIP_ANM" = false ] && echo "true" || echo "false"),
        "selfplay_generation": $([ "$SKIP_SELFPLAY" = false ] && echo "true" || echo "false"),
        "experimental_tournament": $([ "$SKIP_TOURNAMENT" = false ] && echo "true" || echo "false")
    },
    "output_directory": "${OUTPUT_DIR}"
}
EOF

    echo "Summary:"
    cat "${OUTPUT_DIR}/summary.json"
fi

echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review logs in ${LOG_DIR}/"
echo "  2. Aggregate training data: python scripts/build_canonical_dataset.py"
echo "  3. Train models: python scripts/run_canonical_training.py"
echo "  4. Evaluate: python scripts/run_tier_evaluation.py"

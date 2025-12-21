#!/bin/bash
# =============================================================================
# Experimental AI Tournament (D12-D19)
# =============================================================================
# Runs tournament to discover hidden strength in stranded experimental AIs:
#   D12: EBMO (Energy-Based Move Optimization)
#   D13: GMO (Gradient Move Optimization)
#   D14: IG-GMO (Information-Gain GMO)
#   D17: GMO-MCTS Hybrid
#   D18-D19: GMO v2
#
# This tournament compares experimental AIs against production tiers (D7-D11)
# to identify potential Elo gains from underutilized algorithms.
#
# Usage:
#   ./scripts/run_experimental_ai_tournament.sh [board] [players] [games]
#
# Examples:
#   ./scripts/run_experimental_ai_tournament.sh square8 2 100
#   ./scripts/run_experimental_ai_tournament.sh              # Uses defaults
#
# December 2025 - RingRift AI Training Pipeline
# =============================================================================

set -euo pipefail

# Configuration
BOARD="${1:-square8}"
PLAYERS="${2:-2}"
GAMES="${3:-50}"

# Cluster node for tournament (use lambda-h100 for speed)
TOURNAMENT_HOST="${TOURNAMENT_HOST:-100.78.101.123}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_cluster}"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ssh_node() {
    ssh $SSH_OPTS -i "$SSH_KEY" "ubuntu@$TOURNAMENT_HOST" "$@"
}

log "=============================================="
log "Experimental AI Tournament (D12-D19)"
log "=============================================="
log "Board: $BOARD"
log "Players: $PLAYERS"
log "Games per matchup: $GAMES"
log "Host: $TOURNAMENT_HOST"
log "=============================================="

# Update code first
log "Updating code on tournament host..."
ssh_node 'cd ~/ringrift/ai-service && git fetch origin && git reset --hard origin/main && git log -1 --oneline'

# Run the tournament
log "Starting experimental AI tournament..."

ssh_node "cd ~/ringrift/ai-service && source venv/bin/activate && \
    mkdir -p logs results && \
    PYTHONPATH=. python scripts/run_distributed_tournament.py \
        --board-type $BOARD \
        --num-players $PLAYERS \
        --games-per-matchup $GAMES \
        --tiers D7,D8,D9,D10,D11,D12,D13,D14,D17,D18,D19 \
        --output results/experimental_tournament_${BOARD}_${PLAYERS}p.json \
        --workers 4 \
        2>&1 | tee logs/experimental_tournament_${BOARD}_${PLAYERS}p.log"

# Fetch results
log "Fetching results..."
mkdir -p results
scp $SSH_OPTS -i "$SSH_KEY" "ubuntu@$TOURNAMENT_HOST:~/ringrift/ai-service/results/experimental_tournament_${BOARD}_${PLAYERS}p.json" "results/" 2>/dev/null || true

log ""
log "=============================================="
log "Tournament Complete!"
log "=============================================="
log ""
log "Results saved to: results/experimental_tournament_${BOARD}_${PLAYERS}p.json"
log ""
log "View results:"
log "  cat results/experimental_tournament_${BOARD}_${PLAYERS}p.json | python -m json.tool | head -100"
log ""
log "Key metrics to look for:"
log "  - Any D12-D19 with win rate > 50% vs D9-D11"
log "  - Elo ratings higher than current production tiers"
log "  - Low variance results (consistent performance)"

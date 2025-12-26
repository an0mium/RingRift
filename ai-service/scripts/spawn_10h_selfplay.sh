#!/bin/bash
# spawn_10h_selfplay.sh - Spawn long-running selfplay on all GPU nodes
# Prioritizes data-starved configurations for maximum training data generation
#
# Usage: ./scripts/spawn_10h_selfplay.sh [--dry-run]

set -eo pipefail

DRY_RUN="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs/selfplay_10h"
mkdir -p "$LOG_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Node configurations: name|target|port|key|gpu_tier (1=best, 3=entry)
# Tier 1: H100, A100 80GB - assign to largest/hardest boards
# Tier 2: A100 20GB, L40S, A40, RTX 5090/5080 - medium boards
# Tier 3: RTX 4060 Ti, 3060 Ti, 3060 - smaller boards

declare -A NODES=(
  # Tier 1 - High-end (large boards: hexagonal, square19)
  ["nebius-h100-1"]="ubuntu@89.169.111.139|22|/Users/armand/.ssh/id_cluster|1"
  ["runpod-h100"]="root@102.210.171.65|30755|/Users/armand/.runpod/ssh/RunPod-Key-Go|1"
  ["runpod-a100-1"]="root@38.128.233.145|33085|/Users/armand/.runpod/ssh/RunPod-Key-Go|1"
  ["runpod-a100-2"]="root@104.255.9.187|11681|/Users/armand/.runpod/ssh/RunPod-Key-Go|1"

  # Tier 2 - Mid-tier (medium boards)
  ["vultr-a100-20gb"]="root@208.167.249.164|22|/Users/armand/.ssh/id_ed25519|2"
  ["vultr-a100-20gb-2"]="root@140.82.15.69|22|/Users/armand/.ssh/id_ed25519|2"
  ["nebius-backbone-1"]="ubuntu@89.169.112.47|22|/Users/armand/.ssh/id_cluster|2"
  ["nebius-l40s-2"]="ubuntu@89.169.108.182|22|/Users/armand/.ssh/id_cluster|2"
  ["runpod-l40s-2"]="root@193.183.22.62|1630|/Users/armand/.runpod/ssh/RunPod-Key-Go|2"
  ["vast-28918742"]="root@ssh8.vast.ai|38742|/Users/armand/.ssh/id_cluster|2"
  ["vast-28925166"]="root@ssh1.vast.ai|15166|/Users/armand/.ssh/id_cluster|2"
  ["vast-29128356"]="root@ssh7.vast.ai|18356|/Users/armand/.ssh/id_cluster|2"
  ["vast-29031159"]="root@ssh5.vast.ai|31158|/Users/armand/.ssh/id_cluster|2"

  # Tier 3 - Entry (small boards: hex8, square8)
  ["vast-29126088"]="root@ssh5.vast.ai|16088|/Users/armand/.ssh/id_cluster|3"
  ["vast-28889766"]="root@ssh3.vast.ai|19766|/Users/armand/.ssh/id_cluster|3"
  ["vast-29046315"]="root@ssh2.vast.ai|16314|/Users/armand/.ssh/id_cluster|3"
  ["vast-29031161"]="root@ssh2.vast.ai|31160|/Users/armand/.ssh/id_cluster|3"
)

# Priority configs by data starvation (most starved first)
# Format: board|players|priority_weight
PRIORITY_CONFIGS=(
  "hexagonal|4|10"    # 11 games - CRITICAL
  "hexagonal|2|9"     # 24 games - CRITICAL
  "square19|4|8"      # 26 games - CRITICAL
  "square19|2|7"      # 81 games - CRITICAL
  "square19|3|6"      # 156 games - LOW
  "hexagonal|3|5"     # 209 games - LOW
  "hex8|2|4"          # 264 games - LOW
  "square8|3|3"       # 494 games - LOW
  "square8|4|2"       # 5752 games - supplemental
  "hex8|4|1"          # 8059 games - supplemental
)

# Assign configs to nodes based on tier
get_config_for_node() {
  local node="$1"
  local tier="$2"
  local node_index="$3"

  case $tier in
    1) # Tier 1: Large boards (hexagonal, square19)
      case $((node_index % 4)) in
        0) echo "hexagonal|4|5000" ;;
        1) echo "hexagonal|2|5000" ;;
        2) echo "square19|4|3000" ;;
        3) echo "square19|2|3000" ;;
      esac
      ;;
    2) # Tier 2: Mix of boards
      case $((node_index % 6)) in
        0) echo "hexagonal|3|3000" ;;
        1) echo "square19|3|3000" ;;
        2) echo "hexagonal|4|2000" ;;
        3) echo "square19|4|2000" ;;
        4) echo "hex8|2|5000" ;;
        5) echo "square8|3|5000" ;;
      esac
      ;;
    3) # Tier 3: Small boards only
      case $((node_index % 4)) in
        0) echo "hex8|2|8000" ;;
        1) echo "square8|3|8000" ;;
        2) echo "hex8|4|6000" ;;
        3) echo "square8|4|6000" ;;
      esac
      ;;
  esac
}

spawn_selfplay() {
  local node="$1"
  local target="$2"
  local port="$3"
  local key="$4"
  local board="$5"
  local players="$6"
  local games="$7"

  local work_dir="~/ringrift/ai-service"
  # RunPod uses /workspace
  [[ "$node" == runpod-* ]] && work_dir="/workspace/ringrift/ai-service"

  local cmd="cd $work_dir && source venv/bin/activate 2>/dev/null || true && "
  cmd+="pkill -f 'selfplay.py.*--board $board.*--num-players $players' 2>/dev/null || true && sleep 1 && "
  cmd+="nohup python scripts/selfplay.py --board $board --num-players $players --engine gumbel --num-games $games --output-dir data/games "
  cmd+="> /tmp/selfplay_${board}_${players}p_10h.log 2>&1 &"

  if [ "$DRY_RUN" = "--dry-run" ]; then
    log "[DRY-RUN] Would spawn on $node: $board ${players}p x $games games"
    return 0
  fi

  ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$key" -p "$port" "$target" "$cmd" 2>/dev/null
  return $?
}

main() {
  log "Starting 10-hour selfplay deployment..."
  log "Deploying to ${#NODES[@]} GPU nodes"

  if [ "$DRY_RUN" = "--dry-run" ]; then
    warn "DRY RUN MODE - no actual jobs will be spawned"
  fi

  local tier1_idx=0
  local tier2_idx=0
  local tier3_idx=0
  local success_count=0
  local fail_count=0

  for node in "${!NODES[@]}"; do
    IFS='|' read -r target port key tier <<< "${NODES[$node]}"

    case $tier in
      1) config=$(get_config_for_node "$node" "$tier" "$tier1_idx"); ((tier1_idx++)) ;;
      2) config=$(get_config_for_node "$node" "$tier" "$tier2_idx"); ((tier2_idx++)) ;;
      3) config=$(get_config_for_node "$node" "$tier" "$tier3_idx"); ((tier3_idx++)) ;;
    esac

    IFS='|' read -r board players games <<< "$config"

    log "Spawning on $node: $board ${players}p x $games games..."

    if spawn_selfplay "$node" "$target" "$port" "$key" "$board" "$players" "$games"; then
      success "$node: $board ${players}p started"
      ((success_count++))
    else
      error "$node: Failed to spawn"
      ((fail_count++))
    fi
  done

  echo ""
  log "Deployment complete: $success_count succeeded, $fail_count failed"
  log "Jobs will run for approximately 10 hours generating training data"
  log "Monitor with: ./scripts/spawn_10h_selfplay.sh --status"
}

# Status check mode
if [ "${1:-}" = "--status" ]; then
  log "Checking selfplay status on all nodes..."
  for node in "${!NODES[@]}"; do
    IFS='|' read -r target port key tier <<< "${NODES[$node]}"
    count=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$key" -p "$port" "$target" \
      "ps aux | grep 'selfplay.py' | grep -v grep | wc -l" 2>/dev/null || echo "?")
    if [ "$count" = "?" ]; then
      warn "$node: unreachable"
    elif [ "$count" = "0" ]; then
      error "$node: no selfplay running"
    else
      success "$node: $count selfplay job(s)"
    fi
  done
  exit 0
fi

main

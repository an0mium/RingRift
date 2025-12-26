#!/usr/bin/env bash
#
# RingRift Cluster Join Script
#
# Automatically detects node capabilities and assigns appropriate role.
# Sets RINGRIFT_NODE_ROLE environment variable and starts daemons.
#
# Usage:
#   ./scripts/cluster_join.sh              # Auto-detect role and join
#   ./scripts/cluster_join.sh --role X     # Force specific role
#   ./scripts/cluster_join.sh --dry-run    # Show what would happen
#   ./scripts/cluster_join.sh --status     # Check current daemon status
#
# Roles:
#   coordinator      - Leader node, manages pipeline (requires COORDINATOR flag)
#   training_node    - Primary training workloads (high-end GPU: H100, A100, GH200)
#   gpu_selfplay     - GPU-accelerated selfplay (any CUDA GPU)
#   ephemeral        - Volatile nodes (Vast.ai, RunPod, etc.)
#   backbone         - Stable connectivity node for P2P mesh
#
# Environment variables:
#   RINGRIFT_FORCE_COORDINATOR=1  - Force coordinator role (requires manual opt-in)
#   RINGRIFT_EPHEMERAL=1          - Mark as ephemeral (volatile) node
#   RINGRIFT_SKIP_DAEMONS=1       - Skip daemon startup
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="${AI_SERVICE_DIR}/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_debug() { [[ "${VERBOSE:-0}" == "1" ]] && echo -e "${CYAN}[DEBUG]${NC} $*" || true; }

# Parse arguments
DRY_RUN=0
FORCE_ROLE=""
SHOW_STATUS=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --role)
            FORCE_ROLE="$2"
            shift 2
            ;;
        --status)
            SHOW_STATUS=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            cat << 'EOF'
RingRift Cluster Join Script

Usage:
  ./scripts/cluster_join.sh              # Auto-detect role and join
  ./scripts/cluster_join.sh --role X     # Force specific role
  ./scripts/cluster_join.sh --dry-run    # Show what would happen
  ./scripts/cluster_join.sh --status     # Check current daemon status
  ./scripts/cluster_join.sh --verbose    # Enable debug output

Roles:
  coordinator      - Pipeline leader (requires RINGRIFT_FORCE_COORDINATOR=1)
  training_node    - Primary training (H100, A100, GH200, 2xH100)
  gpu_selfplay     - GPU selfplay (any CUDA GPU)
  ephemeral        - Volatile nodes (set RINGRIFT_EPHEMERAL=1)
  backbone         - Stable P2P connectivity node

Auto-detection priority:
  1. RINGRIFT_NODE_ROLE env var (if set)
  2. --role argument
  3. RINGRIFT_FORCE_COORDINATOR=1 -> coordinator
  4. RINGRIFT_EPHEMERAL=1 -> ephemeral
  5. GPU detection -> training_node or gpu_selfplay
  6. No GPU -> backbone (CPU only)

Examples:
  # Join as auto-detected role
  ./scripts/cluster_join.sh

  # Force ephemeral mode (Vast.ai, RunPod)
  RINGRIFT_EPHEMERAL=1 ./scripts/cluster_join.sh

  # Force training_node role
  ./scripts/cluster_join.sh --role training_node

  # Dry run to see what would happen
  ./scripts/cluster_join.sh --dry-run --verbose

EOF
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Detect GPU info
detect_gpu() {
    local gpu_name=""
    local gpu_memory_mb=0
    local has_cuda=0

    # Check for NVIDIA GPU
    if command -v nvidia-smi &>/dev/null; then
        has_cuda=1
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
        gpu_memory_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
    fi

    # Check for Apple MPS
    if [[ "$(uname)" == "Darwin" ]] && system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
        if [[ -z "$gpu_name" ]]; then
            gpu_name=$(system_profiler SPDisplaysDataType 2>/dev/null | grep "Chipset Model" | head -1 | cut -d: -f2 | xargs)
            has_cuda=0  # MPS, not CUDA
        fi
    fi

    echo "${has_cuda}|${gpu_name}|${gpu_memory_mb}"
}

# Detect if this is an ephemeral provider
detect_ephemeral() {
    # Explicit flag
    [[ "${RINGRIFT_EPHEMERAL:-0}" == "1" ]] && echo "1" && return

    # Vast.ai detection
    [[ -f /etc/vast.ai ]] && echo "1" && return
    [[ -d /workspace ]] && [[ -f /workspace/.vast ]] && echo "1" && return

    # RunPod detection
    [[ -n "${RUNPOD_POD_ID:-}" ]] && echo "1" && return
    [[ -f /runpod-volume/.runpod ]] && echo "1" && return

    # Paperspace detection
    [[ -n "${PAPERSPACE_GRADIENT_PROJECT_ID:-}" ]] && echo "1" && return

    echo "0"
}

# Determine role based on hardware
determine_role() {
    local forced_role="${FORCE_ROLE:-${RINGRIFT_NODE_ROLE:-}}"

    # 1. Check for explicit role
    if [[ -n "$forced_role" ]]; then
        echo "$forced_role"
        return
    fi

    # 2. Check for coordinator flag
    if [[ "${RINGRIFT_FORCE_COORDINATOR:-0}" == "1" ]]; then
        echo "coordinator"
        return
    fi

    # 3. Detect hardware
    local gpu_info
    gpu_info=$(detect_gpu)
    local has_cuda gpu_name gpu_memory_mb
    IFS='|' read -r has_cuda gpu_name gpu_memory_mb <<< "$gpu_info"

    local is_ephemeral
    is_ephemeral=$(detect_ephemeral)

    log_debug "GPU detection: has_cuda=$has_cuda, name=$gpu_name, memory_mb=$gpu_memory_mb"
    log_debug "Ephemeral detection: is_ephemeral=$is_ephemeral"

    # 4. Check for ephemeral
    if [[ "$is_ephemeral" == "1" ]]; then
        echo "ephemeral"
        return
    fi

    # 5. GPU-based role selection
    if [[ "$has_cuda" == "1" ]] && [[ -n "$gpu_name" ]]; then
        # High-end GPUs for training
        if echo "$gpu_name" | grep -qiE "(H100|A100|GH200|2xH100|4090|5090)"; then
            # Check memory - training needs at least 40GB for large batches
            if [[ "$gpu_memory_mb" -ge 40000 ]]; then
                echo "training_node"
                return
            fi
        fi
        # Any other CUDA GPU for selfplay
        echo "gpu_selfplay"
        return
    fi

    # 6. No GPU - backbone/CPU role
    echo "backbone"
}

# Get daemon profile for role
get_daemon_profile() {
    local role="$1"
    case "$role" in
        coordinator)
            echo "COORDINATOR,QUEUE_POPULATOR,CLUSTER_DATA_SYNC,MODEL_DISTRIBUTION,AUTO_SYNC,UNIFIED_NODE_HEALTH"
            ;;
        training_node)
            echo "TRAINING_TRIGGER,TRAINING_NODE_WATCHER,CLUSTER_DATA_SYNC,MODEL_DISTRIBUTION"
            ;;
        gpu_selfplay)
            echo "IDLE_RESOURCE,CLUSTER_DATA_SYNC,MODEL_DISTRIBUTION"
            ;;
        ephemeral)
            echo "EPHEMERAL_SYNC,IDLE_RESOURCE"
            ;;
        backbone)
            echo "CLUSTER_DATA_SYNC,AUTO_SYNC"
            ;;
        *)
            log_warn "Unknown role: $role, using minimal profile"
            echo "CLUSTER_DATA_SYNC"
            ;;
    esac
}

# Show current status
show_status() {
    log_info "Current daemon status:"
    echo

    # Check environment
    echo "Environment:"
    echo "  RINGRIFT_NODE_ROLE=${RINGRIFT_NODE_ROLE:-<not set>}"
    echo "  RINGRIFT_EPHEMERAL=${RINGRIFT_EPHEMERAL:-<not set>}"
    echo "  RINGRIFT_FORCE_COORDINATOR=${RINGRIFT_FORCE_COORDINATOR:-<not set>}"
    echo

    # Check GPU
    local gpu_info
    gpu_info=$(detect_gpu)
    local has_cuda gpu_name gpu_memory_mb
    IFS='|' read -r has_cuda gpu_name gpu_memory_mb <<< "$gpu_info"

    echo "Hardware:"
    if [[ "$has_cuda" == "1" ]]; then
        echo "  GPU: $gpu_name (${gpu_memory_mb}MB)"
    else
        echo "  GPU: None (or MPS: $gpu_name)"
    fi
    echo "  Ephemeral: $(detect_ephemeral)"
    echo

    # Determine role
    local role
    role=$(determine_role)
    echo "Detected role: ${role}"
    echo "Daemon profile: $(get_daemon_profile "$role")"
    echo

    # Check running processes
    echo "Running processes:"
    if pgrep -f "master_loop.py" &>/dev/null; then
        log_ok "  master_loop.py - RUNNING"
    else
        log_warn "  master_loop.py - NOT RUNNING"
    fi

    if pgrep -f "launch_daemons.py" &>/dev/null; then
        log_ok "  launch_daemons.py - RUNNING"
    else
        log_warn "  launch_daemons.py - NOT RUNNING"
    fi

    if pgrep -f "p2p_orchestrator" &>/dev/null; then
        log_ok "  p2p_orchestrator - RUNNING"
    else
        log_warn "  p2p_orchestrator - NOT RUNNING"
    fi

    # Check P2P status
    echo
    echo "P2P Status:"
    if curl -s --connect-timeout 2 http://localhost:8770/status &>/dev/null; then
        local status
        status=$(curl -s http://localhost:8770/status)
        local leader alive
        leader=$(echo "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("leader_id", "unknown"))' 2>/dev/null || echo "parse_error")
        alive=$(echo "$status" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("alive_peers", "?"))' 2>/dev/null || echo "?")
        log_ok "  Connected - Leader: $leader, Alive peers: $alive"
    else
        log_warn "  P2P not reachable (port 8770)"
    fi
}

# Main execution
main() {
    if [[ "$SHOW_STATUS" == "1" ]]; then
        show_status
        exit 0
    fi

    log_info "RingRift Cluster Join"
    echo

    # Detect hardware and role
    local gpu_info
    gpu_info=$(detect_gpu)
    local has_cuda gpu_name gpu_memory_mb
    IFS='|' read -r has_cuda gpu_name gpu_memory_mb <<< "$gpu_info"

    local role
    role=$(determine_role)

    local daemon_profile
    daemon_profile=$(get_daemon_profile "$role")

    # Display detection results
    echo "Hardware Detection:"
    if [[ "$has_cuda" == "1" ]]; then
        echo "  GPU: $gpu_name (${gpu_memory_mb}MB VRAM)"
    elif [[ -n "$gpu_name" ]]; then
        echo "  GPU: $gpu_name (MPS)"
    else
        echo "  GPU: None"
    fi
    echo "  Ephemeral: $(detect_ephemeral)"
    echo

    echo "Role Assignment:"
    echo "  Role: ${role}"
    echo "  Daemons: ${daemon_profile}"
    echo

    if [[ "$DRY_RUN" == "1" ]]; then
        log_info "Dry run - would execute:"
        echo "  export RINGRIFT_NODE_ROLE=${role}"
        echo "  cd ${AI_SERVICE_DIR}"
        echo "  PYTHONPATH=. python scripts/master_loop.py"
        exit 0
    fi

    # Set environment variable
    export RINGRIFT_NODE_ROLE="$role"
    log_ok "Set RINGRIFT_NODE_ROLE=${role}"

    # Add to shell profile if not already there
    local shell_rc=""
    if [[ -f ~/.bashrc ]]; then
        shell_rc=~/.bashrc
    elif [[ -f ~/.zshrc ]]; then
        shell_rc=~/.zshrc
    fi

    if [[ -n "$shell_rc" ]]; then
        if ! grep -q "RINGRIFT_NODE_ROLE" "$shell_rc" 2>/dev/null; then
            echo "" >> "$shell_rc"
            echo "# RingRift cluster role (added by cluster_join.sh)" >> "$shell_rc"
            echo "export RINGRIFT_NODE_ROLE=${role}" >> "$shell_rc"
            log_info "Added RINGRIFT_NODE_ROLE to ${shell_rc}"
        fi
    fi

    # Skip daemon startup if requested
    if [[ "${RINGRIFT_SKIP_DAEMONS:-0}" == "1" ]]; then
        log_warn "RINGRIFT_SKIP_DAEMONS=1 - skipping daemon startup"
        log_ok "Cluster join complete (role: ${role})"
        exit 0
    fi

    # Check if daemons already running
    if pgrep -f "master_loop.py" &>/dev/null; then
        log_warn "master_loop.py already running"
        log_info "Kill existing process? (y/N)"
        read -r -n 1 response
        echo
        if [[ "$response" =~ ^[Yy]$ ]]; then
            pkill -f "master_loop.py" || true
            sleep 2
        else
            log_info "Keeping existing process"
            exit 0
        fi
    fi

    # Start daemons
    log_info "Starting daemons for role: ${role}..."

    cd "${AI_SERVICE_DIR}"

    # Check for venv
    if [[ -f venv/bin/activate ]]; then
        source venv/bin/activate
    fi

    # Start master_loop.py in background
    PYTHONPATH=. nohup python scripts/master_loop.py \
        > logs/master_loop.log 2>&1 &

    local pid=$!
    sleep 2

    # Verify it started
    if kill -0 "$pid" 2>/dev/null; then
        log_ok "Started master_loop.py (PID: $pid)"
        log_info "Logs: ${AI_SERVICE_DIR}/logs/master_loop.log"
    else
        log_error "Failed to start master_loop.py"
        log_info "Check logs: ${AI_SERVICE_DIR}/logs/master_loop.log"
        exit 1
    fi

    echo
    log_ok "Cluster join complete!"
    echo "  Role: ${role}"
    echo "  Daemons: ${daemon_profile}"
    echo
    log_info "Use './scripts/cluster_join.sh --status' to check daemon status"
}

main "$@"

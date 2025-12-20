#!/bin/bash
# Vast.ai Canonical Selfplay Script
#
# This script runs canonical self-play generation on a Vast.ai instance,
# including the TS↔Python parity checks that require Node.js.
#
# Usage from local machine (not on Vast.ai):
#   ./vast_canonical_selfplay.sh --create-instance
#   ./vast_canonical_selfplay.sh --instance-id <ID> --num-games 200 --board-type square8
#
# Usage on Vast.ai instance (automatically invoked):
#   ./vast_canonical_selfplay.sh --run-on-instance --num-games 200 --board-type square8
#
# Requirements:
#   - Local machine: vastai CLI configured with API key
#   - Vast.ai instance: Ubuntu with internet access (GPU not required)
#
# The script will:
#   1. Install Node.js 20.x if not present
#   2. Install npm dependencies  
#   3. Install Python dependencies
#   4. Run generate_canonical_selfplay.py with TS↔Python parity validation
#   5. Save results to data/games/canonical_*.db

set -e

# Configuration defaults
NUM_GAMES=${NUM_GAMES:-200}
BOARD_TYPE=${BOARD_TYPE:-square8}
NUM_PLAYERS=${NUM_PLAYERS:-2}
DIFFICULTY_BAND=${DIFFICULTY_BAND:-light}
MIN_RECORDED_GAMES=${MIN_RECORDED_GAMES:-0}
MAX_SOAK_ATTEMPTS=${MAX_SOAK_ATTEMPTS:-1}
DB_PATH_OVERRIDE=""
SUMMARY_PATH_OVERRIDE=""
RESET_DB=false
SKIP_RESOURCE_GUARD=false
RINGRIFT_REPO="https://github.com/SynaptentLLC/RingRift.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
CREATE_INSTANCE=false
RUN_ON_INSTANCE=false
INSTANCE_ID=""
SSH_CMD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --create-instance) CREATE_INSTANCE=true; shift ;;
        --run-on-instance) RUN_ON_INSTANCE=true; shift ;;
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --ssh) SSH_CMD="$2"; shift 2 ;;
        --num-games) NUM_GAMES="$2"; shift 2 ;;
        --board-type|--board) BOARD_TYPE="$2"; shift 2 ;;
        --num-players) NUM_PLAYERS="$2"; shift 2 ;;
        --difficulty-band) DIFFICULTY_BAND="$2"; shift 2 ;;
        --min-recorded-games) MIN_RECORDED_GAMES="$2"; shift 2 ;;
        --max-soak-attempts) MAX_SOAK_ATTEMPTS="$2"; shift 2 ;;
        --db) DB_PATH_OVERRIDE="$2"; shift 2 ;;
        --summary) SUMMARY_PATH_OVERRIDE="$2"; shift 2 ;;
        --reset-db) RESET_DB=true; shift ;;
        --skip-resource-guard) SKIP_RESOURCE_GUARD=true; shift ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --create-instance           Create a new Vast.ai instance"
            echo "  --run-on-instance           Run selfplay on current instance (called from Vast.ai)"
            echo "  --instance-id ID            Vast.ai instance ID to use"
            echo "  --ssh 'ssh cmd'             SSH command for connecting to instance"
            echo "  --num-games N               Number of games to generate (default: 200)"
            echo "  --board-type TYPE           Board type: square8|square19|hexagonal (default: square8)"
            echo "  --board TYPE                Alias for --board-type"
            echo "  --num-players N             Number of players: 2|3|4 (default: 2)"
            echo "  --difficulty-band BAND      AI difficulty: light|canonical (default: light)"
            echo "  --min-recorded-games N      Ensure at least N games are recorded (default: 0)"
            echo "  --max-soak-attempts N        Maximum soak attempts when min-recorded-games > 0 (default: 1)"
            echo "  --db PATH                   Override output DB path"
            echo "  --summary PATH              Override summary JSON path"
            echo "  --reset-db                  Delete DB before generating new games"
            echo "  --skip-resource-guard       Set RINGRIFT_SKIP_RESOURCE_GUARD=1"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

#############################################################################
# FUNCTIONS FOR RUNNING ON VAST.AI INSTANCE
#############################################################################

setup_nodejs() {
    log_info "Checking Node.js installation..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_info "Node.js already installed: $NODE_VERSION"
        
        # Check if version is adequate (v18+)
        MAJOR_VERSION=$(echo $NODE_VERSION | sed 's/v\([0-9]*\).*/\1/')
        if [ "$MAJOR_VERSION" -ge 18 ]; then
            return 0
        fi
        log_warn "Node.js version < 18, will upgrade"
    fi
    
    log_info "Installing Node.js 20.x..."
    
    # Install via NodeSource
    if [ -f /etc/debian_version ]; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
        apt-get install -y nodejs
    elif [ -f /etc/redhat-release ]; then
        curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
        yum install -y nodejs
    else
        log_error "Unsupported OS. Please install Node.js 20+ manually."
        exit 1
    fi
    
    log_info "Node.js installed: $(node --version)"
}

setup_python() {
    log_info "Setting up Python environment..."
    
    cd ~/ringrift/ai-service
    
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install Python dependencies
    pip install --upgrade pip wheel
    pip install -r requirements.txt
    
    log_info "Python environment ready"
}

setup_npm_dependencies() {
    log_info "Installing npm dependencies..."
    
    cd ~/ringrift
    
    # Install npm dependencies for TS parity checks
    npm install
    
    # Build TypeScript if needed
    npm run build 2>/dev/null || log_warn "npm build had issues, continuing..."
    
    log_info "npm dependencies installed"
}

clone_and_setup_repo() {
    log_info "Setting up RingRift repository..."
    
    if [ -d ~/ringrift ]; then
        log_info "Repository exists, pulling latest..."
        cd ~/ringrift
        git fetch --all
        git reset --hard origin/main  # or the appropriate branch
    else
        log_info "Cloning repository..."
        git clone "$RINGRIFT_REPO" ~/ringrift
        cd ~/ringrift
    fi
    
    log_info "Repository ready"
}

run_canonical_selfplay() {
    log_info "Running canonical selfplay generation..."
    log_info "  Board type: $BOARD_TYPE"
    log_info "  Num games: $NUM_GAMES"
    log_info "  Num players: $NUM_PLAYERS"
    log_info "  Difficulty: $DIFFICULTY_BAND"
    log_info "  Min recorded games: $MIN_RECORDED_GAMES"
    log_info "  Max soak attempts: $MAX_SOAK_ATTEMPTS"
    
    cd ~/ringrift/ai-service
    source venv/bin/activate
    
    # Set environment for canonical selfplay
    export PYTHONPATH="$HOME/ringrift/ai-service"
    export RINGRIFT_STRICT_NO_MOVE_INVARIANT=1
    export RINGRIFT_PARITY_VALIDATION=strict
    export RINGRIFT_FORCE_BOOKKEEPING_MOVES=1
    if $SKIP_RESOURCE_GUARD; then
        export RINGRIFT_SKIP_RESOURCE_GUARD=1
    fi
    
    # Create data directories
    mkdir -p data/games logs/selfplay
    
    # Determine output paths
    DB_PATH="data/games/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.db"
    SUMMARY_PATH="data/games/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.summary.json"
    if [ -n "$DB_PATH_OVERRIDE" ]; then
        DB_PATH="$DB_PATH_OVERRIDE"
    fi
    if [ -n "$SUMMARY_PATH_OVERRIDE" ]; then
        SUMMARY_PATH="$SUMMARY_PATH_OVERRIDE"
    fi
    
    # Run canonical selfplay with parity gate
    GATE_ARGS=(
        --board-type "$BOARD_TYPE"
        --num-games "$NUM_GAMES"
        --num-players "$NUM_PLAYERS"
        --difficulty-band "$DIFFICULTY_BAND"
        --min-recorded-games "$MIN_RECORDED_GAMES"
        --max-soak-attempts "$MAX_SOAK_ATTEMPTS"
        --db "$DB_PATH"
        --summary "$SUMMARY_PATH"
    )
    if $RESET_DB; then
        GATE_ARGS+=(--reset-db)
    fi

    python scripts/generate_canonical_selfplay.py \
        "${GATE_ARGS[@]}" \
        2>&1 | tee "logs/selfplay/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.log"
    
    RC=$?
    
    if [ $RC -eq 0 ]; then
        log_info "Canonical selfplay completed successfully!"
        log_info "Database: $DB_PATH"
        log_info "Summary: $SUMMARY_PATH"
        
        # Show summary
        if [ -f "$SUMMARY_PATH" ]; then
            echo ""
            log_info "=== Summary ==="
            cat "$SUMMARY_PATH" | python3 -m json.tool 2>/dev/null || cat "$SUMMARY_PATH"
        fi
    else
        log_error "Canonical selfplay failed with exit code $RC"
    fi
    
    return $RC
}

run_on_instance() {
    log_info "=== Running on Vast.ai Instance ==="
    log_info "Starting canonical selfplay setup..."
    
    # Setup everything
    clone_and_setup_repo
    setup_nodejs
    setup_npm_dependencies
    setup_python
    
    # Run the actual selfplay
    run_canonical_selfplay
}

#############################################################################
# FUNCTIONS FOR LOCAL MACHINE (MANAGING VAST.AI)
#############################################################################

create_vast_instance() {
    log_info "Searching for suitable Vast.ai instances..."
    
    # Search for CPU-heavy, reliable instances
    OFFERS=$(vastai search offers \
        'cpu_cores_effective >= 16 disk_space >= 100 reliability > 0.98' \
        --order 'dph_total' \
        --limit 5 \
        --raw 2>/dev/null)
    
    if [ -z "$OFFERS" ]; then
        log_error "No suitable offers found"
        exit 1
    fi
    
    echo "Available offers:"
    vastai search offers \
        'cpu_cores_effective >= 16 disk_space >= 100 reliability > 0.98' \
        --order 'dph_total' \
        --limit 5 2>/dev/null
    
    echo ""
    read -p "Enter offer ID to rent (or 'q' to quit): " OFFER_ID
    
    if [ "$OFFER_ID" = "q" ]; then
        log_info "Cancelled"
        exit 0
    fi
    
    log_info "Creating instance from offer $OFFER_ID..."
    
    # Create instance with Ubuntu image
    RESULT=$(vastai create instance "$OFFER_ID" \
        --image ubuntu:22.04 \
        --disk 100 \
        --onstart-cmd "apt-get update && apt-get install -y git python3-venv python3-pip curl" \
        --raw 2>&1)
    
    echo "$RESULT"
    
    # Extract instance ID
    NEW_INSTANCE_ID=$(echo "$RESULT" | grep -oP '"new_contract":\s*\K\d+' | head -1)
    
    if [ -n "$NEW_INSTANCE_ID" ]; then
        log_info "Instance created: $NEW_INSTANCE_ID"
        log_info "Wait a few minutes for the instance to start, then run:"
        echo ""
        echo "  ./vast_canonical_selfplay.sh --instance-id $NEW_INSTANCE_ID --num-games $NUM_GAMES --board-type $BOARD_TYPE"
    else
        log_warn "Could not extract instance ID. Check your Vast.ai dashboard."
    fi
}

get_instance_ssh() {
    if [ -z "$INSTANCE_ID" ]; then
        log_error "No instance ID provided. Use --instance-id"
        exit 1
    fi
    
    log_info "Getting SSH info for instance $INSTANCE_ID..."
    
    # Get instance info
    INSTANCE_INFO=$(vastai show instance "$INSTANCE_ID" --raw 2>&1)
    
    # Extract SSH info
    SSH_HOST=$(echo "$INSTANCE_INFO" | grep -oP '"ssh_host":\s*"\K[^"]+' | head -1)
    SSH_PORT=$(echo "$INSTANCE_INFO" | grep -oP '"ssh_port":\s*\K\d+' | head -1)
    
    if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
        log_error "Could not extract SSH info. Instance may not be ready."
        echo "Instance info:"
        echo "$INSTANCE_INFO"
        exit 1
    fi
    
    SSH_CMD="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 -p $SSH_PORT root@$SSH_HOST"
    log_info "SSH Command: $SSH_CMD"
    echo "$SSH_CMD"
}

run_on_remote() {
    if [ -z "$SSH_CMD" ]; then
        SSH_CMD=$(get_instance_ssh)
    fi
    
    log_info "Connecting to instance and running selfplay..."
    
    # First, copy this script to the instance
    SCRIPT_PATH=$(realpath "$0")
    SCRIPT_NAME=$(basename "$0")
    
    log_info "Copying script to instance..."
    scp -o StrictHostKeyChecking=no \
        -P $(echo "$SSH_CMD" | grep -oP '\-p\s*\K\d+') \
        "$SCRIPT_PATH" \
        "root@$(echo "$SSH_CMD" | grep -oP '@\K[^\s]+'):~/$SCRIPT_NAME"
    
    log_info "Running canonical selfplay on instance..."

    REMOTE_ARGS="--run-on-instance --num-games $NUM_GAMES --board-type $BOARD_TYPE --num-players $NUM_PLAYERS --difficulty-band $DIFFICULTY_BAND --min-recorded-games $MIN_RECORDED_GAMES --max-soak-attempts $MAX_SOAK_ATTEMPTS"
    if [ -n "$DB_PATH_OVERRIDE" ]; then
        REMOTE_ARGS="$REMOTE_ARGS --db $DB_PATH_OVERRIDE"
    fi
    if [ -n "$SUMMARY_PATH_OVERRIDE" ]; then
        REMOTE_ARGS="$REMOTE_ARGS --summary $SUMMARY_PATH_OVERRIDE"
    fi
    if $RESET_DB; then
        REMOTE_ARGS="$REMOTE_ARGS --reset-db"
    fi
    if $SKIP_RESOURCE_GUARD; then
        REMOTE_ARGS="$REMOTE_ARGS --skip-resource-guard"
    fi

    $SSH_CMD "chmod +x ~/$SCRIPT_NAME && ~/$SCRIPT_NAME $REMOTE_ARGS"
    
    RC=$?
    
    if [ $RC -eq 0 ]; then
        log_info "Selfplay completed. Retrieving results..."
        retrieve_results
    else
        log_error "Selfplay failed on remote instance"
    fi
    
    return $RC
}

retrieve_results() {
    log_info "Retrieving results from instance..."
    
    # Get SSH components
    SSH_PORT=$(echo "$SSH_CMD" | grep -oP '\-p\s*\K\d+')
    SSH_HOST=$(echo "$SSH_CMD" | grep -oP '@\K[^\s]+')
    
    LOCAL_DIR="ai-service/data/games/vast_results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOCAL_DIR"
    
    # Copy database and summary
    DB_PATH="$DB_PATH_OVERRIDE"
    SUMMARY_PATH="$SUMMARY_PATH_OVERRIDE"
    if [ -z "$DB_PATH" ]; then
        DB_PATH="data/games/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.db"
    fi
    if [ -z "$SUMMARY_PATH" ]; then
        SUMMARY_PATH="data/games/canonical_${BOARD_TYPE}_${NUM_PLAYERS}p.summary.json"
    fi

    DB_REMOTE_PATH="$DB_PATH"
    SUMMARY_REMOTE_PATH="$SUMMARY_PATH"
    if [[ "$DB_REMOTE_PATH" != /* && "$DB_REMOTE_PATH" != "~"* ]]; then
        DB_REMOTE_PATH="ringrift/ai-service/$DB_REMOTE_PATH"
    fi
    if [[ "$SUMMARY_REMOTE_PATH" != /* && "$SUMMARY_REMOTE_PATH" != "~"* ]]; then
        SUMMARY_REMOTE_PATH="ringrift/ai-service/$SUMMARY_REMOTE_PATH"
    fi
    
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        "root@$SSH_HOST:$DB_REMOTE_PATH" \
        "$LOCAL_DIR/" 2>/dev/null && log_info "Downloaded database" || log_warn "No database found"
    
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" \
        "root@$SSH_HOST:$SUMMARY_REMOTE_PATH" \
        "$LOCAL_DIR/" 2>/dev/null && log_info "Downloaded summary" || log_warn "No summary found"
    
    log_info "Results saved to: $LOCAL_DIR"
    ls -la "$LOCAL_DIR/"
}

show_status() {
    log_info "Current Vast.ai instances:"
    vastai show instances 2>&1
}

#############################################################################
# MAIN ENTRY POINT
#############################################################################

if [ "$RUN_ON_INSTANCE" = true ]; then
    run_on_instance
elif [ "$CREATE_INSTANCE" = true ]; then
    create_vast_instance
elif [ -n "$INSTANCE_ID" ]; then
    get_instance_ssh
    run_on_remote
else
    echo "Vast.ai Canonical Selfplay Manager"
    echo ""
    echo "Current instances:"
    show_status
    echo ""
    echo "Usage:"
    echo "  $0 --create-instance                  # Create new instance"
    echo "  $0 --instance-id <ID> --num-games 200 # Run on existing instance"
    echo "  $0 --help                             # Show all options"
fi

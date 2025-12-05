#!/usr/bin/env bash
#
# RingRift Cluster Setup Script
#
# Automates setup of distributed selfplay workers on local Macs.
# Discovers potential workers, tests connectivity, and installs dependencies.
#
# Usage:
#   ./scripts/cluster_setup.sh discover     # Scan network for Macs
#   ./scripts/cluster_setup.sh test         # Test SSH to known workers
#   ./scripts/cluster_setup.sh setup HOST   # Set up a specific worker
#   ./scripts/cluster_setup.sh setup-all    # Set up all workers in cluster_workers.txt
#   ./scripts/cluster_setup.sh start HOST   # Start worker service on HOST
#   ./scripts/cluster_setup.sh status       # Check status of all workers
#
# For workers without SSH, instructions are provided to enable it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="${SCRIPT_DIR}/.."
PROJECT_ROOT="${AI_SERVICE_DIR}/.."
WORKERS_FILE="${SCRIPT_DIR}/cluster_workers.txt"
REMOTE_PROJECT_DIR="~/Development/RingRift"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Discover potential workers on local network via ARP scan
cmd_discover() {
    log_info "Scanning local network for potential workers..."

    # Get local network info
    local gateway
    gateway=$(route get default 2>/dev/null | awk '/gateway/ {print $2}' | head -1)
    local subnet
    subnet=$(echo "${gateway}" | sed 's/\.[0-9]*$/.0\/24/')

    log_info "Network: ${subnet} (gateway: ${gateway})"
    echo

    # Use arp to find hosts (non-invasive scan)
    log_info "Found hosts (from ARP cache):"
    arp -a 2>/dev/null | grep -v incomplete | while read -r line; do
        local ip
        ip=$(echo "$line" | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
        if [[ -n "$ip" ]]; then
            # Test SSH connectivity
            if timeout 2 nc -z "$ip" 22 2>/dev/null; then
                echo -e "  ${GREEN}$ip${NC} - SSH open"
            else
                echo -e "  ${YELLOW}$ip${NC} - SSH closed"
            fi
        fi
    done

    echo
    log_info "To add a worker, append its IP to: ${WORKERS_FILE}"
    log_info "Then run: ./scripts/cluster_setup.sh setup-all"
}

# Test SSH connectivity to all known workers
cmd_test() {
    log_info "Testing SSH connectivity to known workers..."

    if [[ ! -f "${WORKERS_FILE}" ]]; then
        log_error "Workers file not found: ${WORKERS_FILE}"
        exit 1
    fi

    local pass=0
    local fail=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -z "$line" ]] && continue

        local host="$line"
        if timeout 5 ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "echo ok" &>/dev/null; then
            log_ok "${host} - SSH connected"
            ((pass++))
        elif timeout 3 nc -z "$host" 22 2>/dev/null; then
            log_warn "${host} - SSH port open but auth failed (add SSH key)"
            ((fail++))
        else
            log_error "${host} - SSH port closed (enable Remote Login)"
            ((fail++))
        fi
    done < "${WORKERS_FILE}"

    echo
    log_info "Results: ${pass} connected, ${fail} failed"

    if [[ $fail -gt 0 ]]; then
        echo
        echo "To enable SSH on a Mac:"
        echo "  System Preferences → Sharing → Remote Login → ON"
        echo
        echo "To copy your SSH key to a worker:"
        echo "  ssh-copy-id <host>"
    fi
}

# Set up a single worker
cmd_setup() {
    local host="$1"

    if [[ -z "$host" ]]; then
        log_error "Usage: cluster_setup.sh setup HOST"
        exit 1
    fi

    log_info "Setting up worker: ${host}"

    # Test SSH
    if ! timeout 5 ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "echo ok" &>/dev/null; then
        log_error "Cannot SSH to ${host}. Check connectivity and SSH keys."
        exit 1
    fi

    # Check if project exists
    log_info "Checking project directory..."
    if ! ssh "$host" "test -d ${REMOTE_PROJECT_DIR}/.git"; then
        log_info "Cloning project on ${host}..."
        ssh "$host" "mkdir -p ~/Development && cd ~/Development && git clone https://github.com/you/RingRift.git"
    else
        log_info "Updating project on ${host}..."
        ssh "$host" "cd ${REMOTE_PROJECT_DIR} && git pull"
    fi

    # Check Python version
    log_info "Checking Python version..."
    local py_version
    py_version=$(ssh "$host" "python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1")
    local py_major py_minor
    py_major=$(echo "$py_version" | cut -d. -f1)
    py_minor=$(echo "$py_version" | cut -d. -f2)

    local python_cmd="python3"
    if [[ "$py_major" -lt 3 ]] || { [[ "$py_major" -eq 3 ]] && [[ "$py_minor" -lt 10 ]]; }; then
        log_warn "Default Python ${py_version} is too old (need 3.10+)"
        # Look for newer Python
        for ver in 3.13 3.12 3.11 3.10; do
            if ssh "$host" "which python${ver} &>/dev/null"; then
                python_cmd="python${ver}"
                log_info "Found ${python_cmd}"
                break
            fi
            local alt_path="/usr/local/bin/python${ver}"
            if ssh "$host" "test -x ${alt_path}"; then
                python_cmd="${alt_path}"
                log_info "Found ${python_cmd}"
                break
            fi
        done
    fi

    # Set up venv
    log_info "Setting up Python venv..."
    ssh "$host" "cd ${REMOTE_PROJECT_DIR}/ai-service && \
        rm -rf venv && \
        ${python_cmd} -m venv venv && \
        source venv/bin/activate && \
        pip install --upgrade pip"

    # Install requirements
    log_info "Installing Python dependencies (this may take a few minutes)..."
    ssh "$host" "cd ${REMOTE_PROJECT_DIR}/ai-service && \
        source venv/bin/activate && \
        pip install -r requirements.txt"

    log_ok "Worker ${host} setup complete!"
    echo
    echo "To start the worker service:"
    echo "  ./scripts/cluster_setup.sh start ${host}"
}

# Set up all workers
cmd_setup_all() {
    log_info "Setting up all workers from ${WORKERS_FILE}..."

    if [[ ! -f "${WORKERS_FILE}" ]]; then
        log_error "Workers file not found: ${WORKERS_FILE}"
        exit 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -z "$line" ]] && continue

        cmd_setup "$line"
        echo
    done < "${WORKERS_FILE}"

    log_ok "All workers set up!"
}

# Start worker service on a host
cmd_start() {
    local host="$1"

    if [[ -z "$host" ]]; then
        log_error "Usage: cluster_setup.sh start HOST"
        exit 1
    fi

    log_info "Starting worker on ${host}..."

    ssh "$host" "cd ${REMOTE_PROJECT_DIR}/ai-service && \
        source venv/bin/activate && \
        pkill -f 'cluster_worker.py' || true && \
        PYTHONPATH=. nohup python scripts/cluster_worker.py --register-bonjour > /tmp/cluster_worker.log 2>&1 &"

    sleep 2

    # Check if running
    if ssh "$host" "pgrep -f 'cluster_worker.py'" &>/dev/null; then
        log_ok "Worker started on ${host}"
        local worker_log
        worker_log=$(ssh "$host" "cat /tmp/cluster_worker.log | head -5")
        echo "$worker_log"
    else
        log_error "Worker failed to start on ${host}"
        ssh "$host" "cat /tmp/cluster_worker.log"
    fi
}

# Check status of all workers
cmd_status() {
    log_info "Checking status of all workers..."
    echo

    if [[ ! -f "${WORKERS_FILE}" ]]; then
        log_error "Workers file not found: ${WORKERS_FILE}"
        exit 1
    fi

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        line="${line%%#*}"
        line="${line//[[:space:]]/}"
        [[ -z "$line" ]] && continue

        local host="$line"

        # Test HTTP health endpoint
        if curl -s --connect-timeout 2 "http://${host}:8765/health" &>/dev/null; then
            local health
            health=$(curl -s "http://${host}:8765/health")
            local tasks
            tasks=$(echo "$health" | sed -n 's/.*"tasks_completed":\s*\([0-9]*\).*/\1/p' || echo "?")
            log_ok "${host} - Worker running (tasks: ${tasks})"
        elif timeout 3 ssh -o BatchMode=yes "$host" "pgrep -f cluster_worker.py" &>/dev/null; then
            log_warn "${host} - Worker process running but HTTP not responding"
        elif timeout 3 nc -z "$host" 22 2>/dev/null; then
            log_warn "${host} - SSH available but worker not running"
        else
            log_error "${host} - Not reachable"
        fi
    done < "${WORKERS_FILE}"
}

# Show help
cmd_help() {
    cat << 'EOF'
RingRift Cluster Setup

Commands:
  discover    Scan local network for potential workers
  test        Test SSH connectivity to known workers
  setup HOST  Set up a specific worker (clone repo, install deps)
  setup-all   Set up all workers in cluster_workers.txt
  start HOST  Start worker service on HOST
  status      Check status of all workers

Worker File:
  Workers are listed in: scripts/cluster_workers.txt
  Format: one IP/hostname per line, # for comments

Example Workflow:
  1. ./scripts/cluster_setup.sh discover     # Find Macs on network
  2. Edit scripts/cluster_workers.txt        # Add worker IPs
  3. ./scripts/cluster_setup.sh test         # Verify SSH access
  4. ./scripts/cluster_setup.sh setup-all    # Install dependencies
  5. ./scripts/cluster_setup.sh start <host> # Start workers
  6. ./scripts/run_distributed_selfplay_matrix.sh  # Run distributed jobs

Enabling SSH on Mac:
  System Preferences → Sharing → Remote Login → ON

Copying SSH Keys:
  ssh-copy-id <host>

EOF
}

# Main command dispatcher
case "${1:-help}" in
    discover)   cmd_discover ;;
    test)       cmd_test ;;
    setup)      cmd_setup "${2:-}" ;;
    setup-all)  cmd_setup_all ;;
    start)      cmd_start "${2:-}" ;;
    status)     cmd_status ;;
    help|--help|-h) cmd_help ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac

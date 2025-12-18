#!/bin/bash
# Production-grade cluster health check script

# Lambda nodes: name ip
NODES="l-2xh100:100.97.104.89
l-a10:100.91.25.13
l-gh200e:100.88.176.74
l-gh200f:100.104.165.116
l-gh200g:100.104.126.58
l-gh200b:100.83.234.82"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local msg=$2
    case $status in
        OK)   echo -e "${GREEN}[OK]${NC} $msg" ;;
        WARN) echo -e "${YELLOW}[WARN]${NC} $msg" ;;
        FAIL) echo -e "${RED}[FAIL]${NC} $msg" ;;
        INFO) echo -e "${BLUE}[INFO]${NC} $msg" ;;
    esac
}

check_node() {
    local name=$1
    local ip=$2

    if ! timeout 10 ssh -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$ip "echo ok" &>/dev/null; then
        print_status FAIL "$name ($ip): Unreachable"
        return 1
    fi

    local status=$(ssh -o ConnectTimeout=5 ubuntu@$ip 'bash -s' 2>/dev/null << 'REMOTE'
GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | tr -d ' ')
DISK_PCT=$(df -h /home | tail -1 | awk '{print $5}' | tr -d '%')
TRAINING=$(ps aux | grep -E 'training_loop|selfplay|gauntlet' | grep -v grep | wc -l)
PYTHON_PROCS=$(ps aux | grep python | grep -v grep | wc -l)
echo "GPU:${GPU_UTIL}% MEM:${GPU_MEM_USED}/${GPU_MEM_TOTAL}MB DISK:${DISK_PCT}% Procs:$PYTHON_PROCS Train:$TRAINING"
REMOTE
    )

    [ -z "$status" ] && status="Status check failed"
    print_status OK "$name: $status"
}

main() {
    echo "=========================================="
    echo "  RingRift Cluster Health Check"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    print_status INFO "Checking Tailscale..."
    if tailscale status &>/dev/null; then
        print_status OK "Tailscale connected"
    else
        print_status FAIL "Tailscale not running"
    fi
    echo ""

    print_status INFO "Checking Lambda nodes..."
    local failed=0
    local total=0
    while IFS=: read -r name ip; do
        total=$((total + 1))
        check_node "$name" "$ip" || failed=$((failed + 1))
    done <<< "$NODES"
    echo ""

    local healthy=$((total - failed))
    if [ $failed -eq 0 ]; then
        print_status OK "All $total nodes healthy"
    else
        print_status WARN "$healthy/$total nodes healthy"
    fi

    echo ""
    print_status INFO "Checking Vast.ai..."
    if command -v vastai &>/dev/null; then
        VAST_COUNT=$(vastai show instances 2>/dev/null | grep -c running || echo 0)
        print_status OK "$VAST_COUNT instances running"
    else
        print_status WARN "Vast CLI not available"
    fi
    echo "=========================================="
}

main "$@"

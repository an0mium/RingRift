#!/bin/bash
# Production-grade cluster health check script

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

    local status=$(ssh -o ConnectTimeout=5 ubuntu@$ip "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1" 2>/dev/null)
    local disk=$(ssh -o ConnectTimeout=5 ubuntu@$ip "df -h /home | tail -1 | awk '{print \$5}'" 2>/dev/null)
    local procs=$(ssh -o ConnectTimeout=5 ubuntu@$ip "ps aux | grep python | grep -v grep | wc -l" 2>/dev/null)
    local training=$(ssh -o ConnectTimeout=5 ubuntu@$ip "ps aux | grep -E 'training|selfplay|gauntlet' | grep -v grep | wc -l" 2>/dev/null)

    GPU_UTIL=$(echo "$status" | cut -d',' -f1 | tr -d ' ')
    GPU_MEM=$(echo "$status" | cut -d',' -f2 | tr -d ' ')
    GPU_TOTAL=$(echo "$status" | cut -d',' -f3 | tr -d ' ')

    print_status OK "$name: GPU:${GPU_UTIL:-?}% MEM:${GPU_MEM:-?}/${GPU_TOTAL:-?}MB DISK:${disk:-?} Procs:${procs:-?} Train:${training:-?}"
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

    # Load node IPs from config file (gitignored)
    local config_file="$(dirname "$0")/../config/cluster_nodes.env"
    if [ -f "$config_file" ]; then
        source "$config_file"

        # Check H100 node
        if [ -n "$H100_IP" ]; then
            check_node "l-2xh100" "$H100_IP" || failed=$((failed + 1))
            total=$((total + 1))
        fi

        # Check A10 node
        if [ -n "$A10_IP" ]; then
            check_node "l-a10" "$A10_IP" || failed=$((failed + 1))
            total=$((total + 1))
        fi

        # Check GH200 nodes (array from config)
        if [ -n "${GH200_NODES+x}" ]; then
            for node in "${GH200_NODES[@]}"; do
                local ip="${node%%:*}"
                local name="${node##*:}"
                check_node "$name" "$ip" || failed=$((failed + 1))
                total=$((total + 1))
            done
        fi
    else
        print_status WARN "Config file not found: $config_file"
        print_status INFO "Copy config/cluster_nodes.env.example to cluster_nodes.env"
    fi

    echo ""
    local healthy=$((total - failed))
    if [ $total -eq 0 ]; then
        print_status WARN "No nodes configured"
    elif [ $failed -eq 0 ]; then
        print_status OK "All $total nodes healthy"
    else
        print_status WARN "$healthy/$total nodes healthy"
    fi

    echo ""
    print_status INFO "Checking Vast.ai..."
    if command -v vastai &>/dev/null; then
        VAST_COUNT=$(vastai show instances 2>/dev/null | grep -c running || echo 0)
        print_status OK "$VAST_COUNT instances running"
    fi
    echo "=========================================="
}

main "$@"

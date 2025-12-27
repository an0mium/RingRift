#!/bin/bash
# P2P Daemon Status Investigation Script
# Checks P2P daemon status across all GPU nodes in the cluster
# DO NOT auto-fix issues - just investigate and report

set -u

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Output file
REPORT_FILE="/tmp/p2p_status_report_$(date +%Y%m%d_%H%M%S).txt"

echo "P2P Daemon Status Investigation Report" | tee "$REPORT_FILE"
echo "Generated: $(date)" | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Function to check a single node
check_node() {
    local node_name=$1
    local ssh_cmd=$2
    local ringrift_path=$3

    echo -e "${YELLOW}Checking: $node_name${NC}"
    echo "----------------------------------------" | tee -a "$REPORT_FILE"
    echo "Node: $node_name" | tee -a "$REPORT_FILE"

    # Check if SSH connection works
    if ! timeout 10 $ssh_cmd "echo 'SSH OK'" >/dev/null 2>&1; then
        echo -e "${RED}Status: SSH FAILED${NC}" | tee -a "$REPORT_FILE"
        echo "" | tee -a "$REPORT_FILE"
        return 1
    fi

    # Check if P2P daemon is running
    p2p_pids=$($ssh_cmd "pgrep -f 'p2p_daemon|p2p_orchestrator' 2>/dev/null || echo ''")

    if [ -z "$p2p_pids" ]; then
        echo -e "${RED}P2P Running: NO${NC}" | tee -a "$REPORT_FILE"
    else
        echo -e "${GREEN}P2P Running: YES (PIDs: $p2p_pids)${NC}" | tee -a "$REPORT_FILE"
    fi

    # Check port 8770
    port_status=$($ssh_cmd "netstat -tlnp 2>/dev/null | grep :8770 || ss -tlnp 2>/dev/null | grep :8770 || echo 'NOT_LISTENING'")

    if echo "$port_status" | grep -q "NOT_LISTENING"; then
        echo -e "${RED}Port 8770: NOT LISTENING${NC}" | tee -a "$REPORT_FILE"
    else
        echo -e "${GREEN}Port 8770: LISTENING${NC}" | tee -a "$REPORT_FILE"
        echo "Port Details: $port_status" | tee -a "$REPORT_FILE"
    fi

    # Check for P2P logs
    log_path_options="$ringrift_path/logs/p2p*.log /workspace/ringrift/ai-service/logs/p2p*.log ~/ringrift/ai-service/logs/p2p*.log"

    log_found=""
    for log_path in $log_path_options; do
        if $ssh_cmd "test -f $log_path 2>/dev/null" 2>/dev/null; then
            log_found=$log_path
            break
        fi
    done

    if [ -n "$log_found" ]; then
        echo "Log Found: $log_found" | tee -a "$REPORT_FILE"

        # Get last 20 lines of log
        echo "Last 20 lines:" | tee -a "$REPORT_FILE"
        $ssh_cmd "tail -20 $log_found 2>/dev/null" | tee -a "$REPORT_FILE"

        # Check for errors
        error_count=$($ssh_cmd "grep -i 'error\|exception\|crash\|failed' $log_found 2>/dev/null | wc -l")
        if [ "$error_count" -gt 0 ]; then
            echo -e "${RED}Errors in log: $error_count${NC}" | tee -a "$REPORT_FILE"
            echo "Recent errors:" | tee -a "$REPORT_FILE"
            $ssh_cmd "grep -i 'error\|exception\|crash\|failed' $log_found 2>/dev/null | tail -5" | tee -a "$REPORT_FILE"
        fi
    else
        echo -e "${YELLOW}Log: NOT FOUND${NC}" | tee -a "$REPORT_FILE"
    fi

    # Check systemd/supervisor status if applicable
    systemd_status=$($ssh_cmd "systemctl status p2p-daemon 2>/dev/null || systemctl status p2p_orchestrator 2>/dev/null || echo 'NO_SYSTEMD'")

    if ! echo "$systemd_status" | grep -q "NO_SYSTEMD"; then
        echo "Systemd Status:" | tee -a "$REPORT_FILE"
        echo "$systemd_status" | tee -a "$REPORT_FILE"
    fi

    # Check if P2P was ever started (look for old logs or processes)
    if [ -z "$p2p_pids" ]; then
        old_logs=$($ssh_cmd "find $ringrift_path/logs /workspace/ringrift/ai-service/logs ~/ringrift/ai-service/logs -name 'p2p*.log*' 2>/dev/null | head -5 || echo ''")

        if [ -n "$old_logs" ]; then
            echo "Old P2P logs found (daemon was started before):" | tee -a "$REPORT_FILE"
            echo "$old_logs" | tee -a "$REPORT_FILE"
        else
            echo -e "${YELLOW}No evidence of P2P ever running${NC}" | tee -a "$REPORT_FILE"
        fi
    fi

    echo "" | tee -a "$REPORT_FILE"
}

# VAST.AI NODES (top tier first)
echo "=== VAST.AI NODES ===" | tee -a "$REPORT_FILE"

# Multi-GPU nodes
check_node "vast-29129529" \
    "ssh -p 19528 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh6.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29118471" \
    "ssh -p 38470 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh8.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29128352" \
    "ssh -p 18352 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh9.vast.ai" \
    "/workspace/ringrift/ai-service"

# High-end single GPU
check_node "vast-28925166" \
    "ssh -p 15166 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh1.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29128356" \
    "ssh -p 18356 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh7.vast.ai" \
    "/workspace/ringrift/ai-service"

check_node "vast-28918742" \
    "ssh -p 38742 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh8.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29031159" \
    "ssh -p 31158 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh5.vast.ai" \
    "~/ringrift/ai-service"

# Mid-range
check_node "vast-29126088" \
    "ssh -p 16088 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh5.vast.ai" \
    "/workspace/ringrift/ai-service"

check_node "vast-29031161" \
    "ssh -p 31160 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh2.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-28890015" \
    "ssh -p 10014 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh9.vast.ai" \
    "~/ringrift/ai-service"

# Entry-level
check_node "vast-28889766" \
    "ssh -p 19766 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh3.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29046315" \
    "ssh -p 16314 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh2.vast.ai" \
    "~/ringrift/ai-service"

# New RTX 5090 nodes
check_node "vast-29118472" \
    "ssh -p 38472 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh9.vast.ai" \
    "~/ringrift/ai-service"

check_node "vast-29129151" \
    "ssh -p 19150 -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@ssh4.vast.ai" \
    "~/ringrift/ai-service"

# RUNPOD NODES
echo "=== RUNPOD NODES ===" | tee -a "$REPORT_FILE"

check_node "runpod-h100" \
    "ssh -p 30755 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@102.210.171.65" \
    "/workspace/ringrift/ai-service"

check_node "runpod-a100-1" \
    "ssh -p 33085 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@38.128.233.145" \
    "/workspace/ringrift/ai-service"

check_node "runpod-a100-2" \
    "ssh -p 11681 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@104.255.9.187" \
    "/workspace/ringrift/ai-service"

check_node "runpod-a100-storage-1" \
    "ssh -p 30015 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@213.173.105.7" \
    "/workspace/ringrift/ai-service"

check_node "runpod-l40s-2" \
    "ssh -p 1182 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@193.183.22.62" \
    "/workspace/ringrift/ai-service"

check_node "runpod-3090ti-1" \
    "ssh -p 29473 -i ~/.runpod/ssh/RunPod-Key-Go -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@174.94.157.109" \
    "~/ringrift/ai-service"

# VULTR NODES
echo "=== VULTR NODES ===" | tee -a "$REPORT_FILE"

check_node "vultr-a100-20gb" \
    "ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@208.167.249.164" \
    "/root/ringrift/ai-service"

check_node "vultr-a100-20gb-2" \
    "ssh -i ~/.ssh/id_ed25519 -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@140.82.15.69" \
    "/root/ringrift/ai-service"

# NEBIUS NODES
echo "=== NEBIUS NODES ===" | tee -a "$REPORT_FILE"

check_node "nebius-backbone-1" \
    "ssh -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@89.169.112.47" \
    "~/ringrift/ai-service"

check_node "nebius-h100-1" \
    "ssh -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@89.169.111.139" \
    "~/ringrift/ai-service"

check_node "nebius-h100-3" \
    "ssh -i ~/.ssh/id_cluster -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@89.169.110.128" \
    "~/ringrift/ai-service"

echo "" | tee -a "$REPORT_FILE"
echo "========================================" | tee -a "$REPORT_FILE"
echo "Investigation complete!" | tee -a "$REPORT_FILE"
echo "Full report saved to: $REPORT_FILE" | tee -a "$REPORT_FILE"

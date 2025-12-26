#!/bin/bash
# Verify all cluster nodes are on the latest commit
# Generated: 2025-12-25

set -u

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TIMEOUT=15

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}RingRift Cluster Version Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get local commit
LOCAL_COMMIT=$(git rev-parse --short HEAD)
echo -e "${BLUE}Local commit: ${LOCAL_COMMIT}${NC}"
echo ""

check_node() {
    local name=$1
    local ssh_host=$2
    local ssh_port=$3
    local ssh_key=$4
    local ssh_user=$5
    local ringrift_path=$6

    local ssh_cmd="ssh -o ConnectTimeout=${TIMEOUT} -o StrictHostKeyChecking=no"
    ssh_cmd="${ssh_cmd} -p ${ssh_port} -i ${ssh_key} ${ssh_user}@${ssh_host}"

    local git_cmd="cd ${ringrift_path} && git rev-parse --short HEAD 2>&1"

    if remote_commit=$(timeout ${TIMEOUT} ${ssh_cmd} "${git_cmd}" 2>&1 | tail -1); then
        if [[ ${remote_commit} =~ ^[0-9a-f]+$ ]]; then
            if [ "${remote_commit}" = "${LOCAL_COMMIT}" ]; then
                echo -e "${GREEN}✓ ${name}: ${remote_commit} (up to date)${NC}"
            else
                echo -e "${YELLOW}△ ${name}: ${remote_commit} (differs from local: ${LOCAL_COMMIT})${NC}"
            fi
        else
            echo -e "${YELLOW}? ${name}: ${remote_commit}${NC}"
        fi
    else
        echo -e "${YELLOW}✗ ${name}: Failed to connect${NC}"
    fi
}

echo -e "${BLUE}=== Checking Vast.ai Nodes ===${NC}"
check_node "vast-29129529" "ssh6.vast.ai" "19528" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29118471" "ssh8.vast.ai" "38470" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29128352" "ssh9.vast.ai" "18352" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
check_node "vast-28925166" "ssh1.vast.ai" "15166" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29128356" "ssh7.vast.ai" "18356" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
check_node "vast-28918742" "ssh8.vast.ai" "38742" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29031159" "ssh5.vast.ai" "31158" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29126088" "ssh5.vast.ai" "16088" "$HOME/.ssh/id_cluster" "root" "/workspace/ringrift/ai-service"
check_node "vast-29031161" "ssh2.vast.ai" "31160" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-28890015" "ssh9.vast.ai" "10014" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-28889766" "ssh3.vast.ai" "19766" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"
check_node "vast-29046315" "ssh2.vast.ai" "16314" "$HOME/.ssh/id_cluster" "root" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Checking RunPod Nodes ===${NC}"
check_node "runpod-h100" "102.210.171.65" "30690" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
check_node "runpod-a100-1" "38.128.233.145" "33085" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
check_node "runpod-a100-2" "104.255.9.187" "11681" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
check_node "runpod-l40s-2" "193.183.22.62" "1630" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "/workspace/ringrift/ai-service"
check_node "runpod-3090ti-1" "174.94.157.109" "29473" "$HOME/.runpod/ssh/RunPod-Key-Go" "root" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Checking Vultr Nodes ===${NC}"
check_node "vultr-a100-20gb" "208.167.249.164" "22" "$HOME/.ssh/id_ed25519" "root" "/root/ringrift/ai-service"
check_node "vultr-a100-20gb-2" "140.82.15.69" "22" "$HOME/.ssh/id_ed25519" "root" "/root/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Checking Nebius Nodes ===${NC}"
check_node "nebius-backbone-1" "89.169.112.47" "22" "$HOME/.ssh/id_cluster" "ubuntu" "~/ringrift/ai-service"
check_node "nebius-l40s-2" "89.169.108.182" "22" "$HOME/.ssh/id_cluster" "ubuntu" "~/ringrift/ai-service"

echo ""
echo -e "${BLUE}=== Checking Hetzner Nodes ===${NC}"
check_node "hetzner-cpu1" "46.62.147.150" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"
check_node "hetzner-cpu2" "135.181.39.239" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"
check_node "hetzner-cpu3" "46.62.217.168" "22" "$HOME/.ssh/id_cluster" "root" "/root/ringrift/ai-service"

echo ""
echo -e "${BLUE}========================================${NC}"

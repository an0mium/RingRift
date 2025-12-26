#!/bin/bash
# 10-Hour Cluster Automation Script
# Purpose: Maximize GPU utilization for 2000+ Elo AI training
# Created: $(date)

set -e
cd /Users/armand/Development/RingRift/ai-service

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting 10-hour cluster automation..."

# Priority configs needing more data (from earlier analysis)
# hex8_2p: 207 games, hexagonal_*: 11-77 games, square19_*: 27-154 games
PRIORITY_CONFIGS=(
    "hexagonal_2p:hexagonal:2"
    "hexagonal_3p:hexagonal:3"  
    "hexagonal_4p:hexagonal:4"
    "hex8_2p:hex8:2"
    "square19_2p:square19:2"
    "square19_3p:square19:3"
    "square19_4p:square19:4"
)

# Idle nodes from P2P status (with SSH access info)
declare -A NODES
NODES["runpod-a100-1"]="root@38.128.233.145:33085"
NODES["runpod-a100-2"]="root@104.255.9.187:11681"
NODES["runpod-h100"]="root@102.210.171.65:30178"
NODES["runpod-l40s-2"]="root@193.183.22.62:1630"
NODES["nebius-h100-1"]="ubuntu@89.169.111.139"
NODES["nebius-l40s-2"]="ubuntu@89.169.112.47"
NODES["vultr-a100-20gb"]="root@208.167.249.164"

# Function to spawn selfplay on a node
spawn_selfplay() {
    local node=$1
    local config=$2
    local board=$(echo $config | cut -d: -f2)
    local players=$(echo $config | cut -d: -f3)
    local addr=${NODES[$node]}
    
    if [ -z "$addr" ]; then
        log "Unknown node: $node"
        return 1
    fi
    
    local host=$(echo $addr | cut -d@ -f2 | cut -d: -f1)
    local port=$(echo $addr | cut -d: -f2)
    local user=$(echo $addr | cut -d@ -f1)
    
    # Default port is 22 if not specified
    [ -z "$port" ] && port=22
    
    local ssh_cmd="ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no"
    if [ "$port" != "22" ]; then
        ssh_cmd="$ssh_cmd -p $port"
    fi
    
    log "Spawning ${board}_${players}p selfplay on $node..."
    
    # Check if selfplay already running for this config
    local running=$($ssh_cmd $user@$host "pgrep -f 'selfplay.*--board $board.*--num-players $players' 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    
    if [ "$running" -gt "0" ]; then
        log "  Selfplay for ${board}_${players}p already running on $node"
        return 0
    fi
    
    # Spawn selfplay
    $ssh_cmd $user@$host "cd ~/ringrift/ai-service 2>/dev/null || cd /workspace/ringrift/ai-service && \
        source venv/bin/activate 2>/dev/null || true && \
        nohup python scripts/selfplay.py \
            --board $board \
            --num-players $players \
            --engine gumbel \
            --num-games 1000 \
            --output-dir data/games \
            > /tmp/selfplay_${board}_${players}p.log 2>&1 &" 2>/dev/null
    
    log "  Started ${board}_${players}p on $node"
}

# Function to check node status
check_node() {
    local node=$1
    local addr=${NODES[$node]}
    local host=$(echo $addr | cut -d@ -f2 | cut -d: -f1)
    local port=$(echo $addr | cut -d: -f2)
    local user=$(echo $addr | cut -d@ -f1)
    
    [ -z "$port" ] && port=22
    
    local ssh_cmd="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
    if [ "$port" != "22" ]; then
        ssh_cmd="$ssh_cmd -p $port"
    fi
    
    $ssh_cmd $user@$host "echo ok" 2>/dev/null && echo "UP" || echo "DOWN"
}

# Main automation loop
log "Phase 1: Checking node availability..."
for node in "${!NODES[@]}"; do
    status=$(check_node $node)
    log "  $node: $status"
done

log "Phase 2: Spawning priority selfplay jobs..."
node_idx=0
nodes_arr=(${!NODES[@]})
num_nodes=${#nodes_arr[@]}

for config in "${PRIORITY_CONFIGS[@]}"; do
    node=${nodes_arr[$node_idx]}
    spawn_selfplay $node $config || true
    node_idx=$(( (node_idx + 1) % num_nodes ))
done

log "Phase 3: Starting data sync daemon..."
# Enable sync in a background process
PYTHONPATH=. nohup python scripts/unified_data_sync.py --watchdog > logs/unified_data_sync.log 2>&1 &
log "  Data sync daemon started (PID: $!)"

log "Phase 4: Starting master loop with all daemons..."
# Kill existing master loop if running
pkill -f "master_loop.py" 2>/dev/null || true
sleep 2

# Start master loop with full automation
PYTHONPATH=. nohup python scripts/master_loop.py > logs/master_loop_10hr.log 2>&1 &
log "  Master loop started (PID: $!)"

log "Phase 5: Setting up periodic status checks..."
# Create monitoring script
cat << 'MONITOR' > /tmp/cluster_monitor.sh
#!/bin/bash
cd /Users/armand/Development/RingRift/ai-service
while true; do
    echo "=== Cluster Status $(date) ===" >> logs/cluster_status.log
    curl -s http://localhost:8770/status | python3 -c '
import sys,json
try:
    d = json.load(sys.stdin)
    peers = d.get("peers", {})
    total_sp = sum(p.get("selfplay_jobs", 0) for p in peers.values())
    total_tr = sum(p.get("training_jobs", 0) for p in peers.values())
    idle_gpu = sum(1 for p in peers.values() if p.get("has_gpu") and p.get("gpu_percent", 100) < 10)
    print(f"  Selfplay jobs: {total_sp}")
    print(f"  Training jobs: {total_tr}")  
    print(f"  Idle GPU nodes: {idle_gpu}")
except:
    print("  P2P status unavailable")
' >> logs/cluster_status.log
    sleep 300  # Check every 5 minutes
done
MONITOR
chmod +x /tmp/cluster_monitor.sh
nohup /tmp/cluster_monitor.sh > /dev/null 2>&1 &
log "  Status monitor started"

log ""
log "========================================="
log "10-HOUR AUTOMATION STARTED"
log "========================================="
log ""
log "Monitor logs:"
log "  Master loop:  tail -f logs/master_loop_10hr.log"
log "  Data sync:    tail -f logs/unified_data_sync.log" 
log "  Cluster:      tail -f logs/cluster_status.log"
log ""
log "To stop: pkill -f master_loop.py && pkill -f unified_data_sync"
log ""


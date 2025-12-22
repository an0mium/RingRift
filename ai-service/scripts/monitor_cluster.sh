#!/bin/bash
# Cluster monitoring script for Lambda GH200 nodes
# Checks health and restarts failed Gumbel MCTS selfplay processes

NODES="100.65.88.62 100.79.109.120 100.117.177.83 100.99.27.56 100.97.98.26 100.66.65.33 100.104.126.58 100.83.234.82"
LOG_FILE="/tmp/cluster_monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_and_restart_gumbel() {
    local ip=$1
    local board=$2

    # Check if Gumbel selfplay is running
    local proc_count=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | wc -l" 2>/dev/null)
    local game_count=$(ssh -o ConnectTimeout=10 ubuntu@${ip} "wc -l ~/ringrift/ai-service/data/selfplay/gumbel_*.jsonl 2>/dev/null | tail -1 | awk '{print \$1}'" 2>/dev/null)

    echo "$ip ($board): procs=$proc_count games=${game_count:-0}"

    # Restart if no process
    if [ "$proc_count" = "0" ] || [ -z "$proc_count" ]; then
        log "Restarting Gumbel MCTS selfplay on $ip with board=$board"
        ssh -o ConnectTimeout=10 ubuntu@${ip} "cd ~/ringrift/ai-service && source venv/bin/activate && PYTHONPATH=. nohup python scripts/generate_gumbel_selfplay.py --board ${board} --num-players 2 --num-games 5000 --simulation-budget 200 --output data/selfplay/gumbel_${board}_2p.jsonl --allow-fresh-weights > /tmp/gumbel_selfplay.log 2>&1 &" 2>/dev/null
    fi
}

main() {
    log "=== Cluster Health Check ==="

    # Square8 nodes
    for ip in 100.65.88.62 100.79.109.120; do
        check_and_restart_gumbel "$ip" "square8"
    done

    # Square19 nodes
    for ip in 100.117.177.83 100.99.27.56; do
        check_and_restart_gumbel "$ip" "square19"
    done

    # Hexagonal nodes
    for ip in 100.97.98.26 100.66.65.33; do
        check_and_restart_gumbel "$ip" "hexagonal"
    done

    # Hex8 nodes
    for ip in 100.104.126.58 100.83.234.82; do
        check_and_restart_gumbel "$ip" "hex8"
    done

    # Summary
    log "Check complete"
}

# Run once or in loop
if [ "$1" = "--loop" ]; then
    while true; do
        main
        sleep 300  # Check every 5 minutes
    done
else
    main
fi

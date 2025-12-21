#!/bin/bash
# =============================================================================
# Full Cluster Pipeline Deployment
# =============================================================================
# Deploys canonical selfplay for all 12 board/player configs with parity
# validation, plus hexagonal ANM diagnosis.
#
# Usage:
#   ./scripts/deploy_full_cluster_pipeline.sh [command]
#
# Commands:
#   update      - Update code on all cluster nodes
#   canonical   - Launch canonical selfplay on all 12 configs
#   diagnose    - Run hexagonal ANM diagnosis
#   monitor     - Check progress across cluster
#   status      - Show job status on all nodes
#   all         - Run update + canonical + diagnose (default)
#
# December 2025 - RingRift AI Training Pipeline
# =============================================================================

set -euo pipefail

# =============================================================================
# Cluster Node Configuration
# =============================================================================

# GH200 nodes (via Tailscale IPs) - Most powerful, used for hex boards
GH200_HOSTS=(
    "100.123.183.70"   # lambda-gh200-a - hexagonal_2p
    "100.104.34.73"    # lambda-gh200-b - hexagonal_3p
    "100.88.35.19"     # lambda-gh200-c - hexagonal_4p
    "100.75.84.47"     # lambda-gh200-d - square19_2p
    "100.88.176.74"    # lambda-gh200-e - square19_3p
    "100.104.165.116"  # lambda-gh200-f - square19_4p
    "100.104.126.58"   # lambda-gh200-g - hex8_2p
    "100.65.88.62"     # lambda-gh200-h - hex8_3p
    "100.99.27.56"     # lambda-gh200-i - hex8_4p
    "100.96.142.42"    # lambda-gh200-k - square8_2p
    "100.76.145.60"    # lambda-gh200-l - square8_3p
)

# Other Lambda instances
LAMBDA_HOSTS=(
    "100.78.101.123"   # lambda-h100 - square8_4p
)

# Config assignments (board:players -> node index)
declare -A CONFIG_NODES=(
    ["hexagonal:2"]="100.123.183.70"
    ["hexagonal:3"]="100.104.34.73"
    ["hexagonal:4"]="100.88.35.19"
    ["square19:2"]="100.75.84.47"
    ["square19:3"]="100.88.176.74"
    ["square19:4"]="100.104.165.116"
    ["hex8:2"]="100.104.126.58"
    ["hex8:3"]="100.65.88.62"
    ["hex8:4"]="100.99.27.56"
    ["square8:2"]="100.96.142.42"
    ["square8:3"]="100.76.145.60"
    ["square8:4"]="100.78.101.123"
)

ALL_HOSTS=("${GH200_HOSTS[@]}" "${LAMBDA_HOSTS[@]}")
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes"
SSH_KEY="${SSH_KEY:-~/.ssh/id_cluster}"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ssh_node() {
    local host="$1"
    shift
    ssh $SSH_OPTS -i "$SSH_KEY" "ubuntu@$host" "$@"
}

# =============================================================================
# Command: update
# =============================================================================
cmd_update() {
    log "Updating code on all cluster nodes..."

    for host in "${ALL_HOSTS[@]}"; do
        (
            log "  Updating $host..."
            ssh_node "$host" 'cd ~/ringrift/ai-service && git fetch origin && git reset --hard origin/main && git log -1 --oneline' 2>&1 | sed "s/^/    [$host] /"
        ) &
    done
    wait

    log "Code updated on all nodes."
}

# =============================================================================
# Command: canonical - Launch canonical selfplay for all 12 configs
# =============================================================================
cmd_canonical() {
    log "Launching canonical selfplay for all 12 configs..."

    for config in "${!CONFIG_NODES[@]}"; do
        host="${CONFIG_NODES[$config]}"
        board="${config%%:*}"
        players="${config##*:}"

        log "  Starting $board ${players}p on $host..."

        ssh_node "$host" "cd ~/ringrift/ai-service && source venv/bin/activate && \
            mkdir -p data/games logs && \
            nohup PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
                --board $board \
                --num-players $players \
                --num-games 100 \
                --db data/games/canonical_${board}_${players}p.db \
                --summary logs/canonical_${board}_${players}p_summary.json \
                > logs/canonical_${board}_${players}p.log 2>&1 &" &
    done
    wait

    log "Canonical selfplay launched on all nodes."
    log ""
    log "Monitor with: $0 monitor"
}

# =============================================================================
# Command: diagnose - Run hexagonal ANM parity diagnosis
# =============================================================================
cmd_diagnose() {
    log "Running hexagonal ANM diagnosis on lambda-gh200-a..."

    local host="100.123.183.70"

    # Run the diagnosis script on the cluster
    ssh_node "$host" 'cd ~/ringrift/ai-service && source venv/bin/activate && \
        mkdir -p logs/anm_diagnosis && \

        # Step 1: Generate fresh hexagonal games with parity validation
        echo "[diagnose] Step 1: Generating fresh hexagonal games..."
        PYTHONPATH=. python scripts/generate_canonical_selfplay.py \
            --board hexagonal \
            --num-players 2 \
            --num-games 20 \
            --db data/games/hex_anm_diagnosis.db \
            --summary logs/anm_diagnosis/parity_gate.json \
            2>&1 | tee logs/anm_diagnosis/generation.log

        # Step 2: Analyze any ANM divergences
        echo "[diagnose] Step 2: Analyzing ANM divergences..."
        if [ -f logs/anm_diagnosis/parity_gate.json ]; then
            PYTHONPATH=. python scripts/diagnose_anm_divergence.py \
                --parity-gate logs/anm_diagnosis/parity_gate.json \
                --verbose \
                2>&1 | tee logs/anm_diagnosis/analysis.log
        fi

        # Step 3: Summary
        echo "[diagnose] Step 3: Generating summary..."
        echo "=== ANM Diagnosis Complete ===" | tee logs/anm_diagnosis/summary.txt
        sqlite3 data/games/hex_anm_diagnosis.db "SELECT COUNT(*) as games FROM games" 2>/dev/null | tee -a logs/anm_diagnosis/summary.txt
        cat logs/anm_diagnosis/parity_gate.json 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Parity: {d.get(\"canonical_ok\", \"unknown\")}\")" 2>/dev/null | tee -a logs/anm_diagnosis/summary.txt
        '

    log "Hexagonal ANM diagnosis complete."
    log "View results: ssh ubuntu@$host 'cat ~/ringrift/ai-service/logs/anm_diagnosis/summary.txt'"
}

# =============================================================================
# Command: monitor - Check progress across cluster
# =============================================================================
cmd_monitor() {
    log "Monitoring canonical selfplay progress..."
    echo ""

    printf "%-20s %-12s %-10s %-10s %s\n" "HOST" "CONFIG" "GAMES" "STATUS" "LAST_LOG"
    printf "%-20s %-12s %-10s %-10s %s\n" "----" "------" "-----" "------" "--------"

    for config in "${!CONFIG_NODES[@]}"; do
        host="${CONFIG_NODES[$config]}"
        board="${config%%:*}"
        players="${config##*:}"
        db_name="canonical_${board}_${players}p.db"

        result=$(ssh_node "$host" "
            cd ~/ringrift/ai-service 2>/dev/null || exit 1

            # Count games
            games=0
            if [ -f data/games/$db_name ]; then
                games=\$(sqlite3 data/games/$db_name 'SELECT COUNT(*) FROM games' 2>/dev/null || echo 0)
            fi

            # Check if process is running
            running=\$(pgrep -f \"generate_canonical_selfplay.*$board.*$players\" >/dev/null 2>&1 && echo 'running' || echo 'stopped')

            # Last log line
            last_log=''
            if [ -f logs/canonical_${board}_${players}p.log ]; then
                last_log=\$(tail -1 logs/canonical_${board}_${players}p.log 2>/dev/null | cut -c1-40)
            fi

            echo \"\$games|\$running|\$last_log\"
        " 2>/dev/null || echo "0|error|connection failed")

        games=$(echo "$result" | cut -d'|' -f1)
        status=$(echo "$result" | cut -d'|' -f2)
        last_log=$(echo "$result" | cut -d'|' -f3)

        printf "%-20s %-12s %-10s %-10s %s\n" "$host" "${board}_${players}p" "$games" "$status" "$last_log"
    done

    echo ""
    log "Refresh with: $0 monitor"
}

# =============================================================================
# Command: status - Show job status on all nodes
# =============================================================================
cmd_status() {
    log "Checking job status on all nodes..."
    echo ""

    for host in "${ALL_HOSTS[@]}"; do
        echo "=== $host ==="
        ssh_node "$host" '
            cd ~/ringrift/ai-service 2>/dev/null || exit 1

            # Show running selfplay processes
            echo "Running processes:"
            ps aux | grep -E "(generate_canonical|run_diverse)" | grep -v grep | awk "{print \"  PID:\", \$2, \"CPU:\", \$3\"%\", \"MEM:\", \$4\"%\"}" || echo "  None"

            # Show database sizes
            echo "Databases:"
            for db in data/games/canonical_*.db data/games/selfplay_*.db; do
                if [ -f "$db" ]; then
                    games=$(sqlite3 "$db" "SELECT COUNT(*) FROM games" 2>/dev/null || echo 0)
                    size=$(du -h "$db" | cut -f1)
                    echo "  $(basename $db): $games games ($size)"
                fi
            done
        ' 2>/dev/null || echo "  Connection failed"
        echo ""
    done
}

# =============================================================================
# Command: all - Run full pipeline
# =============================================================================
cmd_all() {
    cmd_update
    echo ""
    cmd_canonical
    echo ""
    log "Pipeline deployed. Run '$0 diagnose' after games accumulate for ANM analysis."
}

# =============================================================================
# Main
# =============================================================================

main() {
    local cmd="${1:-all}"

    case "$cmd" in
        update)    cmd_update ;;
        canonical) cmd_canonical ;;
        diagnose)  cmd_diagnose ;;
        monitor)   cmd_monitor ;;
        status)    cmd_status ;;
        all)       cmd_all ;;
        *)
            echo "Usage: $0 [update|canonical|diagnose|monitor|status|all]"
            echo ""
            echo "Commands:"
            echo "  update      - Update code on all cluster nodes"
            echo "  canonical   - Launch canonical selfplay on all 12 configs"
            echo "  diagnose    - Run hexagonal ANM diagnosis"
            echo "  monitor     - Check progress across cluster"
            echo "  status      - Show job status on all nodes"
            echo "  all         - Run update + canonical (default)"
            exit 1
            ;;
    esac
}

main "$@"

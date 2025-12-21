#!/bin/bash
# Monitor canonical selfplay progress across Lambda cluster
# Usage: ./scripts/monitor_canonical_selfplay.sh [--watch]

NODES=(
    "100.97.104.89:lambda-2xh100"
    "100.88.176.74:lambda-gh200-e"
    "100.104.165.116:lambda-gh200-f"
    "100.104.126.58:lambda-gh200-g"
    "100.65.88.62:lambda-gh200-h"
    "100.99.27.56:lambda-gh200-i"
    "100.96.142.42:lambda-gh200-k"
    "100.76.145.60:lambda-gh200-l"
    "100.117.177.83:lambda-gh200-m"
)

SSH_OPTS="-o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes"

show_status() {
    echo "=== CANONICAL SELFPLAY MONITOR ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    echo ""
    printf "%-20s %-18s %-10s %-10s %s\n" "NODE" "CONFIG" "GAMES" "STATUS" "PROGRESS"
    printf "%-20s %-18s %-10s %-10s %s\n" "----" "------" "-----" "------" "--------"

    for entry in "${NODES[@]}"; do
        ip=$(echo $entry | cut -d: -f1)
        name=$(echo $entry | cut -d: -f2)

        # Get process info
        proc_info=$(ssh $SSH_OPTS ubuntu@$ip 'ps aux | grep "generate_canonical" | grep -v grep | head -1' 2>/dev/null)

        if [ -n "$proc_info" ]; then
            # Extract config from command line
            config=$(echo "$proc_info" | sed 's/.*--board \([^ ]*\).*--num-players \([0-9]*\).*/\1_\2p/')

            # Try to get game count from db
            db_path=$(echo "$proc_info" | sed 's/.*--db \([^ ]*\).*/\1/')
            games=$(ssh $SSH_OPTS ubuntu@$ip "python3 -c \"import sqlite3; c=sqlite3.connect('$db_path'); print(c.execute('SELECT COUNT(*) FROM games').fetchone()[0])\" 2>/dev/null" 2>/dev/null || echo "?")

            # Get latest log line for progress
            log_file="logs/canonical_${config}.log"
            progress=$(ssh $SSH_OPTS ubuntu@$ip "tail -1 ~/ringrift/ai-service/$log_file 2>/dev/null | grep -o 'PROGRESS[^|]*' | head -1" 2>/dev/null || echo "")

            printf "%-20s %-18s %-10s %-10s %s\n" "$name" "$config" "$games" "RUNNING" "$progress"
        else
            printf "%-20s %-18s %-10s %-10s %s\n" "$name" "-" "-" "IDLE" ""
        fi
    done

    echo ""
    echo "=== TOTAL CANONICAL DATA ==="

    # Get totals from leader node
    ssh $SSH_OPTS ubuntu@100.88.176.74 'python3 -c "
import sqlite3, os
db_dir = os.path.expanduser(\"~/ringrift/ai-service/data/games\")
totals = {}
for f in os.listdir(db_dir):
    if f.startswith(\"canonical_\") and f.endswith(\".db\"):
        try:
            conn = sqlite3.connect(os.path.join(db_dir, f))
            count = conn.execute(\"SELECT COUNT(*) FROM games\").fetchone()[0]
            # Extract config from filename
            parts = f.replace(\"canonical_\", \"\").replace(\".db\", \"\").split(\"_\")
            if len(parts) >= 2:
                config = f\"{parts[0]}_{parts[1]}\"
                totals[config] = totals.get(config, 0) + count
            conn.close()
        except: pass
for config, count in sorted(totals.items()):
    target = 1000 if \"2p\" in config else 200
    pct = min(100, count * 100 // target)
    print(f\"  {config}: {count} games ({pct}% of {target} target)\")
"' 2>/dev/null
}

if [ "$1" == "--watch" ]; then
    while true; do
        clear
        show_status
        echo ""
        echo "Refreshing in 60s... (Ctrl+C to exit)"
        sleep 60
    done
else
    show_status
fi

#!/bin/bash
# Check training data generation progress across cluster
# Usage: ./check_training_progress.sh

SSH_KEY="$HOME/.ssh/id_ed25519"
TARGET=10000

echo "=============================================="
echo "Training Data Progress - $(date)"
echo "Target: $TARGET games per config"
echo "=============================================="

# Check nebius-h100-1 (has most data)
echo ""
echo "=== Data on nebius-h100-1 ==="
timeout 30 ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$SSH_KEY" \
    ubuntu@89.169.111.139 "
cd ~/ringrift/ai-service
source venv/bin/activate
python3 -c \"
import sqlite3
from pathlib import Path

configs = {}
for db in Path('data/games').glob('canonical_*_?p.db'):
    try:
        conn = sqlite3.connect(str(db))
        count = conn.execute('SELECT COUNT(*) FROM games').fetchone()[0]
        conn.close()
        name = db.stem.replace('canonical_', '')
        configs[name] = count
    except:
        pass

for name in sorted(configs.keys()):
    count = configs[name]
    pct = min(100, count * 100 // $TARGET)
    bar = '=' * (pct // 5) + ' ' * (20 - pct // 5)
    status = '✅' if count >= $TARGET else '🔄'
    print(f'{name:20} [{bar}] {count:6,} / $TARGET {status}')
\"
" 2>/dev/null || echo "Cannot reach nebius-h100-1"

# GPU utilization
echo ""
echo "=== GPU Utilization ==="
for entry in \
    "nebius-h100-1:ubuntu@89.169.111.139:22" \
    "nebius-h100-3:ubuntu@89.169.110.128:22" \
    "runpod-a100-1:root@38.128.233.145:33085" \
    "runpod-a100-2:root@104.255.9.187:11681" \
    "vultr:root@208.167.249.164:22"
do
    name=$(echo "$entry" | cut -d: -f1)
    user_host=$(echo "$entry" | cut -d: -f2)
    port=$(echo "$entry" | cut -d: -f3)

    result=$(timeout 8 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        -p "$port" -i "$SSH_KEY" "$user_host" \
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1" 2>/dev/null)
    procs=$(timeout 8 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
        -p "$port" -i "$SSH_KEY" "$user_host" \
        "ps aux | grep 'selfplay.py' | grep -v grep | wc -l" 2>/dev/null)

    printf "%-20s GPU: %4s  Selfplay: %s\n" "$name" "${result:-?}%" "${procs:-?} procs"
done

echo ""
echo "=============================================="

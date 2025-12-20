#!/bin/bash
LOGFILE="/Users/armand/Development/RingRift/ai-service/logs/cluster_health.log"
DURATION=${1:-10}
CYCLES=$((DURATION * 12))  # 5 min intervals

echo "[$(date)] Monitor starting: $DURATION hours, $CYCLES cycles" >> "$LOGFILE"

for i in $(seq 1 $CYCLES); do
    echo "" >> "$LOGFILE"
    echo "[$(date)] === Cycle $i/$CYCLES ===" >> "$LOGFILE"
    
    # Check A40
    A40_GPU=$(ssh -o ConnectTimeout=15 root@100.97.157.45 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null || echo "OFFLINE")
    A40_PROCS=$(ssh -o ConnectTimeout=15 root@100.97.157.45 "pgrep -c python" 2>/dev/null || echo "0")
    echo "[$(date)] A40: GPU=$A40_GPU, Procs=$A40_PROCS" >> "$LOGFILE"
    
    # Check 5070
    N5070_GPU=$(ssh -o ConnectTimeout=15 root@100.74.40.31 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null || echo "OFFLINE")
    N5070_PROCS=$(ssh -o ConnectTimeout=15 root@100.74.40.31 "pgrep -c python" 2>/dev/null || echo "0")
    echo "[$(date)] 5070: GPU=$N5070_GPU, Procs=$N5070_PROCS" >> "$LOGFILE"
    
    # Check 4080S
    N4080_GPU=$(ssh -o ConnectTimeout=15 root@100.79.143.125 "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null || echo "OFFLINE")
    N4080_PROCS=$(ssh -o ConnectTimeout=15 root@100.79.143.125 "pgrep -c python" 2>/dev/null || echo "0")
    echo "[$(date)] 4080S: GPU=$N4080_GPU, Procs=$N4080_PROCS" >> "$LOGFILE"
    
    # If A40 idle, start work
    if [[ "$A40_GPU" == "0 %" ]] && [[ "$A40_PROCS" -lt 5 ]]; then
        echo "[$(date)] A40 idle - starting selfplay" >> "$LOGFILE"
        ssh root@100.97.157.45 "cd /root/ringrift/ai-service && source venv/bin/activate && PYTHONPATH=. nohup python scripts/run_gpu_selfplay.py --board square19 --num-games 100 --batch-size 64 --engine-mode random-only >/tmp/selfplay.log 2>&1 &" 2>/dev/null
    fi
    
    echo "[$(date)] Sleeping 5 min..." >> "$LOGFILE"
    [ $i -lt $CYCLES ] && sleep 300
done

echo "[$(date)] Monitor complete" >> "$LOGFILE"

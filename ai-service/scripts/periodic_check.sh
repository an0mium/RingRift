#!/bin/bash
# Periodic cluster check - runs every 5 minutes
# Writes summary to /tmp/periodic_check.log

LOG="/tmp/periodic_check.log"
ISSUES="/tmp/cluster_issues.log"

check_and_log() {
    echo "" >> $LOG
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> $LOG

    # Count online Lambda nodes
    lambda_online=0
    lambda_issues=""
    for host in 192.222.51.29 192.222.51.162 192.222.58.122 192.222.57.162 192.222.57.178 192.222.57.79 192.222.56.123 192.222.50.112 192.222.51.150 192.222.51.233 192.222.50.219 192.222.51.204 192.222.51.161 192.222.51.92 192.222.51.215; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@$host "echo OK" >/dev/null 2>&1; then
            lambda_online=$((lambda_online + 1))
        else
            lambda_issues="$lambda_issues $host"
        fi
    done
    echo "Lambda nodes: $lambda_online/15 online" >> $LOG
    if [ -n "$lambda_issues" ]; then
        echo "ISSUE: Offline Lambda nodes:$lambda_issues" >> $LOG
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Offline Lambda:$lambda_issues" >> $ISSUES
    fi

    # Check key benchmarks
    echo "--- Benchmarks ---" >> $LOG

    # sq8 2P
    sq8_2p=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@192.222.51.162 "grep 'Results:' /tmp/benchmark_sq8_2p.log 2>/dev/null | wc -l" 2>/dev/null)
    echo "sq8 2P: $sq8_2p/10 matchups complete" >> $LOG

    # sq8 4P
    sq8_4p=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@192.222.57.79 "grep 'Results:' /tmp/benchmark_sq8_4p.log 2>/dev/null | wc -l" 2>/dev/null)
    echo "sq8 4P: $sq8_4p/10 matchups complete" >> $LOG

    # hex8 2P
    hex8_2p=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ubuntu@192.222.56.123 "grep 'Results:' /tmp/benchmark_hex8_2p.log 2>/dev/null | wc -l" 2>/dev/null)
    echo "hex8 2P: $hex8_2p/10 matchups complete" >> $LOG

    # sq19 2P (Vast)
    sq19_2p=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i ~/.ssh/id_cluster -p 18168 root@ssh5.vast.ai "grep 'Results:' /tmp/benchmark_sq19_2p.log 2>/dev/null | wc -l" 2>/dev/null)
    echo "sq19 2P: $sq19_2p/10 matchups complete" >> $LOG

    # Check for new alerts
    new_alerts=$(tail -5 /tmp/cluster_alerts.log 2>/dev/null | grep -c "ALERT")
    if [ "$new_alerts" -gt 0 ]; then
        echo "WARNING: $new_alerts new alerts in cluster_alerts.log" >> $LOG
    fi
}

# Run check every 5 minutes for 10 hours
echo "Periodic check started at $(date)" > $LOG
for i in $(seq 1 120); do
    check_and_log
    sleep 300  # 5 minutes
done
echo "Periodic check completed at $(date)" >> $LOG

#!/bin/bash
# Monitor Gumbel data accumulation across Lambda cluster
# Usage: ./scripts/monitor_gumbel_data.sh [config]
# Example: ./scripts/monitor_gumbel_data.sh square8_4p

CONFIG="${1:-square8_4p}"
NODES="lambda-gh200-1 lambda-gh200-2 lambda-gh200-3 lambda-gh200-5 lambda-gh200-8 lambda-gh200-training"

echo "=== Gumbel Data Monitor - ${CONFIG} ==="
echo "Timestamp: $(date)"
echo ""

total_games=0
for node in $NODES; do
    count=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ubuntu@$node \
        "cd ~/ringrift/ai-service 2>/dev/null && \
         find data/selfplay/gumbel/${CONFIG} -name '*.jsonl' -exec cat {} + 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    if [ "$count" != "0" ] && [ -n "$count" ]; then
        echo "  $node: $count games"
        total_games=$((total_games + count))
    else
        echo "  $node: 0 games (or unreachable)"
    fi
done

echo ""
echo "TOTAL: $total_games games for ${CONFIG}"
echo "Target: 3,000+ games"
echo "Progress: $(echo "scale=1; $total_games * 100 / 3000" | bc 2>/dev/null || echo "?")%"

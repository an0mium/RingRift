#!/bin/bash
# Monitor hyperparameter tuning jobs across the cluster
# Usage: ./scripts/monitor_hp_tuning.sh [--progress] [--logs]

HOSTS=(
  "lambda-h100:ubuntu@209.20.157.81:square8_2p"
  "lambda-2xh100:ubuntu@192.222.53.22:square8_3p"
  "lambda-a10:ubuntu@150.136.65.197:square8_4p"
  "lambda-gh200-a:ubuntu@192.222.51.29:square19_2p"
  "lambda-gh200-b:ubuntu@192.222.51.167:square19_3p"
  "lambda-gh200-c:ubuntu@192.222.51.162:square19_4p"
  "lambda-gh200-d:ubuntu@192.222.58.122:hexagonal_2p"
  "lambda-gh200-e:ubuntu@192.222.57.162:hexagonal_3p"
  "lambda-gh200-f:ubuntu@192.222.57.178:hexagonal_4p"
)

show_progress() {
  echo "=== HYPERPARAMETER TUNING PROGRESS ==="
  echo ""
  printf "%-15s %-12s %-8s %-10s %s\n" "HOST" "CONFIG" "TRIALS" "BEST" "STATUS"
  echo "--------------------------------------------------------------"

  for entry in "${HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"

    # Check if process is running
    running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f tune_hyperparameters' 2>/dev/null)

    if [ -n "$running" ]; then
      status="RUNNING"
      # Get progress from session file
      result=$(ssh -o ConnectTimeout=5 "$host" "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get(\"trials\",[])), d.get(\"best_score\",-1))' 2>/dev/null" 2>/dev/null)
      trials=$(echo "$result" | awk '{print $1}')
      best=$(echo "$result" | awk '{printf \"%.4f\", $2}')
      [ -z "$trials" ] && trials="0"
      [ "$best" == "-1.0000" ] && best="--"
    else
      status="DONE/STOPPED"
      result=$(ssh -o ConnectTimeout=5 "$host" "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get(\"trials\",[])), d.get(\"best_score\",-1))' 2>/dev/null" 2>/dev/null)
      trials=$(echo "$result" | awk '{print $1}')
      best=$(echo "$result" | awk '{printf \"%.4f\", $2}')
      [ -z "$trials" ] && trials="?"
    fi

    printf "%-15s %-12s %-8s %-10s %s\n" "$name" "$config" "$trials" "$best" "$status"
  done
  echo ""
}

show_logs() {
  echo "=== RECENT LOG OUTPUT ==="
  for entry in "${HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"
    echo ""
    echo "--- $name ($config) ---"
    ssh -o ConnectTimeout=5 "$host" "tail -5 ~/ringrift/ai-service/logs/hp_tuning/${config}.log 2>/dev/null" 2>/dev/null || echo "No log available"
  done
}

show_best_params() {
  echo "=== BEST HYPERPARAMETERS FOUND ==="
  for entry in "${HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"
    echo ""
    echo "--- $config ---"
    ssh -o ConnectTimeout=5 "$host" "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f\"Trials: {len(d.get(\"trials\",[]))}\"); print(f\"Best score: {d.get(\"best_score\",-1):.4f}\"); print(f\"Best params:\"); import pprint; pprint.pprint(d.get(\"best_params\",{}))' 2>/dev/null" 2>/dev/null || echo "No results yet"
  done
}

case "${1:-}" in
  --progress|-p)
    show_progress
    ;;
  --logs|-l)
    show_logs
    ;;
  --best|-b)
    show_best_params
    ;;
  --watch|-w)
    while true; do
      clear
      show_progress
      echo "Refreshing every 60s... (Ctrl+C to stop)"
      sleep 60
    done
    ;;
  *)
    show_progress
    echo "Usage: $0 [--progress|-p] [--logs|-l] [--best|-b] [--watch|-w]"
    ;;
esac

#!/bin/bash
# Monitor all distributed training jobs: HP tuning + selfplay generation
# Usage: ./scripts/monitor_all_jobs.sh [--watch]

# HP Tuning hosts (3 Lambda GPUs)
HP_HOSTS=(
  "lambda-h100:ubuntu@209.20.157.81:square8_2p"
  "lambda-2xh100:ubuntu@192.222.53.22:square8_3p"
  "lambda-a10:ubuntu@150.136.65.197:square8_4p"
)

# Selfplay hosts (6 GH200s)
SELFPLAY_HOSTS=(
  "GH200-a:ubuntu@192.222.51.29:hexagonal_4p"
  "GH200-b:ubuntu@192.222.51.167:square19_4p"
  "GH200-c:ubuntu@192.222.51.162:hexagonal_2p"
  "GH200-d:ubuntu@192.222.58.122:square19_2p"
  "GH200-e:ubuntu@192.222.57.162:hexagonal_3p"
  "GH200-f:ubuntu@192.222.57.178:square19_3p"
)

show_hp_tuning() {
  echo "=============================================="
  echo "   HYPERPARAMETER TUNING STATUS"
  echo "=============================================="
  printf "%-15s %-12s %-8s %-12s %s\n" "HOST" "CONFIG" "TRIAL" "BEST_SCORE" "STATUS"
  echo "----------------------------------------------"

  for entry in "${HP_HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"
    running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f tune_hyperparameters' 2>/dev/null)

    if [ -n "$running" ]; then
      status="RUNNING"
    else
      status="DONE/STOPPED"
    fi

    result=$(ssh -o ConnectTimeout=5 "$host" "cat ~/ringrift/ai-service/logs/hp_tuning/${config}/tuning_session.json 2>/dev/null | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get(\"trials\",[])), d.get(\"best_score\",-1))' 2>/dev/null" 2>/dev/null)
    trials=$(echo "$result" | awk '{print $1}')
    best=$(echo "$result" | awk '{printf "%.6f", $2}')
    [ -z "$trials" ] && trials="0"
    [ "$best" == "-1.000000" ] && best="--"

    printf "%-15s %-12s %-8s %-12s %s\n" "$name" "$config" "$trials/50" "$best" "$status"
  done
  echo ""
}

show_selfplay() {
  echo "=============================================="
  echo "   SELFPLAY DATA GENERATION STATUS"
  echo "=============================================="
  printf "%-15s %-12s %-8s %-10s %s\n" "HOST" "CONFIG" "GAMES" "DB_SIZE" "STATUS"
  echo "----------------------------------------------"

  for entry in "${SELFPLAY_HOSTS[@]}"; do
    IFS=':' read -r name host config <<< "$entry"
    running=$(ssh -o ConnectTimeout=5 "$host" 'pgrep -f "run_self_play_soak|generate_canonical_selfplay"' 2>/dev/null)

    if [ -n "$running" ]; then
      status="RUNNING"
    else
      status="DONE/STOPPED"
    fi

    games=$(ssh -o ConnectTimeout=5 "$host" "sqlite3 ~/ringrift/ai-service/data/games/new_${config}.db 'SELECT COUNT(*) FROM games' 2>/dev/null" 2>/dev/null)
    [ -z "$games" ] && games="0"

    size=$(ssh -o ConnectTimeout=5 "$host" "ls -lh ~/ringrift/ai-service/data/games/new_${config}.db 2>/dev/null | awk '{print \$5}'" 2>/dev/null)
    [ -z "$size" ] && size="--"

    printf "%-15s %-12s %-8s %-10s %s\n" "$name" "$config" "$games" "$size" "$status"
  done
  echo ""
}

show_all() {
  show_hp_tuning
  show_selfplay
  echo "Time: $(date)"
}

case "${1:-}" in
  --watch|-w)
    while true; do
      clear
      show_all
      echo "Refreshing every 60s... (Ctrl+C to stop)"
      sleep 60
    done
    ;;
  --hp)
    show_hp_tuning
    ;;
  --selfplay)
    show_selfplay
    ;;
  *)
    show_all
    echo "Usage: $0 [--watch|-w] [--hp] [--selfplay]"
    ;;
esac

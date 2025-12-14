#!/bin/bash
# ============================================================================
# RingRift Selfplay Throughput Monitor
# ============================================================================
# Monitors selfplay game generation throughput and alerts if below threshold.
#
# Usage:
#   ./selfplay_throughput_monitor.sh [--verbose] [--dry-run]
#
# Environment variables:
#   RINGRIFT_SLACK_WEBHOOK      - Slack webhook URL for alerts
#   RINGRIFT_CLUSTER_API        - Cluster API endpoint
#   RINGRIFT_MIN_THROUGHPUT     - Minimum acceptable games/second (default: 30)
#   RINGRIFT_ALERT_COOLDOWN_MIN - Minutes between repeat alerts (default: 30)
#
# Crontab entry (every 2 minutes):
#   */2 * * * * /opt/ringrift/monitoring/selfplay_throughput_monitor.sh >> /var/log/ringrift-selfplay.log 2>&1
# ============================================================================

set -euo pipefail

# Configuration
CLUSTER_API="${RINGRIFT_CLUSTER_API:-https://cluster.ringrift.ai}"
SLACK_WEBHOOK="${RINGRIFT_SLACK_WEBHOOK:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
MIN_THROUGHPUT="${RINGRIFT_MIN_THROUGHPUT:-30}"
ALERT_COOLDOWN_MIN="${RINGRIFT_ALERT_COOLDOWN_MIN:-30}"
STATE_FILE="/tmp/ringrift_selfplay_monitor_state"
VERBOSE=""
DRY_RUN=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --verbose) VERBOSE="true" ;;
    --dry-run) DRY_RUN="true" ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_verbose() {
  if [ "$VERBOSE" = "true" ]; then
    log "$1"
  fi
}

should_alert() {
  local alert_type="$1"
  local current_time=$(date +%s)

  if [ ! -f "$STATE_FILE" ]; then
    echo "{}" > "$STATE_FILE"
  fi

  local last_alert=$(jq -r ".${alert_type}_last_alert // 0" "$STATE_FILE")
  local cooldown_seconds=$((ALERT_COOLDOWN_MIN * 60))

  if [ $((current_time - last_alert)) -gt "$cooldown_seconds" ]; then
    # Update state file
    jq ".${alert_type}_last_alert = $current_time" "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
    return 0
  fi
  return 1
}

send_slack_alert() {
  local message="$1"
  local severity="${2:-warning}"

  if [ -z "$SLACK_WEBHOOK" ]; then
    log "WARN: RINGRIFT_SLACK_WEBHOOK not set"
    return
  fi

  if [ "$DRY_RUN" = "true" ]; then
    log "DRY-RUN: Would send: $message"
    return
  fi

  local emoji=""
  case $severity in
    critical) emoji=":rotating_light:" ;;
    warning)  emoji=":warning:" ;;
    info)     emoji=":chart_with_upwards_trend:" ;;
  esac

  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${emoji} ${message}\"}" \
    "$SLACK_WEBHOOK" > /dev/null
}

put_cloudwatch_metric() {
  local metric_name="$1"
  local value="$2"
  local unit="${3:-Count}"
  local dimensions="${4:-}"

  if [ "$DRY_RUN" = "true" ]; then
    log "DRY-RUN: CloudWatch metric: $metric_name = $value"
    return
  fi

  local dim_args=""
  if [ -n "$dimensions" ]; then
    dim_args="--dimensions $dimensions"
  fi

  aws cloudwatch put-metric-data \
    --region "$AWS_REGION" \
    --namespace "RingRift/Selfplay" \
    --metric-name "$metric_name" \
    --value "$value" \
    --unit "$unit" \
    $dim_args 2>/dev/null || true
}

# Fetch selfplay stats
log_verbose "Fetching selfplay stats..."
STATS=$(curl -s --connect-timeout 10 --max-time 20 "${CLUSTER_API}/api/selfplay/stats" 2>/dev/null || echo '{}')

# If stats endpoint not available, calculate from cluster status
if [ -z "$STATS" ] || [ "$STATS" = "{}" ]; then
  log_verbose "Selfplay stats endpoint unavailable, calculating from cluster status..."

  CLUSTER_STATUS=$(curl -s --connect-timeout 10 "${CLUSTER_API}/api/cluster/status" 2>/dev/null || echo '{}')

  # Count active selfplay jobs and estimate throughput
  TOTAL_JOBS=$(echo "$CLUSTER_STATUS" | jq '[.peers[] | select(.status == "online") | .selfplay_jobs] | add // 0')

  # Estimate throughput based on job count (rough heuristic: ~0.5 games/sec per job)
  ESTIMATED_THROUGHPUT=$(echo "$TOTAL_JOBS * 0.5" | bc -l 2>/dev/null || echo "0")

  log "Estimated throughput from $TOTAL_JOBS jobs: ${ESTIMATED_THROUGHPUT} g/s"

  put_cloudwatch_metric "EstimatedGamesPerSecond" "$ESTIMATED_THROUGHPUT" "Count/Second"
  put_cloudwatch_metric "ActiveSelfplayJobs" "$TOTAL_JOBS"

  if (( $(echo "$ESTIMATED_THROUGHPUT < $MIN_THROUGHPUT" | bc -l) )); then
    if should_alert "low_throughput"; then
      send_slack_alert "Selfplay throughput low: ~${ESTIMATED_THROUGHPUT} g/s (min: ${MIN_THROUGHPUT})" "warning"
    fi
  fi

  exit 0
fi

# Parse stats from API
TOTAL_THROUGHPUT=$(echo "$STATS" | jq '.total_games_per_second // 0')
TOTAL_GAMES=$(echo "$STATS" | jq '.total_games // 0')
BOARD_STATS=$(echo "$STATS" | jq -r '.by_board_type // {}')

log "Selfplay throughput: ${TOTAL_THROUGHPUT} g/s, total games: ${TOTAL_GAMES}"

# Put metrics
put_cloudwatch_metric "GamesPerSecond" "$TOTAL_THROUGHPUT" "Count/Second" "Environment=production"
put_cloudwatch_metric "TotalGamesGenerated" "$TOTAL_GAMES" "Count" "Environment=production"

# Per-board-type metrics
for board_type in $(echo "$BOARD_STATS" | jq -r 'keys[]' 2>/dev/null); do
  BOARD_THROUGHPUT=$(echo "$BOARD_STATS" | jq -r ".\"$board_type\".games_per_second // 0")
  put_cloudwatch_metric "GamesPerSecond" "$BOARD_THROUGHPUT" "Count/Second" "BoardType=$board_type"
done

# Check throughput threshold
if (( $(echo "$TOTAL_THROUGHPUT < $MIN_THROUGHPUT" | bc -l) )); then
  log "WARNING: Throughput ${TOTAL_THROUGHPUT} g/s below threshold ${MIN_THROUGHPUT}"

  if should_alert "low_throughput"; then
    send_slack_alert "Selfplay throughput BELOW threshold: ${TOTAL_THROUGHPUT} g/s (min: ${MIN_THROUGHPUT})" "warning"
  fi
else
  log_verbose "Throughput OK: ${TOTAL_THROUGHPUT} g/s"
fi

# Check for stalled board types (no games in last 10 minutes)
STALLED_BOARDS=$(echo "$STATS" | jq -r '.by_board_type | to_entries[] | select(.value.games_per_second < 0.1) | .key' 2>/dev/null || true)
if [ -n "$STALLED_BOARDS" ]; then
  log "WARNING: Stalled board types: $STALLED_BOARDS"
fi

log "Monitor complete"

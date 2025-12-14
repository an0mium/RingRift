#!/bin/bash
# ============================================================================
# RingRift Training Pipeline Monitor
# ============================================================================
# Monitors NNUE training jobs, Elo tournaments, and model gating.
#
# Usage:
#   ./training_pipeline_monitor.sh [--verbose] [--dry-run]
#
# Environment variables:
#   RINGRIFT_SLACK_WEBHOOK - Slack webhook URL for alerts
#   RINGRIFT_CLUSTER_API   - Cluster API endpoint
#
# Crontab entry (every 10 minutes):
#   */10 * * * * /opt/ringrift/monitoring/training_pipeline_monitor.sh >> /var/log/ringrift-training.log 2>&1
# ============================================================================

set -euo pipefail

CLUSTER_API="${RINGRIFT_CLUSTER_API:-https://cluster.ringrift.ai}"
SLACK_WEBHOOK="${RINGRIFT_SLACK_WEBHOOK:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
VERBOSE=""
DRY_RUN=""

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
  [ "$VERBOSE" = "true" ] && log "$1"
}

send_slack() {
  local message="$1"
  local severity="${2:-info}"

  [ -z "$SLACK_WEBHOOK" ] && return
  [ "$DRY_RUN" = "true" ] && { log "DRY-RUN: $message"; return; }

  local emoji=":robot_face:"
  case $severity in
    success) emoji=":white_check_mark:" ;;
    warning) emoji=":warning:" ;;
    critical) emoji=":rotating_light:" ;;
  esac

  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${emoji} ${message}\"}" \
    "$SLACK_WEBHOOK" > /dev/null
}

put_metric() {
  local name="$1" value="$2" unit="${3:-Count}"
  [ "$DRY_RUN" = "true" ] && return

  aws cloudwatch put-metric-data \
    --region "$AWS_REGION" \
    --namespace "RingRift/Training" \
    --metric-name "$name" \
    --value "$value" \
    --unit "$unit" 2>/dev/null || true
}

# Fetch training status
log_verbose "Fetching training status..."
TRAINING_STATUS=$(curl -s --connect-timeout 10 "${CLUSTER_API}/api/training/status" 2>/dev/null || echo '{}')

if [ -z "$TRAINING_STATUS" ] || [ "$TRAINING_STATUS" = "{}" ]; then
  log "Training status endpoint unavailable, checking cluster status..."

  CLUSTER=$(curl -s --connect-timeout 10 "${CLUSTER_API}/api/cluster/status" 2>/dev/null || echo '{}')

  TRAINING_JOBS=$(echo "$CLUSTER" | jq '[.peers[] | select(.training_jobs > 0)] | length')
  TRAINING_NODES=$(echo "$CLUSTER" | jq -r '[.peers[] | select(.training_jobs > 0) | .node_id] | join(", ")')

  log "Active training jobs on: ${TRAINING_NODES:-none}"
  put_metric "ActiveTrainingNodes" "$TRAINING_JOBS"

  exit 0
fi

# Parse training status
ACTIVE_TRAINING=$(echo "$TRAINING_STATUS" | jq '.active_jobs // 0')
QUEUED_TRAINING=$(echo "$TRAINING_STATUS" | jq '.queued_jobs // 0')
COMPLETED_TODAY=$(echo "$TRAINING_STATUS" | jq '.completed_today // 0')
FAILED_TODAY=$(echo "$TRAINING_STATUS" | jq '.failed_today // 0')

log "Training: $ACTIVE_TRAINING active, $QUEUED_TRAINING queued, $COMPLETED_TODAY completed today"

put_metric "ActiveTrainingJobs" "$ACTIVE_TRAINING"
put_metric "QueuedTrainingJobs" "$QUEUED_TRAINING"
put_metric "CompletedTrainingJobs" "$COMPLETED_TODAY"
put_metric "FailedTrainingJobs" "$FAILED_TODAY"

# Alert on failures
if [ "$FAILED_TODAY" -gt 0 ]; then
  send_slack "Training pipeline: $FAILED_TODAY jobs failed today" "warning"
fi

# Check Elo tournament status
log_verbose "Checking Elo tournament status..."
ELO_STATUS=$(curl -s --connect-timeout 10 "${CLUSTER_API}/api/elo/status" 2>/dev/null || echo '{}')

if [ -n "$ELO_STATUS" ] && [ "$ELO_STATUS" != "{}" ]; then
  TOURNAMENT_ACTIVE=$(echo "$ELO_STATUS" | jq '.tournament_active // false')
  GAMES_PLAYED=$(echo "$ELO_STATUS" | jq '.games_played // 0')
  TOP_MODEL=$(echo "$ELO_STATUS" | jq -r '.top_model.name // "unknown"')
  TOP_ELO=$(echo "$ELO_STATUS" | jq '.top_model.elo // 0')

  log "Elo tournament: active=$TOURNAMENT_ACTIVE, games=$GAMES_PLAYED, top=$TOP_MODEL ($TOP_ELO)"

  put_metric "EloGamesPlayed" "$GAMES_PLAYED"
  put_metric "TopModelElo" "$TOP_ELO"

  # Celebrate new top model
  if [ "$TOP_ELO" -gt 1700 ]; then
    log_verbose "Top model $TOP_MODEL exceeds 1700 Elo"
  fi
fi

# Check gating status
log_verbose "Checking model gating..."
GATING_STATUS=$(curl -s --connect-timeout 10 "${CLUSTER_API}/api/gating/status" 2>/dev/null || echo '{}')

if [ -n "$GATING_STATUS" ] && [ "$GATING_STATUS" != "{}" ]; then
  PENDING_GATES=$(echo "$GATING_STATUS" | jq '.pending // 0')
  PASSED_TODAY=$(echo "$GATING_STATUS" | jq '.passed_today // 0')
  FAILED_GATES=$(echo "$GATING_STATUS" | jq '.failed_today // 0')

  log "Gating: $PENDING_GATES pending, $PASSED_TODAY passed, $FAILED_GATES failed today"

  put_metric "PendingGates" "$PENDING_GATES"
  put_metric "PassedGates" "$PASSED_TODAY"
  put_metric "FailedGates" "$FAILED_GATES"

  if [ "$PASSED_TODAY" -gt 0 ]; then
    send_slack "Model gating: $PASSED_TODAY models passed today" "success"
  fi
fi

log "Training pipeline monitor complete"

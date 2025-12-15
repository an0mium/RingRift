#!/bin/bash
# ============================================================================
# RingRift Cluster Health Check
# ============================================================================
# Monitors cluster node status, detects offline critical nodes, and sends
# alerts via Slack webhook and CloudWatch.
#
# Usage:
#   ./cluster_health_check.sh [--verbose] [--dry-run]
#
# Environment variables:
#   RINGRIFT_SLACK_WEBHOOK - Slack webhook URL for alerts
#   RINGRIFT_CLUSTER_API   - Cluster API endpoint (default: https://cluster.ringrift.ai)
#   AWS_REGION             - AWS region for CloudWatch (default: us-east-1)
#
# Crontab entry (every 5 minutes):
#   */5 * * * * /opt/ringrift/monitoring/cluster_health_check.sh >> /var/log/ringrift-monitor.log 2>&1
# ============================================================================

set -euo pipefail

# Configuration
CLUSTER_API="${RINGRIFT_CLUSTER_API:-https://cluster.ringrift.ai}"
SLACK_WEBHOOK="${RINGRIFT_SLACK_WEBHOOK:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
LOG_GROUP="/ringrift/cluster-health"
VERBOSE="${1:-}"
DRY_RUN=""
ALERT_STATE_DIR="${RINGRIFT_ALERT_STATE_DIR:-/tmp/ringrift-alerts}"
ALERT_COOLDOWN_SECONDS="${RINGRIFT_ALERT_COOLDOWN:-900}"  # 15 min cooldown between duplicate alerts

# Ensure alert state directory exists
mkdir -p "$ALERT_STATE_DIR"

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

# Check if alert should be throttled (deduplicate within cooldown window)
should_alert() {
  local alert_key="$1"
  local state_file="$ALERT_STATE_DIR/${alert_key}.state"
  local now
  now=$(date +%s)

  if [ -f "$state_file" ]; then
    local last_alert
    last_alert=$(cat "$state_file")
    local elapsed=$((now - last_alert))
    if [ "$elapsed" -lt "$ALERT_COOLDOWN_SECONDS" ]; then
      log_verbose "Throttling alert '$alert_key' (${elapsed}s < ${ALERT_COOLDOWN_SECONDS}s cooldown)"
      return 1  # false = should NOT alert
    fi
  fi

  echo "$now" > "$state_file"
  return 0  # true = should alert
}

# Clear alert state (call when condition resolves)
clear_alert() {
  local alert_key="$1"
  local state_file="$ALERT_STATE_DIR/${alert_key}.state"
  rm -f "$state_file"
}

send_slack_alert() {
  local message="$1"
  local severity="${2:-warning}"
  local alert_key="${3:-}"  # Optional: if provided, deduplicates alerts

  # Generate alert key from message if not provided
  if [ -z "$alert_key" ]; then
    alert_key=$(echo "$message" | md5sum | cut -c1-16)
  fi

  if [ -z "$SLACK_WEBHOOK" ]; then
    log "WARN: RINGRIFT_SLACK_WEBHOOK not set, skipping Slack alert"
    return
  fi

  if [ "$DRY_RUN" = "true" ]; then
    log "DRY-RUN: Would send Slack alert: $message"
    return
  fi

  # Check cooldown
  if ! should_alert "$alert_key"; then
    return
  fi

  local emoji=""
  case $severity in
    critical) emoji=":rotating_light:" ;;
    warning)  emoji=":warning:" ;;
    info)     emoji=":information_source:" ;;
  esac

  curl -s -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"${emoji} ${message}\"}" \
    "$SLACK_WEBHOOK" > /dev/null
}

put_cloudwatch_metric() {
  local metric_name="$1"
  local value="$2"
  local unit="${3:-Count}"

  if [ "$DRY_RUN" = "true" ]; then
    log "DRY-RUN: Would put CloudWatch metric: $metric_name = $value $unit"
    return
  fi

  aws cloudwatch put-metric-data \
    --region "$AWS_REGION" \
    --namespace "RingRift/Cluster" \
    --metric-name "$metric_name" \
    --value "$value" \
    --unit "$unit" \
    --dimensions Environment=production 2>/dev/null || true
}

# Fetch cluster status
log_verbose "Fetching cluster status from $CLUSTER_API..."
STATUS=$(curl -s --connect-timeout 15 --max-time 30 "${CLUSTER_API}/api/cluster/status" 2>/dev/null || echo '{"error": "failed to fetch"}')

if echo "$STATUS" | jq -e '.error' > /dev/null 2>&1; then
  log "ERROR: Failed to fetch cluster status"
  send_slack_alert "Failed to fetch cluster status from $CLUSTER_API" "critical" "fetch_failed"
  exit 1
fi
# Clear fetch failure state on success
clear_alert "fetch_failed"

# Extract metrics
ONLINE_COUNT=$(echo "$STATUS" | jq '[.peers[] | select(.status == "online")] | length')
OFFLINE_COUNT=$(echo "$STATUS" | jq '[.peers[] | select(.status == "offline")] | length')
TOTAL_PEERS=$(echo "$STATUS" | jq '.peer_count')
LEADER_ID=$(echo "$STATUS" | jq -r '.leader_id // "none"')
QUORUM_OK=$(echo "$STATUS" | jq -r '.voter_quorum_ok')
TOTAL_SELFPLAY_JOBS=$(echo "$STATUS" | jq '[.peers[].selfplay_jobs] | add // 0')
TOTAL_TRAINING_JOBS=$(echo "$STATUS" | jq '[.peers[].training_jobs] | add // 0')

log "Cluster status: $ONLINE_COUNT online, $OFFLINE_COUNT offline, leader=$LEADER_ID, quorum=$QUORUM_OK"

# Put CloudWatch metrics
put_cloudwatch_metric "OnlineNodes" "$ONLINE_COUNT"
put_cloudwatch_metric "OfflineNodes" "$OFFLINE_COUNT"
put_cloudwatch_metric "SelfplayJobs" "$TOTAL_SELFPLAY_JOBS"
put_cloudwatch_metric "TrainingJobs" "$TOTAL_TRAINING_JOBS"
put_cloudwatch_metric "QuorumHealthy" "$([ "$QUORUM_OK" = "true" ] && echo 1 || echo 0)"

# Check critical nodes
CRITICAL_NODES=("lambda-h100" "lambda-2xh100" "aws-selfplay" "aws-staging")
CRITICAL_OFFLINE=""

for node in "${CRITICAL_NODES[@]}"; do
  NODE_STATUS=$(echo "$STATUS" | jq -r ".peers[] | select(.node_id == \"$node\") | .status" 2>/dev/null || echo "unknown")
  if [ "$NODE_STATUS" = "offline" ]; then
    CRITICAL_OFFLINE="$CRITICAL_OFFLINE $node"
    log "CRITICAL: Node $node is OFFLINE"
  fi
done

if [ -n "$CRITICAL_OFFLINE" ]; then
  send_slack_alert "CRITICAL nodes OFFLINE:$CRITICAL_OFFLINE" "critical" "critical_offline"
else
  clear_alert "critical_offline"
fi

# Check GH200 cluster health
GH200_ONLINE=$(echo "$STATUS" | jq '[.peers[] | select(.node_id | startswith("lambda-gh200")) | select(.status == "online")] | length')
GH200_TOTAL=$(echo "$STATUS" | jq '[.peers[] | select(.node_id | startswith("lambda-gh200"))] | length')

log_verbose "GH200 cluster: $GH200_ONLINE/$GH200_TOTAL online"
put_cloudwatch_metric "GH200NodesOnline" "$GH200_ONLINE"

if [ "$GH200_ONLINE" -lt 4 ] && [ "$GH200_TOTAL" -gt 0 ]; then
  send_slack_alert "GH200 cluster degraded: only $GH200_ONLINE/$GH200_TOTAL nodes online" "warning" "gh200_degraded"
else
  clear_alert "gh200_degraded"
fi

# Check quorum
if [ "$QUORUM_OK" != "true" ]; then
  send_slack_alert "QUORUM LOST - cluster may not be able to coordinate" "critical" "quorum_lost"
else
  clear_alert "quorum_lost"
fi

# Check for high CPU/memory nodes
HIGH_RESOURCE_NODES=$(echo "$STATUS" | jq -r '.peers[] | select(.status == "online") | select(.cpu_percent > 95 or .memory_percent > 90) | "\(.node_id): CPU=\(.cpu_percent)% MEM=\(.memory_percent)%"')
if [ -n "$HIGH_RESOURCE_NODES" ]; then
  log "WARNING: High resource utilization detected:"
  echo "$HIGH_RESOURCE_NODES" | while read -r line; do
    log "  $line"
  done
fi

# Check for disk pressure
DISK_PRESSURE_NODES=$(echo "$STATUS" | jq -r '.peers[] | select(.status == "online") | select(.disk_percent > 85) | "\(.node_id): DISK=\(.disk_percent)%"')
if [ -n "$DISK_PRESSURE_NODES" ]; then
  log "WARNING: Disk pressure detected:"
  echo "$DISK_PRESSURE_NODES" | while read -r line; do
    log "  $line"
  done
  send_slack_alert "Disk pressure on nodes: $(echo "$DISK_PRESSURE_NODES" | tr '\n' ' ')" "warning" "disk_pressure"
else
  clear_alert "disk_pressure"
fi

log "Health check complete"

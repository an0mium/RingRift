#!/bin/bash
# ============================================================================
# RingRift CloudWatch Setup Script
# ============================================================================
# Creates CloudWatch alarms, dashboard, log groups, and SNS topics for
# RingRift cluster monitoring.
#
# Prerequisites:
#   - AWS CLI configured with appropriate permissions
#   - jq installed
#
# Usage:
#   ./setup_cloudwatch.sh [--dry-run] [--email your@email.com]
# ============================================================================

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
SNS_TOPIC_NAME="ringrift-alerts"
DASHBOARD_NAME="RingRift-Cluster"
DRY_RUN=""
ALERT_EMAIL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN="true"; shift ;;
    --email) ALERT_EMAIL="$2"; shift 2 ;;
    *) shift ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_aws() {
  if [ "$DRY_RUN" = "true" ]; then
    log "DRY-RUN: aws $*"
  else
    aws "$@"
  fi
}

log "Setting up CloudWatch monitoring for RingRift cluster"
log "AWS Account: $AWS_ACCOUNT_ID, Region: $AWS_REGION"

# ============================================================================
# 1. Create SNS Topic
# ============================================================================
log "Creating SNS topic..."
SNS_TOPIC_ARN="arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${SNS_TOPIC_NAME}"

if [ "$DRY_RUN" != "true" ]; then
  aws sns create-topic --name "$SNS_TOPIC_NAME" --region "$AWS_REGION" 2>/dev/null || true
  log "SNS topic created: $SNS_TOPIC_ARN"
fi

if [ -n "$ALERT_EMAIL" ]; then
  log "Subscribing $ALERT_EMAIL to alerts..."
  run_aws sns subscribe \
    --topic-arn "$SNS_TOPIC_ARN" \
    --protocol email \
    --notification-endpoint "$ALERT_EMAIL" \
    --region "$AWS_REGION"
fi

# ============================================================================
# 2. Create Log Groups
# ============================================================================
log "Creating CloudWatch log groups..."

LOG_GROUPS=(
  "/ringrift/cluster-health"
  "/ringrift/selfplay"
  "/ringrift/training"
  "/ringrift/elo-tournaments"
  "/ringrift/application"
)

for log_group in "${LOG_GROUPS[@]}"; do
  run_aws logs create-log-group \
    --log-group-name "$log_group" \
    --region "$AWS_REGION" 2>/dev/null || true

  run_aws logs put-retention-policy \
    --log-group-name "$log_group" \
    --retention-in-days 30 \
    --region "$AWS_REGION" 2>/dev/null || true

  log "  Created: $log_group (30 day retention)"
done

# ============================================================================
# 3. Create CloudWatch Alarms
# ============================================================================
log "Creating CloudWatch alarms..."

# Alarm: Low online node count
run_aws cloudwatch put-metric-alarm \
  --alarm-name "RingRift-LowOnlineNodes" \
  --alarm-description "Fewer than 5 cluster nodes online" \
  --metric-name "OnlineNodes" \
  --namespace "RingRift/Cluster" \
  --statistic Average \
  --period 300 \
  --threshold 5 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --dimensions Name=Environment,Value=production \
  --region "$AWS_REGION"
log "  Created: RingRift-LowOnlineNodes"

# Alarm: Quorum lost
run_aws cloudwatch put-metric-alarm \
  --alarm-name "RingRift-QuorumLost" \
  --alarm-description "Cluster quorum lost" \
  --metric-name "QuorumHealthy" \
  --namespace "RingRift/Cluster" \
  --statistic Minimum \
  --period 60 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --dimensions Name=Environment,Value=production \
  --region "$AWS_REGION"
log "  Created: RingRift-QuorumLost"

# Alarm: Low selfplay throughput
run_aws cloudwatch put-metric-alarm \
  --alarm-name "RingRift-LowSelfplayThroughput" \
  --alarm-description "Selfplay throughput below 20 games/second" \
  --metric-name "GamesPerSecond" \
  --namespace "RingRift/Selfplay" \
  --statistic Average \
  --period 300 \
  --threshold 20 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 3 \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --dimensions Name=Environment,Value=production \
  --region "$AWS_REGION"
log "  Created: RingRift-LowSelfplayThroughput"

# Alarm: Training job failures
run_aws cloudwatch put-metric-alarm \
  --alarm-name "RingRift-TrainingFailures" \
  --alarm-description "Training jobs failing" \
  --metric-name "FailedTrainingJobs" \
  --namespace "RingRift/Training" \
  --statistic Sum \
  --period 3600 \
  --threshold 3 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions "$SNS_TOPIC_ARN" \
  --region "$AWS_REGION"
log "  Created: RingRift-TrainingFailures"

# ============================================================================
# 4. Create CloudWatch Dashboard
# ============================================================================
log "Creating CloudWatch dashboard..."

DASHBOARD_BODY=$(cat <<'EOF'
{
  "widgets": [
    {
      "type": "metric",
      "x": 0,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Cluster Node Status",
        "metrics": [
          ["RingRift/Cluster", "OnlineNodes", "Environment", "production", {"color": "#2ca02c"}],
          ["RingRift/Cluster", "OfflineNodes", "Environment", "production", {"color": "#d62728"}]
        ],
        "view": "timeSeries",
        "stacked": false,
        "period": 300,
        "stat": "Average",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 0,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "GH200 Cluster Status",
        "metrics": [
          ["RingRift/Cluster", "GH200NodesOnline", "Environment", "production"]
        ],
        "view": "timeSeries",
        "period": 300,
        "stat": "Average",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Selfplay Throughput (games/sec)",
        "metrics": [
          ["RingRift/Selfplay", "GamesPerSecond", "Environment", "production", {"color": "#1f77b4"}]
        ],
        "view": "timeSeries",
        "period": 60,
        "stat": "Average",
        "region": "us-east-1",
        "annotations": {
          "horizontal": [
            {"value": 30, "label": "Min Threshold", "color": "#ff7f0e"}
          ]
        }
      }
    },
    {
      "type": "metric",
      "x": 12,
      "y": 6,
      "width": 12,
      "height": 6,
      "properties": {
        "title": "Active Jobs",
        "metrics": [
          ["RingRift/Cluster", "SelfplayJobs", "Environment", "production", {"label": "Selfplay"}],
          ["RingRift/Cluster", "TrainingJobs", "Environment", "production", {"label": "Training"}]
        ],
        "view": "timeSeries",
        "period": 300,
        "stat": "Average",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 0,
      "y": 12,
      "width": 8,
      "height": 6,
      "properties": {
        "title": "Training Pipeline",
        "metrics": [
          ["RingRift/Training", "ActiveTrainingJobs", {"label": "Active"}],
          ["RingRift/Training", "CompletedTrainingJobs", {"label": "Completed"}],
          ["RingRift/Training", "FailedTrainingJobs", {"label": "Failed", "color": "#d62728"}]
        ],
        "view": "timeSeries",
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 8,
      "y": 12,
      "width": 8,
      "height": 6,
      "properties": {
        "title": "Elo Tournament Progress",
        "metrics": [
          ["RingRift/Training", "EloGamesPlayed"],
          ["RingRift/Training", "TopModelElo"]
        ],
        "view": "timeSeries",
        "period": 600,
        "stat": "Maximum",
        "region": "us-east-1"
      }
    },
    {
      "type": "metric",
      "x": 16,
      "y": 12,
      "width": 8,
      "height": 6,
      "properties": {
        "title": "Model Gating",
        "metrics": [
          ["RingRift/Training", "PassedGates", {"color": "#2ca02c"}],
          ["RingRift/Training", "FailedGates", {"color": "#d62728"}],
          ["RingRift/Training", "PendingGates", {"color": "#ff7f0e"}]
        ],
        "view": "timeSeries",
        "period": 3600,
        "stat": "Sum",
        "region": "us-east-1"
      }
    },
    {
      "type": "alarm",
      "x": 0,
      "y": 18,
      "width": 24,
      "height": 3,
      "properties": {
        "title": "Active Alarms",
        "alarms": [
          "arn:aws:cloudwatch:us-east-1:767371459652:alarm:RingRift-LowOnlineNodes",
          "arn:aws:cloudwatch:us-east-1:767371459652:alarm:RingRift-QuorumLost",
          "arn:aws:cloudwatch:us-east-1:767371459652:alarm:RingRift-LowSelfplayThroughput",
          "arn:aws:cloudwatch:us-east-1:767371459652:alarm:RingRift-TrainingFailures"
        ]
      }
    }
  ]
}
EOF
)

run_aws cloudwatch put-dashboard \
  --dashboard-name "$DASHBOARD_NAME" \
  --dashboard-body "$DASHBOARD_BODY" \
  --region "$AWS_REGION"

log "  Created dashboard: $DASHBOARD_NAME"

# ============================================================================
# 5. Summary
# ============================================================================
log ""
log "============================================================================"
log "CloudWatch setup complete!"
log "============================================================================"
log ""
log "Resources created:"
log "  - SNS Topic: $SNS_TOPIC_ARN"
log "  - Log Groups: ${LOG_GROUPS[*]}"
log "  - Alarms: LowOnlineNodes, QuorumLost, LowSelfplayThroughput, TrainingFailures"
log "  - Dashboard: https://${AWS_REGION}.console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#dashboards:name=${DASHBOARD_NAME}"
log ""
log "Next steps:"
log "  1. Confirm email subscription (check inbox for $ALERT_EMAIL)"
log "  2. Set up cron jobs for monitoring scripts"
log "  3. Configure RINGRIFT_SLACK_WEBHOOK environment variable"
log ""

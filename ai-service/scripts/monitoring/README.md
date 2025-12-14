# RingRift Cluster Monitoring

This directory contains monitoring scripts for the RingRift distributed training cluster.

## Quick Start

```bash
# 1. Set up CloudWatch (one-time)
./setup_cloudwatch.sh --email your@email.com

# 2. Test the health check
./cluster_health_check.sh --verbose --dry-run

# 3. Configure environment variables
export RINGRIFT_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export RINGRIFT_CLUSTER_API="https://cluster.ringrift.ai"

# 4. Add to crontab
crontab -e
```

## Scripts

### cluster_health_check.sh

Monitors cluster node status, detects offline critical nodes, and sends alerts.

**Crontab (every 5 minutes):**

```
*/5 * * * * /opt/ringrift/monitoring/cluster_health_check.sh >> /var/log/ringrift-monitor.log 2>&1
```

**Alerts on:**

- Critical nodes offline (lambda-h100, lambda-2xh100, aws-selfplay, aws-staging)
- GH200 cluster degraded (< 4 nodes online)
- Quorum lost
- High CPU/memory utilization (> 95% CPU or > 90% memory)
- Disk pressure (> 85%)

### selfplay_throughput_monitor.sh

Monitors selfplay game generation throughput.

**Crontab (every 2 minutes):**

```
*/2 * * * * /opt/ringrift/monitoring/selfplay_throughput_monitor.sh >> /var/log/ringrift-selfplay.log 2>&1
```

**Alerts on:**

- Throughput below threshold (default: 30 g/s)
- Stalled board types

### training_pipeline_monitor.sh

Monitors NNUE training, Elo tournaments, and model gating.

**Crontab (every 10 minutes):**

```
*/10 * * * * /opt/ringrift/monitoring/training_pipeline_monitor.sh >> /var/log/ringrift-training.log 2>&1
```

**Alerts on:**

- Training job failures
- New models passing gating

### setup_cloudwatch.sh

One-time setup for CloudWatch alarms, dashboard, and log groups.

```bash
./setup_cloudwatch.sh --email alerts@ringrift.ai
```

## Environment Variables

| Variable                      | Description                   | Default                       |
| ----------------------------- | ----------------------------- | ----------------------------- |
| `RINGRIFT_SLACK_WEBHOOK`      | Slack webhook URL for alerts  | (none)                        |
| `RINGRIFT_CLUSTER_API`        | Cluster API endpoint          | `https://cluster.ringrift.ai` |
| `RINGRIFT_MIN_THROUGHPUT`     | Min acceptable games/sec      | `30`                          |
| `RINGRIFT_ALERT_COOLDOWN_MIN` | Minutes between repeat alerts | `30`                          |
| `AWS_REGION`                  | AWS region for CloudWatch     | `us-east-1`                   |

## CloudWatch Dashboard

After running `setup_cloudwatch.sh`, access the dashboard at:
https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=RingRift-Cluster

### Widgets

- **Cluster Node Status**: Online vs offline nodes over time
- **GH200 Cluster Status**: GH200 node availability
- **Selfplay Throughput**: Games/second with threshold line
- **Active Jobs**: Selfplay and training job counts
- **Training Pipeline**: Active, completed, and failed training jobs
- **Elo Tournament Progress**: Games played and top model Elo
- **Model Gating**: Passed, failed, and pending gates
- **Active Alarms**: Current alarm states

## Quick Health Check Commands

```bash
# Check cluster status
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '{
  online: [.peers[] | select(.status=="online") | .node_id],
  offline: [.peers[] | select(.status=="offline") | .node_id] | length,
  leader: .leader_id,
  quorum_ok: .voter_quorum_ok
}'

# Check selfplay throughput
curl -s https://cluster.ringrift.ai/api/selfplay/stats | jq '.total_games_per_second'

# Check GH200 nodes
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '[.peers[] | select(.node_id | startswith("lambda-gh200")) | {id: .node_id, status: .status, gpu: .gpu_percent, jobs: .selfplay_jobs}]'

# Check training status
curl -s https://cluster.ringrift.ai/api/training/status | jq '.'
```

## Troubleshooting

### Node Offline

```bash
# Check node via Tailscale
tailscale status | grep <node-id>

# SSH and check orchestrator
ssh <node-id>
systemctl status ringrift-orchestrator
journalctl -u ringrift-orchestrator -n 50

# Restart orchestrator
sudo systemctl restart ringrift-orchestrator
```

### Low Throughput

```bash
# Check for stuck jobs
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '.peers[] | select(.selfplay_jobs > 100) | {id: .node_id, jobs: .selfplay_jobs}'

# Check for disk pressure
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '.peers[] | select(.disk_percent > 80) | {id: .node_id, disk: .disk_percent}'
```

### Quorum Lost

```bash
# Check voter nodes
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '{voters: .voter_node_ids, alive: .voters_alive, required: .voter_quorum_size}'

# Check leader election
curl -s https://cluster.ringrift.ai/api/cluster/status | jq '{leader: .leader_id, effective_leader: .effective_leader_id}'
```

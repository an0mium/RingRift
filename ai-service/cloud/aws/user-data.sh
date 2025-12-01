#!/bin/bash
# AWS EC2 User Data Script for Self-Play Workers
# This script runs when an EC2 instance starts (or spot instance is fulfilled)

set -e

# Configuration from instance tags/environment
BUCKET="${S3_BUCKET:-ringrift-training-data}"
BOARD_TYPE="${BOARD_TYPE:-square8}"
GAMES="${NUM_GAMES:-10000}"
SEED="${BASE_SEED:-42}"
WORKER_NUM="${WORKER_NUM:-0}"

# Generate unique worker ID from instance ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
WORKER_ID="${INSTANCE_ID}-${WORKER_NUM}"

echo "Starting RingRift self-play worker: ${WORKER_ID}"
echo "Board type: ${BOARD_TYPE}"
echo "Games: ${GAMES}"
echo "Output: s3://${BUCKET}/selfplay/${BOARD_TYPE}/"

# Install dependencies
yum update -y
yum install -y python3 python3-pip git

# Clone repository (or pull from S3)
cd /opt
if [ -d "ringrift-ai" ]; then
    cd ringrift-ai && git pull
else
    git clone https://github.com/YOUR_ORG/ringrift-ai.git ringrift-ai
    cd ringrift-ai
fi

# Install Python dependencies
pip3 install -r ai-service/requirements.txt
pip3 install boto3  # For S3 uploads

# Set up environment
export WORKER_ID="${WORKER_ID}"
export AWS_DEFAULT_REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

# Run self-play worker
cd ai-service
python3 scripts/run_distributed_selfplay.py \
    --num-games "${GAMES}" \
    --board-type "${BOARD_TYPE}" \
    --seed $((SEED + WORKER_NUM * 100000)) \
    --output "s3://${BUCKET}/selfplay/${BOARD_TYPE}" \
    --checkpoint-interval 500 \
    --checkpoint-path "/tmp/checkpoint_${WORKER_ID}.json" \
    --gc-interval 50 \
    --worker-id "${WORKER_ID}" \
    2>&1 | tee /var/log/selfplay.log

# Upload final stats and log to S3
aws s3 cp /var/log/selfplay.log "s3://${BUCKET}/logs/${WORKER_ID}.log"
aws s3 cp "/tmp/worker_${WORKER_ID}_stats.json" "s3://${BUCKET}/stats/${WORKER_ID}_stats.json"

echo "Worker ${WORKER_ID} completed"

# Self-terminate spot instance when done (optional)
# aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}"

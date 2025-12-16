#!/bin/bash
# Disk cleanup script for RingRift AI service
# Removes old model checkpoints when disk usage exceeds threshold

THRESHOLD=80  # Trigger cleanup at 80% disk usage
MODELS_DIR="${RINGRIFT_PATH:-/home/ubuntu/ringrift}/ai-service/models"
LOGS_DIR="${RINGRIFT_PATH:-/home/ubuntu/ringrift}/ai-service/logs"

# Get current disk usage percentage
DISK_USAGE=$(df / | tail -1 | awk '{print int($5)}')

echo "[$(date)] Disk usage: ${DISK_USAGE}%"

if [ "$DISK_USAGE" -ge "$THRESHOLD" ]; then
    echo "[$(date)] Disk usage exceeds ${THRESHOLD}%, starting cleanup..."

    # Remove timestamped model checkpoints (keep only named baselines)
    # Pattern: *_YYYYMM*_YYYYMM*.pth (timestamped checkpoints)
    if [ -d "$MODELS_DIR" ]; then
        BEFORE=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
        find "$MODELS_DIR" -name "*_202[0-9][0-9][0-9]*_202[0-9][0-9][0-9]*.pth" -mtime +1 -delete 2>/dev/null
        AFTER=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
        echo "[$(date)] Models: $BEFORE -> $AFTER"
    fi

    # Clean up old log files (keep last 7 days)
    if [ -d "$LOGS_DIR" ]; then
        find "$LOGS_DIR" -name "*.log" -mtime +7 -delete 2>/dev/null
        find "$LOGS_DIR" -name "*.log.*" -mtime +7 -delete 2>/dev/null
        echo "[$(date)] Cleaned logs older than 7 days"
    fi

    # Clean up stale ai-service directories (common on Vast instances)
    if [ -d "/root/ai-service" ] && [ -d "/root/ringrift/ai-service" ]; then
        echo "[$(date)] Removing stale /root/ai-service directory..."
        rm -rf /root/ai-service
    fi

    # Final disk usage
    DISK_USAGE_AFTER=$(df / | tail -1 | awk '{print int($5)}')
    echo "[$(date)] Cleanup complete. Disk usage: ${DISK_USAGE_AFTER}%"
else
    echo "[$(date)] Disk usage OK, no cleanup needed"
fi

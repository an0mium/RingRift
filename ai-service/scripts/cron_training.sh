#!/bin/bash
# Cron wrapper for automated NNUE training pipeline
#
# To install, add to crontab:
#   crontab -e
#   # Daily at 4 AM:
#   0 4 * * * /path/to/ai-service/scripts/cron_training.sh
#
# Or weekly on Sundays:
#   0 4 * * 0 /path/to/ai-service/scripts/cron_training.sh

set -e

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$AI_SERVICE_DIR"

# Create log directory
mkdir -p logs

# Log file with timestamp
LOG_FILE="logs/cron_training_$(date +%Y%m%d_%H%M%S).log"

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

echo "Starting automated training pipeline at $(date)" >> "$LOG_FILE"
echo "Working directory: $AI_SERVICE_DIR" >> "$LOG_FILE"

# Run the training pipeline
PYTHONPATH="$AI_SERVICE_DIR" python3 scripts/auto_training_pipeline.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Training pipeline finished at $(date) with exit code $EXIT_CODE" >> "$LOG_FILE"

# Keep only last 30 days of logs
find logs -name "cron_training_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE

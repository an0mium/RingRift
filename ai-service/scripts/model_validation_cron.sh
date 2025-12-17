#!/bin/bash
# Model Validation Cron Job
# Runs periodic model validation and cleanup
#
# Add to crontab with:
#   crontab -e
#   # Run every 6 hours
#   0 */6 * * * /path/to/ai-service/scripts/model_validation_cron.sh >> /path/to/ai-service/logs/model_validation.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$AI_SERVICE_ROOT"

echo "=========================================="
echo "Model Validation Cron Job - $(date)"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "$AI_SERVICE_ROOT/.venv/bin/activate" ]; then
    source "$AI_SERVICE_ROOT/.venv/bin/activate"
fi

# Run validation with cleanup and database update
python scripts/validate_models.py --cleanup --update-db

echo ""
echo "Validation completed at $(date)"
echo "=========================================="

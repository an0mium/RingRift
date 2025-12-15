#!/bin/bash
# Cron script for running diverse selfplay on priority configs
# Add to crontab: 0 */4 * * * /path/to/cron_diverse_selfplay.sh >> /var/log/diverse_selfplay_cron.log 2>&1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_SERVICE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$AI_SERVICE_DIR"

# Activate venv if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Configuration
GAMES_PER_MATCHUP=${GAMES_PER_MATCHUP:-50}
LOG_DIR="$AI_SERVICE_DIR/logs/diverse_selfplay"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/cron_$TIMESTAMP.log"

echo "=== Diverse Selfplay Cron Run: $(date) ===" | tee -a "$LOG_FILE"

# Check if another instance is running
if pgrep -f "run_diverse_selfplay.py" > /dev/null; then
    echo "Diverse selfplay already running, skipping..." | tee -a "$LOG_FILE"
    exit 0
fi

# Get current model counts to determine priority
# Priority configs: hexagonal_2p, hexagonal_4p (fewest models)
PRIORITY_CONFIGS=("hexagonal_2p" "hexagonal_4p" "square19_3p")

# Run diverse selfplay for each priority config
for config in "${PRIORITY_CONFIGS[@]}"; do
    echo "Starting diverse selfplay for $config..." | tee -a "$LOG_FILE"

    python3 scripts/run_diverse_selfplay.py \
        --config "$config" \
        --games-per-matchup "$GAMES_PER_MATCHUP" \
        >> "$LOG_FILE" 2>&1 || {
        echo "ERROR: Failed to run diverse selfplay for $config" | tee -a "$LOG_FILE"
        continue
    }

    echo "Completed diverse selfplay for $config" | tee -a "$LOG_FILE"
done

echo "=== Diverse Selfplay Cron Complete: $(date) ===" | tee -a "$LOG_FILE"

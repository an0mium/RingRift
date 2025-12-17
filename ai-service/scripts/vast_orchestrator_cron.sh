#!/bin/bash
# Vast.ai instance orchestrator cron job
# Runs every 30 minutes to:
# 1. Health check all vast instances
# 2. Sync game data from instances with games
# 3. Restart workers on idle/stuck instances

cd /Users/armand/Development/RingRift/ai-service
source venv/bin/activate 2>/dev/null || true

# Run orchestrator with timeout (max 15 minutes)
timeout 900 python -B scripts/vast_lifecycle.py --auto 2>&1 >> logs/vast_orchestrator_cron.log

#!/usr/bin/env python3
"""Continuous Training Runner - Convenience CLI for continuous training loop.

This script provides a simple interface for running the continuous training loop
without needing to go through the daemon manager.

Usage:
    # Run with defaults (hex8_2p and square8_2p)
    python scripts/run_continuous_training.py

    # Custom configs
    python scripts/run_continuous_training.py --config hex8:2 --config square8:4

    # Quick test (10 games, 1 iteration)
    python scripts/run_continuous_training.py --config hex8:2 --games 10 --max-iterations 1

    # Full production run
    python scripts/run_continuous_training.py \
        --config hex8:2 --config square8:2 --config square8:4 \
        --games 1000 --engine gumbel-mcts

Pipeline Flow:
    For each iteration:
    1. Selfplay generates training data
    2. Pipeline auto-triggers: sync -> export -> train -> evaluate -> promote
    3. Cooldown before next iteration

Environment Variables:
    RINGRIFT_CONTINUOUS_GAMES: Default games per iteration (default: 1000)
    RINGRIFT_CONTINUOUS_ENGINE: Default engine (default: gumbel-mcts)
    RINGRIFT_CONTINUOUS_COOLDOWN: Seconds between iterations (default: 60)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Delegate to the continuous loop module
from app.coordination.continuous_loop import main

if __name__ == "__main__":
    # Set default environment variables if not already set
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

    main()

#!/usr/bin/env python3
"""
Neural network training baseline script.

This is a wrapper script that forwards to the main training module.
Preserved for backward compatibility with existing docs and configs.

Usage:
    python scripts/run_nn_training_baseline.py --board square8 --num-players 2
    python scripts/run_nn_training_baseline.py --config config/training.yaml

For full options, see: python -m app.training.train --help
"""

import sys
import subprocess
from pathlib import Path

# Ensure we're running from the ai-service directory
AI_SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))


def main():
    """Forward all arguments to the training module."""
    # Build command: python -m app.training.train [args...]
    cmd = [sys.executable, "-m", "app.training.train"] + sys.argv[1:]

    # Run the training module
    result = subprocess.run(cmd, cwd=AI_SERVICE_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

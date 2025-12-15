#!/usr/bin/env python3
"""Training Loop Monitor - Tracks health and progress of the unified AI loop.

This script monitors the training loop and reports on:
- Training runs and model production
- Data collection rates
- Model promotions
- System health

Usage:
    # One-shot status check
    python scripts/training_monitor.py

    # Continuous monitoring (for cron)
    python scripts/training_monitor.py --log

    # Detailed report
    python scripts/training_monitor.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Monitor] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def load_unified_state() -> Optional[Dict[str, Any]]:
    """Load the unified loop state file."""
    state_path = AI_SERVICE_ROOT / "logs" / "unified_loop" / "unified_loop_state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return None


def check_recent_models(hours: int = 24) -> List[Tuple[str, datetime, int]]:
    """Find models created in the last N hours."""
    models_dir = AI_SERVICE_ROOT / "models"
    cutoff = time.time() - (hours * 3600)
    recent = []

    for pattern in ["*.pt", "*.pth"]:
        for model_path in models_dir.glob(pattern):
            if model_path.stat().st_mtime > cutoff:
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                size = model_path.stat().st_size
                recent.append((model_path.name, mtime, size))

    return sorted(recent, key=lambda x: x[1], reverse=True)


def check_db_health(db_path: Path) -> Tuple[bool, str, int]:
    """Check if a SQLite database is healthy."""
    if not db_path.exists():
        return False, "File not found", 0

    if db_path.stat().st_size == 0:
        return False, "Empty file", 0

    try:
        conn = sqlite3.connect(str(db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()
        if result[0] != "ok":
            conn.close()
            return False, f"Integrity check failed: {result[0]}", 0

        # Try to count games
        try:
            count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
        except:
            count = 0

        conn.close()
        return True, "OK", count
    except Exception as e:
        return False, str(e), 0


def check_gpu_processes() -> List[Dict[str, Any]]:
    """Check running GPU processes."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []

        processes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                if len(parts) >= 3:
                    processes.append({
                        "pid": parts[0],
                        "name": parts[1],
                        "memory_mb": int(parts[2]) if parts[2].isdigit() else 0
                    })
        return processes
    except Exception:
        return []


def check_gpu_utilization() -> Optional[int]:
    """Get current GPU utilization percentage."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split()[0])
    except Exception:
        pass
    return None


def generate_report(verbose: bool = False) -> Dict[str, Any]:
    """Generate a comprehensive training status report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "issues": [],
        "metrics": {}
    }

    # Load state
    state = load_unified_state()
    if not state:
        report["status"] = "error"
        report["issues"].append("Could not load unified loop state")
        return report

    # Basic metrics
    report["metrics"]["total_training_runs"] = state.get("total_training_runs", 0)
    report["metrics"]["total_promotions"] = state.get("total_promotions", 0)
    report["metrics"]["total_data_syncs"] = state.get("total_data_syncs", 0)
    report["metrics"]["consecutive_failures"] = state.get("consecutive_failures", 0)
    report["metrics"]["training_in_progress"] = state.get("training_in_progress", False)

    # Check for issues
    if state.get("consecutive_failures", 0) > 3:
        report["issues"].append(f"High consecutive failures: {state['consecutive_failures']}")

    # Check recent models
    recent_models = check_recent_models(24)
    report["metrics"]["models_last_24h"] = len(recent_models)
    if len(recent_models) == 0:
        report["issues"].append("No new models in last 24 hours")

    # Check GPU
    gpu_util = check_gpu_utilization()
    if gpu_util is not None:
        report["metrics"]["gpu_utilization"] = gpu_util
        if gpu_util < 10:
            report["issues"].append(f"Low GPU utilization: {gpu_util}%")

    # Check key databases
    key_dbs = [
        "data/games/all_jsonl_training.db",
        "data/games/selfplay.db",
        "data/unified_elo.db",
    ]

    db_status = {}
    for db_rel_path in key_dbs:
        db_path = AI_SERVICE_ROOT / db_rel_path
        healthy, msg, count = check_db_health(db_path)
        db_status[db_rel_path] = {
            "healthy": healthy,
            "message": msg,
            "game_count": count
        }
        if not healthy:
            report["issues"].append(f"DB issue - {db_rel_path}: {msg}")

    report["metrics"]["databases"] = db_status

    # Config status
    configs_status = {}
    for config_key, config in state.get("configs", {}).items():
        configs_status[config_key] = {
            "games_since_training": config.get("games_since_training", 0),
            "current_elo": config.get("current_elo", 1500),
        }
    report["metrics"]["configs"] = configs_status

    # Determine overall status
    if len(report["issues"]) == 0:
        report["status"] = "healthy"
    elif len(report["issues"]) <= 2:
        report["status"] = "warning"
    else:
        report["status"] = "critical"

    return report


def main():
    parser = argparse.ArgumentParser(description="Training Loop Monitor")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--log", action="store_true", help="Log output for cron")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    report = generate_report(verbose=args.verbose)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    # Print summary
    status_emoji = {"healthy": "✓", "warning": "⚠", "critical": "✗", "unknown": "?"}
    print(f"\n{'='*60}")
    print(f"Training Loop Status: {status_emoji.get(report['status'], '?')} {report['status'].upper()}")
    print(f"{'='*60}")
    print(f"Timestamp: {report['timestamp']}")
    print()

    # Metrics
    m = report["metrics"]
    print("Metrics:")
    print(f"  Training runs: {m.get('total_training_runs', 'N/A')}")
    print(f"  Promotions: {m.get('total_promotions', 'N/A')}")
    print(f"  Data syncs: {m.get('total_data_syncs', 'N/A')}")
    print(f"  Models (24h): {m.get('models_last_24h', 'N/A')}")
    print(f"  GPU utilization: {m.get('gpu_utilization', 'N/A')}%")
    print(f"  Training in progress: {m.get('training_in_progress', 'N/A')}")
    print()

    # Issues
    if report["issues"]:
        print("Issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
        print()

    # Database status
    if args.verbose and "databases" in m:
        print("Databases:")
        for db_path, status in m["databases"].items():
            icon = "✓" if status["healthy"] else "✗"
            print(f"  {icon} {db_path}: {status['message']} ({status['game_count']} games)")
        print()

    # Config status
    if args.verbose and "configs" in m:
        print("Configs:")
        for config_key, status in m["configs"].items():
            print(f"  {config_key}: {status['games_since_training']} games pending, Elo={status['current_elo']:.0f}")
        print()


if __name__ == "__main__":
    main()

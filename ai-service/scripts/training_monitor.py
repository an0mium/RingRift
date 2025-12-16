#!/usr/bin/env python3
"""Training Loop Monitor - Tracks health and progress of the unified AI loop.

Unified monitoring and alerting for the training pipeline. This script:
- Monitors training runs and model production
- Checks data collection rates and database health
- Tracks model promotions and Elo progression
- Generates alerts for issues (critical/warning/info levels)
- Checks disk usage and system health

Usage:
    # One-shot status check
    python scripts/training_monitor.py

    # Detailed report with alerts
    python scripts/training_monitor.py --verbose

    # JSON output for automation
    python scripts/training_monitor.py --json

    # Cron mode (exit 1 if critical alerts)
    python scripts/training_monitor.py --cron

    # Save alerts to file
    python scripts/training_monitor.py --output alerts.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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

# Alert thresholds
THRESHOLDS = {
    "training_stale_hours": 24,           # Alert if no training for 24h
    "model_stale_hours": 48,              # Alert if no new model in 48h
    "disk_warning_percent": 70,           # Disk usage warning
    "disk_critical_percent": 85,          # Disk usage critical
    "consecutive_failures_warning": 3,    # Failures before warning
    "low_gpu_utilization": 10,            # GPU util % threshold
}

# Priority configs that need attention
PRIORITY_CONFIGS = ["hexagonal_2p", "hexagonal_4p", "square19_3p"]


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: str  # "info", "warning", "critical"
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


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


def check_disk_usage() -> Tuple[float, float]:
    """Check disk usage. Returns (percent_used, free_gb)."""
    data_dir = AI_SERVICE_ROOT / "data"
    try:
        total, used, free = shutil.disk_usage(str(data_dir))
        percent = (used / total) * 100
        return percent, free / (1024**3)
    except Exception:
        return 0.0, 0.0


def check_model_age() -> Tuple[Optional[float], Optional[str]]:
    """Check age of most recent model. Returns (hours_since, model_name)."""
    models_dir = AI_SERVICE_ROOT / "models"
    if not models_dir.exists():
        return None, None

    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        return None, None

    most_recent = max(model_files, key=lambda p: p.stat().st_mtime)
    hours_since = (time.time() - most_recent.stat().st_mtime) / 3600
    return hours_since, most_recent.name


def generate_report(verbose: bool = False) -> Dict[str, Any]:
    """Generate a comprehensive training status report with alerts."""
    alerts: List[Alert] = []
    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "alerts": [],
        "metrics": {}
    }

    # Load state
    state = load_unified_state()
    if not state:
        alerts.append(Alert("critical", "training_loop", "State file not found - loop may not be running"))
    else:
        # Basic metrics
        report["metrics"]["total_training_runs"] = state.get("total_training_runs", 0)
        report["metrics"]["total_promotions"] = state.get("total_promotions", 0)
        report["metrics"]["total_data_syncs"] = state.get("total_data_syncs", 0)
        report["metrics"]["consecutive_failures"] = state.get("consecutive_failures", 0)
        report["metrics"]["training_in_progress"] = state.get("training_in_progress", False)

        # Check consecutive failures
        failures = state.get("consecutive_failures", 0)
        if failures >= THRESHOLDS["consecutive_failures_warning"]:
            alerts.append(Alert("warning", "training_loop", f"High consecutive failures: {failures}"))

        # Check training staleness per config
        now = time.time()
        for config_name, config_data in state.get("configs", {}).items():
            last_training = config_data.get("last_training_time", 0)
            hours_since = (now - last_training) / 3600 if last_training > 0 else float('inf')

            if config_name in PRIORITY_CONFIGS and hours_since > THRESHOLDS["training_stale_hours"]:
                alerts.append(Alert(
                    "warning", "training_stale",
                    f"Priority config {config_name} hasn't trained in {hours_since:.1f}h",
                    {"config": config_name, "hours_since": hours_since}
                ))

        # Config status
        configs_status = {}
        for config_key, config in state.get("configs", {}).items():
            configs_status[config_key] = {
                "games_since_training": config.get("games_since_training", 0),
                "current_elo": config.get("current_elo", 1500),
            }
        report["metrics"]["configs"] = configs_status

    # Check recent models
    recent_models = check_recent_models(24)
    report["metrics"]["models_last_24h"] = len(recent_models)

    # Check model age
    model_hours, model_name = check_model_age()
    if model_hours is not None:
        report["metrics"]["model_age_hours"] = round(model_hours, 1)
        if model_hours > THRESHOLDS["model_stale_hours"]:
            alerts.append(Alert(
                "warning", "model_stale",
                f"No new models in {model_hours:.1f}h (last: {model_name})",
                {"hours_since": model_hours, "last_model": model_name}
            ))
    elif model_name is None:
        alerts.append(Alert("warning", "models", "No model files found"))

    # Check GPU
    gpu_util = check_gpu_utilization()
    if gpu_util is not None:
        report["metrics"]["gpu_utilization"] = gpu_util
        if gpu_util < THRESHOLDS["low_gpu_utilization"]:
            alerts.append(Alert("warning", "gpu", f"Low GPU utilization: {gpu_util}%"))

    # Check disk usage
    disk_percent, disk_free_gb = check_disk_usage()
    report["metrics"]["disk_percent"] = round(disk_percent, 1)
    report["metrics"]["disk_free_gb"] = round(disk_free_gb, 1)
    if disk_percent >= THRESHOLDS["disk_critical_percent"]:
        alerts.append(Alert("critical", "disk_space", f"Disk usage critical: {disk_percent:.1f}%"))
    elif disk_percent >= THRESHOLDS["disk_warning_percent"]:
        alerts.append(Alert("warning", "disk_space", f"Disk usage high: {disk_percent:.1f}%"))

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
            alerts.append(Alert("warning", "database", f"DB issue - {db_rel_path}: {msg}"))

    report["metrics"]["databases"] = db_status

    # Convert alerts to serializable format
    report["alerts"] = [
        {"level": a.level, "category": a.category, "message": a.message, "details": a.details}
        for a in alerts
    ]

    # Count by level
    critical_count = sum(1 for a in alerts if a.level == "critical")
    warning_count = sum(1 for a in alerts if a.level == "warning")
    report["critical_count"] = critical_count
    report["warning_count"] = warning_count

    # Determine overall status
    if critical_count > 0:
        report["status"] = "critical"
    elif warning_count > 0:
        report["status"] = "warning"
    else:
        report["status"] = "healthy"

    return report


def main():
    parser = argparse.ArgumentParser(description="Training Loop Monitor and Alerting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--cron", action="store_true", help="Exit with code 1 if critical alerts")
    parser.add_argument("--output", type=str, help="Save report to JSON file")
    args = parser.parse_args()

    report = generate_report(verbose=args.verbose)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {args.output}")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
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
        print(f"  Disk usage: {m.get('disk_percent', 'N/A')}% ({m.get('disk_free_gb', 'N/A')}GB free)")
        print(f"  Training in progress: {m.get('training_in_progress', 'N/A')}")
        print()

        # Alerts
        alerts = report.get("alerts", [])
        if alerts:
            print(f"Alerts ({report['critical_count']} critical, {report['warning_count']} warning):")
            for alert in alerts:
                icon = {"critical": "X", "warning": "!", "info": "i"}.get(alert["level"], "?")
                print(f"  [{icon}] {alert['category']}: {alert['message']}")
            print()
        else:
            print("No alerts - pipeline healthy\n")

        # Database status (verbose)
        if args.verbose and "databases" in m:
            print("Databases:")
            for db_path, status in m["databases"].items():
                icon = "✓" if status["healthy"] else "✗"
                print(f"  {icon} {db_path}: {status['message']} ({status['game_count']} games)")
            print()

        # Config status (verbose)
        if args.verbose and "configs" in m:
            print("Configs:")
            for config_key, status in m["configs"].items():
                print(f"  {config_key}: {status['games_since_training']} games pending, Elo={status['current_elo']:.0f}")
            print()

    # Exit with error if critical alerts (for cron)
    if args.cron and report.get("critical_count", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

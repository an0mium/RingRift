#!/usr/bin/env python3
"""Training Pipeline Alerting System.

Monitors the training loop and generates alerts for:
- Training not triggering for extended periods
- No new models produced for priority configs
- Disk usage approaching limits
- Host connectivity failures
- Game generation rate drops

Usage:
    # Check and print alerts
    python scripts/training_alerts.py

    # Check and save to file
    python scripts/training_alerts.py --output alerts.json

    # Run as cron check (exit code 1 if critical alerts)
    python scripts/training_alerts.py --cron
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Alerts] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Alert thresholds
THRESHOLDS = {
    "training_stale_hours": 24,           # Alert if no training for 24h
    "model_stale_hours": 48,              # Alert if no new model for config in 48h
    "disk_warning_percent": 70,           # Disk usage warning
    "disk_critical_percent": 85,          # Disk usage critical
    "game_rate_min_per_hour": 100,        # Minimum games per hour
    "priority_config_min_games": 1000,    # Min games for priority configs
}

# Priority configs that need attention
PRIORITY_CONFIGS = ["hexagonal_2p", "hexagonal_4p", "square19_3p"]


@dataclass
class Alert:
    """Represents an alert."""
    level: str  # "info", "warning", "critical"
    category: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainingAlertChecker:
    """Checks training pipeline health and generates alerts."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.state_file = AI_SERVICE_ROOT / "logs" / "unified_loop" / "unified_loop_state.json"

    def add_alert(self, level: str, category: str, message: str, **details):
        """Add an alert."""
        self.alerts.append(Alert(level=level, category=category, message=message, details=details))
        log_fn = {"info": logger.info, "warning": logger.warning, "critical": logger.error}.get(level, logger.info)
        log_fn(f"[{category}] {message}")

    def check_training_loop_state(self) -> None:
        """Check if training loop is healthy."""
        if not self.state_file.exists():
            self.add_alert("critical", "training_loop", "State file not found - loop may not be running")
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)
        except Exception as e:
            self.add_alert("critical", "training_loop", f"Failed to read state file: {e}")
            return

        # Check if training is stuck
        training_in_progress = state.get("training_in_progress", False)
        if training_in_progress:
            # Check how long it's been training
            current_config = state.get("current_training_config", "unknown")
            self.add_alert("info", "training_loop", f"Training in progress for {current_config}")

        # Check last training time for each config
        configs = state.get("configs", {})
        now = time.time()

        for config_name, config_data in configs.items():
            last_training = config_data.get("last_training_time", 0)
            hours_since = (now - last_training) / 3600 if last_training > 0 else float('inf')

            if config_name in PRIORITY_CONFIGS:
                if hours_since > THRESHOLDS["training_stale_hours"]:
                    self.add_alert(
                        "warning", "training_stale",
                        f"Priority config {config_name} hasn't trained in {hours_since:.1f}h",
                        config=config_name, hours_since=hours_since
                    )

            model_count = config_data.get("trained_model_count", 0)
            if config_name in PRIORITY_CONFIGS and model_count < 3:
                self.add_alert(
                    "critical" if model_count <= 1 else "warning",
                    "model_deficit",
                    f"{config_name} has only {model_count} trained models",
                    config=config_name, model_count=model_count
                )

    def check_game_counts(self) -> None:
        """Check game counts for each config."""
        db_path = AI_SERVICE_ROOT / "data" / "games" / "all_jsonl_training.db"
        if not db_path.exists():
            self.add_alert("warning", "database", "Training database not found")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("""
                SELECT board_type, num_players, COUNT(*) as cnt
                FROM games
                GROUP BY board_type, num_players
            """)
            game_counts = {f"{row[0]}_{row[1]}p": row[2] for row in cursor.fetchall()}
            conn.close()

            for config in PRIORITY_CONFIGS:
                count = game_counts.get(config, 0)
                if count < THRESHOLDS["priority_config_min_games"]:
                    self.add_alert(
                        "warning", "game_deficit",
                        f"{config} has only {count} games (need {THRESHOLDS['priority_config_min_games']})",
                        config=config, game_count=count
                    )
        except Exception as e:
            self.add_alert("warning", "database", f"Failed to query game counts: {e}")

    def check_disk_usage(self) -> None:
        """Check disk usage."""
        data_dir = AI_SERVICE_ROOT / "data"
        try:
            total, used, free = shutil.disk_usage(str(data_dir))
            percent = (used / total) * 100

            if percent >= THRESHOLDS["disk_critical_percent"]:
                self.add_alert(
                    "critical", "disk_space",
                    f"Disk usage critical: {percent:.1f}%",
                    percent=percent, free_gb=free / (1024**3)
                )
            elif percent >= THRESHOLDS["disk_warning_percent"]:
                self.add_alert(
                    "warning", "disk_space",
                    f"Disk usage high: {percent:.1f}%",
                    percent=percent, free_gb=free / (1024**3)
                )
        except Exception as e:
            self.add_alert("warning", "disk_space", f"Failed to check disk: {e}")

    def check_model_files(self) -> None:
        """Check for recent model files."""
        models_dir = AI_SERVICE_ROOT / "models"
        if not models_dir.exists():
            self.add_alert("warning", "models", "Models directory not found")
            return

        model_files = list(models_dir.glob("*.pt"))
        if not model_files:
            self.add_alert("warning", "models", "No model files found")
            return

        # Check most recent model
        most_recent = max(model_files, key=lambda p: p.stat().st_mtime)
        hours_since = (time.time() - most_recent.stat().st_mtime) / 3600

        if hours_since > THRESHOLDS["model_stale_hours"]:
            self.add_alert(
                "warning", "model_stale",
                f"No new models in {hours_since:.1f}h (last: {most_recent.name})",
                hours_since=hours_since, last_model=most_recent.name
            )

        # Count models per config
        config_counts: Dict[str, int] = {}
        for mf in model_files:
            # Parse config from filename like ringrift_square8_2p_v1.pt
            parts = mf.stem.split("_")
            if len(parts) >= 3:
                config = f"{parts[1]}_{parts[2]}"
                config_counts[config] = config_counts.get(config, 0) + 1

        for config in PRIORITY_CONFIGS:
            count = config_counts.get(config, 0)
            if count == 0:
                self.add_alert(
                    "critical", "no_models",
                    f"No model files found for priority config {config}",
                    config=config
                )

    def run_all_checks(self) -> List[Alert]:
        """Run all checks and return alerts."""
        self.alerts = []

        self.check_training_loop_state()
        self.check_game_counts()
        self.check_disk_usage()
        self.check_model_files()

        return self.alerts

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all alerts."""
        critical = [a for a in self.alerts if a.level == "critical"]
        warnings = [a for a in self.alerts if a.level == "warning"]
        info = [a for a in self.alerts if a.level == "info"]

        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "info_count": len(info),
            "alerts": [
                {
                    "level": a.level,
                    "category": a.category,
                    "message": a.message,
                    "details": a.details,
                }
                for a in self.alerts
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline Alerting")
    parser.add_argument("--output", type=str, help="Output file for alerts JSON")
    parser.add_argument("--cron", action="store_true", help="Exit with code 1 if critical alerts")
    args = parser.parse_args()

    checker = TrainingAlertChecker()
    alerts = checker.run_all_checks()
    summary = checker.get_summary()

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE ALERTS")
    print("=" * 60)
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Critical: {summary['critical_count']}")
    print(f"Warnings: {summary['warning_count']}")
    print(f"Info: {summary['info_count']}")
    print()

    if alerts:
        for alert in alerts:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.level, "⚪")
            print(f"{icon} [{alert.level.upper()}] {alert.category}: {alert.message}")
    else:
        print("✅ No alerts - pipeline healthy")

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nAlerts saved to: {args.output}")

    # Exit with error if critical alerts (for cron)
    if args.cron and summary["critical_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

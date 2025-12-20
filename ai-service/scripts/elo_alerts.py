#!/usr/bin/env python3
"""ELO Alert System.

Monitors ELO database and sends alerts for significant events:
- New models reaching production threshold
- ELO regressions (top model drops significantly)
- Stale tournaments (no games in X hours)
- Config coverage warnings

Usage:
    python scripts/elo_alerts.py              # Run once
    python scripts/elo_alerts.py --daemon     # Run continuously
    python scripts/elo_alerts.py --test       # Send test alert
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
)

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"
STATE_FILE = AI_SERVICE_ROOT / "data" / ".elo_alert_state.json"

# Alert thresholds
REGRESSION_THRESHOLD = 50  # Alert if top model drops by 50 ELO
STALE_HOURS = 4  # Alert if no games in 4 hours
MIN_GAMES_PER_CONFIG = 100  # Alert if config has fewer games


def get_slack_webhook():
    """Get Slack webhook URL from environment or file."""
    webhook = os.environ.get("RINGRIFT_SLACK_WEBHOOK") or os.environ.get("SLACK_WEBHOOK_URL")
    if webhook:
        return webhook

    webhook_file = Path.home() / ".ringrift_slack_webhook"
    if webhook_file.exists():
        return webhook_file.read_text().strip()

    return None


def send_slack_alert(message: str, alert_type: str = "info"):
    """Send alert to Slack."""
    webhook = get_slack_webhook()

    emoji = {
        "info": ":information_source:",
        "success": ":white_check_mark:",
        "warning": ":warning:",
        "error": ":x:",
        "trophy": ":trophy:",
        "rocket": ":rocket:",
    }.get(alert_type, ":robot_face:")

    if webhook:
        payload = {
            "text": f"{emoji} *RingRift ELO Alert*\n{message}",
            "username": "ELO Monitor",
        }
        try:
            subprocess.run(
                ["curl", "-s", "-X", "POST", "-H", "Content-Type: application/json",
                 "-d", json.dumps(payload), webhook],
                capture_output=True, timeout=10
            )
            return True
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")
            return False
    else:
        print(f"[{alert_type.upper()}] {message}")
        return False


def load_state() -> dict:
    """Load previous alert state."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "last_check": None,
        "top_models": {},
        "production_models": [],
        "last_game_time": None,
    }


def save_state(state: dict):
    """Save alert state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def check_production_promotions(db_path: Path, state: dict) -> list[str]:
    """Check for new production-ready models."""
    alerts = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE rating >= ? AND games_played >= ?
          AND participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
    """, (PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES))

    current_production = []
    for row in cursor.fetchall():
        model_id, rating, games = row
        current_production.append(model_id)

        if model_id not in state.get("production_models", []):
            alerts.append(
                f":rocket: *New Production Model!*\n"
                f"  Model: `{model_id}`\n"
                f"  ELO: {rating:.1f} | Games: {games}"
            )

    conn.close()
    state["production_models"] = current_production
    return alerts


def check_regressions(db_path: Path, state: dict) -> list[str]:
    """Check for ELO regressions in top models."""
    alerts = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT participant_id, rating
        FROM elo_ratings
        WHERE participant_id NOT LIKE 'baseline_%'
        ORDER BY rating DESC
        LIMIT 5
    """)

    current_top = {}
    for row in cursor.fetchall():
        model_id, rating = row
        current_top[model_id] = rating

        old_rating = state.get("top_models", {}).get(model_id)
        if old_rating and old_rating - rating > REGRESSION_THRESHOLD:
            alerts.append(
                f":warning: *ELO Regression Detected*\n"
                f"  Model: `{model_id}`\n"
                f"  Was: {old_rating:.1f} | Now: {rating:.1f} ({rating - old_rating:+.1f})"
            )

    conn.close()
    state["top_models"] = current_top
    return alerts


def check_stale_tournaments(db_path: Path, state: dict) -> list[str]:
    """Check for stale tournament activity."""
    alerts = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT MAX(timestamp) FROM match_history
    """)

    result = cursor.fetchone()
    conn.close()

    if result and result[0]:
        try:
            # timestamp is stored as Unix epoch
            last_game = datetime.fromtimestamp(result[0])
            hours_since = (datetime.now() - last_game).total_seconds() / 3600

            if hours_since > STALE_HOURS:
                # Only alert once per stale period
                last_stale_alert = state.get("last_stale_alert")
                if not last_stale_alert or (datetime.now() - datetime.fromisoformat(last_stale_alert)).total_seconds() > 3600 * 6:
                    alerts.append(
                        f":warning: *Stale Tournaments*\n"
                        f"  No games recorded in {hours_since:.1f} hours\n"
                        f"  Last game: {last_game.strftime('%Y-%m-%d %H:%M') if last_game else 'Unknown'}"
                    )
                    state["last_stale_alert"] = datetime.now().isoformat()

        except Exception as e:
            print(f"Error parsing timestamp: {e}")

    return alerts


def check_config_coverage(db_path: Path) -> list[str]:
    """Check for underrepresented configs."""
    alerts = []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        SELECT board_type, num_players, COUNT(*) as games
        FROM match_history
        GROUP BY board_type, num_players
    """)

    coverage = {f"{r[0]}_{r[1]}p": r[2] for r in cursor.fetchall()}
    conn.close()

    all_configs = [
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hex8_2p", "hex8_3p", "hex8_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    low_coverage = []
    for config in all_configs:
        games = coverage.get(config, 0)
        if games < MIN_GAMES_PER_CONFIG:
            low_coverage.append(f"{config}: {games}")

    if len(low_coverage) > 6:  # More than half have low coverage
        alerts.append(
            f":warning: *Config Coverage Warning*\n"
            f"  {len(low_coverage)}/12 configs have < {MIN_GAMES_PER_CONFIG} games:\n"
            f"  {', '.join(low_coverage[:5])}..."
        )

    return alerts


def run_checks(db_path: Path) -> tuple[list[str], dict]:
    """Run all alert checks."""
    if not db_path.exists():
        return [":x: ELO database not found"], {}

    state = load_state()
    all_alerts = []

    # Run checks
    all_alerts.extend(check_production_promotions(db_path, state))
    all_alerts.extend(check_regressions(db_path, state))
    all_alerts.extend(check_stale_tournaments(db_path, state))

    # Only check coverage once per day
    last_coverage = state.get("last_coverage_check")
    if not last_coverage or (datetime.now() - datetime.fromisoformat(last_coverage)).days >= 1:
        all_alerts.extend(check_config_coverage(db_path))
        state["last_coverage_check"] = datetime.now().isoformat()

    state["last_check"] = datetime.now().isoformat()
    save_state(state)

    return all_alerts, state


def main():
    parser = argparse.ArgumentParser(description="ELO Alert System")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ELO database")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--test", action="store_true", help="Send test alert")

    args = parser.parse_args()

    if args.test:
        print("Sending test alert...")
        success = send_slack_alert(
            "This is a test alert from the ELO monitoring system.\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "If you see this, alerts are working!",
            "info"
        )
        if success:
            print("Test alert sent successfully!")
        else:
            print("Test alert sent to console (no webhook configured)")
        return

    if args.daemon:
        print(f"Starting ELO alert daemon (interval: {args.interval}s)")
        webhook = get_slack_webhook()
        if webhook:
            print(f"Slack webhook configured: {webhook[:50]}...")
        else:
            print("No Slack webhook configured - alerts will go to console")

        while True:
            try:
                alerts, _state = run_checks(args.db)
                for alert in alerts:
                    send_slack_alert(alert, "warning")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Checked - {len(alerts)} alerts")
            except Exception as e:
                print(f"Error in alert check: {e}")

            time.sleep(args.interval)
    else:
        alerts, state = run_checks(args.db)

        print(f"\n{'='*50}")
        print("ELO ALERT CHECK")
        print(f"{'='*50}")
        print(f"Last check: {state.get('last_check', 'Never')}")
        print(f"Production models: {len(state.get('production_models', []))}")
        print(f"Top models tracked: {len(state.get('top_models', {}))}")

        if alerts:
            print(f"\n{len(alerts)} Alert(s):")
            for alert in alerts:
                print(f"\n{alert}")
                send_slack_alert(alert, "warning")
        else:
            print("\nNo alerts.")


if __name__ == "__main__":
    main()

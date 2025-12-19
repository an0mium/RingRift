"""Unified Alerting Module.

Consolidates cluster_alert.sh and webhook logic into one module.
Uses webhooks from config/cluster.yaml.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime
from enum import Enum
from typing import Optional

from scripts.p2p.cluster_config import get_webhook_urls


class AlertLevel(Enum):
    """Alert severity level."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


LEVEL_COLORS = {
    AlertLevel.DEBUG: "#808080",    # Gray
    AlertLevel.INFO: "#36a64f",     # Green
    AlertLevel.WARNING: "#ff9800",  # Orange
    AlertLevel.ERROR: "#f44336",    # Red
    AlertLevel.CRITICAL: "#9c27b0", # Purple
}

LEVEL_EMOJI = {
    AlertLevel.DEBUG: ":bug:",
    AlertLevel.INFO: ":information_source:",
    AlertLevel.WARNING: ":warning:",
    AlertLevel.ERROR: ":x:",
    AlertLevel.CRITICAL: ":rotating_light:",
}


def send_slack_alert(
    webhook_url: str,
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    node_id: str = "",
) -> bool:
    """Send alert to Slack webhook."""
    payload = {
        "attachments": [
            {
                "color": LEVEL_COLORS[level],
                "title": title,
                "text": message,
                "footer": f"RingRift AI | {node_id}" if node_id else "RingRift AI",
                "ts": int(datetime.now().timestamp()),
            }
        ]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Slack alert failed: {e}")
        return False


def send_discord_alert(
    webhook_url: str,
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    node_id: str = "",
) -> bool:
    """Send alert to Discord webhook."""
    # Discord color is decimal, not hex
    color_hex = LEVEL_COLORS[level].lstrip("#")
    color_int = int(color_hex, 16)

    payload = {
        "embeds": [
            {
                "title": f"{LEVEL_EMOJI[level]} {title}",
                "description": message,
                "color": color_int,
                "footer": {"text": f"RingRift AI | {node_id}" if node_id else "RingRift AI"},
                "timestamp": datetime.now().isoformat(),
            }
        ]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"Discord alert failed: {e}")
        return False


def send_alert(
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    node_id: str = "",
    slack_url: Optional[str] = None,
    discord_url: Optional[str] = None,
) -> bool:
    """Send alert to configured webhooks.

    Args:
        title: Alert title.
        message: Alert message body.
        level: Severity level.
        node_id: Optional node identifier.
        slack_url: Override Slack webhook URL.
        discord_url: Override Discord webhook URL.

    Returns:
        True if at least one alert was sent successfully.
    """
    # Get webhooks from config if not provided
    webhooks = get_webhook_urls()
    slack = slack_url or webhooks.get("slack", "")
    discord = discord_url or webhooks.get("discord", "")

    if not slack and not discord:
        print("No webhooks configured - set in config/cluster.yaml or environment")
        return False

    success = False

    if slack:
        if send_slack_alert(slack, title, message, level, node_id):
            success = True

    if discord:
        if send_discord_alert(discord, title, message, level, node_id):
            success = True

    return success


def main():
    """CLI entry point."""
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="Send Cluster Alert")
    parser.add_argument("message", help="Alert message")
    parser.add_argument("--title", default="Cluster Alert", help="Alert title")
    parser.add_argument(
        "--level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Alert level",
    )
    parser.add_argument("--node", default=socket.gethostname(), help="Node identifier")
    args = parser.parse_args()

    level = AlertLevel(args.level)
    success = send_alert(args.title, args.message, level, args.node)

    if success:
        print(f"Alert sent: {args.title}")
    else:
        print("Failed to send alert")
        exit(1)


if __name__ == "__main__":
    main()

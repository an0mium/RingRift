"""Unified Alerting Module.

.. deprecated:: 2025-12
    This module re-exports from the canonical alert infrastructure.
    Import directly from scripts.lib.alerts instead::

        from scripts.lib.alerts import (
            AlertSeverity,
            send_alert,
            send_slack_notification,
            send_discord_notification,
        )

This module is maintained for backwards compatibility with existing scripts.
All functionality has been consolidated into scripts/lib/alerts.py.
"""

from __future__ import annotations

import warnings

# Re-export from canonical location
from scripts.lib.alerts import (
    AlertSeverity,
    DISCORD_COLORS,
    SLACK_EMOJI,
    discord_handler,
    get_discord_webhook,
    get_slack_webhook,
    send_alert,
    send_discord_notification,
    send_slack_notification,
    slack_handler,
)

# Backwards compatibility alias
AlertLevel = AlertSeverity

# Deprecated color mappings - use DISCORD_COLORS instead
LEVEL_COLORS = {
    AlertSeverity.DEBUG: "#808080",
    AlertSeverity.INFO: "#36a64f",
    AlertSeverity.WARNING: "#ff9800",
    AlertSeverity.ERROR: "#f44336",
    AlertSeverity.CRITICAL: "#9c27b0",
}

LEVEL_EMOJI = SLACK_EMOJI  # Use canonical emoji mapping


def _deprecated_send_slack_alert(
    webhook_url: str,
    title: str,
    message: str,
    level: AlertSeverity = AlertSeverity.INFO,
    node_id: str = "",
) -> bool:
    """Deprecated: Use send_slack_notification instead."""
    warnings.warn(
        "send_slack_alert is deprecated, use send_slack_notification from scripts.lib.alerts",
        DeprecationWarning,
        stacklevel=2,
    )
    return send_slack_notification(
        message=message,
        severity=level,
        title=title,
        webhook_url=webhook_url,
    )


def _deprecated_send_discord_alert(
    webhook_url: str,
    title: str,
    message: str,
    level: AlertSeverity = AlertSeverity.INFO,
    node_id: str = "",
) -> bool:
    """Deprecated: Use send_discord_notification instead."""
    warnings.warn(
        "send_discord_alert is deprecated, use send_discord_notification from scripts.lib.alerts",
        DeprecationWarning,
        stacklevel=2,
    )
    return send_discord_notification(
        message=message,
        severity=level,
        title=title,
        webhook_url=webhook_url,
        node_id=node_id,
    )


# Keep old function names for backwards compatibility
send_slack_alert = _deprecated_send_slack_alert
send_discord_alert = _deprecated_send_discord_alert


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
    success = send_alert(
        message=args.message,
        severity=level,
        title=args.title,
        node_id=args.node,
    )

    if success:
        print(f"Alert sent: {args.title}")
    else:
        print("No webhooks configured or alert failed")
        exit(1)


if __name__ == "__main__":
    main()

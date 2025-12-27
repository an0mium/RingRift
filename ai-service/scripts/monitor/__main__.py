"""Monitor Module CLI Entry Point.

Usage:
    python -m scripts.monitor status       # Show cluster status
    python -m scripts.monitor health       # Run health checks
    python -m scripts.monitor metrics      # Smoke check health metrics
    python -m scripts.monitor alert MSG    # Send an alert
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.monitor <command>")
        print("Commands:")
        print("  status  - Show cluster status dashboard")
        print("  health  - Run cluster health checks")
        print("  metrics - Smoke check health metrics")
        print("  alert   - Send an alert message")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Shift args for subcommand

    if command == "status":
        from .dashboard import main as dashboard_main
        dashboard_main()
    elif command == "health":
        from .health import main as health_main
        health_main()
    elif command == "metrics":
        from .health_metrics_smoke import main as metrics_main
        metrics_main()
    elif command == "alert":
        from .alerting import main as alert_main
        alert_main()
    else:
        print(f"Unknown command: {command}")
        print("Valid commands: status, health, alert")
        sys.exit(1)


if __name__ == "__main__":
    main()

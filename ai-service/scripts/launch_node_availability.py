#!/usr/bin/env python3
"""Launch NodeAvailabilityDaemon standalone (December 2025).

Synchronizes cloud provider instance state with distributed_hosts.yaml.

Usage:
    # Dry run (default) - log only, no YAML writes
    python scripts/launch_node_availability.py

    # Enable writes
    python scripts/launch_node_availability.py --enable-writes

    # One-shot mode (run once and exit)
    python scripts/launch_node_availability.py --once

    # Custom check interval
    python scripts/launch_node_availability.py --interval 60

Environment variables:
    RINGRIFT_NODE_AVAILABILITY_ENABLED: Enable daemon (default: true)
    RINGRIFT_NODE_AVAILABILITY_DRY_RUN: Log only mode (default: true)
    RINGRIFT_NODE_AVAILABILITY_INTERVAL: Check interval seconds (default: 300)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.node_availability import (
    NodeAvailabilityConfig,
    NodeAvailabilityDaemon,
    get_node_availability_daemon,
    reset_daemon_instance,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch NodeAvailabilityDaemon to sync provider state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--enable-writes",
        action="store_true",
        help="Enable writes to distributed_hosts.yaml (default: dry run)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check cycle and exit",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Check interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--grace-period",
        type=float,
        default=60.0,
        help="Grace period before marking terminated (default: 60)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


async def run_daemon(args: argparse.Namespace) -> None:
    """Run the daemon."""
    reset_daemon_instance()

    # Configure based on arguments
    config = NodeAvailabilityConfig(
        enabled=True,
        dry_run=not args.enable_writes,
        check_interval_seconds=args.interval,
        grace_period_seconds=args.grace_period,
    )

    daemon = NodeAvailabilityDaemon(config)

    mode = "WRITES ENABLED" if args.enable_writes else "DRY RUN"
    logger.info(f"Starting NodeAvailabilityDaemon ({mode})")
    logger.info(f"  Check interval: {args.interval}s")
    logger.info(f"  Grace period: {args.grace_period}s")
    logger.info(f"  Enabled checkers: {list(daemon._checkers.keys())}")

    if args.once:
        # One-shot mode
        logger.info("Running single check cycle...")
        await daemon._run_cycle()

        # Report results
        stats = daemon._stats
        logger.info(f"Cycle completed in {stats.last_cycle_duration_seconds:.2f}s")
        logger.info(f"  Nodes updated: {stats.nodes_updated}")
        logger.info(f"  Provider checks: {stats.provider_checks}")
        if stats.provider_errors:
            logger.warning(f"  Provider errors: {stats.provider_errors}")

        await daemon.stop()
    else:
        # Continuous mode
        shutdown_event = asyncio.Event()

        def signal_handler():
            logger.info("Shutdown signal received")
            shutdown_event.set()

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        try:
            await daemon.start()

            # Wait for shutdown signal
            await shutdown_event.wait()
        finally:
            await daemon.stop()
            logger.info("Daemon stopped")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from other loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    try:
        asyncio.run(run_daemon(args))
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()

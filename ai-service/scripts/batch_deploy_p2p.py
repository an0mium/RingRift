#!/usr/bin/env python3
"""Batch deploy P2P orchestrator to all cluster nodes.

This script provides one-command P2P deployment across the entire cluster:
- Reads node inventory from distributed_hosts.yaml
- Checks P2P health on each node
- Deploys P2P to nodes missing it
- Reports coverage metrics

Usage:
    # Single check and deploy
    python scripts/batch_deploy_p2p.py

    # Daemon mode (continuous monitoring)
    python scripts/batch_deploy_p2p.py --daemon --interval 300

    # Check only (no deployment)
    python scripts/batch_deploy_p2p.py --check-only

    # Verbose output
    python scripts/batch_deploy_p2p.py -v
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.coordination.p2p_auto_deployer import P2PAutoDeployer, P2PDeploymentConfig


async def main() -> int:
    """Run P2P batch deployment."""
    parser = argparse.ArgumentParser(
        description="Deploy P2P orchestrator to all cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (continuous monitoring)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Check interval in seconds for daemon mode (default: 300)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check coverage, don't deploy",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=90.0,
        help="Minimum coverage threshold percentage (default: 90)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create config
    config = P2PDeploymentConfig(
        check_interval_seconds=args.interval,
        min_coverage_percent=args.min_coverage,
    )

    # Create deployer
    deployer = P2PAutoDeployer(config=config)

    if args.daemon:
        # Daemon mode
        print(f"Starting P2P Auto-Deployer daemon (interval: {args.interval}s)")
        try:
            await deployer.run_daemon()
        except KeyboardInterrupt:
            deployer.stop()
            print("\nDaemon stopped")
        return 0

    # Single check/deploy
    if args.check_only:
        print("Checking P2P coverage (check-only mode)...")
        # Temporarily disable deployment by setting max_concurrent to 0
        orig_max = deployer.config.max_concurrent_deployments
        deployer.config.max_concurrent_deployments = 0

    report = await deployer.check_and_deploy()

    if args.check_only:
        deployer.config.max_concurrent_deployments = orig_max

    # Print summary
    print("\n" + "=" * 50)
    print("P2P COVERAGE REPORT")
    print("=" * 50)
    print(f"Total nodes:        {report.total_nodes}")
    print(f"With P2P:           {report.nodes_with_p2p}")
    print(f"Without P2P:        {report.nodes_without_p2p}")
    print(f"Unreachable:        {report.unreachable_nodes}")
    print(f"Coverage:           {report.coverage_percent:.1f}%")
    print("=" * 50)

    if report.nodes_needing_deployment:
        print(f"\nNodes needing deployment: {report.nodes_needing_deployment}")

    # Return non-zero if below threshold
    if report.coverage_percent < args.min_coverage:
        print(f"\nWARNING: Coverage {report.coverage_percent:.1f}% below threshold {args.min_coverage}%")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

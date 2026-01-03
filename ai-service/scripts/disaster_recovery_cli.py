#!/usr/bin/env python3
"""Disaster Recovery CLI - Command-line interface for disaster recovery operations.

January 2026: Created as part of unified data synchronization plan.

Usage:
    # Restore from S3
    python scripts/disaster_recovery_cli.py restore --source s3 --target ./data

    # Restore from OWC
    python scripts/disaster_recovery_cli.py restore --source owc --target ./data

    # Restore specific configs
    python scripts/disaster_recovery_cli.py restore --source s3 --configs hex8_2p,square8_2p

    # Verify backups
    python scripts/disaster_recovery_cli.py verify

    # Verify with detailed output
    python scripts/disaster_recovery_cli.py verify --detailed
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.disaster_recovery import (
    DisasterRecoveryManager,
    RecoveryConfig,
    RestoreStatus,
)
from app.distributed.cluster_manifest import DataSource


def print_header(title: str) -> None:
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result_summary(result) -> None:
    """Print restore result summary."""
    status_icon = "✓" if result.success else "✗"
    print(f"\n{status_icon} Restore Status: {result.status.value}")
    print(f"  Source: {result.source.value}")
    print(f"  Target: {result.target_dir}")
    print(f"  Files Restored: {len(result.files_restored)}")
    print(f"  Files Failed: {len(result.files_failed)}")
    print(f"  Total Games: {result.total_games_restored:,}")
    print(f"  Total Size: {result.total_size_mb:.1f} MB")
    print(f"  Total Time: {result.total_time_seconds:.1f}s")

    if result.files_restored:
        print("\n  Restored Files:")
        for f in result.files_restored:
            print(f"    ✓ {f.config_key}: {f.game_count:,} games ({f.file_size_mb:.1f} MB)")

    if result.files_failed:
        print("\n  Failed Files:")
        for f in result.files_failed:
            print(f"    ✗ {f.config_key}: {f.error}")

    if result.error:
        print(f"\n  Error: {result.error}")


def print_verification_result(result) -> None:
    """Print backup verification result."""
    status_icon = "✓" if result.fully_backed_up else "⚠"
    redundant_icon = "✓" if result.redundant else "⚠"

    print(f"\n{status_icon} Backup Completeness: {'COMPLETE' if result.fully_backed_up else 'INCOMPLETE'}")
    print(f"{redundant_icon} Redundancy: {'FULL' if result.redundant else 'PARTIAL'}")
    print()
    print(f"  S3 Configs: {len(result.s3_configs)}")
    print(f"  OWC Configs: {len(result.owc_configs)}")
    print(f"  Both (Redundant): {len(result.both_configs)}")
    print(f"  S3 Only: {len(result.s3_only_configs)}")
    print(f"  OWC Only: {len(result.owc_only_configs)}")
    print(f"  Missing: {len(result.missing_configs)}")
    print()
    print(f"  S3 Total Games: {result.s3_total_games:,}")
    print(f"  OWC Total Games: {result.owc_total_games:,}")
    print()
    print(f"  Recommendation: {result.recommendation}")


def print_detailed_verification(result) -> None:
    """Print detailed verification breakdown."""
    if result.both_configs:
        print("\n  Fully Backed Up (Both S3 and OWC):")
        for config in result.both_configs:
            print(f"    ✓ {config}")

    if result.s3_only_configs:
        print("\n  S3 Only (Missing OWC Backup):")
        for config in result.s3_only_configs:
            print(f"    ⚠ {config}")

    if result.owc_only_configs:
        print("\n  OWC Only (Missing S3 Backup):")
        for config in result.owc_only_configs:
            print(f"    ⚠ {config}")

    if result.missing_configs:
        print("\n  Missing All Backups (CRITICAL):")
        for config in result.missing_configs:
            print(f"    ✗ {config}")


async def cmd_restore(args) -> int:
    """Execute restore command."""
    print_header(f"Disaster Recovery - Restore from {args.source.upper()}")

    config = RecoveryConfig()
    if args.s3_bucket:
        config.s3_bucket = args.s3_bucket
    if args.owc_host:
        config.owc_host = args.owc_host

    manager = DisasterRecoveryManager(config)

    target_dir = Path(args.target) if args.target else config.target_dir
    config_keys = args.configs.split(",") if args.configs else None

    print(f"\nRestoring from {args.source.upper()}...")
    if config_keys:
        print(f"  Configs: {', '.join(config_keys)}")
    print(f"  Target: {target_dir}")

    if args.source == "s3":
        result = await manager.restore_from_s3(
            target_dir=target_dir,
            config_keys=config_keys,
        )
    elif args.source == "owc":
        result = await manager.restore_from_owc(
            target_dir=target_dir,
            config_keys=config_keys,
            owc_host=args.owc_host,
        )
    else:
        print(f"Error: Unknown source '{args.source}'")
        return 1

    print_result_summary(result)

    if args.json:
        print("\n--- JSON Output ---")
        print(
            json.dumps(
                {
                    "status": result.status.value,
                    "source": result.source.value,
                    "target_dir": str(result.target_dir),
                    "files_restored": len(result.files_restored),
                    "files_failed": len(result.files_failed),
                    "total_games": result.total_games_restored,
                    "total_size_mb": result.total_size_mb,
                    "total_time_seconds": result.total_time_seconds,
                },
                indent=2,
            )
        )

    return 0 if result.success else 1


async def cmd_verify(args) -> int:
    """Execute verify command."""
    print_header("Disaster Recovery - Verify Backups")

    manager = DisasterRecoveryManager()

    print("\nVerifying backup completeness...")
    result = await manager.verify_backup_completeness()

    print_verification_result(result)

    if args.detailed:
        print_detailed_verification(result)

    if args.json:
        print("\n--- JSON Output ---")
        print(
            json.dumps(
                {
                    "fully_backed_up": result.fully_backed_up,
                    "redundant": result.redundant,
                    "s3_verified": result.s3_verified,
                    "owc_verified": result.owc_verified,
                    "s3_configs_count": len(result.s3_configs),
                    "owc_configs_count": len(result.owc_configs),
                    "both_count": len(result.both_configs),
                    "s3_only_count": len(result.s3_only_configs),
                    "owc_only_count": len(result.owc_only_configs),
                    "missing_count": len(result.missing_configs),
                    "s3_total_games": result.s3_total_games,
                    "owc_total_games": result.owc_total_games,
                    "recommendation": result.recommendation,
                },
                indent=2,
            )
        )

    return 0 if result.fully_backed_up else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Disaster Recovery CLI for RingRift AI Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Restore from S3
    python scripts/disaster_recovery_cli.py restore --source s3 --target ./data/restored

    # Restore specific configs from OWC
    python scripts/disaster_recovery_cli.py restore --source owc --configs hex8_2p,square8_2p

    # Verify backup completeness
    python scripts/disaster_recovery_cli.py verify --detailed

    # Output as JSON
    python scripts/disaster_recovery_cli.py verify --json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Restore command
    restore_parser = subparsers.add_parser(
        "restore", help="Restore data from backup"
    )
    restore_parser.add_argument(
        "--source",
        "-s",
        choices=["s3", "owc"],
        required=True,
        help="Backup source (s3 or owc)",
    )
    restore_parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target directory for restored files",
    )
    restore_parser.add_argument(
        "--configs",
        "-c",
        type=str,
        help="Comma-separated list of configs to restore (default: all)",
    )
    restore_parser.add_argument(
        "--s3-bucket",
        type=str,
        help="S3 bucket override",
    )
    restore_parser.add_argument(
        "--owc-host",
        type=str,
        help="OWC host override",
    )
    restore_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify", help="Verify backup completeness"
    )
    verify_parser.add_argument(
        "--detailed",
        "-d",
        action="store_true",
        help="Show detailed breakdown by config",
    )
    verify_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "restore":
        return asyncio.run(cmd_restore(args))
    elif args.command == "verify":
        return asyncio.run(cmd_verify(args))
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

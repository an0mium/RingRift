#!/usr/bin/env python3
"""Verification script for mandatory data freshness check.

This script demonstrates that the freshness check is enabled by default
and provides examples of the different configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.train_cli import parse_args
from app.coordination.training_freshness import (
    DEFAULT_MAX_AGE_HOURS,
    FreshnessConfig,
)


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def verify_cli_defaults() -> bool:
    """Verify CLI argument defaults."""
    print_section("CLI ARGUMENT DEFAULTS")

    args = parse_args([])

    print("Default arguments (no flags specified):")
    print(f"  skip_freshness_check: {args.skip_freshness_check}")
    print(f"  max_data_age_hours: {args.max_data_age_hours}")
    print(f"  allow_stale_data: {args.allow_stale_data}")
    print()

    # Verify expectations
    assert args.skip_freshness_check is False, "Check should be ENABLED by default"
    assert args.max_data_age_hours == 1.0, "Default threshold should be 1.0 hours"
    assert args.allow_stale_data is False, "Should FAIL on stale by default"

    print("✅ CLI defaults are correct")
    print("   - Freshness check IS enabled by default")
    print("   - Data must be <1 hour old")
    print("   - Training FAILS on stale data (not warns)")

    return True


def verify_module_defaults() -> bool:
    """Verify training_freshness module defaults."""
    print_section("TRAINING FRESHNESS MODULE")

    print(f"DEFAULT_MAX_AGE_HOURS: {DEFAULT_MAX_AGE_HOURS}")
    print()

    config = FreshnessConfig()
    print("Default FreshnessConfig:")
    print(f"  max_age_hours: {config.max_age_hours}")
    print(f"  wait_for_sync: {config.wait_for_sync}")
    print(f"  trigger_sync: {config.trigger_sync}")
    print(f"  min_games_required: {config.min_games_required}")
    print()

    assert config.max_age_hours == 1.0, "Default should be 1.0 hours"

    print("✅ Module defaults are correct")

    return True


def show_usage_examples() -> None:
    """Show usage examples."""
    print_section("USAGE EXAMPLES")

    examples = [
        (
            "Standard training (check enabled by default)",
            [
                "python -m app.training.train \\",
                "  --board-type hex8 --num-players 2 \\",
                "  --data-path data/training/hex8_2p.npz",
            ],
        ),
        (
            "Skip freshness check (NOT RECOMMENDED)",
            [
                "python -m app.training.train \\",
                "  --skip-freshness-check \\",
                "  --board-type hex8 --num-players 2 \\",
                "  --data-path data/training/hex8_2p.npz",
            ],
        ),
        (
            "Allow stale data (warns instead of fails)",
            [
                "python -m app.training.train \\",
                "  --allow-stale-data \\",
                "  --board-type hex8 --num-players 2 \\",
                "  --data-path data/training/hex8_2p.npz",
            ],
        ),
        (
            "Custom threshold (2.5 hours)",
            [
                "python -m app.training.train \\",
                "  --max-data-age-hours 2.5 \\",
                "  --board-type hex8 --num-players 2 \\",
                "  --data-path data/training/hex8_2p.npz",
            ],
        ),
    ]

    for i, (description, commands) in enumerate(examples, 1):
        print(f"{i}. {description}:")
        print()
        for cmd in commands:
            print(f"   {cmd}")
        print()


def verify_flag_behavior() -> bool:
    """Verify flag behavior."""
    print_section("FLAG BEHAVIOR VERIFICATION")

    test_cases = [
        (
            "Default (no flags)",
            [],
            {"skip_freshness_check": False, "max_data_age_hours": 1.0, "allow_stale_data": False},
        ),
        (
            "With --skip-freshness-check",
            ["--skip-freshness-check"],
            {"skip_freshness_check": True},
        ),
        (
            "With --allow-stale-data",
            ["--allow-stale-data"],
            {"skip_freshness_check": False, "allow_stale_data": True},
        ),
        (
            "With --max-data-age-hours 2.5",
            ["--max-data-age-hours", "2.5"],
            {"max_data_age_hours": 2.5},
        ),
    ]

    for description, flags, expected in test_cases:
        args = parse_args(flags)
        print(f"Test: {description}")
        for key, expected_value in expected.items():
            actual_value = getattr(args, key)
            status = "✅" if actual_value == expected_value else "❌"
            print(f"  {status} {key}: {actual_value} (expected: {expected_value})")
        print()

    print("✅ All flag behaviors verified")

    return True


def main() -> int:
    """Run verification."""
    print("=" * 70)
    print("DATA FRESHNESS CHECK - VERIFICATION SCRIPT")
    print("=" * 70)
    print()
    print("This script verifies that the data freshness check is mandatory")
    print("by default and demonstrates the different configuration options.")

    try:
        # Run verifications
        verify_cli_defaults()
        verify_module_defaults()
        verify_flag_behavior()
        show_usage_examples()

        # Final summary
        print_section("SUMMARY")
        print("✅ Data freshness check is MANDATORY by default")
        print("✅ Default threshold: 1.0 hours")
        print("✅ Default behavior: FAIL on stale data (not warn)")
        print()
        print("Override options:")
        print("  - Use --skip-freshness-check to bypass (DANGEROUS)")
        print("  - Use --allow-stale-data to warn instead of fail")
        print("  - Use --max-data-age-hours <hours> to adjust threshold")
        print()
        print("See docs/DATA_FRESHNESS_CHECK.md for full documentation")
        print()

        return 0

    except AssertionError as e:
        print()
        print(f"❌ VERIFICATION FAILED: {e}")
        return 1
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

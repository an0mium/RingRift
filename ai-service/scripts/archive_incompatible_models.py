#!/usr/bin/env python
"""Archive incompatible model checkpoints.

This script scans model directories for checkpoints that are incompatible
with the current architecture and moves them to an archive directory.

This is useful when model architecture changes make old checkpoints unusable
(e.g., changes to movement_channels, in_channels, policy_size).

Usage:
    # Dry run (show what would be archived)
    python scripts/archive_incompatible_models.py --models-dir models --dry-run

    # Archive incompatible hex8 v3 models
    python scripts/archive_incompatible_models.py \
        --models-dir models \
        --model-class HexNeuralNet_v3 \
        --expected-config '{"in_channels": 16, "movement_channels": 48}'

    # Archive all incompatible models (auto-detect)
    python scripts/archive_incompatible_models.py --models-dir models --auto-detect
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.training.model_versioning import (
    check_checkpoint_compatibility,
    get_checkpoint_info,
)


# Known model configurations for auto-detection
KNOWN_CONFIGS: dict[str, dict[str, Any]] = {
    "HexNeuralNet_v3_hex8": {
        "model_class": "HexNeuralNet_v3",
        "board_type": "hex8",
        "expected_config": {
            "in_channels": 16,
            "movement_channels": 48,  # 6 directions * 8 max_distance
            "max_distance": 8,
            "board_size": 9,
        },
    },
    "HexNeuralNet_v3_hexagonal": {
        "model_class": "HexNeuralNet_v3",
        "board_type": "hexagonal",
        "expected_config": {
            "in_channels": 16,
            "movement_channels": 144,  # 6 directions * 24 max_distance
            "max_distance": 24,
            "board_size": 25,
        },
    },
    "HexNeuralNet_v4_hex8": {
        "model_class": "HexNeuralNet_v4",
        "board_type": "hex8",
        "expected_config": {
            "in_channels": 64,
            "movement_channels": 48,
            "max_distance": 8,
            "board_size": 9,
        },
    },
    "HexNeuralNet_v4_hexagonal": {
        "model_class": "HexNeuralNet_v4",
        "board_type": "hexagonal",
        "expected_config": {
            "in_channels": 64,
            "movement_channels": 144,
            "max_distance": 24,
            "board_size": 25,
        },
    },
}


def find_model_files(models_dir: Path, pattern: str = "*.pth") -> list[Path]:
    """Find all model checkpoint files in directory."""
    return list(models_dir.glob(pattern)) + list(models_dir.glob("**/" + pattern))


def infer_expected_config(checkpoint_info: dict[str, Any]) -> dict[str, Any] | None:
    """Try to infer expected config based on checkpoint info."""
    model_class = checkpoint_info.get("model_class")
    config = checkpoint_info.get("config", {})

    if not model_class:
        return None

    board_size = config.get("board_size", 0)
    board_type = "hex8" if board_size <= 9 else "hexagonal"

    config_key = f"{model_class}_{board_type}"
    return KNOWN_CONFIGS.get(config_key)


def check_model_compatibility(
    model_path: Path,
    model_class: str | None = None,
    expected_config: dict[str, Any] | None = None,
    auto_detect: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Check if a model checkpoint is compatible.

    Returns:
        Tuple of (is_compatible, reason, checkpoint_info)
    """
    # Get checkpoint info first
    info = get_checkpoint_info(str(model_path))

    if info.get("error"):
        return False, info["error"], info

    # Auto-detect expected config if needed
    if auto_detect and model_class is None:
        known = infer_expected_config(info)
        if known:
            model_class = known["model_class"]
            expected_config = known["expected_config"]

    if model_class is None:
        # Can't check without knowing expected model class
        return True, "skipped (no expected model class)", info

    # Check compatibility
    compatible, reason = check_checkpoint_compatibility(
        str(model_path),
        model_class,
        expected_config,
    )

    return compatible, reason, info


def archive_model(
    model_path: Path,
    archive_dir: Path,
    reason: str,
    dry_run: bool = False,
) -> bool:
    """Move model to archive directory."""
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Create archive filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{model_path.stem}_{timestamp}{model_path.suffix}"
    archive_path = archive_dir / archive_name

    # Also save reason in a companion file
    reason_file = archive_dir / f"{model_path.stem}_{timestamp}.reason.txt"

    if dry_run:
        print(f"  [DRY RUN] Would archive: {model_path}")
        print(f"            To: {archive_path}")
        print(f"            Reason: {reason}")
        return True

    try:
        shutil.move(str(model_path), str(archive_path))
        reason_file.write_text(f"Archived: {model_path}\nReason: {reason}\nDate: {datetime.now().isoformat()}\n")
        print(f"  Archived: {model_path.name} -> {archive_path.name}")
        return True
    except Exception as e:
        print(f"  ERROR archiving {model_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Archive incompatible model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=None,
        help="Directory to move incompatible models to (default: <models-dir>/archived_incompatible)",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Expected model class name (e.g., HexNeuralNet_v3)",
    )
    parser.add_argument(
        "--expected-config",
        type=str,
        default=None,
        help="Expected config as JSON string (e.g., '{\"in_channels\": 16}')",
    )
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect expected config based on model class and board type",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pth",
        help="Glob pattern for model files (default: *.pth)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without actually moving files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each checkpoint",
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"ERROR: Models directory not found: {models_dir}")
        return 1

    archive_dir = Path(args.archive_dir) if args.archive_dir else models_dir / "archived_incompatible"

    expected_config = None
    if args.expected_config:
        try:
            expected_config = json.loads(args.expected_config)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --expected-config: {e}")
            return 1

    # Find all model files
    model_files = find_model_files(models_dir, args.pattern)
    print(f"Found {len(model_files)} model files in {models_dir}")

    if not model_files:
        print("No model files to check.")
        return 0

    # Check each model
    compatible_count = 0
    incompatible_count = 0
    skipped_count = 0
    archived_count = 0

    for model_path in sorted(model_files):
        # Skip files already in archive directory
        if "archived" in str(model_path):
            skipped_count += 1
            if args.verbose:
                print(f"  Skipped (in archive): {model_path.name}")
            continue

        is_compatible, reason, info = check_model_compatibility(
            model_path,
            model_class=args.model_class,
            expected_config=expected_config,
            auto_detect=args.auto_detect,
        )

        if args.verbose:
            print(f"\n{model_path.name}:")
            print(f"  Model class: {info.get('model_class', 'unknown')}")
            print(f"  Has metadata: {info.get('has_metadata', False)}")
            if info.get("critical_shapes"):
                print(f"  Critical shapes: {info['critical_shapes']}")

        if "skipped" in reason:
            skipped_count += 1
            if args.verbose:
                print(f"  Status: SKIPPED ({reason})")
        elif is_compatible:
            compatible_count += 1
            if args.verbose:
                print(f"  Status: COMPATIBLE")
        else:
            incompatible_count += 1
            print(f"\nINCOMPATIBLE: {model_path.name}")
            print(f"  Reason: {reason}")

            if archive_model(model_path, archive_dir, reason, dry_run=args.dry_run):
                archived_count += 1

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Compatible:   {compatible_count}")
    print(f"  Incompatible: {incompatible_count}")
    print(f"  Skipped:      {skipped_count}")
    if args.dry_run:
        print(f"  Would archive: {archived_count}")
    else:
        print(f"  Archived:     {archived_count}")

    return 0 if incompatible_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

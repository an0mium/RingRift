#!/usr/bin/env python3
"""
Fix Model Checksums Script

Re-computes and updates the checksum stored in model metadata to match
the actual state_dict contents. This fixes "CHECKPOINT INTEGRITY FAILURE"
errors that occur when model files are modified after initial save.

Usage:
    # Scan and report checksum mismatches (dry-run)
    python scripts/fix_model_checksums.py --scan

    # Fix all mismatched checksums
    python scripts/fix_model_checksums.py --fix

    # Fix specific model
    python scripts/fix_model_checksums.py --fix models/canonical_square19_2p.pth
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fix_model_checksums")

AI_SERVICE_ROOT = Path(__file__).parent.parent
MODELS_DIR = AI_SERVICE_ROOT / "models"


def compute_state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    """Compute SHA256 checksum of a model's state_dict."""
    hasher = hashlib.sha256()

    # Sort keys for deterministic ordering
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        # Use numpy bytes for deterministic serialization
        tensor_bytes = tensor.cpu().numpy().tobytes()
        hasher.update(key.encode())
        hasher.update(tensor_bytes)

    return hasher.hexdigest()


def check_model_checksum(model_path: Path) -> tuple[bool, str, str, dict | None]:
    """
    Check if a model's stored checksum matches the actual checksum.

    Returns:
        (is_valid, stored_checksum, actual_checksum, metadata)
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load {model_path}: {e}")
        return False, "", "", None

    # Check if it has versioning metadata
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        stored_checksum = metadata.get("checksum", "")
    else:
        # metadata might be a ModelMetadata object
        stored_checksum = getattr(metadata, "checksum", "")

    if not stored_checksum:
        logger.warning(f"{model_path.name}: No checksum in metadata (legacy model)")
        return True, "", "", metadata  # No checksum = nothing to fix

    # Get state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Compute actual checksum
    actual_checksum = compute_state_dict_checksum(state_dict)

    is_valid = stored_checksum == actual_checksum
    return is_valid, stored_checksum, actual_checksum, metadata


def fix_model_checksum(model_path: Path, dry_run: bool = False) -> bool:
    """
    Fix a model's checksum by recomputing and updating the metadata.

    Returns:
        True if fixed successfully, False otherwise
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load {model_path}: {e}")
        return False

    # Get state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_dict_key = "model_state_dict"
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict_key = "state_dict"
    else:
        logger.warning(f"{model_path.name}: Not a versioned checkpoint, skipping")
        return False

    # Compute correct checksum
    actual_checksum = compute_state_dict_checksum(state_dict)

    # Update metadata
    metadata = checkpoint.get("metadata", {})
    if isinstance(metadata, dict):
        old_checksum = metadata.get("checksum", "")
        metadata["checksum"] = actual_checksum
    else:
        # metadata is an object - convert to dict, update, convert back
        old_checksum = getattr(metadata, "checksum", "")
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
            metadata["checksum"] = actual_checksum
        else:
            logger.warning(f"{model_path.name}: Cannot update metadata object")
            return False

    checkpoint["metadata"] = metadata

    if dry_run:
        logger.info(f"[DRY-RUN] Would fix {model_path.name}:")
        logger.info(f"  Old checksum: {old_checksum[:16]}...")
        logger.info(f"  New checksum: {actual_checksum[:16]}...")
        return True

    # Save with atomic write
    temp_path = model_path.with_suffix(".tmp")
    backup_path = model_path.with_suffix(".bak")

    try:
        # Save to temp file
        torch.save(checkpoint, temp_path)

        # Backup original
        if model_path.exists():
            shutil.copy2(model_path, backup_path)

        # Atomic replace
        temp_path.rename(model_path)

        # Remove backup on success
        backup_path.unlink(missing_ok=True)

        logger.info(f"Fixed {model_path.name}:")
        logger.info(f"  Old checksum: {old_checksum[:16]}...")
        logger.info(f"  New checksum: {actual_checksum[:16]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to save {model_path}: {e}")
        # Restore backup
        if backup_path.exists():
            backup_path.rename(model_path)
        temp_path.unlink(missing_ok=True)
        return False


def scan_models(models_dir: Path) -> list[tuple[Path, bool, str, str]]:
    """
    Scan all models and check their checksums.

    Returns:
        List of (path, is_valid, stored_checksum, actual_checksum)
    """
    results = []
    model_files = list(models_dir.glob("*.pth"))

    logger.info(f"Scanning {len(model_files)} model files...")

    for model_path in sorted(model_files):
        is_valid, stored, actual, _ = check_model_checksum(model_path)
        results.append((model_path, is_valid, stored, actual))

        if not is_valid and stored:  # Has checksum but mismatched
            logger.warning(f"MISMATCH: {model_path.name}")
            logger.warning(f"  Stored:  {stored[:16]}...")
            logger.warning(f"  Actual:  {actual[:16]}...")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fix model checksum mismatches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and report checksum mismatches"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix checksum mismatches"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific model files to check/fix (default: all in models/)"
    )

    args = parser.parse_args()

    if not args.scan and not args.fix:
        parser.print_help()
        return

    # Get models to process
    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        model_paths = list(MODELS_DIR.glob("*.pth"))

    if args.scan:
        print("\n" + "=" * 70)
        print(" Model Checksum Scan")
        print("=" * 70)

        valid_count = 0
        mismatch_count = 0
        no_checksum_count = 0

        for model_path in sorted(model_paths):
            is_valid, stored, actual, _ = check_model_checksum(model_path)

            if not stored:
                no_checksum_count += 1
                print(f"  [NO CHECKSUM] {model_path.name}")
            elif is_valid:
                valid_count += 1
                print(f"  [OK] {model_path.name}")
            else:
                mismatch_count += 1
                print(f"  [MISMATCH] {model_path.name}")
                print(f"      Stored:  {stored[:24]}...")
                print(f"      Actual:  {actual[:24]}...")

        print("=" * 70)
        print(f"  Total:       {len(model_paths)}")
        print(f"  Valid:       {valid_count}")
        print(f"  Mismatched:  {mismatch_count}")
        print(f"  No checksum: {no_checksum_count}")
        print("=" * 70)

        if mismatch_count > 0:
            print("\nRun with --fix to repair mismatched checksums")

    if args.fix:
        print("\n" + "=" * 70)
        print(" Fixing Model Checksums" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 70)

        fixed = 0
        failed = 0
        skipped = 0

        for model_path in sorted(model_paths):
            is_valid, stored, actual, _ = check_model_checksum(model_path)

            if not stored:
                skipped += 1
                continue

            if is_valid:
                skipped += 1
                continue

            if fix_model_checksum(model_path, dry_run=args.dry_run):
                fixed += 1
            else:
                failed += 1

        print("=" * 70)
        print(f"  Fixed:   {fixed}")
        print(f"  Failed:  {failed}")
        print(f"  Skipped: {skipped}")
        print("=" * 70)


if __name__ == "__main__":
    main()

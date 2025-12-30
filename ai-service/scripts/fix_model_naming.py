#!/usr/bin/env python3
"""Fix model file naming to match actual architecture.

This script scans model files, extracts their actual architecture from metadata,
and renames files to accurately reflect their contents.

Problem: Many model files are misnamed (e.g., "hex8_2p_v6.pth" but actually
contain HexNeuralNet_v2 architecture). This causes confusion and makes it
hard to track which models use which architecture.

Usage:
    # Preview changes (dry run)
    python scripts/fix_model_naming.py

    # Apply renames
    python scripts/fix_model_naming.py --apply

    # Scan specific directory
    python scripts/fix_model_naming.py --model-dir models/archive

    # Generate shell script instead of applying
    python scripts/fix_model_naming.py --output-script rename_models.sh
"""

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Mapping from model_class to canonical version string
MODEL_CLASS_TO_VERSION = {
    # V2 architectures
    "HexNeuralNet_v2": "v2",
    "RingRiftCNN_v2": "v2",
    "HexNeuralNet_v2_Lite": "v2_lite",
    "RingRiftCNN_v2_Lite": "v2_lite",
    # V3 architectures
    "HexNeuralNet_v3": "v3",
    "RingRiftCNN_v3": "v3",
    "HexNeuralNet_v3_Lite": "v3_lite",
    "RingRiftCNN_v3_Lite": "v3_lite",
    "RingRiftCNN_v3_Flat": "v3_flat",
    # V4 architectures
    "HexNeuralNet_v4": "v4",
    "RingRiftCNN_v4": "v4",
    # V5 Heavy architectures
    "HexNeuralNet_v5_Heavy": "v5_heavy",
    "RingRiftCNN_v5_Heavy": "v5_heavy",
    "RingRiftCNN_v5": "v5",
    # GNN architectures
    "GNNPolicyNet": "gnn",
    "HybridPolicyNet": "hybrid",
}

# Version patterns to detect in filenames
VERSION_PATTERNS = [
    (r"_v6[-_]xl", "v6_xl"),
    (r"_v6[-_]large", "v6_large"),
    (r"_v5[-_]heavy[-_]xl", "v5_heavy_xl"),
    (r"_v5[-_]heavy[-_]large", "v5_heavy_large"),
    (r"_v5[-_]heavy", "v5_heavy"),
    (r"_v5heavy", "v5_heavy"),
    (r"_v6", "v6"),
    (r"_v5", "v5"),
    (r"_v4", "v4"),
    (r"_v3[-_]flat", "v3_flat"),
    (r"_v3[-_]lite", "v3_lite"),
    (r"_v3", "v3"),
    (r"_v2[-_]lite", "v2_lite"),
    (r"_v2", "v2"),
]


@dataclass
class ModelInfo:
    """Information about a model file."""
    path: Path
    filename: str
    claimed_version: str | None  # Version from filename
    actual_class: str | None  # From metadata
    actual_version: str | None  # Derived from actual_class
    needs_rename: bool
    new_filename: str | None
    error: str | None = None


def extract_version_from_filename(filename: str) -> str | None:
    """Extract version string from filename."""
    filename_lower = filename.lower()
    for pattern, version in VERSION_PATTERNS:
        if re.search(pattern, filename_lower):
            return version
    return None


def get_model_metadata(path: Path) -> tuple[str | None, str | None]:
    """Load model and extract architecture metadata.

    Returns:
        Tuple of (model_class, version) or (None, None) on error
    """
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Try versioning metadata first
        metadata = checkpoint.get("_versioning_metadata", {})
        model_class = metadata.get("model_class")
        version = metadata.get("version")

        if model_class:
            return model_class, version

        # Try model_state_dict inspection
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Infer from layer names
        layer_names = list(state_dict.keys())
        if any("hex" in k.lower() for k in layer_names):
            return "HexNeuralNet_v2", None  # Default assumption
        elif any("gnn" in k.lower() for k in layer_names):
            return "GNNPolicyNet", None
        else:
            return "RingRiftCNN_v2", None  # Default assumption

    except Exception as e:
        return None, str(e)


def compute_new_filename(filename: str, claimed_version: str | None,
                         actual_version: str) -> str:
    """Compute new filename with correct version."""
    if claimed_version is None:
        # No version in filename, just return as-is
        return filename

    if claimed_version == actual_version:
        # Already correct
        return filename

    # Replace version in filename
    new_filename = filename

    # Try each pattern to find and replace
    for pattern, version in VERSION_PATTERNS:
        if version == claimed_version:
            # Found the pattern, replace with actual version
            new_filename = re.sub(
                pattern,
                f"_{actual_version}",
                filename,
                flags=re.IGNORECASE
            )
            break

    return new_filename


def analyze_model_file(path: Path) -> ModelInfo:
    """Analyze a single model file."""
    filename = path.name
    claimed_version = extract_version_from_filename(filename)

    model_class, error_or_version = get_model_metadata(path)

    if model_class is None:
        return ModelInfo(
            path=path,
            filename=filename,
            claimed_version=claimed_version,
            actual_class=None,
            actual_version=None,
            needs_rename=False,
            new_filename=None,
            error=error_or_version,
        )

    actual_version = MODEL_CLASS_TO_VERSION.get(model_class)

    if actual_version is None:
        return ModelInfo(
            path=path,
            filename=filename,
            claimed_version=claimed_version,
            actual_class=model_class,
            actual_version=None,
            needs_rename=False,
            new_filename=None,
            error=f"Unknown model class: {model_class}",
        )

    # Check if rename needed
    needs_rename = (
        claimed_version is not None and
        claimed_version != actual_version
    )

    new_filename = None
    if needs_rename:
        new_filename = compute_new_filename(filename, claimed_version, actual_version)

    return ModelInfo(
        path=path,
        filename=filename,
        claimed_version=claimed_version,
        actual_class=model_class,
        actual_version=actual_version,
        needs_rename=needs_rename,
        new_filename=new_filename,
    )


def scan_models(model_dir: Path) -> list[ModelInfo]:
    """Scan all model files in directory."""
    results = []

    pth_files = sorted(model_dir.glob("**/*.pth"))
    logger.info(f"Found {len(pth_files)} model files")

    for i, path in enumerate(pth_files):
        if (i + 1) % 50 == 0:
            logger.info(f"  Analyzed {i + 1}/{len(pth_files)} files...")

        info = analyze_model_file(path)
        results.append(info)

    return results


def print_summary(results: list[ModelInfo]) -> None:
    """Print analysis summary."""
    total = len(results)
    errors = [r for r in results if r.error]
    needs_rename = [r for r in results if r.needs_rename]
    correct = [r for r in results if not r.needs_rename and not r.error]

    print("\n" + "=" * 70)
    print("MODEL NAMING ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal files scanned: {total}")
    print(f"  Correctly named:   {len(correct)}")
    print(f"  Need renaming:     {len(needs_rename)}")
    print(f"  Errors:            {len(errors)}")

    # Architecture distribution
    arch_counts: dict[str, int] = {}
    for r in results:
        if r.actual_class:
            arch_counts[r.actual_class] = arch_counts.get(r.actual_class, 0) + 1

    print("\nArchitecture distribution:")
    for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
        print(f"  {arch}: {count} files")

    # Mismatch details
    if needs_rename:
        print("\n" + "-" * 70)
        print("FILES NEEDING RENAME:")
        print("-" * 70)

        # Group by mismatch type
        mismatches: dict[str, list[ModelInfo]] = {}
        for r in needs_rename:
            key = f"{r.claimed_version} -> {r.actual_version}"
            if key not in mismatches:
                mismatches[key] = []
            mismatches[key].append(r)

        for mismatch, files in sorted(mismatches.items()):
            print(f"\n{mismatch} ({len(files)} files):")
            for r in files[:5]:  # Show first 5
                print(f"  {r.filename}")
            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more")

    if errors:
        print("\n" + "-" * 70)
        print("FILES WITH ERRORS:")
        print("-" * 70)
        for r in errors[:10]:
            print(f"  {r.filename}: {r.error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


def generate_rename_script(results: list[ModelInfo], output_path: Path) -> None:
    """Generate shell script with rename commands."""
    needs_rename = [r for r in results if r.needs_rename and r.new_filename]

    with open(output_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated model rename script\n")
        f.write(f"# Generated by fix_model_naming.py\n")
        f.write(f"# {len(needs_rename)} files to rename\n\n")
        f.write("set -e\n\n")

        for r in needs_rename:
            old_path = r.path
            new_path = r.path.parent / r.new_filename
            f.write(f'mv "{old_path}" "{new_path}"\n')

        f.write("\necho 'Done! Renamed {} files'\n".format(len(needs_rename)))

    os.chmod(output_path, 0o755)
    logger.info(f"Generated rename script: {output_path}")


def apply_renames(results: list[ModelInfo], dry_run: bool = True) -> int:
    """Apply file renames."""
    needs_rename = [r for r in results if r.needs_rename and r.new_filename]

    if not needs_rename:
        logger.info("No files need renaming")
        return 0

    renamed = 0
    for r in needs_rename:
        old_path = r.path
        new_path = r.path.parent / r.new_filename

        if new_path.exists():
            logger.warning(f"SKIP: Target exists: {new_path}")
            continue

        if dry_run:
            logger.info(f"Would rename: {r.filename} -> {r.new_filename}")
        else:
            old_path.rename(new_path)
            logger.info(f"Renamed: {r.filename} -> {r.new_filename}")
            renamed += 1

    return renamed


def main():
    parser = argparse.ArgumentParser(
        description="Fix model file naming to match actual architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to scan for model files (default: models)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply renames (default: dry run)",
    )
    parser.add_argument(
        "--output-script",
        type=Path,
        help="Generate shell script with rename commands",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show summary, not individual files",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        logger.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Scan models
    logger.info(f"Scanning models in {args.model_dir}...")
    results = scan_models(args.model_dir)

    # Print summary
    print_summary(results)

    # Generate script if requested
    if args.output_script:
        generate_rename_script(results, args.output_script)

    # Apply renames
    needs_rename = [r for r in results if r.needs_rename]
    if needs_rename:
        print("\n" + "=" * 70)
        if args.apply:
            print("APPLYING RENAMES...")
            renamed = apply_renames(results, dry_run=False)
            print(f"Renamed {renamed} files")
        else:
            print("DRY RUN - No files renamed")
            print("Run with --apply to execute renames")
            apply_renames(results, dry_run=True)


if __name__ == "__main__":
    main()

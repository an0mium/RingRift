#!/usr/bin/env python
"""Merge multiple NPZ training datasets preserving all data including policy.

This script properly handles object arrays (ragged policy data) that require
special treatment during concatenation.

Usage:
    python scripts/merge_training_datasets.py \
        --inputs data/training/hex8_2p_v3_policy.npz \
                 data/training/hex8_2p_v3_gpu.npz \
                 data/training/hex8_2p_v3_fresh.npz \
        --output data/training/hex8_2p_v3_merged_full.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def merge_datasets(input_paths: list[Path], output_path: Path, verbose: bool = True) -> int:
    """Merge multiple NPZ datasets preserving all arrays including object arrays.

    Returns number of samples in merged dataset.
    """
    if verbose:
        print(f"Merging {len(input_paths)} datasets...")

    # Collect all arrays by key
    arrays_by_key: dict[str, list] = {}
    metadata_keys = set()
    total_samples = 0

    for path in input_paths:
        if verbose:
            print(f"\nLoading {path.name}...")

        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        n_samples = len(data['features'])
        total_samples += n_samples

        if verbose:
            print(f"  {n_samples} samples, keys: {keys}")

        for key in keys:
            arr = data[key]

            # Check if this is metadata (scalar or small array)
            is_metadata = (
                not hasattr(arr, '__len__') or
                (hasattr(arr, 'shape') and arr.shape == ()) or
                (hasattr(arr, '__len__') and len(arr) != n_samples)
            )

            if is_metadata:
                metadata_keys.add(key)
                # Just keep first value for metadata
                if key not in arrays_by_key:
                    arrays_by_key[key] = arr
            else:
                if key not in arrays_by_key:
                    arrays_by_key[key] = []
                arrays_by_key[key].append(arr)

    if verbose:
        print(f"\nMerging arrays...")
        print(f"  Metadata keys (kept from first file): {metadata_keys}")

    # Merge arrays
    merged = {}
    for key, value in arrays_by_key.items():
        if key in metadata_keys:
            merged[key] = value
        else:
            # Check if object array (ragged)
            if isinstance(value[0], np.ndarray) and value[0].dtype == object:
                # Concatenate object arrays
                merged[key] = np.concatenate(value, axis=0)
                if verbose:
                    print(f"  {key}: object array, shape {merged[key].shape}")
            else:
                # Regular array concatenation
                merged[key] = np.concatenate(value, axis=0)
                if verbose:
                    print(f"  {key}: shape {merged[key].shape}")

    # Verify sample counts match
    n_merged = len(merged['features'])
    assert n_merged == total_samples, f"Sample count mismatch: {n_merged} vs {total_samples}"

    # Save merged dataset
    if verbose:
        print(f"\nSaving to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **merged)

    file_size = output_path.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"  File size: {file_size:.2f} MB")
        print(f"\n✓ Merged {total_samples} samples from {len(input_paths)} datasets")

    return n_merged


def main():
    parser = argparse.ArgumentParser(description="Merge NPZ training datasets")
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="Input NPZ files to merge"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output merged NPZ file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)

    # Validate inputs exist
    for path in input_paths:
        if not path.exists():
            print(f"ERROR: Input file not found: {path}")
            return 1

    n_samples = merge_datasets(input_paths, output_path, verbose=not args.quiet)

    return 0 if n_samples > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

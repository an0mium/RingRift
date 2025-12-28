"""NPZ Combiner - Intelligent combination of multiple NPZ training files.

This module provides utilities for combining historical and fresh training
data with quality-aware weighting and config validation.

Features:
- Combines multiple NPZ files with freshness-based weighting
- Validates config compatibility (board type, player count, feature version)
- Preserves quality scores and metadata
- Supports sample deduplication by game_id

Usage:
    from app.training.npz_combiner import (
        combine_npz_files,
        NPZCombinerConfig,
        validate_npz_compatibility,
    )

    # Combine with freshness weighting
    result = combine_npz_files(
        input_paths=['historical.npz', 'fresh.npz'],
        output_path='combined.npz',
        config=NPZCombinerConfig(
            freshness_weight=2.0,  # Fresh data sampled 2x more
            min_quality_score=0.3,
            deduplicate=True,
        ),
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NPZMetadata:
    """Extracted metadata from an NPZ file."""

    path: Path
    sample_count: int
    board_type: str | None = None
    num_players: int | None = None
    encoder_version: str | None = None
    in_channels: int | None = None
    spatial_size: int | None = None
    policy_size: int | None = None
    export_time: datetime | None = None
    newest_game_time: datetime | None = None
    heuristic_mode: str | None = None  # "full" or "fast"

    @property
    def config_key(self) -> str | None:
        """Return config key like 'hex8_2p'."""
        if self.board_type and self.num_players:
            return f"{self.board_type}_{self.num_players}p"
        return None

    @property
    def age_hours(self) -> float | None:
        """Age in hours since newest game."""
        if not self.newest_game_time:
            return None
        delta = datetime.now(tz=timezone.utc) - self.newest_game_time
        return delta.total_seconds() / 3600


@dataclass
class NPZCombinerConfig:
    """Configuration for NPZ combination."""

    # Freshness weighting: samples from fresher files are weighted higher
    freshness_weight: float = 1.0  # 1.0 = no weighting, 2.0 = fresh 2x more likely
    freshness_half_life_hours: float = 24.0  # Half-life for freshness decay

    # Quality filtering
    min_quality_score: float = 0.0  # 0-1, samples below this are excluded
    quality_weight: float = 1.0  # Weight samples by quality score

    # Deduplication
    deduplicate: bool = False  # Deduplicate by game_id if available
    prefer_fresh_duplicates: bool = True  # When deduplicating, keep fresher sample

    # Validation
    require_matching_config: bool = True  # All files must have same config_key
    require_matching_encoder: bool = True  # All files must have same encoder_version

    # Output limits
    max_samples: int | None = None  # Maximum samples in output
    target_fresh_ratio: float | None = None  # Target ratio of fresh samples (0-1)


@dataclass
class CombineResult:
    """Result of NPZ combination."""

    success: bool
    output_path: Path | None = None
    total_samples: int = 0
    samples_by_source: dict[str, int] = field(default_factory=dict)
    samples_excluded: int = 0  # Low quality or duplicates
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_npz_metadata(npz_path: Path) -> NPZMetadata:
    """Extract metadata from an NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        NPZMetadata with extracted information
    """
    path = Path(npz_path)

    with np.load(path, allow_pickle=True) as data:
        # Get sample count
        if "features" in data.files:
            sample_count = len(data["features"])
        elif "values" in data.files:
            sample_count = len(data["values"])
        else:
            sample_count = 0

        # Extract encoder info
        encoder_version = None
        if "encoder_version" in data.files:
            encoder_version = str(data["encoder_version"].item())
        elif "encoder_type" in data.files:
            encoder_version = str(data["encoder_type"].item())

        # Extract dimensions
        in_channels = None
        if "in_channels" in data.files:
            in_channels = int(data["in_channels"].item())
        elif "features" in data.files:
            in_channels = data["features"].shape[1]

        spatial_size = None
        if "spatial_size" in data.files:
            spatial_size = int(data["spatial_size"].item())

        policy_size = None
        if "policy_size" in data.files:
            policy_size = int(data["policy_size"].item())

        # Extract heuristic mode
        heuristic_mode = None
        if "heuristic_mode" in data.files:
            heuristic_mode = str(data["heuristic_mode"].item())

        # Extract timestamps from metadata dict
        export_time = None
        newest_game_time = None
        if "metadata" in data.files:
            try:
                metadata = data["metadata"].item()
                if isinstance(metadata, dict):
                    if "export_time" in metadata and metadata["export_time"]:
                        export_time = datetime.fromisoformat(
                            metadata["export_time"].replace("Z", "+00:00")
                        )
                    if "newest_game_time" in metadata and metadata["newest_game_time"]:
                        newest_game_time = datetime.fromisoformat(
                            metadata["newest_game_time"].replace("Z", "+00:00")
                        )
            except (TypeError, ValueError, KeyError):
                pass

    # Extract config from filename
    board_type = None
    num_players = None
    name = path.stem.lower()
    for bt in ["hex8", "hexagonal", "square8", "square19"]:
        if bt in name:
            board_type = bt
            break
    for np_ in [2, 3, 4]:
        if f"{np_}p" in name:
            num_players = np_
            break

    return NPZMetadata(
        path=path,
        sample_count=sample_count,
        board_type=board_type,
        num_players=num_players,
        encoder_version=encoder_version,
        in_channels=in_channels,
        spatial_size=spatial_size,
        policy_size=policy_size,
        export_time=export_time,
        newest_game_time=newest_game_time,
        heuristic_mode=heuristic_mode,
    )


def validate_npz_compatibility(
    metadata_list: list[NPZMetadata],
    config: NPZCombinerConfig,
) -> tuple[bool, str | None]:
    """Validate that NPZ files are compatible for combination.

    Args:
        metadata_list: List of NPZMetadata to validate
        config: Combiner configuration

    Returns:
        (is_valid, error_message)
    """
    if len(metadata_list) < 2:
        return True, None

    first = metadata_list[0]

    for meta in metadata_list[1:]:
        # Check config compatibility
        if config.require_matching_config:
            if first.config_key != meta.config_key:
                return False, (
                    f"Config mismatch: {first.path.name} is {first.config_key}, "
                    f"{meta.path.name} is {meta.config_key}"
                )

        # Check encoder compatibility
        if config.require_matching_encoder:
            if first.encoder_version != meta.encoder_version:
                return False, (
                    f"Encoder mismatch: {first.path.name} uses {first.encoder_version}, "
                    f"{meta.path.name} uses {meta.encoder_version}"
                )

        # Check dimension compatibility
        if first.in_channels and meta.in_channels:
            if first.in_channels != meta.in_channels:
                return False, (
                    f"Channel mismatch: {first.path.name} has {first.in_channels} channels, "
                    f"{meta.path.name} has {meta.in_channels} channels"
                )

        if first.spatial_size and meta.spatial_size:
            if first.spatial_size != meta.spatial_size:
                return False, (
                    f"Spatial size mismatch: {first.path.name} has {first.spatial_size}, "
                    f"{meta.path.name} has {meta.spatial_size}"
                )

        if first.policy_size and meta.policy_size:
            if first.policy_size != meta.policy_size:
                return False, (
                    f"Policy size mismatch: {first.path.name} has {first.policy_size}, "
                    f"{meta.path.name} has {meta.policy_size}"
                )

    return True, None


def _calculate_freshness_weight(
    age_hours: float | None,
    config: NPZCombinerConfig,
) -> float:
    """Calculate sampling weight based on freshness.

    Uses exponential decay with configurable half-life.

    Args:
        age_hours: Age of data in hours (None = unknown)
        config: Combiner configuration

    Returns:
        Weight multiplier (1.0 = normal, higher = more likely to sample)
    """
    if age_hours is None:
        return 1.0  # Unknown age = neutral weight

    if config.freshness_weight <= 1.0:
        return 1.0  # No freshness weighting

    # Exponential decay: weight = base_weight * (freshness_weight ^ -age/half_life)
    # Fresh data (age=0) gets full freshness_weight
    # Data at half_life age gets weight = 1.0
    decay = (age_hours / config.freshness_half_life_hours)
    weight = config.freshness_weight ** (1 - decay)

    return max(1.0, weight)  # Never weight below 1.0


def combine_npz_files(
    input_paths: list[str | Path],
    output_path: str | Path,
    config: NPZCombinerConfig | None = None,
) -> CombineResult:
    """Combine multiple NPZ files with quality-aware weighting.

    Args:
        input_paths: List of input NPZ file paths
        output_path: Path for combined output file
        config: Combination configuration

    Returns:
        CombineResult with combination details
    """
    config = config or NPZCombinerConfig()
    output_path = Path(output_path)

    # Extract metadata from all files
    metadata_list = []
    for path in input_paths:
        try:
            meta = extract_npz_metadata(Path(path))
            metadata_list.append(meta)
            logger.info(
                f"Found {meta.sample_count} samples in {meta.path.name} "
                f"(age: {meta.age_hours:.1f}h)" if meta.age_hours else
                f"Found {meta.sample_count} samples in {meta.path.name}"
            )
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to read {path}: {e}")

    if not metadata_list:
        return CombineResult(
            success=False,
            error="No valid NPZ files found",
        )

    # Validate compatibility
    is_valid, error = validate_npz_compatibility(metadata_list, config)
    if not is_valid:
        return CombineResult(
            success=False,
            error=error,
        )

    # Calculate per-file sampling weights
    file_weights = {}
    for meta in metadata_list:
        weight = _calculate_freshness_weight(meta.age_hours, config)
        file_weights[str(meta.path)] = weight
        if weight > 1.0:
            logger.info(f"{meta.path.name}: freshness weight = {weight:.2f}x")

    # Collect samples with weighting
    all_arrays: dict[str, list[np.ndarray]] = {}
    samples_by_source: dict[str, int] = {}
    samples_excluded = 0
    seen_game_ids: set[str] = set()

    for meta in metadata_list:
        path = meta.path
        weight = file_weights[str(path)]

        with np.load(path, allow_pickle=True) as data:
            # Get quality scores if available
            quality_scores = None
            if "quality_score" in data.files:
                quality_scores = data["quality_score"]

            # Get game IDs for deduplication
            game_ids = None
            if config.deduplicate and "game_id" in data.files:
                game_ids = data["game_id"]

            # Build mask for samples to include
            n_samples = meta.sample_count
            include_mask = np.ones(n_samples, dtype=bool)

            # Filter by quality
            if quality_scores is not None and config.min_quality_score > 0:
                include_mask &= (quality_scores >= config.min_quality_score)

            # Deduplicate
            if game_ids is not None:
                for i, gid in enumerate(game_ids):
                    gid_str = str(gid)
                    if gid_str in seen_game_ids:
                        include_mask[i] = False
                    else:
                        seen_game_ids.add(gid_str)

            # Count exclusions
            n_excluded = n_samples - np.sum(include_mask)
            samples_excluded += n_excluded

            # Apply freshness weighting via oversampling
            if weight > 1.0:
                # Duplicate included indices by weight
                include_indices = np.where(include_mask)[0]
                n_to_sample = int(len(include_indices) * weight)
                rng = np.random.default_rng()
                sampled_indices = rng.choice(
                    include_indices,
                    size=min(n_to_sample, len(include_indices) * 3),  # Cap at 3x
                    replace=True,
                )
            else:
                sampled_indices = np.where(include_mask)[0]

            # Collect arrays
            for key in data.files:
                if key not in all_arrays:
                    all_arrays[key] = []

                arr = data[key]
                if len(arr.shape) >= 1 and arr.shape[0] == n_samples:
                    # This is a per-sample array
                    all_arrays[key].append(arr[sampled_indices])
                else:
                    # This is metadata (scalar or fixed array)
                    # Only keep from first file
                    if len(all_arrays[key]) == 0:
                        all_arrays[key].append(arr)

            samples_by_source[path.name] = len(sampled_indices)

    # Concatenate arrays
    save_kwargs = {}
    total_samples = 0

    for key, arrays in all_arrays.items():
        if len(arrays) == 0:
            continue

        # Check if this is per-sample data
        first_arr = arrays[0]
        if len(arrays) > 1 and len(first_arr.shape) >= 1:
            try:
                combined = np.concatenate(arrays, axis=0)
                save_kwargs[key] = combined
                if key == "features":
                    total_samples = len(combined)
            except ValueError:
                # Can't concatenate (different shapes), use first
                save_kwargs[key] = first_arr
        else:
            save_kwargs[key] = first_arr

    # Apply max_samples limit
    if config.max_samples and total_samples > config.max_samples:
        rng = np.random.default_rng()
        keep_indices = rng.choice(
            total_samples,
            size=config.max_samples,
            replace=False,
        )
        keep_indices.sort()

        for key, arr in save_kwargs.items():
            if isinstance(arr, np.ndarray) and len(arr.shape) >= 1 and arr.shape[0] == total_samples:
                save_kwargs[key] = arr[keep_indices]

        total_samples = config.max_samples

    # Update metadata
    first_meta = metadata_list[0]
    save_kwargs["metadata"] = np.asarray({
        "export_time": datetime.now(tz=timezone.utc).isoformat(),
        "newest_game_time": max(
            (m.newest_game_time for m in metadata_list if m.newest_game_time),
            default=None,
        ),
        "game_count": None,  # Unknown after combination
        "sample_count": total_samples,
        "source_files": [str(m.path) for m in metadata_list],
        "combination_config": {
            "freshness_weight": config.freshness_weight,
            "min_quality_score": config.min_quality_score,
        },
    })

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined file
    np.savez_compressed(output_path, **save_kwargs)

    logger.info(
        f"Combined {total_samples} samples from {len(metadata_list)} files → {output_path}"
    )

    return CombineResult(
        success=True,
        output_path=output_path,
        total_samples=total_samples,
        samples_by_source=samples_by_source,
        samples_excluded=samples_excluded,
        metadata={
            "config_key": first_meta.config_key,
            "encoder_version": first_meta.encoder_version,
            "source_count": len(metadata_list),
        },
    )


def discover_and_combine_for_config(
    config_key: str,
    output_path: str | Path,
    data_dirs: list[str | Path] | None = None,
    combiner_config: NPZCombinerConfig | None = None,
) -> CombineResult:
    """Discover all NPZ files for a config and combine them.

    Args:
        config_key: Config key (e.g., 'hex8_2p')
        output_path: Path for combined output file
        data_dirs: Directories to search (default: data/training, data/exports)
        combiner_config: Combination configuration

    Returns:
        CombineResult with combination details
    """
    from pathlib import Path

    if data_dirs is None:
        ai_service_root = Path(__file__).resolve().parents[2]
        data_dirs = [
            ai_service_root / "data" / "training",
            ai_service_root / "data" / "exports",
        ]

    # Discover matching NPZ files
    input_paths = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            continue

        for npz_path in data_dir.glob("**/*.npz"):
            name = npz_path.stem.lower()
            if config_key.lower() in name:
                input_paths.append(npz_path)

    if not input_paths:
        return CombineResult(
            success=False,
            error=f"No NPZ files found for config {config_key}",
        )

    # Sort by modification time (newest first)
    input_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    logger.info(f"Found {len(input_paths)} NPZ files for {config_key}")

    return combine_npz_files(
        input_paths=input_paths,
        output_path=output_path,
        config=combiner_config,
    )

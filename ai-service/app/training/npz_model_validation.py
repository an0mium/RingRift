"""NPZ-to-Model Validation - Ensures NPZ files match neural network architecture.

This module validates that training data (NPZ files) is compatible with the
target neural network model before training starts.

Validates:
- Feature channel count matches model input channels
- Spatial dimensions match board type
- Policy size matches model output
- Encoder version is compatible with model version
- Heuristic mode matches model requirements (v5/v5-heavy)

Usage:
    from app.training.npz_model_validation import (
        validate_npz_for_model,
        NPZModelValidationResult,
        get_expected_dimensions,
    )

    # Validate before training
    result = validate_npz_for_model(
        npz_path='data/training/hex8_2p.npz',
        board_type='hex8',
        num_players=2,
        model_version='v5',
    )
    if not result.valid:
        raise ValueError(f"NPZ validation failed: {result.errors}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Expected dimensions by board type
BOARD_DIMENSIONS = {
    "hex8": {
        "spatial_size": 9,  # 9x9 grid for radius 4
        "num_cells": 61,    # Actual cells (hexagonal)
        "policy_size": 122,  # 61 cells * 2 (place + ring)
    },
    "square8": {
        "spatial_size": 8,
        "num_cells": 64,
        "policy_size": 128,  # 64 * 2
    },
    "square19": {
        "spatial_size": 19,
        "num_cells": 361,
        "policy_size": 722,  # 361 * 2
    },
    "hexagonal": {
        "spatial_size": 25,  # 25x25 grid for radius 12
        "num_cells": 469,
        "policy_size": 938,  # 469 * 2
    },
}

# Expected channels by model version
MODEL_CHANNELS = {
    "v2": {
        "base_channels": 21,  # Fast heuristic features
        "per_player_channels": 4,  # Per-player state
    },
    "v3": {
        "base_channels": 25,
        "per_player_channels": 4,
    },
    "v4": {
        "base_channels": 25,
        "per_player_channels": 4,
    },
    "v5": {
        "base_channels": 21,  # Fast heuristics
        "per_player_channels": 4,
        "heuristic_mode": "fast",
    },
    "v5-heavy": {
        "base_channels": 49,  # Full heuristics
        "per_player_channels": 4,
        "heuristic_mode": "full",
    },
    "v6": {
        "base_channels": 21,
        "per_player_channels": 4,
    },
    "v6-xl": {
        "base_channels": 21,
        "per_player_channels": 4,
    },
}


@dataclass
class ExpectedDimensions:
    """Expected dimensions for a configuration."""

    board_type: str
    num_players: int
    model_version: str
    spatial_size: int
    num_cells: int
    policy_size: int
    in_channels: int
    heuristic_mode: str | None = None


@dataclass
class NPZModelValidationResult:
    """Result of NPZ-to-model validation."""

    valid: bool
    npz_path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    actual: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid


def get_expected_dimensions(
    board_type: str,
    num_players: int,
    model_version: str,
) -> ExpectedDimensions:
    """Get expected dimensions for a configuration.

    Args:
        board_type: Board type (hex8, square8, square19, hexagonal)
        num_players: Number of players (2, 3, 4)
        model_version: Model version (v2, v3, v4, v5, v5-heavy, v6, v6-xl)

    Returns:
        ExpectedDimensions dataclass
    """
    if board_type not in BOARD_DIMENSIONS:
        raise ValueError(f"Unknown board type: {board_type}")

    board_dims = BOARD_DIMENSIONS[board_type]

    # Normalize model version
    model_version = model_version.lower().replace("_", "-")
    if model_version not in MODEL_CHANNELS:
        # Try to find closest match
        if model_version.startswith("v5") and "heavy" in model_version:
            model_version = "v5-heavy"
        elif model_version.startswith("v6") and "xl" in model_version:
            model_version = "v6-xl"
        elif model_version.startswith("v"):
            # Default to base version
            base = model_version.split("-")[0]
            if base in MODEL_CHANNELS:
                model_version = base

    if model_version not in MODEL_CHANNELS:
        raise ValueError(f"Unknown model version: {model_version}")

    model_info = MODEL_CHANNELS[model_version]

    # Calculate expected channels
    # Formula: base_channels + (num_players * per_player_channels)
    in_channels = model_info["base_channels"] + (
        num_players * model_info["per_player_channels"]
    )

    return ExpectedDimensions(
        board_type=board_type,
        num_players=num_players,
        model_version=model_version,
        spatial_size=board_dims["spatial_size"],
        num_cells=board_dims["num_cells"],
        policy_size=board_dims["policy_size"],
        in_channels=in_channels,
        heuristic_mode=model_info.get("heuristic_mode"),
    )


def validate_npz_for_model(
    npz_path: str | Path,
    board_type: str,
    num_players: int,
    model_version: str,
    strict: bool = True,
) -> NPZModelValidationResult:
    """Validate that an NPZ file is compatible with the target model.

    Args:
        npz_path: Path to NPZ file
        board_type: Expected board type
        num_players: Expected number of players
        model_version: Target model version
        strict: If True, treat warnings as errors

    Returns:
        NPZModelValidationResult with validation details
    """
    npz_path = Path(npz_path)
    errors = []
    warnings = []
    actual = {}
    expected_dims = {}

    try:
        expected = get_expected_dimensions(board_type, num_players, model_version)
        expected_dims = {
            "board_type": expected.board_type,
            "num_players": expected.num_players,
            "model_version": expected.model_version,
            "spatial_size": expected.spatial_size,
            "in_channels": expected.in_channels,
            "policy_size": expected.policy_size,
            "heuristic_mode": expected.heuristic_mode,
        }
    except ValueError as e:
        errors.append(str(e))
        return NPZModelValidationResult(
            valid=False,
            npz_path=npz_path,
            errors=errors,
            expected=expected_dims,
        )

    # Read NPZ file
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # Extract actual dimensions
            if "features" in data.files:
                features = data["features"]
                actual["sample_count"] = features.shape[0]
                actual["in_channels"] = features.shape[1]
                if len(features.shape) >= 3:
                    actual["spatial_size"] = features.shape[2]
            else:
                errors.append("NPZ file missing 'features' array")

            # Extract encoder version
            if "encoder_version" in data.files:
                actual["encoder_version"] = str(data["encoder_version"].item())
            elif "encoder_type" in data.files:
                actual["encoder_version"] = str(data["encoder_type"].item())

            # Extract policy size
            if "policy_size" in data.files:
                actual["policy_size"] = int(data["policy_size"].item())
            elif "policy_logits" in data.files:
                actual["policy_size"] = data["policy_logits"].shape[1]

            # Extract heuristic mode
            if "heuristic_mode" in data.files:
                actual["heuristic_mode"] = str(data["heuristic_mode"].item())

            # Extract in_channels from metadata if available
            if "in_channels" in data.files:
                actual["in_channels_metadata"] = int(data["in_channels"].item())

    except (OSError, ValueError, KeyError) as e:
        errors.append(f"Failed to read NPZ file: {e}")
        return NPZModelValidationResult(
            valid=False,
            npz_path=npz_path,
            errors=errors,
            actual=actual,
            expected=expected_dims,
        )

    # Validate channel count
    if "in_channels" in actual:
        if actual["in_channels"] != expected.in_channels:
            errors.append(
                f"Channel mismatch: NPZ has {actual['in_channels']} channels, "
                f"model {model_version} expects {expected.in_channels} "
                f"(base={MODEL_CHANNELS[expected.model_version]['base_channels']} + "
                f"{num_players}*{MODEL_CHANNELS[expected.model_version]['per_player_channels']})"
            )

    # Validate spatial size
    if "spatial_size" in actual:
        if actual["spatial_size"] != expected.spatial_size:
            errors.append(
                f"Spatial size mismatch: NPZ has {actual['spatial_size']}, "
                f"board {board_type} expects {expected.spatial_size}"
            )

    # Validate policy size
    if "policy_size" in actual:
        if actual["policy_size"] != expected.policy_size:
            # This might be a different policy encoding
            warnings.append(
                f"Policy size mismatch: NPZ has {actual['policy_size']}, "
                f"expected {expected.policy_size} for {board_type}"
            )

    # Validate heuristic mode for v5/v5-heavy
    if expected.heuristic_mode:
        actual_mode = actual.get("heuristic_mode")
        if actual_mode and actual_mode != expected.heuristic_mode:
            errors.append(
                f"Heuristic mode mismatch: NPZ uses '{actual_mode}' mode, "
                f"model {model_version} requires '{expected.heuristic_mode}' mode"
            )
        elif not actual_mode and expected.heuristic_mode == "full":
            warnings.append(
                f"NPZ doesn't specify heuristic_mode, but model {model_version} "
                f"requires 'full' mode (49 heuristic features)"
            )

    # Validate encoder version compatibility
    if "encoder_version" in actual:
        actual_version = actual["encoder_version"].lower()
        expected_version = expected.model_version.split("-")[0]  # v5-heavy -> v5

        # v2/v3 are compatible, v4+ are compatible
        if actual_version.startswith("v2") or actual_version.startswith("v3"):
            if not (expected_version.startswith("v2") or expected_version.startswith("v3")):
                warnings.append(
                    f"Encoder version mismatch: NPZ uses {actual['encoder_version']}, "
                    f"model is {model_version}. May have feature alignment issues."
                )
        elif actual_version.startswith("v4") or actual_version.startswith("v5") or actual_version.startswith("v6"):
            if expected_version.startswith("v2") or expected_version.startswith("v3"):
                warnings.append(
                    f"Encoder version mismatch: NPZ uses {actual['encoder_version']}, "
                    f"model is {model_version}. May have feature alignment issues."
                )

    # Convert warnings to errors in strict mode
    if strict:
        errors.extend(warnings)
        warnings = []

    is_valid = len(errors) == 0

    if errors:
        logger.warning(
            f"NPZ validation failed for {npz_path.name}: {'; '.join(errors)}"
        )
    elif warnings:
        logger.info(
            f"NPZ validation passed with warnings for {npz_path.name}: "
            f"{'; '.join(warnings)}"
        )
    else:
        logger.debug(f"NPZ validation passed for {npz_path.name}")

    return NPZModelValidationResult(
        valid=is_valid,
        npz_path=npz_path,
        errors=errors,
        warnings=warnings,
        actual=actual,
        expected=expected_dims,
    )


def validate_before_training(
    data_path: str | Path,
    board_type: str,
    num_players: int,
    model_version: str,
    fail_on_warning: bool = False,
) -> None:
    """Validate NPZ file before training, raising ValueError on failure.

    This is intended to be called at the start of training to catch
    configuration mismatches early.

    Args:
        data_path: Path to NPZ file
        board_type: Expected board type
        num_players: Expected number of players
        model_version: Target model version
        fail_on_warning: If True, warnings also cause failure

    Raises:
        ValueError: If validation fails
        FileNotFoundError: If NPZ file doesn't exist
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    result = validate_npz_for_model(
        npz_path=data_path,
        board_type=board_type,
        num_players=num_players,
        model_version=model_version,
        strict=fail_on_warning,
    )

    if not result.valid:
        error_msg = (
            f"Training data validation failed for {data_path.name}:\n"
            + "\n".join(f"  - {e}" for e in result.errors)
        )
        if result.actual:
            error_msg += f"\n\nActual: {result.actual}"
        if result.expected:
            error_msg += f"\nExpected: {result.expected}"

        raise ValueError(error_msg)

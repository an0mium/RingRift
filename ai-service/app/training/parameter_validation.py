"""Parameter Validation for Training Module.

This module provides validation utilities for training parameters, models, and
datasets. It consolidates validation logic from train.py to improve testability
and reusability.

December 2025: Created for Phase 6A of train.py refactoring.

Usage:
    from app.training.parameter_validation import (
        validate_training_compatibility,
        validate_model_value_head,
        ValidationResult,
    )

    # Validate model and dataset compatibility
    validate_training_compatibility(model, dataset, config)

    # Validate model value head
    validate_model_value_head(model, expected_players=4)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        valid: Whether validation passed
        errors: List of validation error messages
        warnings: List of validation warning messages
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)

    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.valid:
            raise ValueError("\n".join(self.errors))


def validate_policy_size_compatibility(
    model_policy_size: int | None,
    dataset_policy_size: int | None,
) -> ValidationResult:
    """Validate policy size compatibility between model and dataset.

    Args:
        model_policy_size: Policy size from model (None if unknown)
        dataset_policy_size: Policy size from dataset (None if unknown)

    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult()

    if model_policy_size is None or dataset_policy_size is None:
        return result

    if dataset_policy_size > model_policy_size:
        result.add_error(
            f"Dataset policy_size ({dataset_policy_size}) > model policy_size ({model_policy_size}). "
            f"Dataset contains indices the model cannot predict. "
            f"Check board type settings and encoder version."
        )
    elif dataset_policy_size < model_policy_size:
        result.add_warning(
            f"Dataset policy_size ({dataset_policy_size}) < model policy_size ({model_policy_size}). "
            f"Policy targets will be zero-padded (this is normal)."
        )

    return result


def validate_board_type_compatibility(
    model_board_type: str | None,
    dataset_board_type: str | None,
) -> ValidationResult:
    """Validate board type compatibility between model and dataset.

    Args:
        model_board_type: Board type from model (None if unknown)
        dataset_board_type: Board type from dataset (None if unknown)

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult()

    if model_board_type is None or dataset_board_type is None:
        return result

    if model_board_type != dataset_board_type:
        result.add_error(
            f"[CROSS-CONFIG CONTAMINATION] Board type mismatch: "
            f"model expects '{model_board_type}', dataset contains '{dataset_board_type}'. "
            f"This would produce a garbage model. "
            f"Use --board-type to specify the correct board type, or regenerate training data."
        )

    return result


def validate_sample_data(
    dataset: Any,
    num_samples: int = 10,
    policy_size: int = 4500,
) -> ValidationResult:
    """Validate sample data from dataset.

    Args:
        dataset: Training dataset to validate
        num_samples: Number of samples to check
        policy_size: Expected policy size

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult()

    num_samples_to_check = min(num_samples, len(dataset))
    invalid_samples: list[tuple[int, str]] = []

    for i in range(num_samples_to_check):
        try:
            sample = dataset[i]
            # Handle different return formats
            if isinstance(sample, tuple) and len(sample) >= 4:
                _, _, _, policy = sample[:4]
            else:
                continue

            # Check policy vector
            if hasattr(policy, "sum"):
                policy_sum = policy.sum().item()
                if policy_sum > 0 and not (0.5 < policy_sum < 1.5):
                    invalid_samples.append((i, f"policy_sum={policy_sum:.4f}"))
        except Exception as e:
            invalid_samples.append((i, f"error: {e}"))

    if invalid_samples:
        sample_details = ", ".join(f"[{i}]: {msg}" for i, msg in invalid_samples[:5])
        result.add_warning(
            f"Found {len(invalid_samples)} potentially invalid samples: {sample_details}"
        )

    return result


def validate_training_compatibility(
    model: "nn.Module",
    dataset: Any,
    config: Any,
) -> None:
    """Validate model and dataset are compatible before training.

    This function catches common issues early to prevent wasted GPU hours:
    - Policy size mismatches between model and data
    - Board type incompatibility
    - Invalid sample data

    Args:
        model: The neural network model to train
        dataset: The training dataset (RingRiftDataset or similar)
        config: Training configuration

    Raises:
        ValueError: If model/dataset are incompatible or data validation fails
    """
    logger.info("Running training compatibility validation...")

    # Get sizes
    model_policy_size = getattr(model, "policy_size", None)
    dataset_policy_size = getattr(dataset, "policy_size", None)

    # 1. Policy size compatibility
    policy_result = validate_policy_size_compatibility(
        model_policy_size, dataset_policy_size
    )
    for warning in policy_result.warnings:
        logger.info(warning)
    policy_result.raise_if_invalid()

    # 2. Board type compatibility
    model_board_type = getattr(model, "board_type", None)
    dataset_board_type = getattr(dataset, "board_type", None)

    board_result = validate_board_type_compatibility(
        model_board_type, dataset_board_type
    )
    board_result.raise_if_invalid()

    # 3. Sample validation
    policy_size = model_policy_size or dataset_policy_size or 4500
    sample_result = validate_sample_data(dataset, num_samples=10, policy_size=policy_size)
    for warning in sample_result.warnings:
        logger.warning(warning)

    logger.info("Training compatibility validation passed")


def validate_model_value_head(
    model: "nn.Module",
    expected_players: int,
    context: str = "",
) -> None:
    """Validate model value head matches expected player count.

    This prevents training with mismatched value head dimensions, which was
    a root cause of cluster model failures (hex8_4p, square19_3p regressions).

    Args:
        model: Neural network model to validate
        expected_players: Expected number of players (2, 3, or 4)
        context: Description of when validation is happening (for error messages)

    Raises:
        ValueError: If model value head doesn't match expected player count
    """
    import torch.nn as nn

    ctx = f" ({context})" if context else ""

    # Check model's num_players attribute if present
    if hasattr(model, "num_players"):
        model_players = model.num_players
        if model_players != expected_players:
            raise ValueError(
                f"Model value head mismatch{ctx}: model.num_players={model_players} "
                f"but training expects {expected_players} players. "
                f"Use transfer_2p_to_4p.py to resize value head."
            )

    # Check value head output dimension
    # v4/v5-heavy use 3-layer value head (fc1 → fc2 → fc3), others use 2-layer (fc1 → fc2)
    # Check the final layer that outputs to num_players
    final_value_layer = None
    if hasattr(model, "value_fc3"):
        # v4/v5-heavy: value_fc3 is the final output layer
        final_value_layer = model.value_fc3
    elif hasattr(model, "value_fc2"):
        # v2/v3: value_fc2 is the final output layer
        final_value_layer = model.value_fc2

    if final_value_layer is not None:
        out_features = final_value_layer.out_features
        if out_features != expected_players:
            layer_name = "value_fc3" if hasattr(model, "value_fc3") else "value_fc2"
            raise ValueError(
                f"{layer_name} output mismatch{ctx}: out_features={out_features} "
                f"but training expects {expected_players} players. "
                f"Use transfer_2p_to_4p.py to resize value head."
            )

    # Check value_head output dimension (used in some architectures)
    if hasattr(model, "value_head"):
        # value_head might be a Sequential or Linear
        value_head = model.value_head
        if hasattr(value_head, "out_features"):
            out_features = value_head.out_features
            if out_features != expected_players:
                raise ValueError(
                    f"value_head output mismatch{ctx}: out_features={out_features} "
                    f"but training expects {expected_players} players."
                )
        elif isinstance(value_head, nn.Sequential):
            # Check last layer of Sequential
            last_layer = None
            for layer in value_head:
                if hasattr(layer, "out_features"):
                    last_layer = layer
            if last_layer is not None and last_layer.out_features != expected_players:
                raise ValueError(
                    f"value_head Sequential output mismatch{ctx}: "
                    f"out_features={last_layer.out_features} "
                    f"but training expects {expected_players} players."
                )


def validate_architecture_data_compatibility(
    model_version: str,
    detected_num_heuristics: int | None,
    board_type: str,
    data_path: str | None = None,
) -> None:
    """Validate training data is compatible with selected architecture.

    This catches errors early before expensive model initialization:
    - V5-heavy requires at least 21 heuristic features (fast heuristics)
    - V6 requires all 49 heuristic features (full heuristics)

    Args:
        model_version: Model version string (e.g., 'v5-heavy', 'v6')
        detected_num_heuristics: Number of heuristic features in dataset
        board_type: Board type name (e.g., 'hex8', 'square8')
        data_path: Path to training data (for error messages)

    Raises:
        ValueError: If data is incompatible with selected architecture
    """
    # Only validate for architectures that require heuristics
    requires_heuristics_versions = ("v5", "v5-gnn", "v5-heavy", "v6")
    if model_version not in requires_heuristics_versions:
        return

    # Import encoder registry to get requirements
    try:
        from app.training.encoder_registry import get_encoder_config

        version_key = "v6" if model_version == "v6" else "v5-heavy"
        encoder_config = get_encoder_config(board_type, version_key)
    except (ValueError, ImportError):
        # Registry doesn't have this config, skip validation
        return

    # Check if architecture requires heuristics
    if not encoder_config.requires_heuristics:
        return

    min_required = encoder_config.min_heuristic_features
    actual_heuristics = detected_num_heuristics or 0

    if actual_heuristics < min_required:
        version_name = "V6" if model_version == "v6" else "V5-Heavy"
        data_path_str = data_path or "unknown"
        raise ValueError(
            f"\n{'='*70}\n"
            f"ARCHITECTURE-DATA COMPATIBILITY ERROR\n"
            f"{'='*70}\n\n"
            f"Model: {version_name} (--model-version {model_version})\n"
            f"  - Requires at least {min_required} heuristic features\n\n"
            f"Dataset: {data_path_str}\n"
            f"  - Has {actual_heuristics} heuristic features\n\n"
            f"SOLUTIONS:\n"
            f"  1. Re-export data with --full-heuristics flag:\n"
            f"     python scripts/export_replay_dataset.py --full-heuristics ...\n"
            f"  2. Use a different architecture that doesn't require heuristics:\n"
            f"     --model-version v2 or --model-version v4\n"
            f"{'='*70}"
        )


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    model: "nn.Module",
    strict: bool = True,
) -> ValidationResult:
    """Validate checkpoint is compatible with model.

    Args:
        checkpoint: Loaded checkpoint dictionary
        model: Target model to load checkpoint into
        strict: Whether to require exact match

    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult()

    if "model_state_dict" not in checkpoint:
        result.add_error("Checkpoint missing 'model_state_dict'")
        return result

    state_dict = checkpoint["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    if strict:
        if missing_keys:
            result.add_error(f"Missing keys in checkpoint: {list(missing_keys)[:5]}...")
        if unexpected_keys:
            result.add_error(
                f"Unexpected keys in checkpoint: {list(unexpected_keys)[:5]}..."
            )
    else:
        if missing_keys:
            result.add_warning(
                f"Missing keys in checkpoint (will be randomly initialized): "
                f"{list(missing_keys)[:5]}..."
            )
        if unexpected_keys:
            result.add_warning(
                f"Unexpected keys in checkpoint (will be ignored): "
                f"{list(unexpected_keys)[:5]}..."
            )

    return result

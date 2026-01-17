"""Model configuration contract for validation.

This module provides a systematic validation system to prevent model
configuration mismatches. It ensures models are saved and promoted
only when their configuration matches the expected contract for the
given (board_type, num_players) combination.

Key concepts:
- ModelConfigContract: Immutable definition of expected model config
- validate_model_for_save: Pre-save validation (fails fast on mismatch)
- validate_checkpoint_for_promotion: Pre-promotion validation

Usage:
    from app.training.model_config_contract import (
        ModelConfigContract,
        ModelConfigError,
        validate_model_for_save,
    )

    # Validate before saving
    validate_model_for_save(model, BoardType.SQUARE8, num_players=3)

    # Create contract for validation
    contract = ModelConfigContract.for_config(BoardType.SQUARE8, 3)
    violations = contract.validate_model(model)
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from app.ai.neural_net.constants import BOARD_POLICY_SIZES
from app.models import BoardType

logger = logging.getLogger(__name__)


class ModelConfigError(Exception):
    """Raised when model configuration violates contract.

    This exception indicates a fundamental mismatch between the model's
    architecture and the expected configuration. Common causes:
    - Wrong num_players (value head has wrong output dimension)
    - Legacy policy encoding (policy head has wrong size)
    - Board type mismatch (spatial dimensions don't match)
    """

    def __init__(self, message: str, violations: list[str] | None = None):
        super().__init__(message)
        self.violations = violations or []


@dataclass(frozen=True)
class ModelConfigContract:
    """Immutable contract defining expected model configuration.

    A contract specifies the expected configuration for a model
    targeting a specific (board_type, num_players) combination.
    All models claiming to support this configuration must satisfy
    the contract's requirements.

    Attributes:
        board_type: Target board type (square8, hex8, etc.)
        num_players: Number of players (2, 3, or 4)
        policy_size: Expected policy head output dimension
        value_head_outputs: Expected value head output dimension (== num_players)
    """

    board_type: BoardType
    num_players: int
    policy_size: int
    value_head_outputs: int

    @classmethod
    def for_config(cls, board_type: BoardType, num_players: int) -> "ModelConfigContract":
        """Create a contract for the given configuration.

        Args:
            board_type: Target board type
            num_players: Number of players (2, 3, or 4)

        Returns:
            Immutable contract for this configuration

        Raises:
            ValueError: If num_players is not 2, 3, or 4
        """
        if num_players not in (2, 3, 4):
            raise ValueError(f"num_players must be 2, 3, or 4, got {num_players}")

        policy_size = BOARD_POLICY_SIZES.get(board_type)
        if policy_size is None:
            raise ValueError(f"Unknown board type: {board_type}")

        return cls(
            board_type=board_type,
            num_players=num_players,
            policy_size=policy_size,
            value_head_outputs=num_players,
        )

    def validate_model(self, model: nn.Module) -> list[str]:
        """Validate a model against this contract.

        Checks that the model's architecture matches the contract's
        requirements for value head and policy head dimensions.

        Args:
            model: PyTorch model to validate

        Returns:
            List of violation descriptions. Empty list if valid.
        """
        violations = []

        # Check value head output dimension
        # v5_heavy uses value_fc3 as final, older models use value_fc2
        value_final_layer = None
        if hasattr(model, 'value_fc3'):
            value_final_layer = model.value_fc3
        elif hasattr(model, 'value_fc2'):
            value_final_layer = model.value_fc2

        if value_final_layer is not None:
            actual_outputs = value_final_layer.weight.shape[0]
            if actual_outputs != self.value_head_outputs:
                violations.append(
                    f"Value head outputs: expected {self.value_head_outputs}, "
                    f"got {actual_outputs} (num_players mismatch - model has "
                    f"{actual_outputs}-player value head but config expects {self.num_players})"
                )

        # Check policy head output dimension
        # Policy head structure varies by model architecture
        policy_size = self._get_policy_size(model)
        if policy_size is not None and policy_size != self.policy_size:
            # Only flag if significantly different (legacy encoding is ~8x larger)
            if policy_size > self.policy_size * 2:
                violations.append(
                    f"Policy size: expected {self.policy_size}, got {policy_size} "
                    f"(likely legacy_max_n encoding - retrain with board-aware data)"
                )
            elif policy_size < self.policy_size // 2:
                violations.append(
                    f"Policy size: expected {self.policy_size}, got {policy_size} "
                    f"(model was trained for different board type)"
                )

        return violations

    def _get_policy_size(self, model: nn.Module) -> int | None:
        """Extract policy output size from model architecture."""
        # Try common policy head structures

        # CNN v2/v3/v4 style: policy_head with fc layer
        if hasattr(model, 'policy_head'):
            policy_head = model.policy_head
            if hasattr(policy_head, 'fc'):
                return policy_head.fc.weight.shape[0]
            if hasattr(policy_head, 'out_features'):
                return policy_head.out_features

        # Hex v2/v3 style: policy_fc final layer
        if hasattr(model, 'policy_fc'):
            return model.policy_fc.weight.shape[0]

        # Square v2 style: policy_fc2 is the final policy layer
        if hasattr(model, 'policy_fc2'):
            return model.policy_fc2.weight.shape[0]

        # Fallback: look for the largest policy layer output
        # (the final policy layer typically has the largest output = policy_size)
        max_policy_size = None
        for name, module in model.named_modules():
            if 'policy' in name.lower() and hasattr(module, 'weight'):
                if len(module.weight.shape) >= 2:
                    size = module.weight.shape[0]
                    if max_policy_size is None or size > max_policy_size:
                        max_policy_size = size

        return max_policy_size

    def validate_checkpoint_metadata(self, metadata: dict[str, Any]) -> list[str]:
        """Validate checkpoint metadata against this contract.

        Args:
            metadata: Checkpoint metadata dict (from ModelMetadata.to_dict())

        Returns:
            List of violation descriptions. Empty list if valid.
        """
        violations = []

        config = metadata.get('config', {})

        # Check num_players
        checkpoint_num_players = config.get('num_players')
        if checkpoint_num_players is not None and checkpoint_num_players != self.num_players:
            violations.append(
                f"num_players: expected {self.num_players}, "
                f"got {checkpoint_num_players} (checkpoint has wrong player count)"
            )

        # Check policy_size
        checkpoint_policy_size = config.get('policy_size')
        if checkpoint_policy_size is not None:
            if checkpoint_policy_size > self.policy_size * 2:
                violations.append(
                    f"policy_size: expected {self.policy_size}, "
                    f"got {checkpoint_policy_size} (legacy encoding detected)"
                )
            elif checkpoint_policy_size < self.policy_size // 2:
                violations.append(
                    f"policy_size: expected {self.policy_size}, "
                    f"got {checkpoint_policy_size} (wrong board type)"
                )

        # Check board_type if present
        checkpoint_board_type = config.get('board_type')
        if checkpoint_board_type is not None:
            # Handle both string and BoardType
            expected_str = self.board_type.value if hasattr(self.board_type, 'value') else str(self.board_type)
            checkpoint_str = checkpoint_board_type.value if hasattr(checkpoint_board_type, 'value') else str(checkpoint_board_type)

            if checkpoint_str != expected_str:
                violations.append(
                    f"board_type: expected {expected_str}, "
                    f"got {checkpoint_str} (checkpoint is for different board)"
                )

        return violations

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ModelConfigContract("
            f"board={self.board_type.value}, "
            f"players={self.num_players}, "
            f"policy={self.policy_size}, "
            f"value_outputs={self.value_head_outputs})"
        )


def validate_model_for_save(
    model: nn.Module,
    board_type: BoardType,
    num_players: int,
    strict: bool = True,
) -> list[str]:
    """Validate model configuration before saving.

    This is the primary validation entry point. Call this before
    saving any model checkpoint to ensure configuration correctness.

    Args:
        model: PyTorch model to validate
        board_type: Expected board type
        num_players: Expected number of players
        strict: If True, raise exception on violations. If False, log warnings.

    Returns:
        List of violations (empty if valid)

    Raises:
        ModelConfigError: If strict=True and violations found
    """
    contract = ModelConfigContract.for_config(board_type, num_players)
    violations = contract.validate_model(model)

    if violations:
        msg = (
            f"Model configuration violates contract for "
            f"{board_type.value}_{num_players}p:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nThe model cannot be saved with this configuration. "
            + "Please ensure the model was created with matching parameters."
        )
        if strict:
            raise ModelConfigError(msg, violations)
        else:
            logger.warning(msg)

    return violations


def validate_checkpoint_for_promotion(
    metadata: dict[str, Any],
    board_type: BoardType,
    num_players: int,
) -> tuple[bool, list[str]]:
    """Validate checkpoint metadata before promotion to canonical.

    Call this before promoting any model to canonical status.

    Args:
        metadata: Checkpoint metadata dict
        board_type: Target board type
        num_players: Target number of players

    Returns:
        Tuple of (is_valid, violations)
    """
    contract = ModelConfigContract.for_config(board_type, num_players)
    violations = contract.validate_checkpoint_metadata(metadata)
    return len(violations) == 0, violations


# Convenience function for canonical model names
def get_canonical_model_name(board_type: BoardType, num_players: int) -> str:
    """Get the canonical model filename for a configuration.

    Args:
        board_type: Board type
        num_players: Number of players

    Returns:
        Canonical filename like "canonical_hex8_3p.pth"
    """
    return f"canonical_{board_type.value}_{num_players}p.pth"


def validate_model_path_for_config(
    model_path: str,
    expected_board_type: str,
    expected_num_players: int,
    check_architecture: bool = False,
) -> tuple[bool, list[str]]:
    """Validate model file path is appropriate for given config.

    This function performs fast, filename-based validation to catch
    model-config mismatches without loading the model weights.

    January 2026: Added as part of model-config validation system to prevent
    Elo tracking corruption from model-config mismatches.

    Args:
        model_path: Path to model file
        expected_board_type: Expected board type (e.g., "hex8", "square8")
        expected_num_players: Expected number of players (2, 3, or 4)
        check_architecture: If True, load model and verify value head outputs.
            This is slower but catches architecture mismatches.

    Returns:
        Tuple of (is_valid, list_of_violations)

    Example:
        >>> is_valid, violations = validate_model_path_for_config(
        ...     "models/canonical_hex8_2p.pth",
        ...     expected_board_type="hex8",
        ...     expected_num_players=2,
        ... )
        >>> assert is_valid and not violations
    """
    from app.training.validated_model_ref import (
        ValidatedModelRef,
        ModelConfigError as RefError,
    )

    violations: list[str] = []

    # Try to extract config from path
    try:
        ref = ValidatedModelRef.from_path(model_path)
        detected_config = ref.config_key
        expected_config = f"{expected_board_type}_{expected_num_players}p"

        if detected_config != expected_config:
            violations.append(
                f"Path config mismatch: model path '{model_path}' appears to be for "
                f"'{detected_config}' but expected '{expected_config}'"
            )
    except RefError:
        # Cannot extract config from path - this is not necessarily a violation
        # (model might be in a non-standard location)
        logger.debug(
            f"[validate_model_path] Cannot extract config from path: {model_path}"
        )

    # Optional: Load model and check architecture
    if check_architecture and not violations:
        try:
            import torch
            from pathlib import Path

            if not Path(model_path).exists():
                violations.append(f"Model file does not exist: {model_path}")
            else:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

                # Get model state dict
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
                else:
                    state_dict = checkpoint

                # Check value head output dimension
                # Look for value_fc2 or value_fc3 (final layer)
                value_output_dim = None
                for key in ["value_fc3.weight", "value_fc2.weight"]:
                    if key in state_dict:
                        value_output_dim = state_dict[key].shape[0]
                        break

                if value_output_dim is not None and value_output_dim != expected_num_players:
                    violations.append(
                        f"Architecture mismatch: value head has {value_output_dim} outputs "
                        f"but config expects {expected_num_players} players"
                    )

        except (OSError, RuntimeError, KeyError, AttributeError) as e:
            violations.append(f"Failed to load model for architecture check: {e}")

    return len(violations) == 0, violations


def validate_model_matches_config_key(
    model_path: str,
    config_key: str,
    check_architecture: bool = False,
) -> tuple[bool, list[str]]:
    """Validate model path matches a config key.

    Convenience wrapper around validate_model_path_for_config that accepts
    a config_key string instead of separate board_type and num_players.

    Args:
        model_path: Path to model file
        config_key: Config key like "hex8_2p"
        check_architecture: If True, load model and verify value head outputs

    Returns:
        Tuple of (is_valid, list_of_violations)
    """
    from app.training.validated_model_ref import parse_config_key

    parsed = parse_config_key(config_key)
    if parsed is None:
        return False, [f"Invalid config key format: {config_key}"]

    board_type, num_players = parsed
    return validate_model_path_for_config(
        model_path, board_type, num_players, check_architecture
    )

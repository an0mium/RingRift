"""Model factory for RingRift neural network training.

This module centralizes model creation logic, extracting what was previously
~200 lines of model initialization code from train_model().

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from app.models import BoardType
from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    MAX_PLAYERS,
    HexNeuralNet_v2,
    HexNeuralNet_v3,
    RingRiftCNN_v2,
    RingRiftCNN_v3,
    get_policy_size_for_board,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model creation."""

    board_type: BoardType
    board_size: int
    policy_size: int
    in_channels: int
    global_features: int = 20
    history_length: int = 3
    num_players: int = 2
    multi_player: bool = False
    model_version: str = 'v2'
    num_res_blocks: int = 6
    num_filters: int = 96
    dropout: float = 0.08
    feature_version: int = 1


def create_model(config: ModelConfig, device: torch.device | None = None) -> nn.Module:
    """Create a neural network model based on configuration.

    Args:
        config: Model configuration
        device: Device to place model on (optional)

    Returns:
        Initialized neural network model
    """
    use_hex_model = config.board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    use_hex_v3 = use_hex_model and config.model_version == 'v3'
    # Compute hex_radius from board_type: HEX8 has radius 4, HEXAGONAL has radius 12
    hex_radius = 4 if config.board_type == BoardType.HEX8 else 12

    # Determine effective number of players for value head
    if config.multi_player:
        effective_num_players = MAX_PLAYERS
    else:
        effective_num_players = config.num_players

    if use_hex_v3:
        # HexNeuralNet_v3 for hexagonal boards with spatial policy heads
        model = HexNeuralNet_v3(
            in_channels=config.in_channels,
            global_features=config.global_features,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size,
            hex_radius=hex_radius,
            policy_size=config.policy_size,
            num_players=effective_num_players,
        )
        logger.info(
            f"Created HexNeuralNet_v3: board_size={config.board_size}, "
            f"hex_radius={hex_radius}, policy_size={config.policy_size}, "
            f"in_channels={config.in_channels}"
        )
    elif use_hex_model:
        # HexNeuralNet_v2 for hexagonal boards
        model = HexNeuralNet_v2(
            in_channels=config.in_channels,
            global_features=config.global_features,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            board_size=config.board_size,
            hex_radius=hex_radius,
            policy_size=config.policy_size,
            num_players=effective_num_players,
        )
        logger.info(
            f"Created HexNeuralNet_v2: board_size={config.board_size}, "
            f"hex_radius={hex_radius}, policy_size={config.policy_size}, "
            f"in_channels={config.in_channels}"
        )
    elif config.model_version == 'v4':
        # V4 NAS-optimized architecture
        from app.ai.neural_net import RingRiftCNN_v4
        model = RingRiftCNN_v4(
            board_size=config.board_size,
            in_channels=14,  # 14 spatial feature channels per frame
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=effective_num_players,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_attention_heads=4,  # NAS optimal
            dropout=config.dropout,
            initial_kernel_size=5,  # NAS optimal
        )
        logger.info(
            f"Created RingRiftCNN_v4 (NAS): board_size={config.board_size}, "
            f"policy_size={config.policy_size}, blocks={config.num_res_blocks}, "
            f"filters={config.num_filters}"
        )
    elif config.model_version == 'v3':
        # V3 architecture with spatial policy heads
        model = RingRiftCNN_v3(
            board_size=config.board_size,
            in_channels=14,
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_players=effective_num_players,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        )
        logger.info(
            f"Created RingRiftCNN_v3: board_size={config.board_size}, "
            f"policy_size={config.policy_size}, num_players={effective_num_players}"
        )
    else:
        # RingRiftCNN_v2 for square boards (default)
        model = RingRiftCNN_v2(
            board_size=config.board_size,
            in_channels=14,
            global_features=config.global_features,
            history_length=config.history_length,
            policy_size=config.policy_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
            num_players=effective_num_players if config.multi_player else 2,
        )
        logger.info(
            f"Created RingRiftCNN_v2: board_size={config.board_size}, "
            f"policy_size={config.policy_size}"
        )

    # Set feature version for compatibility checking
    with contextlib.suppress(Exception):
        model.feature_version = config.feature_version

    # Move to device if specified
    if device is not None:
        model.to(device)

    return model


def get_board_size(board_type: BoardType) -> int:
    """Get the canonical board size for a board type."""
    if board_type == BoardType.SQUARE19:
        return 19
    elif board_type == BoardType.HEXAGONAL:
        return HEX_BOARD_SIZE  # 25
    elif board_type == BoardType.HEX8:
        return HEX8_BOARD_SIZE  # 9
    else:
        return 8  # Default square8


def compute_in_channels(
    board_type: BoardType,
    history_length: int,
    model_version: str = 'v2',
) -> int:
    """Compute the number of input channels based on board type and history.

    Args:
        board_type: The board type
        history_length: Number of history frames
        model_version: Model version (affects hex channel count)

    Returns:
        Number of input channels for the model
    """
    use_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
    use_hex_v3 = use_hex and model_version == 'v3'

    if use_hex_v3:
        base_channels = 16
    elif use_hex:
        base_channels = 10
    else:
        base_channels = 14

    return base_channels * (history_length + 1)


def get_effective_architecture(
    model_version: str,
    board_type: BoardType,
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
) -> tuple[int, int]:
    """Get effective architecture parameters.

    Args:
        model_version: Model version (v2, v3, v4)
        board_type: Board type (affects defaults for hex)
        num_res_blocks: Override for residual blocks
        num_filters: Override for filters

    Returns:
        Tuple of (effective_blocks, effective_filters)
    """
    use_hex = board_type in (BoardType.HEXAGONAL, BoardType.HEX8)

    if model_version == 'v3' or use_hex:
        default_blocks = 12
        default_filters = 192
    elif model_version == 'v4':
        default_blocks = 13
        default_filters = 128
    else:
        default_blocks = 6
        default_filters = 96

    effective_blocks = num_res_blocks if num_res_blocks is not None else default_blocks
    effective_filters = num_filters if num_filters is not None else default_filters

    return effective_blocks, effective_filters


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> bool:
    """Load model weights from a checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load onto
        strict: Whether to require exact key match

    Returns:
        True if loaded successfully, False otherwise
    """
    import os

    from app.utils.torch_utils import safe_load_checkpoint

    if not os.path.exists(checkpoint_path):
        return False

    try:
        checkpoint = safe_load_checkpoint(
            checkpoint_path,
            map_location=device,
            warn_on_unsafe=False,
        )

        # Handle both raw state_dict and checkpoint dict formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded model weights from {checkpoint_path}")
        return True

    except Exception as e:
        logger.warning(f"Could not load weights from {checkpoint_path}: {e}")
        return False


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device: Device model is on
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if device.type == 'cuda':
        device_ids = [device.index if device.index is not None else 0]
    else:
        device_ids = None

    return DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_summary(model: nn.Module, config: ModelConfig) -> None:
    """Log a summary of the model architecture."""
    param_count = count_parameters(model)
    logger.info(
        f"Model summary: {param_count:,} trainable parameters, "
        f"board_size={config.board_size}, policy_size={config.policy_size}"
    )


__all__ = [
    'ModelConfig',
    'compute_in_channels',
    'count_parameters',
    'create_model',
    'get_board_size',
    'get_effective_architecture',
    'load_model_weights',
    'log_model_summary',
    'wrap_model_ddp',
]

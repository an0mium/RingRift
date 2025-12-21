"""Training configuration resolver for RingRift AI.

This module centralizes parameter resolution and validation for train_model(),
extracting what was previously 200+ lines of parameter handling.

December 2025: Extracted from train.py to improve modularity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.models import BoardType
from app.training.config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTrainingParams:
    """Fully resolved training parameters with all defaults applied."""

    # Core training params
    early_stopping_patience: int = 5
    elo_early_stopping_patience: int = 10
    elo_min_improvement: float = 5.0
    warmup_epochs: int = 1
    lr_scheduler: str = 'cosine'
    lr_min: float = 1e-6
    lr_t0: int = 10
    lr_t_mult: int = 2

    # Model architecture
    model_version: str = 'v2'
    num_res_blocks: int | None = None
    num_filters: int | None = None

    # Data handling
    use_streaming: bool = False
    sampling_weights: str = 'uniform'
    validate_data: bool = True

    # Multi-player
    multi_player: bool = False
    num_players: int = 2

    # Mixed precision
    mixed_precision: bool = False
    amp_dtype: str = 'bfloat16'

    # Fault tolerance
    enable_circuit_breaker: bool = True
    enable_anomaly_detection: bool = True
    gradient_clip_mode: str = 'adaptive'
    gradient_clip_max_norm: float = 1.0

    # Enhancements
    use_integrated_enhancements: bool = True
    enable_curriculum: bool = False
    enable_augmentation: bool = False
    enable_elo_weighting: bool = True
    enable_auxiliary_tasks: bool = True
    enable_background_eval: bool = True

    # Policy label smoothing
    policy_label_smoothing: float = 0.0

    # Regularization
    dropout: float = 0.08

    # Batch configuration
    auto_tune_batch_size: bool = True

    # Hot data buffer
    use_hot_data_buffer: bool = False
    hot_buffer_size: int = 10000
    hot_buffer_mix_ratio: float = 0.3

    # Improvements (2024-12)
    improvements_enabled: list[str] = field(default_factory=list)


def resolve_training_params(
    config: TrainConfig,
    *,
    early_stopping_patience: int | None = None,
    elo_early_stopping_patience: int | None = None,
    elo_min_improvement: float | None = None,
    warmup_epochs: int | None = None,
    lr_scheduler: str | None = None,
    lr_min: float | None = None,
    lr_t0: int = 10,
    lr_t_mult: int = 2,
    model_version: str = 'v2',
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
    use_streaming: bool = False,
    sampling_weights: str = 'uniform',
    validate_data: bool = True,
    multi_player: bool = False,
    num_players: int = 2,
    mixed_precision: bool = False,
    amp_dtype: str = 'bfloat16',
    enable_circuit_breaker: bool = True,
    enable_anomaly_detection: bool = True,
    gradient_clip_mode: str = 'adaptive',
    gradient_clip_max_norm: float = 1.0,
    use_integrated_enhancements: bool = True,
    enable_curriculum: bool = False,
    enable_augmentation: bool = False,
    enable_elo_weighting: bool = True,
    enable_auxiliary_tasks: bool = True,
    enable_background_eval: bool = True,
    policy_label_smoothing: float = 0.0,
    dropout: float = 0.08,
    auto_tune_batch_size: bool = True,
    use_hot_data_buffer: bool = False,
    hot_buffer_size: int = 10000,
    hot_buffer_mix_ratio: float = 0.3,
    # 2024-12 improvements
    spectral_norm: bool = False,
    cyclic_lr: bool = False,
    cyclic_lr_period: int = 5,
    value_whitening: bool = False,
    ema: bool = False,
    ema_decay: float = 0.999,
    stochastic_depth: bool = False,
    stochastic_depth_prob: float = 0.1,
    adaptive_warmup: bool = False,
    hard_example_mining: bool = False,
    hard_example_top_k: float = 0.3,
) -> ResolvedTrainingParams:
    """Resolve all training parameters, applying config defaults where needed.

    Args:
        config: Base training configuration
        **kwargs: Override parameters

    Returns:
        ResolvedTrainingParams with all values set
    """
    # Resolve from config with fallbacks
    resolved = ResolvedTrainingParams(
        early_stopping_patience=(
            early_stopping_patience
            if early_stopping_patience is not None
            else getattr(config, 'early_stopping_patience', 5)
        ),
        elo_early_stopping_patience=(
            elo_early_stopping_patience
            if elo_early_stopping_patience is not None
            else getattr(config, 'elo_early_stopping_patience', 10)
        ),
        elo_min_improvement=(
            elo_min_improvement
            if elo_min_improvement is not None
            else getattr(config, 'elo_min_improvement', 5.0)
        ),
        warmup_epochs=(
            warmup_epochs
            if warmup_epochs is not None
            else getattr(config, 'warmup_epochs', 1)
        ),
        lr_scheduler=(
            lr_scheduler
            if lr_scheduler is not None
            else getattr(config, 'lr_scheduler', 'cosine')
        ),
        lr_min=(
            lr_min
            if lr_min is not None
            else getattr(config, 'lr_min', 1e-6)
        ),
        lr_t0=lr_t0,
        lr_t_mult=lr_t_mult,
        model_version=model_version,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        use_streaming=use_streaming,
        sampling_weights=sampling_weights,
        validate_data=validate_data,
        multi_player=multi_player,
        num_players=num_players,
        mixed_precision=mixed_precision,
        amp_dtype=amp_dtype,
        enable_circuit_breaker=enable_circuit_breaker,
        enable_anomaly_detection=enable_anomaly_detection,
        gradient_clip_mode=gradient_clip_mode,
        gradient_clip_max_norm=gradient_clip_max_norm,
        use_integrated_enhancements=use_integrated_enhancements,
        enable_curriculum=enable_curriculum,
        enable_augmentation=enable_augmentation,
        enable_elo_weighting=enable_elo_weighting,
        enable_auxiliary_tasks=enable_auxiliary_tasks,
        enable_background_eval=enable_background_eval,
        policy_label_smoothing=policy_label_smoothing,
        dropout=dropout,
        auto_tune_batch_size=auto_tune_batch_size,
        use_hot_data_buffer=use_hot_data_buffer,
        hot_buffer_size=hot_buffer_size,
        hot_buffer_mix_ratio=hot_buffer_mix_ratio,
    )

    # Collect enabled improvements
    improvements = []
    if spectral_norm:
        improvements.append("spectral_norm")
    if cyclic_lr:
        improvements.append(f"cyclic_lr(period={cyclic_lr_period})")
    if mixed_precision:
        improvements.append(f"mixed_precision({amp_dtype})")
    if value_whitening:
        improvements.append("value_whitening")
    if ema:
        improvements.append(f"ema(decay={ema_decay})")
    if stochastic_depth:
        improvements.append(f"stochastic_depth(p={stochastic_depth_prob})")
    if adaptive_warmup:
        improvements.append("adaptive_warmup")
    if hard_example_mining:
        improvements.append(f"hard_example_mining(top_k={hard_example_top_k})")
    resolved.improvements_enabled = improvements

    return resolved


def validate_model_id_for_board(
    model_id: str,
    board_type: BoardType,
    use_hex_model: bool,
    num_players: int,
    resume_path: str | None = None,
) -> str:
    """Validate and potentially fix model_id for board type compatibility.

    Args:
        model_id: Current model ID
        board_type: Target board type
        use_hex_model: Whether using hex model
        num_players: Number of players
        resume_path: Path if resuming (stricter validation)

    Returns:
        Validated/fixed model ID

    Raises:
        ValueError: If resuming with incompatible model ID
    """
    from app.utils.canonical_naming import normalize_board_type

    # Check hex model with square ID
    if use_hex_model and "sq8" in model_id.lower():
        if resume_path:
            raise ValueError(
                f"Model ID '{model_id}' contains 'sq8' but board_type is "
                f"{board_type.name} which uses HexNeuralNet architecture. "
                "Use a model ID that reflects the hex board type."
            )
        # Auto-generate appropriate model_id
        board_prefix = "hex8" if board_type == BoardType.HEX8 else "hex"
        new_model_id = f"ringrift_{board_prefix}_{num_players}p"
        logger.warning(
            f"Model ID '{model_id}' is for sq8 but training {board_type.name}. "
            f"Using '{new_model_id}' instead."
        )
        return new_model_id

    # Check square model with hex ID
    if not use_hex_model and ("hex" in model_id.lower() and "sq" not in model_id.lower()):
        if resume_path:
            raise ValueError(
                f"Model ID '{model_id}' appears to be for hex but board_type is "
                f"{board_type.name}. Use a model ID that matches the board type."
            )
        # Auto-generate appropriate model_id
        board_prefix = normalize_board_type(board_type)
        new_model_id = f"ringrift_{board_prefix}_{num_players}p"
        logger.warning(
            f"Model ID '{model_id}' is for hex but training {board_type.name}. "
            f"Using '{new_model_id}' instead."
        )
        return new_model_id

    return model_id


def get_board_size(board_type: BoardType) -> int:
    """Get the canonical board size for a board type.

    Args:
        board_type: The board type

    Returns:
        Board size for CNN spatial dimensions
    """
    from app.ai.neural_net import HEX8_BOARD_SIZE, HEX_BOARD_SIZE

    if board_type == BoardType.SQUARE19:
        return 19
    elif board_type == BoardType.HEXAGONAL:
        return HEX_BOARD_SIZE  # 25
    elif board_type == BoardType.HEX8:
        return HEX8_BOARD_SIZE  # 9
    else:
        return 8  # Default square8


def get_effective_architecture(
    model_version: str,
    use_hex_model: bool,
    num_res_blocks: int | None = None,
    num_filters: int | None = None,
) -> tuple[int, int]:
    """Get effective architecture parameters.

    Args:
        model_version: Model version (v2, v3, v4)
        use_hex_model: Whether using hex model
        num_res_blocks: Override for residual blocks
        num_filters: Override for filters

    Returns:
        Tuple of (effective_blocks, effective_filters)
    """
    if model_version == 'v3' or use_hex_model:
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


__all__ = [
    'ResolvedTrainingParams',
    'get_board_size',
    'get_effective_architecture',
    'resolve_training_params',
    'validate_model_id_for_board',
]

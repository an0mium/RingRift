"""Training context for RingRift Neural Network AI.

December 2025: Extracted from train.py to improve modularity.

This module provides the TrainContext dataclass which holds all shared state
for a training run, enabling decomposition of train_model() into smaller,
testable components.

Usage:
    from app.training.train_context import TrainContext, TrainContextConfig

    context = TrainContext(
        config=train_config,
        resolved=resolved_config,
        device=device,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
    from torch.utils.data import DataLoader, DistributedSampler

    from app.training.train_config import TrainConfig


# =============================================================================
# Resolved Configuration
# =============================================================================


@dataclass
class ResolvedConfig:
    """Fully resolved training configuration.

    Contains all parameters resolved from config, CLI, and defaults.
    This ensures consistent parameter precedence throughout training.
    """

    # Early stopping
    early_stopping_patience: int = 20
    elo_early_stopping_patience: int = 15
    elo_min_improvement: float = 5.0

    # Learning rate
    warmup_epochs: int = 1
    lr_scheduler: str = "cosine"
    lr_min: float = 1e-6
    lr_t0: int = 10
    lr_t_mult: int = 2

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    save_all_epochs: bool = True

    # Resume/transfer
    resume_path: str | None = None
    init_weights_path: str | None = None
    init_weights_strict: bool = False

    # Model architecture
    model_version: str = "v2"
    model_type: str = "cnn"
    num_res_blocks: int | None = None
    num_filters: int | None = None
    freeze_policy: bool = False
    dropout: float = 0.08

    # Data
    use_streaming: bool = False
    data_dir: str | None = None
    sampling_weights: str = "uniform"

    # Players
    multi_player: bool = False
    num_players: int = 2

    # Distributed
    distributed: bool = False
    local_rank: int = -1
    scale_lr: bool = False
    lr_scale_mode: str = "linear"
    find_unused_parameters: bool = False

    # Augmentation
    augment_hex_symmetry: bool = False

    # 2024-12 Training Improvements
    spectral_norm: bool = False
    cyclic_lr: bool = False
    cyclic_lr_period: int = 5
    mixed_precision: bool = False
    amp_dtype: str = "bfloat16"
    value_whitening: bool = False
    value_whitening_momentum: float = 0.99
    ema: bool = False
    ema_decay: float = 0.999
    stochastic_depth: bool = False
    stochastic_depth_prob: float = 0.1
    adaptive_warmup: bool = False
    hard_example_mining: bool = True
    hard_example_top_k: float = 0.3
    enable_outcome_weighted_policy: bool = True
    outcome_weight_scale: float = 0.5
    auto_tune_batch_size: bool = True
    track_calibration: bool = False

    # Hot data buffer
    use_hot_data_buffer: bool = False
    hot_buffer_size: int = 10000
    hot_buffer_mix_ratio: float = 0.3

    # Integrated enhancements
    use_integrated_enhancements: bool = True
    enable_curriculum: bool = True
    enable_augmentation: bool = True
    enable_elo_weighting: bool = True
    enable_auxiliary_tasks: bool = True
    enable_batch_scheduling: bool = False
    enable_background_eval: bool = True

    # Policy and data validation
    policy_label_smoothing: float = 0.0
    validate_data: bool = True
    fail_on_invalid_data: bool = False

    # Fault tolerance
    enable_circuit_breaker: bool = True
    enable_anomaly_detection: bool = True
    gradient_clip_mode: str = "adaptive"
    gradient_clip_max_norm: float = 1.0
    anomaly_spike_threshold: float = 3.0
    anomaly_gradient_threshold: float = 100.0
    enable_graceful_shutdown: bool = True

    # Data freshness
    skip_freshness_check: bool = False
    max_data_age_hours: float = 1.0
    allow_stale_data: bool = False
    disable_stale_fallback: bool = False
    max_sync_failures: int = 5
    max_sync_duration: float = 2700.0

    # Checkpoint averaging
    enable_checkpoint_averaging: bool = True
    num_checkpoints_to_average: int = 5

    # Quality-weighted training
    enable_quality_weighting: bool = True
    quality_weight_blend: float = 0.5
    quality_ranking_weight: float = 0.1

    # Heartbeat
    heartbeat_file: str | None = None
    heartbeat_interval: float = 30.0

    # LR finder
    find_lr: bool = False
    lr_finder_min: float = 1e-7
    lr_finder_max: float = 1.0
    lr_finder_iterations: int = 100

    # Quality discovery
    discover_synced_data: bool = False
    min_quality_score: float = 0.0

    @property
    def config_label(self) -> str:
        """Return config label for metrics (e.g., 'hex8_2p')."""
        # This will be set properly when board_type is available
        return f"unknown_{self.num_players}p"


# =============================================================================
# Training State
# =============================================================================


@dataclass
class TrainingProgress:
    """Tracks training progress for checkpointing and monitoring.

    This is the mutable state that changes during training.
    """

    # Current position
    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0

    # Best metrics
    best_val_loss: float = float("inf")
    best_train_loss: float = float("inf")

    # Current metrics
    current_train_loss: float = float("inf")
    current_val_loss: float = float("inf")

    # Checkpoint tracking
    last_good_checkpoint_path: str | None = None
    last_good_epoch: int = 0

    # Circuit breaker state
    circuit_breaker_rollbacks: int = 0
    max_circuit_breaker_rollbacks: int = 3

    # Anomaly tracking
    anomaly_step: int = 0

    # Per-epoch losses for analysis
    epoch_losses: list[dict[str, float]] = field(default_factory=list)

    # Completion tracking
    epochs_completed: int = 0
    completed_normally: bool = False
    exception: Exception | None = None

    # Timing
    start_time: float = field(default_factory=time.time)


# =============================================================================
# Training Context
# =============================================================================


@dataclass
class TrainContext:
    """All shared state for a training run.

    This dataclass aggregates all components needed for training,
    enabling clean handoff between training phases.
    """

    # ==========================================================================
    # Core configuration
    # ==========================================================================
    config: "TrainConfig"
    resolved: ResolvedConfig
    data_paths: list[str] = field(default_factory=list)
    save_path: str = ""

    # ==========================================================================
    # Device and distributed
    # ==========================================================================
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1

    # ==========================================================================
    # Model
    # ==========================================================================
    model: nn.Module | None = None
    model_version: str = "v2"
    policy_size: int = 0
    board_size: int = 0
    effective_blocks: int = 0
    effective_filters: int = 0
    feature_version: int = 1

    # ==========================================================================
    # Optimizer and schedulers
    # ==========================================================================
    optimizer: "Optimizer | None" = None
    epoch_scheduler: "_LRScheduler | None" = None
    plateau_scheduler: "ReduceLROnPlateau | None" = None
    warmup_scheduler: Any | None = None
    grad_scaler: "GradScaler | None" = None

    # ==========================================================================
    # Data loaders
    # ==========================================================================
    train_loader: "DataLoader | Any | None" = None  # DataLoader or StreamingDataLoader
    val_loader: "DataLoader | Any | None" = None
    train_sampler: "DistributedSampler | None" = None
    use_streaming: bool = False
    has_mp_values: bool = False

    # ==========================================================================
    # Enhancement managers
    # ==========================================================================
    enhancements_manager: Any | None = None  # IntegratedTrainingManager
    training_facade: Any | None = None  # TrainingEnhancementsFacade
    hard_example_miner: Any | None = None  # HardExampleMiner
    quality_trainer: Any | None = None  # QualityWeightedTrainer
    hot_buffer: Any | None = None  # HotDataBuffer
    gradient_surgeon: Any | None = None  # GradientSurgeon

    # ==========================================================================
    # Fault tolerance
    # ==========================================================================
    training_breaker: Any | None = None  # CircuitBreaker
    anomaly_detector: Any | None = None  # AnomalyDetector
    adaptive_clipper: Any | None = None  # AdaptiveGradientClipper
    fixed_clip_norm: float | None = None
    shutdown_handler: Any | None = None  # GracefulShutdownHandler
    rollback_handler: Any | None = None

    # ==========================================================================
    # Checkpointing
    # ==========================================================================
    checkpoint_averager: Any | None = None  # CheckpointAverager
    async_checkpointer: Any | None = None  # AsyncCheckpointer

    # ==========================================================================
    # Monitoring and metrics
    # ==========================================================================
    early_stopper: Any | None = None  # EarlyStopping
    eval_feedback_handler: Any | None = None  # EvaluationFeedbackHandler
    calibration_tracker: Any | None = None  # CalibrationTracker
    metrics_collector: Any | None = None  # MetricsCollector
    heartbeat_monitor: Any | None = None  # HeartbeatMonitor
    dist_metrics: Any | None = None  # DistributedMetrics

    # ==========================================================================
    # Training progress (mutable state)
    # ==========================================================================
    progress: TrainingProgress = field(default_factory=TrainingProgress)

    # ==========================================================================
    # Labels for metrics
    # ==========================================================================
    config_label: str = ""

    # ==========================================================================
    # Convenience properties
    # ==========================================================================

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process in distributed training."""
        if not self.distributed:
            return True
        return self.local_rank <= 0

    @property
    def amp_enabled(self) -> bool:
        """Check if automatic mixed precision is enabled."""
        return self.grad_scaler is not None

    @property
    def amp_dtype(self) -> torch.dtype:
        """Get the AMP dtype."""
        if self.resolved.amp_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16

    @property
    def model_to_save(self) -> nn.Module:
        """Get the model for saving (unwrap DDP if needed)."""
        if self.model is None:
            raise ValueError("Model not initialized")
        if self.distributed and hasattr(self.model, "module"):
            return self.model.module
        return self.model

    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory as Path."""
        return Path(self.resolved.checkpoint_dir)

    # ==========================================================================
    # Factory methods
    # ==========================================================================

    @classmethod
    def from_config(
        cls,
        config: "TrainConfig",
        data_path: str | list[str],
        save_path: str,
        **kwargs: Any,
    ) -> "TrainContext":
        """Create a TrainContext from TrainConfig and parameters.

        Args:
            config: The training configuration
            data_path: Path(s) to training data
            save_path: Path to save the model
            **kwargs: Additional resolved parameters

        Returns:
            A new TrainContext instance
        """
        # Normalize data_path to list
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = list(data_path)

        # Create resolved config from kwargs
        resolved = ResolvedConfig(**{
            k: v for k, v in kwargs.items()
            if hasattr(ResolvedConfig, k)
        })

        return cls(
            config=config,
            resolved=resolved,
            data_paths=data_paths,
            save_path=save_path,
        )

    # ==========================================================================
    # Epoch context creation
    # ==========================================================================

    def create_epoch_context(self) -> "EpochContext":
        """Create an EpochContext for epoch-level training.

        Returns:
            An EpochContext populated from this TrainContext
        """
        from app.training.train_epoch import EpochConfig, EpochContext

        epoch_config = EpochConfig(
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validate_every_n_epochs=1,
            patience=self.resolved.early_stopping_patience,
            distributed=self.distributed,
            board_type=str(self.config.board_type.value) if hasattr(self.config.board_type, 'value') else str(self.config.board_type),
            num_players=self.resolved.num_players,
        )

        return EpochContext(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            config=epoch_config,
            epoch_scheduler=self.epoch_scheduler,
            plateau_scheduler=self.plateau_scheduler,
            grad_scaler=self.grad_scaler,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            training_facade=self.training_facade,
            hard_example_miner=self.hard_example_miner,
            eval_feedback_handler=self.eval_feedback_handler,
            calibration_tracker=self.calibration_tracker,
            hot_buffer=self.hot_buffer,
            training_breaker=self.training_breaker,
            adaptive_clipper=self.adaptive_clipper,
            gradient_surgeon=self.gradient_surgeon,
            dist_metrics=self.dist_metrics,
            quality_trainer=self.quality_trainer,
            is_streaming=self.use_streaming,
            has_mp_values=self.has_mp_values,
        )

    # ==========================================================================
    # State serialization (for checkpointing)
    # ==========================================================================

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Get state dict for checkpointing.

        Returns:
            Dictionary with all checkpoint-worthy state
        """
        state = {
            "epoch": self.progress.epoch,
            "global_step": self.progress.global_step,
            "best_val_loss": self.progress.best_val_loss,
            "epochs_completed": self.progress.epochs_completed,
            "epoch_losses": self.progress.epoch_losses,
            "config_label": self.config_label,
        }

        if self.model is not None:
            state["model_state_dict"] = self.model_to_save.state_dict()

        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.epoch_scheduler is not None:
            state["scheduler_state_dict"] = self.epoch_scheduler.state_dict()

        if self.grad_scaler is not None:
            state["grad_scaler_state_dict"] = self.grad_scaler.state_dict()

        if self.early_stopper is not None and hasattr(self.early_stopper, "state_dict"):
            state["early_stopper_state_dict"] = self.early_stopper.state_dict()

        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state: Dictionary with checkpoint state
        """
        self.progress.epoch = state.get("epoch", 0)
        self.progress.global_step = state.get("global_step", 0)
        self.progress.best_val_loss = state.get("best_val_loss", float("inf"))
        self.progress.epochs_completed = state.get("epochs_completed", 0)
        self.progress.epoch_losses = state.get("epoch_losses", [])

        if "model_state_dict" in state and self.model is not None:
            self.model_to_save.load_state_dict(state["model_state_dict"])

        if "optimizer_state_dict" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if "scheduler_state_dict" in state and self.epoch_scheduler is not None:
            self.epoch_scheduler.load_state_dict(state["scheduler_state_dict"])

        if "grad_scaler_state_dict" in state and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(state["grad_scaler_state_dict"])

        if "early_stopper_state_dict" in state and self.early_stopper is not None:
            if hasattr(self.early_stopper, "load_state_dict"):
                self.early_stopper.load_state_dict(state["early_stopper_state_dict"])

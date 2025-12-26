"""
Advanced Training Utilities for RingRift AI.

This module provides advanced training features:
1. Learning Rate Finder - Find optimal LR range
2. Gradient Checkpointing - Memory-efficient training
3. PFSP Opponent Pool - Prioritized Fictitious Self-Play
4. CMA-ES Auto-Tuning Hook - Trigger HP search on plateau

Usage:
    from app.training.advanced_training import (
        LRFinder,
        GradientCheckpointing,
        PFSPOpponentPool,
        CMAESAutoTuner,
    )
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.utils.torch_utils import safe_load_checkpoint

# December 2025: Import from extracted modules for backward compatibility
from app.training.lr_finder import LRFinder, LRFinderResult
from app.training.gradient_checkpointing import GradientCheckpointing, estimate_memory_savings
from app.training.pfsp_pool import PFSPOpponentPool, OpponentStats

# Import unified sampler base class (2025-12)
try:
    from app.training.elo_weighting import WeightedSamplerBase
    HAS_WEIGHTED_SAMPLER_BASE = True
except ImportError:
    HAS_WEIGHTED_SAMPLER_BASE = False
    WeightedSamplerBase = object  # Fallback to object as base

logger = logging.getLogger(__name__)


# =============================================================================
# 1-3. EXTRACTED MODULES (December 2025)
# =============================================================================
# NOTE: The following classes have been extracted to dedicated modules
# and are imported above for backward compatibility:
# - LRFinder, LRFinderResult -> app.training.lr_finder (280 lines)
# - GradientCheckpointing, estimate_memory_savings -> app.training.gradient_checkpointing (230 lines)
# - PFSPOpponentPool, OpponentStats -> app.training.pfsp_pool (320 lines)
# Total: ~830 lines extracted to improve maintainability.


# =============================================================================
# 4. CMA-ES Auto-Tuning Hook
# =============================================================================


@dataclass
class PlateauConfig:
    """Configuration for plateau detection."""
    patience: int = 10  # Epochs without improvement
    min_delta: float = 0.01  # Minimum improvement to reset patience
    metric: str = "elo"  # "elo", "loss", or "win_rate"
    lookback: int = 20  # Epochs to consider for trend


class CMAESAutoTuner:
    """
    Automatically trigger CMA-ES hyperparameter optimization on plateau.

    Monitors training metrics and launches CMA-ES optimization when
    progress stalls, then integrates optimized hyperparameters.

    Usage:
        auto_tuner = CMAESAutoTuner(
            cmaes_script="scripts/run_gpu_cmaes.py",
            board_type="square8",
        )

        for epoch in range(epochs):
            train_epoch()
            auto_tuner.step(current_elo=elo, current_loss=loss)

            if auto_tuner.should_tune():
                result = auto_tuner.run_optimization()
                if result:
                    # Apply new hyperparameters
                    apply_weights(result['weights'])
    """

    def __init__(
        self,
        cmaes_script: str = "scripts/run_gpu_cmaes.py",
        board_type: str = "square8",
        num_players: int = 2,
        plateau_config: PlateauConfig | None = None,
        output_dir: str = "logs/cmaes_auto",
        min_epochs_between_tuning: int = 50,
        max_auto_tunes: int = 5,
    ):
        """
        Args:
            cmaes_script: Path to CMA-ES optimization script
            board_type: Board type for optimization
            num_players: Number of players
            plateau_config: Plateau detection configuration
            output_dir: Output directory for CMA-ES results
            min_epochs_between_tuning: Minimum epochs between auto-tunes
            max_auto_tunes: Maximum auto-tunes per training run
        """
        self.cmaes_script = cmaes_script
        self.board_type = board_type
        self.num_players = num_players
        self.plateau_config = plateau_config or PlateauConfig()
        self.output_dir = Path(output_dir)
        self.min_epochs_between_tuning = min_epochs_between_tuning
        self.max_auto_tunes = max_auto_tunes

        self._metric_history: deque = deque(maxlen=self.plateau_config.lookback)
        self._epochs_without_improvement = 0
        self._best_metric = float('-inf')
        self._last_tune_epoch = -self.min_epochs_between_tuning
        self._tune_count = 0
        self._current_epoch = 0
        self._is_tuning = False

    def step(
        self,
        current_elo: float | None = None,
        current_loss: float | None = None,
        current_win_rate: float | None = None,
    ) -> None:
        """
        Update with current training metrics.

        Args:
            current_elo: Current Elo rating
            current_loss: Current training loss
            current_win_rate: Current win rate against baseline
        """
        self._current_epoch += 1

        # Get metric value
        metric = self.plateau_config.metric
        if metric == "elo" and current_elo is not None:
            value = current_elo
        elif metric == "loss" and current_loss is not None:
            value = -current_loss  # Lower loss is better
        elif metric == "win_rate" and current_win_rate is not None:
            value = current_win_rate
        else:
            return

        self._metric_history.append(value)

        # Check for improvement
        if value > self._best_metric + self.plateau_config.min_delta:
            self._best_metric = value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

    def should_tune(self) -> bool:
        """Check if CMA-ES optimization should be triggered."""
        if self._is_tuning:
            return False

        if self._tune_count >= self.max_auto_tunes:
            return False

        if (self._current_epoch - self._last_tune_epoch) < self.min_epochs_between_tuning:
            return False

        # Check for plateau
        if self._epochs_without_improvement >= self.plateau_config.patience:
            logger.info(
                f"Plateau detected: {self._epochs_without_improvement} epochs "
                f"without improvement (patience={self.plateau_config.patience})"
            )
            return True

        return False

    def run_optimization(
        self,
        generations: int = 30,
        population_size: int = 15,
        games_per_eval: int = 30,
    ) -> dict[str, Any] | None:
        """
        Run CMA-ES optimization.

        Args:
            generations: Number of CMA-ES generations
            population_size: Population size
            games_per_eval: Games per fitness evaluation

        Returns:
            Dictionary with optimized weights and metrics, or None on failure
        """
        if self._is_tuning:
            logger.warning("Optimization already in progress")
            return None

        self._is_tuning = True
        self._tune_count += 1
        self._last_tune_epoch = self._current_epoch

        # Create output directory
        run_dir = self.output_dir / f"auto_tune_{self._tune_count}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting auto CMA-ES optimization #{self._tune_count}")

        try:
            # Build command
            cmd = [
                "python", self.cmaes_script,
                "--board", self.board_type,
                "--num-players", str(self.num_players),
                "--generations", str(generations),
                "--population-size", str(population_size),
                "--games-per-eval", str(games_per_eval),
                "--output-dir", str(run_dir),
            ]

            # Run CMA-ES
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 4,  # 4 hour timeout
            )

            if result.returncode != 0:
                logger.error(f"CMA-ES failed: {result.stderr}")
                return None

            # Load results
            import json
            results_file = run_dir / "optimized_weights.json"
            if results_file.exists():
                with open(results_file) as f:
                    weights = json.load(f)

                # Reset improvement tracking
                self._epochs_without_improvement = 0

                logger.info(f"CMA-ES optimization complete. Best fitness: {weights.get('fitness', 'N/A')}")

                return {
                    'weights': weights,
                    'run_dir': str(run_dir),
                    'tune_number': self._tune_count,
                }
            else:
                logger.warning("CMA-ES completed but no weights file found")
                return None

        except subprocess.TimeoutExpired:
            logger.error("CMA-ES optimization timed out")
            return None
        except Exception as e:
            logger.error(f"CMA-ES optimization failed: {e}")
            return None
        finally:
            self._is_tuning = False

    def get_status(self) -> dict[str, Any]:
        """Get current auto-tuner status."""
        return {
            'current_epoch': self._current_epoch,
            'epochs_without_improvement': self._epochs_without_improvement,
            'patience': self.plateau_config.patience,
            'best_metric': self._best_metric,
            'tune_count': self._tune_count,
            'max_auto_tunes': self.max_auto_tunes,
            'is_tuning': self._is_tuning,
            'epochs_since_last_tune': self._current_epoch - self._last_tune_epoch,
        }

    def reset(self) -> None:
        """Reset the auto-tuner state."""
        self._metric_history.clear()
        self._epochs_without_improvement = 0
        self._best_metric = float('-inf')
        self._tune_count = 0
        self._current_epoch = 0
        self._last_tune_epoch = -self.min_epochs_between_tuning


# =============================================================================
# Integration Helpers
# =============================================================================


def create_advanced_training_suite(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    board_type: str = "square8",
    num_players: int = 2,
    enable_checkpointing: bool = True,
    enable_pfsp: bool = True,
    enable_auto_tuning: bool = True,
) -> dict[str, Any]:
    """
    Create a suite of advanced training utilities.

    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        board_type: Board type
        num_players: Number of players
        enable_checkpointing: Enable gradient checkpointing
        enable_pfsp: Enable PFSP opponent pool
        enable_auto_tuning: Enable CMA-ES auto-tuning

    Returns:
        Dictionary of utility objects
    """
    suite = {
        'lr_finder': LRFinder(model, optimizer, criterion),
    }

    if enable_checkpointing:
        checkpointing = GradientCheckpointing(model)
        suite['checkpointing'] = checkpointing

    if enable_pfsp:
        pfsp_pool = PFSPOpponentPool()
        suite['pfsp_pool'] = pfsp_pool

    if enable_auto_tuning:
        auto_tuner = CMAESAutoTuner(
            board_type=board_type,
            num_players=num_players,
        )
        suite['auto_tuner'] = auto_tuner

    return suite


# =============================================================================
# Phase 4: Training Stability & Acceleration (2024-12)
# =============================================================================


@dataclass
class StabilityMetrics:
    """Metrics for training stability monitoring."""
    gradient_norm: float
    loss_value: float
    loss_variance: float
    param_update_ratio: float
    is_stable: bool
    warnings: list[str] = field(default_factory=list)


class TrainingStabilityMonitor:
    """
    Monitor training stability and detect issues early.

    Detects:
    - Gradient explosions/vanishing
    - Loss spikes or NaN
    - Parameter update instabilities
    - Learning rate issues
    """

    def __init__(
        self,
        gradient_clip_threshold: float = 10.0,
        loss_spike_threshold: float = 3.0,
        loss_history_size: int = 100,
        param_update_threshold: float = 0.1,
        auto_recover: bool = True,
    ):
        self.gradient_clip_threshold = gradient_clip_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_history_size = loss_history_size
        self.param_update_threshold = param_update_threshold
        self.auto_recover = auto_recover

        self._loss_history: deque = deque(maxlen=loss_history_size)
        self._gradient_history: deque = deque(maxlen=loss_history_size)
        self._last_params: dict[str, torch.Tensor] | None = None
        self._recovery_triggered = False
        self._stability_score = 1.0

    def check_gradients(self, model: nn.Module) -> tuple[float, list[str]]:
        """Check gradient health across all parameters."""
        warnings = []
        total_norm = 0.0
        num_params = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params += 1

                if math.isnan(param_norm) or math.isinf(param_norm):
                    warnings.append(f"NaN/Inf gradient in {name}")
                elif param_norm > self.gradient_clip_threshold * 10:
                    warnings.append(f"Extreme gradient in {name}: {param_norm:.4f}")

        total_norm = math.sqrt(total_norm) if num_params > 0 else 0.0
        self._gradient_history.append(total_norm)

        if total_norm > self.gradient_clip_threshold:
            warnings.append(f"Gradient norm {total_norm:.4f} exceeds threshold")
        elif total_norm < 1e-7:
            warnings.append("Vanishing gradients detected")

        return total_norm, warnings

    def check_loss(self, loss: float) -> tuple[bool, list[str]]:
        """Check if loss is healthy."""
        warnings = []
        is_healthy = True

        if math.isnan(loss) or math.isinf(loss):
            warnings.append(f"Invalid loss value: {loss}")
            is_healthy = False
        else:
            self._loss_history.append(loss)

            if len(self._loss_history) >= 10:
                np.mean(list(self._loss_history)[-10:])
                overall_mean = np.mean(list(self._loss_history))
                overall_std = np.std(list(self._loss_history))

                if overall_std > 0 and abs(loss - overall_mean) > self.loss_spike_threshold * overall_std:
                    warnings.append(f"Loss spike detected: {loss:.4f} vs mean {overall_mean:.4f}")

        return is_healthy, warnings

    def check_param_updates(self, model: nn.Module) -> tuple[float, list[str]]:
        """Check parameter update ratios."""
        warnings = []
        update_ratios = []

        current_params = {name: param.data.clone() for name, param in model.named_parameters()}

        if self._last_params is not None:
            for name, current in current_params.items():
                if name in self._last_params:
                    last = self._last_params[name]
                    diff = (current - last).norm().item()
                    param_norm = current.norm().item()

                    if param_norm > 0:
                        ratio = diff / param_norm
                        update_ratios.append(ratio)

                        if ratio > self.param_update_threshold:
                            warnings.append(f"Large param update in {name}: {ratio:.4f}")

        self._last_params = current_params
        avg_ratio = np.mean(update_ratios) if update_ratios else 0.0
        return avg_ratio, warnings

    def step(
        self,
        model: nn.Module,
        loss: float,
        optimizer: optim.Optimizer | None = None,
    ) -> StabilityMetrics:
        """Run all stability checks and return metrics."""
        all_warnings = []

        grad_norm, grad_warnings = self.check_gradients(model)
        all_warnings.extend(grad_warnings)

        loss_healthy, loss_warnings = self.check_loss(loss)
        all_warnings.extend(loss_warnings)

        update_ratio, update_warnings = self.check_param_updates(model)
        all_warnings.extend(update_warnings)

        loss_var = np.var(list(self._loss_history)) if len(self._loss_history) > 1 else 0.0

        is_stable = loss_healthy and len(all_warnings) == 0

        # Update stability score
        if is_stable:
            self._stability_score = min(1.0, self._stability_score + 0.01)
        else:
            self._stability_score = max(0.0, self._stability_score - 0.1)

        # Auto-recovery if enabled
        if self.auto_recover and not is_stable and optimizer is not None and len(all_warnings) > 3:
            self._trigger_recovery(optimizer)

        return StabilityMetrics(
            gradient_norm=grad_norm,
            loss_value=loss,
            loss_variance=loss_var,
            param_update_ratio=update_ratio,
            is_stable=is_stable,
            warnings=all_warnings,
        )

    def _trigger_recovery(self, optimizer: optim.Optimizer) -> None:
        """Attempt to recover from instability."""
        if self._recovery_triggered:
            return

        logger.warning("Training instability detected, triggering recovery...")

        # Reduce learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            logger.info(f"Reduced LR to {param_group['lr']:.2e}")

        self._recovery_triggered = True

    @property
    def stability_score(self) -> float:
        """Get current stability score (0-1)."""
        return self._stability_score

    def get_summary(self) -> dict[str, Any]:
        """Get stability summary."""
        return {
            'stability_score': self._stability_score,
            'avg_gradient_norm': np.mean(list(self._gradient_history)) if self._gradient_history else 0.0,
            'avg_loss': np.mean(list(self._loss_history)) if self._loss_history else 0.0,
            'loss_variance': np.var(list(self._loss_history)) if len(self._loss_history) > 1 else 0.0,
            'recovery_triggered': self._recovery_triggered,
        }


class AdaptivePrecisionManager:
    """
    Dynamically adjust mixed precision based on training stability.

    Switches between FP32, FP16, and BF16 based on:
    - Loss stability
    - Gradient overflow frequency
    - Training progress
    """

    def __init__(
        self,
        initial_precision: str = "fp16",
        stability_window: int = 100,
        overflow_threshold: float = 0.05,
        auto_downgrade: bool = True,
        auto_upgrade: bool = True,
    ):
        self.current_precision = initial_precision
        self.stability_window = stability_window
        self.overflow_threshold = overflow_threshold
        self.auto_downgrade = auto_downgrade
        self.auto_upgrade = auto_upgrade

        self._overflow_count = 0
        self._step_count = 0
        self._precision_history: list[str] = []
        self._scaler: torch.cuda.amp.GradScaler | None = None

        self._precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

    def setup(self, device: torch.device) -> torch.cuda.amp.GradScaler | None:
        """Setup precision management."""
        if device.type != "cuda":
            self.current_precision = "fp32"
            return None

        if self.current_precision == "fp16":
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

        return self._scaler

    def get_autocast_dtype(self) -> torch.dtype:
        """Get current autocast dtype."""
        return self._precision_map.get(self.current_precision, torch.float32)

    def record_overflow(self, had_overflow: bool) -> None:
        """Record gradient overflow event."""
        self._step_count += 1
        if had_overflow:
            self._overflow_count += 1

        # Check if we should change precision
        if self._step_count >= self.stability_window:
            overflow_rate = self._overflow_count / self._step_count

            if overflow_rate > self.overflow_threshold and self.auto_downgrade:
                self._downgrade_precision()
            elif overflow_rate < self.overflow_threshold * 0.1 and self.auto_upgrade:
                self._upgrade_precision()

            # Reset counters
            self._step_count = 0
            self._overflow_count = 0

    def _downgrade_precision(self) -> None:
        """Downgrade to more stable precision."""
        if self.current_precision == "fp16":
            self.current_precision = "bf16"
            self._scaler = None
            logger.info("Precision downgraded: FP16 -> BF16")
        elif self.current_precision == "bf16":
            self.current_precision = "fp32"
            logger.info("Precision downgraded: BF16 -> FP32")

        self._precision_history.append(self.current_precision)

    def _upgrade_precision(self) -> None:
        """Upgrade to faster precision."""
        if self.current_precision == "fp32":
            self.current_precision = "bf16"
            logger.info("Precision upgraded: FP32 -> BF16")
        elif self.current_precision == "bf16":
            self.current_precision = "fp16"
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info("Precision upgraded: BF16 -> FP16")

        self._precision_history.append(self.current_precision)

    def get_stats(self) -> dict[str, Any]:
        """Get precision management stats."""
        return {
            'current_precision': self.current_precision,
            'overflow_rate': self._overflow_count / max(1, self._step_count),
            'precision_history': self._precision_history[-10:],
        }


class ProgressiveLayerUnfreezing:
    """
    Gradually unfreeze model layers during training.

    Implements discriminative fine-tuning where:
    - Early layers are frozen initially
    - Layers are progressively unfrozen as training progresses
    - Different layers can have different learning rates
    """

    def __init__(
        self,
        model: nn.Module,
        unfreeze_schedule: dict[int, list[str]] | None = None,
        lr_multipliers: dict[str, float] | None = None,
        total_epochs: int = 50,
    ):
        self.model = model
        self.unfreeze_schedule = unfreeze_schedule or {}
        self.lr_multipliers = lr_multipliers or {}
        self.total_epochs = total_epochs

        self._frozen_layers: set = set()
        self._unfrozen_at_epoch: dict[str, int] = {}

    def freeze_all_except(self, layer_names: list[str]) -> None:
        """Freeze all layers except specified ones."""
        for name, param in self.model.named_parameters():
            should_freeze = True
            for unfrozen in layer_names:
                if unfrozen in name:
                    should_freeze = False
                    break

            param.requires_grad = not should_freeze
            if should_freeze:
                self._frozen_layers.add(name)

    def freeze_layers(self, layer_names: list[str]) -> None:
        """Freeze specific layers."""
        for name, param in self.model.named_parameters():
            for frozen_name in layer_names:
                if frozen_name in name:
                    param.requires_grad = False
                    self._frozen_layers.add(name)
                    break

    def unfreeze_layers(self, layer_names: list[str], epoch: int) -> list[str]:
        """Unfreeze specific layers."""
        unfrozen = []
        for name, param in self.model.named_parameters():
            for unfrozen_name in layer_names:
                if unfrozen_name in name and name in self._frozen_layers:
                    param.requires_grad = True
                    self._frozen_layers.discard(name)
                    self._unfrozen_at_epoch[name] = epoch
                    unfrozen.append(name)
                    break
        return unfrozen

    def step(self, epoch: int, optimizer: optim.Optimizer) -> list[str]:
        """Update layer freezing based on epoch."""
        unfrozen = []

        # Check schedule
        if epoch in self.unfreeze_schedule:
            layers_to_unfreeze = self.unfreeze_schedule[epoch]
            unfrozen = self.unfreeze_layers(layers_to_unfreeze, epoch)

            if unfrozen:
                logger.info(f"Epoch {epoch}: Unfroze {len(unfrozen)} layers")

                # Update optimizer with new parameters
                self._update_optimizer_params(optimizer)

        return unfrozen

    def _update_optimizer_params(self, optimizer: optim.Optimizer) -> None:
        """Update optimizer with trainable parameters and LR multipliers."""
        trainable_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                lr_mult = 1.0
                for pattern, mult in self.lr_multipliers.items():
                    if pattern in name:
                        lr_mult = mult
                        break

                trainable_params.append({
                    'params': [param],
                    'lr': optimizer.defaults['lr'] * lr_mult,
                    'name': name,
                })

        # Replace optimizer param groups
        optimizer.param_groups = trainable_params

    def get_status(self) -> dict[str, Any]:
        """Get layer freezing status."""
        return {
            'frozen_count': len(self._frozen_layers),
            'trainable_count': sum(1 for p in self.model.parameters() if p.requires_grad),
            'unfrozen_at_epoch': self._unfrozen_at_epoch.copy(),
        }

    @staticmethod
    def create_default_schedule(
        model: nn.Module,
        total_epochs: int,
        num_stages: int = 4,
    ) -> dict[int, list[str]]:
        """Create a default unfreezing schedule."""
        schedule = {}
        layers = [name for name, _ in model.named_modules()
                  if any(kw in name.lower() for kw in ['layer', 'block', 'encoder', 'decoder'])]

        if not layers:
            return schedule

        stage_size = len(layers) // num_stages
        epoch_step = total_epochs // num_stages

        for i in range(num_stages):
            epoch = i * epoch_step
            start_idx = len(layers) - (i + 1) * stage_size
            end_idx = len(layers) - i * stage_size if i > 0 else len(layers)
            schedule[epoch] = layers[max(0, start_idx):end_idx]

        return schedule


class SWAWithRestarts:
    """
    Stochastic Weight Averaging with periodic restarts.

    Combines SWA's generalization benefits with warm restarts
    for better optimization landscape exploration.
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start: float = 0.75,
        swa_lr: float | None = None,
        restart_period: int = 10,
        num_restarts: int = 3,
    ):
        self.model = model
        self.swa_start = swa_start
        self.swa_lr = swa_lr
        self.restart_period = restart_period
        self.num_restarts = num_restarts

        self._swa_model: torch.optim.swa_utils.AveragedModel | None = None
        self._restart_count = 0
        self._steps_since_restart = 0
        self._is_averaging = False

    def setup(self, device: torch.device) -> None:
        """Initialize SWA model."""
        self._swa_model = torch.optim.swa_utils.AveragedModel(self.model)

    def should_start_averaging(self, progress: float) -> bool:
        """Check if SWA should start based on training progress."""
        return progress >= self.swa_start

    def step(
        self,
        epoch: int,
        total_epochs: int,
        optimizer: optim.Optimizer,
    ) -> bool:
        """Update SWA state and check for restarts."""
        progress = epoch / total_epochs
        did_restart = False

        if self.should_start_averaging(progress):
            if not self._is_averaging:
                self._is_averaging = True
                logger.info(f"SWA started at epoch {epoch}")

                if self.swa_lr is not None:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = self.swa_lr

            # Update SWA model
            if self._swa_model is not None:
                self._swa_model.update_parameters(self.model)

            # Check for restart
            self._steps_since_restart += 1
            if (self._steps_since_restart >= self.restart_period and
                self._restart_count < self.num_restarts):
                self._trigger_restart(optimizer)
                did_restart = True

        return did_restart

    def _trigger_restart(self, optimizer: optim.Optimizer) -> None:
        """Trigger a warm restart."""
        self._restart_count += 1
        self._steps_since_restart = 0

        # Restore original learning rate (will decay again)
        if self.swa_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.swa_lr * 1.5  # Slight bump

        logger.info(f"SWA restart {self._restart_count}/{self.num_restarts}")

    def get_averaged_model(self) -> nn.Module | None:
        """Get the SWA averaged model."""
        return self._swa_model

    def update_bn(self, loader: DataLoader, device: torch.device) -> None:
        """Update batch normalization statistics."""
        if self._swa_model is not None:
            torch.optim.swa_utils.update_bn(loader, self._swa_model, device=device)

    def get_stats(self) -> dict[str, Any]:
        """Get SWA statistics."""
        return {
            'is_averaging': self._is_averaging,
            'restart_count': self._restart_count,
            'steps_since_restart': self._steps_since_restart,
        }


class SmartCheckpointManager:
    """
    Intelligent checkpoint management with minimal I/O overhead.

    .. deprecated:: 2025-12
        For new code, prefer :class:`UnifiedCheckpointManager` from
        ``app.training.checkpoint_unified`` which provides:
        - All features of SmartCheckpointManager
        - Integration with cluster sync
        - Automatic cleanup policies
        - Better async support

    Features:
    - Adaptive checkpointing frequency based on loss improvement
    - Keep only top-k best checkpoints
    - Async checkpoint saving
    - Checkpoint compression
    """

    def __init__(
        self,
        save_dir: Path,
        top_k: int = 3,
        min_interval_epochs: int = 1,
        max_interval_epochs: int = 10,
        improvement_threshold: float = 0.01,
    ):
        self.save_dir = Path(save_dir)
        self.top_k = top_k
        self.min_interval = min_interval_epochs
        self.max_interval = max_interval_epochs
        self.improvement_threshold = improvement_threshold

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: list[tuple[float, Path]] = []
        self._last_save_epoch = -1
        self._last_loss = float('inf')
        self._adaptive_interval = min_interval_epochs

    def should_save(self, epoch: int, loss: float) -> bool:
        """Determine if checkpoint should be saved."""
        if epoch - self._last_save_epoch < self.min_interval:
            return False

        # Always save if significant improvement
        improvement = (self._last_loss - loss) / (abs(self._last_loss) + 1e-8)
        if improvement > self.improvement_threshold:
            self._adaptive_interval = self.min_interval
            return True

        # Adaptive interval based on improvement rate
        if epoch - self._last_save_epoch >= self._adaptive_interval:
            self._adaptive_interval = min(
                self._adaptive_interval + 1,
                self.max_interval
            )
            return True

        return False

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        loss: float,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch:04d}_loss{loss:.4f}.pt"

        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint
        self._checkpoints.append((loss, checkpoint_path))
        self._checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints if exceeding top-k
        while len(self._checkpoints) > self.top_k:
            _, old_path = self._checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")

        self._last_save_epoch = epoch
        self._last_loss = loss

        return checkpoint_path

    def get_best_checkpoint(self) -> Path | None:
        """Get path to best checkpoint."""
        if self._checkpoints:
            return self._checkpoints[0][1]
        return None

    def load_best(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        device: torch.device = torch.device('cpu'),
    ) -> dict[str, Any] | None:
        """Load best checkpoint."""
        best_path = self.get_best_checkpoint()
        if best_path is None or not best_path.exists():
            return None

        checkpoint = safe_load_checkpoint(best_path, map_location=device, warn_on_unsafe=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint

    def get_stats(self) -> dict[str, Any]:
        """Get checkpoint stats."""
        return {
            'num_checkpoints': len(self._checkpoints),
            'best_loss': self._checkpoints[0][0] if self._checkpoints else None,
            'adaptive_interval': self._adaptive_interval,
            'last_save_epoch': self._last_save_epoch,
        }


# Update the suite creation function
def create_phase4_training_suite(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    save_dir: Path,
    device: torch.device,
    total_epochs: int = 50,
    enable_stability_monitor: bool = True,
    enable_adaptive_precision: bool = True,
    enable_progressive_unfreezing: bool = False,
    enable_swa_restarts: bool = True,
    enable_smart_checkpoints: bool = True,
) -> dict[str, Any]:
    """
    Create Phase 4 training utilities suite.

    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        save_dir: Directory for checkpoints
        device: Training device
        total_epochs: Total training epochs
        enable_*: Flags to enable specific features

    Returns:
        Dictionary of Phase 4 utility objects
    """
    suite = {}

    if enable_stability_monitor:
        suite['stability_monitor'] = TrainingStabilityMonitor()

    if enable_adaptive_precision:
        precision_mgr = AdaptivePrecisionManager()
        suite['precision_manager'] = precision_mgr
        suite['scaler'] = precision_mgr.setup(device)

    if enable_progressive_unfreezing:
        schedule = ProgressiveLayerUnfreezing.create_default_schedule(
            model, total_epochs
        )
        suite['layer_unfreezing'] = ProgressiveLayerUnfreezing(
            model, unfreeze_schedule=schedule, total_epochs=total_epochs
        )

    if enable_swa_restarts:
        swa = SWAWithRestarts(model)
        swa.setup(device)
        suite['swa_restarts'] = swa

    if enable_smart_checkpoints:
        suite['checkpoint_manager'] = SmartCheckpointManager(save_dir)

    return suite


# =============================================================================
# Phase 5: Production Optimization (2024-12)
# =============================================================================


class GradientAccumulationScheduler:
    """
    Dynamic gradient accumulation scheduling based on memory pressure.

    Adjusts accumulation steps during training to maximize throughput
    while staying within memory limits. Useful for large batch training
    on memory-constrained GPUs.
    """

    def __init__(
        self,
        initial_accumulation: int = 1,
        min_accumulation: int = 1,
        max_accumulation: int = 16,
        target_memory_fraction: float = 0.85,
        adjustment_interval: int = 100,
        warmup_steps: int = 50,
    ):
        self.accumulation_steps = initial_accumulation
        self.min_accumulation = min_accumulation
        self.max_accumulation = max_accumulation
        self.target_memory_fraction = target_memory_fraction
        self.adjustment_interval = adjustment_interval
        self.warmup_steps = warmup_steps

        self._step_count = 0
        self._memory_history: deque = deque(maxlen=100)
        self._accumulation_history: list[tuple[int, int]] = []
        self._effective_batch_sizes: list[int] = []

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage fraction."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        except RuntimeError:
            return 0.0

    def step(self, batch_size: int) -> int:
        """
        Update accumulation schedule and return current accumulation steps.

        Args:
            batch_size: Current batch size

        Returns:
            Number of gradient accumulation steps to use
        """
        self._step_count += 1

        # Record memory usage
        mem_usage = self.get_memory_usage()
        self._memory_history.append(mem_usage)
        self._effective_batch_sizes.append(batch_size * self.accumulation_steps)

        # Skip adjustment during warmup
        if self._step_count < self.warmup_steps:
            return self.accumulation_steps

        # Adjust at intervals
        if self._step_count % self.adjustment_interval == 0:
            self._adjust_accumulation()

        return self.accumulation_steps

    def _adjust_accumulation(self) -> None:
        """Adjust accumulation based on memory pressure."""
        if not self._memory_history:
            return

        avg_memory = sum(self._memory_history) / len(self._memory_history)
        old_accumulation = self.accumulation_steps

        if avg_memory > self.target_memory_fraction + 0.05:
            # Memory too high, increase accumulation (smaller effective batches)
            self.accumulation_steps = min(
                self.max_accumulation,
                self.accumulation_steps * 2
            )
        elif avg_memory < self.target_memory_fraction - 0.15:
            # Memory has room, decrease accumulation (larger effective batches)
            self.accumulation_steps = max(
                self.min_accumulation,
                self.accumulation_steps // 2
            )

        if self.accumulation_steps != old_accumulation:
            self._accumulation_history.append((self._step_count, self.accumulation_steps))
            logger.info(
                f"Gradient accumulation adjusted: {old_accumulation} -> {self.accumulation_steps} "
                f"(memory: {avg_memory:.1%})"
            )

    def should_step_optimizer(self, micro_step: int) -> bool:
        """Check if optimizer should step after this micro-batch."""
        return (micro_step + 1) % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps

    def get_stats(self) -> dict[str, Any]:
        """Get accumulation statistics."""
        return {
            'current_accumulation': self.accumulation_steps,
            'avg_memory_usage': sum(self._memory_history) / max(1, len(self._memory_history)),
            'effective_batch_size': self._effective_batch_sizes[-1] if self._effective_batch_sizes else 0,
            'adjustment_history': self._accumulation_history[-10:],
        }


class MemoryEfficientAttention:
    """
    Memory-efficient attention implementation using chunked computation.

    When Flash Attention is not available, provides a fallback that
    chunks the attention computation to reduce peak memory usage.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        use_flash_attention: bool = True,
        dropout: float = 0.0,
    ):
        self.chunk_size = chunk_size
        self.dropout = dropout
        self._use_flash = use_flash_attention and self._check_flash_available()
        self._attention_stats: dict[str, float] = {}

    def _check_flash_available(self) -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            # Check for PyTorch 2.0+ scaled_dot_product_attention
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
            # Check for flash-attn package
            import flash_attn  # noqa: F401 - availability check
            return True
        except ImportError:
            return False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Compute attention with memory efficiency.

        Args:
            query: Query tensor [B, N, H, D]
            key: Key tensor [B, S, H, D]
            value: Value tensor [B, S, H, D]
            attn_mask: Optional attention mask
            is_causal: Whether to use causal masking

        Returns:
            Attention output [B, N, H, D]
        """
        if self._use_flash:
            return self._flash_attention(query, key, value, attn_mask, is_causal)
        else:
            return self._chunked_attention(query, key, value, attn_mask, is_causal)

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """Use PyTorch's scaled_dot_product_attention or flash-attn."""
        # Reshape for scaled_dot_product_attention: [B, H, N, D]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        ):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        return out.transpose(1, 2)

    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        """Chunked attention for memory efficiency when flash is unavailable."""
        _B, N, _H, D = query.shape
        S = key.shape[1]

        scale = D ** -0.5
        output = torch.zeros_like(query)

        # Process in chunks
        for i in range(0, N, self.chunk_size):
            end_i = min(i + self.chunk_size, N)
            q_chunk = query[:, i:end_i]

            # Compute attention scores for this chunk
            scores = torch.einsum('bnhd,bshd->bnhs', q_chunk, key) * scale

            if attn_mask is not None:
                scores = scores + attn_mask[:, i:end_i]

            if is_causal:
                # Create causal mask for this chunk
                causal_mask = torch.triu(
                    torch.ones(end_i - i, S, device=query.device, dtype=torch.bool),
                    diagonal=S - end_i + 1
                )
                scores = scores.masked_fill(causal_mask, float('-inf'))

            attn_weights = torch.softmax(scores, dim=-1)

            if self.dropout > 0 and self.training:
                attn_weights = torch.dropout(attn_weights, self.dropout, self.training)

            output[:, i:end_i] = torch.einsum('bnhs,bshd->bnhd', attn_weights, value)

        return output

    @property
    def training(self) -> bool:
        """Check if in training mode."""
        return getattr(self, '_training', True)

    @training.setter
    def training(self, mode: bool) -> None:
        self._training = mode


class ActivationCheckpointingManager:
    """
    Intelligent activation checkpointing for trading compute for memory.

    Selectively checkpoints activations based on:
    - Layer memory footprint
    - Compute cost
    - Current memory pressure
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_ratio: float = 0.5,
        min_layer_size: int = 1024,
        adaptive: bool = True,
    ):
        self.model = model
        self.checkpoint_ratio = checkpoint_ratio
        self.min_layer_size = min_layer_size
        self.adaptive = adaptive

        self._checkpointed_layers: set = set()
        self._layer_costs: dict[str, float] = {}
        self._memory_saved = 0

    def analyze_model(self) -> dict[str, Any]:
        """Analyze model to determine checkpointing strategy."""
        layer_info = {}

        for name, module in self.model.named_modules():
            if self._should_consider_checkpoint(module):
                param_count = sum(p.numel() for p in module.parameters())
                layer_info[name] = {
                    'param_count': param_count,
                    'type': type(module).__name__,
                    'estimated_activation_size': self._estimate_activation_size(module),
                }

        # Sort by activation size and select top ratio for checkpointing
        sorted_layers = sorted(
            layer_info.items(),
            key=lambda x: x[1]['estimated_activation_size'],
            reverse=True
        )

        num_to_checkpoint = int(len(sorted_layers) * self.checkpoint_ratio)
        self._checkpointed_layers = {name for name, _ in sorted_layers[:num_to_checkpoint]}

        return {
            'total_layers': len(layer_info),
            'checkpointed_layers': len(self._checkpointed_layers),
            'layer_info': layer_info,
        }

    def _should_consider_checkpoint(self, module: nn.Module) -> bool:
        """Check if module should be considered for checkpointing."""
        # Only checkpoint significant layers
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
            param_count = sum(p.numel() for p in module.parameters())
            return param_count >= self.min_layer_size
        return False

    def _estimate_activation_size(self, module: nn.Module) -> int:
        """Estimate activation memory size for a module."""
        if isinstance(module, nn.Linear):
            return module.out_features
        elif isinstance(module, nn.Conv2d):
            return module.out_channels
        elif isinstance(module, nn.MultiheadAttention):
            return module.embed_dim
        return 0

    def wrap_model(self) -> nn.Module:
        """Wrap model with activation checkpointing."""

        def checkpoint_wrapper(module):
            """Wrapper that applies checkpointing to forward pass."""
            original_forward = module.forward

            def checkpointed_forward(*args, **kwargs):
                # Use checkpoint for this module
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, use_reentrant=False, **kwargs
                )

            return checkpointed_forward

        # Apply checkpointing to selected layers
        for name, module in self.model.named_modules():
            if name in self._checkpointed_layers:
                module.forward = checkpoint_wrapper(module)
                logger.debug(f"Checkpointing enabled for {name}")

        return self.model

    def get_stats(self) -> dict[str, Any]:
        """Get checkpointing statistics."""
        return {
            'checkpointed_layers': list(self._checkpointed_layers),
            'num_checkpointed': len(self._checkpointed_layers),
            'memory_saved_estimate': self._memory_saved,
        }


class DistributedDataParallelManager:
    """
    Manages Distributed Data Parallel (DDP) training setup.

    Handles:
    - Process group initialization
    - Model wrapping
    - Gradient synchronization
    - Multi-node coordination
    """

    def __init__(
        self,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
        gradient_as_bucket_view: bool = True,
        bucket_cap_mb: int = 25,
    ):
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.broadcast_buffers = broadcast_buffers
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.bucket_cap_mb = bucket_cap_mb

        self._initialized = False
        self._rank = 0
        self._world_size = 1
        self._local_rank = 0

    def setup(self, rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355") -> None:
        """Initialize distributed training environment."""
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
        )

        self._rank = rank
        self._world_size = world_size
        self._local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._initialized = True

        logger.info(f"DDP initialized: rank {rank}/{world_size}, local_rank {self._local_rank}")

    def wrap_model(self, model: nn.Module, device_ids: list[int] | None = None) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        if not self._initialized:
            raise RuntimeError("DDP not initialized. Call setup() first.")

        from torch.nn.parallel import DistributedDataParallel as DDP

        if device_ids is None:
            device_ids = [self._local_rank]

        wrapped = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=self.find_unused_parameters,
            broadcast_buffers=self.broadcast_buffers,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            bucket_cap_mb=self.bucket_cap_mb,
        )

        return wrapped

    def sync_gradients(self) -> None:
        """Synchronize gradients across all processes."""
        import torch.distributed as dist
        if self._initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """All-reduce a tensor across all processes."""
        import torch.distributed as dist

        if not self._initialized:
            return tensor

        dist.all_reduce(tensor)
        if op == "mean":
            tensor /= self._world_size

        return tensor

    def cleanup(self) -> None:
        """Clean up distributed training."""
        import torch.distributed as dist
        if self._initialized:
            dist.destroy_process_group()
            self._initialized = False

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self._rank == 0

    def get_stats(self) -> dict[str, Any]:
        """Get DDP statistics."""
        return {
            'initialized': self._initialized,
            'rank': self._rank,
            'world_size': self._world_size,
            'local_rank': self._local_rank,
            'backend': self.backend,
        }


class DynamicLossScaler:
    """
    Dynamic loss scaling for mixed precision training.

    Implements adaptive loss scaling that:
    - Automatically finds optimal scale
    - Recovers from overflow/underflow
    - Tracks scaling history for debugging
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2**24,
        min_scale: float = 1.0,
    ):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.min_scale = min_scale

        self._growth_tracker = 0
        self._overflow_count = 0
        self._scale_history: list[tuple[int, float]] = []
        self._step_count = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        return loss * self.scale

    def unscale_gradients(self, optimizer: optim.Optimizer) -> bool:
        """
        Unscale gradients and check for overflow.

        Returns:
            True if gradients are valid, False if overflow detected
        """
        has_overflow = False
        inv_scale = 1.0 / self.scale

        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)

                    # Check for inf/nan
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_overflow = True
                        break

            if has_overflow:
                break

        return not has_overflow

    def update(self, overflow: bool) -> None:
        """Update scale based on overflow status."""
        self._step_count += 1

        if overflow:
            # Reduce scale on overflow
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            self._growth_tracker = 0
            self._overflow_count += 1
            self._scale_history.append((self._step_count, self.scale))
            logger.debug(f"Loss scale reduced to {self.scale} due to overflow")
        else:
            # Potentially increase scale
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale = min(self.max_scale, self.scale * self.growth_factor)
                self._growth_tracker = 0
                self._scale_history.append((self._step_count, self.scale))
                logger.debug(f"Loss scale increased to {self.scale}")

    def step(self, optimizer: optim.Optimizer, closure: Callable | None = None) -> bool:
        """
        Perform optimizer step with loss scaling.

        Returns:
            True if step was successful, False if skipped due to overflow
        """
        # Unscale and check for overflow
        valid = self.unscale_gradients(optimizer)

        if valid:
            if closure is not None:
                optimizer.step(closure)
            else:
                optimizer.step()
            self.update(overflow=False)
            return True
        else:
            # Skip step and reduce scale
            optimizer.zero_grad()
            self.update(overflow=True)
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get scaler statistics."""
        return {
            'current_scale': self.scale,
            'overflow_count': self._overflow_count,
            'growth_tracker': self._growth_tracker,
            'scale_history': self._scale_history[-20:],
            'overflow_rate': self._overflow_count / max(1, self._step_count),
        }


class ZeROOptimizer:
    """
    ZeRO (Zero Redundancy Optimizer) style optimizer wrapper.

    Implements ZeRO Stage 1: Optimizer state partitioning across GPUs.
    For full ZeRO, consider using DeepSpeed or FSDP.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        world_size: int = 1,
        rank: int = 0,
        overlap_communication: bool = True,
    ):
        self.optimizer = optimizer
        self.world_size = world_size
        self.rank = rank
        self.overlap_communication = overlap_communication

        self._param_to_partition: dict[int, int] = {}
        self._partition_params: dict[int, list[torch.Tensor]] = {}

        if world_size > 1:
            self._partition_parameters()

    def _partition_parameters(self) -> None:
        """Partition parameters across ranks."""
        all_params = []
        for group in self.optimizer.param_groups:
            all_params.extend(group['params'])

        # Distribute parameters round-robin
        for i, param in enumerate(all_params):
            partition = i % self.world_size
            self._param_to_partition[id(param)] = partition

            if partition not in self._partition_params:
                self._partition_params[partition] = []
            self._partition_params[partition].append(param)

    def step(self, closure: Callable | None = None) -> None:
        """Perform optimizer step with state partitioning."""
        if self.world_size == 1:
            self.optimizer.step(closure)
            return

        import torch.distributed as dist

        # Only update parameters in our partition
        my_params = self._partition_params.get(self.rank, [])

        # Zero gradients for parameters not in our partition
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if id(param) not in [id(p) for p in my_params] and param.grad is not None:
                    param.grad.zero_()

        # Step optimizer
        self.optimizer.step(closure)

        # All-gather updated parameters
        for partition_rank in range(self.world_size):
            partition_params = self._partition_params.get(partition_rank, [])
            for param in partition_params:
                dist.broadcast(param.data, src=partition_rank)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class ElasticTrainingManager:
    """
    Elastic training manager for dynamic worker scaling.

    Handles:
    - Worker joins/leaves during training
    - Checkpoint save/restore for elasticity
    - Batch size adjustment based on worker count
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 8,
        checkpoint_dir: Path = Path("checkpoints/elastic"),
        heartbeat_interval: float = 30.0,
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.heartbeat_interval = heartbeat_interval

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._current_workers = 1
        self._worker_history: list[tuple[float, int]] = []
        self._last_checkpoint_step = 0

    def register_worker(self, worker_id: str) -> int:
        """Register a new worker joining training."""
        self._current_workers = min(self._current_workers + 1, self.max_workers)
        self._worker_history.append((time.time(), self._current_workers))
        logger.info(f"Worker {worker_id} joined. Total workers: {self._current_workers}")
        return self._current_workers

    def remove_worker(self, worker_id: str) -> int:
        """Remove a worker that left training."""
        self._current_workers = max(self._current_workers - 1, self.min_workers)
        self._worker_history.append((time.time(), self._current_workers))
        logger.info(f"Worker {worker_id} left. Total workers: {self._current_workers}")
        return self._current_workers

    def save_elastic_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        step: int,
        extra: dict | None = None,
    ) -> Path:
        """Save checkpoint for elastic recovery."""
        checkpoint_path = self.checkpoint_dir / f"elastic_step_{step}.pt"

        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'workers': self._current_workers,
            'timestamp': time.time(),
        }

        if extra:
            checkpoint.update(extra)

        torch.save(checkpoint, checkpoint_path)
        self._last_checkpoint_step = step

        # Clean old checkpoints (keep last 3)
        checkpoints = sorted(self.checkpoint_dir.glob("elastic_step_*.pt"))
        for old_ckpt in checkpoints[:-3]:
            old_ckpt.unlink()

        return checkpoint_path

    def load_elastic_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
    ) -> int | None:
        """Load latest elastic checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("elastic_step_*.pt"))

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        checkpoint = safe_load_checkpoint(latest, map_location='cpu', warn_on_unsafe=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Loaded elastic checkpoint from step {checkpoint['step']}")
        return checkpoint['step']

    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Get batch size adjusted for current worker count."""
        return base_batch_size * self._current_workers

    def get_stats(self) -> dict[str, Any]:
        """Get elastic training statistics."""
        return {
            'current_workers': self._current_workers,
            'worker_history': self._worker_history[-20:],
            'last_checkpoint_step': self._last_checkpoint_step,
        }


class StreamingNPZLoader:
    """
    Streaming NPZ loader for large datasets that don't fit in memory.

    Supports:
    - Local file streaming
    - S3/GCS URLs (requires boto3/google-cloud-storage)
    - Chunked loading with prefetching
    """

    def __init__(
        self,
        paths: list[str],
        chunk_size: int = 10000,
        prefetch_chunks: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.paths = paths
        self.chunk_size = chunk_size
        self.prefetch_chunks = prefetch_chunks
        self.shuffle = shuffle
        self.seed = seed

        self._rng = np.random.RandomState(seed)
        self._current_chunk: dict[str, np.ndarray] | None = None
        self._chunk_idx = 0
        self._file_idx = 0
        self._total_samples = 0

        # Scan files for total size
        self._scan_files()

    def _scan_files(self) -> None:
        """Scan files to determine total dataset size."""
        for path in self.paths:
            if path.startswith('s3://') or path.startswith('gs://'):
                # For cloud storage, we'd need to fetch metadata
                # For now, estimate based on file size
                self._total_samples += 100000
            else:
                try:
                    with np.load(path, allow_pickle=True) as data:
                        if 'features' in data:
                            self._total_samples += len(data['features'])
                except Exception as e:
                    logger.warning(f"Could not scan {path}: {e}")

    def _load_chunk(self, path: str, start: int, end: int) -> dict[str, np.ndarray]:
        """Load a chunk from file."""
        if path.startswith('s3://'):
            return self._load_s3_chunk(path, start, end)
        elif path.startswith('gs://'):
            return self._load_gcs_chunk(path, start, end)
        else:
            return self._load_local_chunk(path, start, end)

    def _load_local_chunk(self, path: str, start: int, end: int) -> dict[str, np.ndarray]:
        """Load chunk from local file."""
        with np.load(path, allow_pickle=True) as data:
            chunk = {}
            for key in data.files:
                arr = data[key]
                if len(arr) > start:
                    chunk[key] = arr[start:min(end, len(arr))]
            return chunk

    def _load_s3_chunk(self, path: str, start: int, end: int) -> dict[str, np.ndarray]:
        """Load chunk from S3."""
        try:
            import io

            import boto3

            # Parse S3 path
            path = path[5:]  # Remove s3://
            bucket, key = path.split('/', 1)

            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket, Key=key)

            # Load NPZ from bytes
            with np.load(io.BytesIO(response['Body'].read()), allow_pickle=True) as data:
                chunk = {}
                for k in data.files:
                    arr = data[k]
                    if len(arr) > start:
                        chunk[k] = arr[start:min(end, len(arr))]
                return chunk
        except ImportError:
            logger.error("boto3 required for S3 streaming")
            return {}

    def _load_gcs_chunk(self, path: str, start: int, end: int) -> dict[str, np.ndarray]:
        """Load chunk from Google Cloud Storage."""
        try:
            import io

            from google.cloud import storage

            # Parse GCS path
            path = path[5:]  # Remove gs://
            bucket_name, blob_name = path.split('/', 1)

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            with np.load(io.BytesIO(blob.download_as_bytes()), allow_pickle=True) as data:
                chunk = {}
                for k in data.files:
                    arr = data[k]
                    if len(arr) > start:
                        chunk[k] = arr[start:min(end, len(arr))]
                return chunk
        except ImportError:
            logger.error("google-cloud-storage required for GCS streaming")
            return {}

    def __iter__(self):
        """Iterate over chunks."""
        if self.shuffle:
            self._rng.shuffle(self.paths)

        for path in self.paths:
            offset = 0
            while True:
                chunk = self._load_chunk(path, offset, offset + self.chunk_size)
                if not chunk or not any(len(v) > 0 for v in chunk.values()):
                    break
                yield chunk
                offset += self.chunk_size

    def __len__(self) -> int:
        """Return total number of samples."""
        return self._total_samples


class TrainingProfiler:
    """
    Training profiler with PyTorch Profiler and TensorBoard integration.

    Captures:
    - GPU/CPU time per operation
    - Memory allocation timeline
    - Data loading bottlenecks
    - Kernel execution traces
    """

    def __init__(
        self,
        log_dir: Path = Path("runs/profile"),
        profile_memory: bool = True,
        profile_shapes: bool = True,
        record_shapes: bool = True,
        with_stack: bool = False,
        schedule_wait: int = 1,
        schedule_warmup: int = 1,
        schedule_active: int = 3,
        schedule_repeat: int = 2,
    ):
        self.log_dir = Path(log_dir)
        self.profile_memory = profile_memory
        self.profile_shapes = profile_shapes
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.schedule_wait = schedule_wait
        self.schedule_warmup = schedule_warmup
        self.schedule_active = schedule_active
        self.schedule_repeat = schedule_repeat

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._profiler: torch.profiler.profile | None = None
        self._step_count = 0
        self._profile_results: list[dict] = []

    def create_profiler(self) -> torch.profiler.profile:
        """Create and return a PyTorch profiler."""
        schedule = torch.profiler.schedule(
            wait=self.schedule_wait,
            warmup=self.schedule_warmup,
            active=self.schedule_active,
            repeat=self.schedule_repeat,
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.log_dir)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

        return self._profiler

    def step(self) -> None:
        """Step the profiler."""
        if self._profiler is not None:
            self._profiler.step()
            self._step_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        if self._profiler is None:
            return {}

        # Get key averages
        try:
            key_averages = self._profiler.key_averages()

            top_ops = []
            for event in sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]:
                top_ops.append({
                    'name': event.key,
                    'cuda_time_ms': event.cuda_time_total / 1000,
                    'cpu_time_ms': event.cpu_time_total / 1000,
                    'calls': event.count,
                })

            return {
                'top_operations': top_ops,
                'total_steps': self._step_count,
                'log_dir': str(self.log_dir),
            }
        except Exception as e:
            logger.warning(f"Could not generate profile summary: {e}")
            return {}

    def export_chrome_trace(self, path: Path | None = None) -> Path:
        """Export trace for Chrome tracing."""
        if self._profiler is None:
            raise RuntimeError("Profiler not created")

        path = path or self.log_dir / f"trace_{self._step_count}.json"
        self._profiler.export_chrome_trace(str(path))
        return path


class ABModelTester:
    """
    A/B testing framework for comparing model variants.

    Provides statistical comparison of:
    - Win rates
    - Elo differences
    - Loss distributions
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        min_games: int = 100,
        elo_k_factor: float = 32.0,
    ):
        self.confidence_level = confidence_level
        self.min_games = min_games
        self.elo_k_factor = elo_k_factor

        self._results: dict[str, list[dict]] = {}
        self._elo_ratings: dict[str, float] = {}

    def register_model(self, model_id: str, initial_elo: float = 1500.0) -> None:
        """Register a model for A/B testing."""
        self._results[model_id] = []
        self._elo_ratings[model_id] = initial_elo

    def record_game(
        self,
        model_a: str,
        model_b: str,
        winner: str,  # model_a, model_b, or draw
        game_length: int = 0,
        extra: dict | None = None,
    ) -> None:
        """Record a game result."""
        result = {
            'model_a': model_a,
            'model_b': model_b,
            'winner': winner,
            'game_length': game_length,
            'timestamp': time.time(),
        }
        if extra:
            result.update(extra)

        self._results.setdefault(model_a, []).append(result)
        self._results.setdefault(model_b, []).append(result)

        # Update Elo ratings
        self._update_elo(model_a, model_b, winner)

    def _update_elo(self, model_a: str, model_b: str, winner: str) -> None:
        """Update Elo ratings based on game result."""
        elo_a = self._elo_ratings.get(model_a, 1500.0)
        elo_b = self._elo_ratings.get(model_b, 1500.0)

        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 - expected_a

        if winner == model_a:
            actual_a, actual_b = 1.0, 0.0
        elif winner == model_b:
            actual_a, actual_b = 0.0, 1.0
        else:  # draw
            actual_a, actual_b = 0.5, 0.5

        self._elo_ratings[model_a] = elo_a + self.elo_k_factor * (actual_a - expected_a)
        self._elo_ratings[model_b] = elo_b + self.elo_k_factor * (actual_b - expected_b)

    def get_comparison(self, model_a: str, model_b: str) -> dict[str, Any]:
        """Get statistical comparison between two models."""
        # Find head-to-head games
        h2h_games = [
            r for r in self._results.get(model_a, [])
            if r['model_b'] == model_b or r['model_a'] == model_b
        ]

        if len(h2h_games) < self.min_games:
            return {
                'status': 'insufficient_data',
                'games_played': len(h2h_games),
                'min_required': self.min_games,
            }

        # Calculate win rates
        wins_a = sum(1 for g in h2h_games if g['winner'] == model_a)
        wins_b = sum(1 for g in h2h_games if g['winner'] == model_b)
        draws = len(h2h_games) - wins_a - wins_b

        win_rate_a = wins_a / len(h2h_games)
        win_rate_b = wins_b / len(h2h_games)

        # Wilson score confidence interval
        ci_low_a, ci_high_a = self._wilson_ci(wins_a, len(h2h_games))
        ci_low_b, ci_high_b = self._wilson_ci(wins_b, len(h2h_games))

        # Statistical significance
        is_significant = ci_high_a < ci_low_b or ci_high_b < ci_low_a

        return {
            'status': 'complete',
            'games_played': len(h2h_games),
            'model_a': {
                'id': model_a,
                'wins': wins_a,
                'win_rate': win_rate_a,
                'ci_low': ci_low_a,
                'ci_high': ci_high_a,
                'elo': self._elo_ratings.get(model_a, 1500.0),
            },
            'model_b': {
                'id': model_b,
                'wins': wins_b,
                'win_rate': win_rate_b,
                'ci_low': ci_low_b,
                'ci_high': ci_high_b,
                'elo': self._elo_ratings.get(model_b, 1500.0),
            },
            'draws': draws,
            'is_significant': is_significant,
            'elo_difference': self._elo_ratings.get(model_a, 1500.0) - self._elo_ratings.get(model_b, 1500.0),
        }

    def _wilson_ci(self, successes: int, total: int) -> tuple[float, float]:
        """Calculate Wilson score confidence interval."""
        import scipy.stats as stats

        if total == 0:
            return 0.0, 1.0

        p = successes / total
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

        return max(0.0, center - spread), min(1.0, center + spread)

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get sorted leaderboard of all models."""
        leaderboard = []
        for model_id, elo in sorted(self._elo_ratings.items(), key=lambda x: -x[1]):
            games = self._results.get(model_id, [])
            wins = sum(1 for g in games if g['winner'] == model_id)
            leaderboard.append({
                'model_id': model_id,
                'elo': elo,
                'games': len(games),
                'wins': wins,
                'win_rate': wins / max(1, len(games)),
            })
        return leaderboard


def create_phase5_production_suite(
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    save_dir: Path = Path("checkpoints"),
    enable_gradient_accumulation: bool = True,
    enable_activation_checkpointing: bool = False,
    enable_profiling: bool = False,
    enable_ab_testing: bool = True,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
) -> dict[str, Any]:
    """
    Create Phase 5 production optimization suite.

    Args:
        model: Model to train
        optimizer: Optimizer
        device: Training device
        save_dir: Directory for checkpoints
        enable_*: Flags to enable specific features
        distributed_*: Distributed training settings

    Returns:
        Dictionary of Phase 5 utility objects
    """
    suite = {}

    if enable_gradient_accumulation:
        suite['accumulation_scheduler'] = GradientAccumulationScheduler()

    if enable_activation_checkpointing:
        checkpoint_mgr = ActivationCheckpointingManager(model)
        checkpoint_mgr.analyze_model()
        suite['activation_checkpointing'] = checkpoint_mgr

    if enable_profiling:
        profiler = TrainingProfiler(log_dir=save_dir / "profile")
        suite['profiler'] = profiler

    if enable_ab_testing:
        suite['ab_tester'] = ABModelTester()

    # Memory efficient attention helper
    suite['memory_efficient_attention'] = MemoryEfficientAttention()

    # Dynamic loss scaler for mixed precision
    suite['loss_scaler'] = DynamicLossScaler()

    # Elastic training support
    suite['elastic_manager'] = ElasticTrainingManager(checkpoint_dir=save_dir / "elastic")

    # Distributed training if multiple GPUs
    if distributed_world_size > 1:
        ddp_manager = DistributedDataParallelManager()
        suite['ddp_manager'] = ddp_manager

    # ZeRO optimizer wrapper for memory efficiency
    if distributed_world_size > 1:
        suite['zero_optimizer'] = ZeROOptimizer(
            optimizer,
            world_size=distributed_world_size,
            rank=distributed_rank,
        )

    return suite


# =============================================================================
# Temperature Scheduling for Selfplay
# =============================================================================
# NOTE: Temperature scheduling has been consolidated into the canonical module:
# app/training/temperature_scheduling.py
#
# Import from there for the comprehensive ABC-based architecture with:
# - Multiple schedule types (constant, linear, exponential, cosine, step, adaptive)
# - Move-aware temperature adjustment
# - Curriculum-based scheduling
# - Dirichlet noise support
#
# Example:
#     from app.training.temperature_scheduling import (
#         TemperatureConfig,
#         TemperatureScheduler,
#         create_scheduler,
#     )


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty-based sample weighting."""
    method: str = "mc_dropout"  # "mc_dropout", "ensemble", "gradient_norm"
    mc_samples: int = 10  # Number of MC dropout forward passes
    temperature: float = 1.0  # Temperature for softmax
    weight_min: float = 0.1  # Minimum sample weight
    weight_max: float = 3.0  # Maximum sample weight
    uncertainty_scale: str = "linear"  # "linear", "sqrt", "log"
    cache_size: int = 100000  # Max cached uncertainty scores


class UncertaintySampler:
    """Computes uncertainty scores for training samples.

    Uncertain samples (where the model disagrees with itself) are more
    informative and should be weighted higher during training.

    Methods:
    - MC Dropout: Run multiple forward passes with dropout enabled
    - Ensemble: Use variance across ensemble predictions
    - Gradient Norm: Use gradient magnitude as uncertainty proxy
    """

    def __init__(self, model: nn.Module, config: UncertaintyConfig):
        """Initialize the uncertainty sampler.

        Args:
            model: The neural network model
            config: Uncertainty configuration
        """
        self.model = model
        self.config = config
        self._cache: dict[int, float] = {}  # sample_id -> uncertainty

    def compute_uncertainty_mc_dropout(
        self,
        inputs: torch.Tensor,
        sample_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute uncertainty using MC Dropout.

        Runs multiple forward passes with dropout enabled and measures
        variance across predictions.

        Args:
            inputs: Input tensor (batch_size, ...)
            sample_ids: Optional IDs for caching

        Returns:
            Uncertainty scores (batch_size,)
        """
        self.model.train()  # Enable dropout
        inputs.size(0)

        value_preds = []
        policy_preds = []

        with torch.no_grad():
            for _ in range(self.config.mc_samples):
                output = self.model(inputs)

                # Handle both (value, policy) and single output
                if isinstance(output, tuple):
                    value, policy = output[:2]
                else:
                    value = output
                    policy = None

                value_preds.append(value)
                if policy is not None:
                    policy_preds.append(policy)

        # Stack predictions
        value_stack = torch.stack(value_preds, dim=0)  # (mc_samples, batch, ...)
        value_var = value_stack.var(dim=0).mean(dim=-1) if value_stack.dim() > 2 else value_stack.var(dim=0)

        # Policy uncertainty (entropy of mean policy)
        if policy_preds:
            policy_stack = torch.stack(policy_preds, dim=0)
            policy_mean = policy_stack.mean(dim=0)
            # Policy entropy as uncertainty
            policy_entropy = -(policy_mean * (policy_mean + 1e-8).log()).sum(dim=-1)
            uncertainty = value_var.squeeze() + 0.5 * policy_entropy
        else:
            uncertainty = value_var.squeeze()

        # Ensure 1D
        if uncertainty.dim() == 0:
            uncertainty = uncertainty.unsqueeze(0)

        # Cache if sample_ids provided
        if sample_ids is not None:
            for i, sid in enumerate(sample_ids):
                if len(self._cache) < self.config.cache_size:
                    self._cache[sid] = uncertainty[i].item()

        return uncertainty

    def compute_uncertainty_gradient_norm(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
    ) -> torch.Tensor:
        """Compute uncertainty using gradient norm.

        Samples with larger gradients are more informative.

        Args:
            inputs: Input tensor
            targets: Target tensor
            loss_fn: Loss function

        Returns:
            Uncertainty scores (batch_size,)
        """
        self.model.train()
        batch_size = inputs.size(0)
        uncertainties = []

        for i in range(batch_size):
            self.model.zero_grad()
            output = self.model(inputs[i:i+1])

            if isinstance(output, tuple):
                value = output[0]
            else:
                value = output

            loss = loss_fn(value, targets[i:i+1])
            loss.backward()

            # Sum gradient norms across all parameters
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            uncertainties.append(math.sqrt(grad_norm))

        return torch.tensor(uncertainties, device=inputs.device)

    def get_cached_uncertainty(self, sample_id: int) -> float | None:
        """Get cached uncertainty for a sample."""
        return self._cache.get(sample_id)

    def compute_sample_weights(self, uncertainties: torch.Tensor) -> torch.Tensor:
        """Convert uncertainties to sample weights.

        Args:
            uncertainties: Raw uncertainty scores

        Returns:
            Sample weights (normalized to mean=1)
        """
        config = self.config

        # Scale uncertainties
        if config.uncertainty_scale == "sqrt":
            scaled = torch.sqrt(uncertainties + 1e-8)
        elif config.uncertainty_scale == "log":
            scaled = torch.log1p(uncertainties)
        else:
            scaled = uncertainties

        # Normalize to [0, 1] range
        min_u, max_u = scaled.min(), scaled.max()
        if max_u > min_u:
            normalized = (scaled - min_u) / (max_u - min_u)
        else:
            normalized = torch.ones_like(scaled) * 0.5

        # Map to weight range
        weights = config.weight_min + normalized * (config.weight_max - config.weight_min)

        # Normalize so mean weight = 1
        weights = weights / weights.mean()

        return weights

    def clear_cache(self):
        """Clear the uncertainty cache."""
        self._cache.clear()


class UncertaintyWeightedSampler(WeightedSamplerBase):
    """Weighted sampler that prioritizes uncertain samples.

    Uses uncertainty scores to create a sampling distribution that
    oversamples difficult/uncertain examples.

    Inherits from WeightedSamplerBase (2025-12) for common sampling interface.
    """

    def __init__(
        self,
        dataset_size: int,
        uncertainty_scores: np.ndarray | None = None,
        config: UncertaintyConfig | None = None,
    ):
        """Initialize the weighted sampler.

        Args:
            dataset_size: Total number of samples
            uncertainty_scores: Pre-computed uncertainty scores
            config: Uncertainty configuration
        """
        self.dataset_size = dataset_size
        self.config = config or UncertaintyConfig()

        if uncertainty_scores is not None:
            self.update_uncertainty_scores(uncertainty_scores)
        else:
            # Uniform weights initially
            self.weights = np.ones(dataset_size) / dataset_size

    def update_uncertainty_scores(self, uncertainty_scores: np.ndarray):
        """Update sampling weights from uncertainty scores.

        Args:
            uncertainty_scores: Array of uncertainty scores (dataset_size,)
        """
        assert len(uncertainty_scores) == self.dataset_size

        # Scale uncertainties
        if self.config.uncertainty_scale == "sqrt":
            scaled = np.sqrt(uncertainty_scores + 1e-8)
        elif self.config.uncertainty_scale == "log":
            scaled = np.log1p(uncertainty_scores)
        else:
            scaled = uncertainty_scores

        # Normalize to [0, 1]
        min_u, max_u = scaled.min(), scaled.max()
        if max_u > min_u:
            normalized = (scaled - min_u) / (max_u - min_u)
        else:
            normalized = np.ones_like(scaled) * 0.5

        # Map to weight range and normalize to sum=1
        weights = self.config.weight_min + normalized * (self.config.weight_max - self.config.weight_min)
        self.weights = weights / weights.sum()

    # Backwards compatibility alias
    update_weights = update_uncertainty_scores

    # sample() and get_weight() inherited from WeightedSamplerBase


def create_uncertainty_sampler(
    model: nn.Module,
    method: str = "mc_dropout",
    mc_samples: int = 10,
) -> UncertaintySampler:
    """Factory function to create an uncertainty sampler.

    Args:
        model: Neural network model
        method: Uncertainty estimation method
        mc_samples: Number of MC dropout samples

    Returns:
        Configured UncertaintySampler
    """
    config = UncertaintyConfig(
        method=method,
        mc_samples=mc_samples,
    )
    return UncertaintySampler(model, config)
